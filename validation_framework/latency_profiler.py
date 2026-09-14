"""
Latency and Performance Profiler for Q-SENTINEL
Measures sub-millisecond execution timing across individual pipeline layers to verify real-time guarantees.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import sys
import time
import numpy as np
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streaming_pipeline.qkd_network_orchestrator import QKDNetworkOrchestrator
from anomaly_detection.isolation_forest_detector import IsolationForestAnomalyDetector
from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier


def profile_pipeline_latencies(
    models_dir: str = "models",
    n_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Measures timing breakdowns per layer over n_iterations.
    """
    iso_model_path = os.path.join(models_dir, "isolation_forest.joblib")
    iso_scaler_path = os.path.join(models_dir, "isolation_scaler.joblib")
    lgb_model_path = os.path.join(models_dir, "lightgbm_classifier.joblib")
    
    detector = IsolationForestAnomalyDetector()
    detector.load(iso_model_path, iso_scaler_path)
    
    classifier = LightGBMRootCauseClassifier()
    classifier.load(lgb_model_path)
    
    orchestrator = QKDNetworkOrchestrator(enable_audit_logging=False)
    orchestrator.attach_trained_models(detector, classifier)
    
    # Warm-up
    for _ in range(10):
        orchestrator.process_step(dt_seconds=1.0)
        
    total_latencies_ms = []
    layer2_latencies_ms = []
    layer3_latencies_ms = []
    layer4_latencies_ms = []
    layer5_latencies_ms = []
    layer6_latencies_ms = []
    layer7_latencies_ms = []
    
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        
        # 1. Acquire telemetry
        sample = orchestrator.emulator.step(1.0)
        feats = orchestrator.feature_extractor.extract_features(sample)
        
        # Layer 2
        t_l2_start = time.perf_counter()
        anom = orchestrator.anomaly_detector.predict_sample(feats)
        t_l2 = (time.perf_counter() - t_l2_start) * 1000.0
        layer2_latencies_ms.append(t_l2)
        
        # Layer 4 (preliminary attribution, needed before physics validator)
        raw_attr = orchestrator.classifier.predict_sample(feats, None)

        # Layer 3
        t_l3_start = time.perf_counter()
        phys = orchestrator.physics_validator.evaluate(feats, raw_attr.predicted_class, raw_attr.confidence)
        t_l3 = (time.perf_counter() - t_l3_start) * 1000.0
        layer3_latencies_ms.append(t_l3)
        
        # Layer 4
        t_l4_start = time.perf_counter()
        attr = orchestrator.classifier.predict_sample(feats, phys)
        t_l4 = (time.perf_counter() - t_l4_start) * 1000.0
        layer4_latencies_ms.append(t_l4)
        
        # Layer 5
        t_l5_start = time.perf_counter()
        expl = orchestrator.explainer.explain_sample(attr, phys, top_k=3)
        t_l5 = (time.perf_counter() - t_l5_start) * 1000.0
        layer5_latencies_ms.append(t_l5)
        
        # Layer 6
        t_l6_start = time.perf_counter()
        ptct = orchestrator.forecaster.compute_ptct(
            sample.qber,
            feats.get("qber_slope_25", 0.0),
            slope_standard_error=feats.get("qber_slope_25_se", 0.0002),
            qber_acceleration=feats.get("qber_acceleration_25", 0.0),
        )
        t_l6 = (time.perf_counter() - t_l6_start) * 1000.0
        layer6_latencies_ms.append(t_l6)
        
        # Layer 7
        t_l7_start = time.perf_counter()
        remed = orchestrator.remediation_engine.generate_recommendation(attr.predicted_class, feats)
        t_l7 = (time.perf_counter() - t_l7_start) * 1000.0
        layer7_latencies_ms.append(t_l7)
        
        t_total = (time.perf_counter() - t0) * 1000.0
        total_latencies_ms.append(t_total)
        
    return {
        "suite_name": "Pipeline Latency Profiler",
        "iterations": n_iterations,
        "mean_total_latency_ms": float(np.mean(total_latencies_ms)),
        "p95_total_latency_ms": float(np.percentile(total_latencies_ms, 95)),
        "p99_total_latency_ms": float(np.percentile(total_latencies_ms, 99)),
        "layer_breakdowns_mean_ms": {
            "Layer 2 (Isolation Forest)": float(np.mean(layer2_latencies_ms)),
            "Layer 3 (Physics Invariant Check)": float(np.mean(layer3_latencies_ms)),
            "Layer 4 (LightGBM Attribution)": float(np.mean(layer4_latencies_ms)),
            "Layer 5 (SHAP TreeExplainer)": float(np.mean(layer5_latencies_ms)),
            "Layer 6 (PTCT Forecaster)": float(np.mean(layer6_latencies_ms)),
            "Layer 7 (Remediation Optimizer)": float(np.mean(layer7_latencies_ms)),
        },
        "target_5ms_compliant": bool(np.mean(total_latencies_ms) < 5.0),
    }


if __name__ == "__main__":
    res = profile_pipeline_latencies()
    print(f"\n--- {res['suite_name']} ({res['iterations']} iterations) ---")
    print(f"Mean Pipeline Latency:  {res['mean_total_latency_ms']:.3f} ms")
    print(f"P95 Pipeline Latency:   {res['p95_total_latency_ms']:.3f} ms")
    print(f"Target <5.0ms Compliant: {res['target_5ms_compliant']}")
    print("\nLayer Timing Breakdown:")
    for layer, ms in res["layer_breakdowns_mean_ms"].items():
        print(f"  {layer:<35}: {ms:.4f} ms")
