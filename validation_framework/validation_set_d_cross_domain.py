"""
Layer 10 — Validation Set D: Cross-Domain & Non-Stationary Trace Evaluation
Evaluates the trained pipeline against non-stationary diurnal and burst operational traces.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import sys
import numpy as np
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streaming_pipeline.qkd_network_orchestrator import QKDNetworkOrchestrator
from anomaly_detection.isolation_forest_detector import IsolationForestAnomalyDetector
from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier
from config.qkd_system_parameters import ROOT_CAUSE_CLASSES


def generate_diurnal_burst_trace(n_steps: int = 500, seed: int = 7777) -> List[Tuple]:
    """
    Generates dynamic multi-phase trace:
    1. Phase 1 (0-100s): Nominal with diurnal temperature drift
    2. Phase 2 (100-200s): Progressive optical misalignment (polarization drift)
    3. Phase 3 (200-300s): Rapid macrobend fiber attenuation
    4. Phase 4 (300-400s): Intercept-resend eavesdropping burst
    5. Phase 5 (400-500s): Thermal runaway (TEC saturation)
    """
    rng = np.random.RandomState(seed)
    phases = []
    
    for t in range(n_steps):
        if t < 100:
            phases.append(("Normal", 0.0))
        elif t < 200:
            # Misalignment grows from 0.3 to 0.85
            intensity = 0.30 + 0.55 * ((t - 100) / 100.0)
            phases.append(("Optical Misalignment", intensity))
        elif t < 300:
            # Fiber loss jumps to 0.70
            intensity = 0.40 + 0.45 * ((t - 200) / 100.0)
            phases.append(("Channel Attenuation Event", intensity))
        elif t < 400:
            # Eve intercepts 40-70% of pulses
            intensity = 0.50 + 0.35 * rng.uniform(-0.1, 0.2)
            phases.append(("Intercept-Resend", intensity))
        else:
            # APD heating
            intensity = 0.45 + 0.50 * ((t - 400) / 100.0)
            phases.append(("Thermal Drift", intensity))
            
    return phases


def run_validation_set_d(models_dir: str = "models") -> Dict[str, Any]:
    """
    Runs full pipeline over non-stationary multi-regime trace.
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
    
    trace_phases = generate_diurnal_burst_trace(n_steps=500, seed=7777)
    
    correct_attributions = 0
    total_anomalies_detected = 0
    total_true_anomalies = 0
    phase_metrics = {p: {"total": 0, "correct": 0} for p in ROOT_CAUSE_CLASSES}
    
    for fault_type, intensity in trace_phases:
        if fault_type == "Normal":
            orchestrator.emulator.reset_to_nominal()
        else:
            orchestrator.emulator.inject_fault(fault_type, intensity)
            
        result = orchestrator.process_step(dt_seconds=1.0)
        
        is_true_anomaly = (fault_type != "Normal")
        if is_true_anomaly:
            total_true_anomalies += 1
            if result.anomaly.is_anomaly:
                total_anomalies_detected += 1
                
        phase_metrics[fault_type]["total"] += 1
        if result.attribution.predicted_class == fault_type:
            phase_metrics[fault_type]["correct"] += 1
            correct_attributions += 1
            
    overall_accuracy = correct_attributions / len(trace_phases)
    anomaly_detection_rate = total_anomalies_detected / max(1, total_true_anomalies)
    
    phase_accuracies = {
        p: (phase_metrics[p]["correct"] / phase_metrics[p]["total"]) if phase_metrics[p]["total"] > 0 else 0.0
        for p in ROOT_CAUSE_CLASSES if phase_metrics[p]["total"] > 0
    }
    
    return {
        "suite_name": "Validation Set D (Cross-Domain Non-Stationary Multi-Phase Trace)",
        "total_steps": len(trace_phases),
        "overall_attribution_accuracy": float(overall_accuracy),
        "anomaly_detection_rate": float(anomaly_detection_rate),
        "phase_accuracies": phase_accuracies,
    }


if __name__ == "__main__":
    res = run_validation_set_d()
    print(f"\n--- {res['suite_name']} ---")
    print(f"Overall Attribution Accuracy: {res['overall_attribution_accuracy']*100:.2f}%")
    print(f"Anomaly Detection Rate:       {res['anomaly_detection_rate']*100:.2f}%")
    for phase, acc in res["phase_accuracies"].items():
        print(f"  Phase [{phase:<22}]: Accuracy = {acc*100:.1f}%")
