"""
Layer 10 — Validation Set D: Cross-Domain & Non-Stationary Trace Evaluation
Evaluates the trained pipeline against a non-stationary diurnal/burst operational trace
covering ALL 10 fault classes in the ontology.

P3.2: Extended from 5-class/500-step trace to a 10-class/1000-step trace covering:
  Phase 1  (  0- 100): Normal            — nominal baseline
  Phase 2  (100- 200): Optical Misalignment
  Phase 3  (200- 300): Channel Attenuation Event
  Phase 4  (300- 400): Intercept-Resend
  Phase 5  (400- 500): Thermal Drift
  Phase 6  (500- 600): Detector APD Degradation
  Phase 7  (600- 700): Timing Jitter
  Phase 8  (700- 800): Detector Blinding
  Phase 9  (800- 900): Photon Number Splitting
  Phase 10 (900-1000): Time-Shift Attack

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


def generate_diurnal_burst_trace(n_steps: int = 1000, seed: int = 7777) -> List[Tuple[str, float]]:
    """
    Generates a dynamic 10-phase non-stationary trace covering every fault class.

    Each phase ramps intensity gradually to simulate realistic fault onset, then
    holds at peak intensity for the second half of the phase window.

    Phase map (100 steps each):
      0   Normal                — no fault
      1   Optical Misalignment  — V degrades 0.30 → 0.85
      2   Channel Attenuation   — loss ramps 0.40 → 0.85
      3   Intercept-Resend      — Eve fraction 0.50 ± small noise
      4   Thermal Drift         — T ramps 0.45 → 0.95
      5   Detector APD Degradation — DCR boost 0.40 → 0.90
      6   Timing Jitter         — jitter ramps 0.35 → 0.80
      7   Detector Blinding     — intensity 0.60 → 0.95
      8   Photon Number Split.  — intensity 0.50 → 0.85
      9   Time-Shift Attack     — intensity 0.55 → 0.90
    """
    rng = np.random.RandomState(seed)
    phases: List[Tuple[str, float]] = []
    phase_size = n_steps // 10

    def ramp(t_local: int, t_total: int, lo: float, hi: float) -> float:
        """Linear ramp over first half, hold at hi for second half."""
        half = t_total // 2
        if t_local < half:
            return lo + (hi - lo) * (t_local / max(1, half - 1))
        return hi

    fault_specs = [
        ("Normal",                    0.0,   0.0),   # always 0
        ("Optical Misalignment",      0.30,  0.85),
        ("Channel Attenuation Event", 0.40,  0.85),
        ("Intercept-Resend",          0.50,  0.50),  # held constant, noisy
        ("Thermal Drift",             0.45,  0.95),
        ("Detector APD Degradation",  0.40,  0.90),
        ("Timing Jitter",             0.35,  0.80),
        ("Detector Blinding",         0.60,  0.95),
        ("Photon Number Splitting",   0.50,  0.85),
        ("Time-Shift Attack",         0.55,  0.90),
    ]

    for phase_idx, (fault_type, lo, hi) in enumerate(fault_specs):
        for t_local in range(phase_size):
            if fault_type == "Normal":
                intensity = 0.0
            elif fault_type == "Intercept-Resend":
                # Small noise around midpoint to test robustness
                intensity = 0.50 + 0.20 * rng.uniform(-0.1, 0.2)
                intensity = float(np.clip(intensity, 0.0, 1.0))
            else:
                intensity = ramp(t_local, phase_size, lo, hi)
                intensity = float(np.clip(intensity + rng.normal(0.0, 0.01), 0.0, 1.0))
            phases.append((fault_type, intensity))

    return phases


def run_validation_set_d(models_dir: str = "models") -> Dict[str, Any]:
    """
    Runs the full pipeline over the 10-phase non-stationary multi-regime trace.
    Reports per-class accuracy for every fault in the ontology.
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

    trace_phases = generate_diurnal_burst_trace(n_steps=1000, seed=7777)

    correct_attributions = 0
    total_anomalies_detected = 0
    total_true_anomalies = 0

    # Track per-phase (fault-class level) metrics
    phase_metrics: Dict[str, Dict[str, int]] = {
        p: {"total": 0, "correct": 0, "anomaly_detected": 0, "is_fault": int(p != "Normal")}
        for p in ROOT_CAUSE_CLASSES
    }

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
                phase_metrics[fault_type]["anomaly_detected"] += 1

        phase_metrics[fault_type]["total"] += 1
        if result.attribution.predicted_class == fault_type:
            phase_metrics[fault_type]["correct"] += 1
            correct_attributions += 1

    overall_accuracy = correct_attributions / max(1, len(trace_phases))
    anomaly_detection_rate = total_anomalies_detected / max(1, total_true_anomalies)

    # Per-class accuracies and anomaly detection rates
    phase_accuracies: Dict[str, float] = {}
    phase_anomaly_rates: Dict[str, float] = {}
    for p in ROOT_CAUSE_CLASSES:
        m = phase_metrics[p]
        if m["total"] > 0:
            phase_accuracies[p] = m["correct"] / m["total"]
            if m["is_fault"]:
                phase_anomaly_rates[p] = m["anomaly_detected"] / m["total"]

    # Macro-average over fault classes (excluding Normal)
    fault_accs = [phase_accuracies[p] for p in ROOT_CAUSE_CLASSES
                  if p != "Normal" and p in phase_accuracies]
    macro_fault_accuracy = float(np.mean(fault_accs)) if fault_accs else 0.0

    return {
        "suite_name": "Validation Set D — Cross-Domain 10-Class Non-Stationary Trace (P3.2)",
        "total_steps": len(trace_phases),
        "n_phases": 10,
        "overall_attribution_accuracy": float(overall_accuracy),
        "macro_fault_accuracy": macro_fault_accuracy,
        "anomaly_detection_rate": float(anomaly_detection_rate),
        "phase_accuracies": phase_accuracies,
        "phase_anomaly_detection_rates": phase_anomaly_rates,
    }


if __name__ == "__main__":
    res = run_validation_set_d()
    print(f"\n--- {res['suite_name']} ---")
    print(f"Overall Attribution Accuracy : {res['overall_attribution_accuracy']*100:.2f}%")
    print(f"Macro Fault-Class Accuracy   : {res['macro_fault_accuracy']*100:.2f}%")
    print(f"Anomaly Detection Rate       : {res['anomaly_detection_rate']*100:.2f}%")
    print("\nPer-Phase Results:")
    for phase, acc in res["phase_accuracies"].items():
        det_str = ""
        if phase in res["phase_anomaly_detection_rates"]:
            det_str = f"  (Anomaly Det: {res['phase_anomaly_detection_rates'][phase]*100:.1f}%)"
        print(f"  [{phase:<28}]: Attribution = {acc*100:.1f}%{det_str}")
