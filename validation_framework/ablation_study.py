"""
Phase 23: Architecture Ablation Study
Empirically benchmarks:
1. ML Only (LightGBM standalone)
2. Physics Only (Deterministic rule heuristics standalone)
3. Q-Sentinel Hybrid (ML + Physics Invariant Consistency Fusion)
Reports: Macro-F1, False Positive Rate (FPR), and Diagnostic Explainability.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import sys
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.qkd_system_parameters import (
    ROOT_CAUSE_CLASSES,
    ROOT_CAUSE_LABEL_TO_ID,
    ROOT_CAUSE_ID_TO_LABEL,
)
from anomaly_detection.sliding_window_features import FEATURE_COLUMN_NAMES
from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier
from physics_validation.invariant_rule_evaluator import PhysicalInvariantValidator


def evaluate_physics_only(feature_dict: Dict[str, float]) -> str:
    """Standalone deterministic rule engine across 10 classes."""
    qber = feature_dict.get("qber", 0.02)
    skr = feature_dict.get("skr_bps", 10000.0)
    vis = feature_dict.get("visibility", 0.985)
    cnt = feature_dict.get("raw_counts_hz", 2.3e6)
    dcr = feature_dict.get("dark_counts_hz", 500.0)
    temp = feature_dict.get("temperature_celsius", -40.0)
    jitter = feature_dict.get("timing_jitter_ps", 65.0)
    loss = feature_dict.get("channel_attenuation_db", 5.0)
    
    e_opt = (1.0 - vis) / 2.0
    expected_qber = e_opt + (0.5 * (dcr / 1e8)) / max(1e-12, cnt / 1e8)
    surplus = qber - expected_qber
    
    # Attack signatures
    if cnt > 1.0e7 and qber < 0.020:
        return "Detector Blinding"
    if skr == 0.0 and qber < 0.060 and cnt > 1.0e5:
        return "Photon Number Splitting"
    if jitter > 120.0 and qber > 0.065 and vis >= 0.960 and temp <= -35.0:
        return "Time-Shift Attack"
    if surplus > 0.020 and vis >= 0.960 and dcr <= 1500.0 and cnt >= 1.0e6:
        return "Intercept-Resend"
        
    # Fault signatures
    if cnt < 1.2e6 and loss > 6.5:
        return "Channel Attenuation Event"
    if vis < 0.960:
        return "Optical Misalignment"
    if temp > -32.0 and dcr > 1500.0:
        return "Thermal Drift"
    if dcr > 1200.0 and temp <= -35.0:
        return "Detector APD Degradation"
    if jitter > 110.0:
        return "Timing Jitter"
    return "Normal"


def run_ablation_study(
    classifier: LightGBMRootCauseClassifier,
    validator: PhysicalInvariantValidator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Executes empirical ablation study on the test split.
    """
    y_true = np.array(y_test)
    
    preds_ml_only = []
    preds_physics_only = []
    preds_hybrid = []
    
    for i in range(len(X_test)):
        feats = {col: float(X_test[i, idx]) for idx, col in enumerate(FEATURE_COLUMN_NAMES)}
        
        # 1. ML Only
        attr_raw = classifier.predict_sample(feature_dict=feats, physics_validation=None)
        preds_ml_only.append(ROOT_CAUSE_LABEL_TO_ID.get(attr_raw.predicted_class, 0))
        
        # 2. Physics Only
        p_class = evaluate_physics_only(feats)
        preds_physics_only.append(ROOT_CAUSE_LABEL_TO_ID.get(p_class, 0))
        
        # 3. Hybrid Q-Sentinel
        phys_val = validator.evaluate(feats, ml_predicted_class=attr_raw.predicted_class, ml_confidence=attr_raw.confidence)
        attr_hybrid = classifier.predict_sample(feature_dict=feats, physics_validation=phys_val)
        preds_hybrid.append(ROOT_CAUSE_LABEL_TO_ID.get(attr_hybrid.predicted_class, 0))
        
    # Metrics calculation
    def calc_metrics(y_pred):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        # False Positive Rate on Normal class
        norm_idx = ROOT_CAUSE_LABEL_TO_ID["Normal"]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ROOT_CAUSE_CLASSES))))
        fp = np.sum(cm[:, norm_idx]) - cm[norm_idx, norm_idx]
        tn = np.sum(cm) - np.sum(cm[norm_idx, :]) - np.sum(cm[:, norm_idx]) + cm[norm_idx, norm_idx]
        fpr = fp / max(1, fp + tn)
        return float(f1), float(fpr)
        
    f1_ml, fpr_ml = calc_metrics(preds_ml_only)
    f1_phys, fpr_phys = calc_metrics(preds_physics_only)
    f1_hyb, fpr_hyb = calc_metrics(preds_hybrid)
    
    table = [
        {
            "Architecture": "ML Only (LightGBM Standalone)",
            "Macro_F1": round(f1_ml, 4),
            "False_Positive_Rate": round(fpr_ml, 4),
            "Explainability_Depth": "Medium (Feature Weights Only)",
        },
        {
            "Architecture": "Physics Only (Rule Heuristics)",
            "Macro_F1": round(f1_phys, 4),
            "False_Positive_Rate": round(fpr_phys, 4),
            "Explainability_Depth": "High (Rule Determinism, No Probability)",
        },
        {
            "Architecture": "Q-Sentinel Hybrid (ML + Physics Fusion)",
            "Macro_F1": round(f1_hyb, 4),
            "False_Positive_Rate": round(fpr_hyb, 4),
            "Explainability_Depth": "High (5-Point Physics + SHAP)",
        },
    ]
    
    return {
        "ablation_table": table,
        "f1_improvement_pct": round((f1_hyb - f1_ml) * 100.0, 2),
    }
