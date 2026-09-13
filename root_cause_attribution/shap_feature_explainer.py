"""
Phase 12: Explainability Engine (5-Point Explainability Interface + SHAP Local Feature Attribution)
Answers the five operational questions for every alert:
1. What happened?
2. Why?
3. Which measurements support it?
4. How confident are we?
5. What should the operator do?
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import shap
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from config.qkd_system_parameters import (
    ROOT_CAUSE_CLASSES,
    ROOT_CAUSE_LABEL_TO_ID,
    ROOT_CAUSE_ID_TO_LABEL,
    ALARM_SEVERITY_LEVELS,
)
from anomaly_detection.sliding_window_features import FEATURE_COLUMN_NAMES
from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier, RootCauseAttributionResult
from physics_validation.invariant_rule_evaluator import PhysicsValidationResult


@dataclass
class SHAPFeatureContribution:
    """Represents a single feature's local SHAP attribution value and domain interpretation."""
    feature_name: str
    feature_value: float
    shap_value: float
    direction: str
    physics_interpretation: str


@dataclass
class FivePointExplainabilityReport:
    """The 5-Point Explainability Report for QKD Network Operators."""
    what_happened: str                  # Q1: Diagnosis & Alarm Severity
    why_it_happened: str                # Q2: Physical mechanism
    supporting_measurements: List[str]  # Q3: Quantifiable physical evidence & top SHAP drivers
    confidence_assessment: str          # Q4: ML + Physics operational confidence
    recommended_action: str             # Q5: Advisory operator response
    alarm_severity: str                 # 'CRITICAL', 'MAJOR', 'HIGH', 'MEDIUM', 'NORMAL'
    top_shap_contributions: List[SHAPFeatureContribution]


# Action mapping across all 10 fault & attack classes
ADVISORY_ACTION_MAP = {
    "Intercept-Resend": "IMMEDIATE: Terminate current key distillation session, alert SOC, and isolate quantum channel.",
    "Detector Blinding": "IMMEDIATE: Engage optical shutter/variable optical attenuator (VOA) to block blinding light, reset APD bias, and flag incident.",
    "Photon Number Splitting": "IMMEDIATE: PNS attack detected via decoy-state collapse. Abort session, isolate link, and transition to decoy-state recalibration.",
    "Time-Shift Attack": "IMMEDIATE: Time-shift gating asymmetry detected. Randomize gating delays, recalibrate detector sync, and alert SOC.",
    "Detector APD Degradation": "ADVISORY: Inspect APD dark noise baseline, adjust overbias voltage, or switch to redundant receiver detector.",
    "Channel Attenuation Event": "ADVISORY: Reroute quantum transport to alternate optical fiber span (SPAN-B) and inspect physical fiber path.",
    "Optical Misalignment": "ADVISORY: Trigger automated motorized waveplate polarization re-alignment sweep.",
    "Thermal Drift": "ADVISORY: Retune TEC thermoelectric cooling current setpoint to restore -40.0 deg C.",
    "Timing Jitter": "ADVISORY: Execute laser sync delay line re-calibration and FPGA clock phase adjustment.",
    "Normal": "OPERATIONAL: System operating nominally within ETSI GS QKD 014 tolerances. Continue autonomous monitoring.",
}


class SHAPFeatureExplainer:
    """
    Explainability Engine integrating SHAP TreeExplainer with Phase 12 5-Point Explainability Interface.
    """

    def __init__(self, classifier: LightGBMRootCauseClassifier):
        self.classifier = classifier
        if not classifier.is_fitted:
            raise RuntimeError("Classifier must be fitted prior to initializing TreeExplainer.")
        self.explainer = shap.TreeExplainer(classifier.model)

    def explain_sample(
        self,
        attribution_result: RootCauseAttributionResult,
        physics_validation: PhysicsValidationResult,
        top_k: int = 3,
    ) -> FivePointExplainabilityReport:
        """
        Generates complete 5-Point Explainability Report for an incoming telemetry sample.
        """
        vector_2d = attribution_result.feature_vector.reshape(1, -1)
        shap_values = self.explainer.shap_values(vector_2d)
        
        pred_label = attribution_result.predicted_class
        class_idx = ROOT_CAUSE_LABEL_TO_ID.get(pred_label, 0)
        severity = ALARM_SEVERITY_LEVELS.get(pred_label, "NORMAL")
        
        # Handle SHAP multi-class format
        if isinstance(shap_values, list):
            class_shap = shap_values[class_idx][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            class_shap = shap_values[0, :, class_idx]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            class_shap = shap_values[0]
        else:
            class_shap = np.zeros(len(FEATURE_COLUMN_NAMES))
            
        top_indices = np.argsort(np.abs(class_shap))[::-1][:top_k]
        top_contributions: List[SHAPFeatureContribution] = []
        meas_support_list: List[str] = []
        
        for idx in top_indices:
            feat_name = FEATURE_COLUMN_NAMES[idx]
            feat_val = float(attribution_result.feature_vector[idx])
            s_val = float(class_shap[idx])
            direction = "Increases likelihood of fault" if s_val > 0 else "Suppresses fault probability"
            
            top_contributions.append(
                SHAPFeatureContribution(
                    feature_name=feat_name,
                    feature_value=round(feat_val, 4),
                    shap_value=round(s_val, 4),
                    direction=direction,
                    physics_interpretation=f"{feat_name} = {feat_val:.4g} (SHAP impact: {s_val:+.3f})",
                )
            )
            meas_support_list.append(f"{feat_name}: {feat_val:.4g} (SHAP attribution {s_val:+.3f})")
            
        # 1. What happened?
        what = f"Alarm [{severity}]: {pred_label} identified on quantum link."
        
        # 2. Why?
        why = physics_validation.explanation
        
        # 3. Which measurements support it?
        ev = physics_validation.physical_evidence
        meas_support_list.append(f"QBER: {ev.get('measured_qber', 0)*100:.2f}% | Visibility: {ev.get('visibility_measured', 0)*100:.1f}% | APD Temp: {ev.get('detector_temp_celsius', 0)} C")
        
        # 4. How confident are we?
        conf_str = (
            f"Operational Confidence: {physics_validation.operational_confidence_tier} "
            f"(ML Confidence: {attribution_result.confidence*100:.1f}%, "
            f"Physics Consistency: {physics_validation.validation_status} [{physics_validation.physics_consistency_score*100:.0f}%])"
        )
        
        # 5. What should the operator do?
        action = ADVISORY_ACTION_MAP.get(pred_label, "Inspect link.")
        
        return FivePointExplainabilityReport(
            what_happened=what,
            why_it_happened=why,
            supporting_measurements=meas_support_list,
            confidence_assessment=conf_str,
            recommended_action=action,
            alarm_severity=severity,
            top_shap_contributions=top_contributions,
        )
