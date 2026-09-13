"""
Layer 4 — ML Root-Cause Attribution (LightGBM Classifier) with Physics Verification
Multi-class diagnosis across 7 physical operational and security states.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import joblib
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from config.qkd_system_parameters import (
    ROOT_CAUSE_CLASSES,
    ROOT_CAUSE_LABEL_TO_ID,
    ROOT_CAUSE_ID_TO_LABEL,
)
from anomaly_detection.sliding_window_features import FEATURE_COLUMN_NAMES
from physics_validation.invariant_rule_evaluator import PhysicsValidationResult


@dataclass
class RootCauseAttributionResult:
    """Represents the output of Layer 4 Root-Cause Attribution."""
    predicted_class: str
    confidence: float                     # Calibrated confidence probability [0.0, 1.0]
    raw_probabilities: Dict[str, float]   # Unadjusted model probabilities
    calibrated_probabilities: Dict[str, float]  # Physics-adjusted probabilities
    is_physics_verified: bool
    physics_adjustment_applied: float
    feature_vector: np.ndarray


class LightGBMRootCauseClassifier:
    """
    LightGBM Multi-Class Classifier for QKD Anomaly Root-Cause Attribution.
    Trained over temporal telemetry features and integrated with Layer 3 physical verification.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state
        
        self.model: lgb.LGBMClassifier = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            objective="multiclass",
            num_class=len(ROOT_CAUSE_CLASSES),
            verbosity=-1,
            n_jobs=1,
        )
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fits LightGBM on training feature matrix and ground truth class labels.
        
        Args:
            X_train: Array of shape (n_samples, n_features)
            y_train: Integer array of shape (n_samples,) corresponding to ROOT_CAUSE_LABEL_TO_ID
        """
        if len(X_train) == 0:
            raise ValueError("Training dataset cannot be empty.")
            
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict_sample(
        self,
        feature_dict: Dict[str, float],
        physics_validation: Optional[PhysicsValidationResult] = None,
    ) -> RootCauseAttributionResult:
        """
        Infers the root-cause state for a single telemetry sample and applies physics verification.
        
        Args:
            feature_dict: Dictionary mapping feature names to numerical values.
            physics_validation: Optional Layer 3 verification result for Bayesian adjustment.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted or loaded before prediction.")
            
        vector = np.array([[feature_dict[col] for col in FEATURE_COLUMN_NAMES]], dtype=np.float64)
        raw_probs = self.model.predict_proba(vector)[0]
        
        raw_prob_dict = {
            ROOT_CAUSE_ID_TO_LABEL[i]: float(raw_probs[i])
            for i in range(len(ROOT_CAUSE_CLASSES))
        }
        
        # Physics Verification Step
        calibrated_probs = raw_probs.copy()
        physics_adjustment = 0.0
        is_verified = False
        
        if physics_validation is not None:
            mod = physics_validation.physics_confidence_modifier
            sig = physics_validation.primary_physical_signature
            
            # Signature-to-class alignment
            target_class = None
            if "Intercept-Resend" in sig:
                target_class = "Intercept-Resend"
            elif "Attenuation" in sig:
                target_class = "Channel Attenuation"
            elif "Thermal" in sig:
                target_class = "Thermal Drift"
            elif "Misalignment" in sig:
                target_class = "Optical Misalignment"
            elif "Aging" in sig or "Trap" in sig:
                target_class = "APD Aging"
            elif "Jitter" in sig:
                target_class = "Timing Jitter"
            elif "False Positive" in sig or "Nominal" in sig:
                target_class = "Normal"
                
            if target_class is not None and target_class in ROOT_CAUSE_LABEL_TO_ID:
                idx = ROOT_CAUSE_LABEL_TO_ID[target_class]
                if mod > 0:
                    calibrated_probs[idx] += mod
                    is_verified = True
                    physics_adjustment = mod
                elif mod < 0:
                    # Penalize target anomaly class, boost Normal
                    normal_idx = ROOT_CAUSE_LABEL_TO_ID["Normal"]
                    calibrated_probs[normal_idx] += abs(mod)
                    calibrated_probs[idx] = max(0.01, calibrated_probs[idx] - abs(mod))
                    physics_adjustment = mod
                    
        # Re-normalize probability vector
        sum_probs = np.sum(calibrated_probs)
        if sum_probs > 0:
            calibrated_probs = calibrated_probs / sum_probs
        else:
            calibrated_probs = np.ones_like(calibrated_probs) / len(calibrated_probs)
            
        pred_idx = int(np.argmax(calibrated_probs))
        pred_label = ROOT_CAUSE_ID_TO_LABEL[pred_idx]
        confidence = float(calibrated_probs[pred_idx])
        
        calibrated_dict = {
            ROOT_CAUSE_ID_TO_LABEL[i]: float(calibrated_probs[i])
            for i in range(len(ROOT_CAUSE_CLASSES))
        }
        
        return RootCauseAttributionResult(
            predicted_class=pred_label,
            confidence=confidence,
            raw_probabilities=raw_prob_dict,
            calibrated_probabilities=calibrated_dict,
            is_physics_verified=is_verified,
            physics_adjustment_applied=physics_adjustment,
            feature_vector=vector[0],
        )

    def save(self, model_path: str) -> None:
        """Serializes LightGBM model artifact to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump(self.model, model_path)

    def load(self, model_path: str) -> None:
        """Loads serialized LightGBM model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        self.model = joblib.load(model_path)
        self.is_fitted = True
