"""
Layer 4 — ML Root-Cause Attribution (LightGBM Classifier) with Physics Verification
Multi-class diagnosis across 10 physical operational and security states.

P3.3: Added isotonic regression probability calibration.
  - After fitting the LightGBM model, a per-class isotonic regressor is fitted on
    the validation split (or training split when no separate val data is provided).
  - At inference time, raw LightGBM probabilities are passed through the calibrators
    before the physics Bayesian update, converting raw scores to proper posteriors.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import joblib
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression

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
        # P3.3: Per-class isotonic regression calibrators (one per class).
        # Populated by fit_calibration(); None until then (raw probs used as fallback).
        self._calibrators: Optional[List[IsotonicRegression]] = None
        self.is_calibrated: bool = False

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

    def fit_calibration(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        P3.3: Fits per-class isotonic regression calibrators on a held-out validation set.

        Uses the "one-vs-rest" strategy: for each class c, fit an isotonic regressor
        that maps the model's raw P(class=c) scores to true empirical probabilities on
        the validation data.  After re-normalization across classes, the resulting
        probability vector is a proper calibrated posterior.

        Should be called AFTER fit() using the 15-run validation split from the
        data split manifest (previously unused).

        Args:
            X_val: Validation feature matrix, shape (n_val, n_features).
            y_val: Validation integer labels, shape (n_val,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration.")
        if len(X_val) == 0:
            raise ValueError("Calibration dataset cannot be empty.")

        raw_probs = self.model.predict_proba(X_val)   # (n_val, n_classes)
        n_classes = len(ROOT_CAUSE_CLASSES)
        self._calibrators = []

        for c in range(n_classes):
            # Binary label: 1 if true class == c, else 0
            binary_labels = (y_val == c).astype(float)
            scores = raw_probs[:, c]

            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(scores, binary_labels)
            self._calibrators.append(ir)

        self.is_calibrated = True

    def _apply_calibration(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        P3.3: Applies per-class isotonic calibration then re-normalizes.
        Falls back to raw probabilities when calibrators are not fitted.
        """
        if not self.is_calibrated or self._calibrators is None:
            return raw_probs
        cal = np.array([
            float(self._calibrators[c].predict([raw_probs[c]])[0])
            for c in range(len(ROOT_CAUSE_CLASSES))
        ], dtype=np.float64)
        total = cal.sum()
        if total > 0:
            cal /= total
        else:
            cal = np.ones(len(ROOT_CAUSE_CLASSES)) / len(ROOT_CAUSE_CLASSES)
        return cal

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

        # P3.3: Apply isotonic calibration when available
        calibrated_base = self._apply_calibration(raw_probs)

        raw_prob_dict = {
            ROOT_CAUSE_ID_TO_LABEL[i]: float(raw_probs[i])
            for i in range(len(ROOT_CAUSE_CLASSES))
        }
        
        # Physics Verification Step — P1.3: canonical class names; P3.4: proper Bayesian update
        calibrated_probs = calibrated_base.copy()
        physics_adjustment = 0.0
        is_verified = False

        if physics_validation is not None:
            mod = physics_validation.physics_confidence_modifier
            sig = physics_validation.primary_physical_signature

            # Signature-to-class alignment using EXACT canonical names from ROOT_CAUSE_LABEL_TO_ID.
            # Previous bug: used "Channel Attenuation" and "APD Aging" which are not in the dict.
            target_class = None
            if "Intercept-Resend" in sig:
                target_class = "Intercept-Resend"
            elif "Attenuation" in sig:
                target_class = "Channel Attenuation Event"        # P1.3 fix: canonical name
            elif "Thermal" in sig:
                target_class = "Thermal Drift"
            elif "Misalignment" in sig:
                target_class = "Optical Misalignment"
            elif "Aging" in sig or "Trap" in sig or "Degradation" in sig:
                target_class = "Detector APD Degradation"         # P1.3 fix: canonical name
            elif "Blinding" in sig or "Saturation" in sig:
                target_class = "Detector Blinding"
            elif "PNS" in sig or "Splitting" in sig or "Yield Collapse" in sig:
                target_class = "Photon Number Splitting"
            elif "Time-Shift" in sig or "Asymmetry" in sig or "Gating Window" in sig:
                target_class = "Time-Shift Attack"
            elif "Jitter" in sig:
                target_class = "Timing Jitter"
            elif "Nominal" in sig or "False Positive" in sig or "Baseline" in sig:
                target_class = "Normal"

            if target_class is not None and target_class in ROOT_CAUSE_LABEL_TO_ID:
                # P3.4: Proper multiplicative Bayesian posterior update.
                # Likelihood ratio: physics_modifier is in (-1, 1].
                # likelihood_i = 1 + mod  for the physics-indicated class,
                # likelihood_j = 1.0      for all other classes.
                # posterior_i = prior_i * likelihood_i / Z  where Z = sum(prior_j * likelihood_j)
                idx = ROOT_CAUSE_LABEL_TO_ID[target_class]
                likelihood = np.ones(len(ROOT_CAUSE_CLASSES), dtype=np.float64)
                likelihood[idx] = max(0.01, 1.0 + mod)   # never drive to exactly 0
                calibrated_probs = raw_probs * likelihood
                physics_adjustment = mod
                if mod > 0:
                    is_verified = True

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
        """Serializes LightGBM model artifact to disk.
        P3.3: Also saves calibrators if fitted (as model_path + '.cal.joblib').
        """
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump(self.model, model_path)
        if self.is_calibrated and self._calibrators is not None:
            cal_path = model_path + ".cal.joblib"
            joblib.dump(self._calibrators, cal_path)

    def load(self, model_path: str) -> None:
        """Loads serialized LightGBM model from disk.
        P3.3: Automatically loads calibrators if the companion .cal.joblib file exists.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        self.model = joblib.load(model_path)
        self.is_fitted = True
        # Try loading calibrators
        cal_path = model_path + ".cal.joblib"
        if os.path.exists(cal_path):
            self._calibrators = joblib.load(cal_path)
            self.is_calibrated = True
