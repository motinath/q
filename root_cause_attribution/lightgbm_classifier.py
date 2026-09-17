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
        n_estimators: int = 250,
        learning_rate: float = 0.08,
        max_depth: int = 7,
        num_leaves: int = 63,
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
            subsample=0.85,
            colsample_bytree=0.85,
            subsample_freq=1,
            random_state=self.random_state,
            objective="multiclass",
            num_class=len(ROOT_CAUSE_CLASSES),
            verbosity=-1,
            n_jobs=1,
        )
        self.is_fitted: bool = False
        self.class_centroids: Dict[int, np.ndarray] = {}
        self.class_stds: Dict[int, np.ndarray] = {}
        self.ood_threshold: float = 3.0
        self.unknown_confidence_threshold: float = 0.50
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

        # Compute empirical centroids and dispersions for OOD gating
        unique_classes = np.unique(y_train)
        self.class_centroids = {c: np.mean(X_train[y_train == c], axis=0) for c in unique_classes}
        self.class_stds = {c: np.std(X_train[y_train == c], axis=0) + 1e-5 for c in unique_classes}
        in_dist_dists = [
            min(np.mean(np.abs(X_train[i] - self.class_centroids[c]) / self.class_stds[c]) for c in unique_classes)
            for i in range(min(400, len(X_train)))
        ]
        if in_dist_dists:
            self.ood_threshold = float(np.percentile(in_dist_dists, 99.0) * 1.5)

    def is_ood(self, x: np.ndarray) -> bool:
        """Determines if a feature vector lies out-of-distribution relative to known training clusters."""
        if not self.class_centroids:
            return False
        min_d = min(
            np.mean(np.abs(x - self.class_centroids[c]) / self.class_stds[c])
            for c in self.class_centroids
        )
        return bool(min_d > self.ood_threshold)

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

        model_classes = list(self.model.classes_)
        for c in range(n_classes):
            # Binary label: 1 if true class == c, else 0
            binary_labels = (y_val == c).astype(float)
            if c in model_classes and binary_labels.sum() >= 2:
                col_idx = model_classes.index(c)
                scores = raw_probs[:, col_idx]
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(scores, binary_labels)
                self._calibrators.append(ir)
            else:
                self._calibrators.append(None)

        self.is_calibrated = True

    def _apply_calibration(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        P3.3: Applies per-class isotonic calibration then re-normalizes.
        Falls back to raw probabilities when calibrators are not fitted or degenerate.
        """
        if not self.is_calibrated or self._calibrators is None:
            return raw_probs
        cal = np.zeros(len(ROOT_CAUSE_CLASSES), dtype=np.float64)
        n_apply = min(len(self._calibrators), len(raw_probs), len(ROOT_CAUSE_CLASSES))
        for c in range(n_apply):
            if self._calibrators[c] is not None:
                cal[c] = float(self._calibrators[c].predict([raw_probs[c]])[0])
            else:
                cal[c] = raw_probs[c]
        total = cal.sum()
        if total > 0.05:
            cal /= total
        else:
            cal = raw_probs.copy()
            total_raw = cal.sum()
            if total_raw > 0:
                cal /= total_raw
            else:
                cal = np.ones(len(ROOT_CAUSE_CLASSES)) / len(ROOT_CAUSE_CLASSES)
        return cal

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities for batch feature matrix X (shape: n_samples, n_features)."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted or loaded before prediction.")
        model_probs = self.model.predict_proba(X)
        n_samples = len(X)
        n_classes = len(ROOT_CAUSE_CLASSES)
        
        full_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
        for idx, c in enumerate(self.model.classes_):
            if c < n_classes:
                full_probs[:, c] = model_probs[:, idx]

        calibrated = np.zeros_like(full_probs)
        for i in range(n_samples):
            if self.is_calibrated and self._calibrators is not None:
                p = self._apply_calibration(full_probs[i])
            else:
                p = full_probs[i].copy()
            if self.is_ood(X[i]):
                p = np.ones_like(p) / len(p)
                if "Unknown Fault" in ROOT_CAUSE_LABEL_TO_ID:
                    unk_idx = ROOT_CAUSE_LABEL_TO_ID["Unknown Fault"]
                    p[:] = 0.01 / (len(p) - 1)
                    p[unk_idx] = 0.99
            calibrated[i] = p
        return calibrated

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
        model_probs = self.model.predict_proba(vector)[0]
        n_classes = len(ROOT_CAUSE_CLASSES)
        full_raw_probs = np.zeros(n_classes, dtype=np.float64)
        for idx, c in enumerate(self.model.classes_):
            if c < n_classes:
                full_raw_probs[c] = model_probs[idx]

        # P3.3: Apply isotonic calibration when available
        calibrated_base = self._apply_calibration(full_raw_probs)

        raw_prob_dict = {
            ROOT_CAUSE_ID_TO_LABEL[i]: float(full_raw_probs[i])
            for i in range(len(ROOT_CAUSE_CLASSES))
        }
        
        # Physics Verification Step
        calibrated_probs = calibrated_base.copy()
        physics_adjustment = 0.0
        is_verified = False

        if physics_validation is not None:
            mod = physics_validation.physics_confidence_modifier
            sig = physics_validation.primary_physical_signature

            # Signature-to-class alignment supporting both official 9 classes and legacy labels
            target_class = None
            if "Temperature Drift" in sig or "Thermal" in sig:
                target_class = "Temperature Drift" if "Temperature Drift" in ROOT_CAUSE_LABEL_TO_ID else "Thermal Drift"
            elif "Fiber Bend" in sig or "Attenuation" in sig:
                target_class = "Fiber Bend" if "Fiber Bend" in ROOT_CAUSE_LABEL_TO_ID else "Channel Attenuation Event"
            elif "Polarization Drift" in sig or "Misalignment" in sig:
                target_class = "Polarization Drift" if "Polarization Drift" in ROOT_CAUSE_LABEL_TO_ID else "Optical Misalignment"
            elif "Aging" in sig or "Detector Aging" in sig or "Trap" in sig:
                target_class = "Detector Aging" if "Detector Aging" in ROOT_CAUSE_LABEL_TO_ID else "Detector APD Degradation"
            elif "Power Instability" in sig:
                target_class = "Power Instability"
            elif "Humidity" in sig:
                target_class = "Humidity Impact"
            elif "Timing" in sig or "Jitter" in sig:
                target_class = "Timing Misalignment" if "Timing Misalignment" in ROOT_CAUSE_LABEL_TO_ID else "Timing Jitter"
            elif "Intercept-Resend" in sig:
                target_class = "Intercept-Resend"
            elif "Blinding" in sig or "Saturation" in sig:
                target_class = "Detector Blinding"
            elif "PNS" in sig or "Splitting" in sig or "Yield Collapse" in sig:
                target_class = "Photon Number Splitting"
            elif "Time-Shift" in sig:
                target_class = "Time-Shift Attack"
            elif "Nominal" in sig or "False Positive" in sig or "Baseline" in sig:
                target_class = "Normal"

            if target_class is not None and target_class in ROOT_CAUSE_LABEL_TO_ID:
                idx = ROOT_CAUSE_LABEL_TO_ID[target_class]
                likelihood = np.ones(len(ROOT_CAUSE_CLASSES), dtype=np.float64)
                likelihood[idx] = max(0.01, 1.0 + mod)
                calibrated_probs = full_raw_probs * likelihood
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

        # Selective Rejection for Out-of-Distribution / Unknown Faults
        if (self.is_ood(vector[0]) or confidence < getattr(self, "unknown_confidence_threshold", 0.50)) and not is_verified:
            if "Unknown Fault" in ROOT_CAUSE_LABEL_TO_ID:
                pred_label = "Unknown Fault"
        
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
        Saves OOD centroids and thresholds (as model_path + '.meta.joblib').
        """
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump(self.model, model_path)
        if self.is_calibrated and self._calibrators is not None:
            cal_path = model_path + ".cal.joblib"
            joblib.dump(self._calibrators, cal_path)
        meta = {
            "class_centroids": getattr(self, "class_centroids", {}),
            "class_stds": getattr(self, "class_stds", {}),
            "ood_threshold": getattr(self, "ood_threshold", 3.0),
            "unknown_confidence_threshold": getattr(self, "unknown_confidence_threshold", 0.50),
        }
        joblib.dump(meta, model_path + ".meta.joblib")

    def load(self, model_path: str) -> None:
        """Loads serialized LightGBM model from disk.
        P3.3: Automatically loads calibrators if the companion .cal.joblib file exists.
        Loads OOD centroids and thresholds if .meta.joblib exists.
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
        # Try loading OOD metadata
        meta_path = model_path + ".meta.joblib"
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.class_centroids = meta.get("class_centroids", {})
            self.class_stds = meta.get("class_stds", {})
            self.ood_threshold = meta.get("ood_threshold", 3.0)
            self.unknown_confidence_threshold = meta.get("unknown_confidence_threshold", 0.50)


# ==============================================================================
# CHALLENGE EXTENSION: Multi-Label Diagnosis & 3 Operational States
# ==============================================================================

@dataclass
class MultiLabelRootCauseResult:
    """Represents the output of Layer 4 Multi-Label Root-Cause Attribution."""
    operational_state: str                # "Normal", "Diagnosed degradation", "Insufficient evidence"
    active_faults: List[str]              # List of active fault names diagnosed (can be multiple)
    fault_probabilities: Dict[str, float] # Raw model probabilities per fault
    calibrated_probabilities: Dict[str, float] # Isotonic calibrated probabilities
    is_anomaly: bool                      # Whether an anomaly flag was asserted
    is_ood: bool                          # Whether feature vector is OOD
    ood_distance: float                   # Distance to nearest training cluster
    feature_vector: np.ndarray
    details: Dict[str, Any]


CHALLENGE_FAULT_KEYS = [
    "thermal_drift",
    "misalignment",
    "channel_loss",
]

CHALLENGE_FAULT_LABELS = {
    "thermal_drift": "Thermal Drift",
    "misalignment": "Optical Misalignment",
    "channel_loss": "Increased Channel Loss",
}

CHALLENGE_FAULT_TARGET_COLS = {
    "thermal_drift": "has_thermal_drift",
    "misalignment": "has_misalignment",
    "channel_loss": "has_channel_loss",
}


class MultiLabelRootCauseClassifier:
    """
    Layer 4 Multi-Label Root-Cause Classifier for concurrent and single faults.
    Trains one calibrated binary LightGBM classifier per independent physical fault:
      - Thermal Drift (TEC drift)
      - Optical Misalignment (polarization EPC drift)
      - Increased Channel Loss (fiber attenuation/bend)

    Supports:
      1. Single faults (e.g. Thermal Drift alone)
      2. Concurrent combined faults (e.g. Thermal Drift + Misalignment simultaneously)
      3. Operational state classification:
         - "Normal" (no anomaly, no faults)
         - "Diagnosed degradation" (anomaly detected & at least one fault diagnosed)
         - "Insufficient evidence" (anomaly detected, but no known fault >= threshold or OOD)
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

        self.models: Dict[str, lgb.LGBMClassifier] = {}
        self.calibrators: Dict[str, IsotonicRegression] = {}
        self.is_fitted: bool = False
        self.is_calibrated: bool = False

        self.class_centroids: Dict[str, np.ndarray] = {}
        self.class_stds: Dict[str, np.ndarray] = {}
        self.ood_threshold: float = 3.5

    def fit(self, X_train: np.ndarray, y_train_dict: Dict[str, np.ndarray]) -> None:
        """
        Fits binary LightGBM classifiers for each fault key.
        y_train_dict: mapping fault_key -> binary array (0 or 1).
        """
        for key in CHALLENGE_FAULT_KEYS:
            y = y_train_dict[key]
            clf = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                subsample=0.85,
                colsample_bytree=0.85,
                subsample_freq=1,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=1,
            )
            clf.fit(X_train, y)
            self.models[key] = clf

        self.is_fitted = True

        # Fit OOD reference clusters from in-distribution train data
        is_normal = np.ones(len(X_train), dtype=bool)
        for key in CHALLENGE_FAULT_KEYS:
            is_normal &= (y_train_dict[key] == 0)
        
        if np.sum(is_normal) > 0:
            self.class_centroids["normal"] = np.mean(X_train[is_normal], axis=0)
            self.class_stds["normal"] = np.std(X_train[is_normal], axis=0) + 1e-5

        for key in CHALLENGE_FAULT_KEYS:
            mask = y_train_dict[key] == 1
            if np.sum(mask) > 0:
                self.class_centroids[key] = np.mean(X_train[mask], axis=0)
                self.class_stds[key] = np.std(X_train[mask], axis=0) + 1e-5

        # Compute empirical 99th percentile of in-dist normalized distance
        dists = []
        for i in range(min(500, len(X_train))):
            d = min(
                np.mean(np.abs(X_train[i] - self.class_centroids[c]) / self.class_stds[c])
                for c in self.class_centroids
            )
            dists.append(d)
        if dists:
            self.ood_threshold = float(np.percentile(dists, 99.0) * 1.5)

    def fit_calibration(self, X_cal: np.ndarray, y_cal_dict: Dict[str, np.ndarray]) -> None:
        """Fits isotonic regression probability calibrators on calibration set."""
        if not self.is_fitted:
            raise RuntimeError("Models must be fitted before calibration.")

        for key in CHALLENGE_FAULT_KEYS:
            if key in self.models:
                raw_probs = self.models[key].predict_proba(X_cal)[:, 1]
                y_cal = y_cal_dict[key].astype(float)
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(raw_probs, y_cal)
                self.calibrators[key] = ir

        self.is_calibrated = True

    def compute_ood_distance(self, x: np.ndarray) -> float:
        """Computes min normalized Manhattan distance to any known cluster."""
        if not self.class_centroids:
            return 0.0
        return float(min(
            np.mean(np.abs(x - self.class_centroids[c]) / self.class_stds[c])
            for c in self.class_centroids
        ))

    def predict_sample(
        self,
        x: np.ndarray,
        is_anomaly: bool = False,
        threshold: float = 0.50,
    ) -> MultiLabelRootCauseResult:
        """Predicts operational state and active faults for a single feature vector."""
        x_2d = x.reshape(1, -1)
        raw_probs = {}
        cal_probs = {}
        active_faults = []

        for key in CHALLENGE_FAULT_KEYS:
            label = CHALLENGE_FAULT_LABELS[key]
            prob = float(self.models[key].predict_proba(x_2d)[0, 1])
            raw_probs[label] = prob

            if self.is_calibrated and key in self.calibrators:
                cal_p = float(self.calibrators[key].predict([prob])[0])
            else:
                cal_p = prob
            cal_probs[label] = cal_p

            if cal_p >= threshold:
                active_faults.append(label)

        ood_dist = self.compute_ood_distance(x)
        is_ood = bool(ood_dist > self.ood_threshold)

        # Operational State Logic
        if not is_anomaly and len(active_faults) == 0:
            operational_state = "Normal"
        elif is_anomaly:
            if is_ood or len(active_faults) == 0:
                operational_state = "Insufficient evidence"
                active_faults = []
            else:
                operational_state = "Diagnosed degradation"
        else:
            # Fault detected above threshold even if anomaly score was near boundary
            operational_state = "Diagnosed degradation"

        return MultiLabelRootCauseResult(
            operational_state=operational_state,
            active_faults=active_faults,
            fault_probabilities=raw_probs,
            calibrated_probabilities=cal_probs,
            is_anomaly=is_anomaly,
            is_ood=is_ood,
            ood_distance=ood_dist,
            feature_vector=x,
            details={"is_combined_fault": len(active_faults) > 1},
        )

    def predict_batch(
        self,
        X: np.ndarray,
        is_anomaly_flags: Optional[np.ndarray] = None,
        threshold: float = 0.50,
    ) -> List[MultiLabelRootCauseResult]:
        """Runs vectorized inference across a batch of samples."""
        n_samples = len(X)
        if is_anomaly_flags is None:
            is_anomaly_flags = np.zeros(n_samples, dtype=bool)

        fault_labels = [CHALLENGE_FAULT_LABELS[k] for k in CHALLENGE_FAULT_KEYS]
        raw_probs_matrix = {}
        cal_probs_matrix = {}

        for key in CHALLENGE_FAULT_KEYS:
            lbl = CHALLENGE_FAULT_LABELS[key]
            p_raw = self.models[key].predict_proba(X)[:, 1]
            raw_probs_matrix[lbl] = p_raw
            if self.is_calibrated and key in self.calibrators:
                p_cal = self.calibrators[key].predict(p_raw)
            else:
                p_cal = p_raw
            cal_probs_matrix[lbl] = p_cal

        if self.class_centroids:
            cluster_dists = []
            for c in self.class_centroids:
                mu = self.class_centroids[c]
                sigma = self.class_stds[c]
                d = np.mean(np.abs(X - mu) / sigma, axis=1)
                cluster_dists.append(d)
            ood_dists = np.min(np.column_stack(cluster_dists), axis=1)
        else:
            ood_dists = np.zeros(n_samples)

        is_oods = ood_dists > self.ood_threshold

        results = []
        for i in range(n_samples):
            raw_p = {lbl: float(raw_probs_matrix[lbl][i]) for lbl in fault_labels}
            cal_p = {lbl: float(cal_probs_matrix[lbl][i]) for lbl in fault_labels}
            active = [lbl for lbl in fault_labels if cal_p[lbl] >= threshold]
            anom = bool(is_anomaly_flags[i])
            ood = bool(is_oods[i])

            if not anom and len(active) == 0:
                op_state = "Normal"
            elif anom:
                if ood or len(active) == 0:
                    op_state = "Insufficient evidence"
                    active = []
                else:
                    op_state = "Diagnosed degradation"
            else:
                op_state = "Diagnosed degradation"

            results.append(
                MultiLabelRootCauseResult(
                    operational_state=op_state,
                    active_faults=active,
                    fault_probabilities=raw_p,
                    calibrated_probabilities=cal_p,
                    is_anomaly=anom,
                    is_ood=ood,
                    ood_distance=float(ood_dists[i]),
                    feature_vector=X[i],
                    details={"is_combined_fault": len(active) > 1},
                )
            )
        return results

    def save(self, file_path: str) -> None:
        """Serializes models, calibrators, and metadata."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        payload = {
            "models": self.models,
            "calibrators": self.calibrators,
            "is_calibrated": self.is_calibrated,
            "class_centroids": self.class_centroids,
            "class_stds": self.class_stds,
            "ood_threshold": self.ood_threshold,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "random_state": self.random_state,
        }
        joblib.dump(payload, file_path)

    def load(self, file_path: str) -> None:
        """Loads serialized models, calibrators, and metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact not found at {file_path}")
        payload = joblib.load(file_path)
        self.models = payload["models"]
        self.calibrators = payload.get("calibrators", {})
        self.is_calibrated = payload.get("is_calibrated", False)
        self.class_centroids = payload.get("class_centroids", {})
        self.class_stds = payload.get("class_stds", {})
        self.ood_threshold = payload.get("ood_threshold", 3.5)
        self.is_fitted = True

