"""
Phase 8: Always-On ML Anomaly Detection (Isolation Forest)
Runs continuously on 100% of telemetry samples. Trained on the healthy baseline manifold.
Outputs: continuous anomaly score [0.0, 1.0] and operational state (Nominal / Warning / Anomalous).

P2.3: Added `predict_sample_with_baseline_gate` which supplements the Isolation Forest
      score with a deterministic 3-sigma envelope check on the adaptive baselines from
      AdaptiveBaselineEngine.  Any feature outside its 3-sigma envelope elevates the
      operational state to at least "Warning" even when the IF score stays below 0.40.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anomaly_detection.sliding_window_features import FEATURE_COLUMN_NAMES


@dataclass
class AnomalyDetectionResult:
    """Represents the output of Layer 2 ML Anomaly Detection."""
    is_anomaly: bool
    operational_state: str        # 'Nominal', 'Warning', 'Anomalous'
    anomaly_score: float          # Normalized score in [0.0, 1.0]
    raw_decision_score: float     # Direct IsolationForest decision_function value
    telemetry_features: Dict[str, float]
    baseline_violations: Dict[str, float] = None   # P2.3: features outside 3-sigma envelope


class IsolationForestAnomalyDetector:
    """
    Continuous ML Anomaly Detector for QKD Networks.
    Learns normal operating manifold from healthy baseline telemetry.
    """

    def __init__(
        self,
        contamination: float = "auto",   # P2.4: was 0.05 (biases boundary on clean data)
        n_estimators: int = 60,
        random_state: int = 42,
        decision_threshold: float = 0.52,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.decision_threshold = decision_threshold

        self.scaler: StandardScaler = StandardScaler()
        self.model: IsolationForest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.is_fitted: bool = False

    def fit_healthy_baseline(self, X_healthy_train: np.ndarray) -> None:
        """
        Fits the scaler and Isolation Forest model on healthy/nominal baseline data only.

        Args:
            X_healthy_train: 2D array of shape (n_samples, n_features) from nominal runs.
        """
        if len(X_healthy_train) < 10:
            raise ValueError(f"Insufficient healthy training samples ({len(X_healthy_train)}).")

        X_scaled = self.scaler.fit_transform(X_healthy_train)
        self.model.fit(X_scaled)
        self.is_fitted = True

    def fit(self, X_train: np.ndarray) -> None:
        """Compatibility alias for training."""
        self.fit_healthy_baseline(X_train)

    def predict_sample(self, feature_dict: Dict[str, float]) -> AnomalyDetectionResult:
        """
        Runs real-time anomaly inference on a single telemetry sample.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be trained or loaded before prediction.")

        vector = np.array([[feature_dict[col] for col in FEATURE_COLUMN_NAMES]], dtype=np.float64)
        vector_scaled = self.scaler.transform(vector)

        # Raw score: higher is normal, lower is anomalous
        raw_score = float(self.model.decision_function(vector_scaled)[0])

        # Normalized anomaly score [0.0, 1.0] where >0.5 indicates anomaly
        normalized_score = float(np.clip(0.50 - (raw_score * 2.5), 0.0, 1.0))

        # Determine 3-tier operational state
        if normalized_score >= 0.60:
            state = "Anomalous"
            is_anom = True
        elif normalized_score >= 0.40:
            state = "Warning"
            is_anom = False
        else:
            state = "Nominal"
            is_anom = False

        return AnomalyDetectionResult(
            is_anomaly=is_anom,
            operational_state=state,
            anomaly_score=normalized_score,
            raw_decision_score=raw_score,
            telemetry_features=feature_dict,
            baseline_violations={},
        )

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns normalized anomaly scores in [0, 1] for batch array X."""
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be trained or loaded before prediction.")
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        return np.clip(0.50 - (raw_scores * 2.5), 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Returns binary anomaly predictions (1 for anomaly, 0 for nominal) for batch array X."""
        scores = self.score_samples(X)
        th = threshold if threshold is not None else getattr(self, "decision_threshold", 0.52)
        return (scores >= th).astype(int)

    def predict_sample_with_baseline_gate(
        self,
        feature_dict: Dict[str, float],
        adaptive_envelopes: Optional[Dict[str, Any]] = None,
    ) -> AnomalyDetectionResult:
        """
        P2.3: Runs Isolation Forest inference PLUS a deterministic 3-sigma
        adaptive-baseline envelope check.

        If any monitored feature exceeds its current 3-sigma envelope, the result is
        elevated to at least "Warning" (anomaly_score bumped to 0.50) even when the
        Isolation Forest score alone would say "Nominal".  If two or more features
        are outside their envelopes the result is elevated to "Anomalous".

        Args:
            feature_dict:       Current telemetry feature dictionary.
            adaptive_envelopes: Dict[feature_name -> ChannelBaselineEnvelope] from
                                AdaptiveBaselineEngine.update(). May be None or empty.

        Returns:
            AnomalyDetectionResult with baseline_violations populated.
        """
        result = self.predict_sample(feature_dict)

        if not adaptive_envelopes:
            return result

        # Features to gate (exclude noisy ratio features that fluctuate legitimately)
        GATED_FEATURES = {
            "qber", "skr_bps", "raw_counts_hz", "dark_counts_hz",
            "visibility", "temperature_celsius", "timing_jitter_ps",
            "channel_attenuation_db", "qber_roll_mean_25",
        }

        violations: Dict[str, float] = {}
        for feat_name in GATED_FEATURES:
            envelope = adaptive_envelopes.get(feat_name)
            if envelope is None or not envelope.is_calibrated:
                continue
            val = feature_dict.get(feat_name)
            if val is None:
                continue
            lo = envelope.lower_3sigma
            hi = envelope.upper_3sigma
            if lo is not None and hi is not None:
                if val < lo or val > hi:
                    # Store signed deviation in sigma units
                    std_val = getattr(envelope, "std_dev", getattr(envelope, "std", 1e-10))
                    ewma_val = getattr(envelope, "ewma_value", getattr(envelope, "ewma", getattr(envelope, "mean", 0.0)))
                    sigma = max(1e-10, std_val)
                    violations[feat_name] = (val - ewma_val) / sigma

        n_violations = len(violations)

        # Elevate state based on envelope violations
        anomaly_score = result.anomaly_score
        state = result.operational_state
        is_anom = result.is_anomaly

        if n_violations >= 2 and state == "Nominal":
            # Multiple simultaneous envelope breaches → Anomalous
            anomaly_score = max(0.65, anomaly_score)
            state = "Anomalous"
            is_anom = True
        elif n_violations >= 1 and state == "Nominal":
            # Single envelope breach → Warning
            anomaly_score = max(0.50, anomaly_score)
            state = "Warning"
            # is_anomaly stays False for single Warning (keeps existing semantics)

        return AnomalyDetectionResult(
            is_anomaly=is_anom,
            operational_state=state,
            anomaly_score=anomaly_score,
            raw_decision_score=result.raw_decision_score,
            telemetry_features=feature_dict,
            baseline_violations=violations,
        )

    def save(self, model_path: str, scaler_path: str) -> None:
        """Serializes model and scaler artifacts."""
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path: str, scaler_path: str) -> None:
        """Loads serialized model and scaler."""
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model or scaler not found at {model_path} / {scaler_path}")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_fitted = True

