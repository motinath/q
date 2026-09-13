"""
Phase 8: Always-On ML Anomaly Detection (Isolation Forest)
Runs continuously on 100% of telemetry samples. Trained on the healthy baseline manifold.
Outputs: continuous anomaly score [0.0, 1.0] and operational state (Nominal / Warning / Anomalous).
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


class IsolationForestAnomalyDetector:
    """
    Continuous ML Anomaly Detector for QKD Networks.
    Learns normal operating manifold from healthy baseline telemetry.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 60,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
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
