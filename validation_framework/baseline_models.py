"""
VECTOR-Q Baseline Models Suite
Phase 3: Standard Classical & Statistical Baseline Competitors
Governing Standards: ETSI GS QKD 014 / Montgomery SPC / NIST Statistical Standards

Implements:
1. Anomaly Detection: One-Class SVM, EWMA Control Chart, CUSUM Control Chart, Isolation Forest
2. Root Cause Attribution: Random Forest Classifier, Gradient Boosting Classifier, LightGBM
3. Predictive Maintenance: Linear Extrapolation, AR(p) Forecaster, Holt's Linear Trend

Author: Senior Quantum Systems & Applied ML Engineering Team
Version: 3.0.0
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.dataset_governance import OFFICIAL_FAULT_CLASSES, FAULT_LABEL_TO_ID


# ==============================================================================
# 1. ANOMALY DETECTION BASELINES
# ==============================================================================

class OneClassSVManomalyDetector:
    """
    Classical One-Class Support Vector Machine (Schölkopf et al., 2001)
    Non-linear RBF kernel outlier detector.
    """

    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale"):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "OneClassSVManomalyDetector":
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for anomaly, 0 for nominal (inverting scikit-learn convention -1/1)."""
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly score (higher means more anomalous)."""
        # decision_function returns negative for outliers, positive for inliers
        return -self.model.decision_function(X)


class EWMADetector:
    """
    Exponentially Weighted Moving Average (EWMA) Statistical Process Control Chart
    Detects small persistent mean shifts in QBER according to Montgomery (2009).
    """

    def __init__(self, lambda_weight: float = 0.20, l_sigma: float = 3.0):
        self.lambda_weight = lambda_weight
        self.l_sigma = l_sigma
        self.mu_0: float = 0.02
        self.sigma_0: float = 0.001
        self.z_t: float = self.mu_0
        self.step_t: int = 0
        self.is_fitted: bool = False

    def fit(self, qber_series: np.ndarray) -> "EWMADetector":
        """Calibrates nominal in-control mean and standard deviation."""
        self.mu_0 = float(np.mean(qber_series))
        self.sigma_0 = float(max(1e-5, np.std(qber_series)))
        self.z_t = self.mu_0
        self.step_t = 0
        self.is_fitted = True
        return self

    def update(self, qber_val: float) -> Tuple[bool, float, float]:
        """
        Updates EWMA state and returns (is_anomaly, z_t, ucl).
        """
        self.step_t += 1
        lam = self.lambda_weight
        self.z_t = lam * qber_val + (1.0 - lam) * self.z_t
        # Time-varying Upper Control Limit
        term = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * self.step_t))
        ucl = self.mu_0 + self.l_sigma * self.sigma_0 * np.sqrt(max(1e-10, term))
        is_anomaly = bool(self.z_t > ucl)
        return is_anomaly, float(self.z_t), float(ucl)

    def predict(self, qber_series: np.ndarray) -> np.ndarray:
        """Batch evaluation of an array of QBER values."""
        anomalies = []
        z = self.mu_0
        lam = self.lambda_weight
        for t, x in enumerate(qber_series, start=1):
            z = lam * x + (1.0 - lam) * z
            term = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * t))
            ucl = self.mu_0 + self.l_sigma * self.sigma_0 * np.sqrt(max(1e-10, term))
            anomalies.append(1 if z > ucl else 0)
        return np.array(anomalies, dtype=int)


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) Quality Control Chart
    Detects sudden or gradual positive shifts in error rates with fast reaction time.
    """

    def __init__(self, k_slack_factor: float = 0.5, h_threshold_factor: float = 4.0):
        self.k_slack_factor = k_slack_factor
        self.h_threshold_factor = h_threshold_factor
        self.mu_0: float = 0.02
        self.sigma_0: float = 0.001
        self.s_pos: float = 0.0
        self.is_fitted: bool = False

    def fit(self, qber_series: np.ndarray) -> "CUSUMDetector":
        self.mu_0 = float(np.mean(qber_series))
        self.sigma_0 = float(max(1e-5, np.std(qber_series)))
        self.s_pos = 0.0
        self.is_fitted = True
        return self

    def update(self, qber_val: float) -> Tuple[bool, float, float]:
        k = self.k_slack_factor * self.sigma_0
        h = self.h_threshold_factor * self.sigma_0
        self.s_pos = max(0.0, self.s_pos + (qber_val - self.mu_0 - k))
        is_anomaly = bool(self.s_pos > h)
        return is_anomaly, float(self.s_pos), float(h)

    def predict(self, qber_series: np.ndarray) -> np.ndarray:
        anomalies = []
        k = self.k_slack_factor * self.sigma_0
        h = self.h_threshold_factor * self.sigma_0
        s = 0.0
        for x in qber_series:
            s = max(0.0, s + (x - self.mu_0 - k))
            anomalies.append(1 if s > h else 0)
        return np.array(anomalies, dtype=int)


# ==============================================================================
# 2. ROOT CAUSE ATTRIBUTION BASELINES
# ==============================================================================

class RandomForestRCABaseline:
    """Random Forest Classifier baseline (Breiman, 2001) for 8-class fault classification."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRCABaseline":
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class GradientBoostingRCABaseline:
    """Gradient Boosted Decision Trees baseline (Friedman, 2001)."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, random_state: int = 42):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRCABaseline":
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


# ==============================================================================
# 3. PREDICTIVE MAINTENANCE / FORECASTING BASELINES
# ==============================================================================

class LinearTrendForecaster:
    """
    First-Order Linear Extrapolation Baseline.
    Projects QBER forward based on recent linear velocity: Q(t + dt) = Q(t) + v * dt.
    """

    def __init__(self, qber_limit: float = 0.11):
        self.qber_limit = qber_limit

    def forecast_ptct(self, current_qber: float, dqber_dt: float) -> Optional[float]:
        """Returns projected time-to-crossing in seconds, or None if not degrading."""
        if dqber_dt <= 1e-6:
            return None  # No positive degradation slope
        remaining = self.qber_limit - current_qber
        if remaining <= 0.0:
            return 0.0
        return float(remaining / dqber_dt)


class ARIMAForecaster:
    """
    Pure-Python Autoregressive AR(p) Forecaster on differenced series (d=1).
    Estimates parameters via Yule-Walker / Least Squares.
    """

    def __init__(self, p: int = 3, qber_limit: float = 0.11, max_horizon_steps: int = 300):
        self.p = p
        self.qber_limit = qber_limit
        self.max_horizon_steps = max_horizon_steps
        self.phi: np.ndarray = np.ones(p) / p

    def fit(self, history_qber: np.ndarray) -> "ARIMAForecaster":
        """Fits AR(p) coefficients on 1st differences of QBER history."""
        if len(history_qber) <= self.p + 2:
            return self
        diff = np.diff(history_qber)
        # Construct lag matrix
        n = len(diff) - self.p
        if n < 2:
            return self
        X = np.zeros((n, self.p))
        y = diff[self.p:]
        for i in range(n):
            X[i, :] = diff[i:i + self.p][::-1]
        try:
            phi, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.phi = phi
        except Exception:
            pass
        return self

    def forecast_ptct(self, history_qber: np.ndarray, dt_seconds: float = 1.0) -> Optional[float]:
        """Iteratively forecasts QBER differences forward until crossing qber_limit."""
        if len(history_qber) <= self.p + 1:
            return None
        current_q = history_qber[-1]
        if current_q >= self.qber_limit:
            return 0.0

        diff_hist = list(np.diff(history_qber)[-self.p:])
        curr = current_q
        for step in range(1, self.max_horizon_steps + 1):
            lags = np.array(diff_hist[-self.p:][::-1])
            d_next = float(np.dot(self.phi, lags))
            d_next = max(-0.01, min(0.01, d_next))  # Clip numerical divergence
            curr += d_next
            diff_hist.append(d_next)
            if curr >= self.qber_limit:
                return float(step * dt_seconds)

        return None


class HoltWintersLinearForecaster:
    """
    Holt's Linear Exponential Smoothing (Level + Trend, additive).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1, qber_limit: float = 0.11):
        self.alpha = alpha
        self.beta = beta
        self.qber_limit = qber_limit
        self.level: float = 0.02
        self.trend: float = 0.0

    def fit(self, history_qber: np.ndarray) -> "HoltWintersLinearForecaster":
        if len(history_qber) < 2:
            return self
        self.level = history_qber[0]
        self.trend = history_qber[1] - history_qber[0]
        for val in history_qber[1:]:
            prev_level = self.level
            self.level = self.alpha * val + (1.0 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1.0 - self.beta) * self.trend
        return self

    def forecast_ptct(self, dt_seconds: float = 1.0) -> Optional[float]:
        if self.level >= self.qber_limit:
            return 0.0
        if self.trend <= 1e-6:
            return None  # Flat or improving
        steps = (self.qber_limit - self.level) / self.trend
        return float(max(0.0, steps * dt_seconds))
