"""
Layer 6 — Predictive Maintenance (Projected Threshold Crossing Time - PTCT)
Calculates exact time-to-abort forecasts with confidence intervals and physical bounds.

P2.5: slope_standard_error is now passed from the actual least-squares fit residuals
      computed in TelemetryFeatureExtractor._compute_slope_with_se(), replacing the
      previous hardcoded default of 0.0002.

P3.1: Added quadratic PTCT mode.  When |qber_acceleration| > threshold, solves
      the quadratic equation:
          0 = 0.5 * a * t^2 + v * t + (qber_current - qber_limit)
      to give a more accurate forecast for nonlinear fault progression (e.g.,
      Arrhenius thermal runaway, exponential APD degradation).

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import math
import joblib
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from config.qkd_system_parameters import QKDPhysicsConfig


@dataclass
class PTCTForecastResult:
    """Represents the output of Layer 6 Predictive Maintenance."""
    t_cross_seconds: Optional[float]      # Projected seconds until QBER reaches limit (None if stable)
    t_cross_lower_bound_seconds: Optional[float]  # Worst-case 95% confidence bound
    t_cross_upper_bound_seconds: Optional[float]  # Best-case 95% confidence bound
    current_qber: float
    qber_abort_limit: float
    dqber_dt: float                       # Rate of change in QBER per second
    d2qber_dt2: float                     # Second derivative (acceleration) of QBER
    forecast_model: str                   # 'LINEAR' or 'QUADRATIC'
    forecast_trajectory_timestamps: List[float]
    forecast_trajectory_qber: List[float]
    urgency_level: str                    # 'CRITICAL', 'WARNING', 'ADVISORY', 'STABLE'
    recommendation_message: str


class ThresholdCrossingForecaster:
    """
    Projected Threshold Crossing Time (PTCT) Forecaster.
    Computes time remaining before QBER crosses the 11.0% GLLP security abort threshold.

    Selects between a linear (1st-order) and quadratic (2nd-order) forecast model
    based on the magnitude of the measured QBER acceleration.
    Strictly clips trajectories to physical limits: QBER in [0.0, 0.50].
    """

    # P3.1: Minimum |acceleration| to trigger quadratic model (units: fraction/s^2)
    QUADRATIC_ACCEL_THRESHOLD: float = 5e-6

    def __init__(self, config: Optional[QKDPhysicsConfig] = None):
        self.config = config or QKDPhysicsConfig()
        self.qber_limit = self.config.qber_abort_threshold
        self.qber_warning = self.config.qber_warning_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _linear_t_cross(self, qber: float, slope: float) -> Optional[float]:
        """t = (limit - qber) / slope.  Returns None when slope <= 0."""
        if slope <= 1e-6:
            return None
        return max(0.0, (self.qber_limit - qber) / slope)

    def _quadratic_t_cross(self, qber: float, slope: float, accel: float) -> Optional[float]:
        """
        Solves  0.5*a*t^2 + v*t + (qber - limit) = 0  for smallest positive real root.
        Falls back to linear if discriminant < 0 (trajectory never reaches limit).
        """
        c = qber - self.qber_limit          # constant term  (negative when qber < limit)
        b = slope                            # linear term
        a = 0.5 * accel                     # quadratic term

        if abs(a) < 1e-14:                  # Effectively linear
            return self._linear_t_cross(qber, slope)

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None                     # No real crossing in positive time

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b + sqrt_d) / (2.0 * a)
        t2 = (-b - sqrt_d) / (2.0 * a)

        # Pick smallest positive root
        candidates = [t for t in (t1, t2) if t > 0]
        if not candidates:
            return None
        return round(min(candidates), 1)

    def _build_trajectory(
        self,
        qber: float,
        slope: float,
        accel: float,
        horizon: float,
        step: float,
        use_quadratic: bool,
    ) -> Tuple[List[float], List[float]]:
        """Generates time/QBER trajectory arrays."""
        t_steps = list(np.arange(0.0, horizon + step / 2.0, step))
        if use_quadratic:
            q_traj = [
                float(np.clip(qber + slope * t + 0.5 * accel * t * t, 0.0, 0.50))
                for t in t_steps
            ]
        else:
            q_traj = [float(np.clip(qber + slope * t, 0.0, 0.50)) for t in t_steps]
        return t_steps, q_traj

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_ptct(
        self,
        current_qber: float,
        dqber_dt: float,
        slope_standard_error: float = 0.0002,   # P2.5: now passed from actual SE
        qber_acceleration: float = 0.0,          # P3.1: second derivative
        forecast_horizon_seconds: float = 60.0,
        forecast_step_seconds: float = 5.0,
    ) -> PTCTForecastResult:
        """
        Calculates PTCT and projects future QBER trajectory.

        Linear model (default):
            t_cross = (QBER_limit - QBER_current) / (dQBER/dt)

        Quadratic model (when |acceleration| > QUADRATIC_ACCEL_THRESHOLD):
            0 = 0.5 * a * t^2 + v * t + (QBER_current - QBER_limit)

        Confidence interval uses the actual slope SE (P2.5):
            slope_upper = dqber_dt + 1.96 * slope_se   → worst case (earlier crossing)
            slope_lower = max(0, dqber_dt - 1.96 * slope_se) → best case (later crossing)

        Args:
            current_qber:           Current measured QBER.
            dqber_dt:               Linear QBER slope (fraction/s) from feature extractor.
            slope_standard_error:   SE of slope from least-squares fit (P2.5).
            qber_acceleration:      Second derivative d^2QBER/dt^2 from feature extractor (P3.1).
            forecast_horizon_seconds: Projection window.
            forecast_step_seconds:  Trajectory time step.
        """
        current_qber = float(np.clip(current_qber, 0.0, 0.50))

        # Immediate critical condition
        if current_qber >= self.qber_limit:
            return PTCTForecastResult(
                t_cross_seconds=0.0,
                t_cross_lower_bound_seconds=0.0,
                t_cross_upper_bound_seconds=0.0,
                current_qber=current_qber,
                qber_abort_limit=self.qber_limit,
                dqber_dt=dqber_dt,
                d2qber_dt2=qber_acceleration,
                forecast_model="LINEAR",
                forecast_trajectory_timestamps=[0.0],
                forecast_trajectory_qber=[current_qber],
                urgency_level="CRITICAL",
                recommendation_message="CRITICAL: QBER has breached the 11.0% security abort limit. Key distillation terminated.",
            )

        # Stable or improving link
        if dqber_dt <= 1e-6:
            t_steps = list(np.arange(0, forecast_horizon_seconds + 1.0, forecast_step_seconds))
            q_traj = [current_qber for _ in t_steps]
            urgency = "WARNING" if current_qber >= self.qber_warning else "STABLE"
            msg = (
                f"Link operating nominally (QBER = {current_qber*100:.2f}%, "
                f"dQBER/dt = {dqber_dt*100:.4f}%/s). No threshold crossing projected."
            )
            return PTCTForecastResult(
                t_cross_seconds=None,
                t_cross_lower_bound_seconds=None,
                t_cross_upper_bound_seconds=None,
                current_qber=current_qber,
                qber_abort_limit=self.qber_limit,
                dqber_dt=dqber_dt,
                d2qber_dt2=qber_acceleration,
                forecast_model="LINEAR",
                forecast_trajectory_timestamps=t_steps,
                forecast_trajectory_qber=q_traj,
                urgency_level=urgency,
                recommendation_message=msg,
            )

        # P3.1: Decide between quadratic and linear model
        use_quadratic = abs(qber_acceleration) > self.QUADRATIC_ACCEL_THRESHOLD
        forecast_model = "QUADRATIC" if use_quadratic else "LINEAR"

        if use_quadratic:
            t_cross = self._quadratic_t_cross(current_qber, dqber_dt, qber_acceleration)
            if t_cross is None:
                # Quadratic predicts no crossing — fall back to linear
                t_cross = self._linear_t_cross(current_qber, dqber_dt)
                use_quadratic = False
                forecast_model = "LINEAR"
        else:
            t_cross = self._linear_t_cross(current_qber, dqber_dt)

        if t_cross is None:
            t_cross_final = None
            t_cross_lower = None
            t_cross_upper = None
        else:
            t_cross_final = round(t_cross, 1)

            # P2.5: Confidence interval using actual slope SE
            se = max(1e-7, slope_standard_error)
            slope_upper = dqber_dt + 1.96 * se
            slope_lower = max(1e-7, dqber_dt - 1.96 * se)

            if use_quadratic:
                t_lo = self._quadratic_t_cross(current_qber, slope_upper, qber_acceleration)
                t_hi = self._quadratic_t_cross(current_qber, slope_lower, qber_acceleration)
            else:
                t_lo = self._linear_t_cross(current_qber, slope_upper)
                t_hi = self._linear_t_cross(current_qber, slope_lower)

            t_cross_lower = round(max(0.0, t_lo), 1) if t_lo is not None else round(t_cross_final * 0.5, 1)
            t_cross_upper = round(t_hi, 1) if t_hi is not None else round(t_cross_final * 2.0, 1)

        # Build projected trajectory
        horizon = max(forecast_horizon_seconds, (t_cross_final or 0.0) * 1.2)
        t_steps, q_traj = self._build_trajectory(
            current_qber, dqber_dt, qber_acceleration, horizon,
            forecast_step_seconds, use_quadratic
        )

        # Urgency classification
        if t_cross_final is None:
            urgency = "ADVISORY"
        elif t_cross_final <= 15.0:
            urgency = "CRITICAL"
        elif t_cross_final <= 45.0:
            urgency = "WARNING"
        else:
            urgency = "ADVISORY"

        ci_str = (
            f"95% CI: [{t_cross_lower:.1f}s, {t_cross_upper:.1f}s]"
            if t_cross_final is not None else "stable"
        )
        msg = (
            f"URGENCY {urgency}: Projected QBER threshold breach in "
            f"{t_cross_final:.1f}s ({ci_str}). "
            f"Model: {forecast_model}. "
            f"Rate: {dqber_dt*100:.3f}%/s"
            + (f", Accel: {qber_acceleration*1e6:.3f}×10⁻⁶/s²." if use_quadratic else ".")
        )

        return PTCTForecastResult(
            t_cross_seconds=t_cross_final,
            t_cross_lower_bound_seconds=t_cross_lower,
            t_cross_upper_bound_seconds=t_cross_upper,
            current_qber=current_qber,
            qber_abort_limit=self.qber_limit,
            dqber_dt=dqber_dt,
            d2qber_dt2=qber_acceleration,
            forecast_model=forecast_model,
            forecast_trajectory_timestamps=t_steps,
            forecast_trajectory_qber=q_traj,
            urgency_level=urgency,
            recommendation_message=msg,
        )


# ==============================================================================
# CHALLENGE EXTENSION: Dual-Horizon Fixed-Window Quantile Forecaster
# ==============================================================================

@dataclass
class QuantileForecastResult:
    """Represents the output of Layer 6 Dual-Horizon Quantile Forecaster."""
    current_qber: float
    # 60-second horizon quantiles
    qber_60s_p10: float
    qber_60s_p50: float
    qber_60s_p90: float
    # 300-second (5-min) horizon quantiles
    qber_300s_p10: float
    qber_300s_p50: float
    qber_300s_p90: float
    urgency_level: str                    # "CRITICAL", "WARNING", "STABLE"
    recommendation_message: str
    is_crossing_predicted: bool
    predicted_crossing_horizon_s: Optional[int]


class DualHorizonQuantileForecaster:
    """
    Fixed-horizon Quantile Regression Forecaster for QBER predictive maintenance.
    Predicts calibrated distribution quantiles (alpha in {0.10, 0.50, 0.90}) for:
      - Horizon 1: t + 60 seconds (1 minute lookahead)
      - Horizon 2: t + 300 seconds (5 minute lookahead)

    Features:
      - Strictly monotonic quantiles: enforces q_0.10 <= q_0.50 <= q_0.90.
      - Fixed-window lookahead avoiding horizon truncation artifacts.
      - Evaluated against Persistence and Linear Trend baselines using Pinball loss.
    """
    def __init__(
        self,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
        horizons_s: Tuple[int, ...] = (60, 300),
        warning_threshold: float = 0.08,
        abort_threshold: float = 0.11,
        random_state: int = 42,
    ):
        self.quantiles = quantiles
        self.horizons_s = horizons_s
        self.warning_threshold = warning_threshold
        self.abort_threshold = abort_threshold
        self.random_state = random_state

        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_targets: Dict[str, np.ndarray]) -> None:
        """
        Fits LightGBM quantile regressors for each (horizon, quantile) pair.
        y_targets keys must include "target_qber_60s" and "target_qber_300s".
        """
        for h in self.horizons_s:
            col = f"target_qber_{h}s"
            if col not in y_targets:
                continue
            y = y_targets[col]
            valid_mask = ~np.isnan(y)
            X_valid = X_train[valid_mask]
            y_valid = y[valid_mask]

            if len(X_valid) < 100:
                continue

            for q in self.quantiles:
                key = f"{h}s_q{int(q*100)}"
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=5,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=self.random_state,
                    verbosity=-1,
                    n_jobs=1,
                )
                model.fit(X_valid, y_valid)
                self.models[key] = model

        self.is_fitted = True

    def predict_sample(self, x: np.ndarray, current_qber: float) -> QuantileForecastResult:
        """
        Infers monotonic quantile forecasts for 60s and 300s horizons from feature vector x.
        """
        x_2d = x.reshape(1, -1)

        def _predict_horizon(h: int) -> Tuple[float, float, float]:
            vals = []
            for q in self.quantiles:
                key = f"{h}s_q{int(q*100)}"
                if key in self.models:
                    val = float(self.models[key].predict(x_2d)[0])
                else:
                    # Fallback to current QBER if model not available
                    val = current_qber
                vals.append(val)
            # Enforce strict quantile monotonicity: q10 <= q50 <= q90
            vals.sort()
            return float(np.clip(vals[0], 0.0, 0.50)), float(np.clip(vals[1], 0.0, 0.50)), float(np.clip(vals[2], 0.0, 0.50))

        q10_60, q50_60, q90_60 = _predict_horizon(60)
        q10_300, q50_300, q90_300 = _predict_horizon(300)

        # Warning / urgency logic
        if q90_60 >= self.warning_threshold:
            urgency = "CRITICAL"
            is_crossing = True
            horizon = 60
            msg = (
                f"CRITICAL: QBER upper bound (P90: {q90_60*100:.2f}%) projected to breach "
                f"warning threshold ({self.warning_threshold*100:.1f}%) within 60 seconds."
            )
        elif q90_300 >= self.warning_threshold:
            urgency = "WARNING"
            is_crossing = True
            horizon = 300
            msg = (
                f"WARNING: QBER upper bound (P90: {q90_300*100:.2f}%) projected to breach "
                f"warning threshold ({self.warning_threshold*100:.1f}%) within 5 minutes."
            )
        else:
            urgency = "STABLE"
            is_crossing = False
            horizon = None
            msg = (
                f"STABLE: No crossing predicted within 5 minutes. "
                f"P90 @ 60s: {q90_60*100:.2f}%, P90 @ 300s: {q90_300*100:.2f}%."
            )

        return QuantileForecastResult(
            current_qber=current_qber,
            qber_60s_p10=q10_60,
            qber_60s_p50=q50_60,
            qber_60s_p90=q90_60,
            qber_300s_p10=q10_300,
            qber_300s_p50=q50_300,
            qber_300s_p90=q90_300,
            urgency_level=urgency,
            recommendation_message=msg,
            is_crossing_predicted=is_crossing,
            predicted_crossing_horizon_s=horizon,
        )

    def evaluate_against_baselines(
        self,
        X_test: np.ndarray,
        y_true_60: np.ndarray,
        y_true_300: np.ndarray,
        current_qber_test: np.ndarray,
        slope_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Benchmarks Quantile Forecaster against Persistence and Linear Trend baselines.
        Computes Pinball Loss, Interval Coverage, and Median MAE.
        """
        def _pinball(y_t: np.ndarray, y_p: np.ndarray, q: float) -> float:
            err = y_t - y_p
            return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))

        results: Dict[str, Any] = {}

        for h, y_t in [(60, y_true_60), (300, y_true_300)]:
            mask = ~np.isnan(y_t)
            if np.sum(mask) < 20:
                continue

            yt_sub = y_t[mask]
            Xt_sub = X_test[mask]
            qber_sub = current_qber_test[mask]
            slope_sub = slope_test[mask]

            # Quantile Forecaster predictions
            p10 = self.models[f"{h}s_q10"].predict(Xt_sub)
            p50 = self.models[f"{h}s_q50"].predict(Xt_sub)
            p90 = self.models[f"{h}s_q90"].predict(Xt_sub)

            # Monotonic sort
            stacked = np.sort(np.column_stack([p10, p50, p90]), axis=1)
            p10, p50, p90 = stacked[:, 0], stacked[:, 1], stacked[:, 2]

            # Persistence baseline: y_{t+h} = y_t
            pred_persist = qber_sub

            # Linear trend baseline: y_{t+h} = y_t + h * slope
            pred_linear = np.clip(qber_sub + h * slope_sub, 0.0, 0.50)

            # Metrics
            pb_forecaster = (
                _pinball(yt_sub, p10, 0.10) +
                _pinball(yt_sub, p50, 0.50) +
                _pinball(yt_sub, p90, 0.90)
            ) / 3.0
            pb_persist = _pinball(yt_sub, pred_persist, 0.50)
            pb_linear = _pinball(yt_sub, pred_linear, 0.50)

            mae_forecaster = float(np.mean(np.abs(yt_sub - p50)))
            mae_persist = float(np.mean(np.abs(yt_sub - pred_persist)))
            mae_linear = float(np.mean(np.abs(yt_sub - pred_linear)))

            coverage_80 = float(np.mean((yt_sub >= p10) & (yt_sub <= p90)))

            results[f"horizon_{h}s"] = {
                "quantile_forecaster_pinball_loss": round(pb_forecaster, 5),
                "persistence_baseline_pinball_loss": round(pb_persist, 5),
                "linear_baseline_pinball_loss": round(pb_linear, 5),
                "quantile_forecaster_mae": round(mae_forecaster, 5),
                "persistence_baseline_mae": round(mae_persist, 5),
                "linear_baseline_mae": round(mae_linear, 5),
                "coverage_80pct_interval": round(coverage_80, 4),
                "eval_sample_count": int(np.sum(mask)),
            }

        return results

    def save(self, file_path: str) -> None:
        """Serializes forecaster models."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(
            {
                "models": self.models,
                "quantiles": self.quantiles,
                "horizons_s": self.horizons_s,
                "warning_threshold": self.warning_threshold,
                "abort_threshold": self.abort_threshold,
                "random_state": self.random_state,
            },
            file_path,
        )

    def load(self, file_path: str) -> None:
        """Loads serialized forecaster models."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Forecaster artifact not found at {file_path}")
        payload = joblib.load(file_path)
        self.models = payload["models"]
        self.quantiles = payload.get("quantiles", (0.10, 0.50, 0.90))
        self.horizons_s = payload.get("horizons_s", (60, 300))
        self.warning_threshold = payload.get("warning_threshold", 0.08)
        self.abort_threshold = payload.get("abort_threshold", 0.11)
        self.is_fitted = True

