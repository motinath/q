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

import math
import numpy as np
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
