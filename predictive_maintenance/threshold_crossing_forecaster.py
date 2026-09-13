"""
Layer 6 — Predictive Maintenance (Projected Threshold Crossing Time - PTCT)
Calculates exact time-to-abort forecasts with confidence intervals and physical bounds.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

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
    forecast_trajectory_timestamps: List[float]
    forecast_trajectory_qber: List[float]
    urgency_level: str                    # 'CRITICAL', 'WARNING', 'ADVISORY', 'STABLE'
    recommendation_message: str


class ThresholdCrossingForecaster:
    """
    Projected Threshold Crossing Time (PTCT) Forecaster.
    Computes time remaining before QBER crosses the 11.0% GLLP security abort threshold.
    Strictly clips trajectories to physical limits: QBER in [0.0, 0.50].
    """

    def __init__(self, config: Optional[QKDPhysicsConfig] = None):
        self.config = config or QKDPhysicsConfig()
        self.qber_limit = self.config.qber_abort_threshold
        self.qber_warning = self.config.qber_warning_threshold

    def compute_ptct(
        self,
        current_qber: float,
        dqber_dt: float,
        slope_standard_error: float = 0.0002,
        forecast_horizon_seconds: float = 60.0,
        forecast_step_seconds: float = 5.0,
    ) -> PTCTForecastResult:
        """
        Calculates PTCT and project future QBER trajectory.
        
        Formula:
            t_cross = (QBER_limit - QBER_current) / (dQBER / dt)
            
        Confidence interval:
            slope_min = max(1e-7, dqber_dt - 1.96 * slope_se)
            slope_max = dqber_dt + 1.96 * slope_se
            t_cross_lower = (QBER_limit - QBER_current) / slope_max
            t_cross_upper = (QBER_limit - QBER_current) / slope_min
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
                forecast_trajectory_timestamps=[0.0],
                forecast_trajectory_qber=[current_qber],
                urgency_level="CRITICAL",
                recommendation_message="CRITICAL: QBER has breached the 11.0% security abort limit. Key distillation terminated.",
            )

        # Stable or improving link (dQBER/dt <= 0)
        if dqber_dt <= 1e-6:
            # Trajectory is flat / improving
            t_steps = list(np.arange(0, forecast_horizon_seconds + 1.0, forecast_step_seconds))
            q_traj = [current_qber for _ in t_steps]
            
            urgency = "WARNING" if current_qber >= self.qber_warning else "STABLE"
            msg = (
                f"Link operating nominally (QBER = {current_qber*100:.2f}%, dQBER/dt = {dqber_dt*100:.4f}%/s). "
                "No threshold crossing projected."
            )
            
            return PTCTForecastResult(
                t_cross_seconds=None,
                t_cross_lower_bound_seconds=None,
                t_cross_upper_bound_seconds=None,
                current_qber=current_qber,
                qber_abort_limit=self.qber_limit,
                dqber_dt=dqber_dt,
                forecast_trajectory_timestamps=t_steps,
                forecast_trajectory_qber=q_traj,
                urgency_level=urgency,
                recommendation_message=msg,
            )

        # Deteriorating link: positive slope
        delta_qber = self.qber_limit - current_qber
        t_cross = delta_qber / dqber_dt
        
        # Uncertainty bounds
        slope_upper = dqber_dt + 1.96 * max(1e-6, slope_standard_error)
        slope_lower = max(1e-6, dqber_dt - 1.96 * max(1e-6, slope_standard_error))
        
        t_cross_lower = max(0.0, delta_qber / slope_upper)
        t_cross_upper = max(0.0, delta_qber / slope_lower)
        
        # Generate projected trajectory (clipped to physical upper bound 50%)
        t_steps = list(np.arange(0, max(forecast_horizon_seconds, t_cross * 1.2), forecast_step_seconds))
        q_traj = [float(np.clip(current_qber + dqber_dt * t, 0.0, 0.50)) for t in t_steps]
        
        # Determine urgency
        if t_cross <= 15.0:
            urgency = "CRITICAL"
        elif t_cross <= 45.0:
            urgency = "WARNING"
        else:
            urgency = "ADVISORY"
            
        msg = (
            f"URGENCY {urgency}: Projected QBER threshold breach in {t_cross:.1f}s "
            f"(95% CI: [{t_cross_lower:.1f}s, {t_cross_upper:.1f}s]). "
            f"Rate of degradation: {dqber_dt*100:.3f}%/s."
        )
        
        return PTCTForecastResult(
            t_cross_seconds=round(t_cross, 1),
            t_cross_lower_bound_seconds=round(t_cross_lower, 1),
            t_cross_upper_bound_seconds=round(t_cross_upper, 1),
            current_qber=current_qber,
            qber_abort_limit=self.qber_limit,
            dqber_dt=dqber_dt,
            forecast_trajectory_timestamps=t_steps,
            forecast_trajectory_qber=q_traj,
            urgency_level=urgency,
            recommendation_message=msg,
        )
