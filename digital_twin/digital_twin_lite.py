"""
Layer 10 — Digital Twin Lite: Deterministic Single-Link Physics Forward Simulation & Parameter Sweep
Projects forward physical trajectories, conducts multi-dimensional what-if parameter sweeps,
and evaluates critical operational boundaries (e.g., maximum secure distance, thermal abort setpoint).
Governing Standards: ETSI GS QKD 014 / GLLP / ITU-T G.652
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from config.qkd_system_parameters import QKDPhysicsConfig
from physics_engine.optical_channel_models import (
    compute_channel_transmittance,
    compute_dark_count_probability,
    compute_signal_yield,
    compute_quantum_bit_error_rate,
    compute_secret_key_rate,
    compute_thermal_dark_count_rate,
    compute_raw_count_rate,
)


@dataclass
class ForwardProjectionPoint:
    """Represents a single time step in a forward simulated physical trajectory."""
    time_seconds: float
    qber: float
    skr_bps: float
    raw_counts_hz: float
    dark_counts_hz: float
    visibility: float
    temperature_celsius: float
    fiber_loss_db_per_km: float
    channel_attenuation_db: float


@dataclass
class ForwardProjectionResult:
    """Results of a forward deterministic trajectory simulation."""
    horizon_seconds: float
    step_seconds: float
    trajectory: List[ForwardProjectionPoint]
    time_to_abort_seconds: Optional[float]
    final_qber: float
    final_skr_bps: float
    is_secure_throughout: bool


@dataclass
class WhatIfSweepResult:
    """Results of a 1D or multi-dimensional parameter sensitivity sweep."""
    parameter_name: str
    parameter_values: List[float]
    qber_values: List[float]
    skr_values: List[float]
    critical_threshold_value: Optional[float]
    optimal_operating_point: float


class DigitalTwinLite:
    """
    Deterministic Digital Twin for a Single Metropolitan QKD Link.
    Provides forward physical projection and analytical what-if exploration.
    """

    def __init__(self, config: Optional[QKDPhysicsConfig] = None):
        self.config = config or QKDPhysicsConfig()

    def simulate_forward_trajectory(
        self,
        initial_state: Dict[str, float],
        drift_rates: Dict[str, float],
        horizon_seconds: float = 120.0,
        dt_seconds: float = 2.0,
    ) -> ForwardProjectionResult:
        """
        Simulates deterministic forward evolution of link physical state under specified drift rates.
        
        Args:
            initial_state: Dict with keys 'alpha_db_km', 'visibility', 'temperature_c', 'dcr_base', 'jitter_ps'
            drift_rates: Dict with drift per second: 'd_alpha_dt', 'd_vis_dt', 'd_temp_dt', 'd_dcr_dt'
            horizon_seconds: Total simulation time span
            dt_seconds: Time discretization step
        """
        t_values = list(np.arange(0.0, horizon_seconds + dt_seconds / 2.0, dt_seconds))
        trajectory: List[ForwardProjectionPoint] = []
        
        alpha_0 = initial_state.get("alpha_db_km", self.config.nominal_fiber_attenuation_db_per_km)
        vis_0 = initial_state.get("visibility", self.config.nominal_visibility)
        temp_0 = initial_state.get("temperature_c", self.config.nominal_detector_temp_celsius)
        dcr_0 = initial_state.get("dcr_base", self.config.nominal_dark_count_rate_hz)
        jitter_0 = initial_state.get("jitter_ps", self.config.nominal_timing_jitter_ps)
        length_km = initial_state.get("fiber_length_km", self.config.default_fiber_length_km)
        
        d_alpha = drift_rates.get("d_alpha_dt", 0.0)
        d_vis = drift_rates.get("d_vis_dt", 0.0)
        d_temp = drift_rates.get("d_temp_dt", 0.0)
        d_dcr = drift_rates.get("d_dcr_dt", 0.0)
        
        abort_time: Optional[float] = None
        
        for t in t_values:
            alpha = max(0.15, alpha_0 + d_alpha * t)
            vis = float(np.clip(vis_0 + d_vis * t, 0.50, 1.0))
            temp = temp_0 + d_temp * t
            dcr_base = max(10.0, dcr_0 + d_dcr * t)
            
            dcr_therm = compute_thermal_dark_count_rate(
                nominal_dcr_hz=dcr_base,
                current_temp_celsius=temp,
                nominal_temp_celsius=self.config.nominal_detector_temp_celsius,
            )
            jitter_factor = 1.0 + max(0.0, (jitter_0 - 65.0) / 200.0)
            dcr_eff = dcr_therm * jitter_factor
            
            eta_ch = compute_channel_transmittance(alpha, length_km)
            y0 = compute_dark_count_probability(dcr_eff, self.config.pulse_repetition_rate_hz)
            s_yield = compute_signal_yield(eta_ch, self.config.bob_detection_efficiency, self.config.mean_photon_number_signal)
            e_opt = (1.0 - vis) / 2.0
            
            qber = compute_quantum_bit_error_rate(y0, s_yield, e_opt, 0.0)
            skr, _ = compute_secret_key_rate(
                qber=qber,
                repetition_rate_hz=self.config.pulse_repetition_rate_hz,
                y0=y0,
                eta_channel=eta_ch,
                eta_bob=self.config.bob_detection_efficiency,
                mean_photon_number=self.config.mean_photon_number_signal,
                error_correction_efficiency=self.config.error_correction_efficiency,
                critical_qber_threshold=self.config.qber_abort_threshold,
            )
            raw_cnt = compute_raw_count_rate(self.config.pulse_repetition_rate_hz, y0, s_yield)
            
            point = ForwardProjectionPoint(
                time_seconds=round(t, 2),
                qber=round(qber, 4),
                skr_bps=round(skr, 1),
                raw_counts_hz=round(raw_cnt, 1),
                dark_counts_hz=round(dcr_eff, 1),
                visibility=round(vis, 4),
                temperature_celsius=round(temp, 2),
                fiber_loss_db_per_km=round(alpha, 4),
                channel_attenuation_db=round(alpha * length_km, 2),
            )
            trajectory.append(point)
            
            if qber >= self.config.qber_abort_threshold and abort_time is None:
                abort_time = round(t, 2)
                
        final_pt = trajectory[-1]
        is_secure = all(p.qber < self.config.qber_abort_threshold for p in trajectory)
        
        return ForwardProjectionResult(
            horizon_seconds=horizon_seconds,
            step_seconds=dt_seconds,
            trajectory=trajectory,
            time_to_abort_seconds=abort_time,
            final_qber=final_pt.qber,
            final_skr_bps=final_pt.skr_bps,
            is_secure_throughout=is_secure,
        )

    def sweep_fiber_length(
        self,
        min_length_km: float = 5.0,
        max_length_km: float = 80.0,
        steps: int = 40,
        fixed_alpha_db_km: float = 0.20,
    ) -> WhatIfSweepResult:
        """Sweeps fiber link length to identify maximum secure transmission distance."""
        lengths = list(np.linspace(min_length_km, max_length_km, steps))
        qbers: List[float] = []
        skrs: List[float] = []
        cutoff_length: Optional[float] = None
        
        for l in lengths:
            eta_ch = compute_channel_transmittance(fixed_alpha_db_km, l)
            y0 = compute_dark_count_probability(self.config.nominal_dark_count_rate_hz, self.config.pulse_repetition_rate_hz)
            s_yield = compute_signal_yield(eta_ch, self.config.bob_detection_efficiency, self.config.mean_photon_number_signal)
            q = compute_quantum_bit_error_rate(y0, s_yield, self.config.nominal_optical_error, 0.0)
            skr, _ = compute_secret_key_rate(
                qber=q,
                repetition_rate_hz=self.config.pulse_repetition_rate_hz,
                y0=y0,
                eta_channel=eta_ch,
                eta_bob=self.config.bob_detection_efficiency,
                mean_photon_number=self.config.mean_photon_number_signal,
                error_correction_efficiency=self.config.error_correction_efficiency,
                critical_qber_threshold=self.config.qber_abort_threshold,
            )
            qbers.append(round(q, 4))
            skrs.append(round(skr, 1))
            
            if q >= self.config.qber_abort_threshold and cutoff_length is None:
                cutoff_length = round(l, 1)
                
        return WhatIfSweepResult(
            parameter_name="Fiber Length (km)",
            parameter_values=[round(x, 1) for x in lengths],
            qber_values=qbers,
            skr_values=skrs,
            critical_threshold_value=cutoff_length,
            optimal_operating_point=round(lengths[0], 1),
        )

    def sweep_detector_temperature(
        self,
        min_temp_c: float = -45.0,
        max_temp_c: float = 20.0,
        steps: int = 40,
    ) -> WhatIfSweepResult:
        """Sweeps APD detector temperature to identify Arrhenius thermal abort setpoint."""
        temps = list(np.linspace(min_temp_c, max_temp_c, steps))
        qbers: List[float] = []
        skrs: List[float] = []
        crit_temp: Optional[float] = None
        
        eta_ch = compute_channel_transmittance(self.config.nominal_fiber_attenuation_db_per_km, self.config.default_fiber_length_km)
        s_yield = compute_signal_yield(eta_ch, self.config.bob_detection_efficiency, self.config.mean_photon_number_signal)
        
        for t in temps:
            dcr_therm = compute_thermal_dark_count_rate(
                nominal_dcr_hz=self.config.nominal_dark_count_rate_hz,
                current_temp_celsius=t,
                nominal_temp_celsius=self.config.nominal_detector_temp_celsius,
            )
            y0 = compute_dark_count_probability(dcr_therm, self.config.pulse_repetition_rate_hz)
            q = compute_quantum_bit_error_rate(y0, s_yield, self.config.nominal_optical_error, 0.0)
            skr, _ = compute_secret_key_rate(
                qber=q,
                repetition_rate_hz=self.config.pulse_repetition_rate_hz,
                y0=y0,
                eta_channel=eta_ch,
                eta_bob=self.config.bob_detection_efficiency,
                mean_photon_number=self.config.mean_photon_number_signal,
                error_correction_efficiency=self.config.error_correction_efficiency,
                critical_qber_threshold=self.config.qber_abort_threshold,
            )
            qbers.append(round(q, 4))
            skrs.append(round(skr, 1))
            
            if q >= self.config.qber_abort_threshold and crit_temp is None:
                crit_temp = round(t, 1)
                
        return WhatIfSweepResult(
            parameter_name="Detector Temperature (C)",
            parameter_values=[round(x, 1) for x in temps],
            qber_values=qbers,
            skr_values=skrs,
            critical_threshold_value=crit_temp,
            optimal_operating_point=-40.0,
        )

    def sanity_check_against_hand_calculations(self) -> Dict[str, Any]:
        """
        Performs strict analytical sanity check against known manual hand-calculations:
        1. L = 25 km, alpha = 0.20 dB/km -> Loss = 5.0 dB -> Transmittance = 10^(-0.5) = 0.3162277
        2. Signal Yield S = eta_ch * eta_bob * mu = 0.3162277 * 0.15 * 0.50 = 0.023717
        3. Raw count rate R_raw = 1e8 * (5e-6 + 0.023717) = 2,372,208 cps
        """
        eta_ch = compute_channel_transmittance(0.20, 25.0)
        expected_eta = 10.0 ** (-5.0 / 10.0)
        assert math.isclose(eta_ch, expected_eta, rel_tol=1e-5), "Transmittance mismatch"
        
        y0 = compute_dark_count_probability(500.0, 1.0e8)
        assert math.isclose(y0, 5.0e-6, rel_tol=1e-6), "Y0 mismatch"
        
        s = compute_signal_yield(eta_ch, 0.15, 0.50)
        expected_s = expected_eta * 0.15 * 0.50
        assert math.isclose(s, expected_s, rel_tol=1e-5), "Yield mismatch"
        
        r_raw = compute_raw_count_rate(1.0e8, y0, s)
        expected_raw = 1.0e8 * (5.0e-6 + expected_s)
        assert math.isclose(r_raw, expected_raw, rel_tol=1e-4), "Raw count mismatch"
        
        return {
            "status": "PASSED",
            "transmittance_test": {"computed": eta_ch, "expected": expected_eta, "rel_diff": abs(eta_ch - expected_eta)},
            "yield_test": {"computed": s, "expected": expected_s, "rel_diff": abs(s - expected_s)},
            "raw_count_test": {"computed": r_raw, "expected": expected_raw, "rel_diff": abs(r_raw - expected_raw)},
        }

    def perturb_and_forward_simulate(
        self,
        current_telemetry: Dict[str, float],
        delta_params: Dict[str, float],
        horizon_seconds: float = 60.0,
        dt_seconds: float = 2.0,
    ) -> Dict[str, Any]:
        """
        ADD-5: Instant Parameter Perturbation Simulator.
        Applies delta perturbations (e.g. delta_T = +5 C, delta_loss = +2.0 dB, delta_vis = -0.05)
        to current operational telemetry, computes immediate physical state and forward trajectory,
        and forecasts Future QBER, Future SKR, and Future PTCT (seconds to 11.0% abort threshold).
        """
        # 1. Base telemetry values
        temp_0 = float(current_telemetry.get("temperature_celsius", self.config.nominal_detector_temp_celsius))
        vis_0 = float(current_telemetry.get("visibility", self.config.nominal_visibility))
        loss_0 = float(current_telemetry.get("channel_attenuation_db", 5.0))
        fiber_len = float(current_telemetry.get("fiber_length_km", self.config.default_fiber_length_km))
        alpha_0 = loss_0 / max(1e-6, fiber_len)
        dcr_0 = float(current_telemetry.get("dark_counts_hz", self.config.nominal_dark_count_rate_hz))
        jitter_0 = float(current_telemetry.get("timing_jitter_ps", self.config.nominal_timing_jitter_ps))
        curr_qber = float(current_telemetry.get("qber", 0.02))
        curr_skr = float(current_telemetry.get("skr_bps", 10000.0))
        
        # 2. Apply delta perturbations
        temp_p = temp_0 + float(delta_params.get("delta_temperature_c", 0.0))
        vis_p = float(np.clip(vis_0 + float(delta_params.get("delta_visibility", 0.0)), 0.50, 1.0))
        loss_p = max(0.5, loss_0 + float(delta_params.get("delta_loss_db", 0.0)))
        alpha_p = loss_p / max(1e-6, fiber_len)
        dcr_p = max(10.0, dcr_0 + float(delta_params.get("delta_dcr_hz", 0.0)))
        jitter_p = max(30.0, jitter_0 + float(delta_params.get("delta_jitter_ps", 0.0)))
        
        # 3. Immediate impact evaluation (t = 0)
        dcr_therm_0 = compute_thermal_dark_count_rate(
            nominal_dcr_hz=dcr_p,
            current_temp_celsius=temp_p,
            nominal_temp_celsius=self.config.nominal_detector_temp_celsius,
        )
        jitter_factor_0 = 1.0 + max(0.0, (jitter_p - 65.0) / 200.0)
        dcr_eff_0 = dcr_therm_0 * jitter_factor_0
        
        eta_ch_0 = compute_channel_transmittance(alpha_p, fiber_len)
        y0_0 = compute_dark_count_probability(dcr_eff_0, self.config.pulse_repetition_rate_hz)
        s_yield_0 = compute_signal_yield(eta_ch_0, self.config.bob_detection_efficiency, self.config.mean_photon_number_signal)
        e_opt_0 = (1.0 - vis_p) / 2.0
        
        imm_qber = compute_quantum_bit_error_rate(y0_0, s_yield_0, e_opt_0, 0.0)
        imm_skr, _ = compute_secret_key_rate(
            qber=imm_qber,
            repetition_rate_hz=self.config.pulse_repetition_rate_hz,
            y0=y0_0,
            eta_channel=eta_ch_0,
            eta_bob=self.config.bob_detection_efficiency,
            mean_photon_number=self.config.mean_photon_number_signal,
            error_correction_efficiency=self.config.error_correction_efficiency,
            critical_qber_threshold=self.config.qber_abort_threshold,
        )
        imm_raw = compute_raw_count_rate(self.config.pulse_repetition_rate_hz, y0_0, s_yield_0)
        
        # 4. Forward trajectory projection under underlying drift
        d_temp = float(current_telemetry.get("temperature_slope_25", 0.0))
        d_alpha = float(current_telemetry.get("loss_slope_25", 0.0)) / max(1e-6, fiber_len)
        d_vis = float(current_telemetry.get("visibility_slope_25", 0.0))
        
        t_values = list(np.arange(0.0, horizon_seconds + dt_seconds / 2.0, dt_seconds))
        trajectory: List[Dict[str, Any]] = []
        ptct_time: Optional[float] = None
        
        for t in t_values:
            cur_alpha = max(0.15, alpha_p + d_alpha * t)
            cur_vis = float(np.clip(vis_p + d_vis * t, 0.50, 1.0))
            cur_temp = temp_p + d_temp * t
            
            dcr_therm = compute_thermal_dark_count_rate(
                nominal_dcr_hz=dcr_p,
                current_temp_celsius=cur_temp,
                nominal_temp_celsius=self.config.nominal_detector_temp_celsius,
            )
            dcr_eff = dcr_therm * jitter_factor_0
            eta_ch = compute_channel_transmittance(cur_alpha, fiber_len)
            y0 = compute_dark_count_probability(dcr_eff, self.config.pulse_repetition_rate_hz)
            s_yield = compute_signal_yield(eta_ch, self.config.bob_detection_efficiency, self.config.mean_photon_number_signal)
            e_opt = (1.0 - cur_vis) / 2.0
            
            q_step = compute_quantum_bit_error_rate(y0, s_yield, e_opt, 0.0)
            skr_step, _ = compute_secret_key_rate(
                qber=q_step,
                repetition_rate_hz=self.config.pulse_repetition_rate_hz,
                y0=y0,
                eta_channel=eta_ch,
                eta_bob=self.config.bob_detection_efficiency,
                mean_photon_number=self.config.mean_photon_number_signal,
                error_correction_efficiency=self.config.error_correction_efficiency,
                critical_qber_threshold=self.config.qber_abort_threshold,
            )
            
            trajectory.append({
                "time_seconds": round(t, 1),
                "qber": round(q_step, 4),
                "skr_bps": round(skr_step, 1),
                "visibility": round(cur_vis, 4),
                "temperature_celsius": round(cur_temp, 2),
                "channel_attenuation_db": round(cur_alpha * fiber_len, 2),
            })
            
            if q_step >= self.config.qber_abort_threshold and ptct_time is None:
                ptct_time = round(t, 1)
                
        # Security classification
        if imm_qber >= self.config.qber_abort_threshold:
            status = "ABORT_BREACHED"
            ptct_time = 0.0
        elif ptct_time is not None:
            status = "ABORT_PROJECTED"
        elif imm_qber >= 0.08:
            status = "WARNING_ELEVATED"
        else:
            status = "SECURE"
            
        return {
            "current_qber": round(curr_qber, 4),
            "immediate_qber": round(imm_qber, 4),
            "qber_delta": round(imm_qber - curr_qber, 4),
            "current_skr_bps": round(curr_skr, 1),
            "immediate_skr_bps": round(imm_skr, 1),
            "skr_delta_bps": round(imm_skr - curr_skr, 1),
            "immediate_raw_counts_hz": round(imm_raw, 1),
            "projected_ptct_seconds": ptct_time,
            "security_status": status,
            "perturbed_state": {
                "temperature_celsius": round(temp_p, 1),
                "visibility": round(vis_p, 4),
                "channel_attenuation_db": round(loss_p, 2),
                "timing_jitter_ps": round(jitter_p, 1),
                "dark_counts_hz": round(dcr_eff_0, 1),
            },
            "trajectory": trajectory,
        }
