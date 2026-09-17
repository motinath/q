"""
Layer 8 — Literal Performance Optimisation & Mitigation Recommendation Engine (Rule 9)
Evaluates discrete candidate actions per diagnosed fault/attack mode, computes closed-form
physical state projections via Layer 1 optical models, and selects the argmax action under:
J(action) = w1 * (Recovery % / 100) - w2 * (ExecutionTime / T_max) - w3 * Risk

P4.1: Added actuation_mode field to RemediationRecommendation.
  'ADVISORY'   — hardware/environmental faults; operator must confirm before acting.
  'AUTONOMOUS' — performance degradation with low risk; system may act immediately.
  'EMERGENCY'  — active quantum attacks (Intercept-Resend, Detector Blinding,
                 Time-Shift, PNS); emits a JSON actuation signal to stdout so an
                 external controller can immediately terminate the key session.

Governing Standards: ETSI GS QKD 014 / GLLP / Ma-Qi-Zhao-Lo
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import json
import sys
from typing import Dict, Any, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
import numpy as np

from config.qkd_system_parameters import QKDPhysicsConfig, ALARM_SEVERITY_LEVELS
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
class CandidateAction:
    """Represents a single evaluated candidate mitigation action under Rule 9."""
    action_id: str
    action_title: str
    action_description: str
    target_subsystem: str
    execution_time_seconds: float
    operational_risk_score: float         # Risk penalty in [0.0, 1.0]
    post_physical_state: Dict[str, float] # Projected (alpha, V, T, DCR, jitter, eve)
    projected_qber: float
    projected_skr_bps: float
    projected_raw_counts_hz: float
    recovery_percentage: float            # Delta SKR / Deficit * 100%
    utility_score: float                  # Objective J(action)


@dataclass
class RemediationRecommendation:
    """
    Complete output of Layer 8 Literal Optimization.
    Contains the winning argmax action and the full comparative evaluation matrix.
    """
    action_id: str
    action_title: str
    action_description: str
    target_subsystem: str
    alarm_severity: str
    is_advisory_only: bool
    current_qber: float
    expected_post_action_qber: float
    current_skr_bps: float
    expected_post_action_skr_bps: float
    expected_raw_counts_hz: float
    qber_improvement_delta: float
    skr_gain_delta_bps: float
    mitigation_parameters: Dict[str, Any]
    # Rule 9 Optimization Matrix Fields
    candidates: List[CandidateAction] = field(default_factory=list)
    winning_candidate_id: str = ""
    optimization_rationale: str = ""
    post_physical_state: Dict[str, float] = field(default_factory=dict)
    objective_weights: Dict[str, float] = field(default_factory=dict)
    # P4.1: Actuation mode distinguishes operator-advisory from autonomous / emergency
    actuation_mode: str = "ADVISORY"   # 'ADVISORY' | 'AUTONOMOUS' | 'EMERGENCY'


class RemediationOptimizer:
    """
    Literal Argmax Mitigation Optimizer (Rule 9).
    Defines >= 2 candidate actions per fault/attack class, calculates theoretical
    post-action states using L1 optical physics, and selects the argmax utility.
    """

    def __init__(
        self,
        config: Optional[QKDPhysicsConfig] = None,
        w_recovery: float = 0.60,
        w_time: float = 0.20,
        w_risk: float = 0.20,
        t_max_seconds: float = 120.0,
    ):
        self.config = config or QKDPhysicsConfig()
        self.w_recovery = w_recovery
        self.w_time = w_time
        self.w_risk = w_risk
        self.t_max_seconds = t_max_seconds
        
        # Precompute nominal baseline SKR
        eta_nom = compute_channel_transmittance(
            self.config.nominal_fiber_attenuation_db_per_km,
            self.config.default_fiber_length_km
        )
        y0_nom = compute_dark_count_probability(
            self.config.nominal_dark_count_rate_hz,
            self.config.pulse_repetition_rate_hz
        )
        s_nom = compute_signal_yield(eta_nom, self.config.bob_detection_efficiency, self.config.mean_photon_number_signal)
        qber_nom = compute_quantum_bit_error_rate(y0_nom, s_nom, self.config.nominal_optical_error, 0.0)
        self.baseline_skr_bps, _ = compute_secret_key_rate(
            qber_nom, self.config.pulse_repetition_rate_hz, y0_nom, eta_nom,
            self.config.bob_detection_efficiency, self.config.mean_photon_number_signal,
            self.config.error_correction_efficiency, self.config.qber_abort_threshold
        )

    def _evaluate_physical_state(
        self,
        alpha_db_km: float,
        visibility: float,
        temperature_c: float,
        dcr_base_hz: float,
        timing_jitter_ps: float,
        eavesdropping_fraction: float = 0.0,
        mean_photon_number: Optional[float] = None,
        is_blinding_action: bool = False,
        is_abort_action: bool = False,
    ) -> Tuple[float, float, float]:
        """Calculates QBER, SKR, and raw counts via deterministic L1 formulas."""
        if is_abort_action:
            return 0.0, 0.0, 0.0
            
        mu = mean_photon_number if mean_photon_number is not None else self.config.mean_photon_number_signal
        
        dcr_therm = compute_thermal_dark_count_rate(
            nominal_dcr_hz=dcr_base_hz,
            current_temp_celsius=temperature_c,
            nominal_temp_celsius=self.config.nominal_detector_temp_celsius,
        )
        jitter_factor = 1.0 + max(0.0, (timing_jitter_ps - 65.0) / 200.0)
        dcr_eff = dcr_therm * jitter_factor
        
        eta_ch = compute_channel_transmittance(alpha_db_km, self.config.default_fiber_length_km)
        y0 = compute_dark_count_probability(dcr_eff, self.config.pulse_repetition_rate_hz)
        s_yield = compute_signal_yield(eta_ch, self.config.bob_detection_efficiency, mu)
        
        e_opt = (1.0 - visibility) / 2.0
        qber = compute_quantum_bit_error_rate(y0, s_yield, e_opt, eavesdropping_fraction)
        
        skr, _ = compute_secret_key_rate(
            qber=qber,
            repetition_rate_hz=self.config.pulse_repetition_rate_hz,
            y0=y0,
            eta_channel=eta_ch,
            eta_bob=self.config.bob_detection_efficiency,
            mean_photon_number=mu,
            error_correction_efficiency=self.config.error_correction_efficiency,
            critical_qber_threshold=self.config.qber_abort_threshold
        )
        raw_cnt = compute_raw_count_rate(self.config.pulse_repetition_rate_hz, y0, s_yield)
        
        if is_blinding_action:
            # Under optical VOA extinction, blinding light is filtered out, restoring normal flux
            raw_cnt = compute_raw_count_rate(self.config.pulse_repetition_rate_hz, y0, s_yield)
            
        return qber, skr, raw_cnt

    def generate_recommendation(
        self,
        fault_class: str,
        current_features: Dict[str, float],
    ) -> RemediationRecommendation:
        """
        Executes literal optimization over candidate mitigation actions (Rule 9).
        Projects post-action state via L1 physics, evaluates J(action), and returns argmax.
        """
        curr_qber = float(current_features.get("qber", 0.02))
        curr_skr = float(current_features.get("skr_bps", 10000.0))
        curr_vis = float(current_features.get("visibility", 0.985))
        curr_temp = float(current_features.get("temperature_celsius", -40.0))
        curr_dcr = float(current_features.get("dark_counts_hz", 500.0))
        curr_jitter = float(current_features.get("timing_jitter_ps", 65.0))
        curr_loss = float(current_features.get("channel_attenuation_db", 5.0))
        curr_alpha = curr_loss / self.config.default_fiber_length_km
        
        severity = ALARM_SEVERITY_LEVELS.get(fault_class, "NORMAL")
        deficit = max(1.0, self.baseline_skr_bps - curr_skr)
        
        # 1. Generate candidate definitions per diagnosed class (ADD-3: >= 3 candidates per mode: Actions A, B, C)
        candidate_defs: List[Dict[str, Any]] = []
        
        if fault_class == "Optical Misalignment":
            candidate_defs = [
                {
                    "id": "ACTION_POLARIZATION_RECALIBRATION",
                    "title": "Automated Waveplate Polarization Alignment",
                    "desc": "Execute closed-loop motorized waveplate polarization alignment sweep to maximize fringe visibility contrast.",
                    "subsystem": "Optical Polarization Controller",
                    "t_exec": 35.0,
                    "risk": 0.15,
                    "state": {"alpha": curr_alpha, "vis": 0.985, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0},
                },
                {
                    "id": "ACTION_POLARIZATION_FAST_EPC_SWEEP",
                    "title": "Fast Piezoelectric EPC Tracking",
                    "desc": "Rapid piezoelectric fiber squeezer phase compensation for quick partial contrast recovery.",
                    "subsystem": "Optical Polarization Controller",
                    "t_exec": 8.0,
                    "risk": 0.25,
                    "state": {"alpha": curr_alpha, "vis": 0.950, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0},
                },
                {
                    "id": "ACTION_OPTICAL_SESSION_RESTART",
                    "title": "Optical Link Session Restart & Re-keying",
                    "desc": "Warm-restart polarization tracking hardware and re-negotiate cryptographic session basis.",
                    "subsystem": "Optical Link Session Controller",
                    "t_exec": 5.0,
                    "risk": 0.40,
                    "state": {"alpha": curr_alpha, "vis": 0.920, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0},
                },
            ]
            
        elif fault_class in ["Channel Attenuation Event", "Channel Attenuation"]:
            candidate_defs = [
                {
                    "id": "ACTION_SWITCH_ALTERNATE_LINK",
                    "title": "Reroute Quantum Transport to Alternate Span (SPAN-B)",
                    "desc": "Switch quantum and classical optical paths to redundant low-loss fiber link (SPAN-B, alpha=0.20 dB/km).",
                    "subsystem": "Optical Transport Infrastructure",
                    "t_exec": 45.0,
                    "risk": 0.15,
                    "state": {"alpha": self.config.nominal_fiber_attenuation_db_per_km, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0},
                },
                {
                    "id": "ACTION_INTENSITY_BOOST_COMPENSATION",
                    "title": "Alice Transmit Intensity Power Boost",
                    "desc": "Dynamically increase Alice mean photon intensity to overcome moderate channel loss.",
                    "subsystem": "Alice Transmitter Optical Source",
                    "t_exec": 12.0,
                    "risk": 0.40,
                    "state": {"alpha": max(0.20, curr_alpha - 0.20), "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0, "mu": 0.60},
                },
                {
                    "id": "ACTION_RAMAN_FILTER_AND_AMP_RETUNE",
                    "title": "Optical Link Amplification Retuning & Raman Filter Insertion",
                    "desc": "Adjust Raman scattering rejection filter alignment and retune classic co-propagating channel powers.",
                    "subsystem": "Optical Transport Infrastructure",
                    "t_exec": 20.0,
                    "risk": 0.30,
                    "state": {"alpha": max(0.20, curr_alpha - 0.10), "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0},
                },
            ]
            
        elif fault_class in ["Detector APD Degradation", "APD Aging"]:
            candidate_defs = [
                {
                    "id": "ACTION_DETECTOR_BIAS_AND_COOLING_RETUNE",
                    "title": "APD Overbias Voltage & Discriminator Threshold Re-trim",
                    "desc": "Adjust APD overbias voltage by -0.5V and raise discriminator comparator threshold to suppress trap dark counts.",
                    "subsystem": "Bob Receiver SPAD Front-End",
                    "t_exec": 10.0,
                    "risk": 0.15,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr * 0.40, "jitter": curr_jitter, "eve": 0.0},
                },
                {
                    "id": "ACTION_SWITCH_REDUNDANT_SPAD",
                    "title": "Switch to Redundant Cold-Standby SPAD Channel",
                    "desc": "Electrically reroute single-photon clicks to auxiliary low-noise InGaAs SPAD detector channel.",
                    "subsystem": "Bob Receiver SPAD Array",
                    "t_exec": 30.0,
                    "risk": 0.20,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": self.config.nominal_dark_count_rate_hz, "jitter": curr_jitter, "eve": 0.0},
                },
                {
                    "id": "ACTION_APD_THERMAL_ANNEALING",
                    "title": "APD Active Thermal Annealing Cycle",
                    "desc": "Cycle thermoelectric cooler to elevate diode junction temperature for trap recombination before re-chilling.",
                    "subsystem": "Bob Receiver SPAD Front-End",
                    "t_exec": 60.0,
                    "risk": 0.35,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr * 0.50, "jitter": curr_jitter, "eve": 0.0},
                },
            ]
            
        elif fault_class == "Thermal Drift":
            candidate_defs = [
                {
                    "id": "ACTION_TEC_PHASE_COMPENSATION",
                    "title": "Thermoelectric Cooler (TEC) Thermal Restoral",
                    "desc": "Retune closed-loop TEC PID current to restore nominal -40.0 deg C detector setpoint.",
                    "subsystem": "Thermal Stabilization Subsystem",
                    "t_exec": 40.0,
                    "risk": 0.10,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": self.config.nominal_detector_temp_celsius, "dcr": self.config.nominal_dark_count_rate_hz, "jitter": curr_jitter, "eve": 0.0},
                },
                {
                    "id": "ACTION_TEC_RAPID_FEEDFORWARD_CHILL",
                    "title": "Fast Feedforward TEC Coolant Boost",
                    "desc": "Drive aggressive feedforward Peltier current step for rapid temperature quenching to -35.0 deg C.",
                    "subsystem": "Thermal Stabilization Subsystem",
                    "t_exec": 15.0,
                    "risk": 0.25,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": -35.0, "dcr": 700.0, "jitter": curr_jitter, "eve": 0.0},
                },
                {
                    "id": "ACTION_CRYOSTAT_HEATSINK_FAN_STEP",
                    "title": "Cryostat Heat-Sink Fan Boost & Environmental Baffle",
                    "desc": "Ramp secondary enclosure ventilation fan to maximum duty cycle and close thermal heat-shield baffle.",
                    "subsystem": "Thermal Stabilization Subsystem",
                    "t_exec": 8.0,
                    "risk": 0.20,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": -33.0, "dcr": 900.0, "jitter": curr_jitter, "eve": 0.0},
                },
            ]
            
        elif fault_class == "Timing Jitter":
            candidate_defs = [
                {
                    "id": "ACTION_FPGA_CLOCK_REALIGNMENT",
                    "title": "FPGA Master Clock Phase Re-Sync & Window Calibration",
                    "desc": "Execute full clock phase recovery algorithm and re-align receiver gating pulse with Alice master sync.",
                    "subsystem": "Timing Synchronization Subsystem",
                    "t_exec": 35.0,
                    "risk": 0.15,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": self.config.nominal_timing_jitter_ps, "eve": 0.0},
                },
                {
                    "id": "ACTION_OPTICAL_DELAY_LINE_TRIM",
                    "title": "Optical Delay Line Fast Phase Trim",
                    "desc": "Trim motorized optical delay line by small step to center single-photon arrival within gating window.",
                    "subsystem": "Receiver Optical Front-End",
                    "t_exec": 8.0,
                    "risk": 0.10,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": 85.0, "eve": 0.0},
                },
                {
                    "id": "ACTION_DISCRIMINATOR_SLEW_RESET",
                    "title": "Receiver Fast Discriminator Edge-Trigger Reset",
                    "desc": "Reset leading-edge timing discriminator thresholds on Bob SPAD output pulses.",
                    "subsystem": "Timing Synchronization Subsystem",
                    "t_exec": 5.0,
                    "risk": 0.25,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": 95.0, "eve": 0.0},
                },
            ]
            
        elif fault_class == "Intercept-Resend":
            candidate_defs = [
                {
                    "id": "ACTION_EMERGENCY_KEY_SESSION_ABORT",
                    "title": "Emergency Cryptographic Session Abort & Channel Quarantine",
                    "desc": "Immediately terminate current key distillation session, dump contaminated sifted keys, and alert SOC.",
                    "subsystem": "Cryptographic Key Manager & Security Subsystem",
                    "t_exec": 2.0,
                    "risk": 0.05,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0, "abort": True},
                },
                {
                    "id": "ACTION_DECOY_INTENSITY_RANDOMIZATION",
                    "title": "Dynamic Decoy Pulse Randomization & Basis Re-Weighting",
                    "desc": "Shift decoy intensity parameters and increase privacy amplification compression to penalize Eve.",
                    "subsystem": "Alice Transmitter & Sifting Engine",
                    "t_exec": 20.0,
                    "risk": 0.45,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.08},
                },
                {
                    "id": "ACTION_OTDR_FIBER_TAPPING_SWEEP",
                    "title": "OTDR Fiber Tap Localization & Security Quarantine",
                    "desc": "Trigger high-resolution optical time-domain reflectometer pulse to pinpoint eavesdropping backscatter.",
                    "subsystem": "Optical Transport Infrastructure",
                    "t_exec": 50.0,
                    "risk": 0.30,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0, "abort": True},
                },
            ]
            
        elif fault_class == "Detector Blinding":
            candidate_defs = [
                {
                    "id": "ACTION_VOA_PULSE_EXTINCTION",
                    "title": "High-Speed VOA Insertion & Classical Light Extinction",
                    "desc": "Insert 25 dB variable optical attenuation to extinguish blinding CW light and restore Geiger mode.",
                    "subsystem": "Receiver Optical Front-End",
                    "t_exec": 5.0,
                    "risk": 0.12,
                    "state": {"alpha": self.config.nominal_fiber_attenuation_db_per_km, "vis": 0.985, "temp": -40.0, "dcr": 500.0, "jitter": 65.0, "eve": 0.0, "blinding_fixed": True},
                },
                {
                    "id": "ACTION_HARDWARE_SHUTTER_AND_APD_QUENCH",
                    "title": "Mechanical Shutter Closure & APD Power Cycle",
                    "desc": "Close physical shutter on Bob input port and cycle APD high-voltage bias power supply.",
                    "subsystem": "Bob Optical Enclosure & APD Bias Supply",
                    "t_exec": 25.0,
                    "risk": 0.20,
                    "state": {"alpha": self.config.nominal_fiber_attenuation_db_per_km, "vis": 0.985, "temp": -40.0, "dcr": 500.0, "jitter": 65.0, "eve": 0.0, "blinding_fixed": True},
                },
                {
                    "id": "ACTION_SPECTRAL_NOTCH_FILTER_INSERTION",
                    "title": "Spectral Notch Filtering & APD Overcurrent Trip",
                    "desc": "Engage narrow dielectric bandpass filter and trigger active overcurrent protection on detector cathodes.",
                    "subsystem": "Receiver Optical Front-End",
                    "t_exec": 15.0,
                    "risk": 0.28,
                    "state": {"alpha": self.config.nominal_fiber_attenuation_db_per_km, "vis": 0.985, "temp": -40.0, "dcr": 500.0, "jitter": 65.0, "eve": 0.0, "blinding_fixed": True},
                },
            ]
            
        elif fault_class == "Photon Number Splitting":
            candidate_defs = [
                {
                    "id": "ACTION_ADAPTIVE_DECOY_REOPTIMIZATION",
                    "title": "Adaptive Decoy-State Intensity Optimization (mu -> 0.25)",
                    "desc": "Lower Alice mean photon number from 0.50 to 0.25 photons/pulse, suppressing multi-photon emission.",
                    "subsystem": "Alice Variable Optical Attenuator & Decoy Modulator",
                    "t_exec": 18.0,
                    "risk": 0.18,
                    "state": {"alpha": self.config.nominal_fiber_attenuation_db_per_km, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0, "mu": 0.25},
                },
                {
                    "id": "ACTION_SWITCH_ENTANGLEMENT_BBM92",
                    "title": "Reconfigure Link Protocol to Entangled-Photon BBM92",
                    "desc": "Switch from weak coherent pulse BB84 to true entangled photon pair source to eliminate PNS vulnerability.",
                    "subsystem": "Quantum Optical Source Subsystem",
                    "t_exec": 80.0,
                    "risk": 0.45,
                    "state": {"alpha": self.config.nominal_fiber_attenuation_db_per_km, "vis": 0.985, "temp": -40.0, "dcr": 500.0, "jitter": 65.0, "eve": 0.0},
                },
                {
                    "id": "ACTION_PRIVACY_AMPLIFICATION_COMPRESSION_INFLATION",
                    "title": "Dynamic Privacy Amplification Compression Inflation",
                    "desc": "Elevate Toeplitz matrix compression ratio by 40% to account for worst-case multi-photon information leakage.",
                    "subsystem": "Alice & Bob Key Post-Processing",
                    "t_exec": 10.0,
                    "risk": 0.35,
                    "state": {"alpha": self.config.nominal_fiber_attenuation_db_per_km, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0, "mu": 0.35},
                },
            ]
            
        elif fault_class == "Time-Shift Attack":
            candidate_defs = [
                {
                    "id": "ACTION_GATING_TEMPORAL_DITHERING",
                    "title": "Gate Window Temporal Dithering & Random Delay Injection",
                    "desc": "Inject pseudo-random sub-nanosecond jitter into detector gating windows to invalidate Eve timing delay.",
                    "subsystem": "Bob Receiver Gating Synchronization",
                    "t_exec": 12.0,
                    "risk": 0.15,
                    "state": {"alpha": self.config.nominal_fiber_attenuation_db_per_km, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": 70.0, "eve": 0.0},
                },
                {
                    "id": "ACTION_FULL_DETECTOR_EFFICIENCY_CALIBRATION",
                    "title": "Multi-Detector Efficiency Balancing & Gate Re-Alignment",
                    "desc": "Perform automated 4-channel detector efficiency curve equalization and gate profile re-centering.",
                    "subsystem": "Bob Receiver Electronics",
                    "t_exec": 55.0,
                    "risk": 0.28,
                    "state": {"alpha": self.config.nominal_fiber_attenuation_db_per_km, "vis": 0.985, "temp": -40.0, "dcr": 500.0, "jitter": 65.0, "eve": 0.0},
                },
                {
                    "id": "ACTION_DUAL_WINDOW_COINCIDENCE_FILTERING",
                    "title": "Dual-Window Coincidence Filtering & Phase Randomization",
                    "desc": "Shorten detection coincidence gate to 800 ps and randomize receiver detection clock edges.",
                    "subsystem": "Bob Receiver Gating Synchronization",
                    "t_exec": 20.0,
                    "risk": 0.25,
                    "state": {"alpha": self.config.nominal_fiber_attenuation_db_per_km, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": 78.0, "eve": 0.0},
                },
            ]
            
        else: # Normal baseline
            candidate_defs = [
                {
                    "id": "ACTION_NOMINAL_PASSIVE_MONITORING",
                    "title": "Autonomous Passive Telemetry Monitoring",
                    "desc": "Maintain nominal link operation, stream continuous telemetry, and preserve key distillation.",
                    "subsystem": "Vector Q Supervisor",
                    "t_exec": 1.0,
                    "risk": 0.01,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0},
                },
                {
                    "id": "ACTION_PREVENTATIVE_CALIBRATION_CHECK",
                    "title": "Non-Disruptive Optical Baseline Verification Sweep",
                    "desc": "Verify dark noise baseline and fringe visibility during sifting phase pauses without key loss.",
                    "subsystem": "Diagnostics & Calibration Engine",
                    "t_exec": 10.0,
                    "risk": 0.04,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0},
                },
                {
                    "id": "ACTION_PREDICTIVE_DISPERSION_PRECALIBRATION",
                    "title": "Predictive Fiber Dispersion & Polarization Pre-Calibration",
                    "desc": "Record seasonal polarization rotation baseline and update tracking feedforward table.",
                    "subsystem": "Optical Polarization Controller",
                    "t_exec": 25.0,
                    "risk": 0.08,
                    "state": {"alpha": curr_alpha, "vis": curr_vis, "temp": curr_temp, "dcr": curr_dcr, "jitter": curr_jitter, "eve": 0.0},
                },
            ]
            
        # 2. Project physical states and evaluate utility for all candidates
        evaluated_candidates: List[CandidateAction] = []
        for cdef in candidate_defs:
            st = cdef["state"]
            is_abort = st.get("abort", False)
            is_blind_fix = st.get("blinding_fixed", False)
            mu_val = st.get("mu", None)
            
            q_post, skr_post, raw_post = self._evaluate_physical_state(
                alpha_db_km=st["alpha"],
                visibility=st["vis"],
                temperature_c=st["temp"],
                dcr_base_hz=st["dcr"],
                timing_jitter_ps=st["jitter"],
                eavesdropping_fraction=st.get("eve", 0.0),
                mean_photon_number=mu_val,
                is_blinding_action=is_blind_fix,
                is_abort_action=is_abort,
            )
            
            # Compute physical recovery %
            if is_abort:
                # Emergency security abort intentionally terminates key to safeguard forward secrecy
                rec_pct = 100.0  # Full security preservation
            elif fault_class == "Normal":
                rec_pct = 100.0
            else:
                gain = max(0.0, skr_post - curr_skr)
                rec_pct = float(np.clip((gain / deficit) * 100.0, 0.0, 100.0))
                
            # Compute Rule 9 Objective Utility: J(action)
            norm_rec = rec_pct / 100.0
            norm_time = min(1.0, cdef["t_exec"] / self.t_max_seconds)
            risk = cdef["risk"]
            utility = (self.w_recovery * norm_rec) - (self.w_time * norm_time) - (self.w_risk * risk)
            
            cand_obj = CandidateAction(
                action_id=cdef["id"],
                action_title=cdef["title"],
                action_description=cdef["desc"],
                target_subsystem=cdef["subsystem"],
                execution_time_seconds=cdef["t_exec"],
                operational_risk_score=risk,
                post_physical_state=st,
                projected_qber=round(q_post, 4),
                projected_skr_bps=round(skr_post, 1),
                projected_raw_counts_hz=round(raw_post, 1),
                recovery_percentage=round(rec_pct, 1),
                utility_score=round(utility, 4),
            )
            evaluated_candidates.append(cand_obj)
            
        # 3. Select Argmax Action: a* = argmax_a J(a)
        utilities = [c.utility_score for c in evaluated_candidates]
        best_idx = int(np.argmax(utilities))
        winner = evaluated_candidates[best_idx]

        # 4. P4.1: Determine actuation mode from fault class
        # EMERGENCY — active quantum attacks requiring immediate autonomous response
        # AUTONOMOUS — performance faults with well-understood remediation, low risk
        # ADVISORY   — all other cases; operator confirmation required
        EMERGENCY_CLASSES = {"Intercept-Resend", "Detector Blinding", "Photon Number Splitting", "Time-Shift Attack"}
        AUTONOMOUS_CLASSES = {"Thermal Drift", "Timing Jitter"}
        if fault_class in EMERGENCY_CLASSES:
            actuation_mode = "EMERGENCY"
        elif fault_class in AUTONOMOUS_CLASSES and winner.operational_risk_score <= 0.25:
            actuation_mode = "AUTONOMOUS"
        else:
            actuation_mode = "ADVISORY"

        # 5. Formulate Optimization Rationale (ADD-3: Multi-Candidate Trade-off Matrix)
        rationale_lines = [
            f"Literal Argmax Optimization: Evaluated {len(evaluated_candidates)} physical candidate actions.",
            f"Objective Function: J(a) = {self.w_recovery:.2f}*(Recovery/100) - {self.w_time:.2f}*(t_exec/{self.t_max_seconds:.0f}s) - {self.w_risk:.2f}*Risk.",
            f"Selected Action '{winner.action_title}' (ID: {winner.action_id}) achieved maximum utility J = {winner.utility_score:.4f}.",
            f"Trade-off Profile: {winner.recovery_percentage:.1f}% Recovery in {winner.execution_time_seconds:.0f}s (Operational Risk: {winner.operational_risk_score:.2f}).",
            f"Actuation Mode: {actuation_mode}.",
        ]
        alternatives = [c for i, c in enumerate(evaluated_candidates) if i != best_idx]
        alt_summaries = [
            f"'{alt.action_title}' (J = {alt.utility_score:.4f}, Rec: {alt.recovery_percentage:.1f}%, Time: {alt.execution_time_seconds:.0f}s, Risk: {alt.operational_risk_score:.2f})"
            for alt in alternatives
        ]
        if alt_summaries:
            rationale_lines.append(f"Rejected Alternatives: {'; '.join(alt_summaries)}.")
        rationale = " ".join(rationale_lines)

        # P4.1: Emit structured actuation signal to stdout for EMERGENCY actions
        # A real field controller can pipe stdout to trigger hardware responses.
        if actuation_mode == "EMERGENCY":
            actuation_signal = {
                "actuation_mode": "EMERGENCY",
                "action_id": winner.action_id,
                "fault_class": fault_class,
                "alarm_severity": severity,
                "action_title": winner.action_title,
                "target_subsystem": winner.target_subsystem,
                "execution_time_seconds": winner.execution_time_seconds,
                "current_qber": round(curr_qber, 4),
                "trigger": "VECTOR_Q_AUTONOMOUS_EMERGENCY_RESPONSE",
            }
            print(
                f"[VECTOR Q EMERGENCY ACTUATION] {json.dumps(actuation_signal)}",
                file=sys.stderr,
                flush=True,
            )

        return RemediationRecommendation(
            action_id=winner.action_id,
            action_title=winner.action_title,
            action_description=winner.action_description,
            target_subsystem=winner.target_subsystem,
            alarm_severity=severity,
            is_advisory_only=(actuation_mode == "ADVISORY"),
            current_qber=round(curr_qber, 4),
            expected_post_action_qber=winner.projected_qber,
            current_skr_bps=round(curr_skr, 1),
            expected_post_action_skr_bps=winner.projected_skr_bps,
            expected_raw_counts_hz=winner.projected_raw_counts_hz,
            qber_improvement_delta=round(curr_qber - winner.projected_qber, 4),
            skr_gain_delta_bps=round(winner.projected_skr_bps - curr_skr, 1),
            mitigation_parameters={
                "winning_candidate": winner.action_id,
                "utility_score": winner.utility_score,
                "recovery_pct": winner.recovery_percentage,
                "execution_time_s": winner.execution_time_seconds,
                "risk_score": winner.operational_risk_score,
            },
            candidates=evaluated_candidates,
            winning_candidate_id=winner.action_id,
            optimization_rationale=rationale,
            post_physical_state=winner.post_physical_state,
            objective_weights={
                "w_recovery": self.w_recovery,
                "w_time": self.w_time,
                "w_risk": self.w_risk,
                "t_max_seconds": self.t_max_seconds,
            },
            actuation_mode=actuation_mode,
        )
