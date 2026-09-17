"""
Physics-Calibrated Quantum Telemetry Emulator (Layer 1, Source B)
Calibrated to BB84 / Decoy-State Optical Physics Models and Standard Attacks.
Governing Standards: ETSI GS QKD 014 / GLLP / Ma-Qi-Zhao-Lo / Makarov et al.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import time
import math
import numpy as np
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from config.qkd_system_parameters import QKDPhysicsConfig, ROOT_CAUSE_ID_TO_LABEL
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
class QuantumTelemetrySample:
    """
    Standard synchronized QKD telemetry sample record across physical layers.
    Conforms to ETSI GS QKD 014 data schema and VECTOR-Q Dataset Governance.
    """
    timestamp: float
    qber: float                     # Fraction [0.0, 0.5] (e.g. 0.02 = 2.0%)
    skr_bps: float                  # Secret Key Rate in bits per second
    raw_counts_hz: float            # Raw single-photon count rate (cps / Hz)
    dark_counts_hz: float           # Dark count rate (Hz)
    visibility: float               # Fringe / Polarization visibility [0.0, 1.0]
    temperature_celsius: float      # APD detector temperature (deg C)
    timing_jitter_ps: float         # FWHM timing jitter (ps)
    channel_attenuation_db: float   # Total optical channel loss (dB)
    fiber_loss_db_per_km: float     # Loss coefficient (dB/km)
    eavesdropping_fraction: float   # Intercept-resend fraction [0.0, 1.0]
    active_fault_label: str         # Ground truth fault class (for validation/analysis)
    fault_intensity: float          # Injected perturbation magnitude

    # Environmental Observables (Phase 1 Governance)
    humidity_relative_pct: float = 45.0      # Enclosure relative humidity (%)
    vibration_g: float = 0.02                # Mechanical vibration / acoustic shock (g)
    supply_voltage_v: float = 3.30           # Transceiver DC supply rail voltage (V)
    fiber_strain_ue: float = 15.0            # Fiber conduit mechanical micro-strain (με)

    # Maintenance & Operating History Metadata (Phase 1 Governance)
    device_operating_hours: float = 1200.0   # Cumulative active operating hours
    hours_since_calibration: float = 48.0    # Operating hours since last optical recalibration
    trap_aging_index: float = 0.05           # APD crystal trap defect accumulation [0.0, 1.0]
    maintenance_event_count: int = 1         # Count of historical service interventions

    # Hardware Control Registers & Diagnostics (Challenge Protocol)
    tec_drive_current_ma: float = 1200.0     # Thermoelectric cooler drive current (mA)
    epc_bias_voltage_v: float = 0.0          # Electronic polarization controller bias (V)
    voa_attenuation_db: float = 0.0          # Variable optical attenuator compensation (dB)
    invalid_reading_flag: int = 0            # 1 if measurement corrupted/missing, 0 if valid
    run_id: str = "run_default"              # Unique episode run identifier

    @property
    def channel_loss_db(self) -> float:
        """Standardized alias for channel_attenuation_db."""
        return self.channel_attenuation_db


class QuantumTelemetryEmulator:
    """
    Physics-Calibrated Telemetry Emulator.
    Simulates genuine time-series telemetry based on exact physical optical models
    with physical noise (Poissonian counting statistics and detector thermal noise).
    Supports the complete 9-class official fault ontology and multi-channel environmental dynamics.
    """

    def __init__(
        self,
        config: Optional[QKDPhysicsConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or QKDPhysicsConfig()
        self.rng = np.random.RandomState(random_seed)
        
        # Nominal physical state variables
        self.fiber_length_km: float = self.config.default_fiber_length_km
        self.fiber_attenuation_db_per_km: float = self.config.nominal_fiber_attenuation_db_per_km
        self.visibility: float = self.config.nominal_visibility
        self.temperature_celsius: float = self.config.nominal_detector_temp_celsius
        self.nominal_dcr_hz: float = self.config.nominal_dark_count_rate_hz
        self.timing_jitter_ps: float = self.config.nominal_timing_jitter_ps
        self.eavesdropping_fraction: float = 0.0
        self.mean_photon_number: float = self.config.mean_photon_number_signal
        self.bob_efficiency: float = self.config.bob_detection_efficiency
        self.repetition_rate_hz: float = self.config.pulse_repetition_rate_hz

        # Environmental & Maintenance State
        self.humidity_pct: float = 45.0
        self.vibration_g: float = 0.02
        self.supply_voltage_v: float = 3.30
        self.fiber_strain_ue: float = 15.0
        self.device_operating_hours: float = 1200.0
        self.hours_since_calibration: float = 48.0
        self.trap_aging_index: float = 0.05
        self.maintenance_event_count: int = 1

        # Hardware Control Registers & Diagnostics (Challenge Protocol)
        self.tec_drive_current_ma: float = 1200.0
        self.epc_bias_voltage_v: float = 0.0
        self.voa_attenuation_db: float = 0.0
        self.invalid_reading_flag: int = 0
        self.current_run_id: str = "run_default"
        
        # Fault injection state
        self.active_fault: str = "Normal"
        self.fault_intensity: float = 0.0
        self.simulation_step_index: int = 0
        self.internal_clock: float = time.time()

    def set_seed(self, seed: int) -> None:
        """Sets random seed for reproducible validation sets."""
        self.rng = np.random.RandomState(seed)

    def inject_fault(self, fault_name: Union[str, int], intensity: float = 1.0) -> None:
        """
        Perturbs physical state parameters corresponding to specific physical fault modes.
        Supports both official 9-class taxonomy, combined faults, and legacy labels.
        """
        if isinstance(fault_name, int):
            fault_name = ROOT_CAUSE_ID_TO_LABEL.get(fault_name, "Normal")
        
        if fault_name in ["Combined Fault", "Combined", "Thermal Drift + Misalignment"]:
            self.active_fault = "Combined Fault"
        elif fault_name in ["Unfamiliar Condition", "Unknown Fault"]:
            self.active_fault = "Unknown Fault"
        else:
            self.active_fault = fault_name
        self.fault_intensity = float(np.clip(intensity, 0.0, 1.0))

        # Update physical hardware control registers to reflect disturbance
        if self.active_fault in ["Temperature Drift", "Thermal Drift"]:
            self.tec_drive_current_ma = max(200.0, 1200.0 - 950.0 * self.fault_intensity)
        elif self.active_fault in ["Polarization Drift", "Optical Misalignment"]:
            self.epc_bias_voltage_v = 3.2 * self.fault_intensity
        elif self.active_fault in ["Fiber Bend", "Channel Attenuation Event"]:
            self.voa_attenuation_db = 0.55 * self.fault_intensity
        elif self.active_fault == "Combined Fault":
            self.tec_drive_current_ma = max(200.0, 1200.0 - 950.0 * self.fault_intensity)
            self.epc_bias_voltage_v = 3.2 * self.fault_intensity

    def apply_actuation(self, action_id: str, parameter_value: Optional[float] = None) -> Dict[str, Any]:
        """
        Applies a physical corrective action to hardware control registers.
        Physically alters subsequent measurements with continuous state preservation.
        """
        if "TEC" in action_id or "THERMAL" in action_id or action_id == "ACTION_TEC_PHASE_COMPENSATION":
            self.tec_drive_current_ma = 1200.0
            if self.active_fault in ["Temperature Drift", "Thermal Drift"]:
                self.active_fault = "Normal"
                self.fault_intensity = 0.0
            elif self.active_fault == "Combined Fault":
                self.active_fault = "Polarization Drift"
            return {"status": "SUCCESS", "control": "tec_drive_current_ma", "value": 1200.0}
        elif "POLARIZATION" in action_id or "EPC" in action_id or action_id in ["ACTION_POLARIZATION_FAST_EPC", "ACTION_POLARIZATION_RECALIBRATION"]:
            self.epc_bias_voltage_v = 0.0
            if self.active_fault in ["Polarization Drift", "Optical Misalignment"]:
                self.active_fault = "Normal"
                self.fault_intensity = 0.0
            elif self.active_fault == "Combined Fault":
                self.active_fault = "Temperature Drift"
            return {"status": "SUCCESS", "control": "epc_bias_voltage_v", "value": 0.0}
        elif "ATTENUATION" in action_id or "VOA" in action_id:
            self.voa_attenuation_db = 0.0
            if self.active_fault in ["Fiber Bend", "Channel Attenuation Event"]:
                self.active_fault = "Normal"
                self.fault_intensity = 0.0
            return {"status": "SUCCESS", "control": "voa_attenuation_db", "value": 0.0}
        return {"status": "NOOP", "control": None, "value": None}

    def reset_to_nominal(self) -> None:
        """Restores all parameters to nominal healthy operating values."""
        self.active_fault = "Normal"
        self.fault_intensity = 0.0
        self.tec_drive_current_ma = 1200.0
        self.epc_bias_voltage_v = 0.0
        self.voa_attenuation_db = 0.0
        self.invalid_reading_flag = 0
        self.fiber_attenuation_db_per_km = self.config.nominal_fiber_attenuation_db_per_km
        self.visibility = self.config.nominal_visibility
        self.temperature_celsius = self.config.nominal_detector_temp_celsius
        self.nominal_dcr_hz = self.config.nominal_dark_count_rate_hz
        self.timing_jitter_ps = self.config.nominal_timing_jitter_ps
        self.eavesdropping_fraction = 0.0
        self.mean_photon_number = self.config.mean_photon_number_signal
        self.humidity_pct = 45.0
        self.vibration_g = 0.02
        self.supply_voltage_v = 3.30
        self.fiber_strain_ue = 15.0

    reset = reset_to_nominal
    clear_fault = reset_to_nominal

    def step(self, dt_seconds: float = 1.0) -> QuantumTelemetrySample:
        """
        Advances the physical simulation by dt_seconds and computes the exact physical state.
        Adheres strictly to governing physical optical and environmental equations.
        """
        self.simulation_step_index += 1
        self.internal_clock += dt_seconds
        self.device_operating_hours += dt_seconds / 3600.0
        self.hours_since_calibration += dt_seconds / 3600.0
        
        # Base physical parameters
        alpha = self.config.nominal_fiber_attenuation_db_per_km
        vis = self.config.nominal_visibility
        temp = self.config.nominal_detector_temp_celsius
        dcr_base = self.config.nominal_dark_count_rate_hz
        jitter = self.config.nominal_timing_jitter_ps
        gamma_eve = 0.0
        mean_mu = self.config.mean_photon_number_signal
        humidity = 45.0
        vibration = 0.02
        voltage = 3.30
        strain = 15.0
        
        # Attack & Fault flags
        fault = self.active_fault
        intensity = self.fault_intensity

        # Official 9-Class Ontology + Legacy Mapping
        if fault in ["Temperature Drift", "Thermal Drift"]:
            # Temperature rises from -40 C up to +10 C (TEC cooler fault)
            temp = self.config.nominal_detector_temp_celsius + 50.0 * intensity
            
        elif fault in ["Fiber Bend", "Channel Attenuation Event", "Channel Attenuation"]:
            # Fiber loss increases from 0.20 dB/km up to 0.75 dB/km (fiber macrobend / splice degradation)
            alpha = self.config.nominal_fiber_attenuation_db_per_km + 0.55 * intensity
            strain = 15.0 + 80.0 * intensity
            
        elif fault in ["Polarization Drift", "Optical Misalignment"]:
            # Visibility drops from 0.985 down to ~0.70
            vis = max(0.60, self.config.nominal_visibility - 0.28 * intensity)
            
        elif fault in ["Detector Aging", "Detector APD Degradation", "APD Aging"]:
            # Dark count rate increases 4x to 10x due to trap accumulation
            dcr_base = self.config.nominal_dark_count_rate_hz * (1.0 + 8.0 * intensity)
            self.trap_aging_index = float(np.clip(0.05 + 0.90 * intensity, 0.0, 1.0))
            
        elif fault in ["Timing Misalignment", "Timing Jitter"]:
            # Jitter expands from 65 ps to 350 ps
            jitter = self.config.nominal_timing_jitter_ps + 285.0 * intensity
            
        elif fault == "Power Instability":
            # Laser diode power or wavelength dither
            mean_mu = self.config.mean_photon_number_signal * max(0.20, 1.0 - 0.45 * intensity)
            voltage = 3.30 - 0.65 * intensity
            vibration = 0.02 + 0.15 * intensity

        elif fault == "Humidity Impact":
            # Enclosure humidity surge causing optical surface condensation
            humidity = 45.0 + 48.0 * intensity
            alpha = self.config.nominal_fiber_attenuation_db_per_km + 0.25 * intensity
            vis = max(0.85, self.config.nominal_visibility - 0.06 * intensity)

        elif fault in ["Combined Fault", "Combined", "Thermal Drift + Misalignment"]:
            # Concurrent thermal degradation and polarization misalignment
            temp = self.config.nominal_detector_temp_celsius + 50.0 * intensity
            vis = max(0.60, self.config.nominal_visibility - 0.28 * intensity)
            
        elif fault == "Unknown Fault":
            # Out-of-distribution synthetic perturbation (non-standard multi-frequency modulation)
            vis = max(0.75, self.config.nominal_visibility - 0.12 * intensity)
            jitter = self.config.nominal_timing_jitter_ps + 50.0 * intensity
            alpha = self.config.nominal_fiber_attenuation_db_per_km + 0.15 * intensity
            vibration = 0.02 + 0.40 * intensity
            # Non-linear phase dither
            gamma_eve = 0.08 * intensity

        # Legacy attack mappings
        elif fault == "Intercept-Resend":
            gamma_eve = 0.15 + 0.55 * intensity
        elif fault == "Time-Shift Attack":
            jitter = self.config.nominal_timing_jitter_ps + 70.0 + 85.0 * intensity
        
        # Add realistic physical measurement noise
        temp_noisy = temp + self.rng.normal(0, 0.25)
        jitter_noisy = max(20.0, jitter + self.rng.normal(0, 2.0))
        vis_noisy = float(np.clip(vis + self.rng.normal(0, 0.002), 0.50, 0.9999))
        alpha_noisy = max(0.10, alpha + self.rng.normal(0, 0.005))
        humidity_noisy = float(np.clip(humidity + self.rng.normal(0, 0.5), 10.0, 99.0))
        vibration_noisy = max(0.001, vibration + self.rng.normal(0, 0.005))
        voltage_noisy = float(voltage + self.rng.normal(0, 0.01))
        strain_noisy = max(0.0, strain + self.rng.normal(0, 0.5))

        is_timeshift = (fault in ["Time-Shift Attack"])
        is_timing_attack = is_timeshift or (fault in ["Timing Misalignment", "Timing Jitter"])
        is_blinding = (fault in ["Detector Blinding", "Blinding Attack"])
        is_pns = (fault in ["Photon Number Splitting", "PNS Attack"])
        
        # Compute thermal dark counts from physical APD temperature model
        dcr_thermal = compute_thermal_dark_count_rate(
            nominal_dcr_hz=dcr_base,
            current_temp_celsius=temp_noisy,
            nominal_temp_celsius=self.config.nominal_detector_temp_celsius,
        )
        # Jitter contributes additional background noise capture in gating window
        jitter_noise_factor = 1.0 + max(0.0, (jitter_noisy - 65.0) / 200.0)
        dcr_effective = dcr_thermal * jitter_noise_factor
        
        # Optical error from fringe visibility e_opt = (1 - V) / 2
        e_opt = (1.0 - vis_noisy) / 2.0
        
        # Exact channel transmission & signal yield
        eta_channel = compute_channel_transmittance(
            fiber_attenuation_db_per_km=alpha_noisy,
            fiber_length_km=self.fiber_length_km
        )
        y0 = compute_dark_count_probability(
            dark_count_rate_hz=dcr_effective,
            repetition_rate_hz=self.repetition_rate_hz
        )
        signal_yield = compute_signal_yield(
            eta_channel=eta_channel,
            eta_bob=self.bob_efficiency,
            mean_photon_number=mean_mu
        )
        
        # Exact QBER
        qber = compute_quantum_bit_error_rate(
            y0=y0,
            signal_yield=signal_yield,
            optical_error_rate=e_opt,
            eavesdropping_fraction=gamma_eve
        )
        
        qber_mismatch_term = 0.0
        # If Time-Shift Attack, add characteristic basis-dependent shift error
        if is_timeshift:
            # P4.4: Physically correct time-shift model.
            # Step 1 — Basis-dependent efficiency mismatch:
            #   H-basis detection efficiency is reduced by up to 40% (intensity-scaled),
            #   +45-basis remains at nominal.  This asymmetry is the signature that
            #   distinguishes a time-shift attack from hardware timing jitter.
            eta_mismatch = 0.40 * intensity          # fraction of H-basis clicks suppressed
            # Effective detection probability for H-basis pulses
            eta_h  = self.bob_efficiency * (1.0 - eta_mismatch)
            eta_45 = self.bob_efficiency              # unaffected basis
            # Combined average yield (equal basis probability in BB84)
            avg_eta = 0.5 * eta_h + 0.5 * eta_45
            # Recompute signal yield and QBER under the mismatch model
            signal_yield_ts = compute_signal_yield(
                eta_channel=eta_channel,
                eta_bob=avg_eta,
                mean_photon_number=mean_mu
            )
            # Basis mismatch introduces extra QBER: Eve's shifted window maps H→+45 errors
            qber_mismatch_term = 0.5 * eta_mismatch * signal_yield_ts / max(1e-15, y0 + signal_yield_ts)
            
        # If timing attack, extra frame error
        if fault == "Time-Shift Attack" or is_timing_attack:
            qber_shift_delta = qber_mismatch_term + 0.025 * intensity   # additional Bob-side frame error
            qber = float(np.clip(qber + qber_shift_delta, 0.0, 0.50))
            
        # Add physical counting fluctuation to QBER
        qber_measured = float(np.clip(qber + self.rng.normal(0.0, 0.0008), 0.0, 0.50))
        
        # SKR calculation (GLLP / Decoy-state bound with 11% cutoff)
        skr_bps, _ = compute_secret_key_rate(
            qber=qber_measured,
            repetition_rate_hz=self.repetition_rate_hz,
            y0=y0,
            eta_channel=eta_channel,
            eta_bob=self.bob_efficiency,
            mean_photon_number=mean_mu,
            error_correction_efficiency=self.config.error_correction_efficiency,
            critical_qber_threshold=self.config.qber_abort_threshold
        )
        
        # Raw count rate (clicks/s)
        raw_counts = compute_raw_count_rate(
            repetition_rate_hz=self.repetition_rate_hz,
            y0=y0,
            signal_yield=signal_yield
        )
        raw_counts_noisy = max(100.0, raw_counts * (1.0 + self.rng.normal(0.0, 0.008)))
        
        # Special Attack Signatures:
        if is_blinding:
            # Makarov et al. (2009): CW bright illumination saturates APD
            # Raw click rate explodes > 10,000,000 cps, QBER drops to near zero
            raw_counts_noisy = float(1.2e7 + 2.0e7 * intensity + self.rng.normal(0.0, 50000.0))
            qber_measured = float(np.clip(0.005 + 0.004 * (1.0 - intensity) + self.rng.normal(0.0, 0.0004), 0.001, 0.015))
            skr_bps = 0.0  # Classical saturation invalidates single-photon security proof
            
        elif is_pns:
            # Brassard et al. (2000): Eve splits multi-photons, suppresses single photons
            # Decoy-state analysis detects information leakage -> distilled SKR drops strictly to 0.0 bps
            # Raw counts moderately reduced, QBER remains moderate / low
            raw_counts_noisy = max(1000.0, raw_counts * (0.50 - 0.15 * intensity) * (1.0 + self.rng.normal(0.0, 0.01)))
            qber_measured = float(np.clip(qber_measured + 0.01 * intensity, 0.015, 0.045))
            skr_bps = 0.0  # GLLP decoy privacy amplification fails completely
            
        total_loss_db = alpha_noisy * self.fiber_length_km
        
        return QuantumTelemetrySample(
            timestamp=self.internal_clock,
            qber=qber_measured,
            skr_bps=skr_bps,
            raw_counts_hz=raw_counts_noisy,
            dark_counts_hz=dcr_effective,
            visibility=vis_noisy,
            temperature_celsius=temp_noisy,
            timing_jitter_ps=jitter_noisy,
            channel_attenuation_db=total_loss_db,
            fiber_loss_db_per_km=alpha_noisy,
            eavesdropping_fraction=gamma_eve,
            active_fault_label=self.active_fault,
            fault_intensity=intensity,
            humidity_relative_pct=humidity_noisy,
            vibration_g=vibration_noisy,
            supply_voltage_v=voltage_noisy,
            fiber_strain_ue=strain_noisy,
            device_operating_hours=self.device_operating_hours,
            hours_since_calibration=self.hours_since_calibration,
            trap_aging_index=self.trap_aging_index,
            maintenance_event_count=self.maintenance_event_count,
            tec_drive_current_ma=self.tec_drive_current_ma,
            epc_bias_voltage_v=self.epc_bias_voltage_v,
            voa_attenuation_db=self.voa_attenuation_db,
            invalid_reading_flag=self.invalid_reading_flag,
            run_id=self.current_run_id,
        )
