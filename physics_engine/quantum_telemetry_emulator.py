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
    Conforms to ETSI GS QKD 014 data schema.
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


class QuantumTelemetryEmulator:
    """
    Physics-Calibrated Telemetry Emulator.
    Simulates genuine time-series telemetry based on exact physical optical models
    with physical noise (Poissonian counting statistics and detector thermal noise).
    Supports the complete 10-class fault and quantum attack ontology.
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
        Supports both string class names and integer IDs.
        
        Standard 10-Class Ontology:
            - 'Normal' (0): Baseline nominal operating condition
            - 'Optical Misalignment' (1): Polarization rotation / interferometer drift (reduces V)
            - 'Channel Attenuation Event' (2): Fiber bend, dirty connector or extra physical loss
            - 'Detector APD Degradation' (3): Trap degradation causing elevated baseline DCR
            - 'Thermal Drift' (4): TEC failure or cooler saturation raising APD temperature
            - 'Timing Jitter' (5): Clock phase jitter or laser diode timing dispersion
            - 'Intercept-Resend' (6): Eve intercepts fraction gamma of optical pulses
            - 'Detector Blinding' (7): CW bright light saturation (Makarov et al., 2009)
            - 'Photon Number Splitting' (8): Multi-photon pulse splitting on weak coherent source
            - 'Time-Shift Attack' (9): Temporal gating offset / efficiency mismatch (Zhao et al., 2008)
        """
        if isinstance(fault_name, int):
            fault_name = ROOT_CAUSE_ID_TO_LABEL.get(fault_name, "Normal")
        self.active_fault = fault_name
        self.fault_intensity = float(np.clip(intensity, 0.0, 1.0))

    def reset_to_nominal(self) -> None:
        """Restores all parameters to nominal healthy operating values."""
        self.active_fault = "Normal"
        self.fault_intensity = 0.0
        self.fiber_attenuation_db_per_km = self.config.nominal_fiber_attenuation_db_per_km
        self.visibility = self.config.nominal_visibility
        self.temperature_celsius = self.config.nominal_detector_temp_celsius
        self.nominal_dcr_hz = self.config.nominal_dark_count_rate_hz
        self.timing_jitter_ps = self.config.nominal_timing_jitter_ps
        self.eavesdropping_fraction = 0.0

    reset = reset_to_nominal

    def step(self, dt_seconds: float = 1.0) -> QuantumTelemetrySample:
        """
        Advances the physical simulation by dt_seconds and computes the exact physical state.
        Adheres strictly to governing physical optical equations.
        """
        self.simulation_step_index += 1
        self.internal_clock += dt_seconds
        
        # Base physical parameters
        alpha = self.config.nominal_fiber_attenuation_db_per_km
        vis = self.config.nominal_visibility
        temp = self.config.nominal_detector_temp_celsius
        dcr_base = self.config.nominal_dark_count_rate_hz
        jitter = self.config.nominal_timing_jitter_ps
        gamma_eve = 0.0
        
        # Attack flags
        is_blinding = (self.active_fault == "Detector Blinding")
        is_pns = (self.active_fault == "Photon Number Splitting")
        is_timeshift = (self.active_fault == "Time-Shift Attack")
        
        # Apply physical fault perturbations according to active fault mode
        intensity = self.fault_intensity
        if self.active_fault == "Intercept-Resend":
            # Eve intercepts a fraction gamma in range [0.15, 0.70] depending on intensity
            gamma_eve = 0.15 + 0.55 * intensity
            
        elif self.active_fault in ["Optical Misalignment"]:
            # Visibility drops from 0.985 down to ~0.70
            vis = max(0.60, self.config.nominal_visibility - 0.28 * intensity)
            
        elif self.active_fault in ["Channel Attenuation Event", "Channel Attenuation"]:
            # Fiber loss increases from 0.20 dB/km up to 0.75 dB/km (fiber macrobend / splice degradation)
            alpha = self.config.nominal_fiber_attenuation_db_per_km + 0.55 * intensity
            
        elif self.active_fault in ["Detector APD Degradation", "APD Aging"]:
            # Dark count rate increases 4x to 10x due to trap accumulation
            dcr_base = self.config.nominal_dark_count_rate_hz * (1.0 + 8.0 * intensity)
            
        elif self.active_fault in ["Thermal Drift"]:
            # Temperature rises from -40 C up to +10 C (TEC cooler fault)
            temp = self.config.nominal_detector_temp_celsius + 50.0 * intensity
            
        elif self.active_fault == "Timing Jitter":
            # Jitter expands from 65 ps to 350 ps
            jitter = self.config.nominal_timing_jitter_ps + 285.0 * intensity
            
        elif is_timeshift:
            # Time-shift attack: gate timing offset / asymmetry (130 - 220 ps)
            # P4.4: Also model basis-dependent detection efficiency mismatch.
            # In a real time-shift attack Eve shifts the gate window so that one
            # basis (e.g. H/V) has significantly lower detection efficiency than
            # the other (+/- 45 basis). This creates an asymmetric click pattern
            # that differs from hardware timing jitter (which affects all bases equally).
            jitter = self.config.nominal_timing_jitter_ps + 70.0 + 85.0 * intensity

        # Add micro-fluctuations (shot noise & physical drift)
        alpha_noisy = max(0.15, alpha + self.rng.normal(0.0, 0.003))
        vis_noisy = float(np.clip(vis + self.rng.normal(0.0, 0.0015), 0.50, 0.999))
        temp_noisy = temp + self.rng.normal(0.0, 0.10)
        jitter_noisy = max(40.0, jitter + self.rng.normal(0.0, 1.5))
        
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
            mean_photon_number=self.mean_photon_number
        )
        
        # QBER calculation
        qber = compute_quantum_bit_error_rate(
            y0=y0,
            signal_yield=signal_yield,
            optical_error_rate=e_opt,
            eavesdropping_fraction=gamma_eve
        )
        
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
            signal_yield_ts = compute_signal_yield(eta_channel, avg_eta / self.bob_efficiency, self.mean_photon_number)
            # Basis mismatch introduces extra QBER: Eve's shifted window maps H→+45 errors
            qber_mismatch_term = 0.5 * eta_mismatch * signal_yield_ts / max(1e-15, y0 + signal_yield_ts)
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
            mean_photon_number=self.mean_photon_number,
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
            fault_intensity=intensity
        )
