"""
Quantum System Configuration and Physical Constants
Standard: ETSI GS QKD 014 / GLLP / Decoy-State BB84 Protocols
Governing Specifications: ITU-T Y.3800 / Shor-Preskill / Ma-Qi-Zhao-Lo
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List

# Physical constants
PLANCK_CONSTANT_H: float = 6.62607015e-34  # J*s (Origin: Theory-derived CODATA)
SPEED_OF_LIGHT_VACUUM_C: float = 2.99792458e8  # m/s (Origin: Theory-derived SI)
FIBER_REFRACTIVE_INDEX_N: float = 1.4682  # Standard SMF-28 at 1550 nm (Origin: Theory-derived ITU-T)
SPEED_OF_LIGHT_FIBER: float = SPEED_OF_LIGHT_VACUUM_C / FIBER_REFRACTIVE_INDEX_N
BOLTZMANN_CONSTANT_K: float = 1.380649e-23  # J/K
ELECTRON_CHARGE_Q: float = 1.602176634e-19  # C


@dataclass(frozen=True)
class QKDPhysicsConfig:
    """
    Physical parameters defining the optical channel, transceivers, and detectors.
    Every parameter has a documented physical or operational origin.
    """
    # Optical channel
    standard_wavelength_nm: float = 1550.0                # Origin: Calibration-derived (C-band)
    nominal_fiber_attenuation_db_per_km: float = 0.20     # Origin: Theory-derived (ITU-T G.652 SMF-28)
    default_fiber_length_km: float = 25.0                 # Origin: Operator-configured (Metropolitan link)
    
    # Transmitter (Alice)
    pulse_repetition_rate_hz: float = 1.0e8               # Origin: Calibration-derived (100 MHz clock rate)
    mean_photon_number_signal: float = 0.50               # Origin: Theory-derived (Decoy-state BB84 optimum)
    mean_photon_number_decoy: float = 0.10                # Origin: Theory-derived (Decoy intensity)
    mean_photon_number_vacuum: float = 0.00               # Origin: Theory-derived
    
    # Receiver (Bob)
    bob_detection_efficiency: float = 0.15                # Origin: Calibration-derived (Optics + APD quantum eff.)
    nominal_visibility: float = 0.985                     # Origin: Calibration-derived (Fringe contrast)
    nominal_optical_error: float = 0.0075                 # Origin: Theory-derived e_opt = (1 - V) / 2
    nominal_dark_count_rate_hz: float = 500.0             # Origin: Calibration-derived (Gated InGaAs SPAD @ -40 C)
    nominal_detector_temp_celsius: float = -40.0          # Origin: Calibration-derived (TEC target setpoint)
    nominal_timing_jitter_ps: float = 65.0                # Origin: Calibration-derived (FWHM time tagging jitter)
    
    # Security & Protocol Bounds
    qber_abort_threshold: float = 0.110                   # Origin: Theory-derived (Shor-Preskill / GLLP 11.0% limit)
    qber_warning_threshold: float = 0.080                 # Origin: Operator-configured (Early warning trigger)
    error_correction_efficiency: float = 1.16             # Origin: Theory-derived (Cascade / LDPC f(e) factor)

    # Quantum Attack Detection Thresholds (Layer 9 Intelligence)
    detector_blinding_counts_threshold_hz: float = 1.0e7  # Origin: Makarov et al. (2009) saturation threshold (>10 Mcps @ 100 MHz clock)
    detector_blinding_max_qber: float = 0.020             # Blinding forces deterministic clicks (low QBER)
    time_shift_jitter_threshold_ps: float = 120.0         # Origin: Zhao et al. (2008) gate offset limit
    time_shift_min_qber: float = 0.065                    # Elevated error in shifted gating window
    pns_max_qber_threshold: float = 0.060                 # PNS maintains moderate QBER while attacking multi-photons


# Layer 0: Standardized 10-Class Fault and Attack Ontology
ROOT_CAUSE_CLASSES: List[str] = [
    "Normal",
    "Optical Misalignment",
    "Channel Attenuation Event",
    "Detector APD Degradation",
    "Thermal Drift",
    "Timing Jitter",
    "Intercept-Resend",
    "Detector Blinding",
    "Photon Number Splitting",
    "Time-Shift Attack",
]

ROOT_CAUSE_LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(ROOT_CAUSE_CLASSES)}
ROOT_CAUSE_ID_TO_LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(ROOT_CAUSE_CLASSES)}

# Layer 0: Telecom Alarm Severity Mapping
ALARM_SEVERITY_LEVELS: Dict[str, str] = {
    "Normal": "NORMAL",                      # Healthy baseline
    "Optical Misalignment": "MEDIUM",        # Polarization drift / interferometer contrast loss
    "Channel Attenuation Event": "HIGH",     # Macro-bend / connector loss
    "Detector APD Degradation": "MAJOR",     # APD trap buildup
    "Thermal Drift": "MEDIUM",               # Environmental thermal swing
    "Timing Jitter": "MEDIUM",               # Clock sync phase wander
    "Intercept-Resend": "CRITICAL",          # Active quantum eavesdropping
    "Detector Blinding": "CRITICAL",         # Optical saturation physical attack
    "Photon Number Splitting": "CRITICAL",   # Multi-photon pulse splitting attack
    "Time-Shift Attack": "CRITICAL",         # Gating window phase attack
}
