"""
VECTOR-Q Dataset Governance & Specification Framework
Single Source of Truth for all Telemetry Features, Splits, and Fault Taxonomies.
Governing Standards: ETSI GS QKD 014 / ITU-T Y.3800 / GLLP Decoy-State BB84

Author: Senior Quantum Systems & Applied ML Engineering Team
Version: 3.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import os

# ==============================================================================
# 1. OFFICIAL STANDARDIZED FAULT TAXONOMY (9-CLASS ONTOLOGY)
# ==============================================================================

OFFICIAL_FAULT_CLASSES: List[str] = [
    "Normal",
    "Temperature Drift",
    "Fiber Bend",
    "Polarization Drift",
    "Detector Aging",
    "Timing Misalignment",
    "Power Instability",
    "Humidity Impact",
    "Unknown Fault",
]

FAULT_LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(OFFICIAL_FAULT_CLASSES)}
FAULT_ID_TO_LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(OFFICIAL_FAULT_CLASSES)}

# Operational Alarm Severity Mapping
FAULT_ALARM_SEVERITY: Dict[str, str] = {
    "Normal": "NORMAL",
    "Temperature Drift": "MEDIUM",
    "Fiber Bend": "HIGH",
    "Polarization Drift": "MEDIUM",
    "Detector Aging": "MAJOR",
    "Timing Misalignment": "MEDIUM",
    "Power Instability": "HIGH",
    "Humidity Impact": "MEDIUM",
    "Unknown Fault": "CRITICAL",
    # Legacy compatibility aliases
    "Thermal Drift": "MEDIUM",
    "Optical Misalignment": "MEDIUM",
    "Channel Attenuation": "HIGH",
    "Channel Attenuation Event": "HIGH",
    "Detector APD Degradation": "MAJOR",
    "APD Aging": "MAJOR",
    "Timing Jitter": "MEDIUM",
    "Intercept-Resend": "CRITICAL",
    "Time-Shift Attack": "CRITICAL",
    "Detector Blinding": "CRITICAL",
    "Photon Number Splitting": "CRITICAL",
}

# Physical mechanism explanations
FAULT_PHYSICAL_MECHANISMS: Dict[str, str] = {
    "Normal": "System operates nominally within calibrated ITU-T / ETSI specifications.",
    "Temperature Drift": "Thermoelectric cooler (TEC) setpoint drift or ambient heating elevates detector dark carrier generation.",
    "Fiber Bend": "Macrobend or splice strain increases channel attenuation, reducing single-photon yield below noise floor.",
    "Polarization Drift": "Thermal/mechanical birefringence rotation degrades interferometric visibility and increases optical error.",
    "Detector Aging": "Semiconductor single-photon avalanche diode (SPAD) trap accumulation raises baseline dark count rate.",
    "Timing Misalignment": "Receiver clock phase wander or laser jitter widens gating window offset, capturing noise photons.",
    "Power Instability": "Alice laser diode drive rail fluctuation or central wavelength drift destabilizes photon flux.",
    "Humidity Impact": "Enclosure relative humidity surge or optical connector condensation introduces scatter and loss.",
    "Unknown Fault": "Out-of-distribution anomaly without sufficient domain support; requires operator physical inspection.",
}


# ==============================================================================
# 2. OFFICIAL FEATURE SPECIFICATION (34 TOTAL FEATURES)
# ==============================================================================

# Group 1: Operational Observables (7 features)
OPERATIONAL_FEATURES: List[str] = [
    "qber",                    # Quantum Bit Error Rate [0.0, 0.50]
    "skr_bps",                 # Usable Secret Key Rate (bps)
    "channel_loss_db",         # Total optical channel attenuation (dB)
    "dark_counts_hz",          # Detector dark count rate (Hz)
    "raw_counts_hz",           # Total raw photon detection click rate (Hz)
    "visibility",              # Optical fringe visibility [0.0, 1.0]
    "timing_jitter_ps",        # Receiver timing jitter FWHM (ps)
]

# Group 2: Environmental Observables (5 features)
ENVIRONMENTAL_FEATURES: List[str] = [
    "temperature_celsius",     # Detector / Enclosure temperature (°C)
    "humidity_relative_pct",   # Enclosure relative humidity (%)
    "vibration_g",             # Mechanical vibration / acoustic acceleration (g)
    "supply_voltage_v",        # Transceiver DC supply rail voltage (V)
    "fiber_strain_ue",         # Micro-strain induced on fiber conduit (με)
]

# Group 3: Operating History & Maintenance Metadata (4 features)
MAINTENANCE_FEATURES: List[str] = [
    "device_operating_hours",  # Cumulative active operating hours
    "hours_since_calibration", # Operating hours elapsed since last optical recalibration
    "trap_aging_index",        # Cumulative detector crystal trap degradation metric [0.0, 1.0]
    "maintenance_event_count", # Count of historical maintenance interventions
]

# Group 4: Physical Ratios & Statistical Moments (18 features, W=25 rolling window)
DERIVED_MOMENT_FEATURES: List[str] = [
    # Domain Ratios
    "signal_to_noise_ratio",          # raw_counts / max(1, dark_counts)
    "optical_error_ratio",            # e_opt = (1 - visibility) / 2
    "qber_to_visibility_mismatch",    # measured_qber - optical_error_ratio
    "skr_to_qber_ratio",              # skr / max(1e-5, qber)
    # Rolling Statistical Moments (W=25)
    "qber_roll_mean_25",
    "qber_roll_std_25",
    "qber_iqr_25",
    "raw_counts_roll_mean_25",
    "raw_counts_roll_std_25",
    "dark_counts_roll_mean_25",
    "dark_counts_roll_std_25",
    "temp_roll_mean_25",
    "humidity_roll_mean_25",
    # Temporal Dynamics (Slopes & Accelerations)
    "qber_slope_25",                  # dQBER/dt (velocity)
    "qber_acceleration_25",           # d²QBER/dt² (acceleration)
    "corr_temp_qber",                 # Corr(Temperature, QBER)
    "corr_humidity_loss",             # Corr(Humidity, Channel Loss)
    "corr_strain_vis",                # Corr(Fiber Strain, Visibility)
]

# Official 34 Feature Vector Specification (Aligned with Model Checkpoints)
ALL_FEATURE_COLUMNS: List[str] = [
    # 1. Raw Telemetry Observables
    "qber",
    "skr_bps",
    "raw_counts_hz",
    "dark_counts_hz",
    "visibility",
    "temperature_celsius",
    "timing_jitter_ps",
    "channel_attenuation_db",
    # 2. Physical Domain Ratios
    "count_to_dark_ratio",
    "signal_to_noise_ratio",
    "optical_error_ratio",
    "qber_to_visibility_mismatch",
    "skr_to_qber_ratio",
    # 3. Statistical Moments (W=25)
    "qber_roll_mean_25",
    "qber_roll_std_25",
    "qber_roll_var_25",
    "qber_iqr_25",
    "raw_counts_roll_mean_25",
    "raw_counts_roll_std_25",
    "dark_counts_roll_mean_25",
    "dark_counts_roll_std_25",
    "visibility_roll_mean_25",
    "visibility_roll_std_25",
    "temp_roll_mean_25",
    "jitter_roll_mean_25",
    # 4. Temporal Derivatives
    "qber_slope_25",
    "qber_acceleration_25",
    "raw_counts_slope_25",
    "dark_counts_slope_25",
    "visibility_slope_25",
    "temp_slope_25",
    # 5. Cross-Channel Correlations
    "corr_temp_qber",
    "corr_vis_qber",
    "corr_counts_loss",
]

TOTAL_FEATURE_COUNT: int = len(ALL_FEATURE_COLUMNS)  # 34 features


# ==============================================================================
# 3. OFFICIAL DATASET PARTITION SPECIFICATION
# ==============================================================================

@dataclass(frozen=True)
class DatasetPartitionSpec:
    """Defines an official partition of the VECTOR-Q evaluation dataset."""
    partition_name: str
    target_sample_count: int
    fault_classes_included: List[str]
    description: str
    file_path: str


OFFICIAL_DATASET_SPECS: Dict[str, DatasetPartitionSpec] = {
    "training": DatasetPartitionSpec(
        partition_name="training",
        target_sample_count=4000,
        fault_classes_included=OFFICIAL_FAULT_CLASSES[:-1],  # Excludes Unknown Fault
        description="Standard training partition containing healthy baselines and 8 known fault modes.",
        file_path="data/training_dataset.parquet",
    ),
    "validation": DatasetPartitionSpec(
        partition_name="validation",
        target_sample_count=1000,
        fault_classes_included=OFFICIAL_FAULT_CLASSES[:-1],  # Excludes Unknown Fault
        description="Held-out calibration partition for isotonic regression and hyperparameter tuning.",
        file_path="data/validation_dataset.parquet",
    ),
    "testing_known": DatasetPartitionSpec(
        partition_name="testing_known",
        target_sample_count=1000,
        fault_classes_included=OFFICIAL_FAULT_CLASSES[:-1],  # Known faults only
        description="Independent held-out test partition for evaluating known fault top-1 and top-3 accuracy.",
        file_path="data/testing_known_dataset.parquet",
    ),
    "testing_mixed": DatasetPartitionSpec(
        partition_name="testing_mixed",
        target_sample_count=600,
        fault_classes_included=["Temperature Drift", "Fiber Bend", "Polarization Drift", "Detector Aging", "Humidity Impact", "Power Instability"],
        description="Composite scenarios containing simultaneous/overlapping multi-fault degradation.",
        file_path="data/testing_mixed_dataset.parquet",
    ),
    "testing_unknown": DatasetPartitionSpec(
        partition_name="testing_unknown",
        target_sample_count=500,
        fault_classes_included=["Unknown Fault"],
        description="Out-of-distribution degradation sequences unseen during training to validate selective rejection.",
        file_path="data/testing_unknown_dataset.parquet",
    ),
}
