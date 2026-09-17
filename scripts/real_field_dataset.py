"""
VECTOR-Q OpenQKD-Inspired Simulation Telemetry Dataset Generator
Module: Physics-Based Synthetic Telemetry Validation
Governing Standards: ETSI GS QKD 014 / ITU-T Y.3800

Generates physics-based synthetic telemetry representing an OpenQKD-inspired simulation
(22.7 km SMF-28 underground conduit based on Geneva-CERN / OpenQKD field testbed parameters).
100% synthetic, emulator-generated telemetry produced by QuantumTelemetryEmulator.
Includes modeled diurnal temperature swings, transit mechanical vibrations, slow birefringence drift,
and an afternoon junction-box macro-bend maintenance event.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.qkd_system_parameters import QKDPhysicsConfig
from config.dataset_governance import (
    ALL_FEATURE_COLUMNS,
    FAULT_LABEL_TO_ID,
    FAULT_ID_TO_LABEL,
)
from physics_engine.quantum_telemetry_emulator import QuantumTelemetryEmulator, QuantumTelemetrySample
from anomaly_detection.sliding_window_features import TelemetryFeatureExtractor


def generate_real_field_dataset(output_dir: str = "data") -> pd.DataFrame:
    """
    Generates physics-based synthetic telemetry (1,440 samples, 1-minute cadence)
    representing an OpenQKD-inspired simulation of a 22.7 km dark fiber link.
    100% synthetic, emulator-generated telemetry via QuantumTelemetryEmulator.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(42)

    # 22.7 km metropolitan link
    cfg = QKDPhysicsConfig(default_fiber_length_km=22.7)
    emulator = QuantumTelemetryEmulator(config=cfg, random_seed=42)
    extractor = TelemetryFeatureExtractor(max_buffer_size=35)

    # 1,440 minutes in 24 hours
    n_minutes = 1440
    records = []

    # Warmup feature ring buffer
    for _ in range(35):
        sample = emulator.step(dt_seconds=60.0)
        extractor.add_sample(sample)

    for m in range(n_minutes):
        hour_of_day = (m / 60.0) % 24.0

        # 1. Diurnal solar cycle
        # Ambient temperature ranges from 16.5 C (night) to 33.5 C (afternoon peak at 14:00)
        t_ambient = 25.0 + 8.5 * np.sin(2 * np.pi * (hour_of_day - 8.0) / 24.0) + rng.normal(0, 0.25)
        # APD cryostat maintains -40 C with slight daytime thermal load (+/- 0.6 C)
        cryo_load = max(0.0, (t_ambient - 22.0) * 0.05)
        detector_temp = -40.0 + cryo_load + rng.normal(0, 0.08)

        # 2. Rush hour transit vibrations (morning 07:30-09:30, evening 17:00-19:00)
        is_rush_morning = (7.5 <= hour_of_day <= 9.5)
        is_rush_evening = (17.0 <= hour_of_day <= 19.0)
        base_vib = 0.015
        if is_rush_morning or is_rush_evening:
            base_vib = 0.075 + 0.035 * np.sin(np.pi * (hour_of_day % 2.0) / 2.0)
        vibration_g = max(0.005, base_vib + rng.normal(0, 0.008))

        # Conduit micro-strain fluctuates with vibration and soil thermal expansion
        conduit_strain = 14.0 + 120.0 * vibration_g + 0.25 * (t_ambient - 20.0) + rng.normal(0, 0.4)

        # 3. Slow diurnal polarization drift (optical visibility oscillation)
        # Solar heating rotates SMF-28 birefringence axes
        vis_base = 0.985 - 0.012 * (0.5 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 6.0) / 24.0))
        visibility = float(np.clip(vis_base + rng.normal(0, 0.0015), 0.95, 0.992))

        # 4. Field Maintenance Incident: Fiber Tray Macrobend Event
        # Local telecom crew maintenance at junction vault between 13:30 (m=810) and 14:30 (m=870)
        is_incident = (810 <= m <= 870)
        incident_intensity = 0.0
        if is_incident:
            # Macro-bend ramps up, reaches peak at m=840, then resolved
            incident_progress = (m - 810) / 60.0
            incident_intensity = float(np.sin(np.pi * incident_progress) * 0.75)
            fault_label = "Fiber Bend"
            alpha_effective = 0.20 + 0.45 * incident_intensity
            conduit_strain += 75.0 * incident_intensity
        else:
            fault_label = "Normal"
            alpha_effective = 0.20 + rng.normal(0, 0.002)

        # Update emulator physical state
        emulator.temperature_celsius = detector_temp
        emulator.fiber_attenuation_db_per_km = alpha_effective
        emulator.visibility = visibility
        emulator.vibration_g = vibration_g
        emulator.fiber_strain_ue = conduit_strain
        if is_incident:
            emulator.active_fault = "Fiber Bend"
            emulator.fault_intensity = incident_intensity
        else:
            emulator.active_fault = "Normal"
            emulator.fault_intensity = 0.0

        sample = emulator.step(dt_seconds=60.0)
        extractor.add_sample(sample)
        features = extractor.extract_features(sample)

        # Build standardized telemetry record
        row = {col: float(features.get(col, 0.0)) for col in ALL_FEATURE_COLUMNS}
        row["timestamp"] = 1789600000.0 + (m * 60.0)
        row["fault_class"] = fault_label
        row["fault_id"] = FAULT_LABEL_TO_ID.get(fault_label, 0)
        row["fault_intensity"] = float(incident_intensity)
        row["is_anomaly"] = int(is_incident)
        records.append(row)

    df = pd.DataFrame(records)

    # Save Parquet & CSV
    parquet_path = os.path.join(output_dir, "real_field_telemetry.parquet")
    csv_path = os.path.join(output_dir, "real_field_telemetry.csv")
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    # Compute SHA-256 hashes
    def get_hash(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    h_parquet = get_hash(parquet_path)
    h_csv = get_hash(csv_path)

    print("=" * 78)
    print("OPENQKD-INSPIRED SIMULATION TELEMETRY DATASET GENERATED")
    print("=" * 78)
    print(f"Deployment: 22.7 km SMF-28 OpenQKD-Inspired Simulation (Geneva-CERN spec)")
    print(f"Duration:   24.0 Hours (1,440 continuous samples at 1-min cadence)")
    print(f"Nominal:    {np.sum(df['is_anomaly'] == 0)} samples")
    print(f"Incident:   {np.sum(df['is_anomaly'] == 1)} samples (Fiber Bend Junction Box Maintenance)")
    print(f"Parquet:    {parquet_path} (SHA-256: {h_parquet})")
    print(f"CSV:        {csv_path} (SHA-256: {h_csv})")
    print("=" * 78)

    return df


if __name__ == "__main__":
    generate_real_field_dataset()
