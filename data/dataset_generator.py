"""
VECTOR-Q Standardized Benchmark Dataset Generator
Phase 1 & Phase 2: Data Foundation & Reproducible Benchmark Datasets
Generates Physics-Calibrated QKD Telemetry Datasets (Normal, Single-Fault, Mixed-Fault, Unknown-Fault)
Governing Standards: ETSI GS QKD 014 / ITU-T Y.3800 / GLLP Decoy-State BB84

Author: Senior Quantum Systems & Applied ML Engineering Team
Version: 3.0.0
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.qkd_system_parameters import QKDPhysicsConfig
from config.dataset_governance import (
    OFFICIAL_FAULT_CLASSES,
    FAULT_LABEL_TO_ID,
    FAULT_ID_TO_LABEL,
    ALL_FEATURE_COLUMNS,
    TOTAL_FEATURE_COUNT,
)
from physics_engine.quantum_telemetry_emulator import QuantumTelemetryEmulator, QuantumTelemetrySample
from anomaly_detection.sliding_window_features import TelemetryFeatureExtractor, FEATURE_COLUMN_NAMES


def compute_file_sha256(filepath: str) -> str:
    """Computes cryptographic SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class VectorQDatasetGenerator:
    """
    Generates reproducible, physics-governed telemetry benchmark datasets
    for training, validation, multi-fault stress testing, and OOD unknown fault detection.
    """

    def __init__(self, random_seed: int = 42, output_dir: str = "data"):
        self.random_seed = random_seed
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.rng = np.random.RandomState(random_seed)

    def _generate_sequence(
        self,
        emulator: QuantumTelemetryEmulator,
        extractor: TelemetryFeatureExtractor,
        n_samples: int,
        fault_label: str,
        fault_intensity: float = 0.0,
        dt_seconds: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Generates a steady sequence of telemetry points under a given fault regime."""
        records: List[Dict[str, Any]] = []
        if fault_label != "Normal":
            emulator.inject_fault(fault_label, fault_intensity)
        else:
            emulator.clear_fault()

        for _ in range(n_samples):
            sample = emulator.step(dt_seconds=dt_seconds)
            features = extractor.extract_features(sample)
            
            # Row dict with governance features + metadata
            row: Dict[str, Any] = {col: float(features.get(col, 0.0)) for col in ALL_FEATURE_COLUMNS}
            row["timestamp"] = float(sample.timestamp)
            row["fault_class"] = fault_label
            row["fault_id"] = FAULT_LABEL_TO_ID.get(fault_label, FAULT_LABEL_TO_ID.get("Unknown Fault", 8))
            row["fault_intensity"] = float(fault_intensity)
            row["is_anomaly"] = int(fault_label != "Normal")
            records.append(row)

        return records

    def generate_normal_dataset(self, n_samples: int = 2000) -> pd.DataFrame:
        """
        Generates healthy baseline telemetry across nominal operating fluctuations,
        varying fiber lengths (20 km to 60 km), and daytime thermal drift ranges.
        """
        records: List[Dict[str, Any]] = []
        sub_lengths = [25.0, 35.0, 50.0, 60.0]
        samples_per_sub = n_samples // len(sub_lengths)

        for i, length in enumerate(sub_lengths):
            seed = self.random_seed + i * 17
            cfg = QKDPhysicsConfig(default_fiber_length_km=length)
            emu = QuantumTelemetryEmulator(config=cfg, random_seed=seed)
            ext = TelemetryFeatureExtractor(max_buffer_size=35)
            
            # Warm up feature extractor ring buffer
            for _ in range(30):
                ext.add_sample(emu.step(1.0))

            seq = self._generate_sequence(emu, ext, samples_per_sub, fault_label="Normal", fault_intensity=0.0)
            records.extend(seq)

        df = pd.DataFrame(records)
        return df

    def generate_single_fault_dataset(self, samples_per_class: int = 250) -> pd.DataFrame:
        """
        Generates isolated, single-fault degradation sequences across the 8 known classes
        (Normal + 7 physical fault modes) across multiple perturbation intensities.
        """
        records: List[Dict[str, Any]] = []
        known_faults = [c for c in OFFICIAL_FAULT_CLASSES if c != "Unknown Fault"]
        intensities = [0.25, 0.50, 0.75, 1.00]

        for fault_idx, fault_name in enumerate(known_faults):
            if fault_name == "Normal":
                emu = QuantumTelemetryEmulator(random_seed=self.random_seed + 101)
                ext = TelemetryFeatureExtractor(max_buffer_size=35)
                for _ in range(30):
                    ext.add_sample(emu.step(1.0))
                records.extend(self._generate_sequence(emu, ext, samples_per_class, "Normal", 0.0))
                continue

            per_intensity = samples_per_class // len(intensities)
            for j, intensity in enumerate(intensities):
                seed = self.random_seed + fault_idx * 50 + j * 7
                emu = QuantumTelemetryEmulator(random_seed=seed)
                ext = TelemetryFeatureExtractor(max_buffer_size=35)
                for _ in range(30):
                    ext.add_sample(emu.step(1.0))

                seq = self._generate_sequence(emu, ext, per_intensity, fault_name, intensity)
                records.extend(seq)

        df = pd.DataFrame(records)
        return df

    def generate_mixed_fault_dataset(self, n_samples: int = 800) -> pd.DataFrame:
        """
        Generates composite, overlapping multi-fault degradations
        (e.g., simultaneous thermal drift + fiber attenuation, polarization rotation + timing jitter).
        """
        records: List[Dict[str, Any]] = []
        composite_scenarios = [
            ("Temperature Drift", "Humidity Impact", 0.6, 0.5),
            ("Fiber Bend", "Polarization Drift", 0.7, 0.4),
            ("Power Instability", "Timing Misalignment", 0.5, 0.6),
            ("Detector Aging", "Temperature Drift", 0.8, 0.4),
        ]
        per_scenario = n_samples // len(composite_scenarios)

        for s_idx, (f1, f2, int1, int2) in enumerate(composite_scenarios):
            seed = self.random_seed + 500 + s_idx * 23
            emu = QuantumTelemetryEmulator(random_seed=seed)
            ext = TelemetryFeatureExtractor(max_buffer_size=35)
            for _ in range(30):
                ext.add_sample(emu.step(1.0))

            # Inject primary fault into emulator state, secondary compound perturbation
            emu.inject_fault(f1, int1)
            for _ in range(per_scenario):
                sample = emu.step(1.0)
                # Apply compound physics modification to simulate co-occurring secondary fault
                if f2 == "Humidity Impact":
                    sample.humidity_relative_pct = float(np.clip(sample.humidity_relative_pct + 35.0 * int2, 20.0, 95.0))
                    sample.channel_attenuation_db += 1.2 * int2
                elif f2 == "Polarization Drift":
                    sample.visibility = max(0.65, sample.visibility - 0.15 * int2)
                elif f2 == "Timing Misalignment":
                    sample.timing_jitter_ps += 100.0 * int2
                elif f2 == "Temperature Drift":
                    sample.temperature_celsius += 25.0 * int2

                features = ext.extract_features(sample)
                row = {col: float(features.get(col, 0.0)) for col in ALL_FEATURE_COLUMNS}
                row["timestamp"] = float(sample.timestamp)
                row["fault_class"] = f"{f1} + {f2}"
                row["fault_id"] = FAULT_LABEL_TO_ID.get(f1, 8)
                row["fault_intensity"] = float(max(int1, int2))
                row["is_anomaly"] = 1
                records.append(row)

        df = pd.DataFrame(records)
        return df

    def generate_unknown_fault_dataset(self, n_samples: int = 600) -> pd.DataFrame:
        """
        Generates out-of-distribution synthetic anomalies that deviate strongly from
        all 8 known training modes to rigorously benchmark selective rejection.
        """
        records: List[Dict[str, Any]] = []
        seed = self.random_seed + 999
        emu = QuantumTelemetryEmulator(random_seed=seed)
        ext = TelemetryFeatureExtractor(max_buffer_size=35)
        for _ in range(30):
            ext.add_sample(emu.step(1.0))

        # Synthetic non-linear anomalous regimes
        regimes = [
            ("Unknown High-Frequency Polarization Modulation", 0.7),
            ("Synthetic Multi-Tone Phase Dither", 0.85),
            ("Anomalous Backscatter Chirp", 0.65),
        ]
        per_regime = n_samples // len(regimes)

        for r_idx, (r_name, r_intensity) in enumerate(regimes):
            emu.inject_fault("Unknown Fault", r_intensity)
            for step_i in range(per_regime):
                sample = emu.step(1.0)
                # Introduce exotic out-of-distribution coupling (e.g. vibration modulating visibility non-linearly)
                sample.vibration_g = 0.35 + 0.15 * np.sin(step_i * 0.4)
                sample.visibility = float(np.clip(0.88 - 0.20 * np.sin(step_i * 0.2) ** 2, 0.55, 0.99))
                sample.supply_voltage_v = float(3.30 + 0.45 * np.cos(step_i * 0.3))

                features = ext.extract_features(sample)
                row = {col: float(features.get(col, 0.0)) for col in ALL_FEATURE_COLUMNS}
                row["timestamp"] = float(sample.timestamp)
                row["fault_class"] = "Unknown Fault"
                row["fault_id"] = FAULT_LABEL_TO_ID["Unknown Fault"]
                row["fault_intensity"] = float(r_intensity)
                row["is_anomaly"] = 1
                records.append(row)

        df = pd.DataFrame(records)
        return df

    def generate_challenge_episodes(
        self,
        episodes_per_condition: int = 50,
        timesteps_per_episode: int = 400,
        dt_seconds: float = 1.0,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generates official episode dataset for the 1-link challenge scope across 6 conditions:
          - normal
          - thermal_drift
          - optical_misalignment
          - channel_loss
          - combined_fault (thermal + misalignment concurrently)
          - unfamiliar (zero-day non-thermal anomaly)

        Performs run-level group splitting (60% train, 10% calibration, 10% validation, 20% test).
        Zero cross-run data leakage. Target horizons at t+60s and t+300s.
        """
        conditions = [
            ("normal", "Normal", 0.0),
            ("thermal_drift", "Temperature Drift", 0.65),
            ("optical_misalignment", "Polarization Drift", 0.70),
            ("channel_loss", "Fiber Bend", 0.70),
            ("combined_fault", "Combined Fault", 0.65),
            ("unfamiliar", "Unknown Fault", 0.80),
        ]

        n_train = int(episodes_per_condition * 0.60)
        n_cal = int(episodes_per_condition * 0.10)
        n_val = int(episodes_per_condition * 0.10)
        # Remaining is test

        all_records: List[Dict[str, Any]] = []
        run_manifest: Dict[str, str] = {}

        for cond_slug, fault_name, def_intensity in conditions:
            for ep_idx in range(episodes_per_condition):
                run_id = f"ep_{cond_slug}_{ep_idx:03d}"

                if ep_idx < n_train:
                    split = "train"
                elif ep_idx < n_train + n_cal:
                    split = "calibration"
                elif ep_idx < n_train + n_cal + n_val:
                    split = "validation"
                else:
                    split = "test"
                run_manifest[run_id] = split

                seed = self.random_seed + hash(run_id) % 100000
                emu = QuantumTelemetryEmulator(random_seed=seed)
                emu.current_run_id = run_id
                ext = TelemetryFeatureExtractor(max_buffer_size=35)

                # Warm up buffer
                for _ in range(30):
                    ext.add_sample(emu.step(dt_seconds))

                # Onset between timestep 35 and 55 (except normal)
                onset_step = 40 + (ep_idx % 20) if fault_name != "Normal" else 9999
                mitigation_step = onset_step + 140 if (ep_idx % 2 == 1 and fault_name != "Normal") else 9999

                ep_records: List[Dict[str, Any]] = []

                for step_t in range(timesteps_per_episode):
                    # Check fault onset
                    if step_t == onset_step:
                        intensity = float(np.clip(def_intensity + (ep_idx % 5) * 0.04 - 0.08, 0.40, 0.95))
                        emu.inject_fault(fault_name, intensity)

                    # Check mitigation actuation
                    if step_t == mitigation_step:
                        if fault_name in ["Temperature Drift"]:
                            emu.apply_actuation("ACTION_TEC_PHASE_COMPENSATION")
                        elif fault_name in ["Polarization Drift"]:
                            emu.apply_actuation("ACTION_POLARIZATION_FAST_EPC")
                        elif fault_name in ["Fiber Bend"]:
                            emu.apply_actuation("ACTION_ATTENUATION_COMPENSATION")
                        elif fault_name == "Combined Fault":
                            emu.apply_actuation("ACTION_TEC_PHASE_COMPENSATION")
                            emu.apply_actuation("ACTION_POLARIZATION_FAST_EPC")

                    sample = emu.step(dt_seconds)
                    features = ext.extract_features(sample)

                    # Standardized observable features
                    row = {col: float(features.get(col, 0.0)) for col in ALL_FEATURE_COLUMNS}
                    row["run_id"] = run_id
                    row["timestamp"] = float(sample.timestamp)
                    row["step_index"] = step_t
                    row["split"] = split

                    # Multi-label ground truth targets (clean separation from inputs)
                    active = emu.active_fault
                    row["has_thermal_drift"] = int(active in ["Temperature Drift", "Combined Fault"])
                    row["has_misalignment"] = int(active in ["Polarization Drift", "Combined Fault"])
                    row["has_channel_loss"] = int(active in ["Fiber Bend"])
                    row["is_unfamiliar"] = int(active in ["Unknown Fault"])
                    row["is_anomaly"] = int(active != "Normal")

                    # Hardware control registers & flags
                    row["tec_drive_current_ma"] = float(sample.tec_drive_current_ma)
                    row["epc_bias_voltage_v"] = float(sample.epc_bias_voltage_v)
                    row["voa_attenuation_db"] = float(sample.voa_attenuation_db)
                    row["invalid_reading_flag"] = int(sample.invalid_reading_flag)

                    ep_records.append(row)

                # Compute future forecast targets for this run (no cross-run leakage)
                for i in range(len(ep_records)):
                    # Target at +60s and +300s (or max available in episode)
                    target_60_idx = i + 60
                    target_300_idx = i + 300
                    ep_records[i]["target_qber_60s"] = (
                        ep_records[target_60_idx]["qber"] if target_60_idx < len(ep_records) else np.nan
                    )
                    ep_records[i]["target_skr_60s"] = (
                        ep_records[target_60_idx]["skr_bps"] if target_60_idx < len(ep_records) else np.nan
                    )
                    ep_records[i]["target_qber_300s"] = (
                        ep_records[target_300_idx]["qber"] if target_300_idx < len(ep_records) else np.nan
                    )
                    ep_records[i]["target_skr_300s"] = (
                        ep_records[target_300_idx]["skr_bps"] if target_300_idx < len(ep_records) else np.nan
                    )

                all_records.extend(ep_records)

        df = pd.DataFrame(all_records)

        # Save challenge dataset artifacts
        parquet_path = os.path.join(self.output_dir, "challenge_episodes.parquet")
        csv_path = os.path.join(self.output_dir, "challenge_episodes.csv")
        manifest_path = os.path.join(self.output_dir, "challenge_split_manifest.json")

        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        df.to_csv(csv_path, index=False)

        split_info = {
            "total_runs": len(run_manifest),
            "episodes_per_condition": episodes_per_condition,
            "timesteps_per_episode": timesteps_per_episode,
            "total_records": len(df),
            "parquet_sha256": compute_file_sha256(parquet_path),
            "csv_sha256": compute_file_sha256(csv_path),
            "run_manifest": run_manifest,
            "split_counts": {
                "train": int(np.sum(df["split"] == "train")),
                "calibration": int(np.sum(df["split"] == "calibration")),
                "validation": int(np.sum(df["split"] == "validation")),
                "test": int(np.sum(df["split"] == "test")),
            },
        }

        with open(manifest_path, "w") as f:
            json.dump(split_info, f, indent=2)

        print(f"[CHALLENGE DATASET] Generated {len(df)} records across {len(run_manifest)} runs -> {parquet_path}")
        return df, split_info

    def save_dataset(self, df: pd.DataFrame, base_filename: str) -> Dict[str, str]:
        """Saves dataframe as both .parquet and .csv, returning file paths and hashes."""
        parquet_path = os.path.join(self.output_dir, f"{base_filename}.parquet")
        csv_path = os.path.join(self.output_dir, f"{base_filename}.csv")

        # Save Parquet
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        # Save CSV
        df.to_csv(csv_path, index=False)

        return {
            "parquet": parquet_path,
            "parquet_sha256": compute_file_sha256(parquet_path),
            "csv": csv_path,
            "csv_sha256": compute_file_sha256(csv_path),
            "rows": len(df),
            "columns": list(df.columns),
        }

    def generate_all(self) -> Dict[str, Any]:
        """Executes full generation pipeline and exports manifest."""
        manifest: Dict[str, Any] = {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "governing_standard": "ETSI GS QKD 014 / ITU-T Y.3800",
            "feature_count": TOTAL_FEATURE_COUNT,
            "feature_columns": ALL_FEATURE_COLUMNS,
            "datasets": {},
        }

        print("1/4 Generating Normal Telemetry Dataset...")
        df_normal = self.generate_normal_dataset(n_samples=2000)
        manifest["datasets"]["normal"] = self.save_dataset(df_normal, "normal_dataset")

        print("2/4 Generating Single-Fault Benchmark Dataset...")
        df_single = self.generate_single_fault_dataset(samples_per_class=250)
        manifest["datasets"]["single_fault"] = self.save_dataset(df_single, "single_fault_dataset")

        print("3/4 Generating Mixed-Fault Benchmark Dataset...")
        df_mixed = self.generate_mixed_fault_dataset(n_samples=800)
        manifest["datasets"]["mixed_fault"] = self.save_dataset(df_mixed, "mixed_fault_dataset")

        print("4/4 Generating Unknown-Fault (OOD) Benchmark Dataset...")
        df_unknown = self.generate_unknown_fault_dataset(n_samples=600)
        manifest["datasets"]["unknown_fault"] = self.save_dataset(df_unknown, "unknown_fault_dataset")

        # Save Manifest
        manifest_path = os.path.join(self.output_dir, "dataset_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[SUCCESS] All 4 benchmark datasets and manifest written to {self.output_dir}/")
        return manifest


def generate_all_datasets(output_dir: str = "data") -> Dict[str, Any]:
    """Convenience entry point for generating all datasets."""
    generator = VectorQDatasetGenerator(output_dir=output_dir)
    return generator.generate_all()


if __name__ == "__main__":
    generate_all_datasets()
