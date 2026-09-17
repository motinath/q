"""
Phase 7: Statistical & Multi-Scale Temporal Feature Layer (Rolling W=25)
Extracts statistical (mean, std, variance, IQR), temporal (slope, acceleration), physical domain ratios,
and cross-channel correlations (Corr(T, QBER), Corr(V, QBER), Corr(Counts, Loss)).
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
import collections
from typing import List, Dict, Any, Optional, Tuple
from physics_engine.quantum_telemetry_emulator import QuantumTelemetrySample


FEATURE_COLUMN_NAMES = [
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
    
    # 3. Statistical Features (Rolling Window W=25)
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
    
    # 4. Temporal Derivatives & Dynamics (Slope & Acceleration)
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


class TelemetryFeatureExtractor:
    """
    Maintains a ring buffer of historical telemetry snapshots and computes
    statistical distributions, derivatives, and cross-channel correlations.
    """

    def __init__(self, max_buffer_size: int = 35, window_size: Optional[int] = None):
        if window_size is not None:
            max_buffer_size = max(max_buffer_size, window_size + 10)
        self.max_buffer_size = max_buffer_size
        self.history: collections.deque = collections.deque(maxlen=max_buffer_size)

    def reset(self) -> None:
        """Clears buffer state."""
        self.history.clear()

    def add_sample(self, sample: QuantumTelemetrySample) -> None:
        """Appends a new sample to the internal ring buffer."""
        self.history.append(sample)

    @staticmethod
    def _compute_slope(values: np.ndarray, timestamps: np.ndarray) -> float:
        """Computes linear least-squares slope (dy/dt)."""
        n = len(values)
        if n < 2:
            return 0.0
        dt = timestamps - timestamps[0]
        if np.all(dt == 0):
            return 0.0
        dt_mean = np.mean(dt)
        val_mean = np.mean(values)
        denom = np.sum((dt - dt_mean) ** 2)
        if denom == 0:
            return 0.0
        return float(np.sum((dt - dt_mean) * (values - val_mean)) / denom)

    @staticmethod
    def _compute_slope_with_se(values: np.ndarray, timestamps: np.ndarray) -> Tuple[float, float]:
        """
        P2.5: Computes linear least-squares slope AND its standard error.

        Returns:
            (slope, slope_standard_error)
        """
        n = len(values)
        if n < 3:
            return 0.0, 0.0002  # default SE when window too small
        dt = timestamps - timestamps[0]
        if np.all(dt == 0):
            return 0.0, 0.0002
        dt_mean = np.mean(dt)
        val_mean = np.mean(values)
        ss_xx = np.sum((dt - dt_mean) ** 2)
        if ss_xx < 1e-20:
            return 0.0, 0.0002
        slope = float(np.sum((dt - dt_mean) * (values - val_mean)) / ss_xx)
        residuals = values - (val_mean + slope * (dt - dt_mean))
        mse = float(np.sum(residuals ** 2) / max(1, n - 2))
        se = float(np.sqrt(mse / ss_xx))
        return slope, se

    @staticmethod
    def _compute_acceleration(values: np.ndarray, timestamps: np.ndarray) -> float:
        """Computes second derivative (acceleration d^2y/dt^2)."""
        n = len(values)
        if n < 3:
            return 0.0
        dt = timestamps - timestamps[0]
        if len(np.unique(dt)) < 3:
            return 0.0
        try:
            poly = np.polyfit(dt, values, deg=2)
            return float(2.0 * poly[0])  # y = a*t^2 + b*t + c -> y'' = 2a
        except Exception:
            return 0.0

    @staticmethod
    def _compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Computes Pearson correlation coefficient between two channels."""
        if len(x) < 3 or np.std(x) < 1e-6 or np.std(y) < 1e-6:
            return 0.0
        try:
            r = np.corrcoef(x, y)[0, 1]
            return 0.0 if np.isnan(r) else float(np.clip(r, -1.0, 1.0))
        except Exception:
            return 0.0

    def extract_features(self, current_sample: Optional[QuantumTelemetrySample] = None) -> Dict[str, float]:
        """
        Extracts full engineered feature dictionary for the current telemetry sample.
        """
        if current_sample is not None:
            self.add_sample(current_sample)
            
        if len(self.history) == 0:
            raise ValueError("Cannot extract features from empty history buffer.")
            
        latest = self.history[-1]
        
        # Physical domain ratios
        cnt = max(1.0, latest.raw_counts_hz)
        dcr = max(1.0, latest.dark_counts_hz)
        vis = float(np.clip(latest.visibility, 0.01, 0.9999))
        
        count_to_dark = cnt / dcr
        snr = max(0.0, (cnt - dcr)) / dcr
        optical_error_ratio = (1.0 - vis) / 2.0
        qber_vis_mismatch = latest.qber - optical_error_ratio
        skr_to_qber = latest.skr_bps / max(1e-4, latest.qber)
        
        # History arrays
        hist_list = list(self.history)
        w = min(len(hist_list), 25)
        hist_slice = hist_list[-w:]
        
        ts_arr = np.array([s.timestamp for s in hist_slice])
        qber_arr = np.array([s.qber for s in hist_slice])
        cnt_arr = np.array([s.raw_counts_hz for s in hist_slice])
        dcr_arr = np.array([s.dark_counts_hz for s in hist_slice])
        vis_arr = np.array([s.visibility for s in hist_slice])
        temp_arr = np.array([s.temperature_celsius for s in hist_slice])
        jit_arr = np.array([s.timing_jitter_ps for s in hist_slice])
        loss_arr = np.array([s.channel_attenuation_db for s in hist_slice])
        hum_arr = np.array([getattr(s, "humidity_relative_pct", 45.0) for s in hist_slice])
        strain_arr = np.array([getattr(s, "fiber_strain_ue", 15.0) for s in hist_slice])
        
        # Statistical moments (IQR, Mean, Std, Var)
        q75, q25 = np.percentile(qber_arr, [75, 25]) if len(qber_arr) >= 4 else (qber_arr[-1], qber_arr[-1])
        qber_iqr = q75 - q25
        
        features: Dict[str, float] = {
            # 1. Raw Observables
            "qber": float(latest.qber),
            "skr_bps": float(latest.skr_bps),
            "raw_counts_hz": float(latest.raw_counts_hz),
            "dark_counts_hz": float(latest.dark_counts_hz),
            "visibility": float(latest.visibility),
            "temperature_celsius": float(latest.temperature_celsius),
            "timing_jitter_ps": float(latest.timing_jitter_ps),
            "channel_attenuation_db": float(latest.channel_attenuation_db),
            "channel_loss_db": float(latest.channel_attenuation_db),

            # Environmental Observables
            "humidity_relative_pct": float(getattr(latest, "humidity_relative_pct", 45.0)),
            "vibration_g": float(getattr(latest, "vibration_g", 0.02)),
            "supply_voltage_v": float(getattr(latest, "supply_voltage_v", 3.30)),
            "fiber_strain_ue": float(getattr(latest, "fiber_strain_ue", 15.0)),

            # Maintenance Features
            "device_operating_hours": float(getattr(latest, "device_operating_hours", 1200.0)),
            "hours_since_calibration": float(getattr(latest, "hours_since_calibration", 48.0)),
            "trap_aging_index": float(getattr(latest, "trap_aging_index", 0.05)),
            "maintenance_event_count": float(getattr(latest, "maintenance_event_count", 1)),
            
            # 2. Domain Ratios
            "count_to_dark_ratio": float(count_to_dark),
            "signal_to_noise_ratio": float(snr),
            "optical_error_ratio": float(optical_error_ratio),
            "qber_to_visibility_mismatch": float(qber_vis_mismatch),
            "skr_to_qber_ratio": float(skr_to_qber),
            
            # 3. Statistical (W=25)
            "qber_roll_mean_25": float(np.mean(qber_arr)),
            "qber_roll_std_25": float(np.std(qber_arr)) if w > 1 else 0.0,
            "qber_roll_var_25": float(np.var(qber_arr)) if w > 1 else 0.0,
            "qber_iqr_25": float(qber_iqr),
            "raw_counts_roll_mean_25": float(np.mean(cnt_arr)),
            "raw_counts_roll_std_25": float(np.std(cnt_arr)) if w > 1 else 0.0,
            "dark_counts_roll_mean_25": float(np.mean(dcr_arr)),
            "dark_counts_roll_std_25": float(np.std(dcr_arr)) if w > 1 else 0.0,
            "visibility_roll_mean_25": float(np.mean(vis_arr)),
            "visibility_roll_std_25": float(np.std(vis_arr)) if w > 1 else 0.0,
            "temp_roll_mean_25": float(np.mean(temp_arr)),
            "humidity_roll_mean_25": float(np.mean(hum_arr)),
            "jitter_roll_mean_25": float(np.mean(jit_arr)),
            
            # 4. Temporal Derivatives
            # P2.5: qber slope computed with SE for PTCT confidence interval
            "qber_slope_25": self._compute_slope(qber_arr, ts_arr),
            "qber_slope_25_se": self._compute_slope_with_se(qber_arr, ts_arr)[1],  # SE sidecar
            "qber_acceleration_25": self._compute_acceleration(qber_arr, ts_arr),
            "raw_counts_slope_25": self._compute_slope(cnt_arr, ts_arr),
            "dark_counts_slope_25": self._compute_slope(dcr_arr, ts_arr),
            "visibility_slope_25": self._compute_slope(vis_arr, ts_arr),
            "temp_slope_25": self._compute_slope(temp_arr, ts_arr),
            
            # 5. Cross-Channel Correlations
            "corr_temp_qber": self._compute_correlation(temp_arr, qber_arr),
            "corr_vis_qber": self._compute_correlation(vis_arr, qber_arr),
            "corr_counts_loss": self._compute_correlation(cnt_arr, loss_arr),
            "corr_humidity_loss": self._compute_correlation(hum_arr, loss_arr),
            "corr_strain_vis": self._compute_correlation(strain_arr, vis_arr),
        }
        
        return features

    def get_feature_vector(self, current_sample: Optional[QuantumTelemetrySample] = None) -> np.ndarray:
        """Returns ordered feature vector as NumPy array."""
        feat_dict = self.extract_features(current_sample)
        return np.array([feat_dict[col] for col in FEATURE_COLUMN_NAMES], dtype=np.float64)
