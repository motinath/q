"""
P4.5 — Model Drift Monitor for Isolation Forest Anomaly Detector

Tracks the rolling distribution of anomaly scores on "Normal"-classified samples
and detects statistically significant distributional shifts using KL divergence.
When model drift is detected (distribution shift beyond threshold), the monitor
raises a flag indicating that the model should be retrained on recent field data.

This is essential for long-running field deployments where the nominal baseline
may drift over time due to component aging, fiber plant changes, or environmental
baseline shifts.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import scipy.stats


@dataclass
class DriftDetectionResult:
    """Result of a model drift check."""
    is_drift_detected: bool
    kl_divergence: float
    drift_threshold: float
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    n_reference_samples: int
    n_current_samples: int
    recommendation: str


class ModelDriftMonitor:
    """
    Monitors the distribution of IsolationForest anomaly scores on Normal-classified
    samples to detect model drift.

    Strategy:
      1. Accumulate a reference distribution from the first N "Normal" samples
         (default 200) immediately after training/deployment.
      2. Maintain a sliding window of the most recent M "Normal" samples (default 200).
      3. Periodically compute KL-divergence between reference and current distributions.
      4. If KL-div > threshold (default 0.15), flag drift and recommend retraining.

    KL divergence is estimated via histogram binning; both distributions are
    discretized into 20 bins over the [0, 1] anomaly score range.
    """

    def __init__(
        self,
        reference_window_size: int = 200,
        current_window_size: int = 200,
        kl_threshold: float = 0.15,
        n_bins: int = 20,
    ):
        """
        Args:
            reference_window_size: Number of initial Normal samples to collect as baseline.
            current_window_size:   Size of the rolling window for current distribution.
            kl_threshold:          KL divergence above which drift is flagged.
            n_bins:                Number of histogram bins for KL-div estimation.
        """
        self.reference_window_size = reference_window_size
        self.current_window_size = current_window_size
        self.kl_threshold = kl_threshold
        self.n_bins = n_bins

        # Reference distribution (frozen after collecting reference_window_size samples)
        self._reference_scores: List[float] = []
        self._reference_frozen: bool = False

        # Current rolling window
        self._current_scores: deque = deque(maxlen=current_window_size)

        # Cached reference histogram for efficient repeated KL-div checks
        self._reference_hist: Optional[np.ndarray] = None
        self._bin_edges: Optional[np.ndarray] = None

    def add_sample(self, anomaly_score: float, predicted_class: str) -> None:
        """
        Logs an anomaly score for a single sample.

        Args:
            anomaly_score:    The IsolationForest anomaly score [0, 1].
            predicted_class:  The attributed root-cause class from LightGBM.

        Only samples predicted as "Normal" are used for drift monitoring.
        """
        if predicted_class != "Normal":
            return  # Drift monitor only tracks the Normal baseline

        # Build reference distribution first
        if not self._reference_frozen:
            self._reference_scores.append(anomaly_score)
            if len(self._reference_scores) >= self.reference_window_size:
                self._freeze_reference()
            return

        # Reference is frozen; add to current rolling window
        self._current_scores.append(anomaly_score)

    def _freeze_reference(self) -> None:
        """
        Freezes the reference distribution and computes the reference histogram.
        Called once after collecting reference_window_size Normal samples.
        """
        if len(self._reference_scores) < 10:
            return  # insufficient data

        ref_arr = np.array(self._reference_scores, dtype=np.float64)
        self._reference_hist, self._bin_edges = np.histogram(
            ref_arr, bins=self.n_bins, range=(0.0, 1.0), density=True
        )
        # Add small epsilon to avoid log(0) in KL-div
        self._reference_hist = self._reference_hist + 1e-10
        self._reference_hist = self._reference_hist / self._reference_hist.sum()
        self._reference_frozen = True

    def check_drift(self) -> Optional[DriftDetectionResult]:
        """
        Computes KL-divergence between the reference and current distributions.

        Returns None if insufficient data; otherwise returns a DriftDetectionResult
        indicating whether statistically significant drift was detected.
        """
        if not self._reference_frozen:
            return None  # Reference not ready

        if len(self._current_scores) < 50:
            return None  # Insufficient current data

        cur_arr = np.array(list(self._current_scores), dtype=np.float64)
        cur_hist, _ = np.histogram(
            cur_arr, bins=self.n_bins, range=(0.0, 1.0), density=True
        )
        cur_hist = cur_hist + 1e-10
        cur_hist = cur_hist / cur_hist.sum()

        # KL divergence: D_KL(P_ref || P_cur) = sum(P_ref * log(P_ref / P_cur))
        kl_div = float(np.sum(self._reference_hist * np.log(self._reference_hist / cur_hist)))

        ref_mean = float(np.mean(self._reference_scores))
        ref_std = float(np.std(self._reference_scores))
        cur_mean = float(np.mean(cur_arr))
        cur_std = float(np.std(cur_arr))

        is_drift = kl_div > self.kl_threshold

        if is_drift:
            recommendation = (
                f"Model drift detected (KL-div = {kl_div:.4f} > {self.kl_threshold:.4f}). "
                f"The distribution of Normal-class anomaly scores has shifted significantly "
                f"(reference mean={ref_mean:.3f}±{ref_std:.3f}, "
                f"current mean={cur_mean:.3f}±{cur_std:.3f}). "
                f"Recommend retraining the IsolationForest and LightGBM models on recent "
                f"field telemetry to recalibrate the baseline manifold."
            )
        else:
            recommendation = (
                f"No significant drift detected (KL-div = {kl_div:.4f} < {self.kl_threshold:.4f}). "
                f"Model baseline remains stable."
            )

        return DriftDetectionResult(
            is_drift_detected=is_drift,
            kl_divergence=kl_div,
            drift_threshold=self.kl_threshold,
            reference_mean=ref_mean,
            current_mean=cur_mean,
            reference_std=ref_std,
            current_std=cur_std,
            n_reference_samples=len(self._reference_scores),
            n_current_samples=len(self._current_scores),
            recommendation=recommendation,
        )

    def reset(self) -> None:
        """Clears all accumulated data and unfreezes the reference distribution."""
        self._reference_scores.clear()
        self._current_scores.clear()
        self._reference_frozen = False
        self._reference_hist = None
        self._bin_edges = None
