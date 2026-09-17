"""
Module B: Confidence Drift Monitor for VECTOR Q
Tracks the live classifier confidence stream using two complementary algorithms:
1. ADWIN (Adaptive Windowing) to detect gradual statistical drift in model certainty.
2. Page-Hinkley Test on (1.0 - confidence) for low-latency detection of abrupt confidence drops.

Triggers candidate retraining when confidence drops indicate physical regime shifts.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class DriftSignal:
    """Telemetry signal emitted by the confidence drift monitor."""
    adwin_triggered: bool
    page_hinkley_triggered: bool
    current_confidence: float
    window_mean_confidence: float
    samples_observed: int
    drift_detected: bool
    recommendation: str


class ADWIN:
    """
    Pure-Python implementation of the ADWIN (Adaptive Windowing) algorithm
    (Bifet & Gavalda, 2007) for detecting distribution shifts without prior knowledge.
    Maintains a variable-length window that shrinks when a statistically significant
    change in mean is detected via Hoeffding bounds.
    """

    def __init__(self, delta: float = 0.002, max_window: int = 1000):
        self.delta = float(delta)
        self.max_window = int(max_window)
        self.window: deque = deque(maxlen=self.max_window)
        self.total_sum: float = 0.0

    @property
    def estimation(self) -> float:
        """Current estimated mean of the window."""
        if not self.window:
            return 1.0
        return self.total_sum / len(self.window)

    def update(self, value: float) -> bool:
        """
        Ingests a new confidence observation.
        Returns True if a statistically significant drift was detected and older
        elements were dropped.
        """
        val = float(value)
        if len(self.window) == self.max_window:
            self.total_sum -= self.window[0]
        self.window.append(val)
        self.total_sum += val

        n = len(self.window)
        if n < 30:
            return False

        drift_detected = False
        # Test subwindow split points
        prefix_sum = 0.0
        for i in range(1, n - 5):
            prefix_sum += self.window[i - 1]
            n0 = i
            n1 = n - i
            if n0 < 10 or n1 < 10:
                continue

            mean0 = prefix_sum / n0
            mean1 = (self.total_sum - prefix_sum) / n1

            # Harmonic mean of sample sizes
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            delta_p = self.delta / n
            epsilon_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / max(1e-12, delta_p)))

            if abs(mean0 - mean1) > epsilon_cut:
                # Distribution shift confirmed: drop older elements up to split point
                drift_detected = True
                for _ in range(n0):
                    dropped = self.window.popleft()
                    self.total_sum -= dropped
                break

        return drift_detected


class PageHinkley:
    """
    Page-Hinkley cumulative sum test for rapid detection of abrupt shifts.
    Specifically monitors x = (1.0 - confidence) to trigger on surges in uncertainty.
    """

    def __init__(self, threshold: float = 50.0, alpha: float = 0.005):
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.cumsum: float = 0.0
        self.min_cumsum: float = 0.0
        self.n_samples: int = 0

    def update(self, confidence: float) -> bool:
        """
        Ingests a new confidence sample.
        Returns True if cumulative drift exceeds threshold.
        """
        self.n_samples += 1
        x = 1.0 - float(confidence)
        self.cumsum += (x - self.alpha)
        self.min_cumsum = min(self.min_cumsum, self.cumsum)

        ph_diff = self.cumsum - self.min_cumsum
        if ph_diff > self.threshold:
            # Trigger and reset
            self.cumsum = 0.0
            self.min_cumsum = 0.0
            return True
        return False


class ModelDriftMonitor:
    """
    Dual-engine Confidence Drift Monitor for live QKD ML classifiers.
    Combines ADWIN (gradual drift) + Page-Hinkley (abrupt drops).

    OPERATIONAL CALIBRATION NOTE:
    The default hyperparameters (adwin_delta=0.002, ph_threshold=25.0, ph_alpha=0.005) represent
    initial baseline thresholds derived from generic statistical drift benchmarks. In live field
    deployments, these parameters must be calibrated against the empirical confidence variance
    measured across a 24-hour nominal link burn-in:
      - For high-variance links (e.g. aerial fiber with diurnal solar swings): widen delta to 0.005
        and raise ph_threshold to 40-50 to avoid false retraining alerts.
      - For ultra-stable links (e.g. deep subterranean SMF-28): tighten delta to 0.001 and ph_threshold
        to 15-20 to catch subtle detector degradation rapidly.
    """

    def __init__(
        self,
        adwin_delta: float = 0.002,
        ph_threshold: float = 25.0,
        ph_alpha: float = 0.005,
    ):
        self.adwin_delta = float(adwin_delta)
        self.ph_threshold = float(ph_threshold)
        self.ph_alpha = float(ph_alpha)

        self.adwin = ADWIN(delta=self.adwin_delta)
        self.page_hinkley = PageHinkley(threshold=self.ph_threshold, alpha=self.ph_alpha)
        self.samples_observed: int = 0
        self.recent_signals: List[DriftSignal] = []

    def calibrate_thresholds(self, baseline_confidence_scores: List[float]) -> Dict[str, float]:
        """
        Site-specific calibration hook: adjusts delta and ph_threshold based on empirical
        nominal variance (std) of calibrated classifier probabilities.
        """
        if len(baseline_confidence_scores) < 30:
            return {"status": "insufficient_data", "adwin_delta": self.adwin_delta, "ph_threshold": self.ph_threshold}

        conf_std = float(np.std(baseline_confidence_scores)) if "np" in globals() else 0.02
        # Scale delta proportionally to empirical noise floor
        new_delta = max(0.0005, min(0.01, conf_std * 0.10))
        new_ph_thresh = max(10.0, min(100.0, 15.0 + conf_std * 500.0))

        self.adwin_delta = new_delta
        self.ph_threshold = new_ph_thresh
        self.adwin = ADWIN(delta=new_delta)
        self.page_hinkley = PageHinkley(threshold=new_ph_thresh, alpha=self.ph_alpha)

        return {
            "status": "calibrated",
            "empirical_conf_std": round(conf_std, 4),
            "calibrated_adwin_delta": round(new_delta, 5),
            "calibrated_ph_threshold": round(new_ph_thresh, 2),
        }

    def update(self, confidence: float) -> DriftSignal:
        """
        Evaluates classifier posterior confidence and updates both drift detectors.
        """
        conf = float(confidence)
        self.samples_observed += 1

        adwin_fired = self.adwin.update(conf)
        ph_fired = self.page_hinkley.update(conf)

        drift_detected = adwin_fired or ph_fired
        if adwin_fired and ph_fired:
            rec = "CRITICAL: Dual ADWIN + Page-Hinkley drift detected. Hardware regime shift likely."
        elif adwin_fired:
            rec = "WARNING: ADWIN detected statistical confidence degradation. Candidate retrain advised."
        elif ph_fired:
            rec = "WARNING: Page-Hinkley detected abrupt confidence collapse. Check hardware calibration."
        else:
            rec = "NOMINAL: Model confidence distribution stable."

        signal = DriftSignal(
            adwin_triggered=adwin_fired,
            page_hinkley_triggered=ph_fired,
            current_confidence=round(conf, 4),
            window_mean_confidence=round(self.adwin.estimation, 4),
            samples_observed=self.samples_observed,
            drift_detected=drift_detected,
            recommendation=rec,
        )
        self.recent_signals.append(signal)
        if len(self.recent_signals) > 100:
            self.recent_signals.pop(0)

        return signal


def should_trigger_retrain(
    labeled_buffer_size: int,
    drift_signal: Optional[DriftSignal],
    days_since_last_train: float,
    min_labels_threshold: int = 50,
    max_retrain_interval_days: float = 30.0,
) -> Tuple[bool, str]:
    """
    Multi-criteria decision rule for automated retraining trigger.
    Evaluates:
      1. Buffer saturation (enough high-confidence human ground truth)
      2. ADWIN or Page-Hinkley drift alarm on confidence stream
      3. Elapsed time floor (scheduled cadence)
    """
    if drift_signal and (drift_signal.adwin_triggered or drift_signal.page_hinkley_triggered):
        if labeled_buffer_size >= 10:
            return True, f"Drift alarm fired with {labeled_buffer_size} new verified labels available"
        return False, f"Drift alarm fired but insufficient labels in buffer ({labeled_buffer_size}/10 required)"

    if labeled_buffer_size >= min_labels_threshold:
        return True, f"Feedback buffer reached threshold ({labeled_buffer_size} >= {min_labels_threshold})"

    if days_since_last_train >= max_retrain_interval_days and labeled_buffer_size >= 15:
        return True, f"Scheduled retraining due ({days_since_last_train:.1f} days elapsed with {labeled_buffer_size} labels)"

    return False, "Retraining conditions not met"
