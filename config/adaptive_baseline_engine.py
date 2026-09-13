"""
Phase 15: Adaptive Baseline Engine
Derives portable statistical envelopes (mean, variance, 3-sigma bounds, EWMA) per link during healthy calibration.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ChannelBaselineEnvelope:
    """Represents the learned statistical operating manifold of a deployed QKD link."""
    metric_name: str
    mean: float
    variance: float
    std_dev: float
    ewma_value: float
    lower_3sigma: float
    upper_3sigma: float
    is_calibrated: bool


class AdaptiveBaselineEngine:
    """
    Learns link-specific normal operating bounds dynamically.
    Replaces static thresholds with calibrated statistical boundaries.
    """

    def __init__(self, alpha_ewma: float = 0.05, warmup_samples: int = 30):
        self.alpha_ewma = alpha_ewma
        self.warmup_samples = warmup_samples
        self.sample_count = 0
        
        # Tracked continuous metrics
        self.metrics = ["qber", "skr_bps", "raw_counts_hz", "dark_counts_hz", "visibility", "temperature_celsius", "timing_jitter_ps"]
        self.history: Dict[str, list] = {m: [] for m in self.metrics}
        self.ewma: Dict[str, float] = {}

    def update(self, sample_dict: Dict[str, float], is_healthy: bool = True) -> Dict[str, ChannelBaselineEnvelope]:
        """
        Updates the adaptive baseline with a new telemetry sample.
        Updates running statistics if the sample belongs to a healthy calibration period.
        """
        self.sample_count += 1
        envelopes = {}
        
        for m in self.metrics:
            val = float(sample_dict.get(m, 0.0))
            if is_healthy:
                self.history[m].append(val)
                if len(self.history[m]) > 200:
                    self.history[m].pop(0)
                    
                # Update EWMA
                if m not in self.ewma:
                    self.ewma[m] = val
                else:
                    self.ewma[m] = self.alpha_ewma * val + (1.0 - self.alpha_ewma) * self.ewma[m]
                    
            arr = np.array(self.history[m]) if len(self.history[m]) > 0 else np.array([val])
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr)) if len(arr) > 1 else 1e-4
            var_val = float(np.var(arr)) if len(arr) > 1 else 1e-8
            
            ewma_val = self.ewma.get(m, val)
            is_calib = len(self.history[m]) >= self.warmup_samples
            
            envelopes[m] = ChannelBaselineEnvelope(
                metric_name=m,
                mean=mean_val,
                variance=var_val,
                std_dev=std_val,
                ewma_value=ewma_val,
                lower_3sigma=max(0.0, mean_val - 3.0 * std_val),
                upper_3sigma=mean_val + 3.0 * std_val,
                is_calibrated=is_calib,
            )
            
        return envelopes
