"""
Unit & Integration Tests for Challenge Extensions:
1. MultiLabelRootCauseClassifier (Layer 4)
2. DualHorizonQuantileForecaster (Layer 6)
3. FairOptimizationBenchmark (Layer 8)
"""

import os
import numpy as np
import pytest

from root_cause_attribution.lightgbm_classifier import (
    MultiLabelRootCauseClassifier,
    CHALLENGE_FAULT_KEYS,
    CHALLENGE_FAULT_LABELS,
)
from predictive_maintenance.threshold_crossing_forecaster import (
    DualHorizonQuantileForecaster,
    QuantileForecastResult,
)
from validation_framework.fair_optimization_benchmark import (
    FairOptimizationBenchmark,
    PolicyRunMetrics,
)


def test_multilabel_root_cause_classifier():
    np.random.seed(42)
    n_samples = 300
    n_features = 34
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # 3 independent binary targets
    y_dict = {
        "thermal_drift": (np.random.rand(n_samples) > 0.7).astype(np.int32),
        "misalignment": (np.random.rand(n_samples) > 0.7).astype(np.int32),
        "channel_loss": (np.random.rand(n_samples) > 0.8).astype(np.int32),
    }

    clf = MultiLabelRootCauseClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X[:200], {k: v[:200] for k, v in y_dict.items()})
    clf.fit_calibration(X[200:250], {k: v[200:250] for k, v in y_dict.items()})

    assert clf.is_fitted
    assert clf.is_calibrated

    # Test single prediction
    test_x = X[260]
    res = clf.predict_sample(test_x, is_anomaly=False)
    assert res.operational_state in ["Normal", "Diagnosed degradation", "Insufficient evidence"]
    assert isinstance(res.fault_probabilities, dict)
    assert len(res.fault_probabilities) == 3

    # Test vectorized batch prediction
    batch_res = clf.predict_batch(X[250:], is_anomaly_flags=np.zeros(50, dtype=bool))
    assert len(batch_res) == 50


def test_quantile_forecaster_monotonicity_and_urgency():
    np.random.seed(42)
    n_samples = 400
    n_features = 34
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Synthetic targets
    y_60 = np.clip(0.02 + 0.05 * np.random.rand(n_samples), 0.0, 0.50).astype(np.float32)
    y_300 = np.clip(0.02 + 0.08 * np.random.rand(n_samples), 0.0, 0.50).astype(np.float32)

    forecaster = DualHorizonQuantileForecaster(
        quantiles=(0.10, 0.50, 0.90),
        horizons_s=(60, 300),
        warning_threshold=0.08,
        random_state=42,
    )
    forecaster.fit(X[:300], {"target_qber_60s": y_60[:300], "target_qber_300s": y_300[:300]})
    assert forecaster.is_fitted

    # Monotonicity test
    for i in range(10):
        pred = forecaster.predict_sample(X[300 + i], current_qber=0.03)
        assert isinstance(pred, QuantileForecastResult)
        # Check strict monotonicity
        assert pred.qber_60s_p10 <= pred.qber_60s_p50 <= pred.qber_60s_p90
        assert pred.qber_300s_p10 <= pred.qber_300s_p50 <= pred.qber_300s_p90
        assert pred.urgency_level in ["CRITICAL", "WARNING", "STABLE"]


def test_fair_optimization_single_matched_trial():
    bench = FairOptimizationBenchmark(classifier=None)
    res_a, res_b = bench.run_single_matched_trial(
        condition="Thermal Drift",
        fault_name="Temperature Drift",
        fault_intensity=0.75,
        run_id="test_run_01",
        seed=42,
        timesteps=50,
        onset_step=15,
    )

    assert isinstance(res_a, PolicyRunMetrics)
    assert isinstance(res_b, PolicyRunMetrics)
    assert res_a.policy_name == "Default (No-Action)"
    assert res_b.policy_name == "VECTOR-Q Autonomous"
    # Policy B should produce equal or more keys than Policy A
    assert res_b.total_keys_bits >= res_a.total_keys_bits
