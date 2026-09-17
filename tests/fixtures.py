"""
Test Fixtures for VECTOR Q Advanced Features

Provides reusable test data and mock objects.

Author: VECTOR Q Development Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict


@pytest.fixture
def sample_telemetry_features() -> Dict[str, float]:
    """Generate sample telemetry feature dictionary."""
    return {
        'qber': 0.06,
        'skr_bps': 1000,
        'raw_counts_hz': 15000,
        'dark_counts_hz': 3000,
        'singles_alice_hz': 20000,
        'singles_bob_hz': 20000,
        'visibility': 0.98,
        'temperature_celsius': 25.0,
        'timing_jitter_ps': 100.0,
        'qber_slope_25': 0.0001,
        'qber_acceleration_25': 0.00001,
        'temp_slope_25': 0.1,
        'dark_counts_slope_25': 10.0
    }


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    """Generate synthetic training data for models."""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'duration': np.random.exponential(150, n_samples),
        'event': np.random.binomial(1, 0.7, n_samples),
        'qber_slope': np.random.normal(0.0002, 0.0001, n_samples),
        'qber_accel': np.random.normal(0.00001, 0.000005, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'dark_counts': np.random.normal(3000, 500, n_samples),
        'visibility': np.random.uniform(0.95, 0.99, n_samples),
        'qber': np.random.uniform(0.04, 0.12, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_network_topology():
    """Generate mock network topology for multi-link tests."""
    class MockTopology:
        def __init__(self):
            self.links = {
                "link_0": None,
                "link_1": None,
                "link_2": None,
                "link_3": None,
                "link_4": None
            }
            # Mesh topology
            self.adjacency_matrix = np.array([
                [0, 1, 1, 0, 0],
                [1, 0, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [0, 1, 1, 0, 1],
                [0, 0, 1, 1, 0]
            ])
            self.nodes = list(self.links.keys())
    
    return MockTopology()


@pytest.fixture
def mock_hardware_telemetry():
    """Generate mock hardware telemetry sample."""
    from hardware_interface.base_hardware_interface import HardwareTelemetry
    import time
    
    return HardwareTelemetry(
        timestamp=time.time(),
        link_id="test_link",
        qber=0.055,
        skr=950.0,
        raw_key_rate=1425.0,
        signal_counts=14500,
        dark_counts=3200,
        singles_alice=19500,
        singles_bob=19500,
        temperature=26.0,
        optical_power=-46.0,
        visibility=0.975,
        bit_error_rate=0.055,
        status="operational",
        alarm_state="normal"
    )
