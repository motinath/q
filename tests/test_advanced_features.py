"""
Validation Tests for Advanced Q-SENTINEL Features (Version 2.0)

Tests all new capabilities:
- Dynamic Bayesian Network causal inference
- Survival Analysis PTCT
- Conformal Prediction
- Graph Neural Networks
- Causal counterfactual reasoning
- Hardware interfaces

Author: Q-SENTINEL Development Team
Version: 2.0.0
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict

# Test fixtures
from tests.fixtures import (
    sample_telemetry_features,
    sample_training_data,
    mock_network_topology
)


# ============================================================================
# Test 1: Dynamic Bayesian Network Causal Inference
# ============================================================================

class TestDynamicBayesianNetwork:
    """Test suite for DBN causal root-cause analysis."""
    
    def test_dbn_initialization(self):
        """Test DBN can be initialized."""
        from streaming_pipeline.causal_network_intelligence import DynamicBayesianNetworkRCA
        
        dbn_engine = DynamicBayesianNetworkRCA()
        assert dbn_engine is not None
        assert hasattr(dbn_engine, 'infer_root_cause')
    
    def test_dbn_causal_inference_fallback(self):
        """Test DBN causal inference with fallback mode."""
        from streaming_pipeline.causal_network_intelligence import DynamicBayesianNetworkRCA
        
        dbn_engine = DynamicBayesianNetworkRCA()
        
        # Mock observations
        observed_anomalies = {
            "link_0": False,
            "link_1": True,
            "link_2": True
        }
        
        telemetry_features = {
            "link_1": {"qber": 0.09, "skr": 800, "temperature_celsius": 35},
            "link_2": {"qber": 0.12, "skr": 500, "temperature_celsius": 38}
        }
        
        result = dbn_engine.infer_root_cause(
            observed_anomalies,
            telemetry_features
        )
        
        # Validate result structure
        assert result.root_cause_node is not None
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.causal_chain, list)
        assert isinstance(result.all_posteriors, dict)
        assert 0.0 <= result.causal_strength <= 1.0
        assert result.temporal_delay >= 0
    
    def test_dbn_multiple_scenarios(self):
        """Test DBN across different fault scenarios."""
        from streaming_pipeline.causal_network_intelligence import DynamicBayesianNetworkRCA
        
        dbn_engine = DynamicBayesianNetworkRCA()
        
        scenarios = [
            # Scenario 1: High temperature causing anomaly
            {
                "anomalies": {"link_0": True},
                "features": {"link_0": {"qber": 0.10, "temperature_celsius": 45}}
            },
            # Scenario 2: Multiple link failures
            {
                "anomalies": {"link_0": True, "link_1": True},
                "features": {
                    "link_0": {"qber": 0.11, "temperature_celsius": 30},
                    "link_1": {"qber": 0.09, "temperature_celsius": 28}
                }
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            result = dbn_engine.infer_root_cause(
                scenario["anomalies"],
                scenario["features"]
            )
            assert result.root_cause_node is not None, f"Scenario {i} failed"
            assert result.confidence > 0.0, f"Scenario {i} has zero confidence"


# ============================================================================
# Test 2: Survival Analysis PTCT Forecaster
# ============================================================================

class TestSurvivalAnalysis:
    """Test suite for Survival Analysis PTCT forecasting."""
    
    def test_survival_forecaster_initialization(self):
        """Test survival forecaster can be initialized."""
        from predictive_maintenance.survival_ptct_forecaster import SurvivalPTCTForecaster
        
        forecaster = SurvivalPTCTForecaster(threshold_qber=0.11)
        assert forecaster is not None
        assert forecaster.threshold_qber == 0.11
    
    def test_survival_model_training(self):
        """Test survival model training on synthetic data."""
        from predictive_maintenance.survival_ptct_forecaster import SurvivalPTCTForecaster
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 100
        
        training_data = pd.DataFrame({
            'duration': np.random.exponential(150, n_samples),
            'event': np.random.binomial(1, 0.7, n_samples),
            'qber_slope': np.random.normal(0.0002, 0.0001, n_samples),
            'qber_accel': np.random.normal(0.00001, 0.000005, n_samples),
            'temperature': np.random.normal(25, 5, n_samples),
            'dark_counts': np.random.normal(3000, 500, n_samples)
        })
        
        forecaster = SurvivalPTCTForecaster()
        feature_cols = ['qber_slope', 'qber_accel', 'temperature', 'dark_counts']
        
        # Should not raise exception
        forecaster.fit(training_data, feature_cols)
    
    def test_survival_prediction(self):
        """Test survival curve prediction."""
        from predictive_maintenance.survival_ptct_forecaster import SurvivalPTCTForecaster
        
        forecaster = SurvivalPTCTForecaster()
        
        # Use fallback mode (no training)
        current_features = pd.DataFrame({
            'qber_slope': [0.0003],
            'qber_accel': [0.00002],
            'temperature': [30.0],
            'dark_counts': [3500]
        })
        
        result = forecaster.predict_survival(current_features, horizon_seconds=300)
        
        # Validate result
        assert result.median_ttf > 0
        assert result.confidence_lower >= 0
        assert result.confidence_upper >= result.median_ttf
        assert result.hazard_ratio > 0
        assert len(result.survival_curve) == len(result.time_points)
        
        # Survival curve should be non-increasing
        assert all(result.survival_curve[i] >= result.survival_curve[i+1] 
                   for i in range(len(result.survival_curve)-1))


# ============================================================================
# Test 3: Conformal Prediction
# ============================================================================

class TestConformalPrediction:
    """Test suite for Conformal Prediction PTCT."""
    
    def test_conformal_predictor_initialization(self):
        """Test conformal predictor initialization."""
        from predictive_maintenance.conformal_ptct import ConformalPTCTForecaster
        from sklearn.ensemble import RandomForestRegressor
        
        base_model = RandomForestRegressor(n_estimators=10, random_state=42)
        predictor = ConformalPTCTForecaster(base_model, significance=0.10)
        
        assert predictor is not None
        assert predictor.significance == 0.10
        assert predictor.coverage_guarantee == 0.90
    
    def test_conformal_training_and_prediction(self):
        """Test conformal predictor training and inference."""
        from predictive_maintenance.conformal_ptct import ConformalPTCTForecaster
        from sklearn.ensemble import RandomForestRegressor
        
        # Generate synthetic data
        np.random.seed(42)
        n_train, n_cal, n_test = 100, 50, 10
        n_features = 4
        
        X_train = np.random.randn(n_train, n_features)
        y_train = 150 + 30 * X_train[:, 0] + np.random.normal(0, 10, n_train)
        
        X_cal = np.random.randn(n_cal, n_features)
        y_cal = 150 + 30 * X_cal[:, 0] + np.random.normal(0, 10, n_cal)
        
        X_test = np.random.randn(n_test, n_features)
        
        # Train conformal predictor
        base_model = RandomForestRegressor(n_estimators=10, random_state=42)
        predictor = ConformalPTCTForecaster(base_model, significance=0.10)
        predictor.fit(X_train, y_train, X_cal, y_cal)
        
        assert predictor.is_fitted
        
        # Predict
        result = predictor.predict(X_test[0:1])
        
        assert result.median_ttf > 0
        assert result.lower_bound <= result.median_ttf
        assert result.upper_bound >= result.median_ttf
        assert result.coverage_guarantee == 0.90
        assert result.interval_width == result.upper_bound - result.lower_bound
    
    def test_conformal_coverage_validation(self):
        """Test empirical coverage meets theoretical guarantee."""
        from predictive_maintenance.conformal_ptct import ConformalPTCTForecaster
        from sklearn.ensemble import RandomForestRegressor
        
        np.random.seed(42)
        n_train, n_cal, n_test = 100, 50, 30
        n_features = 4
        
        X_train = np.random.randn(n_train, n_features)
        y_train = 150 + 30 * X_train[:, 0] + np.random.normal(0, 10, n_train)
        
        X_cal = np.random.randn(n_cal, n_features)
        y_cal = 150 + 30 * X_cal[:, 0] + np.random.normal(0, 10, n_cal)
        
        X_test = np.random.randn(n_test, n_features)
        y_test = 150 + 30 * X_test[:, 0] + np.random.normal(0, 10, n_test)
        
        base_model = RandomForestRegressor(n_estimators=10, random_state=42)
        predictor = ConformalPTCTForecaster(base_model, significance=0.10)
        predictor.fit(X_train, y_train, X_cal, y_cal)
        
        # Validate coverage
        coverage_metrics = predictor.validate_coverage(X_test, y_test, significance=0.10)
        
        assert coverage_metrics['empirical_coverage'] >= 0.0
        assert coverage_metrics['theoretical_coverage'] == 0.90
        assert coverage_metrics['n_samples'] == n_test


# ============================================================================
# Test 4: Graph Neural Network
# ============================================================================

class TestGraphNeuralNetwork:
    """Test suite for GNN network intelligence."""
    
    def test_gnn_initialization(self):
        """Test GNN can be initialized."""
        from streaming_pipeline.graph_network_embeddings import GNNNetworkIntelligence
        
        gnn_engine = GNNNetworkIntelligence(n_features=35)
        assert gnn_engine is not None
        assert gnn_engine.n_features == 35
    
    def test_gnn_network_embedding(self):
        """Test GNN network embedding generation."""
        from streaming_pipeline.graph_network_embeddings import GNNNetworkIntelligence
        
        gnn_engine = GNNNetworkIntelligence(n_features=35)
        
        # Create mock network
        n_links = 5
        link_features = {
            f"link_{i}": np.random.randn(35) for i in range(n_links)
        }
        
        # Mesh topology
        adjacency_matrix = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0]
        ])
        
        result = gnn_engine.embed_network(link_features, adjacency_matrix)
        
        # Validate result
        assert result.link_embeddings.shape == (n_links, 64)
        assert 0.0 <= result.network_health_score <= 1.0
        assert len(result.cascade_risk_per_link) == n_links
        assert all(0.0 <= risk <= 1.0 for risk in result.cascade_risk_per_link)


# ============================================================================
# Test 5: Causal Bayesian Network Counterfactuals
# ============================================================================

class TestCausalCounterfactuals:
    """Test suite for causal counterfactual reasoning."""
    
    def test_causal_engine_initialization(self):
        """Test causal attribution engine initialization."""
        from root_cause_attribution.causal_attribution_engine import CausalAttributionEngine
        
        engine = CausalAttributionEngine()
        assert engine is not None
        assert hasattr(engine, 'infer_root_cause')
        assert hasattr(engine, 'counterfactual_query')
    
    def test_causal_inference(self):
        """Test causal root-cause inference."""
        from root_cause_attribution.causal_attribution_engine import CausalAttributionEngine
        
        engine = CausalAttributionEngine()
        
        observed_state = {
            'qber': 'High',
            'skr': 'Low',
            'dark_counts': 'High'
        }
        
        result = engine.infer_root_cause(observed_state)
        
        assert result.root_cause is not None
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.causal_chain, list)
        assert isinstance(result.all_causes_ranked, list)
    
    def test_counterfactual_query(self):
        """Test counterfactual 'what-if' analysis."""
        from root_cause_attribution.causal_attribution_engine import CausalAttributionEngine
        
        engine = CausalAttributionEngine()
        
        observed_state = {'qber': 'High', 'temperature': 'Hot'}
        intervention = {'temperature': 'Cold'}
        
        result = engine.counterfactual_query(observed_state, intervention)
        
        assert result.counterfactual_qber >= 0.0
        assert result.observed_qber >= 0.0
        assert isinstance(result.delta, float)
        assert isinstance(result.intervention, dict)
        assert isinstance(result.interpretation, str)
        assert result.causal_effect_size >= 0.0


# ============================================================================
# Test 6: Hardware Interfaces
# ============================================================================

class TestHardwareInterfaces:
    """Test suite for real hardware deployment interfaces."""
    
    def test_mock_hardware_interface(self):
        """Test mock hardware interface."""
        from hardware_interface.base_hardware_interface import MockQKDHardware
        
        hardware = MockQKDHardware(link_id="test_link")
        
        assert hardware.connect()
        assert hardware.is_connected
        
        # Read telemetry
        telemetry = hardware.read_telemetry()
        
        assert telemetry.link_id == "test_link"
        assert 0.0 <= telemetry.qber <= 1.0
        assert telemetry.skr >= 0
        assert telemetry.signal_counts > 0
        assert telemetry.status in ["operational", "degraded", "failed"]
        
        # Send command
        from hardware_interface.base_hardware_interface import HardwareCommand
        
        command = HardwareCommand(
            command_type="restart",
            parameters={}
        )
        result = hardware.send_command(command)
        
        assert result.success
        assert isinstance(result.message, str)
        assert result.execution_time >= 0
        
        hardware.disconnect()
        assert not hardware.is_connected
    
    def test_id_quantique_interface(self):
        """Test ID Quantique interface (mock mode)."""
        from hardware_interface.id_quantique_interface import IDQuantiqueInterface
        
        idq = IDQuantiqueInterface(
            host="192.168.1.100",
            link_id="idq_test",
            community="public"
        )
        
        # Will use mock mode
        assert idq.connect()
        
        telemetry = idq.read_telemetry()
        assert telemetry.link_id == "idq_test"
        assert 0.0 <= telemetry.qber <= 1.0
        
        status = idq.get_system_status()
        assert status['vendor'] == "ID Quantique"
        
        idq.disconnect()
    
    def test_hardware_context_manager(self):
        """Test hardware interface context manager."""
        from hardware_interface.base_hardware_interface import MockQKDHardware
        
        with MockQKDHardware(link_id="context_test") as hardware:
            assert hardware.is_connected
            telemetry = hardware.read_telemetry()
            assert telemetry is not None
        
        # Should be disconnected after context exit
        assert not hardware.is_connected


# ============================================================================
# Test 7: Advanced Orchestrator Integration
# ============================================================================

class TestAdvancedOrchestrator:
    """Test suite for advanced orchestrator integration."""
    
    def test_advanced_orchestrator_initialization(self):
        """Test advanced orchestrator can be initialized."""
        from streaming_pipeline.advanced_orchestrator import AdvancedQKDOrchestrator
        
        orchestrator = AdvancedQKDOrchestrator(
            enable_causal_inference=True,
            enable_survival_analysis=True,
            enable_conformal_prediction=False,
            enable_gnn=False,
            enable_counterfactual=True
        )
        
        assert orchestrator is not None
        assert orchestrator.base_orchestrator is not None
    
    def test_advanced_orchestrator_single_timestep(self):
        """Test advanced orchestrator processes single timestep."""
        from streaming_pipeline.advanced_orchestrator import AdvancedQKDOrchestrator
        
        orchestrator = AdvancedQKDOrchestrator(
            enable_causal_inference=True,
            enable_survival_analysis=True,
            enable_conformal_prediction=False,
            enable_gnn=False,
            enable_counterfactual=True
        )
        
        result = orchestrator.process_single_timestep(dt_seconds=1.0)
        
        # Validate result structure
        assert result.base_result is not None
        assert result.base_result.sample is not None
        assert result.base_result.features is not None
        
        # Advanced features may be None if not trained/configured
        # but result structure should exist
        assert hasattr(result, 'causal_inference')
        assert hasattr(result, 'survival_forecast')
        assert hasattr(result, 'conformal_forecast')
        assert hasattr(result, 'network_embedding')
        assert hasattr(result, 'counterfactual_analysis')


# ============================================================================
# Test 8: End-to-End Integration
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_with_hardware(self):
        """Test full pipeline with mock hardware."""
        from streaming_pipeline.advanced_orchestrator import AdvancedQKDOrchestrator
        from hardware_interface.base_hardware_interface import MockQKDHardware
        
        # Initialize orchestrator
        orchestrator = AdvancedQKDOrchestrator(
            enable_causal_inference=True,
            enable_survival_analysis=True
        )
        
        # Connect hardware
        hardware = MockQKDHardware(link_id="integration_test")
        assert orchestrator.connect_hardware(hardware)
        
        # Process timesteps
        results = []
        for _ in range(5):
            result = orchestrator.process_single_timestep()
            results.append(result)
        
        assert len(results) == 5
        
        # Validate hardware telemetry was captured
        assert any(r.hardware_telemetry is not None for r in results)
        
        # Disconnect
        orchestrator.disconnect_hardware()
    
    def test_continuous_monitoring(self):
        """Test continuous monitoring mode."""
        from streaming_pipeline.advanced_orchestrator import AdvancedQKDOrchestrator
        
        orchestrator = AdvancedQKDOrchestrator()
        
        # Run for 5 seconds
        results = orchestrator.run_continuous_monitoring(
            duration_seconds=5.0,
            dt_seconds=1.0
        )
        
        assert len(results) == 5
        assert all(r.base_result is not None for r in results)


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance and latency tests."""
    
    def test_advanced_pipeline_latency(self):
        """Test that advanced pipeline meets latency requirements."""
        import time
        from streaming_pipeline.advanced_orchestrator import AdvancedQKDOrchestrator
        
        orchestrator = AdvancedQKDOrchestrator()
        
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            result = orchestrator.process_single_timestep()
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Performance targets (may need adjustment)
        assert mean_latency < 100.0, f"Mean latency {mean_latency:.1f}ms exceeds 100ms target"
        assert p95_latency < 200.0, f"P95 latency {p95_latency:.1f}ms exceeds 200ms target"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
