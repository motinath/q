"""
Advanced QKD Network Orchestrator (Version 2.0)

Integrates cutting-edge AI capabilities:
- Dynamic Bayesian Network for causal root-cause analysis
- Survival Analysis for PTCT forecasting
- Conformal Prediction for calibrated confidence intervals
- Graph Neural Networks for network-wide intelligence
- Causal Bayesian Network for counterfactual reasoning
- Real hardware deployment interfaces

Author: VECTOR Q Development Team
Version: 2.0.0
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from streaming_pipeline.qkd_network_orchestrator import QKDNetworkOrchestrator, PipelineStepResult
from config.qkd_system_parameters import QKDPhysicsConfig

# Advanced engines
from streaming_pipeline.causal_network_intelligence import (
    DynamicBayesianNetworkRCA,
    CausalInferenceResult
)
from predictive_maintenance.survival_ptct_forecaster import (
    SurvivalPTCTForecaster,
    SurvivalForecastResult
)
from predictive_maintenance.conformal_ptct import (
    ConformalPTCTForecaster,
    ConformalForecastResult
)
from streaming_pipeline.graph_network_embeddings import (
    GNNNetworkIntelligence,
    NetworkEmbeddingResult
)
from root_cause_attribution.causal_attribution_engine import (
    CausalAttributionEngine,
    CounterfactualResult
)

# Hardware interface
from hardware_interface import BaseQKDHardwareInterface, HardwareTelemetry

# Continuous Learning Flywheel & Security Rigor
from model_lifecycle.drift_monitor import ModelDriftMonitor, DriftSignal
from physics_engine.finite_key_analysis import FiniteKeySecurityAnalyzer, FiniteKeySecurityResult
from remediation_engine.adaptive_decoy_optimizer import AdaptiveDecoyOptimizer


@dataclass
class AdvancedPipelineResult:
    """Extended pipeline result with advanced AI capabilities."""
    # Base result from standard pipeline
    base_result: PipelineStepResult
    
    # Advanced causal analysis
    causal_inference: Optional[CausalInferenceResult] = None
    counterfactual_analysis: Optional[CounterfactualResult] = None
    
    # Advanced PTCT forecasting
    survival_forecast: Optional[SurvivalForecastResult] = None
    conformal_forecast: Optional[ConformalForecastResult] = None
    
    # Network-wide intelligence (if multi-link)
    network_embedding: Optional[NetworkEmbeddingResult] = None
    
    # Hardware telemetry (if connected to real hardware)
    hardware_telemetry: Optional[HardwareTelemetry] = None

    # Continuous Learning Flywheel & Finite-Key Cryptographic Rigor
    drift_signal: Optional[DriftSignal] = None
    finite_key_result: Optional[FiniteKeySecurityResult] = None
    decoy_recommendation: Optional[Dict[str, Any]] = None



class AdvancedQKDOrchestrator:
    """
    Advanced orchestrator with state-of-the-art AI capabilities.
    
    Extends base orchestrator with:
    1. Causal inference (DBN + Causal Bayesian Network)
    2. Advanced PTCT (Survival Analysis + Conformal Prediction)
    3. Network-wide intelligence (GNN)
    4. Real hardware integration
    5. Counterfactual reasoning
    """
    
    def __init__(
        self,
        config: Optional[QKDPhysicsConfig] = None,
        base_orchestrator: Optional[QKDNetworkOrchestrator] = None,
        hardware_interface: Optional[BaseQKDHardwareInterface] = None,
        enable_causal_inference: bool = True,
        enable_survival_analysis: bool = True,
        enable_conformal_prediction: bool = True,
        enable_gnn: bool = False,  # Requires multi-link network
        enable_counterfactual: bool = True,
        enable_drift_monitoring: bool = True,
        enable_finite_key_analysis: bool = True,
        enable_adaptive_decoy_optimizer: bool = True
    ):
        """
        Initialize advanced orchestrator.
        
        Args:
            config: System configuration
            base_orchestrator: Base orchestrator (or will create new one)
            hardware_interface: Real hardware interface (optional)
            enable_*: Feature flags for advanced engines
        """
        self.config = config or QKDPhysicsConfig()
        self.base_orchestrator = base_orchestrator or QKDNetworkOrchestrator(config=self.config)
        self.hardware_interface = hardware_interface
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize advanced engines
        self.causal_inference_engine = None
        self.survival_forecaster = None
        self.conformal_forecaster = None
        self.gnn_engine = None
        self.counterfactual_engine = None
        
        if enable_causal_inference:
            try:
                self.causal_inference_engine = DynamicBayesianNetworkRCA()
                self.logger.info("✓ Causal inference engine initialized")
            except Exception as e:
                self.logger.warning(f"Causal inference disabled: {e}")
        
        if enable_survival_analysis:
            try:
                self.survival_forecaster = SurvivalPTCTForecaster(
                    threshold_qber=self.config.qber_abort_threshold
                )
                self.logger.info("✓ Survival analysis forecaster initialized")
            except Exception as e:
                self.logger.warning(f"Survival analysis disabled: {e}")
        
        if enable_conformal_prediction:
            try:
                # Will be initialized after training
                self.conformal_forecaster = None
                self.logger.info("✓ Conformal prediction enabled (requires training)")
            except Exception as e:
                self.logger.warning(f"Conformal prediction disabled: {e}")
        
        if enable_gnn:
            try:
                self.gnn_engine = GNNNetworkIntelligence(n_features=35)
                self.logger.info("✓ Graph Neural Network initialized")
            except Exception as e:
                self.logger.warning(f"GNN disabled: {e}")
        
        if enable_counterfactual:
            try:
                self.counterfactual_engine = CausalAttributionEngine()
                self.logger.info("✓ Counterfactual reasoning engine initialized")
            except Exception as e:
                self.logger.warning(f"Counterfactual reasoning disabled: {e}")
        
        # Continuous Learning Flywheel & Security Rigor components
        self.drift_monitor = None
        self.finite_key_analyzer = None
        self.adaptive_decoy_optimizer = None

        if enable_drift_monitoring:
            try:
                self.drift_monitor = ModelDriftMonitor()
                self.logger.info("✓ Confidence drift monitor (ADWIN + Page-Hinkley) initialized")
            except Exception as e:
                self.logger.warning(f"Drift monitor disabled: {e}")

        if enable_finite_key_analysis:
            try:
                self.finite_key_analyzer = FiniteKeySecurityAnalyzer()
                self.logger.info("✓ Finite-key security analyzer (Tomamichel-Lim bound) initialized")
            except Exception as e:
                self.logger.warning(f"Finite-key analyzer disabled: {e}")

        if enable_adaptive_decoy_optimizer:
            try:
                self.adaptive_decoy_optimizer = AdaptiveDecoyOptimizer()
                self.logger.info("✓ Adaptive decoy optimizer (Thompson Sampling MAB, Advisory Mode) initialized")
            except Exception as e:
                self.logger.warning(f"Adaptive decoy optimizer disabled: {e}")
        
        self.step_counter = 0
    
    def attach_survival_model(
        self,
        trained_model: SurvivalPTCTForecaster
    ):
        """Attach pre-trained survival analysis model."""
        self.survival_forecaster = trained_model
        self.logger.info("✓ Survival model attached")
    
    def attach_conformal_predictor(
        self,
        trained_predictor: ConformalPTCTForecaster
    ):
        """Attach pre-trained conformal predictor."""
        self.conformal_forecaster = trained_predictor
        self.logger.info("✓ Conformal predictor attached")
    
    def attach_gnn_model(
        self,
        trained_gnn: GNNNetworkIntelligence
    ):
        """Attach pre-trained GNN model."""
        self.gnn_engine = trained_gnn
        self.logger.info("✓ GNN model attached")
    
    def process_single_timestep(
        self,
        dt_seconds: float = 1.0,
        link_id: Optional[str] = None
    ) -> AdvancedPipelineResult:
        """
        Process single timestep with advanced AI.
        
        Args:
            dt_seconds: Time step duration
            link_id: Link identifier (for hardware telemetry)
        
        Returns:
            AdvancedPipelineResult with all analyses
        """
        self.step_counter += 1
        
        # Step 1: Run base pipeline
        base_result = self.base_orchestrator.process_step(dt_seconds)
        
        # Step 2: Read hardware telemetry if available
        hardware_telemetry = None
        if self.hardware_interface and self.hardware_interface.is_connected:
            try:
                hardware_telemetry = self.hardware_interface.read_telemetry()
                self.logger.debug(f"Hardware QBER: {hardware_telemetry.qber:.4f}")
            except Exception as e:
                self.logger.warning(f"Hardware read failed: {e}")
        
        # Step 3: Causal inference (if anomaly detected)
        causal_result = None
        if self.causal_inference_engine and base_result.anomaly.is_anomaly:
            try:
                observed_anomalies = {link_id or "link_0": True}
                telemetry_features = {link_id or "link_0": base_result.features}
                
                causal_result = self.causal_inference_engine.infer_root_cause(
                    observed_anomalies,
                    telemetry_features
                )
                self.logger.info(f"Causal root cause: {causal_result.root_cause_node} (confidence: {causal_result.confidence:.3f})")
            except Exception as e:
                self.logger.error(f"Causal inference failed: {e}")
        
        # Step 4: Advanced PTCT forecasting
        survival_result = None
        conformal_result = None
        
        if self.survival_forecaster and getattr(self.survival_forecaster, 'is_fitted', False):
            try:
                # Prepare features for survival model
                current_features = pd.DataFrame({
                    'qber_slope': [base_result.features.get('qber_slope_25', 0.0)],
                    'qber_accel': [base_result.features.get('qber_acceleration_25', 0.0)],
                    'temperature': [base_result.features.get('temperature_celsius', 25.0)],
                    'dark_counts': [base_result.features.get('dark_counts_hz', 3000)]
                })
                
                survival_result = self.survival_forecaster.predict_survival(
                    current_features,
                    horizon_seconds=300.0
                )
                self.logger.info(f"Survival TTF: {survival_result.median_ttf:.1f}s (HR: {survival_result.hazard_ratio:.2f})")
            except Exception as e:
                self.logger.error(f"Survival forecasting failed: {e}")
        
        if self.conformal_forecaster and self.conformal_forecaster.is_fitted:
            try:
                feature_vector = np.array([
                    base_result.features.get('qber_slope_25', 0.0),
                    base_result.features.get('qber_acceleration_25', 0.0),
                    base_result.features.get('temperature_celsius', 25.0),
                    base_result.features.get('dark_counts_hz', 3000)
                ])
                
                conformal_result = self.conformal_forecaster.predict(feature_vector)
                self.logger.info(f"Conformal interval: [{conformal_result.lower_bound:.1f}, {conformal_result.upper_bound:.1f}]s with {conformal_result.coverage_guarantee:.1%} coverage")
            except Exception as e:
                self.logger.error(f"Conformal prediction failed: {e}")
        
        # Step 5: Counterfactual analysis (if enabled and anomaly)
        counterfactual_result = None
        if self.counterfactual_engine and base_result.anomaly.is_anomaly:
            try:
                # Discretize current state
                observed_state = self._discretize_state(base_result.features)
                
                # Test intervention: cool detector
                intervention = {'temperature': 'Cold'}
                
                counterfactual_result = self.counterfactual_engine.counterfactual_query(
                    observed_state,
                    intervention
                )
                self.logger.info(f"Counterfactual: {counterfactual_result.interpretation}")
            except Exception as e:
                self.logger.error(f"Counterfactual analysis failed: {e}")
        
        # Step 6: GNN network embedding (if available)
        network_embedding = None
        if self.gnn_engine:
            try:
                # For single link, create minimal topology
                link_features = {link_id or "link_0": self._extract_feature_vector(base_result.features)}
                adjacency_matrix = np.array([[1]])  # Self-loop for single link
                
                network_embedding = self.gnn_engine.embed_network(
                    link_features,
                    adjacency_matrix
                )
                self.logger.info(f"Network health: {network_embedding.network_health_score:.3f}")
            except Exception as e:
                self.logger.error(f"GNN embedding failed: {e}")
        
        # Step 7: Continuous Confidence Drift Monitoring (Flywheel)
        drift_signal = None
        if self.drift_monitor:
            try:
                if hasattr(base_result, 'diagnosis') and base_result.diagnosis:
                    current_conf = float(base_result.diagnosis.confidence)
                elif hasattr(base_result, 'anomaly') and base_result.anomaly:
                    current_conf = float(np.clip(1.0 - getattr(base_result.anomaly, 'anomaly_score', 0.0), 0.50, 1.0))
                else:
                    current_conf = 0.95
                drift_signal = self.drift_monitor.update(current_conf)
                if drift_signal.drift_detected:
                    self.logger.warning(f"Confidence drift alert: {drift_signal.recommendation}")
            except Exception as e:
                self.logger.error(f"Drift monitoring failed: {e}")

        # Step 8: Finite-Key Cryptographic Security Bound (Tomamichel-Lim)
        finite_key_result = None
        if self.finite_key_analyzer and hasattr(base_result, 'sample') and base_result.sample:
            try:
                loss_db = getattr(base_result.sample, 'channel_attenuation_db', getattr(base_result.sample, 'channel_loss_db', 5.0))
                finite_key_result = self.finite_key_analyzer.compute_finite_key_bound(
                    qber_z=base_result.sample.qber,
                    raw_rate_hz=base_result.sample.raw_counts_hz,
                    dark_count_hz=base_result.sample.dark_counts_hz,
                    visibility=base_result.sample.visibility,
                    loss_db=loss_db,
                    block_size_n=1e7,
                )
            except Exception as e:
                self.logger.error(f"Finite key analysis failed: {e}")

        # Step 9: Proactive Adaptive Decoy Optimization (Advisory / Shadow Mode)
        decoy_recommendation = None
        if self.adaptive_decoy_optimizer and hasattr(base_result, 'sample') and base_result.sample:
            try:
                finite_skr = finite_key_result.finite_key_rate_bps if finite_key_result else 0.0
                decoy_recommendation = self.adaptive_decoy_optimizer.step(
                    qber=base_result.sample.qber,
                    skr_bps=base_result.sample.skr_bps,
                    is_anomaly=base_result.anomaly.is_anomaly,
                    finite_key_rate_bps=finite_skr,
                )
            except Exception as e:
                self.logger.error(f"Adaptive decoy optimization failed: {e}")

        return AdvancedPipelineResult(
            base_result=base_result,
            causal_inference=causal_result,
            counterfactual_analysis=counterfactual_result,
            survival_forecast=survival_result,
            conformal_forecast=conformal_result,
            network_embedding=network_embedding,
            hardware_telemetry=hardware_telemetry,
            drift_signal=drift_signal,
            finite_key_result=finite_key_result,
            decoy_recommendation=decoy_recommendation
        )
    
    def _discretize_state(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Discretize continuous features for causal network."""
        qber = features.get('qber', 0.05)
        temp = features.get('temperature_celsius', 25.0)
        
        # QBER bins
        if qber < 0.05:
            qber_discrete = 'Low'
        elif qber < 0.08:
            qber_discrete = 'Normal'
        elif qber < 0.11:
            qber_discrete = 'High'
        else:
            qber_discrete = 'Critical'
        
        # Temperature bins
        if temp < -20:
            temp_discrete = 'Cold'
        elif temp < 10:
            temp_discrete = 'Cool'
        elif temp < 40:
            temp_discrete = 'Normal'
        else:
            temp_discrete = 'Hot'
        
        return {
            'qber': qber_discrete,
            'temperature': temp_discrete
        }
    
    def _extract_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Extract 35-feature vector for GNN."""
        # Standard feature order (must match training)
        feature_names = [
            'qber', 'skr_bps', 'raw_counts_hz', 'dark_counts_hz',
            'singles_alice_hz', 'singles_bob_hz', 'visibility',
            'temperature_celsius', 'timing_jitter_ps'
        ]
        
        vector = []
        for name in feature_names:
            vector.append(features.get(name, 0.0))
        
        # Pad to 35 features
        while len(vector) < 35:
            vector.append(0.0)
        
        return np.array(vector[:35])
    
    def connect_hardware(self, hardware_interface: BaseQKDHardwareInterface) -> bool:
        """
        Connect to real QKD hardware.
        
        Args:
            hardware_interface: Hardware interface object
        
        Returns:
            True if connection successful
        """
        try:
            if hardware_interface.connect():
                self.hardware_interface = hardware_interface
                self.logger.info(f"✓ Connected to hardware: {hardware_interface.link_id}")
                return True
            else:
                self.logger.error("Hardware connection failed")
                return False
        except Exception as e:
            self.logger.error(f"Hardware connection error: {e}")
            return False
    
    def disconnect_hardware(self):
        """Disconnect from hardware."""
        if self.hardware_interface:
            try:
                self.hardware_interface.disconnect()
                self.logger.info("✓ Hardware disconnected")
            except Exception as e:
                self.logger.warning(f"Hardware disconnection error: {e}")
            finally:
                self.hardware_interface = None
    
    def run_continuous_monitoring(
        self,
        duration_seconds: float = 60.0,
        dt_seconds: float = 1.0
    ) -> List[AdvancedPipelineResult]:
        """
        Run continuous monitoring loop.
        
        Args:
            duration_seconds: Total monitoring duration
            dt_seconds: Timestep interval
        
        Returns:
            List of all pipeline results
        """
        results = []
        num_steps = int(duration_seconds / dt_seconds)
        
        self.logger.info(f"Starting continuous monitoring: {num_steps} steps over {duration_seconds}s")
        
        for step in range(num_steps):
            try:
                result = self.process_single_timestep(dt_seconds=dt_seconds)
                results.append(result)
                
                # Log critical events
                if result.base_result.anomaly.is_anomaly:
                    self.logger.warning(f"Step {step}: Anomaly detected - {result.base_result.attribution.predicted_class}")
                
                if step % 10 == 0:
                    self.logger.info(f"Step {step}/{num_steps}: QBER={result.base_result.sample.qber:.4f}")
            
            except Exception as e:
                self.logger.error(f"Step {step} failed: {e}")
            
            time.sleep(dt_seconds)
        
        self.logger.info(f"✓ Continuous monitoring complete: {len(results)} samples")
        return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Advanced QKD Orchestrator - Example\n")
    
    # Initialize advanced orchestrator
    config = QKDPhysicsConfig()
    orchestrator = AdvancedQKDOrchestrator(
        config=config,
        enable_causal_inference=True,
        enable_survival_analysis=True,
        enable_conformal_prediction=False,  # Requires training
        enable_gnn=False,  # Requires multi-link network
        enable_counterfactual=True
    )
    
    print("Processing single timestep...\n")
    
    # Process single timestep
    result = orchestrator.process_single_timestep()
    
    print(f"Base Pipeline:")
    print(f"  QBER: {result.base_result.sample.qber:.4f}")
    print(f"  Anomaly: {result.base_result.anomaly.is_anomaly}")
    print(f"  Root Cause: {result.base_result.attribution.predicted_class}")
    print(f"  Confidence: {result.base_result.attribution.confidence:.3f}")
    
    if result.causal_inference:
        print(f"\nCausal Analysis:")
        print(f"  Root Cause Node: {result.causal_inference.root_cause_node}")
        print(f"  Confidence: {result.causal_inference.confidence:.3f}")
        print(f"  Causal Strength: {result.causal_inference.causal_strength:.3f}")
    
    if result.survival_forecast:
        print(f"\nSurvival Analysis:")
        print(f"  Median TTF: {result.survival_forecast.median_ttf:.1f}s")
        print(f"  Hazard Ratio: {result.survival_forecast.hazard_ratio:.2f}x")
    
    if result.counterfactual_analysis:
        print(f"\nCounterfactual:")
        print(f"  {result.counterfactual_analysis.interpretation}")
    
    print("\n✓ Advanced orchestrator example complete")
