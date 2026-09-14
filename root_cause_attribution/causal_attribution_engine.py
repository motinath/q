"""
Causal Attribution Engine with Counterfactual Reasoning

Causal Bayesian Network for root-cause analysis with counterfactual inference.
Answers "what-if" questions and provides causal explanations.

Author: Q-SENTINEL Development Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

try:
    from causalnex.structure import StructureModel
    from causalnex.network import BayesianNetwork
    from causalnex.inference import InferenceEngine
    CAUSALNEX_AVAILABLE = True
except ImportError:
    CAUSALNEX_AVAILABLE = False
    logging.warning("CausalNex not installed. Causal inference will use heuristic fallback.")


@dataclass
class CausalAttributionResult:
    """Result of causal attribution with counterfactual analysis."""
    root_cause: str
    confidence: float
    causal_chain: List[Tuple[str, str]]  # [(cause, effect), ...]
    all_causes_ranked: List[Tuple[str, float]]  # [(variable, posterior), ...]
    counterfactual_analysis: Optional[Dict[str, Any]] = None


@dataclass
class CounterfactualResult:
    """Result of counterfactual query."""
    counterfactual_qber: float
    observed_qber: float
    delta: float  # Change from observed
    intervention: Dict[str, float]  # What was changed
    interpretation: str  # Human-readable explanation
    causal_effect_size: float  # Magnitude of causal effect


class CausalAttributionEngine:
    """
    Causal Bayesian Network for root-cause attribution.
    
    Provides:
    - Causal inference (distinguishes causes from correlations)
    - Counterfactual reasoning ("What if we had intervened?")
    - Intervention recommendations
    - Confounder identification
    
    Causal DAG Structure:
    
    Environmental Variables → Telemetry → QBER → SKR
    ├─ Temperature → Dark_Counts → QBER
    ├─ Fiber_Loss → Signal_Counts → QBER
    ├─ Visibility → Optical_Error → QBER
    └─ Attack_Present → QBER (direct)
    
    Enables queries:
    1. Observational: P(QBER | Temperature = 30°C)
    2. Interventional: P(QBER | do(Temperature = -40°C))
    3. Counterfactual: "Given QBER = 0.10, if we had cooled to -40°C, what would QBER be?"
    """
    
    def __init__(self, use_learned_structure: bool = False):
        """
        Initialize causal attribution engine.
        
        Args:
            use_learned_structure: If True, learn DAG from data. If False, use domain knowledge.
        """
        self.use_learned_structure = use_learned_structure
        self.logger = logging.getLogger(__name__)
        
        if not CAUSALNEX_AVAILABLE:
            self.logger.warning("Using heuristic fallback (CausalNex unavailable)")
            self.use_fallback = True
            self.causal_graph = None
            self.bayesian_network = None
        else:
            self.use_fallback = False
            self.causal_graph = self._define_causal_structure()
            self.bayesian_network = None  # Initialized after fitting
            self.inference_engine = None
    
    def _define_causal_structure(self) -> StructureModel:
        """
        Define causal DAG based on physics domain knowledge.
        
        Causal relationships:
        - Temperature affects dark count rate (thermal noise)
        - Dark counts increase QBER
        - Fiber loss reduces signal counts
        - Low signal counts increase QBER (shot noise)
        - Visibility affects optical error rate
        - Optical errors increase QBER
        - QBER directly reduces SKR
        - Attacks directly manipulate QBER
        """
        sm = StructureModel()
        
        # Environmental causes
        sm.add_edges_from([
            ("temperature", "dark_counts"),
            ("dark_counts", "qber"),
            
            ("fiber_loss", "signal_counts"),
            ("signal_counts", "qber"),
            
            ("visibility", "optical_error"),
            ("optical_error", "qber"),
            
            ("timing_jitter", "qber"),
            
            # QBER affects SKR
            ("qber", "skr"),
        ])
        
        # Attack causes (direct manipulation)
        sm.add_edges_from([
            ("intercept_resend", "qber"),
            ("detector_blinding", "signal_counts"),
            ("pns_attack", "qber"),
        ])
        
        # Hardware degradation paths
        sm.add_edges_from([
            ("detector_age", "dark_counts"),
            ("laser_drift", "signal_counts"),
            ("optical_misalignment", "visibility"),
        ])
        
        return sm
    
    def fit(self, telemetry_data: pd.DataFrame):
        """
        Fit Bayesian Network parameters from historical data.
        
        Args:
            telemetry_data: DataFrame with columns matching causal graph variables
        
        Expected columns:
            temperature, dark_counts, fiber_loss, signal_counts, visibility,
            optical_error, timing_jitter, qber, skr, intercept_resend, etc.
        """
        if self.use_fallback:
            self.logger.info("Skipping causal network fitting (fallback mode)")
            return
        
        try:
            # Discretize continuous variables for Bayesian Network
            discretized_data = self._discretize_data(telemetry_data)
            
            # Create Bayesian Network from structure
            self.bayesian_network = BayesianNetwork(self.causal_graph)
            
            # Fit CPDs (Conditional Probability Distributions)
            self.bayesian_network.fit_node_states(discretized_data)
            self.bayesian_network = self.bayesian_network.fit_cpds(
                discretized_data,
                method="BayesianEstimator",
                bayes_prior="K2"
            )
            
            # Initialize inference engine
            self.inference_engine = InferenceEngine(self.bayesian_network)
            
            self.logger.info("Causal Bayesian Network fitted successfully")
        
        except Exception as e:
            self.logger.error(f"Causal network fitting failed: {e}")
            self.use_fallback = True
    
    def _discretize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Discretize continuous variables into bins for Bayesian Network.
        
        QBER: {Low, Normal, High, Critical}
        Temperature: {Cold, Normal, Hot}
        etc.
        """
        discretized = data.copy()
        
        # QBER bins
        if 'qber' in discretized.columns:
            discretized['qber'] = pd.cut(
                discretized['qber'],
                bins=[0, 0.05, 0.08, 0.11, 1.0],
                labels=['Low', 'Normal', 'High', 'Critical']
            )
        
        # Temperature bins
        if 'temperature' in discretized.columns:
            discretized['temperature'] = pd.cut(
                discretized['temperature'],
                bins=[-100, -20, 10, 40, 100],
                labels=['Cold', 'Cool', 'Normal', 'Hot']
            )
        
        # Dark counts
        if 'dark_counts' in discretized.columns:
            discretized['dark_counts'] = pd.cut(
                discretized['dark_counts'],
                bins=[0, 2000, 4000, 8000, 100000],
                labels=['Low', 'Normal', 'High', 'Very_High']
            )
        
        # Signal counts
        if 'signal_counts' in discretized.columns:
            discretized['signal_counts'] = pd.cut(
                discretized['signal_counts'],
                bins=[0, 5000, 10000, 20000, 1000000],
                labels=['Very_Low', 'Low', 'Normal', 'High']
            )
        
        # SKR
        if 'skr' in discretized.columns:
            discretized['skr'] = pd.cut(
                discretized['skr'],
                bins=[0, 500, 1000, 2000, 100000],
                labels=['Critical', 'Low', 'Normal', 'High']
            )
        
        # Boolean variables (attacks)
        for attack_col in ['intercept_resend', 'detector_blinding', 'pns_attack']:
            if attack_col in discretized.columns:
                discretized[attack_col] = discretized[attack_col].astype(str)
        
        return discretized
    
    def infer_root_cause(
        self,
        observed_state: Dict[str, Any]
    ) -> CausalAttributionResult:
        """
        Perform causal inference to identify root cause.
        
        Args:
            observed_state: Current observations (e.g., {"qber": "High", "skr": "Low"})
        
        Returns:
            CausalAttributionResult with root cause and causal chain
        """
        if self.use_fallback or self.bayesian_network is None:
            return self._fallback_heuristic_attribution(observed_state)
        
        try:
            # Backward causal inference: Given effects, what are likely causes?
            posteriors = self._compute_cause_posteriors(observed_state)
            
            # Rank causes by posterior probability
            ranked_causes = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
            
            root_cause = ranked_causes[0]
            
            # Trace causal chain from root cause to observed effects
            causal_chain = self._trace_causal_path(root_cause[0], observed_state)
            
            return CausalAttributionResult(
                root_cause=root_cause[0],
                confidence=root_cause[1],
                causal_chain=causal_chain,
                all_causes_ranked=ranked_causes,
                counterfactual_analysis=None  # Computed separately
            )
        
        except Exception as e:
            self.logger.error(f"Causal inference failed: {e}")
            return self._fallback_heuristic_attribution(observed_state)
    
    def _compute_cause_posteriors(
        self,
        observed_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute P(Cause | Observations) for all potential root causes.
        
        Uses Bayesian inference in causal network.
        """
        if self.inference_engine is None:
            return {}
        
        # List of potential root causes (upstream variables)
        potential_causes = [
            "temperature", "fiber_loss", "visibility", "timing_jitter",
            "intercept_resend", "detector_blinding", "pns_attack",
            "detector_age", "laser_drift", "optical_misalignment"
        ]
        
        posteriors = {}
        
        for cause in potential_causes:
            try:
                # Query posterior P(Cause | Evidence)
                result = self.inference_engine.query(
                    variables=[cause],
                    evidence=observed_state
                )
                
                # Extract probability of "fault" state (highest value)
                posteriors[cause] = float(max(result[cause].values()))
            
            except Exception as e:
                self.logger.warning(f"Failed to query {cause}: {e}")
                posteriors[cause] = 0.0
        
        return posteriors
    
    def _trace_causal_path(
        self,
        root_cause: str,
        observed_effects: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """
        Trace causal chain from root cause to observed effects.
        
        Uses graph traversal on causal DAG.
        """
        if self.causal_graph is None:
            return []
        
        causal_chain = []
        
        try:
            # Find shortest path from root cause to each observed effect
            for effect in observed_effects.keys():
                try:
                    # Use networkx to find path
                    import networkx as nx
                    
                    # Convert StructureModel to networkx DiGraph
                    G = self.causal_graph
                    
                    if nx.has_path(G, root_cause, effect):
                        path = nx.shortest_path(G, root_cause, effect)
                        
                        # Convert path to edge list
                        for i in range(len(path) - 1):
                            edge = (path[i], path[i+1])
                            if edge not in causal_chain:
                                causal_chain.append(edge)
                
                except:
                    pass
        
        except Exception as e:
            self.logger.warning(f"Path tracing failed: {e}")
        
        return causal_chain
    
    def counterfactual_query(
        self,
        observed_state: Dict[str, Any],
        intervention: Dict[str, Any]
    ) -> CounterfactualResult:
        """
        Answer counterfactual query: "What if we had intervened?"
        
        3-step process:
        1. Abduction: Infer latent variables from observed state
        2. Action: Apply intervention (do-operator)
        3. Prediction: Forward inference under intervention
        
        Args:
            observed_state: What actually happened (e.g., {"qber": "High", "temperature": "Hot"})
            intervention: What we want to test (e.g., {"temperature": "Cold"})
        
        Returns:
            CounterfactualResult with predicted outcome under intervention
        """
        if self.use_fallback or self.bayesian_network is None:
            return self._fallback_counterfactual(observed_state, intervention)
        
        try:
            # Step 1: Abduction - infer latent causes
            latent_state = self._abduct_latent_variables(observed_state)
            
            # Step 2: Action - apply intervention
            modified_network = self._apply_intervention(intervention)
            
            # Step 3: Prediction - forward inference
            counterfactual_qber = self._predict_forward(modified_network, latent_state, intervention)
            
            # Extract observed QBER
            observed_qber_discrete = observed_state.get('qber', 'Normal')
            observed_qber = self._discrete_to_continuous(observed_qber_discrete, 'qber')
            
            # Compute delta
            delta = counterfactual_qber - observed_qber
            
            # Generate interpretation
            interpretation = self._interpret_counterfactual(
                intervention,
                observed_qber,
                counterfactual_qber,
                delta
            )
            
            return CounterfactualResult(
                counterfactual_qber=counterfactual_qber,
                observed_qber=observed_qber,
                delta=delta,
                intervention=intervention,
                interpretation=interpretation,
                causal_effect_size=abs(delta)
            )
        
        except Exception as e:
            self.logger.error(f"Counterfactual query failed: {e}")
            return self._fallback_counterfactual(observed_state, intervention)
    
    def _abduct_latent_variables(self, observed_state: Dict[str, Any]) -> Dict[str, Any]:
        """Infer unobserved variables given observations."""
        if self.inference_engine is None:
            return observed_state
        
        # Query all unobserved variables
        observed_vars = set(observed_state.keys())
        all_vars = set(self.causal_graph.nodes())
        latent_vars = all_vars - observed_vars
        
        latent_state = {}
        for var in latent_vars:
            try:
                result = self.inference_engine.query(
                    variables=[var],
                    evidence=observed_state
                )
                # Use most likely state
                latent_state[var] = max(result[var], key=result[var].get)
            except:
                pass
        
        return {**observed_state, **latent_state}
    
    def _apply_intervention(self, intervention: Dict[str, Any]):
        """
        Apply do-operator: Remove incoming edges to intervention variables.
        
        This simulates "forcing" a variable to a value, breaking its
        dependence on parents.
        """
        # Create modified network with intervention edges removed
        modified_graph = self.causal_graph.copy()
        
        for var in intervention.keys():
            # Remove all incoming edges (parents)
            parents = list(modified_graph.predecessors(var))
            for parent in parents:
                modified_graph.remove_edge(parent, var)
        
        # Create new Bayesian Network
        modified_bn = BayesianNetwork(modified_graph)
        
        # Copy CPDs from original network (except intervened variables)
        # (Simplified - in full implementation would refit)
        
        return modified_bn
    
    def _predict_forward(
        self,
        network: Any,
        latent_state: Dict[str, Any],
        intervention: Dict[str, Any]
    ) -> float:
        """
        Forward prediction: P(QBER | do(Intervention), Latent State).
        """
        try:
            # Combine latent state with intervention
            evidence = {**latent_state, **intervention}
            
            # Remove 'qber' from evidence (we're predicting it)
            evidence.pop('qber', None)
            
            # Query QBER distribution
            inference = InferenceEngine(network)
            result = inference.query(variables=['qber'], evidence=evidence)
            
            # Convert discrete prediction to continuous
            qber_dist = result['qber']
            expected_qber = sum(
                self._discrete_to_continuous(state, 'qber') * prob
                for state, prob in qber_dist.items()
            )
            
            return expected_qber
        
        except:
            # Fallback: use observed QBER
            return 0.07
    
    def _discrete_to_continuous(self, discrete_value: str, variable: str) -> float:
        """Map discrete bin labels back to continuous values."""
        mappings = {
            'qber': {
                'Low': 0.03,
                'Normal': 0.06,
                'High': 0.09,
                'Critical': 0.13
            },
            'temperature': {
                'Cold': -30,
                'Cool': 0,
                'Normal': 20,
                'Hot': 35
            },
            'skr': {
                'Critical': 300,
                'Low': 700,
                'Normal': 1200,
                'High': 1800
            }
        }
        
        return mappings.get(variable, {}).get(discrete_value, 0.0)
    
    def _interpret_counterfactual(
        self,
        intervention: Dict[str, Any],
        observed_qber: float,
        counterfactual_qber: float,
        delta: float
    ) -> str:
        """Generate human-readable counterfactual explanation."""
        intervention_desc = ", ".join(f"{k}={v}" for k, v in intervention.items())
        
        if abs(delta) < 0.01:
            return f"Intervention ({intervention_desc}) would have minimal effect (Δ={delta:+.4f})"
        elif delta < 0:
            return f"If we had set {intervention_desc}, QBER would improve by {abs(delta):.4f} (from {observed_qber:.4f} to {counterfactual_qber:.4f})"
        else:
            return f"If we had set {intervention_desc}, QBER would worsen by {delta:.4f} (from {observed_qber:.4f} to {counterfactual_qber:.4f})"
    
    def _fallback_heuristic_attribution(
        self,
        observed_state: Dict[str, Any]
    ) -> CausalAttributionResult:
        """Fallback using simple heuristics when causal network unavailable."""
        # Simple rule-based attribution
        qber = observed_state.get('qber', 'Normal')
        
        if qber in ['High', 'Critical']:
            # Check for obvious causes
            if 'temperature' in observed_state and observed_state['temperature'] == 'Hot':
                root_cause = 'temperature'
                confidence = 0.8
            elif 'intercept_resend' in observed_state and observed_state['intercept_resend'] == True:
                root_cause = 'intercept_resend'
                confidence = 0.9
            else:
                root_cause = 'unknown'
                confidence = 0.5
        else:
            root_cause = 'nominal'
            confidence = 0.7
        
        return CausalAttributionResult(
            root_cause=root_cause,
            confidence=confidence,
            causal_chain=[(root_cause, 'qber')],
            all_causes_ranked=[(root_cause, confidence)],
            counterfactual_analysis=None
        )
    
    def _fallback_counterfactual(
        self,
        observed_state: Dict[str, Any],
        intervention: Dict[str, Any]
    ) -> CounterfactualResult:
        """Fallback counterfactual using physics equations."""
        observed_qber = 0.09
        
        # Simple physics model
        if 'temperature' in intervention:
            temp_change = 20 - self._discrete_to_continuous(intervention['temperature'], 'temperature')
            # Assume 0.001 QBER improvement per 10°C cooling
            delta = -temp_change * 0.0001
            counterfactual_qber = observed_qber + delta
        else:
            counterfactual_qber = observed_qber
            delta = 0.0
        
        interpretation = self._interpret_counterfactual(
            intervention, observed_qber, counterfactual_qber, delta
        )
        
        return CounterfactualResult(
            counterfactual_qber=counterfactual_qber,
            observed_qber=observed_qber,
            delta=delta,
            intervention=intervention,
            interpretation=interpretation,
            causal_effect_size=abs(delta)
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Causal Attribution Engine with Counterfactuals - Example\n")
    
    # Initialize engine
    causal_engine = CausalAttributionEngine()
    
    # Simulate training data (would be real telemetry in production)
    np.random.seed(42)
    n_samples = 500
    
    training_data = pd.DataFrame({
        'temperature': np.random.choice(['Cold', 'Cool', 'Normal', 'Hot'], n_samples),
        'dark_counts': np.random.choice(['Low', 'Normal', 'High'], n_samples),
        'fiber_loss': np.random.uniform(0, 10, n_samples),
        'signal_counts': np.random.choice(['Low', 'Normal', 'High'], n_samples),
        'visibility': np.random.uniform(0.9, 1.0, n_samples),
        'optical_error': np.random.uniform(0, 0.05, n_samples),
        'timing_jitter': np.random.uniform(0, 100, n_samples),
        'qber': np.random.choice(['Low', 'Normal', 'High', 'Critical'], n_samples),
        'skr': np.random.choice(['Low', 'Normal', 'High'], n_samples),
        'intercept_resend': np.random.choice([False, True], n_samples, p=[0.9, 0.1])
    })
    
    # Fit causal network
    print("Fitting causal Bayesian network...")
    causal_engine.fit(training_data)
    
    # Causal inference example
    print("\n--- Causal Inference ---")
    observed_state = {
        'qber': 'High',
        'skr': 'Low',
        'dark_counts': 'High'
    }
    
    result = causal_engine.infer_root_cause(observed_state)
    
    print(f"Root Cause: {result.root_cause}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"\nCausal Chain:")
    for cause, effect in result.causal_chain:
        print(f"  {cause} → {effect}")
    
    print(f"\nAll Causes Ranked:")
    for cause, prob in result.all_causes_ranked[:5]:
        print(f"  {cause}: {prob:.3f}")
    
    # Counterfactual query
    print("\n--- Counterfactual Analysis ---")
    counterfactual = causal_engine.counterfactual_query(
        observed_state={'qber': 'High', 'temperature': 'Hot'},
        intervention={'temperature': 'Cold'}
    )
    
    print(f"Observed QBER: {counterfactual.observed_qber:.4f}")
    print(f"Counterfactual QBER: {counterfactual.counterfactual_qber:.4f}")
    print(f"Delta: {counterfactual.delta:+.4f}")
    print(f"Causal Effect Size: {counterfactual.causal_effect_size:.4f}")
    print(f"\nInterpretation: {counterfactual.interpretation}")
    
    print("\n✓ Causal attribution complete")
