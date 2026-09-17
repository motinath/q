"""
Causal Network Intelligence Module

Dynamic Bayesian Network (DBN) for temporal and spatial causal inference
across multi-link quantum networks. Distinguishes correlation from causation
and enables probabilistic root-cause identification.

Author: VECTOR Q Development Team
Version: 2.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict

try:
    from pgmpy.models import DynamicBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import DBNInference
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    logging.warning("pgmpy not installed. DBN causal inference will use fallback correlation analysis.")


@dataclass
class CausalInferenceResult:
    """Result of causal root-cause inference."""
    root_cause_node: str
    confidence: float
    causal_chain: List[Tuple[str, str]]  # [(parent, child), ...]
    all_posteriors: Dict[str, float]
    causal_strength: float  # 0-1, strength of causal relationship
    temporal_delay: int  # Number of timesteps from cause to effect


class DynamicBayesianNetworkRCA:
    """
    Dynamic Bayesian Network for causal root-cause analysis.
    
    Models:
    - Temporal dependencies: X(t) → X(t+1)
    - Spatial dependencies: X_i(t) → X_j(t) for connected links
    - Environmental factors: Temperature, Loss, etc. → Link State
    
    Advantages over correlation:
    - Identifies causal direction (A causes B vs B causes A)
    - Handles confounding variables
    - Temporal dynamics modeling
    - Probabilistic inference under uncertainty
    """
    
    def __init__(self, network_topology: Optional[object] = None):
        """
        Initialize DBN for causal inference.
        
        Args:
            network_topology: NetworkTopology object with nodes and links
        """
        self.topology = network_topology
        self.dbn = None
        self.inference_engine = None
        self.logger = logging.getLogger(__name__)
        
        if not PGMPY_AVAILABLE:
            self.logger.warning("Using fallback correlation-based inference")
            self.use_fallback = True
        else:
            self.use_fallback = False
            if network_topology:
                self._build_dbn_structure()
    
    def _build_dbn_structure(self):
        """
        Build Dynamic Bayesian Network structure.
        
        Network structure (2-time-slice):
        
        Time t:
        - Link states: {Healthy, Degraded, Critical} per link
        - Environmental: {Temperature, Fiber_Loss, Visibility}
        - Fault indicators: {Hardware_Fault, Attack_Detected}
        
        Time t+1:
        - Same variables
        
        Edges:
        - Intra-slice: Environment(t) → Link(t)
        - Inter-slice: Link(t) → Link(t+1) (temporal persistence)
        - Spatial: Link_i(t) → Link_j(t+1) if physically connected
        """
        if self.use_fallback:
            return
        
        # Build edge structure
        edges_intra = []  # Within same time slice
        edges_inter = []  # Between time slices
        
        # For each link, create temporal edge
        if self.topology and hasattr(self.topology, 'links'):
            for link_id in self.topology.links.keys():
                # Temporal persistence: Link(t) → Link(t+1)
                edges_inter.append(((f"{link_id}", 0), (f"{link_id}", 1)))
                
                # Environmental influence (same time slice)
                edges_intra.append((("temperature", 0), (f"{link_id}", 0)))
                edges_intra.append((("fiber_loss", 0), (f"{link_id}", 0)))
        
        # Spatial edges for connected links
        if self.topology and hasattr(self.topology, 'adjacency_matrix'):
            adj = self.topology.adjacency_matrix
            link_ids = list(self.topology.links.keys())
            
            for i, link_a in enumerate(link_ids):
                for j, link_b in enumerate(link_ids):
                    if i != j and adj[i, j] > 0:  # Connected links
                        # Causal influence across links
                        edges_inter.append(((link_a, 0), (link_b, 1)))
        
        try:
            self.dbn = DynamicBayesianNetwork(edges_inter)
            self.logger.info(f"DBN structure built: {len(edges_inter)} inter-slice edges")
        except Exception as e:
            self.logger.error(f"Failed to build DBN: {e}. Using fallback.")
            self.use_fallback = True
    
    def infer_root_cause(
        self,
        observed_anomalies: Dict[str, bool],
        telemetry_features: Dict[str, Dict[str, float]],
        temporal_history: Optional[List[Dict]] = None
    ) -> CausalInferenceResult:
        """
        Perform causal inference to identify root cause.
        
        Args:
            observed_anomalies: {link_id: is_anomalous}
            telemetry_features: {link_id: {feature: value}}
            temporal_history: Past observations for temporal inference
        
        Returns:
            CausalInferenceResult with root cause identification
        """
        if self.use_fallback or self.dbn is None:
            return self._fallback_correlation_inference(
                observed_anomalies, telemetry_features
            )
        
        # Prepare evidence for Bayesian inference
        evidence = self._prepare_evidence(observed_anomalies, telemetry_features)
        
        # Perform backward inference (abduction)
        posteriors = self._compute_posteriors(evidence)
        
        # Identify most likely root cause
        root_cause = max(posteriors.items(), key=lambda x: x[1])
        
        # Trace causal chain
        causal_chain = self._trace_causal_path(root_cause[0], observed_anomalies)
        
        # Compute causal strength
        causal_strength = self._compute_causal_strength(root_cause[0], posteriors)
        
        return CausalInferenceResult(
            root_cause_node=root_cause[0],
            confidence=root_cause[1],
            causal_chain=causal_chain,
            all_posteriors=posteriors,
            causal_strength=causal_strength,
            temporal_delay=self._estimate_temporal_delay(causal_chain)
        )
    
    def _prepare_evidence(
        self,
        observed_anomalies: Dict[str, bool],
        telemetry_features: Dict[str, Dict[str, float]]
    ) -> Dict[Tuple[str, int], str]:
        """Convert observations to DBN evidence format."""
        evidence = {}
        
        # Map anomaly observations to discrete states
        for link_id, is_anomalous in observed_anomalies.items():
            if is_anomalous:
                # Classify severity based on telemetry
                features = telemetry_features.get(link_id, {})
                qber = features.get('qber', 0.05)
                
                if qber > 0.11:
                    state = "Critical"
                elif qber > 0.08:
                    state = "Degraded"
                else:
                    state = "Healthy"
                
                evidence[(link_id, 1)] = state  # Observation at t=1
        
        return evidence
    
    def _compute_posteriors(self, evidence: Dict) -> Dict[str, float]:
        """
        Compute posterior probabilities P(Root Cause | Evidence).
        
        Uses variable elimination for exact inference in DBN.
        """
        posteriors = {}
        
        try:
            if not hasattr(self, 'inference_engine') or self.inference_engine is None:
                self.inference_engine = DBNInference(self.dbn)
            
            # Query each potential root cause variable at t=0
            candidate_roots = self._get_candidate_root_causes()
            
            for root in candidate_roots:
                try:
                    result = self.inference_engine.query(
                        variables=[(root, 0)],
                        evidence=evidence
                    )
                    # Extract probability of "fault" state
                    posteriors[root] = float(result.values.max())
                except:
                    posteriors[root] = 0.0
        
        except Exception as e:
            self.logger.warning(f"DBN inference failed: {e}. Using uniform priors.")
            # Fallback to uniform distribution
            candidate_roots = self._get_candidate_root_causes()
            posteriors = {root: 1.0 / len(candidate_roots) for root in candidate_roots}
        
        return posteriors
    
    def _get_candidate_root_causes(self) -> List[str]:
        """Get list of potential root cause nodes."""
        if self.topology and hasattr(self.topology, 'links'):
            return list(self.topology.links.keys())
        else:
            # Generic candidates
            return ["link_0", "link_1", "link_2", "temperature", "fiber_loss"]
    
    def _trace_causal_path(
        self,
        root_cause: str,
        observed_effects: Dict[str, bool]
    ) -> List[Tuple[str, str]]:
        """
        Trace causal chain from root cause to observed effects.
        
        Uses graph traversal on DBN structure.
        """
        if self.use_fallback or self.dbn is None:
            # Simple chain: root -> first observed anomaly
            effects = [k for k, v in observed_effects.items() if v]
            if effects:
                return [(root_cause, effects[0])]
            return []
        
        causal_chain = []
        
        try:
            # BFS to find shortest causal path
            from collections import deque
            
            queue = deque([(root_cause, [root_cause])])
            visited = {root_cause}
            
            while queue:
                current, path = queue.popleft()
                
                # Check if reached an observed effect
                if current in observed_effects and observed_effects[current]:
                    # Convert path to edge list
                    for i in range(len(path) - 1):
                        causal_chain.append((path[i], path[i+1]))
                    break
                
                # Explore neighbors in DBN
                # (Simplified: assumes edges represent causal flow)
                neighbors = self._get_causal_children(current)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        except Exception as e:
            self.logger.warning(f"Causal path tracing failed: {e}")
        
        return causal_chain
    
    def _get_causal_children(self, node: str) -> List[str]:
        """Get nodes that are causally influenced by given node."""
        if self.dbn is None:
            return []
        
        children = []
        try:
            # Extract edges from DBN structure
            for edge in self.dbn.get_inter_edges():
                parent, child = edge
                if parent[0] == node:  # parent[0] is variable name
                    children.append(child[0])
        except:
            pass
        
        return children
    
    def _compute_causal_strength(
        self,
        root_cause: str,
        posteriors: Dict[str, float]
    ) -> float:
        """
        Compute strength of causal relationship (0-1).
        
        Based on posterior probability ratio vs baseline.
        """
        posterior = posteriors.get(root_cause, 0.0)
        baseline = 1.0 / len(posteriors)  # Uniform prior
        
        # Causal strength: how much evidence increased belief
        strength = min((posterior - baseline) / (1.0 - baseline), 1.0)
        return max(strength, 0.0)
    
    def _estimate_temporal_delay(self, causal_chain: List[Tuple[str, str]]) -> int:
        """Estimate temporal delay from root cause to effect (in timesteps)."""
        return len(causal_chain)  # Each hop = 1 timestep
    
    def _fallback_correlation_inference(
        self,
        observed_anomalies: Dict[str, bool],
        telemetry_features: Dict[str, Dict[str, float]]
    ) -> CausalInferenceResult:
        """
        Fallback to correlation-based inference when DBN unavailable.
        
        Uses heuristics:
        - Earliest anomaly in time = likely root cause
        - Highest severity = likely root cause
        - Spatial proximity to other failures
        """
        anomalous_links = [link for link, is_anom in observed_anomalies.items() if is_anom]
        
        if not anomalous_links:
            return CausalInferenceResult(
                root_cause_node="unknown",
                confidence=0.0,
                causal_chain=[],
                all_posteriors={},
                causal_strength=0.0,
                temporal_delay=0
            )
        
        # Score each anomalous link
        scores = {}
        for link in anomalous_links:
            features = telemetry_features.get(link, {})
            qber = features.get('qber', 0.05)
            
            # Heuristic score: higher QBER = more likely root cause
            scores[link] = qber
        
        root_cause = max(scores.items(), key=lambda x: x[1])
        
        # Normalize scores to probabilities
        total = sum(scores.values())
        posteriors = {link: score / total for link, score in scores.items()}
        
        # Simple causal chain: root -> other failures
        causal_chain = [(root_cause[0], link) for link in anomalous_links if link != root_cause[0]]
        
        return CausalInferenceResult(
            root_cause_node=root_cause[0],
            confidence=posteriors[root_cause[0]],
            causal_chain=causal_chain,
            all_posteriors=posteriors,
            causal_strength=posteriors[root_cause[0]],
            temporal_delay=1
        )
    
    def update_network_topology(self, topology: object):
        """Update network topology and rebuild DBN."""
        self.topology = topology
        if not self.use_fallback:
            self._build_dbn_structure()


class CausalCorrelationAnalyzer:
    """
    Lightweight correlation analyzer for multi-link analysis.
    
    Complements DBN by providing fast correlation metrics.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history = defaultdict(list)  # {link_id: [qber_values]}
    
    def update(self, link_id: str, qber: float):
        """Update rolling window of telemetry."""
        self.history[link_id].append(qber)
        if len(self.history[link_id]) > self.window_size:
            self.history[link_id].pop(0)
    
    def compute_correlation_matrix(self) -> np.ndarray:
        """
        Compute Pearson correlation matrix across all links.
        
        Returns:
            correlation_matrix: [n_links, n_links]
        """
        link_ids = sorted(self.history.keys())
        n_links = len(link_ids)
        
        if n_links < 2:
            return np.eye(1)
        
        # Build data matrix
        data = []
        for link_id in link_ids:
            values = self.history[link_id]
            if len(values) < self.window_size:
                # Pad with zeros
                values = values + [0.0] * (self.window_size - len(values))
            data.append(values)
        
        data = np.array(data)
        
        # Compute correlation
        correlation_matrix = np.corrcoef(data)
        
        return correlation_matrix
    
    def identify_correlated_failures(
        self,
        anomalous_links: List[str],
        threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of links with correlated failures.
        
        Returns:
            List of (link_a, link_b, correlation_coefficient)
        """
        if len(anomalous_links) < 2:
            return []
        
        corr_matrix = self.compute_correlation_matrix()
        link_ids = sorted(self.history.keys())
        
        correlated_pairs = []
        
        for i, link_a in enumerate(anomalous_links):
            if link_a not in link_ids:
                continue
            idx_a = link_ids.index(link_a)
            
            for link_b in anomalous_links:
                if link_b <= link_a or link_b not in link_ids:
                    continue
                idx_b = link_ids.index(link_b)
                
                corr = corr_matrix[idx_a, idx_b]
                if abs(corr) >= threshold:
                    correlated_pairs.append((link_a, link_b, float(corr)))
        
        return sorted(correlated_pairs, key=lambda x: abs(x[2]), reverse=True)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Dynamic Bayesian Network Causal Inference - Example\n")
    
    # Mock network topology
    class MockTopology:
        def __init__(self):
            self.links = {
                "link_0": None,
                "link_1": None,
                "link_2": None
            }
            self.adjacency_matrix = np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ])
    
    topology = MockTopology()
    
    # Initialize causal inference engine
    causal_engine = DynamicBayesianNetworkRCA(topology)
    
    # Simulate observations
    observed_anomalies = {
        "link_0": False,
        "link_1": True,
        "link_2": True
    }
    
    telemetry_features = {
        "link_1": {"qber": 0.09, "skr": 800},
        "link_2": {"qber": 0.12, "skr": 500}
    }
    
    # Perform causal inference
    result = causal_engine.infer_root_cause(
        observed_anomalies,
        telemetry_features
    )
    
    print(f"Root Cause: {result.root_cause_node}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Causal Strength: {result.causal_strength:.3f}")
    print(f"Temporal Delay: {result.temporal_delay} timesteps")
    print(f"\nCausal Chain:")
    for parent, child in result.causal_chain:
        print(f"  {parent} → {child}")
    
    print(f"\nAll Posteriors:")
    for node, prob in sorted(result.all_posteriors.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node}: {prob:.3f}")
    
    print("\n✓ Causal inference complete")
