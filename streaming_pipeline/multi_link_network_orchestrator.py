"""
Multi-Link QKD Network Orchestrator with Cascading Failure Detection

Extends single-link orchestrator to support multi-node quantum networks with:
- Network-wide anomaly aggregation
- Inter-link correlation analysis
- Cascading failure detection and propagation
- Network health scoring and criticality assessment

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time

from config.network_topology import NetworkTopology, QKDLink, QKDNode
from streaming_pipeline.qkd_network_orchestrator import QKDNetworkOrchestrator
from physics_engine.quantum_telemetry_emulator import QuantumTelemetryEmulator


@dataclass
class LinkDiagnosticState:
    """Complete diagnostic state for a single link."""
    link_id: str
    timestamp: float
    is_anomaly: bool
    anomaly_score: float
    predicted_class: str
    ml_confidence: float
    physics_confidence: float
    unified_trust_score: float
    qber: float
    skr_bps: float
    ptct_seconds: Optional[float]
    urgency_level: str
    recommended_action: str


@dataclass
class NetworkCorrelationResult:
    """Result of inter-link correlation analysis."""
    correlated_link_pairs: List[Tuple[str, str]]
    correlation_coefficients: Dict[Tuple[str, str], float]
    suspected_common_cause: Optional[str]
    cascade_risk_score: float  # 0.0 to 1.0


@dataclass
class CascadingFailureAlert:
    """Alert for detected cascading failure."""
    trigger_link_id: str
    affected_link_ids: List[str]
    propagation_path: List[str]
    estimated_cascade_time_seconds: float
    network_availability_impact: float  # Fraction of network capacity lost
    recommended_isolation_actions: List[str]


class MultiLinkNetworkOrchestrator:
    """
    Network-wide orchestrator managing multiple QKD links simultaneously.
    
    Performs:
    - Independent per-link diagnostics via single-link orchestrators
    - Spatial-temporal correlation analysis across links
    - Cascading failure detection using graph propagation
    - Network-wide health scoring and criticality ranking
    """
    
    def __init__(
        self,
        topology: NetworkTopology,
        correlation_window_size: int = 50,
        correlation_threshold: float = 0.75,
        cascade_detection_enabled: bool = True
    ):
        """
        Args:
            topology: Network topology definition
            correlation_window_size: Window for computing inter-link correlations
            correlation_threshold: Correlation coefficient threshold for flagging
            cascade_detection_enabled: Enable cascading failure detection
        """
        self.topology = topology
        self.correlation_window_size = correlation_window_size
        self.correlation_threshold = correlation_threshold
        self.cascade_detection_enabled = cascade_detection_enabled
        
        # Per-link orchestrators
        self.link_orchestrators: Dict[str, QKDNetworkOrchestrator] = {}
        self.link_emulators: Dict[str, QuantumTelemetryEmulator] = {}
        
        # Initialize orchestrators for each link
        for link_id, link in topology.links.items():
            if link.is_active:
                emulator = QuantumTelemetryEmulator(
                    fiber_distance_km=link.fiber_distance_km,
                    nominal_alpha=link.nominal_alpha_db_per_km
                )
                orchestrator = QKDNetworkOrchestrator(telemetry_source=emulator)
                
                self.link_emulators[link_id] = emulator
                self.link_orchestrators[link_id] = orchestrator
        
        # Diagnostic history for correlation analysis
        self.link_diagnostic_history: Dict[str, deque] = {
            link_id: deque(maxlen=correlation_window_size)
            for link_id in self.link_orchestrators.keys()
        }
        
        # Network-wide state
        self.network_health_score: float = 1.0
        self.active_cascading_failures: List[CascadingFailureAlert] = []
    
    def process_network_timestep(self) -> Dict[str, LinkDiagnosticState]:
        """
        Process one timestep for entire network.
        
        Returns:
            Dictionary mapping link_id to diagnostic state
        """
        timestamp = time.time()
        link_states = {}
        
        # Step 1: Process each link independently
        for link_id, orchestrator in self.link_orchestrators.items():
            try:
                result = orchestrator.process_single_timestep()
                
                state = LinkDiagnosticState(
                    link_id=link_id,
                    timestamp=timestamp,
                    is_anomaly=result.anomaly_result.is_anomaly,
                    anomaly_score=result.anomaly_result.anomaly_score,
                    predicted_class=result.attribution_result.predicted_class,
                    ml_confidence=result.attribution_result.ml_confidence,
                    physics_confidence=result.physics_result.physics_confidence,
                    unified_trust_score=result.physics_result.unified_trust_score,
                    qber=result.features.qber,
                    skr_bps=result.features.skr_bps,
                    ptct_seconds=result.ptct_result.time_to_crossing_seconds 
                                 if result.ptct_result else None,
                    urgency_level=result.ptct_result.urgency_level 
                                 if result.ptct_result else "STABLE",
                    recommended_action=result.remediation_result.selected_action.action_name
                                      if result.remediation_result else "Monitor"
                )
                
                link_states[link_id] = state
                self.link_diagnostic_history[link_id].append(state)
                
            except Exception as e:
                print(f"[ERROR] Link {link_id} processing failed: {e}")
                continue
        
        # Step 2: Network-wide correlation analysis
        correlation_result = self._analyze_inter_link_correlations(link_states)
        
        # Step 3: Cascading failure detection
        if self.cascade_detection_enabled:
            cascade_alerts = self._detect_cascading_failures(link_states, correlation_result)
            self.active_cascading_failures = cascade_alerts
        
        # Step 4: Compute network health score
        self.network_health_score = self._compute_network_health_score(link_states)
        
        return link_states
    
    def _analyze_inter_link_correlations(
        self, 
        current_states: Dict[str, LinkDiagnosticState]
    ) -> NetworkCorrelationResult:
        """
        Analyze correlations between link anomaly scores and QBER trends.
        
        Detects:
        - Temporally correlated anomalies (common environmental cause)
        - Spatially adjacent link failures (physical plant issue)
        """
        correlated_pairs = []
        correlation_coeffs = {}
        
        link_ids = list(current_states.keys())
        
        # Compute pairwise Pearson correlations over recent window
        for i, link_a in enumerate(link_ids):
            for link_b in link_ids[i+1:]:
                history_a = self.link_diagnostic_history[link_a]
                history_b = self.link_diagnostic_history[link_b]
                
                if len(history_a) < 10 or len(history_b) < 10:
                    continue  # Insufficient data
                
                # Extract QBER time series
                qber_a = np.array([s.qber for s in history_a])
                qber_b = np.array([s.qber for s in history_b])
                
                # Pearson correlation
                if np.std(qber_a) > 1e-6 and np.std(qber_b) > 1e-6:
                    corr = np.corrcoef(qber_a, qber_b)[0, 1]
                    
                    if abs(corr) > self.correlation_threshold:
                        correlated_pairs.append((link_a, link_b))
                        correlation_coeffs[(link_a, link_b)] = corr
        
        # Infer common cause
        suspected_cause = None
        cascade_risk = 0.0
        
        if len(correlated_pairs) >= 2:
            # Multiple links correlated → likely environmental or network-wide
            suspected_cause = "Environmental (Temperature/Vibration) or Central Hub Fault"
            cascade_risk = min(1.0, len(correlated_pairs) / len(link_ids))
        elif len(correlated_pairs) == 1:
            link_a, link_b = correlated_pairs[0]
            # Check if links share a common node
            link_a_obj = self.topology.links[link_a]
            link_b_obj = self.topology.links[link_b]
            
            shared_nodes = set([link_a_obj.node_a, link_a_obj.node_b]) & \
                          set([link_b_obj.node_a, link_b_obj.node_b])
            
            if shared_nodes:
                suspected_cause = f"Node Fault at {shared_nodes.pop()}"
                cascade_risk = 0.6
            else:
                suspected_cause = "Parallel Fiber Bundle Stress"
                cascade_risk = 0.4
        
        return NetworkCorrelationResult(
            correlated_link_pairs=correlated_pairs,
            correlation_coefficients=correlation_coeffs,
            suspected_common_cause=suspected_cause,
            cascade_risk_score=cascade_risk
        )
    
    def _detect_cascading_failures(
        self,
        current_states: Dict[str, LinkDiagnosticState],
        correlation_result: NetworkCorrelationResult
    ) -> List[CascadingFailureAlert]:
        """
        Detect cascading failures using graph-based propagation analysis.
        
        Algorithm:
        1. Identify trigger link (highest anomaly score with CRITICAL urgency)
        2. Use BFS to find downstream links at risk
        3. Estimate propagation time using topology propagation delays
        """
        alerts = []
        
        # Find critical anomalies
        critical_links = [
            link_id for link_id, state in current_states.items()
            if state.is_anomaly and state.urgency_level in ["CRITICAL", "CRITICAL_PTCT"]
        ]
        
        if not critical_links:
            return alerts
        
        # For each critical link, simulate failure propagation
        for trigger_link_id in critical_links:
            trigger_state = current_states[trigger_link_id]
            trigger_link = self.topology.links[trigger_link_id]
            
            # BFS from trigger link's nodes
            affected_links = []
            propagation_path = [trigger_link_id]
            visited_nodes = set()
            
            queue = deque([(trigger_link.node_a, 0.0), (trigger_link.node_b, 0.0)])
            
            while queue:
                node_id, elapsed_ms = queue.popleft()
                
                if node_id in visited_nodes:
                    continue
                visited_nodes.add(node_id)
                
                # Find links connected to this node
                node = self.topology.nodes[node_id]
                for connected_link_id in node.connected_links:
                    if connected_link_id == trigger_link_id:
                        continue
                    
                    connected_link = self.topology.links[connected_link_id]
                    
                    # Check if this link is also degrading
                    if connected_link_id in current_states:
                        conn_state = current_states[connected_link_id]
                        
                        # If anomalous and temporally close, likely cascade
                        if conn_state.is_anomaly and conn_state.anomaly_score > 0.6:
                            affected_links.append(connected_link_id)
                            propagation_path.append(connected_link_id)
                            
                            # Add downstream nodes to queue
                            next_node = (connected_link.node_b 
                                       if connected_link.node_a == node_id 
                                       else connected_link.node_a)
                            
                            prop_delay = connected_link.fiber_distance_km / 200.0  # ms
                            queue.append((next_node, elapsed_ms + prop_delay))
            
            if len(affected_links) >= 1:
                # Calculate network availability impact
                total_links = len(self.topology.links)
                impact = (len(affected_links) + 1) / total_links
                
                # Generate isolation recommendations
                isolation_actions = []
                
                # Recommend isolating the trigger link
                isolation_actions.append(
                    f"Isolate link {trigger_link_id} ({trigger_link.node_a} ↔ {trigger_link.node_b})"
                )
                
                # Check for backup routes
                backup_routes = self.topology.find_backup_routes(trigger_link_id)
                if backup_routes:
                    isolation_actions.append(
                        f"Activate backup route: {' → '.join(backup_routes[0])}"
                    )
                else:
                    isolation_actions.append(
                        f"WARNING: No backup route available for {trigger_link_id}"
                    )
                
                alert = CascadingFailureAlert(
                    trigger_link_id=trigger_link_id,
                    affected_link_ids=affected_links,
                    propagation_path=propagation_path,
                    estimated_cascade_time_seconds=elapsed_ms / 1000.0 if queue else 0.0,
                    network_availability_impact=impact,
                    recommended_isolation_actions=isolation_actions
                )
                alerts.append(alert)
        
        return alerts
    
    def _compute_network_health_score(
        self, 
        link_states: Dict[str, LinkDiagnosticState]
    ) -> float:
        """
        Compute network-wide health score [0.0, 1.0].
        
        Formula:
            Health = (Σ priority_weight * link_health) / (Σ priority_weight)
        
        Where link_health = unified_trust_score if not anomalous, else 0.5 * trust_score
        """
        if not link_states:
            return 0.0
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for link_id, state in link_states.items():
            link = self.topology.links[link_id]
            
            # Priority weights: 1→3.0, 2→2.0, 3→1.0
            priority_weight = 4.0 - link.priority
            
            # Link health
            if state.urgency_level == "CRITICAL_PTCT":
                link_health = 0.2
            elif state.is_anomaly:
                link_health = 0.5 * state.unified_trust_score
            else:
                link_health = state.unified_trust_score
            
            weighted_sum += priority_weight * link_health
            weight_sum += priority_weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def get_network_summary(self) -> Dict:
        """Generate comprehensive network summary report."""
        link_summaries = {}
        
        for link_id, history in self.link_diagnostic_history.items():
            if not history:
                continue
            
            latest = history[-1]
            link = self.topology.links[link_id]
            
            link_summaries[link_id] = {
                "link": f"{link.node_a} ↔ {link.node_b}",
                "distance_km": link.fiber_distance_km,
                "priority": link.priority,
                "status": "CRITICAL" if latest.urgency_level in ["CRITICAL", "CRITICAL_PTCT"]
                         else "ANOMALOUS" if latest.is_anomaly
                         else "NOMINAL",
                "qber": f"{latest.qber:.4f}",
                "skr_mbps": f"{latest.skr_bps / 1e6:.3f}",
                "predicted_class": latest.predicted_class,
                "trust_score": f"{latest.unified_trust_score:.3f}",
                "ptct_seconds": latest.ptct_seconds if latest.ptct_seconds else "N/A"
            }
        
        return {
            "network_health_score": f"{self.network_health_score:.3f}",
            "active_links": len([l for l in self.topology.links.values() if l.is_active]),
            "anomalous_links": len([h[-1] for h in self.link_diagnostic_history.values() 
                                   if h and h[-1].is_anomaly]),
            "cascading_failures": len(self.active_cascading_failures),
            "link_details": link_summaries,
            "cascade_alerts": [
                {
                    "trigger": alert.trigger_link_id,
                    "affected": alert.affected_link_ids,
                    "impact": f"{alert.network_availability_impact:.1%}",
                    "actions": alert.recommended_isolation_actions
                }
                for alert in self.active_cascading_failures
            ]
        }
