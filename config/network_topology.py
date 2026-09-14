"""
Multi-Link QKD Network Topology Configuration

Defines network topology, link dependency graph, and propagation characteristics
for multi-node quantum key distribution networks.

Supports:
- Star, mesh, and ring topologies
- Link failure propagation modeling
- Inter-node correlation analysis
- Network-wide performance metrics

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class TopologyType(Enum):
    """Network topology types."""
    STAR = "star"
    MESH = "mesh"
    RING = "ring"
    HYBRID = "hybrid"


@dataclass
class QKDLink:
    """Single quantum link specification."""
    link_id: str
    node_a: str
    node_b: str
    fiber_distance_km: float
    nominal_alpha_db_per_km: float = 0.20
    priority: int = 1  # 1=critical, 2=high, 3=medium
    is_active: bool = True
    backup_link_ids: List[str] = field(default_factory=list)


@dataclass
class QKDNode:
    """Quantum network node (Alice or Bob transceiver)."""
    node_id: str
    node_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_trusted_node: bool = False  # Trusted repeater vs end-user
    connected_links: List[str] = field(default_factory=list)


@dataclass
class NetworkTopology:
    """Complete network topology definition."""
    topology_type: TopologyType
    nodes: Dict[str, QKDNode]
    links: Dict[str, QKDLink]
    adjacency_matrix: Optional[np.ndarray] = None
    propagation_delay_matrix_ms: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Build adjacency and propagation matrices."""
        if self.adjacency_matrix is None:
            self._build_adjacency_matrix()
        if self.propagation_delay_matrix_ms is None:
            self._build_propagation_delay_matrix()
    
    def _build_adjacency_matrix(self) -> None:
        """Construct binary adjacency matrix from link definitions."""
        n_nodes = len(self.nodes)
        node_id_to_idx = {node_id: i for i, node_id in enumerate(sorted(self.nodes.keys()))}
        
        adj = np.zeros((n_nodes, n_nodes), dtype=np.int32)
        
        for link in self.links.values():
            if not link.is_active:
                continue
            i = node_id_to_idx[link.node_a]
            j = node_id_to_idx[link.node_b]
            adj[i, j] = 1
            adj[j, i] = 1  # Bidirectional
        
        self.adjacency_matrix = adj
    
    def _build_propagation_delay_matrix(self) -> None:
        """Calculate link propagation delays (fiber speed ~200,000 km/s)."""
        n_nodes = len(self.nodes)
        node_id_to_idx = {node_id: i for i, node_id in enumerate(sorted(self.nodes.keys()))}
        
        delays = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(delays, 0.0)
        
        fiber_speed_km_per_ms = 200.0  # ~2/3 speed of light
        
        for link in self.links.values():
            if not link.is_active:
                continue
            i = node_id_to_idx[link.node_a]
            j = node_id_to_idx[link.node_b]
            delay_ms = link.fiber_distance_km / fiber_speed_km_per_ms
            delays[i, j] = delay_ms
            delays[j, i] = delay_ms
        
        self.propagation_delay_matrix_ms = delays
    
    def get_critical_paths(self) -> List[List[str]]:
        """Identify critical paths (paths containing only priority=1 links)."""
        critical_links = [lid for lid, link in self.links.items() 
                         if link.priority == 1 and link.is_active]
        
        # Return all connected components using critical links
        paths = []
        visited = set()
        
        for link_id in critical_links:
            link = self.links[link_id]
            if link.node_a not in visited:
                path = self._trace_critical_path(link.node_a, critical_links, visited)
                if len(path) > 1:
                    paths.append(path)
        
        return paths
    
    def _trace_critical_path(self, start_node: str, critical_links: List[str], 
                            visited: Set[str]) -> List[str]:
        """DFS to trace a critical path from start_node."""
        path = [start_node]
        visited.add(start_node)
        
        # Find connected nodes via critical links
        for link_id in critical_links:
            link = self.links[link_id]
            if link.node_a == start_node and link.node_b not in visited:
                path.extend(self._trace_critical_path(link.node_b, critical_links, visited)[1:])
            elif link.node_b == start_node and link.node_a not in visited:
                path.extend(self._trace_critical_path(link.node_a, critical_links, visited)[1:])
        
        return path
    
    def find_backup_routes(self, failed_link_id: str) -> List[List[str]]:
        """Find alternative routes when a link fails."""
        failed_link = self.links[failed_link_id]
        source = failed_link.node_a
        target = failed_link.node_b
        
        # BFS to find shortest path excluding failed link
        from collections import deque
        
        queue = deque([(source, [source])])
        visited = {source}
        alternative_paths = []
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                alternative_paths.append(path)
                continue
            
            # Find neighbors via active links (excluding failed link)
            for link_id, link in self.links.items():
                if link_id == failed_link_id or not link.is_active:
                    continue
                
                neighbor = None
                if link.node_a == current:
                    neighbor = link.node_b
                elif link.node_b == current:
                    neighbor = link.node_a
                
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return alternative_paths
    
    def get_node_index_map(self) -> Dict[str, int]:
        """Return mapping from node_id to matrix index."""
        return {node_id: i for i, node_id in enumerate(sorted(self.nodes.keys()))}


# ============================================================================
# Pre-Defined Network Topologies
# ============================================================================

def create_default_metropolitan_network() -> NetworkTopology:
    """
    Default 5-node metropolitan mesh network (typical Tier-1 city deployment).
    
    Topology:
        Central Hub (CH) connected to 4 edge nodes (EN1-EN4)
        Additional mesh links between edge nodes for redundancy
    """
    nodes = {
        "CH": QKDNode("CH", "Central Hub", 13.0827, 80.2707, is_trusted_node=True, 
                     connected_links=["L1", "L2", "L3", "L4"]),
        "EN1": QKDNode("EN1", "Edge Node 1 - North", 13.1500, 80.2707, 
                      connected_links=["L1", "L5"]),
        "EN2": QKDNode("EN2", "Edge Node 2 - East", 13.0827, 80.3500, 
                      connected_links=["L2", "L6"]),
        "EN3": QKDNode("EN3", "Edge Node 3 - South", 13.0100, 80.2707, 
                      connected_links=["L3", "L7"]),
        "EN4": QKDNode("EN4", "Edge Node 4 - West", 13.0827, 80.2000, 
                      connected_links=["L4", "L8"]),
    }
    
    links = {
        # Star topology: Central Hub to all edge nodes (priority=1)
        "L1": QKDLink("L1", "CH", "EN1", fiber_distance_km=12.5, priority=1),
        "L2": QKDLink("L2", "CH", "EN2", fiber_distance_km=10.8, priority=1),
        "L3": QKDLink("L3", "CH", "EN3", fiber_distance_km=15.2, priority=1),
        "L4": QKDLink("L4", "CH", "EN4", fiber_distance_km=11.3, priority=1),
        
        # Mesh backup links between edge nodes (priority=2)
        "L5": QKDLink("L5", "EN1", "EN2", fiber_distance_km=18.7, priority=2, 
                     backup_link_ids=["L1", "L2"]),
        "L6": QKDLink("L6", "EN2", "EN3", fiber_distance_km=16.4, priority=2,
                     backup_link_ids=["L2", "L3"]),
        "L7": QKDLink("L7", "EN3", "EN4", fiber_distance_km=19.1, priority=2,
                     backup_link_ids=["L3", "L4"]),
        "L8": QKDLink("L8", "EN4", "EN1", fiber_distance_km=17.5, priority=2,
                     backup_link_ids=["L4", "L1"]),
    }
    
    return NetworkTopology(
        topology_type=TopologyType.HYBRID,
        nodes=nodes,
        links=links
    )


def create_ring_topology(n_nodes: int = 6, avg_distance_km: float = 20.0) -> NetworkTopology:
    """Create a ring topology with n_nodes."""
    nodes = {}
    links = {}
    
    for i in range(n_nodes):
        node_id = f"N{i}"
        nodes[node_id] = QKDNode(
            node_id=node_id,
            node_name=f"Node {i}",
            connected_links=[f"L{i}", f"L{(i-1) % n_nodes}"]
        )
    
    for i in range(n_nodes):
        link_id = f"L{i}"
        links[link_id] = QKDLink(
            link_id=link_id,
            node_a=f"N{i}",
            node_b=f"N{(i+1) % n_nodes}",
            fiber_distance_km=avg_distance_km + np.random.uniform(-5, 5),
            priority=1
        )
    
    return NetworkTopology(
        topology_type=TopologyType.RING,
        nodes=nodes,
        links=links
    )


def create_full_mesh_topology(n_nodes: int = 4) -> NetworkTopology:
    """Create a fully connected mesh topology."""
    nodes = {}
    links = {}
    
    for i in range(n_nodes):
        node_id = f"N{i}"
        nodes[node_id] = QKDNode(
            node_id=node_id,
            node_name=f"Node {i}"
        )
    
    link_idx = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            link_id = f"L{link_idx}"
            links[link_id] = QKDLink(
                link_id=link_id,
                node_a=f"N{i}",
                node_b=f"N{j}",
                fiber_distance_km=15.0 + np.random.uniform(-5, 5),
                priority=1
            )
            nodes[f"N{i}"].connected_links.append(link_id)
            nodes[f"N{j}"].connected_links.append(link_id)
            link_idx += 1
    
    return NetworkTopology(
        topology_type=TopologyType.MESH,
        nodes=nodes,
        links=links
    )
