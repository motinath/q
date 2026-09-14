"""
Graph Neural Network for Quantum Network Intelligence

Graph Convolutional Networks (GCN) + Graph Attention Networks (GAT) for
learning network-wide representations and predicting cascading failures.

Author: Q-SENTINEL Development Team
Version: 2.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    logging.warning("PyTorch Geometric not installed. GNN will use MLP fallback.")


@dataclass
class NetworkEmbeddingResult:
    """Result of GNN embedding."""
    link_embeddings: np.ndarray  # [n_links, embedding_dim]
    network_health_score: float  # [0, 1]
    cascade_risk_per_link: np.ndarray  # [n_links]
    attention_weights: Optional[np.ndarray] = None  # [n_links, n_links]


class QuantumNetworkGNN(nn.Module):
    """
    Graph Neural Network for quantum network telemetry.
    
    Architecture:
    1. Graph Attention Layers: Learn edge importance
    2. Graph Convolutional Layers: Aggregate neighborhood information
    3. Link-level prediction heads: Cascade risk, anomaly score
    4. Network-level prediction: Global health score
    
    Input:
    - Node features: [QBER, SKR, Temperature, Dark Counts, ...] per link
    - Edge index: Adjacency matrix (which links are connected)
    
    Output:
    - Link embeddings: 64-dim learned representations
    - Cascade risk: Probability of cascading failure per link
    - Network health: Overall network health score [0, 1]
    """
    
    def __init__(
        self,
        n_features: int = 35,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize GNN architecture.
        
        Args:
            n_features: Number of input features per node (link)
            hidden_dim: Hidden layer dimension
            embedding_dim: Final embedding dimension
            n_heads: Number of attention heads in GAT
            dropout: Dropout probability
        """
        super().__init__()
        
        if not PYTORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNN")
        
        # Graph Attention Layers (multi-head)
        self.gat1 = GATConv(n_features, hidden_dim, heads=n_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=n_heads, dropout=dropout)
        
        # Graph Convolutional Layer
        self.gcn1 = GCNConv(hidden_dim * n_heads, embedding_dim)
        
        # Link-level prediction heads
        self.cascade_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.anomaly_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Network-level prediction head (operates on global pooling)
        self.health_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.dropout = dropout
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features [n_nodes, n_features]
            edge_index: Edge connectivity [2, n_edges]
            batch: Batch assignment for graph-level pooling
        
        Returns:
            embeddings: [n_nodes, embedding_dim]
            cascade_risk: [n_nodes, 1]
            anomaly_score: [n_nodes, 1]
            health_score: [n_graphs, 1] (or [1] if single graph)
        """
        # Multi-head graph attention
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_attn, attention_weights = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x_attn)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph convolution
        embeddings = self.gcn1(x, edge_index)
        embeddings = F.relu(embeddings)
        
        # Link-level predictions
        cascade_risk = self.cascade_predictor(embeddings)
        anomaly_score = self.anomaly_predictor(embeddings)
        
        # Network-level health (global mean pooling)
        if batch is None:
            batch = torch.zeros(embeddings.size(0), dtype=torch.long, device=embeddings.device)
        
        pooled = global_mean_pool(embeddings, batch)
        health_score = self.health_predictor(pooled)
        
        return embeddings, cascade_risk, anomaly_score, health_score


class GNNNetworkIntelligence:
    """
    High-level interface for GNN-based network intelligence.
    
    Provides:
    - Network embedding generation
    - Cascade risk prediction
    - Anomaly scoring
    - Transfer learning across different network topologies
    """
    
    def __init__(
        self,
        n_features: int = 35,
        model_path: Optional[str] = None
    ):
        """
        Initialize GNN intelligence engine.
        
        Args:
            n_features: Number of telemetry features per link
            model_path: Path to pretrained model weights (optional)
        """
        self.n_features = n_features
        self.logger = logging.getLogger(__name__)
        
        if not PYTORCH_GEOMETRIC_AVAILABLE:
            self.logger.warning("Using MLP fallback (PyTorch Geometric unavailable)")
            self.use_fallback = True
            self.model = None
        else:
            self.use_fallback = False
            self.model = QuantumNetworkGNN(n_features=n_features)
            
            if model_path:
                self._load_model(model_path)
            
            self.model.eval()  # Start in evaluation mode
    
    def embed_network(
        self,
        link_features: Dict[str, np.ndarray],
        adjacency_matrix: np.ndarray
    ) -> NetworkEmbeddingResult:
        """
        Generate network embeddings and predictions.
        
        Args:
            link_features: {link_id: feature_vector[35]}
            adjacency_matrix: [n_links, n_links] connectivity matrix
        
        Returns:
            NetworkEmbeddingResult with embeddings and predictions
        """
        if self.use_fallback or self.model is None:
            return self._fallback_mlp_embedding(link_features, adjacency_matrix)
        
        try:
            # Convert to PyTorch Geometric format
            data = self._prepare_graph_data(link_features, adjacency_matrix)
            
            # Forward pass
            with torch.no_grad():
                embeddings, cascade_risk, anomaly_score, health_score = self.model(
                    data.x,
                    data.edge_index
                )
            
            # Convert to numpy
            link_embeddings = embeddings.cpu().numpy()
            cascade_risk_np = cascade_risk.squeeze().cpu().numpy()
            network_health = float(health_score.squeeze().cpu().item())
            
            return NetworkEmbeddingResult(
                link_embeddings=link_embeddings,
                network_health_score=network_health,
                cascade_risk_per_link=cascade_risk_np,
                attention_weights=None  # Could extract from GAT layers
            )
        
        except Exception as e:
            self.logger.error(f"GNN embedding failed: {e}. Using fallback.")
            return self._fallback_mlp_embedding(link_features, adjacency_matrix)
    
    def _prepare_graph_data(
        self,
        link_features: Dict[str, np.ndarray],
        adjacency_matrix: np.ndarray
    ) -> Data:
        """
        Convert to PyTorch Geometric Data object.
        
        Returns:
            Data object with x (features) and edge_index (connectivity)
        """
        # Stack features in consistent order
        link_ids = sorted(link_features.keys())
        feature_matrix = np.stack([link_features[link_id] for link_id in link_ids])
        
        # Convert adjacency matrix to edge index (COO format)
        edge_index = self._adjacency_to_edge_index(adjacency_matrix)
        
        # Create PyTorch tensors
        x = torch.tensor(feature_matrix, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index_tensor)
    
    def _adjacency_to_edge_index(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Convert adjacency matrix to edge index format.
        
        Args:
            adj_matrix: [n, n] adjacency matrix
        
        Returns:
            edge_index: [2, n_edges] COO format
        """
        # Find non-zero entries (edges)
        rows, cols = np.where(adj_matrix > 0)
        edge_index = np.stack([rows, cols], axis=0)
        return edge_index
    
    def train_on_data(
        self,
        training_graphs: List[Tuple[Dict, np.ndarray, Dict]],
        n_epochs: int = 50,
        learning_rate: float = 0.001
    ):
        """
        Train GNN on historical network data.
        
        Args:
            training_graphs: List of (link_features, adj_matrix, labels)
                labels: {"cascade_labels": [n_links], "health_label": float}
            n_epochs: Number of training epochs
            learning_rate: Learning rate for Adam optimizer
        """
        if self.use_fallback or self.model is None:
            self.logger.warning("Training not available in fallback mode")
            return
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Convert training data to batches
        data_list = []
        for features, adj, labels in training_graphs:
            data = self._prepare_graph_data(features, adj)
            
            # Add labels
            data.cascade_y = torch.tensor(labels["cascade_labels"], dtype=torch.float32).unsqueeze(1)
            data.health_y = torch.tensor([labels["health_label"]], dtype=torch.float32).unsqueeze(1)
            
            data_list.append(data)
        
        # Training loop
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            for data in data_list:
                optimizer.zero_grad()
                
                embeddings, cascade_pred, anomaly_pred, health_pred = self.model(
                    data.x,
                    data.edge_index
                )
                
                # Compute losses
                cascade_loss = F.binary_cross_entropy(cascade_pred, data.cascade_y)
                health_loss = F.mse_loss(health_pred, data.health_y)
                
                loss = cascade_loss + health_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}")
        
        self.model.eval()
        self.logger.info("Training complete")
    
    def save_model(self, path: str):
        """Save trained model weights."""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            self.logger.info(f"Model saved to {path}")
    
    def _load_model(self, path: str):
        """Load pretrained model weights."""
        try:
            self.model.load_state_dict(torch.load(path))
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def _fallback_mlp_embedding(
        self,
        link_features: Dict[str, np.ndarray],
        adjacency_matrix: np.ndarray
    ) -> NetworkEmbeddingResult:
        """
        Fallback to simple MLP when GNN unavailable.
        
        Uses feature averaging over neighbors instead of graph convolution.
        """
        link_ids = sorted(link_features.keys())
        n_links = len(link_ids)
        
        # Simple feature projection (simulate embedding)
        feature_matrix = np.stack([link_features[link_id] for link_id in link_ids])
        embeddings = feature_matrix[:, :min(64, feature_matrix.shape[1])]  # Take first 64 features
        
        # Pad if needed
        if embeddings.shape[1] < 64:
            embeddings = np.pad(embeddings, ((0, 0), (0, 64 - embeddings.shape[1])))
        
        # Simple heuristic cascade risk (based on QBER)
        qber_values = feature_matrix[:, 0] if feature_matrix.shape[1] > 0 else np.zeros(n_links)
        cascade_risk = np.clip((qber_values - 0.05) / 0.06, 0, 1)  # Normalize QBER to [0, 1]
        
        # Network health (inverse of mean QBER)
        network_health = float(1.0 - np.mean(cascade_risk))
        
        return NetworkEmbeddingResult(
            link_embeddings=embeddings,
            network_health_score=network_health,
            cascade_risk_per_link=cascade_risk,
            attention_weights=None
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Graph Neural Network for Quantum Networks - Example\n")
    
    # Create mock network (5 links in mesh topology)
    n_links = 5
    n_features = 35
    
    link_features = {
        f"link_{i}": np.random.randn(n_features) for i in range(n_links)
    }
    
    # Mesh topology adjacency matrix
    adjacency_matrix = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ])
    
    # Initialize GNN intelligence
    gnn_engine = GNNNetworkIntelligence(n_features=n_features)
    
    # Generate embeddings
    result = gnn_engine.embed_network(link_features, adjacency_matrix)
    
    print(f"Link Embeddings Shape: {result.link_embeddings.shape}")
    print(f"Network Health Score: {result.network_health_score:.3f}")
    print(f"\nCascade Risk per Link:")
    for i, risk in enumerate(result.cascade_risk_per_link):
        print(f"  link_{i}: {risk:.3f}")
    
    # Simulate training (if PyTorch Geometric available)
    if PYTORCH_GEOMETRIC_AVAILABLE:
        print("\nSimulating GNN training...")
        
        # Generate synthetic training data
        training_graphs = []
        for _ in range(10):
            features = {f"link_{i}": np.random.randn(n_features) for i in range(n_links)}
            labels = {
                "cascade_labels": np.random.randint(0, 2, n_links).astype(float),
                "health_label": np.random.rand()
            }
            training_graphs.append((features, adjacency_matrix, labels))
        
        gnn_engine.train_on_data(training_graphs, n_epochs=20, learning_rate=0.01)
        
        # Re-embed after training
        result_trained = gnn_engine.embed_network(link_features, adjacency_matrix)
        print(f"\nPost-training Network Health: {result_trained.network_health_score:.3f}")
    
    print("\n✓ GNN embedding complete")
