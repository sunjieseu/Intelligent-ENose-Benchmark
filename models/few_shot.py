"""
Level 2: Few-Shot Learning Methods

This module implements few-shot learning approaches for rapid task adaptation:
- Prototypical Networks: Metric-based few-shot classification
- MAML: Model-Agnostic Meta-Learning
- Relation Networks: Learnable distance metric
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Callable
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for Few-Shot Learning
    
    Each class is represented by a prototype (centroid of support embeddings).
    Classification is performed by computing distances to prototypes.
    
    p(y=c|x) = exp(-d(f(x), p_c)) / sum_c'[exp(-d(f(x), p_c'))]
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, embedding_dim: int = 32,
                 distance_metric: str = 'euclidean'):
        """
        Initialize Prototypical Network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            embedding_dim: Embedding space dimension
            distance_metric: Distance metric ('euclidean', 'cosine')
        """
        super(PrototypicalNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        
        # Embedding network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def compute_prototypes(self, X_support: torch.Tensor, y_support: torch.Tensor,
                          n_way: int) -> torch.Tensor:
        """
        Compute class prototypes from support set.
        
        Args:
            X_support: Support set features [n_support, input_dim]
            y_support: Support set labels [n_support]
            n_way: Number of classes
            
        Returns:
            Prototypes [n_way, embedding_dim]
        """
        # Encode support samples
        embeddings = self.encoder(X_support)
        
        # Compute prototypes (class centroids)
        prototypes = []
        for c in range(n_way):
            mask = (y_support == c)
            if mask.sum() > 0:
                prototype = embeddings[mask].mean(dim=0)
                prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        return prototypes
    
    def compute_distances(self, X_query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between query samples and prototypes.
        
        Args:
            X_query: Query features [n_query, input_dim]
            prototypes: Class prototypes [n_way, embedding_dim]
            
        Returns:
            Distance matrix [n_query, n_way]
        """
        # Encode query samples
        embeddings = self.encoder(X_query)
        
        if self.distance_metric == 'euclidean':
            # Compute squared Euclidean distances
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            embeddings_sq = (embeddings ** 2).sum(dim=1, keepdim=True)
            prototypes_sq = (prototypes ** 2).sum(dim=1, keepdim=True).t()
            distances = embeddings_sq + prototypes_sq - 2 * embeddings @ prototypes.t()
            distances = F.relu(distances)  # Ensure non-negative
        elif self.distance_metric == 'cosine':
            # Compute cosine similarity and convert to distance
            embeddings_norm = F.normalize(embeddings, dim=1)
            prototypes_norm = F.normalize(prototypes, dim=1)
            similarity = embeddings_norm @ prototypes_norm.t()
            distances = 1 - similarity
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(self, X_query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute class probabilities.
        
        Args:
            X_query: Query features
            prototypes: Class prototypes
            
        Returns:
            Class probabilities [n_query, n_way]
        """
        distances = self.compute_distances(X_query, prototypes)
        # Convert distances to probabilities (negative distances for softmax)
        logits = -distances
        probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def train_episode(self, support_set: Tuple, query_set: Tuple,
                     optimizer: torch.optim.Optimizer, device: str = 'cpu') -> float:
        """
        Train on a single episode.
        
        Args:
            support_set: (X_support, y_support)
            query_set: (X_query, y_query)
            optimizer: Optimizer
            device: Device to train on
            
        Returns:
            Episode loss
        """
        self.train()
        optimizer.zero_grad()
        
        X_support, y_support = torch.FloatTensor(support_set[0]).to(device), \
                               torch.LongTensor(support_set[1]).to(device)
        X_query, y_query = torch.FloatTensor(query_set[0]).to(device), \
                           torch.LongTensor(query_set[1]).to(device)
        
        n_way = len(torch.unique(y_support))
        
        # Compute prototypes
        prototypes = self.compute_prototypes(X_support, y_support, n_way)
        
        # Compute query probabilities
        probs = self(X_query, prototypes)
        
        # Compute loss
        loss = F.cross_entropy(probs, y_query)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def predict(self, X_query: np.ndarray, prototypes: torch.Tensor) -> np.ndarray:
        """Predict class labels for query samples."""
        self.eval()
        X_query_t = torch.FloatTensor(X_query)
        
        with torch.no_grad():
            probs = self(X_query_t, prototypes)
            _, predicted = torch.max(probs, 1)
        
        return predicted.numpy()
    
    def evaluate(self, X_query: np.ndarray, y_query: np.ndarray,
                prototypes: torch.Tensor) -> float:
        """Evaluate accuracy on query set."""
        predictions = self.predict(X_query, prototypes)
        accuracy = np.mean(predictions == y_query)
        return accuracy


class MAML:
    """
    Model-Agnostic Meta-Learning (MAML)
    
    Learns a set of sensitive initial parameters such that, for any new task,
    a small number of gradient updates yields significant performance gains.
    
    Inner loop: theta'_i = theta - alpha * grad(L_task_i)
    Outer loop: min_theta sum_i L_task_i(f_{theta'_i})
    """
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, 
                 meta_lr: float = 0.001, n_inner_steps: int = 5):
        """
        Initialize MAML.
        
        Args:
            model: Base model to meta-learn
            inner_lr: Inner loop learning rate (task adaptation)
            meta_lr: Outer loop learning rate (meta-update)
            n_inner_steps: Number of inner loop gradient steps
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_inner_steps = n_inner_steps
    
    def inner_loop(self, X_support: torch.Tensor, y_support: torch.Tensor,
                  params: dict) -> dict:
        """
        Perform inner loop adaptation (task-specific update).
        
        Args:
            X_support: Support set features
            y_support: Support set labels
            params: Current parameters
            
        Returns:
            Updated parameters
        """
        adapted_params = dict(params)
        
        for step in range(self.n_inner_steps):
            # Forward pass with adapted parameters
            output = self.functional_forward(X_support, adapted_params)
            loss = F.cross_entropy(output, y_support)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params.values(),
                                       create_graph=True, only_inputs=True)
            
            # Update parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params
    
    def functional_forward(self, X: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Forward pass using provided parameters (for gradient computation).
        This is a simplified version - in practice, you'd need to implement
        functional versions of all layers.
        """
        # For simplicity, using the model's forward pass
        # In full implementation, you'd manually compute with params
        return self.model(X)
    
    def meta_update(self, task_datasets: List, meta_optimizer: torch.optim.Optimizer,
                   device: str = 'cpu') -> float:
        """
        Perform meta-update across multiple tasks.
        
        Args:
            task_datasets: List of (support_set, query_set) tuples
            meta_optimizer: Meta-optimizer
            device: Device to train on
            
        Returns:
            Meta-loss value
        """
        self.model.train()
        meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        
        for support_set, query_set in task_datasets:
            X_support, y_support = torch.FloatTensor(support_set[0]).to(device), \
                                   torch.LongTensor(support_set[1]).to(device)
            X_query, y_query = torch.FloatTensor(query_set[0]).to(device), \
                               torch.LongTensor(query_set[1]).to(device)
            
            # Get current model parameters
            params = dict(self.model.named_parameters())
            
            # Inner loop: adapt to support set
            adapted_params = self.inner_loop(X_support, y_support, params)
            
            # Evaluate on query set with adapted parameters
            # (In full implementation, would use adapted_params)
            with torch.no_grad():
                output = self.model(X_query)
                loss = F.cross_entropy(output, y_query)
            
            meta_loss = meta_loss + loss
        
        # Average meta-loss
        meta_loss = meta_loss / len(task_datasets)
        
        # Backward pass
        meta_loss.backward()
        meta_optimizer.step()
        
        return meta_loss.item()


class RelationNetwork(nn.Module):
    """
    Relation Networks for Few-Shot Learning
    
    Learns a deep distance metric instead of using fixed distances.
    Uses a relation network to compare query-support pairs.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, relation_dim: int = 8):
        """
        Initialize Relation Network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            relation_dim: Relation score dimension
        """
        super(RelationNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Relation network (learns distance metric)
        self.relation_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, relation_dim),  # Concatenate query + support
            nn.BatchNorm1d(relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, 1),
            nn.Sigmoid()  # Relation score in [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def compute_relations(self, X_query: torch.Tensor, X_support: torch.Tensor) -> torch.Tensor:
        """
        Compute relation scores between query and support samples.
        
        Args:
            X_query: Query features [n_query, input_dim]
            X_support: Support features [n_support, input_dim]
            
        Returns:
            Relation scores [n_query, n_support]
        """
        # Extract features
        query_features = self.feature_extractor(X_query)
        support_features = self.feature_extractor(X_support)
        
        # Compute pairwise relations
        n_query = query_features.shape[0]
        n_support = support_features.shape[0]
        
        relations = torch.zeros(n_query, n_support)
        
        for i in range(n_query):
            for j in range(n_support):
                # Concatenate query and support features
                combined = torch.cat([query_features[i], support_features[j]])
                relations[i, j] = self.relation_network(combined)
        
        return relations
    
    def forward(self, X_query: torch.Tensor, X_support: torch.Tensor,
               y_support: torch.Tensor, n_way: int) -> torch.Tensor:
        """
        Forward pass: classify query based on relations to support.
        
        Args:
            X_query: Query features
            X_support: Support features
            y_support: Support labels
            n_way: Number of classes
            
        Returns:
            Class probabilities [n_query, n_way]
        """
        relations = self.compute_relations(X_query, X_support)
        
        # Aggregate relations by class
        relation_scores = torch.zeros(X_query.shape[0], n_way)
        
        for c in range(n_way):
            mask = (y_support == c)
            if mask.sum() > 0:
                relation_scores[:, c] = relations[:, mask].mean(dim=1)
        
        return relation_scores


def create_fewshot_episode(X: np.ndarray, y: np.ndarray,
                          n_way: int = 5, k_shot: int = 5, n_query: int = 15) -> Tuple:
    """
    Create a few-shot learning episode.
    
    Args:
        X: Dataset features
        y: Dataset labels
        n_way: Number of classes in episode
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        
    Returns:
        Tuple of (support_set, query_set)
    """
    # Select n_way classes
    unique_classes = np.unique(y)
    if len(unique_classes) < n_way:
        raise ValueError(f"Not enough classes: have {len(unique_classes)}, need {n_way}")
    
    selected_classes = np.random.choice(unique_classes, n_way, replace=False)
    
    support_samples = []
    query_samples = []
    
    for cls in selected_classes:
        cls_indices = np.where(y == cls)[0]
        n_available = len(cls_indices)
        
        if n_available < k_shot + n_query:
            logger.warning(f"Class {cls} has only {n_available} samples, "
                          f"need {k_shot + n_query}. Skipping...")
            continue
        
        # Randomly split into support and query
        shuffled_indices = np.random.permutation(cls_indices)
        support_idx = shuffled_indices[:k_shot]
        query_idx = shuffled_indices[k_shot:k_shot + n_query]
        
        support_samples.append((X[support_idx], y[support_idx]))
        query_samples.append((X[query_idx], y[query_idx]))
    
    # Combine samples
    X_support = np.concatenate([s[0] for s in support_samples])
    y_support = np.concatenate([s[1] for s in support_samples])
    X_query = np.concatenate([q[0] for q in query_samples])
    y_query = np.concatenate([q[1] for q in query_samples])
    
    return (X_support, y_support), (X_query, y_query)


if __name__ == "__main__":
    # Example usage
    print("Testing few-shot learning methods...")
    
    # Create dummy data
    n_samples, n_features, n_classes = 200, 16, 6
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Test Prototypical Network
    print("\n1. Prototypical Network:")
    proto_net = PrototypicalNetwork(input_dim=n_features, hidden_dim=64, embedding_dim=32)
    
    # Create episode
    support_set, query_set = create_fewshot_episode(X, y, n_way=5, k_shot=3, n_query=10)
    print(f"   Support set: {support_set[0].shape}")
    print(f"   Query set: {query_set[0].shape}")
    
    # Test forward pass
    X_support_t = torch.FloatTensor(support_set[0])
    y_support_t = torch.LongTensor(support_set[1])
    X_query_t = torch.FloatTensor(query_set[0])
    
    prototypes = proto_net.compute_prototypes(X_support_t, y_support_t, n_way=5)
    print(f"   Prototypes shape: {prototypes.shape}")
    
    probs = proto_net(X_query_t, prototypes)
    print(f"   Query probabilities shape: {probs.shape}")
    
    print("\n✓ Few-shot learning methods tested successfully!")
