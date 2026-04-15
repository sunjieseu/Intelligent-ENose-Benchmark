"""
Level 1: Feature-Level Fusion Methods

This module implements various feature fusion strategies for multi-sensor E-nose data:
- Simple concatenation
- Principal Component Analysis (PCA)
- Attention-based fusion
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConcatenationFusion:
    """
    Simple concatenation of multi-sensor features.
    
    This is the most direct and widely adopted baseline method for feature-level fusion.
    Assumes the j-th sensor yields a D_j-dimensional feature vector x_j,
    the fused representation is constructed by stacking all M sensor features:
    
    x = [x_1^T, x_2^T, ..., x_M^T]^T
    """
    
    def __init__(self):
        self.fusion_dim = None
    
    def fit(self, X_list):
        """
        Fit the fusion model (no-op for concatenation).
        
        Args:
            X_list: List of feature arrays from different sensors
        """
        self.fusion_dim = sum(x.shape[1] if x.ndim > 1 else 1 for x in X_list)
        logger.info(f"Concatenation fusion: total dimension = {self.fusion_dim}")
    
    def transform(self, X_list):
        """
        Concatenate features from multiple sensors.
        
        Args:
            X_list: List of feature arrays [N, D_j]
            
        Returns:
            Concatenated feature array [N, sum(D_j)]
        """
        return np.concatenate(X_list, axis=1)
    
    def fit_transform(self, X_list):
        """Fit and transform in one step."""
        self.fit(X_list)
        return self.transform(X_list)


class PCAFusion:
    """
    Principal Component Analysis for dimensionality reduction and feature fusion.
    
    PCA projects the concatenated high-dimensional features onto a lower-dimensional
    orthogonal subspace, retaining the top K principal components that capture
    the largest variance. This method effectively removes linear redundancy while
    preserving essential information.
    """
    
    def __init__(self, n_components: int = 10, variance_threshold: Optional[float] = None):
        """
        Initialize PCA fusion.
        
        Args:
            n_components: Number of principal components to retain
            variance_threshold: Alternative - retain components explaining this much variance
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.fusion_dim = None
    
    def fit(self, X_list):
        """
        Fit PCA on concatenated features.
        
        Args:
            X_list: List of feature arrays from different sensors
        """
        # Concatenate features first
        X_concat = np.concatenate(X_list, axis=1)
        self.fusion_dim = X_concat.shape[1]
        
        # Determine number of components
        if self.variance_threshold is not None:
            # Fit PCA with all components first to analyze variance
            pca_full = PCA()
            pca_full.fit(X_concat)
            
            # Find number of components that explains threshold variance
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.searchsorted(cumsum_var, self.variance_threshold) + 1
            logger.info(f"Variance threshold {self.variance_threshold} -> {n_components} components")
            
            self.pca = PCA(n_components=n_components)
        else:
            self.pca = PCA(n_components=self.n_components)
        
        self.pca.fit(X_concat)
        logger.info(f"PCA fusion: {self.fusion_dim} -> {self.pca.n_components_} components, "
                    f"explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
    
    def transform(self, X_list):
        """
        Transform concatenated features using PCA.
        
        Args:
            X_list: List of feature arrays
            
        Returns:
            PCA-transformed features
        """
        X_concat = np.concatenate(X_list, axis=1)
        return self.pca.transform(X_concat)
    
    def fit_transform(self, X_list):
        """Fit and transform in one step."""
        self.fit(X_list)
        return self.transform(X_list)


class AttentionFusion(nn.Module):
    """
    Attention-based sensor-level fusion.
    
    These models dynamically learn the importance weights of each sensor channel
    based on input data, enabling adaptive emphasis on the most informative features.
    
    x_fused = sum(alpha_j * x_j), where alpha = Softmax(W * [x_1, ..., x_M] + b)
    """
    
    def __init__(self, sensor_dim: int, n_sensors: int, hidden_dim: int = 32):
        """
        Initialize attention fusion module.
        
        Args:
            sensor_dim: Dimension of each sensor's feature vector
            n_sensors: Number of sensors in the array
            hidden_dim: Hidden dimension for attention computation
        """
        super(AttentionFusion, self).__init__()
        
        self.sensor_dim = sensor_dim
        self.n_sensors = n_sensors
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, X):
        """
        Compute attention-weighted fusion.
        
        Args:
            X: Input tensor of shape [batch_size, n_sensors, sensor_dim]
            
        Returns:
            Fused feature tensor [batch_size, sensor_dim]
        """
        batch_size, n_sensors, sensor_dim = X.shape
        
        # Compute attention scores
        # X: [batch_size, n_sensors, sensor_dim] -> [batch_size * n_sensors, sensor_dim]
        X_reshaped = X.view(-1, sensor_dim)
        
        # Compute attention weights: [batch_size * n_sensors, 1]
        attention_scores = self.attention_net(X_reshaped)
        
        # Reshape and apply softmax: [batch_size, n_sensors]
        attention_scores = attention_scores.view(batch_size, n_sensors)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention weights: [batch_size, n_sensors, 1] * [batch_size, n_sensors, sensor_dim]
        attention_weights = attention_weights.unsqueeze(2)
        X_weighted = attention_weights * X
        
        # Sum across sensors: [batch_size, sensor_dim]
        X_fused = X_weighted.sum(dim=1)
        
        return X_fused, attention_weights
    
    def fit(self, X_list, y=None, epochs=50, lr=0.001, device='cpu'):
        """
        Train attention fusion (unsupervised or supervised).
        
        Args:
            X_list: List of sensor feature arrays
            y: Labels (optional, for supervised training)
            epochs: Number of training epochs
            lr: Learning rate
            device: Device to train on
            
        Returns:
            Self
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(np.stack(X_list, axis=1))
        X_tensor = X_tensor.to(device)
        
        # Move to device
        self.to(device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Simple reconstruction loss for unsupervised learning
        criterion = nn.MSELoss()
        
        logger.info(f"Training attention fusion for {epochs} epochs")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            X_fused, attention_weights = self(X_tensor)
            
            # Reconstruction loss: try to reconstruct original features
            X_reconstructed = X_fused.unsqueeze(1).expand_as(X_tensor)
            loss = criterion(X_reconstructed, X_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        return self
    
    def transform(self, X_list):
        """
        Transform using trained attention fusion.
        
        Args:
            X_list: List of sensor feature arrays
            
        Returns:
            Attention-weighted fused features
        """
        X_tensor = torch.FloatTensor(np.stack(X_list, axis=1))
        
        with torch.no_grad():
            X_fused, attention_weights = self(X_tensor)
        
        return X_fused.numpy()


class FeatureFusionPipeline:
    """
    Pipeline for comparing different feature fusion methods.
    """
    
    def __init__(self, method: str = 'concatenation', **kwargs):
        """
        Initialize fusion pipeline.
        
        Args:
            method: Fusion method ('concatenation', 'pca', 'attention')
            **kwargs: Additional arguments for the fusion method
        """
        self.method = method
        
        if method == 'concatenation':
            self.fusion = ConcatenationFusion()
        elif method == 'pca':
            self.fusion = PCAFusion(**kwargs)
        elif method == 'attention':
            self.fusion = AttentionFusion(**kwargs)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
    
    def fit_transform(self, X_list):
        """Fit and transform using the selected method."""
        return self.fusion.fit_transform(X_list)
    
    def transform(self, X_list):
        """Transform using the fitted method."""
        return self.fusion.transform(X_list)


if __name__ == "__main__":
    # Example usage
    print("Testing feature fusion methods...")
    
    # Create dummy multi-sensor data
    n_samples = 100
    n_sensors = 4
    sensor_dim = 10
    
    X_list = [np.random.randn(n_samples, sensor_dim) for _ in range(n_sensors)]
    
    # Test concatenation
    print("\n1. Concatenation Fusion:")
    concat_fusion = ConcatenationFusion()
    X_concat = concat_fusion.fit_transform(X_list)
    print(f"   Input: {n_sensors} sensors × {sensor_dim} dims")
    print(f"   Output: {X_concat.shape}")
    
    # Test PCA
    print("\n2. PCA Fusion:")
    pca_fusion = PCAFusion(n_components=15)
    X_pca = pca_fusion.fit_transform(X_list)
    print(f"   Input: {n_sensors} sensors × {sensor_dim} dims")
    print(f"   Output: {X_pca.shape}")
    
    # Test Attention
    print("\n3. Attention Fusion:")
    attn_fusion = AttentionFusion(sensor_dim=sensor_dim, n_sensors=n_sensors)
    X_tensor = torch.FloatTensor(np.stack(X_list, axis=1))
    X_attn, attn_weights = attn_fusion(X_tensor)
    print(f"   Input: {n_sensors} sensors × {sensor_dim} dims")
    print(f"   Output: {X_attn.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    print("\n✓ All fusion methods tested successfully!")
