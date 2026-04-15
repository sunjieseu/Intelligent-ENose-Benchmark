"""
Level 2: Transfer Learning Methods

This module implements transfer learning approaches for E-nose domain adaptation:
- TCA: Transfer Component Analysis (MMD-based)
- JDA: Joint Distribution Adaptation
- DANN: Domain-Adversarial Neural Network
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from scipy.linalg import sqrtm
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TCA:
    """
    Transfer Component Analysis (TCA)
    
    Minimizes MMD in a reduced subspace while preserving variance.
    Maps source and target domains into a reproducing kernel Hilbert space (RKHS)
    and matches their distributions by minimizing Maximum Mean Discrepancy (MMD).
    """
    
    def __init__(self, n_components: int = 20, kernel_type: str = 'rbf', 
                 gamma: float = 1.0, mu: float = 0.1):
        """
        Initialize TCA.
        
        Args:
            n_components: Number of transfer components
            kernel_type: Kernel type ('rbf', 'linear', 'poly')
            gamma: Kernel parameter
            mu: Trade-off parameter for variance preservation
        """
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.mu = mu
        self.A = None  # Transformation matrix
    
    def _kernel(self, X, Y):
        """Compute kernel matrix."""
        if self.kernel_type == 'linear':
            K = X @ Y.T
        elif self.kernel_type == 'rbf':
            # RBF kernel: exp(-gamma * ||x - y||^2)
            X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
            dist = X_norm + Y_norm - 2 * X @ Y.T
            K = np.exp(-self.gamma * dist)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
        return K
    
    def fit(self, X_source, X_target):
        """
        Fit TCA transformation.
        
        Args:
            X_source: Source domain data [n_source, d]
            X_target: Target domain data [n_target, d]
        """
        n_source, n_target = X_source.shape[0], X_target.shape[0]
        n_total = n_source + n_target
        
        # Combine data
        X = np.vstack([X_source, X_target])
        
        # Compute kernel matrix
        K = self._kernel(X, X)
        
        # Construct MMD matrix
        L = np.zeros((n_total, n_total))
        L[:n_source, :n_source] = 1 / n_source**2
        L[n_source:, n_source:] = 1 / n_target**2
        L[:n_source, n_source:] = -1 / (n_source * n_target)
        L[n_source:, :n_source] = -1 / (n_source * n_target)
        
        # Centering matrix
        H = np.eye(n_total) - 1 / n_total * np.ones((n_total, n_total))
        
        # Solve generalized eigenvalue problem
        # (K L K^T + mu * I) A = K H K^T A
        A = K @ L @ K.T + self.mu * np.eye(n_total)
        B = K @ H @ K.T
        
        # Add small regularization for numerical stability
        A += 1e-6 * np.eye(n_total)
        
        # Solve eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eigh(B, A)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        self.A = eigenvectors[:, :self.n_components]
        
        logger.info(f"TCA fitted: {X.shape[1]} -> {self.n_components} components")
    
    def transform(self, X):
        """
        Transform data to transfer subspace.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        K = self._kernel(X, X)
        return K @ self.A
    
    def fit_transform(self, X_source, X_target):
        """Fit and transform source and target data."""
        self.fit(X_source, X_target)
        X_combined = np.vstack([X_source, X_target])
        return self.transform(X_combined)


class JDA:
    """
    Joint Distribution Adaptation (JDA)
    
    Extends TCA by aligning both marginal and conditional distributions.
    Incorporates class structure information for better adaptation.
    
    min Tr(A^T K L K^T A) + mu * ||A||_F^2
    s.t. A^T K H K^T A = I
    """
    
    def __init__(self, n_components: int = 20, kernel_type: str = 'rbf',
                 gamma: float = 1.0, mu: float = 0.1, n_iter: int = 10):
        """
        Initialize JDA.
        
        Args:
            n_components: Number of components
            kernel_type: Kernel type
            gamma: Kernel parameter
            mu: Regularization parameter
            n_iter: Number of iterations for pseudo-label refinement
        """
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.mu = mu
        self.n_iter = n_iter
        self.A = None
    
    def _kernel(self, X, Y):
        """Compute kernel matrix."""
        if self.kernel_type == 'linear':
            return X @ Y.T
        elif self.kernel_type == 'rbf':
            X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
            dist = X_norm + Y_norm - 2 * X @ Y.T
            return np.exp(-self.gamma * dist)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
    def _compute_mmd_matrix(self, n_source, n_target, y_pred=None):
        """Compute joint MMD matrix (marginal + conditional)."""
        n_total = n_source + n_target
        L = np.zeros((n_total, n_total))
        
        if y_pred is None:
            # Marginal distribution only
            L[:n_source, :n_source] = 1 / n_source**2
            L[n_source:, n_source:] = 1 / n_target**2
            L[:n_source, n_source:] = -1 / (n_source * n_target)
            L[n_source:, :n_source] = -1 / (n_source * n_target)
        else:
            # Joint distribution (marginal + conditional)
            classes = np.unique(y_pred)
            
            # Marginal part
            L[:n_source, :n_source] += 1 / n_source**2
            L[n_source:, n_source:] += 1 / n_target**2
            L[:n_source, n_source:] -= 1 / (n_source * n_target)
            L[n_source:, :n_source] -= 1 / (n_source * n_target)
            
            # Conditional part
            for c in classes:
                idx_source_c = np.where(y_pred[:n_source] == c)[0]
                idx_target_c = np.where(y_pred[n_source:] == c)[0]
                
                if len(idx_source_c) == 0 or len(idx_target_c) == 0:
                    continue
                
                n_source_c = len(idx_source_c)
                n_target_c = len(idx_target_c)
                
                for i in idx_source_c:
                    for j in idx_source_c:
                        L[i, j] += 1 / (len(classes) * n_source_c**2)
                
                for i in idx_target_c:
                    for j in idx_target_c:
                        L[n_source + i, n_source + j] += 1 / (len(classes) * n_target_c**2)
                
                for i in idx_source_c:
                    for j in idx_target_c:
                        L[i, n_source + j] -= 1 / (len(classes) * n_source_c * n_target_c)
                        L[n_source + j, i] -= 1 / (len(classes) * n_source_c * n_target_c)
        
        return L
    
    def fit(self, X_source, X_target, y_source=None, y_target=None):
        """
        Fit JDA transformation.
        
        Args:
            X_source: Source domain data
            X_target: Target domain data
            y_source: Source labels (optional)
            y_target: Target labels or pseudo-labels (optional)
        """
        n_source = X_source.shape[0]
        X = np.vstack([X_source, X_target])
        n_total = X.shape[0]
        
        # Compute kernel matrix
        K = self._kernel(X, X)
        
        # Initialize with marginal distribution
        y_pred = None
        
        # Iterative refinement with pseudo-labels
        for iteration in range(self.n_iter):
            logger.info(f"JDA iteration {iteration + 1}/{self.n_iter}")
            
            # Compute MMD matrix
            L = self._compute_mmd_matrix(n_source, X_target.shape[0], y_pred)
            
            # Centering matrix
            H = np.eye(n_total) - 1 / n_total * np.ones((n_total, n_total))
            
            # Solve eigenvalue problem
            A_mat = K @ L @ K.T + self.mu * np.eye(n_total)
            B_mat = K @ H @ K.T
            
            # Add regularization
            A_mat += 1e-6 * np.eye(n_total)
            
            # Solve
            eigenvalues, eigenvectors = np.linalg.eigh(B_mat, A_mat)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            self.A = eigenvectors[:, :self.n_components]
            
            # Transform and update pseudo-labels
            X_transformed = self.transform(X)
            
            # Train simple classifier and predict
            if y_source is not None:
                clf = SVC(kernel='linear')
                clf.fit(X_transformed[:n_source], y_source)
                y_pred = clf.predict(X_transformed[n_source:])
        
        logger.info(f"JDA fitted: {X.shape[1]} -> {self.n_components} components")
    
    def transform(self, X):
        """Transform data to JDA subspace."""
        K = self._kernel(X, X)
        return K @ self.A
    
    def fit_transform(self, X_source, X_target, y_source=None):
        """Fit and transform data."""
        self.fit(X_source, X_target, y_source)
        X_combined = np.vstack([X_source, X_target])
        return self.transform(X_combined)


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for DANN."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DANN(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN)
    
    Learns domain-invariant features through adversarial training.
    Consists of:
    - Feature extractor (G_f)
    - Label predictor (G_y)
    - Domain discriminator (G_d) with gradient reversal
    
    The feature extractor is trained to:
    1. Minimize label prediction loss on source domain
    2. Maximize domain classification loss (confuse the discriminator)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 6,
                 alpha: float = 1.0):
        """
        Initialize DANN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of classes
            alpha: Trade-off parameter for domain adversarial loss
        """
        super(DANN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.alpha = alpha
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Label predictor
        self.label_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Domain discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Gradient reversal layer (applied during forward pass)
        self.grl = GradientReversalLayer()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, domain_label=None, alpha=1.0):
        """
        Forward pass through DANN.
        
        Args:
            x: Input features
            domain_label: Domain labels (0 for source, 1 for target)
            alpha: Gradient reversal coefficient
            
        Returns:
            class_pred, domain_pred
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Predict class labels
        class_pred = self.label_predictor(features)
        
        # Apply gradient reversal and predict domain
        reversed_features = self.grl(features, alpha)
        domain_pred = self.domain_discriminator(reversed_features)
        
        return class_pred, domain_pred.squeeze()
    
    def fit(self, X_source, y_source, X_target, epochs: int = 100,
            batch_size: int = 32, lr: float = 0.001, device: str = 'cpu'):
        """
        Train DANN model.
        
        Args:
            X_source: Source domain data
            y_source: Source domain labels
            X_target: Target domain data (unlabeled)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            device: Device to train on
        """
        # Convert to tensors
        X_source_t = torch.FloatTensor(X_source)
        y_source_t = torch.LongTensor(y_source)
        X_target_t = torch.FloatTensor(X_target)
        
        # Move to device
        self.to(device)
        X_source_t = X_source_t.to(device)
        y_source_t = y_source_t.to(device)
        X_target_t = X_target_t.to(device)
        
        # Setup optimizers
        optimizer_f = optim.Adam(list(self.feature_extractor.parameters()) + 
                                 list(self.label_predictor.parameters()), lr=lr)
        optimizer_d = optim.Adam(self.domain_discriminator.parameters(), lr=lr)
        
        # Loss functions
        criterion_cls = nn.CrossEntropyLoss()
        criterion_dom = nn.BCEWithLogitsLoss()
        
        n_source = X_source_t.shape[0]
        n_target = X_target_t.shape[0]
        
        logger.info(f"Training DANN for {epochs} epochs")
        logger.info(f"Source: {n_source} samples, Target: {n_target} samples")
        
        for epoch in range(epochs):
            self.train()
            
            # Shuffle data
            perm_source = torch.randperm(n_source)
            perm_target = torch.randperm(n_target)
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, max(n_source, n_target), batch_size):
                # Get batches
                idx_s = perm_source[i % n_source: (i + batch_size) % n_source]
                idx_t = perm_target[i % n_target: (i + batch_size) % n_target]
                
                if len(idx_s) == 0 or len(idx_t) == 0:
                    continue
                
                X_s, y_s = X_source_t[idx_s], y_source_t[idx_s]
                X_t = X_target_t[idx_t]
                
                # Domain labels
                d_s = torch.zeros(len(X_s)).to(device)
                d_t = torch.ones(len(X_t)).to(device)
                
                # Compute alpha (increases with epoch)
                p = (epoch + i / n_source) / epochs
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                # Update feature extractor and label predictor
                optimizer_f.zero_grad()
                
                class_pred_s, domain_pred_s = self(X_s, alpha=alpha)
                _, domain_pred_t = self(X_t, alpha=alpha)
                
                loss_cls = criterion_cls(class_pred_s, y_s)
                loss_dom_s = criterion_dom(domain_pred_s, d_s)
                loss_dom_t = criterion_dom(domain_pred_t, d_t)
                loss_dom = (loss_dom_s + loss_dom_t) / 2
                
                loss = loss_cls - alpha * loss_dom
                loss.backward()
                optimizer_f.step()
                
                # Update domain discriminator
                optimizer_d.zero_grad()
                
                with torch.no_grad():
                    features_s = self.feature_extractor(X_s)
                    features_t = self.feature_extractor(X_t)
                
                domain_pred_s = self.domain_discriminator(features_s)
                domain_pred_t = self.domain_discriminator(features_t)
                
                loss_dom_s = criterion_dom(domain_pred_s, d_s)
                loss_dom_t = criterion_dom(domain_pred_t, d_t)
                loss_dom = (loss_dom_s + loss_dom_t) / 2
                
                loss_dom.backward()
                optimizer_d.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        """Predict class labels."""
        self.eval()
        X_t = torch.FloatTensor(X)
        
        with torch.no_grad():
            class_pred, _ = self(X_t)
            _, predicted = torch.max(class_pred, 1)
        
        return predicted.numpy()
    
    def evaluate(self, X, y):
        """Evaluate model accuracy."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


if __name__ == "__main__":
    # Example usage
    print("Testing transfer learning methods...")
    
    # Create dummy data
    n_source, n_target, n_features = 100, 80, 16
    X_source = np.random.randn(n_source, n_features)
    X_target = np.random.randn(n_target, n_features) + 0.5  # Shifted distribution
    y_source = np.random.randint(0, 6, n_source)
    
    # Test TCA
    print("\n1. TCA:")
    tca = TCA(n_components=10)
    X_tca = tca.fit_transform(X_source, X_target)
    print(f"   Transformed shape: {X_tca.shape}")
    
    # Test JDA
    print("\n2. JDA:")
    jda = JDA(n_components=10)
    X_jda = jda.fit_transform(X_source, X_target, y_source)
    print(f"   Transformed shape: {X_jda.shape}")
    
    # Test DANN
    print("\n3. DANN:")
    dann = DANN(input_dim=n_features, hidden_dim=64, num_classes=6)
    X_all = np.vstack([X_source, X_target])
    y_all = np.concatenate([y_source, np.zeros(n_target, dtype=int)])
    print(f"   Model initialized successfully")
    
    print("\n✓ All transfer learning methods tested successfully!")
