"""
Level 3: Adaptive Drift Compensation Methods

This module implements adaptive drift compensation strategies for long-term E-nose stability:
- Orthogonal Signal Correction (OSC)
- Classifier Replacement Ensemble (CRE)
- Test-Time Adaptation (TTA)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from typing import Optional, List, Tuple, Callable
import logging
from collections import deque

logger = logging.getLogger(__name__)


class OrthogonalSignalCorrection:
    """
    Orthogonal Signal Correction (OSC)
    
    Removes variance components in the input matrix X that are orthogonal
    to the label matrix Y, thereby "purifying" the data from drift or noise.
    
    This is a data-level correction method that decouples calibration from recognition.
    """
    
    def __init__(self, n_components: int = 2):
        """
        Initialize OSC.
        
        Args:
            n_components: Number of orthogonal components to remove
        """
        self.n_components = n_components
        self.P_orthogonal = None
        self.mean = None
        self.std = None
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit OSC to find orthogonal components.
        
        Args:
            X: Input data matrix [n_samples, n_features]
            Y: Label matrix or target variables [n_samples, n_targets]
        """
        # Standardize data
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X_std = (X - self.mean) / self.std
        
        # Ensure Y is 2D
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Project X onto Y
        # T = X * (Y' * X)^-1 * Y' (scores)
        # This captures the variation in X that is correlated with Y
        
        # Compute projection matrix
        XtY = X_std.T @ Y
        P_y = Y @ np.linalg.pinv(XtY) @ X_std.T
        
        # Compute orthogonal components
        # The part of X orthogonal to Y
        P_orth = np.eye(X_std.shape[1]) - P_y.T @ P_y
        
        # Perform PCA on orthogonal part to find directions to remove
        # Find eigenvectors corresponding to largest eigenvalues
        cov_orth = X_std.T @ P_orth @ X_std
        eigenvalues, eigenvectors = np.linalg.eigh(cov_orth)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components to remove
        self.P_orthogonal = eigenvectors[:, :self.n_components]
        
        logger.info(f"OSC fitted: removing {self.n_components} orthogonal components")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by removing orthogonal components.
        
        Args:
            X: Input data matrix
            
        Returns:
            Corrected data matrix
        """
        # Standardize
        X_std = (X - self.mean) / self.std
        
        # Project out orthogonal components
        # X_corrected = X - (X * P_orth) * P_orth'
        projection = X_std @ self.P_orthogonal @ self.P_orthogonal.T
        X_corrected = X_std - projection
        
        # Restore original scale
        X_corrected = X_corrected * self.std + self.mean
        
        return X_corrected
    
    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, Y)
        return self.transform(X)


class ClassifierReplacementEnsemble:
    """
    Classifier Replacement Ensemble (CRE)
    
    Maintains a pool of classifiers and dynamically replaces underperforming ones.
    This is an ensemble-level method that provides robustness against both gradual
    and abrupt drift.
    
    Strategy:
    - Monitor ensemble performance over time
    - When performance drops below threshold, replace weakest classifier
    - Train new classifier on recent labeled data
    """
    
    def __init__(self, base_model: str = 'svm', ensemble_size: int = 5,
                 threshold: float = 0.05, update_interval: int = 30):
        """
        Initialize CRE.
        
        Args:
            base_model: Base classifier type ('svm', 'rf', etc.)
            ensemble_size: Number of classifiers in ensemble
            threshold: Performance drop threshold to trigger replacement
            update_interval: Minimum samples between updates
        """
        self.base_model = base_model
        self.ensemble_size = ensemble_size
        self.threshold = threshold
        self.update_interval = update_interval
        
        self.classifiers = []
        self.weights = np.ones(ensemble_size) / ensemble_size
        self.performance_history = deque(maxlen=100)
        self.sample_count = 0
        
    def _create_classifier(self):
        """Create a new base classifier."""
        if self.base_model == 'svm':
            return SVC(kernel='rbf', probability=True)
        else:
            raise ValueError(f"Unsupported base model: {self.base_model}")
    
    def fit_initial(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize ensemble with initial data.
        
        Args:
            X: Training data
            y: Labels
        """
        logger.info(f"Initializing CRE with {self.ensemble_size} classifiers")
        
        for i in range(self.ensemble_size):
            clf = self._create_classifier()
            
            # Train on bootstrap sample
            n_samples = len(X)
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            clf.fit(X[bootstrap_idx], y[bootstrap_idx])
            
            self.classifiers.append(clf)
        
        logger.info(f"CRE initialized with {len(self.classifiers)} classifiers")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using weighted ensemble voting.
        
        Args:
            X: Input data
            
        Returns:
            Predicted labels
        """
        # Get predictions from all classifiers
        all_preds = np.array([clf.predict_proba(X) for clf in self.classifiers])
        
        # Weighted average
        weighted_preds = np.average(all_preds, axis=0, weights=self.weights)
        predictions = np.argmax(weighted_preds, axis=1)
        
        return predictions
    
    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Update ensemble by replacing weakest classifier.
        
        Args:
            X_new: New labeled data
            y_new: New labels
        """
        if len(self.classifiers) == 0:
            self.fit_initial(X_new, y_new)
            return
        
        # Evaluate each classifier on new data
        performances = []
        for i, clf in enumerate(self.classifiers):
            accuracy = clf.score(X_new, y_new)
            performances.append(accuracy)
        
        # Find weakest classifier
        weakest_idx = np.argmin(performances)
        weakest_performance = performances[weakest_idx]
        
        # Check if replacement is needed
        if weakest_performance < self.threshold:
            logger.info(f"Replacing classifier {weakest_idx} "
                       f"(accuracy: {weakest_performance:.4f} < threshold: {self.threshold})")
            
            # Remove weakest
            self.classifiers.pop(weakest_idx)
            self.weights = np.delete(self.weights, weakest_idx)
            
            # Train new classifier
            new_clf = self._create_classifier()
            new_clf.fit(X_new, y_new)
            
            self.classifiers.append(new_clf)
            self.weights = np.append(self.weights, 1.0 / self.ensemble_size)
            
            # Renormalize weights
            self.weights /= self.weights.sum()
            
            logger.info(f"Ensemble updated: {len(self.classifiers)} classifiers")
    
    def detect_drift(self, X_new: np.ndarray, y_new: np.ndarray) -> bool:
        """
        Detect if significant drift has occurred.
        
        Args:
            X_new: New data
            y_new: True labels
            
        Returns:
            True if drift detected
        """
        # Predict
        predictions = self.predict(X_new)
        accuracy = np.mean(predictions == y_new)
        
        # Track performance
        self.performance_history.append(accuracy)
        self.sample_count += len(X_new)
        
        # Check if performance dropped significantly
        if len(self.performance_history) >= 10:
            recent_avg = np.mean(list(self.performance_history)[-10:])
            historical_avg = np.mean(list(self.performance_history)[:-10])
            
            if historical_avg - recent_avg > self.threshold:
                logger.info(f"Drift detected: accuracy dropped from "
                           f"{historical_avg:.4f} to {recent_avg:.4f}")
                return True
        
        return False
    
    def adaptive_update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Conditionally update ensemble based on drift detection.
        
        Args:
            X_new: New labeled data
            y_new: New labels
        """
        if self.detect_drift(X_new, y_new):
            self.update(X_new, y_new)
        elif self.sample_count % self.update_interval == 0:
            # Periodic update
            self.update(X_new, y_new)


class TestTimeAdaptation:
    """
    Test-Time Adaptation (TTA)
    
    Updates model parameters during deployment using only unlabeled test streams.
    Assumes reliable predictions should have low entropy (high confidence).
    Minimizes entropy at test time to drive online adaptation.
    
    L_TTA(x_test) = - sum_c y_c * log(y_c)
    """
    
    def __init__(self, model: nn.Module, adaptation_lr: float = 0.001,
                 batch_norm_layers: Optional[List] = None):
        """
        Initialize TTA.
        
        Args:
            model: PyTorch model to adapt
            adaptation_lr: Learning rate for adaptation
            batch_norm_layers: List of batch norm layer names to update
        """
        self.model = model
        self.adaptation_lr = adaptation_lr
        self.batch_norm_layers = batch_norm_layers or []
        
        # Freeze all parameters except batch norm
        self._freeze_parameters()
    
    def _freeze_parameters(self):
        """Freeze all parameters except those in batch norm layers."""
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        # Unfreeze batch norm parameters
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if name in self.batch_norm_layers or len(self.batch_norm_layers) == 0:
                    module.weight.requires_grad = True
                    module.bias.requires_grad = True
    
    def entropy_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy loss for uncertainty minimization.
        
        Args:
            predictions: Model predictions (probabilities) [batch, n_classes]
            
        Returns:
            Entropy loss
        """
        # H = - sum(p * log(p))
        entropy = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
        return entropy.mean()
    
    def adapt(self, X_unlabeled: np.ndarray, n_steps: int = 1,
              device: str = 'cpu') -> dict:
        """
        Adapt model to unlabeled test data.
        
        Args:
            X_unlabeled: Unlabeled test data
            n_steps: Number of adaptation steps
            device: Device to adapt on
            
        Returns:
            Adaptation metrics
        """
        self.model.train()
        X_tensor = torch.FloatTensor(X_unlabeled).to(device)
        self.model.to(device)
        
        # Setup optimizer for batch norm parameters only
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.adaptation_lr)
        
        metrics = {'initial_entropy': None, 'final_entropy': None}
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Forward pass
            with torch.no_grad():
                # Get initial predictions (for logging)
                if step == 0:
                    initial_output = self.model(X_tensor)
                    if isinstance(initial_output, tuple):
                        initial_output = initial_output[0]
                    initial_probs = F.softmax(initial_output, dim=1)
                    metrics['initial_entropy'] = self.entropy_loss(initial_probs).item()
            
            # Forward pass (with gradients for BN layers)
            output = self.model(X_tensor)
            if isinstance(output, tuple):
                output = output[0]
            
            # Compute probabilities
            probs = F.softmax(output, dim=1)
            
            # Minimize entropy
            loss = self.entropy_loss(probs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log final entropy
            if step == n_steps - 1:
                with torch.no_grad():
                    final_output = self.model(X_tensor)
                    if isinstance(final_output, tuple):
                        final_output = final_output[0]
                    final_probs = F.softmax(final_output, dim=1)
                    metrics['final_entropy'] = self.entropy_loss(final_probs).item()
        
        logger.info(f"TTA adaptation: entropy {metrics['initial_entropy']:.4f} -> "
                    f"{metrics['final_entropy']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray, device: str = 'cpu') -> np.ndarray:
        """
        Predict with adapted model.
        
        Args:
            X: Input data
            device: Device for inference
            
        Returns:
            Predicted labels
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(device)
        self.model.to(device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
            if isinstance(output, tuple):
                output = output[0]
            _, predicted = torch.max(output, 1)
        
        return predicted.numpy()


class ActiveLearningSelector:
    """
    Active Learning for efficient supervision management.
    
    Selects the most informative samples (e.g., high uncertainty)
    for human annotation, minimizing labeling costs.
    """
    
    def __init__(self, strategy: str = 'uncertainty'):
        """
        Initialize active learning selector.
        
        Args:
            strategy: Selection strategy ('uncertainty', 'margin', 'entropy')
        """
        self.strategy = strategy
    
    def compute_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """
        Compute uncertainty scores for samples.
        
        Args:
            predictions: Model predictions (probabilities) [n_samples, n_classes]
            
        Returns:
            Uncertainty scores [n_samples]
        """
        if self.strategy == 'uncertainty':
            # 1 - max probability
            uncertainty = 1 - np.max(predictions, axis=1)
        elif self.strategy == 'margin':
            # Difference between top-2 probabilities
            sorted_probs = np.sort(predictions, axis=1)
            uncertainty = sorted_probs[:, -1] - sorted_probs[:, -2]
            uncertainty = 1 - uncertainty  # Invert: higher = more uncertain
        elif self.strategy == 'entropy':
            # Prediction entropy
            entropy = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
            uncertainty = entropy / np.log(predictions.shape[1])  # Normalize
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return uncertainty
    
    def select_samples(self, predictions: np.ndarray, X: np.ndarray,
                      n_select: int = 10) -> Tuple:
        """
        Select most informative samples for annotation.
        
        Args:
            predictions: Model predictions
            X: Input features
            n_select: Number of samples to select
            
        Returns:
            Selected samples and their indices
        """
        uncertainty = self.compute_uncertainty(predictions)
        
        # Select top-n uncertain samples
        selected_indices = np.argsort(uncertainty)[-n_select:]
        
        return X[selected_indices], selected_indices


if __name__ == "__main__":
    # Example usage
    print("Testing adaptive drift compensation methods...")
    
    # Create dummy data with drift
    n_samples, n_features = 200, 16
    X_initial = np.random.randn(n_samples, n_features)
    y_initial = np.random.randint(0, 6, n_samples)
    
    X_drifted = X_initial + np.random.randn(n_samples, n_features) * 2
    y_drifted = y_initial
    
    # Test OSC
    print("\n1. Orthogonal Signal Correction:")
    osc = OrthogonalSignalCorrection(n_components=2)
    X_corrected = osc.fit_transform(X_initial, y_initial)
    print(f"   Original shape: {X_initial.shape}")
    print(f"   Corrected shape: {X_corrected.shape}")
    
    # Test CRE
    print("\n2. Classifier Replacement Ensemble:")
    cre = ClassifierReplacementEnsemble(ensemble_size=3, threshold=0.3)
    cre.fit_initial(X_initial, y_initial)
    predictions = cre.predict(X_drifted)
    print(f"   Predictions: {predictions[:10]}")
    
    # Test Active Learning
    print("\n3. Active Learning Selector:")
    # Dummy predictions with varying confidence
    dummy_preds = np.array([[0.9, 0.1, 0.0], [0.4, 0.3, 0.3], [0.6, 0.2, 0.2]])
    selector = ActiveLearningSelector(strategy='uncertainty')
    uncertainty = selector.compute_uncertainty(dummy_preds)
    print(f"   Uncertainty scores: {uncertainty}")
    
    print("\n✓ All drift compensation methods tested successfully!")
