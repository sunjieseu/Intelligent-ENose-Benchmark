"""
Data Preprocessing Utilities

This module provides utilities for:
- Data normalization and preprocessing
- Batch splitting for drift evaluation
- Episode creation for few-shot learning
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def normalize_data(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, object]:
    """
    Normalize data using various methods.
    
    Args:
        X: Input data [n_samples, n_features]
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, scaler


def apply_scaler(X: np.ndarray, scaler) -> np.ndarray:
    """Apply a fitted scaler to new data."""
    return scaler.transform(X)


def split_into_batches(X: np.ndarray, y: np.ndarray, 
                       batch_indices: np.ndarray,
                       batch_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into a specific batch.
    
    Args:
        X: All features
        y: All labels
        batch_indices: Batch assignment for each sample
        batch_id: ID of batch to extract
        
    Returns:
        Tuple of (X_batch, y_batch)
    """
    mask = (batch_indices == batch_id)
    return X[mask], y[mask]


def create_temporal_batches(X: np.ndarray, y: np.ndarray, 
                           n_batches: int = 10) -> List[Tuple]:
    """
    Split data into temporal batches (simulating drift).
    
    Args:
        X: Features
        y: Labels
        n_batches: Number of batches
        
    Returns:
        List of (X_batch, y_batch) tuples
    """
    n_samples = len(X)
    batch_size = n_samples // n_batches
    
    batches = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < n_batches - 1 else n_samples
        
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        batches.append((X_batch, y_batch))
    
    return batches


def add_synthetic_drift(X: np.ndarray, drift_magnitude: float = 0.1,
                       drift_direction: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Add synthetic drift to data for testing.
    
    Args:
        X: Original data
        drift_magnitude: Magnitude of drift
        drift_direction: Direction of drift (random if None)
        
    Returns:
        Data with added drift
    """
    if drift_direction is None:
        drift_direction = np.random.randn(X.shape[1])
        drift_direction = drift_direction / np.linalg.norm(drift_direction)
    
    # Add linear drift
    n_samples = X.shape[0]
    drift = np.linspace(0, drift_magnitude, n_samples).reshape(-1, 1) * drift_direction
    
    X_drifted = X + drift
    
    return X_drifted


def create_cross_validation_splits(X: np.ndarray, y: np.ndarray,
                                   n_folds: int = 5, 
                                   stratified: bool = True) -> List[Tuple]:
    """
    Create cross-validation splits.
    
    Args:
        X: Features
        y: Labels
        n_folds: Number of folds
        stratified: Whether to stratify by class
        
    Returns:
        List of (X_train, y_train, X_test, y_test) tuples
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    splits = []
    
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        splits.append((X_train, y_train, X_test, y_test))
    
    return splits


def balance_dataset(X: np.ndarray, y: np.ndarray, 
                   sampling_strategy: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset by class.
    
    Args:
        X: Features
        y: Labels
        sampling_strategy: 'undersample' or 'oversample'
        
    Returns:
        Balanced (X, y)
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    target_count = np.min(class_counts) if sampling_strategy == 'undersample' else np.max(class_counts)
    
    X_balanced = []
    y_balanced = []
    
    for cls in unique_classes:
        cls_mask = (y == cls)
        X_cls = X[cls_mask]
        y_cls = y[cls_mask]
        
        if sampling_strategy == 'undersample':
            # Random undersampling
            indices = np.random.choice(len(X_cls), target_count, replace=False)
        elif sampling_strategy == 'oversample':
            # Random oversampling
            indices = np.random.choice(len(X_cls), target_count, replace=True)
        else:
            raise ValueError(f"Unknown strategy: {sampling_strategy}")
        
        X_balanced.append(X_cls[indices])
        y_balanced.append(y_cls[indices])
    
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.concatenate(y_balanced)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    
    logger.info(f"Dataset balanced: {len(X_balanced)} samples "
                f"(was {len(X)} with min class size {np.min(class_counts)})")
    
    return X_balanced, y_balanced


def augment_timeseries(X: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Augment time series data with noise.
    
    Args:
        X: Time series data
        noise_level: Standard deviation of noise
        
    Returns:
        Augmented data
    """
    noise = np.random.randn(*X.shape) * noise_level
    X_augmented = X + noise
    
    return X_augmented


def create_episodes_for_fewshot(X: np.ndarray, y: np.ndarray,
                                n_way: int, k_shot: int, 
                                n_query: int = 15,
                                n_episodes: int = 100) -> List[Tuple]:
    """
    Create multiple few-shot learning episodes.
    
    Args:
        X: Dataset features
        y: Dataset labels
        n_way: Number of classes per episode
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        n_episodes: Number of episodes to create
        
    Returns:
        List of ((X_support, y_support), (X_query, y_query)) tuples
    """
    episodes = []
    
    unique_classes = np.unique(y)
    if len(unique_classes) < n_way:
        raise ValueError(f"Not enough classes: have {len(unique_classes)}, need {n_way}")
    
    for _ in range(n_episodes):
        # Select n_way classes
        selected_classes = np.random.choice(unique_classes, n_way, replace=False)
        
        support_samples = []
        query_samples = []
        
        for cls in selected_classes:
            cls_indices = np.where(y == cls)[0]
            n_available = len(cls_indices)
            
            if n_available < k_shot + n_query:
                continue
            
            # Split into support and query
            shuffled_indices = np.random.permutation(cls_indices)
            support_idx = shuffled_indices[:k_shot]
            query_idx = shuffled_indices[k_shot:k_shot + n_query]
            
            support_samples.append((X[support_idx], y[support_idx]))
            query_samples.append((X[query_idx], y[query_idx]))
        
        if len(support_samples) == n_way:
            # Combine samples
            X_support = np.concatenate([s[0] for s in support_samples])
            y_support = np.concatenate([s[1] for s in support_samples])
            X_query = np.concatenate([q[0] for q in query_samples])
            y_query = np.concatenate([q[1] for q in query_samples])
            
            episodes.append(((X_support, y_support), (X_query, y_query)))
    
    logger.info(f"Created {len(episodes)} few-shot episodes")
    
    return episodes


if __name__ == "__main__":
    # Example usage
    print("Testing data utilities...")
    
    # Test normalization
    X = np.random.randn(100, 10)
    
    print("\n1. Normalization:")
    X_norm, scaler = normalize_data(X, method='standard')
    print(f"   Original mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"   Normalized mean: {X_norm.mean():.4f}, std: {X_norm.std():.4f}")
    
    # Test temporal batches
    print("\n2. Temporal Batches:")
    y = np.random.randint(0, 6, 100)
    batches = create_temporal_batches(X, y, n_batches=5)
    print(f"   Created {len(batches)} batches")
    for i, (X_batch, y_batch) in enumerate(batches):
        print(f"   Batch {i}: {X_batch.shape[0]} samples")
    
    # Test synthetic drift
    print("\n3. Synthetic Drift:")
    X_drifted = add_synthetic_drift(X, drift_magnitude=0.5)
    print(f"   Original range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"   Drifted range: [{X_drifted.min():.4f}, {X_drifted.max():.4f}]")
    
    # Test episode creation
    print("\n4. Few-Shot Episodes:")
    episodes = create_episodes_for_fewshot(X, y, n_way=3, k_shot=2, n_episodes=5)
    print(f"   Created {len(episodes)} episodes")
    if len(episodes) > 0:
        (X_s, y_s), (X_q, y_q) = episodes[0]
        print(f"   Episode 0: Support {X_s.shape}, Query {X_q.shape}")
    
    print("\n✓ Data utilities tested successfully!")
