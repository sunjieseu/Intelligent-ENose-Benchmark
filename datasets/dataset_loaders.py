"""
Unified Dataset Loader for Intelligent E-Nose Benchmark

This module provides a consistent interface for loading and preprocessing
various E-nose datasets including UCSD, CQU, and others.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Base class for dataset loading and preprocessing."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
    def load(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
        
    def preprocess(self, X: np.ndarray, method: str = 'standard', 
                   handle_missing: bool = True, remove_outliers: bool = False,
                   outlier_threshold: float = 3.0) -> np.ndarray:
        """
        Preprocess data with various normalization methods.
        
        Args:
            X: Raw data matrix
            method: Normalization method ('standard', 'minmax', 'robust')
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            Preprocessed data matrix
        """
        X_processed = X.copy()
        
        # Handle missing values
        if handle_missing:
            if np.isnan(X_processed).any():
                logger.info("Handling missing values with median imputation")
                median = np.nanmedian(X_processed, axis=0)
                X_processed = np.where(np.isnan(X_processed), median, X_processed)
        
        # Remove outliers
        if remove_outliers:
            z_scores = np.abs((X_processed - X_processed.mean(axis=0)) / X_processed.std(axis=0))
            outlier_mask = (z_scores > outlier_threshold).any(axis=1)
            if outlier_mask.any():
                logger.info(f"Removing {outlier_mask.sum()} outliers")
                X_processed = X_processed[~outlier_mask]
        
        # Normalize
        if method == 'standard':
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            X_processed = scaler.fit_transform(X_processed)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return X_processed


class UCSDLoader(DatasetLoader):
    """Loader for UCSD Gas Sensor Array Drift Dataset."""
    
    def load(self, 
             source_batches: List[int] = [1, 2],
             target_batches: List[int] = [3, 4, 5],
             feature_type: str = 'standard',
             normalize: bool = True,
             normalize_method: str = 'standard') -> Tuple:
        """
        Load UCSD dataset with specified source and target batches.

        The default protocol extracts 8 features per sensor (16 sensors),
        yielding the standard 128-dimensional feature vector as described in
        Vergara et al. (2012) and used throughout the published UCSD drift
        literature. A legacy 16-dimensional variant using raw steady-state
        readings only is available via feature_type='steady_state'.

        Args:
            source_batches: List of batch numbers for source domain
            target_batches: List of batch numbers for target domain
            feature_type: Feature extraction mode ('standard' for 128-dim,
                'steady_state' for legacy 16-dim, 'raw' for raw readings)
            normalize: Whether to normalize data
            normalize_method: Normalization method
            
        Returns:
            Tuple of (X_source, y_source, X_target, y_target)
        """
        data_path = os.path.join(self.data_dir, 'ucsd', 'Datos3096.txt')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"UCSD dataset not found at {data_path}. "
                f"Please run datasets/download_ucsd.sh first."
            )
        
        logger.info(f"Loading UCSD dataset from {data_path}")
        
        # Load raw data
        df = pd.read_csv(data_path, sep='\t', header=None)
        df.columns = [f'sensor_{i}' for i in range(16)] + ['batch', 'class']
        
        if feature_type == 'raw':
            feature_cols = [f'sensor_{i}' for i in range(16)]
            X = df[feature_cols].values
            logger.info("Using raw 16-dimensional sensor readings")
        elif feature_type == 'steady_state':
            feature_cols = [f'sensor_{i}' for i in range(16)]
            X = df[feature_cols].values
            logger.info("Using legacy 16-dimensional steady-state features")
        else:
            X = self._extract_128d_features(df)
            logger.info("Using standard 128-dimensional feature protocol")
        
        y = df['class'].values - 1  # Convert to 0-indexed
        batch_nums = df['batch'].values
        
        # Split into source and target domains
        source_mask = np.isin(batch_nums, source_batches)
        target_mask = np.isin(batch_nums, target_batches)
        
        X_source = X[source_mask]
        y_source = y[source_mask]
        X_target = X[target_mask]
        y_target = y[target_mask]
        
        # Normalize
        if normalize:
            logger.info(f"Normalizing data with {normalize_method} method")
            X_source = self.preprocess(X_source, method=normalize_method)
            X_target = self.preprocess(X_target, method=normalize_method)
        
        logger.info(f"Source domain: {X_source.shape[0]} samples, "
                    f"Target domain: {X_target.shape[0]} samples")
        logger.info(f"Feature dimensionality: {X_source.shape[1]}")
        logger.info(f"Number of classes: {len(np.unique(y_target))}")
        
        return X_source, y_source, X_target, y_target
    
    @staticmethod
    def _extract_128d_features(df: pd.DataFrame) -> np.ndarray:
        """Extract 8 standard UCSD features per sensor (16 sensors = 128 dims).
        
        Features per sensor (Vergara et al., 2012):
          0  max_response        -- maximum sensor response
          1  rel_max_response    -- relative max response (to baseline)
          2  slope              -- average slope of response
          3  integral           -- integral of response curve
          4  time_to_max        -- time to reach maximum response
          5  steady_state       -- steady-state response value
          6  recovery_slope     -- slope during recovery phase
          7  area_ratio         -- ratio of integral to max response
        """
        n_samples = len(df)
        n_sensors = 16
        features = np.zeros((n_samples, n_sensors * 8))
        
        for s in range(n_sensors):
            col = df.columns[s]
            sensor_vals = df[col].values.astype(float)
            
            # Find baseline (first non-NaN value or median of initial segment)
            baseline = np.nanmedian(sensor_vals[:50]) if np.isnan(sensor_vals[:50]).any() else np.median(sensor_vals[:50])
            
            # Max response
            max_resp = np.max(sensor_vals) - baseline
            
            # Relative max response
            rel_max = max_resp / (abs(baseline) + 1e-10)
            
            # Average slope
            valid = sensor_vals[np.isfinite(sensor_vals)]
            if len(valid) > 1:
                slope = np.mean(np.diff(valid))
            else:
                slope = 0.0
            
            # Integral (trapezoidal approximation)
            integral = np.trapz(valid, dx=1.0) if len(valid) > 1 else 0.0
            
            # Time to max (index of max response normalized)
            max_idx = np.argmax(sensor_vals) if len(sensor_vals) > 0 else 0
            time_to_max = max_idx / max(len(sensor_vals) - 1, 1)
            
            # Steady-state value (mean of last 10% of readings)
            n_tail = max(int(len(sensor_vals) * 0.1), 1)
            steady_state = np.mean(sensor_vals[-n_tail:]) - baseline if len(sensor_vals) >= n_tail else max_resp
            
            # Recovery slope (slope of last 20% of readings)
            n_rec = max(int(len(sensor_vals) * 0.2), 2)
            if len(sensor_vals) >= n_rec:
                recovery_slope = np.mean(np.diff(sensor_vals[-n_rec:]))
            else:
                recovery_slope = 0.0
            
            # Area ratio
            area_ratio = integral / (abs(max_resp) + 1e-10) if abs(max_resp) > 1e-10 else 0.0
            
            start_idx = s * 8
            features[:, start_idx:start_idx + 8] = np.array([
                max_resp, rel_max, slope, integral,
                time_to_max, steady_state, recovery_slope, area_ratio
            ])
        
        return features


class CQULoader(DatasetLoader):
    """Loader for CQU E-Nose Drift Dataset."""
    
    def load(self,
             source_batch: int = 1,
             target_batch: int = 2,
             feature_type: str = 'steady_state',
             normalize: bool = True,
             normalize_method: str = 'standard') -> Tuple:
        """
        Load CQU dataset with specified source and target batches.
        
        Args:
            source_batch: Source batch number
            target_batch: Target batch number
            feature_type: Type of features
            normalize: Whether to normalize data
            normalize_method: Normalization method
            
        Returns:
            Tuple of (X_source, y_source, X_target, y_target)
        """
        data_dir = os.path.join(self.data_dir, 'cqu')
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"CQU dataset directory not found at {data_dir}. "
                f"Please run datasets/download_cqu.sh first."
            )
        
        # Load batch data
        source_path = os.path.join(data_dir, f'batch{source_batch}.csv')
        target_path = os.path.join(data_dir, f'batch{target_batch}.csv')
        
        if not os.path.exists(source_path) or not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Batch files not found. Expected: {source_path}, {target_path}"
            )
        
        logger.info(f"Loading CQU dataset: source={source_batch}, target={target_batch}")
        
        df_source = pd.read_csv(source_path)
        df_target = pd.read_csv(target_path)
        
        # Extract features and labels
        feature_cols = [col for col in df_source.columns if col.startswith('sensor_')]
        X_source = df_source[feature_cols].values
        y_source = df_source['class'].values
        
        X_target = df_target[feature_cols].values
        y_target = df_target['class'].values
        
        # Normalize
        if normalize:
            logger.info(f"Normalizing data with {normalize_method} method")
            X_source = self.preprocess(X_source, method=normalize_method)
            X_target = self.preprocess(X_target, method=normalize_method)
        
        logger.info(f"Source domain: {X_source.shape[0]} samples, "
                    f"Target domain: {X_target.shape[0]} samples")
        
        return X_source, y_source, X_target, y_target


def load_ucsd(*args, **kwargs) -> Tuple:
    """Convenience function to load UCSD dataset."""
    loader = UCSDLoader()
    return loader.load(*args, **kwargs)


def load_cqu(*args, **kwargs) -> Tuple:
    """Convenience function to load CQU dataset."""
    loader = CQULoader()
    return loader.load(*args, **kwargs)


def preprocess_data(X: np.ndarray, method: str = 'standard',
                    handle_missing: bool = True, remove_outliers: bool = False,
                    outlier_threshold: float = 3.0) -> np.ndarray:
    """
    Convenience function for data preprocessing.
    
    Args:
        X: Raw data matrix
        method: Normalization method
        handle_missing: Whether to handle missing values
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold
        
    Returns:
        Preprocessed data matrix
    """
    loader = DatasetLoader()
    return loader.preprocess(X, method=method, handle_missing=handle_missing,
                            remove_outliers=remove_outliers, 
                            outlier_threshold=outlier_threshold)


def create_fewshot_task(dataset_name: str,
                       n_way: int = 5,
                       k_shot: int = 3,
                       n_query: int = 10,
                       **dataset_kwargs) -> Tuple:
    """
    Create an N-way K-shot few-shot learning task.
    
    Args:
        dataset_name: Name of dataset ('ucsd' or 'cqu')
        n_way: Number of classes
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        **dataset_kwargs: Additional arguments for dataset loader
        
    Returns:
        Tuple of (support_set, query_set) where each is (X, y)
    """
    # Load dataset
    if dataset_name == 'ucsd':
        X_source, y_source, X_target, y_target = load_ucsd(**dataset_kwargs)
    elif dataset_name == 'cqu':
        X_source, y_source, X_target, y_target = load_cqu(**dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Use target domain for few-shot task
    X = X_target
    y = y_target
    
    # Get unique classes
    unique_classes = np.unique(y)
    if len(unique_classes) < n_way:
        raise ValueError(f"Dataset has {len(unique_classes)} classes, "
                        f"but requested {n_way}-way task")
    
    # Randomly select n_way classes
    selected_classes = np.random.choice(unique_classes, n_way, replace=False)
    
    support_samples = []
    query_samples = []
    
    for cls in selected_classes:
        cls_mask = (y == cls)
        cls_X = X[cls_mask]
        cls_y = y[cls_mask]
        
        # Randomly split into support and query
        n_total = len(cls_X)
        n_support = k_shot
        n_query_actual = min(n_query, n_total - n_support)
        
        indices = np.random.permutation(n_total)
        support_indices = indices[:n_support]
        query_indices = indices[n_support:n_support + n_query_actual]
        
        support_samples.append((cls_X[support_indices], cls_y[support_indices]))
        query_samples.append((cls_X[query_indices], cls_y[query_indices]))
    
    # Combine samples
    X_support = np.concatenate([s[0] for s in support_samples])
    y_support = np.concatenate([s[1] for s in support_samples])
    X_query = np.concatenate([q[0] for q in query_samples])
    y_query = np.concatenate([q[1] for q in query_samples])
    
    support_set = (X_support, y_support)
    query_set = (X_query, y_query)
    
    logger.info(f"Created {n_way}-way {k_shot}-shot task")
    logger(f"Support set: {X_support.shape[0]} samples, "
           f"Query set: {X_query.shape[0]} samples")
    
    return support_set, query_set


if __name__ == "__main__":
    # Example usage
    print("Testing UCSD dataset loader...")
    try:
        X_source, y_source, X_target, y_target = load_ucsd(
            source_batches=[1, 2],
            target_batches=[3, 4, 5],
            normalize=True
        )
        print(f"✓ UCSD loaded successfully")
        print(f"  Source: {X_source.shape}, Target: {X_target.shape}")
    except Exception as e:
        print(f"✗ UCSD loading failed: {e}")
    
    print("\nTesting few-shot task creation...")
    try:
        support_set, query_set = create_fewshot_task(
            dataset_name='ucsd',
            n_way=5,
            k_shot=3,
            n_query=10,
            source_batches=[1],
            target_batches=[2]
        )
        print(f"✓ Few-shot task created successfully")
        print(f"  Support: {support_set[0].shape}, Query: {query_set[0].shape}")
    except Exception as e:
        print(f"✗ Few-shot task creation failed: {e}")
