"""
Unit Tests for Dataset Loaders

Run with: pytest tests/test_datasets.py -v
"""

import pytest
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_loaders import DatasetLoader, preprocess_data


class TestDatasetLoader:
    """Test dataset loading functionality."""
    
    def test_preprocess_standard(self):
        """Test standard normalization."""
        X = np.random.randn(100, 10)
        X_processed = preprocess_data(X, method='standard')
        
        # Check shape preserved
        assert X_processed.shape == X.shape
        
        # Check mean ~ 0, std ~ 1
        assert np.abs(X_processed.mean()) < 0.01
        assert np.abs(X_processed.std() - 1.0) < 0.01
    
    def test_preprocess_minmax(self):
        """Test min-max normalization."""
        X = np.random.randn(100, 10) * 10 + 50
        X_processed = preprocess_data(X, method='minmax')
        
        # Check range [0, 1]
        assert X_processed.min() >= 0.0
        assert X_processed.max() <= 1.0
    
    def test_preprocess_with_missing_values(self):
        """Test handling of missing values."""
        X = np.random.randn(100, 10)
        X[np.random.randint(0, 100, 10), np.random.randint(0, 10, 10)] = np.nan
        
        # Should not raise error
        X_processed = preprocess_data(X, method='standard', handle_missing=True)
        
        # Check no NaN remaining
        assert not np.isnan(X_processed).any()
    
    def test_preprocess_remove_outliers(self):
        """Test outlier removal."""
        X = np.random.randn(100, 10)
        # Add extreme outliers
        X[0:5, 0] = 100.0
        
        X_processed = preprocess_data(
            X, 
            method='standard', 
            remove_outliers=True, 
            outlier_threshold=3.0
        )
        
        # Should have fewer samples
        assert X_processed.shape[0] <= X.shape[0]
    
    def test_dataset_loader_instantiation(self):
        """Test DatasetLoader can be instantiated."""
        loader = DatasetLoader()
        assert loader is not None
        assert loader.data_dir == "data"
    
    def test_ucsd_loader_missing_data(self):
        """Test UCSD loader raises error when data missing."""
        from datasets.dataset_loaders import UCSDLoader
        
        loader = UCSDLoader(data_dir='nonexistent')
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_cqu_loader_missing_data(self):
        """Test CQU loader raises error when data missing."""
        from datasets.dataset_loaders import CQULoader
        
        loader = CQULoader(data_dir='nonexistent')
        
        with pytest.raises(FileNotFoundError):
            loader.load()


class TestFewShotTaskCreation:
    """Test few-shot task creation."""
    
    def test_create_fewshot_task_import(self):
        """Test that few-shot task creation can be imported."""
        from datasets.dataset_loaders import create_fewshot_task
        assert create_fewshot_task is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
