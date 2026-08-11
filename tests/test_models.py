"""
Unit Tests for Models

Run with: pytest tests/test_models.py -v
"""

import pytest
import numpy as np
import torch
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFeatureFusion:
    """Test feature fusion methods."""
    
    def test_concatenation_fusion(self):
        """Test concatenation fusion."""
        from models.feature_fusion import ConcatenationFusion
        
        X_list = [np.random.randn(100, 10) for _ in range(4)]
        
        fusion = ConcatenationFusion()
        X_fused = fusion.fit_transform(X_list)
        
        # Check output shape
        assert X_fused.shape == (100, 40)
    
    def test_pca_fusion(self):
        """Test PCA fusion."""
        from models.feature_fusion import PCAFusion
        
        X_list = [np.random.randn(100, 10) for _ in range(4)]
        
        fusion = PCAFusion(n_components=15)
        X_fused = fusion.fit_transform(X_list)
        
        # Check output shape
        assert X_fused.shape == (100, 15)
    
    def test_attention_fusion(self):
        """Test attention fusion."""
        from models.feature_fusion import AttentionFusion
        
        n_sensors = 4
        sensor_dim = 10
        X = torch.randn(32, n_sensors, sensor_dim)
        
        fusion = AttentionFusion(sensor_dim=sensor_dim, n_sensors=n_sensors)
        X_fused, attention_weights = fusion(X)
        
        # Check output shapes
        assert X_fused.shape == (32, sensor_dim)
        assert attention_weights.shape == (32, n_sensors)


class TestTransferLearning:
    """Test transfer learning methods."""
    
    def test_tca_initialization(self):
        """Test TCA initialization."""
        from models.transfer_learning import TCA
        
        tca = TCA(n_components=10, kernel_type='rbf')
        assert tca is not None
        assert tca.n_components == 10
    
    def test_tca_transform(self):
        """Test TCA transformation."""
        from models.transfer_learning import TCA
        
        X_source = np.random.randn(100, 128)
        X_target = np.random.randn(80, 128)
        
        tca = TCA(n_components=10)
        X_transformed = tca.fit_transform(X_source, X_target)
        
        # Check output shape
        assert X_transformed.shape[0] == 180
        assert X_transformed.shape[1] == 10
    
    def test_jda_initialization(self):
        """Test JDA initialization."""
        from models.transfer_learning import JDA
        
        jda = JDA(n_components=10)
        assert jda is not None
    
    def test_dann_initialization(self):
        """Test DANN model initialization."""
        from models.transfer_learning import DANN
        
        dann = DANN(input_dim=128, hidden_dim=64, num_classes=6)
        assert dann is not None
        
        # Check it's a PyTorch module
        assert isinstance(dann, torch.nn.Module)


class TestFewShot:
    """Test few-shot learning methods."""
    
    def test_protonet_initialization(self):
        """Test Prototypical Network initialization."""
        from models.few_shot import PrototypicalNetwork
        
        proto_net = PrototypicalNetwork(
            input_dim=128,
            hidden_dim=64,
            embedding_dim=32
        )
        assert proto_net is not None
    
    def test_protonet_compute_prototypes(self):
        """Test prototype computation."""
        from models.few_shot import PrototypicalNetwork
        
        proto_net = PrototypicalNetwork(input_dim=128, hidden_dim=64, embedding_dim=32)

        X_support = torch.randn(15, 128)  # 5 classes × 3 samples
        y_support = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        
        prototypes = proto_net.compute_prototypes(X_support, y_support, n_way=5)
        
        # Check prototype shape
        assert prototypes.shape == (5, 32)
    
    def test_protonet_forward(self):
        """Test forward pass."""
        from models.few_shot import PrototypicalNetwork
        
        proto_net = PrototypicalNetwork(input_dim=128, hidden_dim=64, embedding_dim=32)
        
        X_query = torch.randn(50, 128)
        prototypes = torch.randn(5, 32)
        
        probs = proto_net(X_query, prototypes)
        
        # Check output shape and probabilities sum to 1
        assert probs.shape == (50, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(50))
    
    def test_create_fewshot_episode(self):
        """Test episode creation."""
        from models.few_shot import create_fewshot_episode
        
        X = np.random.randn(200, 128)
        y = np.random.randint(0, 6, 200)
        
        support_set, query_set = create_fewshot_episode(
            X, y, n_way=5, k_shot=3, n_query=10
        )
        
        X_support, y_support = support_set
        X_query, y_query = query_set
        
        # Check shapes
        assert X_support.shape[0] == 15  # 5 ways × 3 shots
        assert X_query.shape[0] == 50    # 5 ways × 10 queries


class TestDriftCompensation:
    """Test drift compensation methods."""
    
    def test_osc_initialization(self):
        """Test OSC initialization."""
        from models.drift_compensation import OrthogonalSignalCorrection
        
        osc = OrthogonalSignalCorrection(n_components=2)
        assert osc is not None
    
    def test_osc_transform(self):
        """Test OSC transformation."""
        from models.drift_compensation import OrthogonalSignalCorrection
        
        X = np.random.randn(100, 128)
        y = np.random.randint(0, 6, 100)
        
        osc = OrthogonalSignalCorrection(n_components=2)
        X_corrected = osc.fit_transform(X, y)
        
        # Check shape preserved
        assert X_corrected.shape == X.shape
    
    def test_cre_initialization(self):
        """Test Classifier Replacement Ensemble initialization."""
        from models.drift_compensation import ClassifierReplacementEnsemble
        
        cre = ClassifierReplacementEnsemble(
            ensemble_size=5,
            threshold=0.05
        )
        assert cre is not None
    
    def test_cre_predict(self):
        """Test CRE prediction."""
        from models.drift_compensation import ClassifierReplacementEnsemble
        
        X_train = np.random.randn(100, 128)
        y_train = np.random.randint(0, 6, 100)
        X_test = np.random.randn(50, 128)
        
        cre = ClassifierReplacementEnsemble(ensemble_size=3)
        cre.fit_initial(X_train, y_train)
        
        predictions = cre.predict(X_test)
        
        # Check predictions
        assert len(predictions) == 50
        assert all(0 <= p < 6 for p in predictions)
    
    def test_test_time_adaptation(self):
        """Test Test-Time Adaptation."""
        from models.drift_compensation import TestTimeAdaptation
        from models.transfer_learning import DANN
        
        # Create a simple model
        model = DANN(input_dim=128, hidden_dim=32, num_classes=6)
        
        tta = TestTimeAdaptation(model, adaptation_lr=0.001)
        assert tta is not None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_metrics_import(self):
        """Test that metrics can be imported."""
        from utils.metrics import compute_all_metrics, compute_bwt, compute_fwd
        assert compute_all_metrics is not None
        assert compute_bwt is not None
        assert compute_fwd is not None
    
    def test_compute_all_metrics(self):
        """Test metrics computation."""
        from utils.metrics import compute_all_metrics
        
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        metrics = compute_all_metrics(y_true, y_pred)
        
        # Check all metrics present
        assert 'accuracy' in metrics
        assert 'f1_macro' in metrics
        assert metrics['accuracy'] == 1.0
    
    def test_data_utils_import(self):
        """Test data utils can be imported."""
        from utils.data_utils import normalize_data, create_temporal_batches
        assert normalize_data is not None
        assert create_temporal_batches is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
