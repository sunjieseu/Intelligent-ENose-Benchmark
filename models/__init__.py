"""
Intelligent E-Nose Models Package

This package contains implementations of various algorithms for robust E-nose systems:
- Level 1: Feature-level fusion methods
- Level 2: Transfer learning and few-shot learning methods
- Level 3: Adaptive drift compensation methods
"""

from .feature_fusion import (
    ConcatenationFusion,
    PCAFusion,
    AttentionFusion
)

from .transfer_learning import (
    DANN,
    TCA,
    JDA
)

from .few_shot import (
    PrototypicalNetwork,
    MAML,
    RelationNetwork
)

from .drift_compensation import (
    OrthogonalSignalCorrection,
    ClassifierReplacementEnsemble,
    TestTimeAdaptation
)

__all__ = [
    # Feature fusion
    'ConcatenationFusion',
    'PCAFusion',
    'AttentionFusion',
    # Transfer learning
    'DANN',
    'TCA',
    'JDA',
    # Few-shot learning
    'PrototypicalNetwork',
    'MAML',
    'RelationNetwork',
    # Drift compensation
    'OrthogonalSignalCorrection',
    'ClassifierReplacementEnsemble',
    'TestTimeAdaptation',
]
