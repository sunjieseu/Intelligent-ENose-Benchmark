"""
Utility Functions Package
"""

from .metrics import (
    compute_all_metrics,
    compute_bwt,
    compute_fwd,
    accuracy_score
)

from .data_utils import (
    normalize_data,
    split_into_batches,
    create_episodes_for_fewshot
)

__all__ = [
    'compute_all_metrics',
    'compute_bwt',
    'compute_fwd',
    'accuracy_score',
    'normalize_data',
    'split_into_batches',
    'create_episodes_for_fewshot'
]
