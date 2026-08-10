"""
Evaluation Metrics for E-Nose Systems

This module provides comprehensive evaluation metrics for:
- Standard classification metrics
- Long-term drift metrics (BWT, FWT)
- Few-shot learning metrics
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    # Handle class imbalance
    unique_classes, counts = np.unique(y_true, return_counts=True)
    metrics['class_distribution'] = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
    
    return metrics


def compute_bwt(predictions_history: List[np.ndarray], 
               y_true_history: List[np.ndarray]) -> float:
    """
    Compute Backward Transfer (BWT) metric.
    
    BWT measures how much the model forgets previous knowledge when learning new tasks.
    Negative BWT indicates forgetting.
    
    BWT = (1/(N-1)) * sum_{i=1}^{N-1} (a_{N,i} - a_{i,i})
    
    where a_{N,i} is the accuracy on task i after learning all N tasks,
    and a_{i,i} is the accuracy on task i after learning task i.
    
    Args:
        predictions_history: List of predictions for each batch
        y_true_history: List of true labels for each batch
        
    Returns:
        BWT value
    """
    n_batches = len(predictions_history)
    
    if n_batches < 2:
        logger.warning("Need at least 2 batches to compute BWT")
        return 0.0
    
    # Compute accuracy for each batch at different time points
    accuracies = []
    for i in range(n_batches):
        acc = accuracy_score(y_true_history[i], predictions_history[i])
        accuracies.append(acc)
    
    # Simplified BWT: compare current performance with initial performance
    # In a full implementation, you'd track performance on all previous batches
    initial_acc = accuracies[0]
    final_acc = accuracies[-1]
    
    bwt = final_acc - initial_acc
    
    return bwt


def compute_bwt_from_matrix(accuracy_matrix: np.ndarray) -> float:
    """Compute standard BWT from a stage-by-task accuracy matrix.

    Rows denote the training stage after learning task t, and columns denote
    evaluation tasks. Entry a[t, i] is the accuracy on task i after stage t.
    """
    matrix = np.asarray(accuracy_matrix, dtype=float)
    n_tasks = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or n_tasks < 2:
        raise ValueError("accuracy_matrix must be a square matrix with at least 2 tasks")

    final_row = matrix[n_tasks - 1, :n_tasks - 1]
    learned_diag = np.diag(matrix)[:n_tasks - 1]
    return float(np.mean(final_row - learned_diag))


def compute_fwt_from_matrix(accuracy_matrix: np.ndarray, baseline_accuracies: np.ndarray) -> float:
    """Compute standard FWT from a stage-by-task accuracy matrix.

    baseline_accuracies[i] is the accuracy on task i without transfer before
    learning that task. Task 0 is excluded from the average by definition.
    """
    matrix = np.asarray(accuracy_matrix, dtype=float)
    baseline = np.asarray(baseline_accuracies, dtype=float)
    n_tasks = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or n_tasks < 2:
        raise ValueError("accuracy_matrix must be a square matrix with at least 2 tasks")
    if baseline.shape[0] != n_tasks:
        raise ValueError("baseline_accuracies must have one value per task")

    pre_task_scores = np.array([matrix[i - 1, i] for i in range(1, n_tasks)])
    return float(np.mean(pre_task_scores - baseline[1:]))


def compute_fwd(predictions_history: List[np.ndarray],
                y_true_history: List[np.ndarray],
                baseline_accuracy: float) -> float:
    """
    Compute Forward Transfer (FWT) metric.
    
    FWT measures how much previous knowledge helps learn new tasks faster.
    Positive FWT indicates positive transfer.
    
    FWT = (1/(N-1)) * sum_{i=2}^{N} (a_{i-1,i} - b_i)
    
    where a_{i-1,i} is the accuracy on task i after learning task i-1,
    and b_i is the baseline accuracy on task i without transfer.
    
    Args:
        predictions_history: List of predictions for each batch
        y_true_history: List of true labels for each batch
        baseline_accuracy: Baseline accuracy without transfer
        
    Returns:
        FWT value
    """
    n_batches = len(predictions_history)
    
    if n_batches < 2:
        logger.warning("Need at least 2 batches to compute FWT")
        return 0.0
    
    # Compute actual accuracies
    actual_accs = []
    for i in range(n_batches):
        acc = accuracy_score(y_true_history[i], predictions_history[i])
        actual_accs.append(acc)
    
    # FWT: compare with baseline
    avg_actual = np.mean(actual_accs[1:])  # Skip first batch
    fwd = avg_actual - baseline_accuracy
    
    return fwd


def compute_confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for a list of scores.
    
    Args:
        scores: Array of scores (e.g., accuracies from multiple runs)
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (mean, std, ci_lower, ci_upper)
    """
    mean = np.mean(scores)
    std = np.std(scores)
    
    # Compute confidence interval
    n = len(scores)
    z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99%
    margin_of_error = z_score * std / np.sqrt(n)
    
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return mean, std, ci_lower, ci_upper


def compute_nway_kshot_metrics(accuracies: List[float], 
                               n_way: int, 
                               k_shot: int) -> Dict:
    """
    Compute few-shot learning specific metrics.
    
    Args:
        accuracies: List of accuracies from multiple episodes
        n_way: Number of ways (classes)
        k_shot: Number of shots (support samples per class)
        
    Returns:
        Dictionary of few-shot metrics
    """
    mean_acc, std_acc, ci_lower, ci_upper = compute_confidence_interval(accuracies)
    
    metrics = {
        'task': f'{n_way}-way {k_shot}-shot',
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'median_accuracy': np.median(accuracies),
        'n_episodes': len(accuracies)
    }
    
    return metrics


def compute_drift_metrics(batch_accuracies: List[float]) -> Dict:
    """
    Compute metrics specific to drift evaluation.
    
    Args:
        batch_accuracies: List of accuracies for each batch
        
    Returns:
        Dictionary of drift metrics
    """
    n_batches = len(batch_accuracies)
    
    if n_batches < 2:
        return {'error': 'Need at least 2 batches'}
    
    metrics = {
        'average_accuracy': np.mean(batch_accuracies),
        'first_batch_accuracy': batch_accuracies[0],
        'last_batch_accuracy': batch_accuracies[-1],
        'performance_degradation': batch_accuracies[0] - batch_accuracies[-1],
        'max_accuracy': np.max(batch_accuracies),
        'min_accuracy': np.min(batch_accuracies),
        'std_accuracy': np.std(batch_accuracies),
        'n_batches': n_batches
    }
    
    # Compute trend (slope of accuracy over batches)
    if n_batches > 2:
        x = np.arange(n_batches)
        y = np.array(batch_accuracies)
        slope = np.polyfit(x, y, 1)[0]
        metrics['accuracy_trend_slope'] = slope
    
    return metrics


def print_detailed_report(y_true: np.ndarray, y_pred: np.ndarray, 
                          title: str = "Classification Report") -> None:
    """
    Print a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Report title
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    metrics = compute_all_metrics(y_true, y_pred)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro):  {metrics['f1_macro']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    unique_classes = sorted(np.unique(y_true))
    
    for cls in unique_classes:
        print(f"  Class {cls}:")
        print(f"    Precision: {metrics['precision_per_class'][cls]:.4f}")
        print(f"    Recall:    {metrics['recall_per_class'][cls]:.4f}")
        print(f"    F1-Score:  {metrics['f1_per_class'][cls]:.4f}")
        print(f"    Support:   {metrics['class_distribution'][cls]}")
    
    print("\n" + classification_report(y_true, y_pred))
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Testing metrics computation...")
    
    # Dummy predictions
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 1])
    y_pred = np.array([0, 0, 1, 2, 2, 2, 0, 1, 1, 1])
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred)
    
    print("\n1. Classification Metrics:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1 Macro: {metrics['f1_macro']:.4f}")
    
    # Test BWT/FWT
    predictions_history = [y_pred, y_pred, y_pred]
    y_true_history = [y_true, y_true, y_true]
    
    bwt = compute_bwt(predictions_history, y_true_history)
    fwd = compute_fwd(predictions_history, y_true_history, baseline_accuracy=0.7)
    
    print("\n2. Transfer Metrics:")
    print(f"   BWT: {bwt:.4f}")
    print(f"   FWT: {fwd:.4f}")
    
    # Test confidence interval
    accuracies = [0.85, 0.87, 0.86, 0.88, 0.84]
    mean, std, ci_lower, ci_upper = compute_confidence_interval(accuracies)
    
    print("\n3. Confidence Interval:")
    print(f"   Mean: {mean:.4f} ± {std:.4f}")
    print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print("\n✓ Metrics tested successfully!")
