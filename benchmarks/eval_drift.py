#!/usr/bin/env python3
"""
Benchmark Script: Evaluate Drift Compensation Performance

This script evaluates algorithms under sensor drift scenarios,
computing metrics like FWT (Forward Transfer), BWT (Backward Transfer),
and average accuracy across batches.

Usage:
    python benchmarks/eval_drift.py --config configs/ucisd_drift.yaml
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import logging
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_loaders import load_ucsd
from models.transfer_learning import DANN, TCA, JDA
from models.drift_compensation import ClassifierReplacementEnsemble, OrthogonalSignalCorrection
from utils.metrics import compute_bwt, compute_fwd, compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_sequential_batches(config):
    """Evaluate model performance across sequential batches (simulating drift)."""
    logger.info("=" * 60)
    logger.info("Evaluating Drift Compensation Performance")
    logger.info("=" * 60)
    
    dataset_config = config['dataset']
    n_batches = dataset_config.get('n_batches', 10)
    
    # Load all batches sequentially
    logger.info(f"Loading {n_batches} batches from UCSD dataset")
    
    # Store results for each batch
    batch_accuracies = []
    batch_models = []
    
    # Reference model (trained on batch 1, no updates)
    reference_accs = []
    
    # Adaptive model (with drift compensation)
    adaptive_accs = []
    
    for batch_idx in range(1, n_batches + 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating Batch {batch_idx}/{n_batches}")
        logger.info(f"{'='*40}")
        
        # Load current batch as target domain
        # In practice, you'd load each batch separately
        X_source, y_source, X_target, y_target = load_ucsd(
            source_batches=[1],
            target_batches=[batch_idx],
            normalize=True
        )
        
        # Baseline: Static model trained on batch 1
        if batch_idx == 1:
            # Train baseline model
            baseline_model = SVC(kernel='rbf', probability=True)
            baseline_model.fit(X_source, y_source)
            logger.info("Baseline model trained on batch 1")
        
        # Evaluate baseline
        baseline_pred = baseline_model.predict(X_target)
        baseline_acc = accuracy_score(y_target, baseline_pred)
        reference_accs.append(baseline_acc)
        
        logger.info(f"Baseline accuracy (no adaptation): {baseline_acc:.4f}")
        
        # Method 1: Transfer Learning (DANN)
        if config.get('methods', {}).get('dann', False):
            logger.info("Applying DANN domain adaptation...")
            dann = DANN(
                input_dim=X_source.shape[1],
                hidden_dim=128,
                num_classes=len(np.unique(y_target)),
                alpha=1.0
            )
            
            # Train DANN
            dann.fit(X_source, y_source, X_target, epochs=50, batch_size=32)
            
            dann_pred = dann.predict(X_target)
            dann_acc = accuracy_score(y_target, dann_pred)
            logger.info(f"DANN accuracy: {dann_acc:.4f}")
        
        # Method 2: TCA + Classifier
        if config.get('methods', {}).get('tca', False):
            logger.info("Applying TCA + SVM...")
            tca = TCA(n_components=20)
            X_combined = np.vstack([X_source, X_target])
            X_transformed = tca.fit_transform(X_source, X_target)
            
            X_source_t = X_transformed[:len(X_source)]
            X_target_t = X_transformed[len(X_source):]
            
            tca_svm = SVC(kernel='rbf')
            tca_svm.fit(X_source_t, y_source)
            tca_pred = tca_svm.predict(X_target_t)
            tca_acc = accuracy_score(y_target, tca_pred)
            logger.info(f"TCA+SVM accuracy: {tca_acc:.4f}")
        
        # Method 3: Classifier Replacement Ensemble
        if config.get('methods', {}).get('cre', False):
            logger.info("Applying Classifier Replacement Ensemble...")
            
            if batch_idx == 1:
                # Initialize ensemble
                cre = ClassifierReplacementEnsemble(
                    ensemble_size=5,
                    threshold=0.05
                )
                cre.fit_initial(X_source, y_source)
                logger.info("CRE ensemble initialized")
            else:
                # Update ensemble with new data
                # In real scenario, you'd have some labeled data
                n_labeled = min(50, len(X_target))
                labeled_idx = np.random.choice(len(X_target), n_labeled, replace=False)
                
                cre.update(X_target[labeled_idx], y_target[labeled_idx])
                logger.info(f"CRE ensemble updated with {n_labeled} samples")
            
            cre_pred = cre.predict(X_target)
            cre_acc = accuracy_score(y_target, cre_pred)
            adaptive_accs.append(cre_acc)
            logger.info(f"CRE accuracy: {cre_acc:.4f}")
        
        batch_accuracies.append({
            'batch': batch_idx,
            'baseline': baseline_acc,
        })
    
    # Compute transfer metrics
    logger.info("\n" + "=" * 60)
    logger.info("Transfer Learning Metrics")
    logger.info("=" * 60)
    
    # Average accuracy
    avg_acc = np.mean([r['baseline'] for r in batch_accuracies])
    logger.info(f"Average Accuracy (baseline): {avg_acc:.4f}")
    
    # Final batch accuracy
    final_acc = batch_accuracies[-1]['baseline']
    logger.info(f"Final Batch Accuracy (baseline): {final_acc:.4f}")
    
    # Performance degradation
    degradation = batch_accuracies[0]['baseline'] - batch_accuracies[-1]['baseline']
    logger.info(f"Performance Degradation: {degradation:.4f}")
    
    if len(adaptive_accs) > 0:
        avg_adaptive_acc = np.mean(adaptive_accs)
        logger.info(f"Average Accuracy (adaptive): {avg_adaptive_acc:.4f}")
        logger.info(f"Improvement over baseline: {avg_adaptive_acc - avg_acc:.4f}")
    
    # Compute BWT and FWT if we have sequential results
    # (In a full implementation, you'd store all predictions for this)
    
    # Save results
    if config.get('evaluation', {}).get('save_results', True):
        output_dir = config['evaluation'].get('output_dir', 'results/drift')
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'batch_accuracies': batch_accuracies,
            'average_accuracy': avg_acc,
            'final_accuracy': final_acc,
            'degradation': degradation,
        }
        
        if len(adaptive_accs) > 0:
            results['adaptive_average'] = avg_adaptive_acc
        
        import json
        results_path = os.path.join(output_dir, 'drift_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    return batch_accuracies


def main():
    parser = argparse.ArgumentParser(description='Evaluate drift compensation performance')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Evaluate drift compensation
    batch_accuracies = evaluate_sequential_batches(config)
    
    logger.info("Drift evaluation completed!")


if __name__ == "__main__":
    main()
