#!/usr/bin/env python3
"""
Benchmark Script: Train Static Baseline Models

This script trains and evaluates static baseline models on E-nose datasets
without any domain adaptation or drift compensation.

Usage:
    python benchmarks/train_baseline.py --config configs/ucisd_baseline.yaml
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import logging
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_loaders import load_ucsd, load_cqu
from models.feature_fusion import ConcatenationFusion, PCAFusion
from utils.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_config):
    """Create model based on configuration."""
    model_name = model_config.get('name', 'svm')
    
    if model_name == 'svm':
        return SVC(kernel='rbf', probability=True, C=1.0)
    elif model_name == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_baseline(config):
    """Train baseline model according to configuration."""
    logger.info("=" * 60)
    logger.info("Training Static Baseline Model")
    logger.info("=" * 60)
    
    # Load dataset
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == 'ucsd':
        X_source, y_source, X_target, y_target = load_ucsd(
            source_batches=dataset_config.get('source_batches', [1, 2]),
            target_batches=dataset_config.get('target_batches', [3, 4, 5]),
            normalize=dataset_config.get('normalize', True)
        )
    elif dataset_name == 'cqu':
        X_source, y_source, X_target, y_target = load_cqu(
            source_batch=dataset_config.get('source_batch', 1),
            target_batch=dataset_config.get('target_batch', 2),
            normalize=dataset_config.get('normalize', True)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Source: {X_source.shape[0]} samples, Target: {X_target.shape[0]} samples")
    logger.info(f"Number of classes: {len(np.unique(y_target))}")
    
    # Feature fusion (if specified)
    fusion_config = config.get('fusion', None)
    if fusion_config:
        logger.info(f"Applying feature fusion: {fusion_config['method']}")
        
        if fusion_config['method'] == 'pca':
            from models.feature_fusion import PCAFusion
            fusion = PCAFusion(n_components=fusion_config.get('n_components', 10))
            X_source = fusion.fit_transform(X_source)
            X_target = fusion.transform(X_target)
        elif fusion_config['method'] == 'attention':
            # Attention fusion requires multi-sensor data structure
            logger.info("Attention fusion requires multi-sensor data, skipping...")
    
    logger.info(f"Feature dimension after fusion: {X_source.shape[1]}")
    
    # Create model
    model_config = config['model']
    model = create_model(model_config)
    
    logger.info(f"Model: {model_config['name']}")
    
    # Train model on source domain
    logger.info("Training on source domain...")
    model.fit(X_source, y_source)
    
    # Evaluate on source domain
    source_pred = model.predict(X_source)
    source_accuracy = accuracy_score(y_source, source_pred)
    logger.info(f"Source domain accuracy: {source_accuracy:.4f}")
    
    # Evaluate on target domain (no adaptation)
    target_pred = model.predict(X_target)
    target_accuracy = accuracy_score(y_target, target_pred)
    logger.info(f"Target domain accuracy (no adaptation): {target_accuracy:.4f}")
    
    # Compute comprehensive metrics
    metrics = compute_all_metrics(y_target, target_pred)
    
    logger.info("\n" + "=" * 60)
    logger.info("Baseline Performance Summary")
    logger.info("=" * 60)
    logger.info(f"Source Accuracy: {source_accuracy:.4f}")
    logger.info(f"Target Accuracy: {target_accuracy:.4f}")
    logger.info(f"Performance Drop: {(source_accuracy - target_accuracy):.4f}")
    logger.info(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    
    # Save results
    if config.get('evaluation', {}).get('save_results', True):
        output_dir = config['evaluation'].get('output_dir', 'results/baseline')
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'source_accuracy': source_accuracy,
            'target_accuracy': target_accuracy,
            'metrics': metrics,
            'predictions': target_pred.tolist(),
            'true_labels': y_target.tolist()
        }
        
        results_path = os.path.join(output_dir, f'{dataset_name}_baseline_results.yaml')
        with open(results_path, 'w') as f:
            yaml.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    # Detailed classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_target, target_pred))
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Train static baseline model')
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
    
    # Train baseline
    model, metrics = train_baseline(config)
    
    logger.info("Baseline training completed!")


if __name__ == "__main__":
    main()
