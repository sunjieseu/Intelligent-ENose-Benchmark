#!/usr/bin/env python3
"""
Benchmark Script: Evaluate Few-Shot Learning Performance

This script evaluates N-way K-shot few-shot learning performance
using episodic evaluation protocol.

Usage:
    python benchmarks/eval_fewshot.py --config configs/ucisd_fewshot.yaml
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import logging
from tqdm import tqdm
from typing import Tuple, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_loaders import load_ucsd, create_fewshot_task
from models.few_shot import PrototypicalNetwork, create_fewshot_episode
from utils.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_fewshot(config):
    """Evaluate few-shot learning performance."""
    logger.info("=" * 60)
    logger.info("Evaluating Few-Shot Learning Performance")
    logger.info("=" * 60)
    
    # Load configuration
    dataset_config = config['dataset']
    fewshot_config = config['fewshot']
    
    n_way = fewshot_config['n_way']
    k_shot = fewshot_config['k_shot']
    n_query = fewshot_config.get('n_query', 15)
    n_episodes = fewshot_config.get('n_episodes', 1000)
    
    dataset_name = dataset_config['name']
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Task: {n_way}-way {k_shot}-shot")
    logger.info(f"Query samples per class: {n_query}")
    logger.info(f"Evaluation episodes: {n_episodes}")
    
    # Load dataset
    logger.info(f"\nLoading dataset...")
    X_source, y_source, X_target, y_target = load_ucsd(
        source_batches=dataset_config.get('source_batches', [1, 2]),
        target_batches=dataset_config.get('target_batches', [3]),
        normalize=True
    )
    
    logger.info(f"Target domain: {X_target.shape[0]} samples")
    logger.info(f"Number of classes: {len(np.unique(y_target))}")
    
    # Initialize model
    model_config = config['model']
    model_name = model_config['name']
    
    if model_name == 'protonet':
        model = PrototypicalNetwork(
            input_dim=X_target.shape[1],
            hidden_dim=model_config.get('hidden_dim', 64),
            embedding_dim=model_config.get('embedding_dim', 32)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    logger.info(f"Model: {model_name}")
    
    # Training phase
    if config.get('training', {}).get('train', True):
        logger.info("\n" + "=" * 40)
        logger.info("Training Phase")
        logger.info("=" * 40)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
        device = config['training'].get('device', 'cpu')
        
        model.to(device)
        
        # Episodic training
        logger.info(f"Training for {n_episodes} episodes...")
        
        episode_losses = []
        
        for episode in tqdm(range(n_episodes), desc="Training"):
            # Create training episode
            support_set, query_set = create_fewshot_episode(
                X_target, y_target,
                n_way=n_way,
                k_shot=k_shot,
                n_query=n_query
            )
            
            # Train on episode
            loss = model.train_episode(support_set, query_set, optimizer, device)
            episode_losses.append(loss)
        
        avg_loss = np.mean(episode_losses[-100:])
        logger.info(f"Average loss (last 100 episodes): {avg_loss:.4f}")
    
    # Evaluation phase
    logger.info("\n" + "=" * 40)
    logger.info("Evaluation Phase")
    logger.info("=" * 40)
    
    n_test_episodes = config.get('evaluation', {}).get('n_test_episodes', 600)
    
    accuracies = []
    f1_scores = []
    
    logger.info(f"Evaluating on {n_test_episodes} episodes...")
    
    for episode in tqdm(range(n_test_episodes), desc="Evaluating"):
        # Create test episode
        support_set, query_set = create_fewshot_episode(
            X_target, y_target,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query
        )
        
        X_support, y_support = support_set
        X_query, y_query = query_set
        
        # Compute prototypes from support set
        X_support_t = torch.FloatTensor(X_support)
        y_support_t = torch.LongTensor(y_support)
        
        with torch.no_grad():
            prototypes = model.compute_prototypes(X_support_t, y_support_t, n_way)
        
        # Predict query samples
        predictions = model.predict(X_query, prototypes)
        
        # Compute metrics
        accuracy = np.mean(predictions == y_query)
        accuracies.append(accuracy)
    
    # Compute statistics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    ci_95 = 1.96 * std_accuracy / np.sqrt(len(accuracies))
    
    logger.info("\n" + "=" * 60)
    logger.info("Few-Shot Learning Performance Summary")
    logger.info("=" * 60)
    logger.info(f"Task: {n_way}-way {k_shot}-shot")
    logger.info(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"95% Confidence Interval: ±{ci_95:.4f}")
    logger.info(f"Min Accuracy: {np.min(accuracies):.4f}")
    logger.info(f"Max Accuracy: {np.max(accuracies):.4f}")
    
    # Save results
    if config.get('evaluation', {}).get('save_results', True):
        output_dir = config['evaluation'].get('output_dir', 'results/fewshot')
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'task': f'{n_way}-way {k_shot}-shot',
            'n_episodes': n_test_episodes,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'ci_95': ci_95,
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'all_accuracies': accuracies
        }
        
        import json
        results_path = os.path.join(output_dir, f'fewshot_{n_way}way_{k_shot}shot_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    return mean_accuracy, std_accuracy


def compare_different_shots(config):
    """Evaluate performance with different K-shot values."""
    logger.info("\n" + "=" * 60)
    logger.info("Comparing Different K-shot Values")
    logger.info("=" * 60)
    
    k_values = config.get('fewshot', {}).get('k_values', [1, 3, 5, 10])
    n_way = config['fewshot']['n_way']
    
    results = {}
    
    for k_shot in k_values:
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating {n_way}-way {k_shot}-shot")
        logger.info(f"{'='*40}")
        
        # Update config
        config['fewshot']['k_shot'] = k_shot
        
        # Evaluate
        mean_acc, std_acc = evaluate_fewshot(config)
        
        results[k_shot] = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc
        }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("K-shot Comparison Summary")
    logger.info("=" * 60)
    
    for k_shot, res in results.items():
        logger.info(f"{k_shot}-shot: {res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate few-shot learning performance')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--compare-shots', action='store_true',
                       help='Compare different K-shot values')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    if args.compare_shots:
        results = compare_different_shots(config)
    else:
        mean_acc, std_acc = evaluate_fewshot(config)
    
    logger.info("Few-shot evaluation completed!")


if __name__ == "__main__":
    main()
