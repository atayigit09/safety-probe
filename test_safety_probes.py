"""
Evaluate trained safety probes on test data using cached features.

This script:
1. Loads pre-cached test features from a specified directory
2. Loads trained probe models from a training run directory
3. Evaluates all probe types: linear (single + concat), MLP (single + concat)
4. Generates comparison visualizations and metrics
"""

import os
import json
import pickle
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import torch.nn as nn


from train_probes import MLPProbe

def load_test_features(cache_dir):
    """Load cached test features and labels."""
    cache_path = os.path.join(cache_dir, "hidden_states.pt")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    cache_data = torch.load(cache_path)
    print(f"✓ Test features loaded from {cache_path}")
    
    # Load labels
    entropy_path = os.path.join(cache_dir, "safety_entropy_labels_tbg.npy")
    score_path = os.path.join(cache_dir, "safety_score_labels_tbg.npy")

    safety_entropy = np.load(entropy_path)
    safety_score = np.load(score_path)
    #joint risk is 
    alpha = 0.7
    joint_risk_target = (alpha * safety_score) + ((1 - alpha) * safety_entropy)


    #first 650 samples are for harmful prompts, rest are for benigin prompts

    print(f"✓ Test labels loaded")
    print(f"  Samples: {len(safety_entropy)}")
    print(f"  Safety Entropy range: [{safety_entropy.min():.3f}, {safety_entropy.max():.3f}]")
    print(f"  Safety Score range: [{safety_score.min():.3f}, {safety_score.max():.3f}]")
    print(f"  Joint Risk range: [{joint_risk_target.min():.3f}, {joint_risk_target.max():.3f}]")
    
    return cache_data['tbg_states'], safety_entropy, safety_score, joint_risk_target


def predict_linear(model, X):
    """Make predictions using linear model."""
    return np.clip(model.predict(X), 0, 1)


def predict_mlp(model, X, batch_size=32, device='mps'):
    """Make predictions using MLP model."""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            predictions.append(preds)
    
    return np.concatenate(predictions)


def evaluate_probe(model, X, y_true, probe_type='linear', device='mps', threshold=0.5, num_harmful=650):
    """Evaluate a single probe and return metrics.
    
    Args:
        model: Trained probe model
        X: Input features
        y_true: Ground truth labels
        probe_type: 'linear' or 'mlp'
        device: Device for computation
        threshold: Threshold for classifying harmful vs benign
        num_harmful: Number of harmful samples (first N samples)
    """
    if probe_type == 'linear':
        y_pred = predict_linear(model, X)
    else:  # mlp
        y_pred = predict_mlp(model, X, device=device)
    
    # Split predictions into harmful and benign
    harmful_preds = y_pred[:num_harmful]
    benign_preds = y_pred[num_harmful:]
    
    harmful_true = y_true[:num_harmful]
    benign_true = y_true[num_harmful:]
    
    # False Positives: benign prompts predicted as harmful (pred > threshold)
    fp_count = np.sum(benign_preds > threshold)
    fp_rate = fp_count / len(benign_preds) * 100 if len(benign_preds) > 0 else 0
    
    # False Negatives: harmful prompts with true label > threshold but predicted below threshold
    harmful_above_threshold = harmful_true > threshold
    fn_count = np.sum((harmful_above_threshold) & (harmful_preds <= threshold))
    fn_rate = fn_count / len(harmful_preds) * 100 if len(harmful_preds) > 0 else 0
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'predictions': y_pred,
        'false_positive_rate': fp_rate,
        'false_negative_rate': fn_rate,
        'fp_count': int(fp_count),
        'fn_count': int(fn_count),
        'num_benign': len(benign_preds),
        'num_harmful': len(harmful_preds),
    }
    
    return metrics


def load_trained_probe(probe_dir, probe_type):
    """Load trained probe from directory."""
    probe_file = os.path.join(probe_dir, f'safety_probe.pkl')
    
    if not os.path.exists(probe_file):
        raise FileNotFoundError(f"Probe file not found: {probe_file}")
    
    with open(probe_file, 'rb') as f:
        probe_data = pickle.load(f)
    
    return probe_data


def evaluate_all_probes(train_dir, test_cache_dir, device='mps', threshold=0.5):
    """Evaluate all trained probes on test data.
    
    Args:
        train_dir: Directory containing trained probes
        test_cache_dir: Directory containing test features
        device: Device for computation
        threshold: Threshold for harmful/benign classification
    """
    
    print("=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)
    
    # Load test features
    test_features, safety_entropy, safety_score, joint_risk = load_test_features(test_cache_dir)
    num_layers = len(test_features)
    num_samples = test_features[0].shape[0]
    
    print(f"\n✓ Test data loaded:")
    print(f"  Number of layers: {num_layers}")
    print(f"  Samples: {num_samples}")
    print(f"    - Harmful: 650")
    print(f"    - Benign: {num_samples - 650}")
    print(f"  Feature dim: {test_features[0].shape[1]}")
    print(f"  Threshold for FP/FN: {threshold}")
    
    results = {
        'test_samples': num_samples,
        'threshold': threshold,
        'probes': {}
    }
    
    # Evaluate each probe type
    for probe_type in ['linear', 'mlp']:
        print("\n" + "=" * 80)
        print(f"EVALUATING {probe_type.upper()} PROBES")
        print("=" * 80)
        
        probe_dir = os.path.join(train_dir, probe_type)
        
        if not os.path.exists(probe_dir):
            print(f"⚠ Probe directory not found: {probe_dir}")
            continue
        
        # Load probe data
        probe_data = load_trained_probe(probe_dir, probe_type)
        
        results['probes'][probe_type] = {}
        
        # Evaluate single best layer
        print(f"\n--- Single Best Layer (Layer {probe_data['best_layer']}) ---")
        best_layer = probe_data['best_layer']
        best_model = probe_data['best_model']
        
        X_test = test_features[best_layer].numpy() if isinstance(test_features[best_layer], torch.Tensor) else test_features[best_layer]
        
        metrics = evaluate_probe(best_model, X_test, joint_risk, probe_type, device, threshold)
        
        print(f"  Test MSE: {metrics['mse']:.4f}")
        print(f"  Test MAE: {metrics['mae']:.4f}")
        print(f"  Test R²: {metrics['r2']:.4f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.2f}% ({metrics['fp_count']}/{metrics['num_benign']} benign)")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.2f}% ({metrics['fn_count']}/{metrics['num_harmful']} harmful)")
        
        results['probes'][probe_type]['single_layer'] = {
            'layer': best_layer,
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae']),
            'r2': float(metrics['r2']),
            'false_positive_rate': float(metrics['false_positive_rate']),
            'false_negative_rate': float(metrics['false_negative_rate']),
            'fp_count': metrics['fp_count'],
            'fn_count': metrics['fn_count'],
            'predictions': metrics['predictions'].tolist()
        }
        
        # Evaluate concatenated layers
        print(f"\n--- Concatenated Layers {probe_data['concatenated_layer_indices']} ---")
        concat_model = probe_data['concatenated_model']
        concat_layers = probe_data['concatenated_layer_indices']
        
        # Concatenate features
        concat_features = []
        for layer_idx in concat_layers:
            layer_feat = test_features[layer_idx].numpy() if isinstance(test_features[layer_idx], torch.Tensor) else test_features[layer_idx]
            concat_features.append(layer_feat)
        X_test_concat = np.concatenate(concat_features, axis=1)
        
        concat_metrics = evaluate_probe(concat_model, X_test_concat, joint_risk, probe_type, device, threshold)
        
        print(f"  Test MSE: {concat_metrics['mse']:.4f}")
        print(f"  Test MAE: {concat_metrics['mae']:.4f}")
        print(f"  Test R²: {concat_metrics['r2']:.4f}")
        print(f"  False Positive Rate: {concat_metrics['false_positive_rate']:.2f}% ({concat_metrics['fp_count']}/{concat_metrics['num_benign']} benign)")
        print(f"  False Negative Rate: {concat_metrics['false_negative_rate']:.2f}% ({concat_metrics['fn_count']}/{concat_metrics['num_harmful']} harmful)")
        
        results['probes'][probe_type]['concatenated'] = {
            'layers': concat_layers,
            'mse': float(concat_metrics['mse']),
            'mae': float(concat_metrics['mae']),
            'r2': float(concat_metrics['r2']),
            'false_positive_rate': float(concat_metrics['false_positive_rate']),
            'false_negative_rate': float(concat_metrics['false_negative_rate']),
            'fp_count': concat_metrics['fp_count'],
            'fn_count': concat_metrics['fn_count'],
            'predictions': concat_metrics['predictions'].tolist()
        }
        
        # Comparison
        improvement_mse = ((metrics['mse'] - concat_metrics['mse']) / metrics['mse'] * 100)
        improvement_r2 = concat_metrics['r2'] - metrics['r2']
        
        print(f"\n  Improvement (Concat vs Single):")
        print(f"    MSE: {improvement_mse:+.2f}% {'(better)' if improvement_mse > 0 else '(worse)'}")
        print(f"    R²: {improvement_r2:+.4f} {'(better)' if improvement_r2 > 0 else '(worse)'}")
    
    return results, joint_risk


def create_evaluation_visualizations(results, y_true, output_dir):
    """Create comprehensive evaluation visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    probe_types = list(results['probes'].keys())
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MSE Comparison
    ax = axes[0, 0]
    x_pos = np.arange(len(probe_types) * 2)
    width = 0.35
    
    mse_single = [results['probes'][pt]['single_layer']['mse'] for pt in probe_types]
    mse_concat = [results['probes'][pt]['concatenated']['mse'] for pt in probe_types]
    
    ax.bar(x_pos[::2] - width/2, mse_single, width, label='Single Layer', alpha=0.8)
    ax.bar(x_pos[::2] + width/2, mse_concat, width, label='Concatenated', alpha=0.8)
    ax.set_xlabel('Probe Type')
    ax.set_ylabel('MSE (lower is better)')
    ax.set_title('Test MSE Comparison')
    ax.set_xticks(x_pos[::2])
    ax.set_xticklabels([pt.upper() for pt in probe_types])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # R² Comparison
    ax = axes[0, 1]
    r2_single = [results['probes'][pt]['single_layer']['r2'] for pt in probe_types]
    r2_concat = [results['probes'][pt]['concatenated']['r2'] for pt in probe_types]
    
    ax.bar(x_pos[::2] - width/2, r2_single, width, label='Single Layer', alpha=0.8)
    ax.bar(x_pos[::2] + width/2, r2_concat, width, label='Concatenated', alpha=0.8)
    ax.set_xlabel('Probe Type')
    ax.set_ylabel('R² (higher is better)')
    ax.set_title('Test R² Comparison')
    ax.set_xticks(x_pos[::2])
    ax.set_xticklabels([pt.upper() for pt in probe_types])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Prediction distribution for best probe
    ax = axes[1, 0]
    for pt in probe_types:
        preds_single = np.array(results['probes'][pt]['single_layer']['predictions'])
        preds_concat = np.array(results['probes'][pt]['concatenated']['predictions'])
        
        ax.hist(preds_single, bins=30, alpha=0.5, label=f'{pt.upper()} Single', density=True)
        ax.hist(preds_concat, bins=30, alpha=0.5, label=f'{pt.upper()} Concat', density=True)
    
    ax.hist(y_true, bins=30, alpha=0.3, label='Ground Truth', density=True, color='black', linestyle='--', histtype='step', linewidth=2)
    ax.set_xlabel('Joint Risk Score')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scatter plot: predictions vs ground truth
    ax = axes[1, 1]
    colors = ['blue', 'red']
    markers = ['o', 's']
    
    for idx, pt in enumerate(probe_types):
        preds_concat = np.array(results['probes'][pt]['concatenated']['predictions'])
        ax.scatter(y_true, preds_concat, alpha=0.5, s=20, c=colors[idx], marker=markers[idx], label=f'{pt.upper()} Concat')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect prediction')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predictions')
    ax.set_title('Predictions vs Ground Truth (Concatenated Models)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'evaluation_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained safety probes on test data')
    parser.add_argument('--train_dir', type=str, default = 'runs/test4',
                       help='Directory containing trained probes (e.g., runs/test4)')
    parser.add_argument('--test_cache_dir', type=str, default='feature_caches/llama3.2_test',
                       help='Directory containing cached test features')
    parser.add_argument('--device', type=str, default='mps', choices=['cuda', 'mps', 'cpu'],
                       help='Device to use for MLP evaluation')
    parser.add_argument('--threshold', type=float, default=0.0050,
                       help='Threshold for harmful/benign classification (default: 0.0)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save evaluation results (default: train_dir/evaluation)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.train_dir, 'evaluation')
    
    print("=" * 80)
    print("SAFETY PROBE EVALUATION")
    print("=" * 80)
    print(f"Train directory: {args.train_dir}")
    print(f"Test cache directory: {args.test_cache_dir}")
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Run evaluation
    results, y_true = evaluate_all_probes(args.train_dir, args.test_cache_dir, device, args.threshold)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    create_evaluation_visualizations(results, y_true, args.output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    for probe_type in results['probes'].keys():
        print(f"\n{probe_type.upper()} Probes:")
        print(f"  Single Layer:")
        print(f"    MSE: {results['probes'][probe_type]['single_layer']['mse']:.4f}")
        print(f"    R²: {results['probes'][probe_type]['single_layer']['r2']:.4f}")
        print(f"    FP Rate: {results['probes'][probe_type]['single_layer']['false_positive_rate']:.2f}%")
        print(f"    FN Rate: {results['probes'][probe_type]['single_layer']['false_negative_rate']:.2f}%")
        print(f"  Concatenated:")
        print(f"    MSE: {results['probes'][probe_type]['concatenated']['mse']:.4f}")
        print(f"    R²: {results['probes'][probe_type]['concatenated']['r2']:.4f}")
        print(f"    FP Rate: {results['probes'][probe_type]['concatenated']['false_positive_rate']:.2f}%")
        print(f"    FN Rate: {results['probes'][probe_type]['concatenated']['false_negative_rate']:.2f}%")
    
    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
