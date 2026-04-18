import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import Ridge
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_curve, auc, roc_auc_score
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu

import warnings
warnings.filterwarnings('ignore')

from train_probes import MLPProbe

def load_probes(probe_path):
    """Load saved probe models from checkpoint."""
    print(f"Loading probes from: {probe_path}")
    
    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Probe file not found: {probe_path}")
    
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)
    
    probe_type = probe_data.get('probe_type', 'mlp')
    print(f"Loaded {probe_type.upper()} probes for model: {probe_data['model_name']}")
    print(f"Best layer: {probe_data['best_layer']}")
    print(f"Concatenated layers: {probe_data['concatenated_layer_indices']}")
    
    return probe_data


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
    joint_risk_path = os.path.join(cache_dir, "joint_risk_target_labels_tbg.npy")
    
    safety_entropy = np.load(entropy_path)
    safety_score = np.load(score_path)
    #joint risk is 
    alpha = 0.7
    joint_risk_target = (alpha * safety_score) + ((1 - alpha) * safety_entropy)

    print(f"✓ Test labels loaded")
    print(f"  Samples: {len(safety_entropy)}")
    print(f"  Safety Entropy range: [{safety_entropy.min():.3f}, {safety_entropy.max():.3f}]")
    print(f"  Safety Score range: [{safety_score.min():.3f}, {safety_score.max():.3f}]")
    print(f"  Joint Risk range: [{joint_risk_target.min():.3f}, {joint_risk_target.max():.3f}]")
    
    return cache_data['tbg_states'], safety_entropy, safety_score, joint_risk_target

def predict_with_mlp_probe(probe_model, features, device='mps', batch_size=32):
    """Make predictions using an MLP probe model."""
    probe_model.eval()
    # Convert features to float32 if they're in a different dtype (e.g., float16)
    if isinstance(features, torch.Tensor):
        features = features.float()
    else:
        features = torch.FloatTensor(features)
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            outputs = probe_model(batch_X)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions)


def predict_with_linear_probe(probe_model, features):
    """Make predictions using a linear (Ridge) probe model."""
    # Convert tensor to numpy if needed
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    predictions = probe_model.predict(features)
    return np.clip(predictions, 0, 1)


def evaluate_probes_on_datasets(probe_data, benign_features, harmful_features, device='mps'):
    """Evaluate probes on both benign and harmful datasets."""
    print("Evaluating probes on test datasets...")
    
    probe_type = probe_data.get('probe_type', 'mlp')
    is_mlp = probe_type == 'mlp'
    
    results = {
        'benign': {},
        'harmful': {},
        'best_layer': probe_data['best_layer'],
        'concatenated_layers': probe_data['concatenated_layer_indices'],
        'probe_type': probe_type
    }
    
    # Choose prediction function based on probe type
    if is_mlp:
        predict_fn = lambda model, features: predict_with_mlp_probe(model, features, device)
    else:
        predict_fn = predict_with_linear_probe
    
    # Evaluate single-layer probes
    print(f"\nEvaluating single-layer {probe_type.upper()} probes...")
    for layer_idx, probe_model in enumerate(probe_data['all_models']):
        print(f"Layer {layer_idx}:")
        
        # Benign predictions
        benign_pred = predict_fn(probe_model, benign_features[layer_idx])
        results['benign'][f'layer_{layer_idx}'] = {
            'predictions': benign_pred,
            'mean': np.mean(benign_pred),
            'std': np.std(benign_pred),
            'min': np.min(benign_pred),
            'max': np.max(benign_pred)
        }
        
        # Harmful predictions
        harmful_pred = predict_fn(probe_model, harmful_features[layer_idx])
        results['harmful'][f'layer_{layer_idx}'] = {
            'predictions': harmful_pred,
            'mean': np.mean(harmful_pred),
            'std': np.std(harmful_pred),
            'min': np.min(harmful_pred),
            'max': np.max(harmful_pred)
        }
        
        print(f"  Benign - Mean: {results['benign'][f'layer_{layer_idx}']['mean']:.4f}, Std: {results['benign'][f'layer_{layer_idx}']['std']:.4f}")
        print(f"  Harmful - Mean: {results['harmful'][f'layer_{layer_idx}']['mean']:.4f}, Std: {results['harmful'][f'layer_{layer_idx}']['std']:.4f}")
    
    # Evaluate concatenated probe
    print(f"\nEvaluating concatenated {probe_type.upper()} probe...")
    concat_model = probe_data['concatenated_model']
    concat_layers = probe_data['concatenated_layer_indices']
    
    # Concatenate features from specified layers
    benign_concat_features = np.concatenate([benign_features[i] for i in concat_layers], axis=1)
    harmful_concat_features = np.concatenate([harmful_features[i] for i in concat_layers], axis=1)
    
    # Benign predictions
    benign_pred_concat = predict_fn(concat_model, benign_concat_features)
    results['benign']['concatenated'] = {
        'predictions': benign_pred_concat,
        'mean': np.mean(benign_pred_concat),
        'std': np.std(benign_pred_concat),
        'min': np.min(benign_pred_concat),
        'max': np.max(benign_pred_concat)
    }
    
    # Harmful predictions
    harmful_pred_concat = predict_fn(concat_model, harmful_concat_features)
    results['harmful']['concatenated'] = {
        'predictions': harmful_pred_concat,
        'mean': np.mean(harmful_pred_concat),
        'std': np.std(harmful_pred_concat),
        'min': np.min(harmful_pred_concat),
        'max': np.max(harmful_pred_concat)
    }
    
    print(f"Concatenated Benign - Mean: {results['benign']['concatenated']['mean']:.4f}, Std: {results['benign']['concatenated']['std']:.4f}")
    print(f"Concatenated Harmful - Mean: {results['harmful']['concatenated']['mean']:.4f}, Std: {results['harmful']['concatenated']['std']:.4f}")
    
    return results


def create_visualizations(results, output_dir="results/probe_evaluation"):
    """Create comprehensive visualizations of probe evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    probe_type = results.get('probe_type', 'mlp')
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Enhanced Score Distributions with Violin Plots
    create_enhanced_distributions(results, output_dir, probe_type)
    
    # 2. ROC Curves and AUC Analysis
    create_roc_analysis(results, output_dir, probe_type)
    
    # 3. Statistical Analysis and Effect Size
    create_statistical_analysis(results, output_dir, probe_type)
    
    # 4. Mean and std comparison across layers
    create_layer_comparison(results, output_dir, probe_type)
    
    # 5. Clustering visualization
    create_clustering_visualization(results, output_dir, probe_type)
    
    print(f"Visualizations saved to: {output_dir}")


def create_enhanced_distributions(results, output_dir, probe_type='mlp'):
    """Create enhanced distribution visualizations with violin plots and density plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Best layer results
    best_layer = results['best_layer']
    benign_best = results['benign'][f'layer_{best_layer}']['predictions']
    harmful_best = results['harmful'][f'layer_{best_layer}']['predictions']
    
    # Concatenated results
    benign_concat = results['benign']['concatenated']['predictions']
    harmful_concat = results['harmful']['concatenated']['predictions']
    
    # Violin plots for best layer
    ax1 = axes[0, 0]
    data_best = [benign_best, harmful_best]
    parts = ax1.violinplot(data_best, positions=[1, 2], showmeans=True, showmedians=True)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Benign', 'Harmful'])
    ax1.set_title(f'Best Layer {best_layer} - Violin Plot ({probe_type.upper()})')
    ax1.set_ylabel('Safety Score')
    ax1.grid(True, alpha=0.3)
    
    # Violin plots for concatenated
    ax2 = axes[0, 1]
    data_concat = [benign_concat, harmful_concat]
    parts = ax2.violinplot(data_concat, positions=[1, 2], showmeans=True, showmedians=True)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Benign', 'Harmful'])
    ax2.set_title(f'Concatenated Layers - Violin Plot ({probe_type.upper()})')
    ax2.set_ylabel('Safety Score')
    ax2.grid(True, alpha=0.3)
    
    # Density plots for best layer
    ax3 = axes[1, 0]
    ax3.hist(benign_best, bins=50, alpha=0.6, label='Benign', density=True, color='blue')
    ax3.hist(harmful_best, bins=50, alpha=0.6, label='Harmful', density=True, color='red')
    
    # Add kernel density estimation
    from scipy.stats import gaussian_kde
    kde_benign = gaussian_kde(benign_best)
    kde_harmful = gaussian_kde(harmful_best)
    x_range = np.linspace(min(min(benign_best), min(harmful_best)), 
                         max(max(benign_best), max(harmful_best)), 200)
    ax3.plot(x_range, kde_benign(x_range), 'b-', linewidth=2, alpha=0.8, label='Benign KDE')
    ax3.plot(x_range, kde_harmful(x_range), 'r-', linewidth=2, alpha=0.8, label='Harmful KDE')
    
    ax3.set_title(f'Best Layer {best_layer} - Density Plot ({probe_type.upper()})')
    ax3.set_xlabel('Safety Score')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Density plots for concatenated
    ax4 = axes[1, 1]
    ax4.hist(benign_concat, bins=50, alpha=0.6, label='Benign', density=True, color='blue')
    ax4.hist(harmful_concat, bins=50, alpha=0.6, label='Harmful', density=True, color='red')
    
    kde_benign_concat = gaussian_kde(benign_concat)
    kde_harmful_concat = gaussian_kde(harmful_concat)
    x_range_concat = np.linspace(min(min(benign_concat), min(harmful_concat)), 
                                max(max(benign_concat), max(harmful_concat)), 200)
    ax4.plot(x_range_concat, kde_benign_concat(x_range_concat), 'b-', linewidth=2, alpha=0.8, label='Benign KDE')
    ax4.plot(x_range_concat, kde_harmful_concat(x_range_concat), 'r-', linewidth=2, alpha=0.8, label='Harmful KDE')
    
    ax4.set_title(f'Concatenated Layers - Density Plot ({probe_type.upper()})')
    ax4.set_xlabel('Safety Score')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'enhanced_score_distributions_{probe_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_roc_analysis(results, output_dir, probe_type='mlp'):
    """Create ROC curve analysis for separability assessment."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Best layer results
    best_layer = results['best_layer']
    benign_best = results['benign'][f'layer_{best_layer}']['predictions']
    harmful_best = results['harmful'][f'layer_{best_layer}']['predictions']
    
    # Concatenated results
    benign_concat = results['benign']['concatenated']['predictions']
    harmful_concat = results['harmful']['concatenated']['predictions']
    
    datasets = [
        (f'Best Layer ({probe_type.upper()})', benign_best, harmful_best),
        (f'Concatenated ({probe_type.upper()})', benign_concat, harmful_concat)
    ]
    
    for idx, (name, benign_scores, harmful_scores) in enumerate(datasets):
        ax = axes[idx]
        
        # Create labels: 0 for benign, 1 for harmful
        y_true = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(harmful_scores))])
        y_scores = np.concatenate([benign_scores, harmful_scores])
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name} - ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
                label=f'Optimal threshold: {optimal_threshold:.3f}')
        ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_analysis_{probe_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_statistical_analysis(results, output_dir, probe_type='mlp'):
    """Create statistical analysis visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Best layer results
    best_layer = results['best_layer']
    benign_best = results['benign'][f'layer_{best_layer}']['predictions']
    harmful_best = results['harmful'][f'layer_{best_layer}']['predictions']
    
    # Concatenated results
    benign_concat = results['benign']['concatenated']['predictions']
    harmful_concat = results['harmful']['concatenated']['predictions']
    
    datasets = [
        (f'Best Layer ({probe_type.upper()})', benign_best, harmful_best),
        (f'Concatenated ({probe_type.upper()})', benign_concat, harmful_concat)
    ]
    
    for idx, (name, benign_scores, harmful_scores) in enumerate(datasets):
        # Scatter plot with decision boundary
        ax_scatter = axes[idx, 0]
        
        # Create scatter plot
        y_positions = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(harmful_scores))])
        all_scores = np.concatenate([benign_scores, harmful_scores])
        
        # Add some jitter to y-axis for better visualization
        y_jittered = y_positions + np.random.normal(0, 0.05, len(y_positions))
        
        ax_scatter.scatter(benign_scores, np.zeros(len(benign_scores)) + np.random.normal(0, 0.05, len(benign_scores)), 
                          alpha=0.6, color='blue', label='Benign', s=20)
        ax_scatter.scatter(harmful_scores, np.ones(len(harmful_scores)) + np.random.normal(0, 0.05, len(harmful_scores)), 
                          alpha=0.6, color='red', label='Harmful', s=20)
        
        # Add decision boundary
        optimal_threshold = (np.mean(benign_scores) + np.mean(harmful_scores)) / 2
        ax_scatter.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, 
                          label=f'Decision boundary: {optimal_threshold:.3f}')
        
        ax_scatter.set_xlabel('Safety Score')
        ax_scatter.set_ylabel('Label (0=Benign, 1=Harmful)')
        ax_scatter.set_title(f'{name} - Scatter Plot with Decision Boundary')
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax_box = axes[idx, 1]
        data = [benign_scores, harmful_scores]
        bp = ax_box.boxplot(data, labels=['Benign', 'Harmful'], patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax_box.set_ylabel('Safety Score')
        ax_box.set_title(f'{name} - Box Plot Comparison')
        ax_box.grid(True, alpha=0.3)
        
        # Add statistical annotations
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(benign_scores) - 1) * np.var(benign_scores, ddof=1) + 
                             (len(harmful_scores) - 1) * np.var(harmful_scores, ddof=1)) / 
                            (len(benign_scores) + len(harmful_scores) - 2))
        cohens_d = (np.mean(harmful_scores) - np.mean(benign_scores)) / pooled_std
        
        # Add text annotation
        ax_box.text(0.02, 0.98, f"Cohen's d: {cohens_d:.3f}\n"
                                f"Mean diff: {np.mean(harmful_scores) - np.mean(benign_scores):.3f}\n"
                                f"Separation: {abs(np.mean(harmful_scores) - np.mean(benign_scores)):.3f}", 
                   transform=ax_box.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'statistical_analysis_{probe_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_layer_comparison(results, output_dir, probe_type='mlp'):
    """Create layer comparison visualizations."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    layers = list(range(len(results['benign']) - 1))  # Exclude concatenated
    benign_means = [results['benign'][f'layer_{i}']['mean'] for i in layers]
    harmful_means = [results['harmful'][f'layer_{i}']['mean'] for i in layers]
    benign_stds = [results['benign'][f'layer_{i}']['std'] for i in layers]
    harmful_stds = [results['harmful'][f'layer_{i}']['std'] for i in layers]
    
    best_layer = results['best_layer']
    
    axes[0].plot(layers, benign_means, 'o-', label='Benign', linewidth=2, markersize=6)
    axes[0].plot(layers, harmful_means, 's-', label='Harmful', linewidth=2, markersize=6)
    axes[0].axvline(best_layer, color='red', linestyle='--', alpha=0.7, label='Best Layer')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean Safety Score')
    axes[0].set_title(f'Mean Safety Scores by Layer ({probe_type.upper()})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(layers, benign_stds, 'o-', label='Benign', linewidth=2, markersize=6)
    axes[1].plot(layers, harmful_stds, 's-', label='Harmful', linewidth=2, markersize=6)
    axes[1].axvline(best_layer, color='red', linestyle='--', alpha=0.7, label='Best Layer')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_title(f'Score Standard Deviation by Layer ({probe_type.upper()})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'layer_comparison_{probe_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_clustering_visualization(results, output_dir, probe_type='mlp'):
    """Create clustering visualization to show separation between benign and harmful."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Use best layer and concatenated results for clustering
    best_layer = results['best_layer']
    
    # Prepare data for clustering
    datasets = [
        (f'Best Layer ({probe_type.upper()})', results['benign'][f'layer_{best_layer}']['predictions'], 
         results['harmful'][f'layer_{best_layer}']['predictions']),
        (f'Concatenated ({probe_type.upper()})', results['benign']['concatenated']['predictions'], 
         results['harmful']['concatenated']['predictions'])
    ]
    
    for idx, (name, benign_scores, harmful_scores) in enumerate(datasets):
        # Combine scores and create labels
        all_scores = np.concatenate([benign_scores, harmful_scores])
        true_labels = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(harmful_scores))])
        
        # Reshape for clustering
        X = all_scores.reshape(-1, 1)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        # Plot clustering results
        ax = axes[idx, 0]
        colors = ['blue', 'red']
        for i in range(2):
            mask = cluster_labels == i
            ax.scatter(range(np.sum(mask)), X[mask, 0], c=colors[i], alpha=0.6, 
                      label=f'Cluster {i}')
        ax.set_title(f'{name} - K-means Clustering (Silhouette: {silhouette_avg:.3f})')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Safety Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot true labels vs cluster assignments
        ax = axes[idx, 1]
        scatter = ax.scatter(all_scores, true_labels, c=cluster_labels, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Safety Score')
        ax.set_ylabel('True Label (0=Benign, 1=Harmful)')
        ax.set_title(f'{name} - True Labels vs Clusters')
        ax.grid(True, alpha=0.3)
        
        # Add separation line
        threshold = (np.mean(benign_scores) + np.mean(harmful_scores)) / 2
        ax.axvline(threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Threshold: {threshold:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'clustering_analysis_{probe_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_summary_statistics(results):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    probe_type = results.get('probe_type', 'mlp')
    best_layer = results['best_layer']
    concat_layers = results['concatenated_layers']
    
    print(f"\nProbe Type: {probe_type.upper()}")
    print(f"Best Layer: {best_layer}")
    print(f"Concatenated Layers: {concat_layers}")
    
    print(f"\nBest Layer {best_layer} Results:")
    benign_best = results['benign'][f'layer_{best_layer}']
    harmful_best = results['harmful'][f'layer_{best_layer}']
    
    print(f"  Benign  - Mean: {benign_best['mean']:.4f}, Std: {benign_best['std']:.4f}")
    print(f"  Harmful - Mean: {harmful_best['mean']:.4f}, Std: {harmful_best['std']:.4f}")
    print(f"  Separation: {harmful_best['mean'] - benign_best['mean']:.4f}")
    
    print(f"\nConcatenated Layers {concat_layers} Results:")
    benign_concat = results['benign']['concatenated']
    harmful_concat = results['harmful']['concatenated']
    
    print(f"  Benign  - Mean: {benign_concat['mean']:.4f}, Std: {benign_concat['std']:.4f}")
    print(f"  Harmful - Mean: {harmful_concat['mean']:.4f}, Std: {harmful_concat['std']:.4f}")
    print(f"  Separation: {harmful_concat['mean'] - benign_concat['mean']:.4f}")
    
    # Enhanced statistical analysis
    print(f"\n" + "="*50)
    print("ENHANCED STATISTICAL ANALYSIS")
    print("="*50)
    
    datasets = [
        ('Best Layer', benign_best['predictions'], harmful_best['predictions']),
        ('Concatenated', benign_concat['predictions'], harmful_concat['predictions'])
    ]
    
    for name, benign_scores, harmful_scores in datasets:
        print(f"\n{name} Analysis:")
        
        # ROC AUC
        y_true = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(harmful_scores))])
        y_scores = np.concatenate([benign_scores, harmful_scores])
        roc_auc = roc_auc_score(y_true, y_scores)
        print(f"  ROC AUC: {roc_auc:.4f} (1.0 = perfect separation, 0.5 = random)")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(benign_scores) - 1) * np.var(benign_scores, ddof=1) + 
                             (len(harmful_scores) - 1) * np.var(harmful_scores, ddof=1)) / 
                            (len(benign_scores) + len(harmful_scores) - 2))
        cohens_d = (np.mean(harmful_scores) - np.mean(benign_scores)) / pooled_std
        print(f"  Cohen's d: {cohens_d:.4f} (0.2=small, 0.5=medium, 0.8=large effect)")
        
        # Statistical tests
        ks_stat, ks_pvalue = ks_2samp(benign_scores, harmful_scores)
        print(f"  Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.2e}")
        
        mw_stat, mw_pvalue = mannwhitneyu(harmful_scores, benign_scores, alternative='two-sided')
        print(f"  Mann-Whitney U test: statistic={mw_stat:.4f}, p-value={mw_pvalue:.2e}")
        
        # Optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        
        # Classification accuracy at optimal threshold
        predictions = (y_scores >= optimal_threshold).astype(int)
        accuracy = np.mean(predictions == y_true)
        print(f"  Accuracy at optimal threshold: {accuracy:.4f}")
        
        # Calculate overlap
        def calculate_overlap(mean1, std1, mean2, std2):
            """Calculate overlap between two normal distributions."""
            separation = abs(mean1 - mean2)
            combined_std = (std1 + std2) / 2
            return max(0, 1 - separation / (2 * combined_std))
        
        overlap = calculate_overlap(np.mean(benign_scores), np.std(benign_scores), 
                                  np.mean(harmful_scores), np.std(harmful_scores))
        print(f"  Distribution overlap: {overlap:.3f} (lower is better)")
    
    print("="*80)


def read_model_name_from_args(source_path, probe_type):
    """Read model name from arguments.json file in the source path."""
    args_file = os.path.join(source_path, probe_type, 'arguments.json')
    
    if not os.path.exists(args_file):
        raise FileNotFoundError(f"Arguments file not found: {args_file}")
    
    with open(args_file, 'r') as f:
        args_data = json.load(f)
    
    return args_data.get('model_name', 'meta-llama/Llama-3.2-3B-Instruct'), args_data.get('probe_name', 'safety_probe')

def main():
    parser = argparse.ArgumentParser(description='Evaluate safety probes on test datasets')
    parser.add_argument('--source_path', default='runs/test5',
                       help='Base path to the training run (e.g., runs/test1)')
    parser.add_argument('--probe_type', default='both',
                       choices=['mlp', 'linear', 'both'],
                       help='Type of probe to evaluate: mlp, linear, or both (default: both)')
    parser.add_argument('--cache_dir', type=str, default='feature_caches/llama3.2_test',
                       help='Directory containing cached test features')
    parser.add_argument('--device', default='mps',
                       help='Device to use (mps, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(66)
    np.random.seed(66)
    
    # Determine which probe types to evaluate
    eval_mlp = args.probe_type in ['mlp', 'both']
    eval_linear = args.probe_type in ['linear', 'both']
    
    probe_types_to_eval = []
    if eval_mlp:
        probe_types_to_eval.append('mlp')
    if eval_linear:
        probe_types_to_eval.append('linear')
    
    print("Starting safety probe evaluation...")
    print(f"Source path: {args.source_path}")
    print(f"Evaluating probe types: {', '.join([p.upper() for p in probe_types_to_eval])}")
    print(f"Device: {args.device}")
    

    # Load test features
    test_features, safety_entropy, safety_score, joint_risk = load_test_features(args.cache_dir)
    #first 650 are harmful, rest are benign
    harmful_features = [layer_features[:650] for layer_features in test_features]
    benign_features = [layer_features[650:] for layer_features in test_features]
    
    # Evaluate each probe type
    for probe_type in probe_types_to_eval:
        print("\n" + "="*80)
        print(f"EVALUATING {probe_type.upper()} PROBES")
        print("="*80)
        
        # Construct paths
        probe_dir = os.path.join(args.source_path, probe_type)
        output_dir = os.path.join(args.source_path, f'evaluation_{probe_type}')
        
        # Read probe name from arguments.json
        _, probe_name = read_model_name_from_args(args.source_path, probe_type)
        
        # Construct probe path
        probe_filename = f"{probe_name}.pkl"
        probe_path = os.path.join(probe_dir, probe_filename)
        
        print(f"Probe path: {probe_path}")
        print(f"Output directory: {output_dir}")
        
        # Load probes
        probe_data = load_probes(probe_path)
        
        # Evaluate probes
        results = evaluate_probes_on_datasets(probe_data, benign_features, harmful_features, args.device)
        
        # Print summary statistics
        print_summary_statistics(results)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(results, output_dir)
        
        # Save results
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and 'predictions' in subvalue:
                        json_results[key][subkey] = {
                            'mean': float(subvalue['mean']),
                            'std': float(subvalue['std']),
                            'min': float(subvalue['min']),
                            'max': float(subvalue['max'])
                        }
                    else:
                        json_results[key][subkey] = subvalue
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n{probe_type.upper()} evaluation completed!")
        print(f"Results saved to: {output_dir}")
        print(f"Summary statistics saved to: {results_file}")
    
    print("\n" + "="*80)
    print("ALL EVALUATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
