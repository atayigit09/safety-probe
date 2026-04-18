import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings('ignore')

class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, activation='silu', dropout=0.4):
        super(MLPProbe, self).__init__()
        # The traceback specifically mentions 'layer1'
        self.layer1 = nn.Linear(input_dim, hidden_dim * 2)
        self.act = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU()
        }[activation] 
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        return self.sigmoid(x).squeeze(-1)
    
# --- Linear Training ---
def train_single_linear_probe(X_train, y_train, X_val, y_val, alpha=1.0):
    """Train a single linear (Ridge) probe."""
    model = Ridge(alpha=alpha) 
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = np.clip(model.predict(X_train), 0, 1)
    val_pred = np.clip(model.predict(X_val), 0, 1)
    
    # Metrics
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    return model, {
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_r2': train_r2,
        'val_r2': val_r2
    }


def train_linear_layer_probes(hidden_states, labels, val_size=0.2, ridge_alpha=1.0):
    """Train linear (Ridge) probes for each layer."""
    results = {
        'layer_results': [],
        'models': [],
        'best_layer': None,
        'best_mse': float('inf'),
        'model_type': 'linear'
    }
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=42)
    
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    
    for layer_idx, layer_states in enumerate(tqdm(hidden_states, desc="Training linear probes")):
        # Convert to numpy
        X = layer_states.numpy()
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Train linear probe
        model, metrics = train_single_linear_probe(X_train, y_train, X_val, y_val, alpha=ridge_alpha)
        
        layer_result = {
            'layer': layer_idx,
            'train_mse': metrics['train_mse'],
            'val_mse': metrics['val_mse'],
            'train_mae': metrics['train_mae'],
            'val_mae': metrics['val_mae'],
            'train_r2': metrics['train_r2'],
            'val_r2': metrics['val_r2'],
        }
        
        results['layer_results'].append(layer_result)
        results['models'].append(model)
        
        # Track best validation performance (lowest MSE)
        if metrics['val_mse'] < results['best_mse']:
            results['best_mse'] = metrics['val_mse']
            results['best_layer'] = layer_idx
        
        print(f"Layer {layer_idx:2d}: Train MSE: {metrics['train_mse']:.4f}, Val MSE: {metrics['val_mse']:.4f}, Val R²: {metrics['val_r2']:.4f}")
    
    return results
    
def train_mlp_probe(X_train, y_train, X_val, y_val, 
                    hidden_dim=256, activation='relu', dropout=0.1,
                    learning_rate=0.001, num_epochs=100, batch_size=32,
                    patience=10, device='mps'):
    """Train MLP probe with early stopping."""
    input_dim = X_train.shape[1]
    
    # Convert to torch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = MLPProbe(input_dim, hidden_dim, activation, dropout).to(device)
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss /= len(val_dataset)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    return model

def predict_mlp(model, X, batch_size=32, device='cuda'):
    """Make predictions using MLP model."""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions)


def train_linear_layer_probes(hidden_states, labels, val_size=0.2, ridge_alpha=1.0):
    """Train linear (Ridge) probes for each layer."""
    results = {
        'layer_results': [],
        'models': [],
        'best_layer': None,
        'best_mse': float('inf'),
        'model_type': 'linear'
    }
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=42)
    
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    
    for layer_idx, layer_states in enumerate(tqdm(hidden_states, desc="Training linear probes")):
        # Convert to numpy
        X = layer_states.numpy()
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Train linear probe
        model, metrics = train_single_linear_probe(X_train, y_train, X_val, y_val, alpha=ridge_alpha)
        
        layer_result = {
            'layer': layer_idx,
            'train_mse': metrics['train_mse'],
            'val_mse': metrics['val_mse'],
            'train_mae': metrics['train_mae'],
            'val_mae': metrics['val_mae'],
            'train_r2': metrics['train_r2'],
            'val_r2': metrics['val_r2'],
        }
        
        results['layer_results'].append(layer_result)
        results['models'].append(model)
        
        # Track best validation performance (lowest MSE)
        if metrics['val_mse'] < results['best_mse']:
            results['best_mse'] = metrics['val_mse']
            results['best_layer'] = layer_idx
        
        print(f"Layer {layer_idx:2d}: Train MSE: {metrics['train_mse']:.4f}, Val MSE: {metrics['val_mse']:.4f}, Val R²: {metrics['val_r2']:.4f}")
    
    return results


def load_hidden_states(cache_path):

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    cache_data = torch.load(cache_path)
    print(f"Hidden states loaded from {cache_path}")

    if cache_data['metadata']:
        print(f"  Metadata: {cache_data['metadata']}")
    
    hidden_states_dict = {
        'tbg': cache_data['tbg_states'],
    }
    
    return hidden_states_dict, cache_data['metadata']


def load_labels(cache_dir):
    #loads the npy files
    safety_entropy_tbg = np.load(os.path.join(cache_dir, "safety_entropy_labels_tbg.npy"))
    safety_score_tbg = np.load(os.path.join(cache_dir, "safety_score_labels_tbg.npy"))
    return safety_entropy_tbg, safety_score_tbg
    

def train_layer_probes(hidden_states, labels, val_size=0.2, 
                       hidden_dim=256, activation='relu', 
                       dropout=0.1, learning_rate=0.001, num_epochs=100, 
                       mlp_batch_size=32, patience=10, probe_device='mps'):

    results = {
        'layer_results': [],
        'models': [],
        'best_layer': None,
        'best_mse': float('inf'),
        'model_type': 'mlp' 
    }
    
    # Split data (no stratification for continuous labels)
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=42)

    
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    
    for layer_idx, layer_states in enumerate(tqdm(hidden_states, desc="Training probes")):
        # Convert to numpy
        X = layer_states.numpy()
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        model = train_mlp_probe(
            X_train, y_train, X_val, y_val,
            hidden_dim=hidden_dim, activation=activation, dropout=dropout,
            learning_rate=learning_rate, num_epochs=num_epochs,
            batch_size=mlp_batch_size, patience=patience, device=probe_device
        )
            
            # Predict
        train_pred = predict_mlp(model, X_train, batch_size=mlp_batch_size, device=probe_device)
        val_pred = predict_mlp(model, X_val, batch_size=mlp_batch_size, device=probe_device)

        
        # Evaluate with regression metrics
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
  
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
   
        layer_result = {
            'layer': layer_idx,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
        }
        
        results['layer_results'].append(layer_result)
        results['models'].append(model)
        
        # Track best validation performance (lowest MSE)
        if val_mse < results['best_mse']:
            results['best_mse'] = val_mse
            results['best_layer'] = layer_idx
        
        print(f"Layer {layer_idx:2d}: Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, Val R²: {val_r2:.4f}")
    
    return results


def train_concatenated_layers_linear_probe(hidden_states, labels, layer_indices=[0, 1, 2, 3, 4], 
                                          val_size=0.2, ridge_alpha=1.0):
    """Train a linear (Ridge) probe on concatenated layers."""
    print(f"\nTraining linear probe on concatenated layers: {layer_indices}")
    
    # Concatenate features from specified layers
    concatenated_features = []
    for layer_idx in layer_indices:
        concatenated_features.append(hidden_states[layer_idx].numpy())
    
    X = np.concatenate(concatenated_features, axis=1)
    print(f"Concatenated feature shape: {X.shape}")
    print(f"  Original feature dim per layer: {hidden_states[0].shape[1]}")
    print(f"  Total concatenated dim: {X.shape[1]}")
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=42)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    
    # Train linear probe
    model, metrics = train_single_linear_probe(X_train, y_train, X_val, y_val, alpha=ridge_alpha)
    
    results = {
        'layer_indices': layer_indices,
        'model': model,
        'train_mse': metrics['train_mse'],
        'val_mse': metrics['val_mse'], 
        'train_mae': metrics['train_mae'],
        'val_mae': metrics['val_mae'],
        'train_r2': metrics['train_r2'],
        'val_r2': metrics['val_r2'],
        'feature_dim': X.shape[1]
    }
    
    print(f"\nConcatenated Layers {layer_indices} Results (Linear):")
    print(f"  Train MSE: {metrics['train_mse']:.4f}, MAE: {metrics['train_mae']:.4f}, R²: {metrics['train_r2']:.4f}")
    print(f"  Val   MSE: {metrics['val_mse']:.4f}, MAE: {metrics['val_mae']:.4f}, R²: {metrics['val_r2']:.4f}")
    
    return results


def train_concatenated_layers_probe(hidden_states, labels, layer_indices=[0, 1, 2, 3, 4], 
                                   val_size=0.2, hidden_dim=256, 
                                   activation='relu', dropout=0.1, learning_rate=0.001, 
                                   num_epochs=100, mlp_batch_size=32, patience=10, probe_device='mps'):

    print(f"\nTraining MLP probe on concatenated layers: {layer_indices}")
    
    # Concatenate features from specified layers
    concatenated_features = []
    for layer_idx in layer_indices:
        concatenated_features.append(hidden_states[layer_idx].numpy())
    
    X = np.concatenate(concatenated_features, axis=1)
    print(f"Concatenated feature shape: {X.shape}")
    print(f"  Original feature dim per layer: {hidden_states[0].shape[1]}")
    print(f"  Total concatenated dim: {X.shape[1]}")
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=42)

    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    

    model = train_mlp_probe(
        X_train, y_train, X_val, y_val,
        hidden_dim=hidden_dim, activation=activation, dropout=dropout,
        learning_rate=learning_rate, num_epochs=num_epochs,
        batch_size=mlp_batch_size, patience=patience, device=probe_device
    )
        
    # Predict
    train_pred = predict_mlp(model, X_train, batch_size=mlp_batch_size, device=probe_device)
    val_pred = predict_mlp(model, X_val, batch_size=mlp_batch_size, device=probe_device)

    # Evaluate
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    results = {
        'layer_indices': layer_indices,
        'model': model,
        'train_mse': train_mse,
        'val_mse': val_mse, 
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'feature_dim': X.shape[1]
    }
    
    print(f"\nConcatenated Layers {layer_indices} Results:")
    print(f"  Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"  Val   MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return results


def get_next_test_directory(base_dir="runs", dir_name="test"):
    """Find the next available directory (dir_name1, dir_name2, etc.)"""
    os.makedirs(base_dir, exist_ok=True)
    
    dir_num = 1
    while True:
        test_dir = os.path.join(base_dir, f"{dir_name}{dir_num}")
        if not os.path.exists(test_dir):
            return test_dir
        dir_num += 1

def save_results(probe_results, concatenated_results, model_name, output_dir=None, args=None, probe_type='mlp'):
    """Save the trained probes and results (both single-layer and concatenated)."""
    if output_dir is None:
        # Use automatic directory numbering with custom name
        dir_name = args.output_dir if args else "test"
        test_dir = get_next_test_directory(dir_name=dir_name)
        output_dir = os.path.join(test_dir, probe_type)  # mlp or linear
    
    os.makedirs(output_dir, exist_ok=True)
    
    save_data = {
        'model_name': model_name,
        'probe_type': probe_type,
        'best_layer': probe_results['best_layer'],
        'best_model': probe_results['models'][probe_results['best_layer']],
        'all_models': probe_results['models'],
        'results': probe_results['layer_results'],
        
        # Add concatenated model information
        'concatenated_model': concatenated_results['model'],
        'concatenated_layer_indices': concatenated_results['layer_indices'],
        'concatenated_results': {
            'train_mse': concatenated_results['train_mse'],
            'val_mse': concatenated_results['val_mse'],
    
            'train_mae': concatenated_results['train_mae'],
            'val_mae': concatenated_results['val_mae'],
           
            'train_r2': concatenated_results['train_r2'],
            'val_r2': concatenated_results['val_r2'],
            'feature_dim': concatenated_results['feature_dim']
        }
    }
    
    # Use probe_type in filename
    probe_name = getattr(args, 'probe_name', f'safety_probe_{probe_type}')
    output_file = os.path.join(output_dir, f'{probe_name}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    # Save arguments as JSON file
    if args is not None:
        args_dict = vars(args)  # Convert argparse.Namespace to dict
        args_file = os.path.join(output_dir, 'arguments.json')
        with open(args_file, 'w') as f:
            json.dump(args_dict, f, indent=2)
        print(f"Arguments saved to: {args_file}")
    
    print(f"Safety probes saved to: {output_file}")
    return output_file, output_dir


def create_visualizations(probe_results, concatenated_results, output_dir=None, args=None, probe_type='mlp'):
    """Create and save visualization plots for regression probes."""
    if output_dir is None:
        # Use automatic directory numbering with custom name
        dir_name = args.output_dir if args else "test"
        test_dir = get_next_test_directory(dir_name=dir_name)
        output_dir = os.path.join(test_dir, probe_type)  # mlp or linear
    
    os.makedirs(output_dir, exist_ok=True)
    
    layer_nums = [r['layer'] for r in probe_results['layer_results']]
    train_mses = [r['train_mse'] for r in probe_results['layer_results']]
    val_mses = [r['val_mse'] for r in probe_results['layer_results']]

    train_r2s = [r['train_r2'] for r in probe_results['layer_results']]
    val_r2s = [r['val_r2'] for r in probe_results['layer_results']]

    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(layer_nums, train_mses, 'o-', label='Train', alpha=0.7)
    plt.plot(layer_nums, val_mses, 's-', label='Validation', alpha=0.7)
    plt.axvline(probe_results['best_layer'], color='red', linestyle='--', alpha=0.5, label='Best Layer')
    plt.xlabel('Layer')
    plt.ylabel('MSE (lower is better)')
    plt.title(f'Safety Entropy Regression - MSE by Layer ({probe_type.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(layer_nums, train_r2s, 'o-', label='Train', alpha=0.7)
    plt.plot(layer_nums, val_r2s, 's-', label='Validation', alpha=0.7)
    plt.axvline(probe_results['best_layer'], color='red', linestyle='--', alpha=0.5, label='Best Layer')
    plt.xlabel('Layer')
    plt.ylabel('R² Score (higher is better)')
    plt.title(f'Safety Entropy Regression - R² by Layer ({probe_type.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    best_layer_idx = probe_results['best_layer']
    best_result = probe_results['layer_results'][best_layer_idx]
    
    models_compared = ['Best Single\nLayer', 'Concatenated\n5 Layers']
    val_mses_compared = [best_result['val_mse'], concatenated_results['val_mse']]
    
    x = np.arange(len(models_compared))
    width = 0.35
    
    plt.bar(x - width/2, val_mses_compared, width, label='Validation', alpha=0.7)
    plt.xlabel('Model Type')
    plt.ylabel('MSE (lower is better)')
    plt.title(f'MSE Comparison: Single vs Concatenated ({probe_type.upper()})')
    plt.xticks(x, models_compared)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'safety_probe_results_{probe_type}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Train safety entropy probes for Llama 3B')
    parser.add_argument('--output_dir', default='test',
                       help='Base name for output directories under runs/ (default: test)')
    parser.add_argument('--concat_layers', type=int, nargs='+', default=[16, 20, 25, 27, 28],
                       help='Layer indices to concatenate (default: 0 1 2 3 4)')
    parser.add_argument('--alpha', type=float, default=0.75,
                       help='Alpha value for joint risk target calculation (default: 0.7)')
    
    # MLP-specific arguments
    parser.add_argument('--device', default='mps',
                       help='Device to use (mps, cuda, cpu)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for MLP (default: 256)')
    parser.add_argument('--activation', type=str, default='silu',
                       choices=['relu', 'gelu', 'tanh', 'silu'],
                       help='Activation function for MLP (default: relu)')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate for MLP (default: 0.1)')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='Learning rate for MLP training (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Maximum epochs for MLP training (default: 100)')
    parser.add_argument('--mlp_batch_size', type=int, default=128,
                       help='Batch size for MLP training (default: 32)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience for MLP (default: 10)')
    
    parser.add_argument('--cache_dir', type=str, default='feature_caches/llama3.2_train',
                       help='Directory for caching hidden states (default: feature_cache)')
    parser.add_argument('--ridge_alpha', type=float, default=1.0,
                       help='Ridge regression alpha for linear probes (default: 1.0)')
    parser.add_argument('--probe_name', type=str, default='safety_probe',
                       help='Base name for saved probe files (default: safety_probe)')

    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(66)
    np.random.seed(66)
    
   
    cache_dir = args.cache_dir
    hidden_state_path = f"{cache_dir}/hidden_states.pt"


    print("Loading hidden states from cache...")
    hidden_states_dict, metadata = load_hidden_states(hidden_state_path)
    args.model_name = metadata.get('model_name', 'unknown_model')
    safety_entropy_tbg, safety_score_tbg= load_labels(cache_dir)
    tbg_hidden_states = hidden_states_dict['tbg']

    # Compute joint risk target for training label
    print("Computing joint risk target for training labels...")
    alpha = args.alpha

    joint_risk_target = (alpha * safety_score_tbg) + ((1 - alpha) * safety_entropy_tbg)

    #select first 100 samples for quick testing
    #tbg_hidden_states = [layer_states[:100] for layer_states in tbg_hidden_states]
    #joint_risk_target = joint_risk_target[:100]

    print("-" * 80)
    print(f"Hidden states ready!")
    print(f"  TBG - Number of layers: {len(tbg_hidden_states)}")
    print(f"  TBG - Shape per layer: {tbg_hidden_states[0].shape}")
    print(f"  TBG - Total samples: {tbg_hidden_states[0].shape[0]}")


    print("-" * 80)
    

    
    # Determine device for probe training
    probe_device = torch.device(args.device)
    
    # Get the base test directory (always train all 4 variants)
    dir_name = args.output_dir if args.output_dir else "test"
    base_test_dir = get_next_test_directory(dir_name=dir_name)
    
    # ============================================================================
    # TRAIN MLP PROBES (Single-layer and Concatenated)
    # ============================================================================
    print("\n" + "="*80)
    print("TRAINING MLP PROBES")
    print("="*80)
    
    # Train single-layer MLP probes
    print("\n" + "="*80)
    print("Training single-layer MLP probes...")
    print("="*80)
    mlp_probe_results = train_layer_probes(
        tbg_hidden_states, joint_risk_target,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        mlp_batch_size=args.mlp_batch_size,
        patience=args.patience,
        probe_device=probe_device,
    )
    
    print(f"\nBest layer: {mlp_probe_results['best_layer']} with validation MSE: {mlp_probe_results['best_mse']:.4f}")
    mlp_best_layer_result = mlp_probe_results['layer_results'][mlp_probe_results['best_layer']]
    print(f"Val MSE: {mlp_best_layer_result['val_mse']:.4f}")
    print(f"Val R²: {mlp_best_layer_result['val_r2']:.4f}")
    
    # Train concatenated layers MLP probe
    print("\n" + "="*80)
    print(f"Training concatenated layers MLP probe (layers {args.concat_layers})...")
    print("="*80)
    mlp_concatenated_results = train_concatenated_layers_probe(
        tbg_hidden_states, joint_risk_target, 
        layer_indices=args.concat_layers,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        mlp_batch_size=args.mlp_batch_size,
        patience=args.patience,
        probe_device=probe_device
    )
    
    # Compare MLP results
    print("\n" + "="*80)
    print("MLP COMPARISON: Single Best Layer vs Concatenated Layers")
    print("="*80)
    print(f"\nSingle Best Layer (Layer {mlp_probe_results['best_layer']}):")
    print(f"  Val MSE: {mlp_best_layer_result['val_mse']:.4f}")
    print(f"  Val R²: {mlp_best_layer_result['val_r2']:.4f}")
    
    print(f"\nConcatenated Layers {args.concat_layers}:")
    print(f"  Val MSE: {mlp_concatenated_results['val_mse']:.4f}")
    print(f"  Val R²: {mlp_concatenated_results['val_r2']:.4f}")
    
    mlp_mse_improvement = ((mlp_best_layer_result['val_mse'] - mlp_concatenated_results['val_mse']) / 
                       mlp_best_layer_result['val_mse'] * 100)
    mlp_r2_improvement = mlp_concatenated_results['val_r2'] - mlp_best_layer_result['val_r2']
    
    print(f"\nImprovement:")
    print(f"  Val MSE: {mlp_mse_improvement:+.2f}% {'(better)' if mlp_mse_improvement > 0 else '(worse)'}")
    print(f"  Val R²: {mlp_r2_improvement:+.4f} {'(better)' if mlp_r2_improvement > 0 else '(worse)'}")
    print("="*80)
    
    # Save MLP results
    print("\nSaving MLP results...")
    mlp_output_dir = os.path.join(base_test_dir, "mlp")
    mlp_output_file, mlp_output_dir = save_results(mlp_probe_results, mlp_concatenated_results, 
                                                    args.model_name, mlp_output_dir, args, probe_type='mlp')
    
    # Create MLP visualizations
    print("Creating MLP visualizations...")
    create_visualizations(mlp_probe_results, mlp_concatenated_results, mlp_output_dir, args, probe_type='mlp')

    # ============================================================================
    # TRAIN LINEAR PROBES (Single-layer and Concatenated)
    # ============================================================================
    print("\n" + "="*80)
    print("TRAINING LINEAR PROBES")
    print("="*80)
    
    # Train single-layer linear probes
    print("\n" + "="*80)
    print("Training single-layer linear probes...")
    print("="*80)
    linear_probe_results = train_linear_layer_probes(
        tbg_hidden_states, joint_risk_target,
        ridge_alpha=args.ridge_alpha
    )
    
    print(f"\nBest layer: {linear_probe_results['best_layer']} with validation MSE: {linear_probe_results['best_mse']:.4f}")
    linear_best_layer_result = linear_probe_results['layer_results'][linear_probe_results['best_layer']]
    print(f"Val MSE: {linear_best_layer_result['val_mse']:.4f}")
    print(f"Val R²: {linear_best_layer_result['val_r2']:.4f}")
    
    # Train concatenated layers linear probe
    print("\n" + "="*80)
    print(f"Training concatenated layers linear probe (layers {args.concat_layers})...")
    print("="*80)
    linear_concatenated_results = train_concatenated_layers_linear_probe(
        tbg_hidden_states, joint_risk_target,
        layer_indices=args.concat_layers,
        ridge_alpha=args.ridge_alpha
    )
    
    # Compare linear results
    print("\n" + "="*80)
    print("LINEAR COMPARISON: Single Best Layer vs Concatenated Layers")
    print("="*80)
    print(f"\nSingle Best Layer (Layer {linear_probe_results['best_layer']}):") 
    print(f"  Val MSE: {linear_best_layer_result['val_mse']:.4f}")
    print(f"  Val R²: {linear_best_layer_result['val_r2']:.4f}")
    
    print(f"\nConcatenated Layers {args.concat_layers}:")
    print(f"  Val MSE: {linear_concatenated_results['val_mse']:.4f}")
    print(f"  Val R²: {linear_concatenated_results['val_r2']:.4f}")
    
    linear_mse_improvement = ((linear_best_layer_result['val_mse'] - linear_concatenated_results['val_mse']) / 
                           linear_best_layer_result['val_mse'] * 100)
    linear_r2_improvement = linear_concatenated_results['val_r2'] - linear_best_layer_result['val_r2']
    
    print(f"\nImprovement:")
    print(f"  Val MSE: {linear_mse_improvement:+.2f}% {'(better)' if linear_mse_improvement > 0 else '(worse)'}")
    print(f"  Val R²: {linear_r2_improvement:+.4f} {'(better)' if linear_r2_improvement > 0 else '(worse)'}")
    print("="*80)
    
    # Save linear results
    print("\nSaving linear results...")
    linear_output_dir = os.path.join(base_test_dir, "linear")
    linear_output_file, linear_output_dir = save_results(linear_probe_results, linear_concatenated_results,
                                                          args.model_name, linear_output_dir, args, probe_type='linear')
    
    # Create linear visualizations
    print("Creating linear visualizations...")
    create_visualizations(linear_probe_results, linear_concatenated_results, linear_output_dir, args, probe_type='linear')

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll models saved to: {base_test_dir}")
    print(f"  MLP probes: {os.path.join(base_test_dir, 'mlp')}")
    print(f"  Linear probes: {os.path.join(base_test_dir, 'linear')}")
    
    # Cross-comparison
    print("\n" + "="*80)
    print("CROSS-COMPARISON: MLP vs LINEAR (Best Concatenated Models)")
    print("="*80)
    print(f"\nMLP Concatenated:")
    print(f"  Val MSE: {mlp_concatenated_results['val_mse']:.4f}")
    print(f"  Val R²: {mlp_concatenated_results['val_r2']:.4f}")
    
    print(f"\nLinear Concatenated:")
    print(f"  Val MSE: {linear_concatenated_results['val_mse']:.4f}")
    print(f"  Val R²: {linear_concatenated_results['val_r2']:.4f}")
    
    cross_mse_improvement = ((linear_concatenated_results['val_mse'] - mlp_concatenated_results['val_mse']) / 
                             linear_concatenated_results['val_mse'] * 100)
    cross_r2_improvement = mlp_concatenated_results['val_r2'] - linear_concatenated_results['val_r2']
    
    print(f"\nMLP vs Linear:")
    print(f"  MSE Improvement: {cross_mse_improvement:+.2f}% {'(MLP better)' if cross_mse_improvement > 0 else '(Linear better)'}")
    print(f"  R² Improvement: {cross_r2_improvement:+.4f} {'(MLP better)' if cross_r2_improvement > 0 else '(Linear better)'}")
    print("="*80)

    
    

if __name__ == "__main__":
    main()
