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
import warnings
warnings.filterwarnings('ignore')


class MLPProbe(nn.Module):
    """Two-layer MLP probe with activation function for regression."""
    def __init__(self, input_dim, hidden_dim=256, activation='relu', dropout=0.1):
        super(MLPProbe, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, 1)  # Output single value for regression
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x.squeeze(-1)  # Remove last dimension for scalar output


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


def load_hidden_states(cache_path):
    """
    Load hidden states from disk.
    
    Args:
        cache_path: Path to the cache file
        
    Returns:
        hidden_states_dict: Dictionary with 'tbg' and 'slt' keys
        metadata: Dictionary with metadata
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    cache_data = torch.load(cache_path)
    print(f"Hidden states loaded from {cache_path}")
    print(f"  Layers: {cache_data['num_layers']}")
    print(f"  Samples: {cache_data['num_samples']}")
    print(f"  Hidden dim: {cache_data['hidden_dim']}")
    
    if cache_data['metadata']:
        print(f"  Metadata: {cache_data['metadata']}")
    
    hidden_states_dict = {
        'tbg': cache_data['tbg_states'],
        'slt': cache_data['slt_states']
    }
    
    return hidden_states_dict, cache_data['metadata']

def load_labels(cache_dir):
    #loads the npy files
    safety_entropy = np.load(os.path.join(cache_dir, "safety_entropy_labels.npy"))
    safety_score = np.load(os.path.join(cache_dir, "safety_score_labels.npy"))
    return safety_entropy, safety_score


def train_layer_probes(hidden_states, labels, test_size=0.2, val_size=0.2, 
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
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")
    
    for layer_idx, layer_states in enumerate(tqdm(hidden_states, desc="Training probes")):
        # Convert to numpy
        X = layer_states.numpy()
        
        # Split data
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
        

        model = train_mlp_probe(
            X_train, y_train, X_val, y_val,
            hidden_dim=hidden_dim, activation=activation, dropout=dropout,
            learning_rate=learning_rate, num_epochs=num_epochs,
            batch_size=mlp_batch_size, patience=patience, device=probe_device
        )
            
            # Predict
        train_pred = predict_mlp(model, X_train, batch_size=mlp_batch_size, device=probe_device)
        val_pred = predict_mlp(model, X_val, batch_size=mlp_batch_size, device=probe_device)
        test_pred = predict_mlp(model, X_test, batch_size=mlp_batch_size, device=probe_device)
       
        
        # Evaluate with regression metrics
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        layer_result = {
            'layer': layer_idx,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2
        }
        
        results['layer_results'].append(layer_result)
        results['models'].append(model)
        
        # Track best validation performance (lowest MSE)
        if val_mse < results['best_mse']:
            results['best_mse'] = val_mse
            results['best_layer'] = layer_idx
        
        print(f"Layer {layer_idx:2d}: Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, Test MSE: {test_mse:.4f}, Val R²: {val_r2:.4f}")
    
    return results


def train_concatenated_layers_probe(hidden_states, labels, layer_indices=[0, 1, 2, 3, 4], 
                                   test_size=0.2, val_size=0.2, hidden_dim=256, 
                                   activation='relu', dropout=0.1, learning_rate=0.001, 
                                   num_epochs=100, mlp_batch_size=32, patience=10, probe_device='mps'):
    """
    Train a probe on concatenated features from multiple layers.
    
    Args:
        hidden_states: List of tensors, one per layer
        labels: Target labels (safety entropy values)
        layer_indices: List of layer indices to concatenate (default: first 5 layers)
        test_size: Fraction of data for testing
        val_size: Fraction of remaining data for validation
        use_mlp: If True, use 2-layer MLP; if False, use linear regression
        hidden_dim: Hidden dimension for MLP
        activation: Activation function for MLP ('relu', 'gelu', 'tanh', 'silu')
        dropout: Dropout rate for MLP
        learning_rate: Learning rate for MLP training
        num_epochs: Maximum epochs for MLP training
        mlp_batch_size: Batch size for MLP training
        patience: Early stopping patience for MLP
        probe_device: Device for MLP training
    
    Returns:
        Dictionary with results and trained model
    """
    print(f"\nTraining probe on concatenated layers: {layer_indices}")
    
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
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size/(1-test_size), random_state=42)
    
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    

    model = train_mlp_probe(
        X_train, y_train, X_val, y_val,
        hidden_dim=hidden_dim, activation=activation, dropout=dropout,
        learning_rate=learning_rate, num_epochs=num_epochs,
        batch_size=mlp_batch_size, patience=patience, device=probe_device
    )
        
    # Predict
    train_pred = predict_mlp(model, X_train, batch_size=mlp_batch_size, device=probe_device)
    val_pred = predict_mlp(model, X_val, batch_size=mlp_batch_size, device=probe_device)
    test_pred = predict_mlp(model, X_test, batch_size=mlp_batch_size, device=probe_device)

    # Evaluate
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    results = {
        'layer_indices': layer_indices,
        'model': model,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'feature_dim': X.shape[1]
    }
    
    print(f"\nConcatenated Layers {layer_indices} Results:")
    print(f"  Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"  Val   MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    print(f"  Test  MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    return results


def save_results(probe_results, concatenated_results, model_name, n_samples, output_dir="models/safety_probes"):
    """Save the trained probes and results (both single-layer and concatenated)."""
    os.makedirs(output_dir, exist_ok=True)
    
    save_data = {
        'model_name': model_name,
        'best_layer': probe_results['best_layer'],
        'best_model': probe_results['models'][probe_results['best_layer']],
        'all_models': probe_results['models'],
        'results': probe_results['layer_results'],
        'n_samples': n_samples,
        'probe_model_type': probe_results.get('model_type', 'linear'),  # 'linear' or 'mlp'
        
        # Add concatenated model information
        'concatenated_model': concatenated_results['model'],
        'concatenated_model_type': concatenated_results.get('model_type', 'linear'),
        'concatenated_layer_indices': concatenated_results['layer_indices'],
        'concatenated_results': {
            'train_mse': concatenated_results['train_mse'],
            'val_mse': concatenated_results['val_mse'],
            'test_mse': concatenated_results['test_mse'],
            'train_mae': concatenated_results['train_mae'],
            'val_mae': concatenated_results['val_mae'],
            'test_mae': concatenated_results['test_mae'],
            'train_r2': concatenated_results['train_r2'],
            'val_r2': concatenated_results['val_r2'],
            'test_r2': concatenated_results['test_r2'],
            'feature_dim': concatenated_results['feature_dim']
        }
    }
    
    # Add model type suffix to filename
    model_type_suffix = '_mlp' 
    output_file = os.path.join(output_dir, f'llama3b_safety_probes{model_type_suffix}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Safety probes saved to: {output_file}")
    return output_file


def create_visualizations(probe_results, concatenated_results, output_dir="models/safety_probes"):
    """Create and save visualization plots for regression probes."""
    layer_nums = [r['layer'] for r in probe_results['layer_results']]
    train_mses = [r['train_mse'] for r in probe_results['layer_results']]
    val_mses = [r['val_mse'] for r in probe_results['layer_results']]
    test_mses = [r['test_mse'] for r in probe_results['layer_results']]
    
    train_r2s = [r['train_r2'] for r in probe_results['layer_results']]
    val_r2s = [r['val_r2'] for r in probe_results['layer_results']]
    test_r2s = [r['test_r2'] for r in probe_results['layer_results']]
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(layer_nums, train_mses, 'o-', label='Train', alpha=0.7)
    plt.plot(layer_nums, val_mses, 's-', label='Validation', alpha=0.7)
    plt.plot(layer_nums, test_mses, '^-', label='Test', alpha=0.7)
    plt.axvline(probe_results['best_layer'], color='red', linestyle='--', alpha=0.5, label='Best Layer')
    plt.xlabel('Layer')
    plt.ylabel('MSE (lower is better)')
    plt.title('Safety Entropy Regression - MSE by Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(layer_nums, train_r2s, 'o-', label='Train', alpha=0.7)
    plt.plot(layer_nums, val_r2s, 's-', label='Validation', alpha=0.7)
    plt.plot(layer_nums, test_r2s, '^-', label='Test', alpha=0.7)
    plt.axvline(probe_results['best_layer'], color='red', linestyle='--', alpha=0.5, label='Best Layer')
    plt.xlabel('Layer')
    plt.ylabel('R² Score (higher is better)')
    plt.title('Safety Entropy Regression - R² by Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    best_layer_idx = probe_results['best_layer']
    best_result = probe_results['layer_results'][best_layer_idx]
    
    models_compared = ['Best Single\nLayer', 'Concatenated\n5 Layers']
    test_mses_compared = [best_result['test_mse'], concatenated_results['test_mse']]
    val_mses_compared = [best_result['val_mse'], concatenated_results['val_mse']]
    
    x = np.arange(len(models_compared))
    width = 0.35
    
    plt.bar(x - width/2, val_mses_compared, width, label='Validation', alpha=0.7)
    plt.bar(x + width/2, test_mses_compared, width, label='Test', alpha=0.7)
    plt.xlabel('Model Type')
    plt.ylabel('MSE (lower is better)')
    plt.title('MSE Comparison: Single vs Concatenated')
    plt.xticks(x, models_compared)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'safety_probe_results.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train safety entropy probes for Llama 3B')
    parser.add_argument('--model_name', default='meta-llama/Llama-3.2-3B-Instruct', 
                       help='Hugging Face model name')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for hidden state extraction (default: 4)')
    parser.add_argument('--output_dir', default='models/safety_probes',
                       help='Output directory for saved models')
    parser.add_argument('--concat_layers', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help='Layer indices to concatenate (default: 0 1 2 3 4)')
    
    # MLP-specific arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for MLP (default: 256)')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'gelu', 'tanh', 'silu'],
                       help='Activation function for MLP (default: relu)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for MLP (default: 0.1)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for MLP training (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Maximum epochs for MLP training (default: 100)')
    parser.add_argument('--mlp_batch_size', type=int, default=32,
                       help='Batch size for MLP training (default: 32)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience for MLP (default: 10)')
    
    parser.add_argument('--cache_dir', type=str, default='feature_cache',
                       help='Directory for caching hidden states (default: feature_cache)')

    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
   
    cache_dir = args.cache_dir
    hidden_state_path = f"{cache_dir}/hidden_states_tbg_slt.pt"


    print("Loading hidden states from cache...")
    hidden_states_dict, metadata = load_hidden_states(hidden_state_path)
    safety_entropy, safety_score = load_labels(cache_dir)
    tbg_hidden_states = hidden_states_dict['tbg']
    slt_hidden_states = hidden_states_dict['slt']
        

    print("-" * 80)
    print(f"Hidden states ready!")
    print(f"  TBG - Number of layers: {len(tbg_hidden_states)}")
    print(f"  TBG - Shape per layer: {tbg_hidden_states[0].shape}")
    print(f"  TBG - Total samples: {tbg_hidden_states[0].shape[0]}")
    print(f"  SLT - Number of layers: {len(slt_hidden_states)}")
    print(f"  SLT - Shape per layer: {slt_hidden_states[0].shape}")
    print(f"  SLT - Total samples: {slt_hidden_states[0].shape[0]}")

    print(f"Label shape: {safety_score.shape}")
    print("-" * 80)
    

    
    # Determine device for probe training
    probe_device = torch.device('mps')
    
    # Train single-layer probes
    print("\n" + "="*80)
    print("Training single-layer probes...")
    print("="*80)
    probe_results = train_layer_probes(
        tbg_hidden_states, safety_score,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        mlp_batch_size=args.mlp_batch_size,
        patience=args.patience,
        probe_device=probe_device
    )
    
    print(f"\nBest layer: {probe_results['best_layer']} with validation MSE: {probe_results['best_mse']:.4f}")
    best_layer_result = probe_results['layer_results'][probe_results['best_layer']]
    print(f"Test MSE: {best_layer_result['test_mse']:.4f}")
    print(f"Test R²: {best_layer_result['test_r2']:.4f}")
    
    # Train concatenated layers probe
    print("\n" + "="*80)
    print(f"Training concatenated layers probe (layers {args.concat_layers})...")
    print("="*80)
    concatenated_results = train_concatenated_layers_probe(
        tbg_hidden_states, safety_score, 
        layer_indices=args.concat_layers,
        use_mlp=args.use_mlp,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        mlp_batch_size=args.mlp_batch_size,
        patience=args.patience,
        probe_device=probe_device
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON: Single Best Layer vs Concatenated Layers")
    print("="*80)
    print(f"\nSingle Best Layer (Layer {probe_results['best_layer']}):")
    print(f"  Val MSE: {best_layer_result['val_mse']:.4f}")
    print(f"  Test MSE: {best_layer_result['test_mse']:.4f}")
    print(f"  Test R²: {best_layer_result['test_r2']:.4f}")
    
    print(f"\nConcatenated Layers {args.concat_layers}:")
    print(f"  Val MSE: {concatenated_results['val_mse']:.4f}")
    print(f"  Test MSE: {concatenated_results['test_mse']:.4f}")
    print(f"  Test R²: {concatenated_results['test_r2']:.4f}")
    
    mse_improvement = ((best_layer_result['test_mse'] - concatenated_results['test_mse']) / 
                       best_layer_result['test_mse'] * 100)
    r2_improvement = concatenated_results['test_r2'] - best_layer_result['test_r2']
    
    print(f"\nImprovement:")
    print(f"  Test MSE: {mse_improvement:+.2f}% {'(better)' if mse_improvement > 0 else '(worse)'}")
    print(f"  Test R²: {r2_improvement:+.4f} {'(better)' if r2_improvement > 0 else '(worse)'}")
    print("="*80)
    
    # Save results
    print("\nSaving results...")
    output_file = save_results(probe_results, concatenated_results, args.model_name, len(safety_score), args.output_dir)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(probe_results, concatenated_results, args.output_dir)
    
    print("\nTraining completed successfully!")
    print(f"Models saved to: {args.output_dir}")
    

if __name__ == "__main__":
    main()
