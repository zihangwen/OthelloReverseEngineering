import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import pickle
import os
import argparse
import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from helper_fns import create_feature_names

# ======================================================
# Sparse Linear Model (all neurons)
# ======================================================
class SparseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    def get_weights(self):
        return (self.linear.weight.detach().cpu().numpy(),
                self.linear.bias.detach().cpu().numpy())

# ======================================================
# Train sparse model for ALL neurons
# ======================================================
def train_all_neurons(X: np.ndarray, y: np.ndarray,
                      feature_names: list, layer: int,
                      l1_lambda=1e-3, lr=1e-3, epochs=200, top_k=10, batch_size=32):
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print(f"Input features: {X_train_scaled.shape[1]}")
    print(f"Output neurons: {y_train.shape[1]}")
    
    # Clear GPU memory before creating model
    if t.cuda.is_available():
        t.cuda.empty_cache()
    
    model = SparseLinear(X_train_scaled.shape[1], y_train.shape[1]).to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    if device == "cuda":
        print(f"GPU memory allocated: {t.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Ensure gradients are enabled
    t.set_grad_enabled(True)

    # Create DataLoaders
    train_dataset = TensorDataset(
        t.tensor(X_train_scaled, dtype=t.float32),
        t.tensor(y_train, dtype=t.float32)
    )
    test_dataset = TensorDataset(
        t.tensor(X_test_scaled, dtype=t.float32),
        t.tensor(y_test, dtype=t.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training tracking
    train_r2s = []
    test_r2s = []

    print("Training progress:")
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            
            # MSE loss
            mse_loss = criterion(pred, batch_y)
            
            # L1 regularization on the linear layer weights
            l1_loss = l1_lambda * model.linear.weight.abs().sum()
            
            total_loss = mse_loss + l1_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
        
        # Evaluation
        model.eval()
        with t.no_grad():
            # Train evaluation
            train_preds = []
            train_targets = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                pred = model(batch_x).cpu()
                train_preds.append(pred)
                train_targets.append(batch_y)
            
            train_preds = t.cat(train_preds, dim=0).numpy()
            train_targets = t.cat(train_targets, dim=0).numpy()
            
            # Test evaluation
            test_preds = []
            test_targets = []
            test_loss = 0
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                test_loss += criterion(pred, batch_y).item()
                test_preds.append(pred.cpu())
                test_targets.append(batch_y.cpu())
            
            test_preds = t.cat(test_preds, dim=0).numpy()
            test_targets = t.cat(test_targets, dim=0).numpy()
        
        # Calculate R² scores (average across all neurons)
        train_r2 = r2_score(train_targets.flatten(), train_preds.flatten())
        test_r2 = r2_score(test_targets.flatten(), test_preds.flatten())
        
        train_r2s.append(train_r2)
        test_r2s.append(test_r2)
        
        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

    # Final evaluation
    model.eval()
    with t.no_grad():
        final_test_preds = []
        final_test_targets = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu()
            final_test_preds.append(pred)
            final_test_targets.append(batch_y)
    
    final_test_preds = t.cat(final_test_preds, dim=0).numpy()
    final_test_targets = t.cat(final_test_targets, dim=0).numpy()
    
    # Final R² score (average across all neurons)
    final_test_r2 = r2_score(final_test_targets.flatten(), final_test_preds.flatten())

    weights, bias = model.get_weights()
    sparsity = np.mean(weights == 0.0)

    print(f"\n==== Results for Layer {layer} (All {weights.shape[0]} neurons) ====")
    print(f"Final Test R² = {final_test_r2:.4f}")
    print(f"Sparsity = {sparsity*100:.1f}%")
    print(f"Weight matrix shape: {weights.shape}")

    # Calculate per-neuron R² scores
    per_neuron_r2 = []
    for neuron_idx in range(weights.shape[0]):
        neuron_r2 = r2_score(final_test_targets[:, neuron_idx], final_test_preds[:, neuron_idx])
        per_neuron_r2.append(neuron_r2)
    
    # Find top performing neurons
    top_neuron_idx = np.argsort(per_neuron_r2)[-10:][::-1]
    print(f"\nTop 10 neurons by R²:")
    for rank, neuron_idx in enumerate(top_neuron_idx, 1):
        print(f"  {rank}. Neuron {neuron_idx}: R²={per_neuron_r2[neuron_idx]:.4f}")

    return {
        "layer": layer,
        "weights": weights,  # Shape: (2048, 320) - numpy array
        "bias": bias,  # Shape: (2048,) - numpy array
        "test_r2": final_test_r2,  # Overall R² score
        "sparsity": sparsity,  # Overall sparsity
        "per_neuron_r2": per_neuron_r2,  # R² score for each neuron
    }

# ======================================================
# Data loader for all layers
# ======================================================
def load_ground_truth_data_and_activations(dataset_size=6000, target_layer=5):
    """Load ground truth features and target layer activations directly"""
    print("Loading model...")
    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    model = utils.get_model(model_name, device="cpu")
    
    print("Loading dataset and generating ground truth features...")
    # Use the same custom function as decision tree training
    custom_functions = [othello_utils.games_batch_to_board_state_flipped_played_BLC]
    
    test_data = construct_othello_dataset(
        custom_functions=custom_functions,
        split="test", 
        device="cpu", 
        n_inputs=dataset_size
    )
    
    encoded_inputs = test_data["encoded_inputs"]
    ground_truth_features = test_data[othello_utils.games_batch_to_board_state_flipped_played_BLC.__name__]
    
    # Generate target layer activations
    print(f"Generating activations for Layer {target_layer}...")
    neuron_acts = []
    ground_truth_samples = []
    
    for i, game_tokens in enumerate(encoded_inputs):
        if len(game_tokens) >= 30:  # Use same move range as decision tree training
            game_tensor = t.tensor(game_tokens[5:30], device="cpu").unsqueeze(0)  # moves 5-30
            with t.no_grad():
                with model.trace(game_tensor, scan=False, validate=False):
                    mlp_acts = model.blocks[target_layer].mlp.hook_post.output.save()
            neuron_acts.append(mlp_acts.squeeze(0))
            
            # Get corresponding ground truth features for the same games/moves
            game_features = ground_truth_features[i, 5:30, :]
            ground_truth_samples.append(game_features.cpu().numpy())
    
    if neuron_acts:
        # Concatenate all activations and ground truth features
        activations = t.cat(neuron_acts, dim=0).cpu().numpy()
        ground_truth = np.concatenate(ground_truth_samples, axis=0)
        
        print(f"Ground truth features shape: {ground_truth.shape}")
        print(f"Layer {target_layer} activations shape: {activations.shape}")
        
        return ground_truth, activations
    else:
        return None, None



# ======================================================
# MAIN: Train All Layers
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, default=6000, help="Dataset size")
    args = parser.parse_args()

    # Train all layers 0-7
    layers_to_train = list(range(8))  # 0-7
    neurons_per_layer = 2048
    
    print(f"Dataset size: {args.dataset_size}")
    
    # Load ground truth features once (same for all layers)
    print("\nLoading ground truth features...")
    ground_truth_features, _ = load_ground_truth_data_and_activations(
        dataset_size=args.dataset_size, target_layer=0  # Just to get ground truth features
    )
    
    if ground_truth_features is None:
        print("Error: Could not load ground truth features")
        exit(1)
    
    print(f"Ground truth features shape: {ground_truth_features.shape}")
    
    # Create feature names once
    func_name = othello_utils.games_batch_to_board_state_flipped_played_BLC.__name__
    feature_names = create_feature_names(ground_truth_features.shape[1], func_name)
    
    # Train each layer
    all_results = {}
    for layer_idx, layer in enumerate(layers_to_train):
        print(f"\n{'='*60}")
        print(f"TRAINING LAYER {layer} ({layer_idx + 1}/{len(layers_to_train)})")
        print(f"{'='*60}")
        
        # Load layer-specific activations
        _, layer_activations = load_ground_truth_data_and_activations(
            dataset_size=args.dataset_size, target_layer=layer
        )
        
        if layer_activations is None:
            print(f"Error: Could not load activations for layer {layer}")
            continue
            
        print(f"Layer {layer} activations shape: {layer_activations.shape}")
        
        # Train model for this layer
        print(f"Training model for layer {layer}...")
        results = train_all_neurons(
            ground_truth_features, layer_activations, feature_names, layer,
            l1_lambda=1e-4, lr=1e-3, epochs=100, top_k=10, batch_size=128
        )
        
        # Store results
        all_results[f"layer_{layer}"] = results
        
        # Save individual layer results
        os.makedirs("lasso_results", exist_ok=True)
        layer_out_path = f"lasso_results/layer{layer}_results.pkl"
        with open(layer_out_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved layer {layer} results to {layer_out_path}")
        
        # Clear GPU memory
        if t.cuda.is_available():
            t.cuda.empty_cache()
        
        print(f"Layer {layer} completed")
    
    # Print summary
    print(f"\nSummary:")
    for layer in layers_to_train:
        if f"layer_{layer}" in all_results:
            results = all_results[f"layer_{layer}"]
            print(f"  Layer {layer}: Test R² = {results['test_r2']:.4f}, Sparsity = {results['sparsity']*100:.1f}%")