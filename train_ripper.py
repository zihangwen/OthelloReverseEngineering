
import os
import numpy as np
import pandas as pd
import wittgenstein as lw
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import Parallel, delayed
import json
import torch as t
import sys

# Add parent directory to path to import circuits modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuits import othello_utils, utils
from circuits.eval_sae_as_classifier import construct_othello_dataset

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Data loading
def load_ground_truth_data_and_activations(dataset_size=6000, target_layer=5):
    """Load ground truth features and target layer activations directly"""
    print("Loading model...")
    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    model = utils.get_model(model_name, device=device)
    
    print("Loading dataset and generating ground truth features...")
    # Use the same custom function as decision tree training
    custom_functions = [othello_utils.games_batch_to_board_state_flipped_played_BLC]
    
    test_data = construct_othello_dataset(
        custom_functions=custom_functions,
        split="test", 
        device=device, 
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
            game_tensor = t.tensor(game_tokens[5:30], device=device).unsqueeze(0)  # moves 5-30
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

# Train RIPPER on one neuron
def train_ripper_on_neuron(ground_truth_features, layer_activations, target_layer, target_neuron, sample_size=50000):
    """Train RIPPER on a specific neuron using ground truth features"""
    X = ground_truth_features  # Ground truth features (same for all layers)
    Y = layer_activations[:, target_neuron]  # Target neuron activations

    # Binary labels: Strong (1) = top 10% activations
    p90 = np.percentile(Y, 90)
    y_binary = (Y >= p90).astype(int)

    # Sample for efficiency
    n_samples = min(sample_size, len(Y))
    idx = np.random.choice(len(Y), size=n_samples, replace=False)
    X_sampled, y_sampled = X[idx], y_binary[idx]

    # Train/test split (consistent with decision tree training: 80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, test_size=0.2, stratify=y_sampled, random_state=42
    )

    # Convert to DataFrame (required by wittgenstein)
    df_train, df_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
    df_train['label'], df_test['label'] = y_train, y_test

    ripper = lw.RIPPER()
    ripper.fit(df_train.drop(columns=['label']), df_train['label'], pos_class=1)

    # Evaluate
    y_pred = ripper.predict(df_test.drop(columns=['label']))
    report = classification_report(df_test['label'], y_pred, target_names=['Non-Strong', 'Strong'], output_dict=True)

    return {
        "layer": target_layer,
        "neuron": target_neuron,
        "rules": str(ripper.ruleset_),
        "threshold": float(p90),
        "class_balance": {
            "non_strong": int((y_binary == 0).sum()),
            "strong": int((y_binary == 1).sum())
        },
        "test_report": report
    }

def train_and_collect(layer, neuron, ground_truth_features, layer_activations):
    try:
        return train_ripper_on_neuron(ground_truth_features, layer_activations, layer, neuron)
    except Exception as e:
        return {"layer": layer, "neuron": neuron, "error": str(e)}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, help="Which layer to train on (0-7)")
    parser.add_argument("--cores", type=int, default=32, help="Number of CPU cores")
    parser.add_argument("--neurons", type=int, default=2048, help="Total neurons in layer")
    args = parser.parse_args()

    # Load data with ground truth features 
    ground_truth_features, layer_activations = load_ground_truth_data_and_activations(target_layer=args.layer)

    if ground_truth_features is None or layer_activations is None:
        print(f"Error: Could not load data for layer {args.layer}")
        exit(1)

    # Get all neurons in this layer
    assigned_neurons = list(range(args.neurons))
    print(f"Machine assigned: Layer {args.layer}, Neurons 0 → {args.neurons-1}")
    print(f"Using ground truth features (same for all layers): {ground_truth_features.shape}")
    print(f"Layer {args.layer} activations: {layer_activations.shape}")
    
    #results = train_and_collect(args.layer, 766, ground_truth_features, layer_activations)

    # Train in parallel across neurons
    results = Parallel(n_jobs=args.cores, verbose=5)(
        delayed(train_and_collect)(args.layer, n, ground_truth_features, layer_activations)
        for n in assigned_neurons
    )

    # Save results
    os.makedirs("ripper_results", exist_ok=True)
    out_path = f"ripper_results/layer{args.layer}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results for Layer {args.layer} to {out_path}")