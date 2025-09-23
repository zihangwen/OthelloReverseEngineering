"""
Train per-square is_placed probes for each layer of OthelloGPT.

For each layer and each square position (64 total), trains a separate 
logistic regression probe to detect if that square was just played.
Evaluates F1 score per layer by aggregating predictions across all squares.
"""

import json
from dataclasses import dataclass
from joblib import Parallel, delayed

import numpy as np
import torch as t
from torch import Tensor
import einops
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformer_lens import HookedTransformer

import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from typing import Dict, List, Tuple

t.set_grad_enabled(False)

# ============ Data Structures ============


@dataclass
class ProbeResults:
    """Results for a single square's probe."""
    layer: int
    square: int
    probe: LogisticRegression
    train_f1: float
    test_f1: float


# ============ Data Loading ============


def load_model_and_data(model_name: str = "Baidicoot/Othello-GPT-Transformer-Lens", n_train: int = 100000, n_test: int = 10000, device: str = "cuda") -> Tuple[HookedTransformer, Tensor, np.ndarray, Tensor, np.ndarray]:
    """
    Returns:
        model: OthelloGPT model
        train_data: Training game data
        train_labels: Training is_placed labels (n_samples, seq_len, 64)
        test_data: Test game data
        test_labels: Test is_placed labels
    """
    model = utils.get_model(model_name, device)

    train_data = construct_othello_dataset(
        custom_functions=[],
        n_inputs=n_train,
        split="train",
        device="cpu",
    )
    train_encoded_inputs = t.tensor(train_data["encoded_inputs"], dtype=t.long, device="cpu")
    train_decoded_inputs = np.array(train_data["decoded_inputs"])
    train_labels = np.zeros((n_train, model.cfg.n_ctx, 64))
    batch_indices = einops.repeat(np.arange(n_train), "n_train -> n_train seq", seq = model.cfg.n_ctx)
    seq_indices = einops.repeat(np.arange(model.cfg.n_ctx), "seq -> n_train seq", n_train = n_train)
    train_labels[batch_indices, seq_indices, train_decoded_inputs] = 1

    test_data = construct_othello_dataset(
        custom_functions=[],
        n_inputs=n_test,
        split="test",
        device="cpu",
    )
    test_encoded_inputs = t.tensor(test_data["encoded_inputs"], dtype=t.long, device="cpu")
    test_decoded_inputs = np.array(test_data["decoded_inputs"])
    test_labels = np.zeros((n_test, model.cfg.n_ctx, 64))
    batch_indices = einops.repeat(np.arange(n_test), "n_test -> n_test seq", seq = model.cfg.n_ctx)
    seq_indices = einops.repeat(np.arange(model.cfg.n_ctx), "seq -> n_test seq", n_test = n_test)
    test_labels[batch_indices, seq_indices, test_decoded_inputs] = 1

    return model, train_encoded_inputs, train_labels, test_encoded_inputs, test_labels
    

# ============ Feature Extraction ============


def extract_activations_by_layer(
    model: HookedTransformer, data : Tensor, batch_size : int = 256, device = "cuda"
) -> Dict[int, np.ndarray]:
    """
    Extract activations from each layer of the model.

    Args:
        model: OthelloGPT model
        data: train or test encoded inputs

    Returns:
        Dict mapping layer_idx -> activations array (n_games, n_tokens, hidden_dim)
    """
    keys = [f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)]
    activations = {layer : np.empty((data.shape[0], data.shape[1], model.cfg.d_model)) for layer in range(model.cfg.n_layers)}

    for i in range(0, len(data), batch_size):
        batch_inputs = data[i : i + batch_size].to(device)
        _, cache = model.run_with_cache(
            batch_inputs,
            names_filter=lambda name: name in keys,
        )

        for layer, key in enumerate(keys):
            activations[layer][i : i + batch_size] = cache[key].detach().cpu().numpy()
        
    return activations


# ============ Probe Training ============


def prepare_probe_data(
    activations: Dict[int, np.ndarray], labels: np.ndarray, layer: int, square: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training a probe for a specific square.

    Args:
        activations: Layer activations (n_samples, n_tokens, hidden_dim)
        labels: Is_placed labels (n_samples, n_tokens, 64)
        square_idx: Which square (0-63) to train probe for

    Returns:
        X: Flattened activations (n_games * n_tokens, hidden_dim)
        y: Binary labels for this square (n_games * n_tokens,)
    """
    X = einops.rearrange(activations[layer], "batch seq d_model -> (batch seq) d_model")
    y = labels[..., square].flatten()
    return X, y


def train_single_probe(
    train_activations: Dict[int, np.ndarray], 
    test_activations: Dict[int, np.ndarray],
    train_labels: np.ndarray, 
    test_labels,
    layer: int, 
    square: int
) -> ProbeResults:
    """
    Train and evaluate a single probe.

    Returns:
        ProbeResults with trained probe and metrics
    """
    X_train, y_train = prepare_probe_data(train_activations, train_labels, layer=layer, square=square)
    X_test, y_test = prepare_probe_data(test_activations, test_labels, layer=layer, square=square)

    probe = LogisticRegression(
        fit_intercept=False,
        #class_weight='balanced',
        random_state=42,
    )

    probe.fit(X_train, y_train)

    train_preds = probe.predict(X_train)
    train_f1 = f1_score(y_train, train_preds)

    test_preds = probe.predict(X_test)
    test_f1 = f1_score(y_test, test_preds)

    return ProbeResults(
        layer=layer, 
        square=square, 
        probe=probe, 
        train_f1=train_f1, 
        test_f1=test_f1,
    )


def train_probes_for_layer(
    train_activations: Dict[int, np.ndarray], 
    test_activations: Dict[int, np.ndarray],
    train_labels: np.ndarray, 
    test_labels,
    layer: int, 
    n_jobs: int = -1
) -> List[ProbeResults]:
    """
    Train probes for all 64 squares at a specific layer.

    Args:
        layer_idx: Which layer we're training probes for
        layer_activations: Activations from this layer
        train_labels: Training labels for all squares
        test_labels: Test labels for all squares

    Returns:
        List of 64 ProbeResults, one per square
    """
    # Skip the 4 center squares that are never played in Othello
    squares_to_probe = [i for i in range(64) if i not in [27, 28, 35, 36]]

    results = Parallel(n_jobs=n_jobs)(
        delayed(train_single_probe)(
            train_activations, 
            test_activations, 
            train_labels, 
            test_labels, 
            layer, 
            square,
        )
        for square in squares_to_probe
    )
    return results


def train_all_probes(
    train_activations: Dict[int, np.ndarray],
    test_activations: Dict[int, np.ndarray],
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    layers_to_probe: List[int] = None,
    n_jobs: int = -1
) -> Dict[int, List[ProbeResults]]:
    """
    Train probes for all layers and squares.
    """
    if layers_to_probe is None:
        layers_to_probe = list(train_activations.keys())
    
    results = {}
    for layer in layers_to_probe:
        print(f"Training probes for layer {layer}...")
        results[layer] = train_probes_for_layer(
            train_activations,
            test_activations, 
            train_labels,
            test_labels,
            layer,
            n_jobs=n_jobs
        )
    
    return results

if __name__ == "__main__":
    model, train_data, train_labels, test_data, test_labels = load_model_and_data(n_train = 10000, n_test = 10000)

    train_activations = extract_activations_by_layer(model, train_data)
    test_activations = extract_activations_by_layer(model, test_data)

    results = train_all_probes(train_activations, test_activations, train_labels, test_labels)

    # Save
    for layer in range(model.cfg.n_layers):
        coef_dict = {probe.square: probe.probe.coef_.squeeze() for probe in results[layer]}

        # insert -1000, so sigmoid viz will show blank
        probe = np.stack([
            coef_dict.get(i, np.zeros(model.cfg.d_model)) for i in range(64)
        ], axis=-1)

        probe = einops.rearrange(probe, "d_model (row col) -> d_model row col", row=8)

        probe_tensor = t.from_numpy(probe).float()
        t.save(probe_tensor, f"resid_{layer}_played.pth")
        
        print(f"Saved layer {layer} probe to played_probes/resid_{layer}_played.pth")


    # Calculate layer-wise F1 averages
    f1_summary = {}
    for layer in range(model.cfg.n_layers):
        train_f1s = [probe.train_f1 for probe in results[layer]]
        test_f1s = [probe.test_f1 for probe in results[layer]]
        
        f1_summary[f"layer_{layer}"] = {
            "train_f1_mean": np.mean(train_f1s),
            "train_f1_std": np.std(train_f1s),
            "test_f1_mean": np.mean(test_f1s),
            "test_f1_std": np.std(test_f1s),
            "n_squares": len(results[layer])  # Should be 60
        }

    # Save to JSON
    with open("f1_summary.json", "w") as f:
        json.dump(f1_summary, f, indent=2)