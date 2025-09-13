"""
Train decision tree on neuron using continuous features
of board state projection onto probe directions
"""

import torch as t
from torch import Tensor
import numpy as np
import einops
from nnsight.models import NNsightModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
import neel_utils as neel_utils
from cosine_sims import (
    load_board_state_probes,
    load_flipped_probes,
    load_played_probes,
    get_mine_theirs_normed,
    get_blank_normed,
    get_flipped_normed,
    get_played_normed,
)

import json
import pickle
import gzip
from typing import Tuple
from jaxtyping import Int, Float, jaxtyped
from typeguard import typechecked
from functools import partial
from dataclasses import dataclass
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm


jaxtyped = partial(jaxtyped, typechecker=typechecked)
device = "cuda" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)
CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent


MIDDLE_SQUARES = [27, 28, 35, 36]
ALL_SQUARES = [i for i in range(64) if i not in MIDDLE_SQUARES]


@dataclass
class DecisionTreeResults:
    """Results for a single square's decision tree."""

    layer: int
    neuron: int
    tree: DecisionTreeRegressor
    train_R2: float
    train_MSE: float
    test_R2: float
    test_MSE: float


def load_model(
    model_name: str = "Baidicoot/Othello-GPT-Transformer-Lens",
    device=device,
) -> NNsightModel:
    return utils.get_model(model_name, device)


@jaxtyped
def load_data(
    n_train: int = 10000,
    n_test: int = 10000,
) -> Tuple[Int[Tensor, "n_train_games n_moves"], Int[Tensor, "n_test_games n_moves"]]:
    """
    Returns:
        model: OthelloGPT model
        train_data: Training game data
        test_data: Test game data
    """
    train_data = construct_othello_dataset(
        custom_functions=[],
        n_inputs=n_train,
        split="train",
        device="cpu",
    )
    train_encoded_inputs = t.tensor(
        train_data["encoded_inputs"], dtype=t.long, device="cpu"
    )

    test_data = construct_othello_dataset(
        custom_functions=[],
        n_inputs=n_test,
        split="test",
        device="cpu",
    )
    test_encoded_inputs = t.tensor(
        test_data["encoded_inputs"], dtype=t.long, device="cpu"
    )

    return train_encoded_inputs, test_encoded_inputs


@jaxtyped
def extract_activations_for_layer(
    model: NNsightModel,
    data: Int[Tensor, "n_games n_total_moves"],
    layer: int,
    batch_size: int = 256,
    device="cuda",
) -> Tuple[
    Float[Tensor, "n_games n_moves d_model"], Float[np.ndarray, "n_games n_moves d_mlp"]
]:
    """
    Extract activations from each layer of the model.

    Args:
        model: OthelloGPT model
        data: train or test encoded inputs

    Returns:
        Dict mapping layer_idx -> activations array (n_games, n_tokens, hidden_dim)
    """
    keys = [f"blocks.{layer}.hook_resid_pre", f"blocks.{layer}.mlp.hook_post"]
    #resid = t.empty((data.shape[0], data.shape[1], model.cfg.d_model))
    #mlp = np.ndarray((data.shape[0], data.shape[1], model.cfg.d_mlp))
    resid = t.empty((data.shape[0], 26, model.cfg.d_model))
    mlp = np.ndarray((data.shape[0], 26, model.cfg.d_mlp))

    for i in range(0, len(data), batch_size):
        batch_inputs = data[i : i + batch_size].to(device)
        _, cache = model.run_with_cache(
            batch_inputs,
            names_filter=lambda name: name in keys,
        )

        # Only focus on moves 5-30
        #resid[i : i + batch_size] = cache[keys[0]].detach().cpu()
        #mlp[i : i + batch_size] = cache[keys[1]].detach().cpu()
        resid[i : i + batch_size] = cache[keys[0]][:, 5:31].detach().cpu()
        mlp[i : i + batch_size] = cache[keys[1]][:, 5:31].detach().cpu()

    return resid, mlp


@jaxtyped
def prepare_dt_train_data_for_layer(
    model: NNsightModel,
    train_resid_acts: Float[Tensor, "n_train_games n_moves d_model"],
    train_mlp_post_acts: Float[np.ndarray, "n_train_games n_moves d_mlp"],
    test_resid_acts: Float[Tensor, "n_test_games n_moves d_model"],
    test_mlp_post_acts: Float[np.ndarray, "n_test_games n_moves d_mlp"],
    layer: int,
    batch_size: int = 256,
) -> Tuple[
    Float[np.ndarray, "n_train_samples n_feats"],
    Float[np.ndarray, "n_train_samples d_mlp"],
    Float[np.ndarray, "n_test_samples n_feats"],
    Float[np.ndarray, "n_test_samples d_mlp"],
]:
    board_state_probes = load_board_state_probes(
        model,
        path=f"{PARENT_DIR}/linear_probes/resid_{{layer}}_board_state.pth"
    )
    flipped_probes = load_flipped_probes(
        model,
        path=f"{PARENT_DIR}/flipped_probes/resid_{{layer}}_flipped.pth",
    )
    played_probes = load_played_probes(
        model,
        path=f"{PARENT_DIR}/played_probes/resid_{{layer}}_played.pth",
    )

    mine_theirs = get_mine_theirs_normed(board_state_probes, normalize=False)[layer - 1]
    blank = get_blank_normed(board_state_probes, normalize=False)[layer - 1]
    flipped = get_flipped_normed(flipped_probes, normalize=False)[layer - 1]
    played = get_played_normed(played_probes, normalize=False)[layer - 1]

    blank_flat = einops.rearrange(blank, "d_model row col -> d_model (row col)")
    blank_selected = blank_flat[:, ALL_SQUARES] 
    
    played_flat = einops.rearrange(played, "d_model row col -> d_model (row col)")
    played_selected = played_flat[:, ALL_SQUARES]
    
    mine_theirs_flat = einops.rearrange(mine_theirs, "d_model row col -> d_model (row col)")

    flipped_flat = einops.rearrange(flipped, "d_model row col -> d_model (row col)")

    all_probe = t.cat([
        mine_theirs_flat,   
        blank_selected,    
        flipped_flat,  
        played_selected   
    ], dim=1)

    train_resid_acts = einops.rearrange(
        train_resid_acts, "n_games n_moves d_model -> (n_games n_moves) d_model"
    )
    X_train = np.ndarray((train_resid_acts.shape[0], all_probe.shape[1]))
    for i in range(0, len(train_resid_acts), batch_size):
        batch_inputs = train_resid_acts[i : i + batch_size].to(device)
        X_train[i : i + batch_size] = (
            einops.einsum(
                batch_inputs, all_probe, "batch d_model, d_model feats -> batch feats"
            )
            .detach()
            .cpu()
            .numpy()
        )

    test_resid_acts = einops.rearrange(
        test_resid_acts, "n_games n_moves d_model -> (n_games n_moves) d_model"
    )
    X_test = np.ndarray((test_resid_acts.shape[0], all_probe.shape[1]))
    for i in range(0, len(test_resid_acts), batch_size):
        batch_inputs = test_resid_acts[i : i + batch_size].to(device)
        X_test[i : i + batch_size] = (
            einops.einsum(
                batch_inputs, all_probe, "batch d_model, d_model feats -> batch feats"
            )
            .detach()
            .cpu()
            .numpy()
        )

    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    epsilon = 1e-8
    X_train_scaled = (X_train - X_train_mean) / (X_train_std + epsilon)

    X_test_scaled = (X_test - X_train_mean) / (X_train_std + epsilon)

    y_train = einops.rearrange(
        train_mlp_post_acts, "n_games n_moves d_mlp -> (n_games n_moves) d_mlp"
    )
    y_test = einops.rearrange(
        test_mlp_post_acts, "n_games n_moves d_mlp -> (n_games n_moves) d_mlp"
    )

    return X_train_scaled, y_train, X_test_scaled, y_test


def train_dt_for_layer(
    model: NNsightModel,
    X_train: Float[np.ndarray, "n_train_samples n_feats"],
    y_train: Float[np.ndarray, "n_train_samples d_mlp"],
    X_test: Float[np.ndarray, "n_test_samples n_feats"],
    y_test: Float[np.ndarray, "n_test_samples d_mlp"],
    layer: int,
    depth: int = 3,
    n_jobs: int = -1,
) -> list[DecisionTreeResults]:
    # Convert to plain numpy arrays (joblib no like jaxtyping)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    
    def worker(neuron):
        tree = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42,
            min_samples_leaf=50,
            min_samples_split=100,
            criterion="squared_error",
        )
        
        y_train_neuron = y_train[:, neuron]
        y_test_neuron = y_test[:, neuron]
        
        tree.fit(X_train, y_train_neuron)
        
        train_r2 = tree.score(X_train, y_train_neuron)
        test_r2 = tree.score(X_test, y_test_neuron)
        train_mse = mean_squared_error(y_train_neuron, tree.predict(X_train))
        test_mse = mean_squared_error(y_test_neuron, tree.predict(X_test))
        
        return DecisionTreeResults(
            layer=layer,
            neuron=neuron,
            tree=tree,
            train_R2=train_r2,
            train_MSE=train_mse,
            test_R2=test_r2,
            test_MSE=test_mse,
        )
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(neuron)
        for neuron in tqdm(range(model.cfg.d_mlp), desc=f"Training layer {layer} trees")
    )
    return results


def save_layer_results(results, layer, save_dir):
    """Save all trees in one compressed pickle"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"layer_{layer}_trees.pkl.gz"
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(results, f)
    return save_path


if __name__ == "__main__":
    layer = 5
    depth = 4

    model = load_model()

    train_data, test_data = load_data(n_train=6000, n_test=500)

    train_resid_acts, train_mlp_acts = extract_activations_for_layer(
        model, train_data, layer=layer
    )
    test_resid_acts, test_mlp_acts = extract_activations_for_layer(
        model, test_data, layer=layer
    )

    X_train, y_train, X_test, y_test = prepare_dt_train_data_for_layer(
        model,
        train_resid_acts,
        train_mlp_acts,
        test_resid_acts,
        test_mlp_acts,
        layer
    )

    results = train_dt_for_layer(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        layer=layer,
        depth=depth
    )

    save_path = save_layer_results(results, layer, save_dir="results")
    print(f"Saved results to {save_path}")