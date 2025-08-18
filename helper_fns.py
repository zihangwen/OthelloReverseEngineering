import torch as t
import numpy as np
import einops
import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import to_numpy
from torch import Tensor
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from typing import Optional, List
# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree
import arena_utils as arena_utils
from simulate_activations_with_dts import (
    compute_kl_divergence,
    compute_top_n_accuracy,
)

MIDDLE_SQUARES = [27, 28, 35, 36]
ALL_SQUARES = [i for i in range(64) if i not in MIDDLE_SQUARES]
tracer_kwargs = {"validate": True, "scan": True}

# %%
def get_neuron_decision_tree(data: dict, layer: int, neuron_idx: int, function_name: str):
    """Extract the decision tree for a specific neuron."""
    if layer not in data:
        raise ValueError(f"Layer {layer} not found in data. Available layers: {list(data.keys())}")
    
    if function_name not in data[layer]:
        available_funcs = list(data[layer].keys())
        raise ValueError(f"Function {function_name} not found. Available: {available_funcs}")
    
    multi_output_model = data[layer][function_name]['decision_tree']['model']
    
    if neuron_idx >= len(multi_output_model.estimators_):
        raise ValueError(f"Neuron {neuron_idx} not found. Max neuron index: {len(multi_output_model.estimators_) - 1}")
    
    neuron_tree = multi_output_model.estimators_[neuron_idx]
    r2_scores = data[layer][function_name]['decision_tree']['r2']
    neuron_r2 = r2_scores[neuron_idx]
    
    return neuron_tree, neuron_r2

# %%
def get_neuron_binary_decision_tree(data: dict, layer: int, neuron_idx: int, function_name: str):
    """Extract the decision tree for a specific neuron."""
    if layer not in data:
        raise ValueError(f"Layer {layer} not found in data. Available layers: {list(data.keys())}")
    
    if function_name not in data[layer]:
        available_funcs = list(data[layer].keys())
        raise ValueError(f"Function {function_name} not found. Available: {available_funcs}")
    
    multi_output_model = data[layer][function_name]['binary_decision_tree']['model']
    
    if neuron_idx >= len(multi_output_model.estimators_):
        raise ValueError(f"Neuron {neuron_idx} not found. Max neuron index: {len(multi_output_model.estimators_) - 1}")
    
    neuron_tree = multi_output_model.estimators_[neuron_idx]
    f1_scores = data[layer][function_name]['binary_decision_tree']['f1']
    neuron_f1 = f1_scores[neuron_idx]
    
    return neuron_tree, neuron_f1

# %%
# def visualize_decision_tree(tree_model, neuron_idx: int, layer: int, r2_score: float,
#                           feature_names: List[str], max_depth: Optional[int] = None,
#                           save_path: Optional[str] = None):
#     """Visualize a decision tree for a specific neuron."""
#     plt.figure(figsize=(20, 12))
    
#     plot_tree(
#         tree_model,
#         feature_names=feature_names,
#         filled=True,
#         rounded=True,
#         fontsize=8,
#         max_depth=max_depth
#     )
    
#     plt.title(f"Decision Tree for Layer {layer}, Neuron {neuron_idx}\nR² Score: {r2_score:.4f}", 
#               fontsize=16, pad=20)
    
#     if save_path:
#         plt.savefig(f"{save_path}/dt_layer_{layer}_neuron_{neuron_idx}.png", dpi=300, bbox_inches='tight')
#         print(f"Saved visualization to {save_path}")
    
#     plt.show()

# %%
def neuron_intervention(
    model,
    layers_neurons: dict[list],
    game_batch_BL: t.Tensor,
    ablation_method: str = "zero",
):
    allowed_methods = ["mean", "max", "zero"]
    # allowed_methods = ["zero"]
    assert ablation_method in allowed_methods, (
        f"Invalid ablation method. Must be one of {allowed_methods}"
    )

    mean_activations = {}
    max_activations = {}

    # Get clean logits and mean submodule activations
    with t.no_grad(), model.trace(game_batch_BL, **tracer_kwargs):
        for layer in range(model.cfg.n_layers):
            original_input_BLD = model.blocks[layer].mlp.hook_post.output
            if ablation_method == "mean":
                mean_activations[layer] = original_input_BLD.mean(dim=(0, 1)).save()
            elif ablation_method == "max":
                # max_activations_temp = original_input_BLD.max(dim=0).values
                max_activations[layer] = original_input_BLD.max(dim=(0, 1)).values.save()
            elif ablation_method == "zero":
                # No need to do anything for zero ablation, just save the original input
                pass
        logits_clean_BLV = model.unembed.output.save()
    
    with t.no_grad(), model.trace(game_batch_BL, **tracer_kwargs):
        for layer, neuron_indices in layers_neurons.items():
            original_input_BLD = model.blocks[layer].mlp.hook_post.output
            if ablation_method == "mean":
                original_input_BLD[:, :, neuron_indices] = mean_activations[layer][neuron_indices]
            elif ablation_method == "max":
                original_input_BLD[:, :, neuron_indices] = max_activations[layer][neuron_indices]
            elif ablation_method == "zero":
                original_input_BLD[:, :, neuron_indices] = 0.0
        
        logits_patch_BLV = model.unembed.output.save()
    
    return logits_clean_BLV, logits_patch_BLV

# %%
def calculate_ablation_scores_game_move(model, layers_neurons, focus_games_id, focus_legal_moves, move, ablation_method = "zero"):
    logits_clean_BLV, logits_patch_BLV = neuron_intervention(
        model,
        layers_neurons=layers_neurons,
        game_batch_BL=focus_games_id,
        ablation_method=ablation_method,
    )

    logits_clean_BLV_move = logits_clean_BLV[:, move].unsqueeze(1) # [1, 1, ids]
    logits_patch_BLV_move = logits_patch_BLV[:, move].unsqueeze(1) # [1, 1, ids]
    focus_legal_moves_move = focus_legal_moves[:, move].unsqueeze(1).unsqueeze(-1)  # [1, 1, row, col, 1]

    kl_div_BL = compute_kl_divergence(logits_clean_BLV_move, logits_patch_BLV_move)

    _, _, clean_accuracy = compute_top_n_accuracy(
        logits_clean_BLV_move, focus_legal_moves_move
    )

    _, _, patch_accuracy = compute_top_n_accuracy(
        logits_patch_BLV_move, focus_legal_moves_move
    )

    return kl_div_BL.mean().item(), clean_accuracy, patch_accuracy

# %%
def calculate_ablation_scores_square(model, layers_neurons, board_seqs_id, valid_move_square_mask, valid_move_number, token_id, ablation_method = "zero"):
    logits_clean_BLV, logits_patch_BLV = neuron_intervention(
        model,
        layers_neurons=layers_neurons,
        game_batch_BL=board_seqs_id,
        ablation_method=ablation_method,
    )

    kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)

    logits_clean_rank_token = (logits_clean_BLV > logits_clean_BLV[...,token_id].unsqueeze(-1)).sum(-1)
    clean_total = valid_move_square_mask.sum()
    clean_correct = ((logits_clean_rank_token < valid_move_number) * valid_move_square_mask).sum()
    clean_accuracy = clean_correct / clean_total

    logits_patch_rank_token = (logits_patch_BLV > logits_patch_BLV[...,token_id].unsqueeze(-1)).sum(-1)
    patch_total = valid_move_square_mask.sum()
    patch_correct = ((logits_patch_rank_token < valid_move_number) * valid_move_square_mask).sum()
    patch_accuracy = patch_correct / patch_total

    return kl_div_BL.mean().item(), clean_accuracy.item(), patch_accuracy.item()

# %%
def get_board_states_and_legal_moves(
    games_square: Int[Tensor, "n_games n_moves"],
) -> tuple[
    Int[Tensor, "n_games n_moves rows cols"],
    Int[Tensor, "n_games n_moves rows cols"],
    list,
]:
    """
    Returns the following:
        states:                 (n_games, n_moves, 8, 8): tensor of board states after each move
        legal_moves:            (n_games, n_moves, 8, 8): tensor of 1s for legal moves, 0s for illegal moves
        legal_moves_annotation: (n_games, n_moves, 8, 8): list containing strings of "o" for legal moves (for plotting)
    """
    # Create tensors to store the board state & legal moves
    n_games, n_moves = games_square.shape
    states = t.zeros((n_games, n_moves, 8, 8), dtype=t.int32)
    legal_moves = t.zeros((n_games, n_moves, 8, 8), dtype=t.int32)

    # Loop over each game, populating state & legal moves tensors after each move
    for n in range(n_games):
        board = arena_utils.OthelloBoardState()
        for i in range(n_moves):
            board.umpire(games_square[n, i].item())
            states[n, i] = t.from_numpy(board.state)
            legal_moves[n, i].flatten()[board.get_valid_moves()] = 1

    # Convert legal moves to annotation
    legal_moves_annotation = np.where(to_numpy(legal_moves), "o", "").tolist()

    return states, legal_moves, legal_moves_annotation

# %%
def plot_probe_outputs(
    cache: ActivationCache,
    probe_dict : dict[int, Tensor],
    layer: int,
    game_index: int,
    move: int,
    title: str = "Probe outputs",
):
    linear_probe = probe_dict[layer]
    residual_stream = cache["resid_post", layer][game_index, move]
    probe_out = einops.einsum(
        residual_stream,
        linear_probe,
        "d_model, d_model row col options -> options row col",
    )

    arena_utils.plot_board_values(
        probe_out.softmax(dim=0),
        title=title,
        width=900,
        height=400,
        board_titles=["P(Mine)", "P(Empty)", "P(Their's)"],
        # text=BOARD_LABELS_2D,
    )

# %%
def get_w_in(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    """
    Returns the input weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_in = model.W_in[layer, :, neuron].detach().clone()
    if normalize:
        w_in /= w_in.norm(dim=0, keepdim=True)
    return w_in

def get_w_out(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    """
    Returns the output weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_out = model.W_out[layer, neuron, :].detach().clone()
    if normalize:
        w_out /= w_out.norm(dim=0, keepdim=True)
    return w_out

def get_w_out_all(
    model: HookedTransformer,
    normalize: bool = False,
) -> Float[Tensor, "layer neuron d_model"]:
    """
    Returns the output weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_out = model.W_out.detach().clone()
    if normalize:
        w_out /= w_out.norm(dim=-1, keepdim=True)
    return w_out

def get_w_U(
    model: HookedTransformer,
    normalize: bool = False,
) -> Float[Tensor, "d_model token_id"]:
    """
    Returns the W_U weights for the model.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_U = model.W_U[:, 1:].detach().clone()  # Exclude the "pass" move
    if normalize:
        w_U /= w_U.norm(dim=0, keepdim=True)
    return w_U

def calculate_neuron_input_weights(
    model: HookedTransformer, probe: Float[Tensor, "d_model row col"], layer: int, neuron: int
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the input weights for the given neuron, at each square on the board, projected
    along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    w_in = get_w_in(model, layer, neuron, normalize=True)

    return einops.einsum(w_in, probe, "d_model, d_model row col -> row col")

def calculate_neuron_output_weights(
    model: HookedTransformer, probe: Float[Tensor, "d_model row col"], layer: int, neuron: int
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the output weights for the given neuron, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    w_out = get_w_out(model, layer, neuron, normalize=True)

    return einops.einsum(w_out, probe, "d_model, d_model row col -> row col")

# %%
def create_pbs_feature_names(n_features: int) -> List[str]:
    """Create feature names based on the actual feature structure:
    (192) + (64 + 64 + 5) + (64) = 389 dimensional vector
    - Board state: 192 one-hot (8x8x3 mine/empty/theirs)
    - Last move: 64 one-hot move + 64 pre-occupied + 5 coordinates  
    - Flipped moves: 64 binary encoding of flipped squares
    
    Square notation: A0-H7 where A0 is top-left, H7 is bottom-right
    """
    feature_names = []
    idx = 0
    
    # First 192: Board state (8x8x3 = mine/empty/theirs)
    for square_idx in range(min(64, (n_features - idx) // 3)):
        row = square_idx // 8  
        col = square_idx % 8
        square = chr(ord('A') + row) + str(col)
        
        # Add the 3 states for this square
        if idx < n_features:
            feature_names.append(f"prev_board_{square}_theirs")
            idx += 1
        if idx < n_features:
            feature_names.append(f"prev_board_{square}_empty") 
            idx += 1
        if idx < n_features:
            feature_names.append(f"prev_board_{square}_mine")
            idx += 1

    # Next 64: Last move one-hot encoding (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_just_played")
        idx += 1
    
    # Next 64: Pre-occupied squares (A0-H7)  
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_pre_occupied")
        idx += 1
    
    # Next 5: Move coordinates and player info
    coord_names = ["move_row", "move_col", "move_number", "white_played", "black_played"]
    for i in range(min(5, n_features - idx)):
        feature_names.append(coord_names[i])
        idx += 1
    
    # Last 64: Flipped squares (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_flipped")
        idx += 1
    
    # Add any remaining features as generic (shouldn't happen with 389 total)
    while idx < n_features:
        feature_names.append(f"Feature_{idx}")
        idx += 1
    
    return feature_names


def create_bs_feature_names(n_features: int) -> List[str]:
    """Create feature names based on the actual feature structure:
    (192) + (64 + 64 + 5) + (64) = 389 dimensional vector
    - Board state: 192 one-hot (8x8x3 mine/empty/theirs)
    - Last move: 64 one-hot move + 64 pre-occupied + 5 coordinates  
    - Flipped moves: 64 binary encoding of flipped squares
    
    Square notation: A0-H7 where A0 is top-left, H7 is bottom-right
    """
    feature_names = []
    idx = 0
    
    # First 192: Board state (8x8x3 = mine/empty/theirs)
    for square_idx in range(min(64, (n_features - idx) // 3)):
        row = square_idx // 8  
        col = square_idx % 8
        square = chr(ord('A') + row) + str(col)
        
        # Add the 3 states for this square
        if idx < n_features:
            feature_names.append(f"{square}_mine")
            idx += 1
        if idx < n_features:
            feature_names.append(f"{square}_empty") 
            idx += 1
        if idx < n_features:
            feature_names.append(f"{square}_theirs")
            idx += 1

    # Next 64: Last move one-hot encoding (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_just_played")
        idx += 1
    
    # Next 64: Pre-occupied squares (A0-H7)  
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_pre_occupied")
        idx += 1
    
    # Next 5: Move coordinates and player info
    coord_names = ["move_row", "move_col", "move_number", "white_played", "black_played"]
    for i in range(min(5, n_features - idx)):
        feature_names.append(coord_names[i])
        idx += 1
    
    # Last 64: Flipped squares (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + row) + str(col)
        feature_names.append(f"{square}_flipped")
        idx += 1
    
    # Add any remaining features as generic (shouldn't happen with 389 total)
    while idx < n_features:
        feature_names.append(f"Feature_{idx}")
        idx += 1
    
    return feature_names


def create_feature_names(n_features: int, function_name) -> List[str]:
    if function_name == "games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC":
        return create_pbs_feature_names(n_features)
    elif function_name == "games_batch_to_input_tokens_flipped_bs_classifier_input_BLC":
        return create_bs_feature_names(n_features)
    else:
        raise ValueError(f"Unknown function name: {function_name}. Cannot create feature names.")
    
# %% PLOTTING LOG PROBS
# First 10 moves of game 0
# sample_input = t.tensor(test_data["encoded_inputs"][0][:10]).to(device)
# with model.trace(sample_input):
#     logits = model.unembed.output.save()
# logprobs = logits.log_softmax(dim=-1)

# logprobs_board = t.full(size=(8, 8), fill_value=-13.0, device=device)
# logprobs_board.flatten()[ALL_SQUARES] = logprobs[
#     0, 0, 1:
# ]  # the [1:] is to filter out logits for the "pass" move

# arena_utils.plot_board_values(logprobs_board, title="Example Log Probs", width=500)

# %% PLOTTING LOG PROBS with ANNOTATED TOKEN IDS and BOARD LABELS
# TOKEN_IDS_2D = np.array(
#     [str(i) if i in ALL_SQUARES else "" for i in range(64)]
# ).reshape(8, 8)
# BOARD_LABELS_2D = np.array(
#     ["ABCDEFGH"[i // 8] + f"{i % 8}" for i in range(64)]
# ).reshape(8, 8)

# print(TOKEN_IDS_2D)
# print(BOARD_LABELS_2D)

# arena_utils.plot_board_values(
#     t.stack([logprobs_board, logprobs_board]),  # shape (2, 8, 8)
#     title="Example Log Probs (with annotated token IDs)",
#     width=800,
#     text=np.stack([TOKEN_IDS_2D, BOARD_LABELS_2D]),  # shape (2, 8, 8)
#     board_titles=["Labelled by token ID", "Labelled by board label"],
# )

# %% PLOTTING LOG PROBS (10 MOVES)
# logprobs_multi_board = t.full(size=(10, 8, 8), fill_value=-13.0, device=device)
# logprobs_multi_board.flatten(1, -1)[:, ALL_SQUARES] = logprobs[
#     0, :, 1:
# ]  # we now do all 10 moves at once

# arena_utils.plot_board_values(
#     logprobs_multi_board,
#     title="Example Log Probs",
#     width=1000,
#     boards_per_row=5,
#     board_titles=[f"Logprobs after move {i}" for i in range(1, 11)],
# )

# %% PLOTTING BOARD STATES AND LEGAL MOVES (10 MOVES)
# board_states = t.zeros((10, 8, 8), dtype=t.int32)
# legal_moves = t.zeros((10, 8, 8), dtype=t.int32)

# board = arena_utils.OthelloBoardState()
# for i, token_id in enumerate(sample_input.squeeze()):
#     # board.umpire takes a square index (i.e. from 0 to 63) and makes a move on the board
#     board.umpire(arena_utils.id_to_square(token_id))

#     # board.state gives us the 8x8 numpy array of 0 (blank), -1 (black), 1 (white)
#     board_states[i] = t.from_numpy(board.state)

#     # board.get_valid_moves() gives us a list of the indices of squares that are legal to play next
#     legal_moves[i].flatten()[board.get_valid_moves()] = 1

# # Turn `legal_moves` into strings, with "o" where the move is legal and empty string where illegal
# legal_moves_annotation = np.where(to_numpy(legal_moves), "o", "").tolist()

# arena_utils.plot_board_values(
#     board_states,
#     title="Board states",
#     width=1000,
#     boards_per_row=5,
#     board_titles=[f"State after move {i}" for i in range(1, 11)],
#     text=legal_moves_annotation,
# )
