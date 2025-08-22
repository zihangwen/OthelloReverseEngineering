# %%
import pickle
from collections import defaultdict
import torch as t
import numpy as np
import einops
from rich import print as rprint
from rich.table import Column, Table
from rich.console import Console
from rich.terminal_theme import MONOKAI

# from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from transformer_lens.utils import to_numpy, get_act_name
# from transformer_lens import ActivationCache, HookedTransformer
# from torch import Tensor
# from IPython.display import HTML, display
# from jaxtyping import Bool, Float, Int

import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
import arena_utils as arena_utils
from helper_fns import (
    # MIDDLE_SQUARES,
    neuron_intervention,
    ALL_SQUARES,
    get_board_states_and_legal_moves,
    calculate_ablation_scores_game_move,
    calculate_ablation_scores_square,
    # plot_probe_outputs,
    get_w_in,
    # get_w_out,
    calculate_neuron_input_weights,
    calculate_neuron_output_weights,
    create_feature_names,
    get_neuron_decision_tree,
    get_neuron_binary_decision_tree,
    # visualize_decision_tree,
)
# from simulate_activations_with_dts import (
#     compute_kl_divergence,
#     compute_top_n_accuracy,
# )

device = "cuda" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")
# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = utils.get_model(model_name, device)
# %% Load the test dataset and process
test_size = 500
custom_functions = [
    # othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
    othello_utils.games_batch_to_board_state_classifier_input_BLC, # (board state)
]
test_data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=test_size,
    split="test", 
    device=device,
)

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)
board_seqs_square = t.tensor(test_data["decoded_inputs"]).long().to(device)

# board_states, legal_moves, legal_moves_annotation = get_board_states_and_legal_moves(board_seqs_square)
# legal_moves = legal_moves.to(device=device, dtype=t.float32)

board_states_custom = test_data["games_batch_to_board_state_classifier_input_BLC"]
board_states_custom = einops.rearrange(board_states_custom, "B L (R1 R2 C) -> B L R1 R2 C", R1=8, R2=8)

# %%
n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp

probe_dict = {i : t.load(
    f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
)['linear_probe'].squeeze() for i in range(model.cfg.n_layers)}

probe_t = t.stack([probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]
blank_probe = probe_t[..., 1] - (probe_t[..., 0] + probe_t[..., 2]) * 0.5  # [layer, d_model, row, col]
my_probe = probe_t[..., 0] - probe_t[..., 2]  # [layer, d_model, row, col]

blank_probe_normalized = blank_probe / blank_probe.norm(dim=1, keepdim=True)
my_probe_normalized = my_probe / my_probe.norm(dim=1, keepdim=True)
blank_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# %% writing neuron
w_out = model.W_out.detach().clone() # [layer, neuron, d_model]
# w_out_nomalized = w_out / w_out.norm(dim=-1, keepdim=True)
W_U = model.W_U[:, 1:].detach().clone()  # [d_model, 60]
# W_U_normalized = W_U / W_U.norm(dim=0, keepdim=True)

write_attribution = einops.einsum(
    w_out,
    W_U,
    "layer neuron d_model, d_model id -> layer neuron id",
)

write_attribution_square = t.zeros((n_layers, n_neurons, 8, 8), device=device, dtype=t.float32)
write_attribution_square.flatten(start_dim=-2, end_dim=-1)[..., ALL_SQUARES] = write_attribution

# %% ----- ----- ----- ----- ----- ----- specific square ----- ----- ----- ----- ----- ----- %% #
# Calculate neuron attribution for a specific square
square_idx = 25
token_id = arena_utils.SQUARE_TO_ID[square_idx]
print(f"Square {square_idx} ({arena_utils.to_board_label(square_idx)}), Token ID: {token_id}")

valid_move_square_mask = legal_moves.flatten(start_dim=-2, end_dim=-1)[..., square_idx] # [game, seq]
valid_move_number = legal_moves.sum(dim=(-2,-1))  # [game, seq]
