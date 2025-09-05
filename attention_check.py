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
import os

# from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import graphviz

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
    calculate_ablation_scores_square_probability,
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

device = "cuda:1" if t.cuda.is_available() else "cpu"
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
    # othello_utils.games_batch_to_valid_moves_BLRRC, # (legal move)
]
test_data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=test_size,
    split="test", 
    device=device,
)

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)
board_seqs_square = t.tensor(test_data["decoded_inputs"]).long().to(device)

board_states, legal_moves, legal_moves_annotation = get_board_states_and_legal_moves(board_seqs_square)
legal_moves = legal_moves.to(device=device, dtype=t.float32)

# %% writing neuron
n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head

W_Q = model.W_Q.detach().clone()  # [layer, head, d_model, d_head]
W_K = model.W_K.detach().clone()  # [layer, head, d_model, d_head]
W_O = model.W_O.detach().clone()  # [layer, head, d_head, d_model]
W_V = model.W_V.detach().clone()  # [layer, head, d_model, d_head]

W_E = model.W_E[1:].detach().clone()  # [vocab_size, d_model]
W_U = model.W_U[:, 1:].detach().clone()  # [d_model, 60]

# W_OV: a linear map describing what information gets moved from source to destination, in the residual stream.
# In other words, if x is a vector in the residual stream, then x^T*W_OV is the vector written to the residual stream at the destination position, if the destination token only pays attention to the source token at the position of the vector x.
W_OV = einops.einsum(
    W_V,
    W_O,
    "layer head d_model1 d_head, layer head d_head d_model2 -> layer head d_model1 d_model2",
)  # [layer, head, d_model, d_model]

# W_OV_full: a linear map describing what information gets moved from source to destination, in a start-to-end sense.
W_OV_full = einops.einsum(
    W_E,
    W_OV,
    W_U,
    "vocab1 d_model1, layer head d_model1 d_model2, d_model2 vocab2 -> layer head vocab1 vocab2",
)

# W_QK: a bilinear form describing where information is moved to and from in the residual stream (i.e. which residual stream vectors attend to which others).
W_QK = einops.einsum(
    W_Q,
    W_K,
    "layer head d_model1 d_head, layer head d_model2 d_head -> layer head d_model1 d_model2",
)  # [layer, head, d_model, d_model]

W_QK_full = einops.einsum(
    W_E,
    W_QK,
    W_E,
    "vocab1 d_model1, layer head d_model1 d_model2, vocab2 d_model2 -> layer head vocab1 vocab2",
)

# W_Q_composition

# w_out = model.W_out.detach().clone() # [layer, neuron, d_model]
# # w_out_nomalized = w_out / w_out.norm(dim=-1, keepdim=True)
# W_U = model.W_U[:, 1:].detach().clone()  # [d_model, 60]
# # W_U_normalized = W_U / W_U.norm(dim=0, keepdim=True)

# write_attribution = einops.einsum(
#     w_out,
#     W_U,
#     "layer neuron d_model, d_model id -> layer neuron id",
# )

# write_attribution_square = t.zeros((n_layers, n_neurons, 8, 8), device=device, dtype=t.float32)
# write_attribution_square.flatten(start_dim=-2, end_dim=-1)[..., ALL_SQUARES] = write_attribution

# %% heatmap for QK circuits
for layer in range(n_layers):
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f"W_QK_full Heatmaps for Layer {layer}", fontsize=16)
    for head in range(n_heads):
        ax = axs[head // 4, head % 4]
        im = ax.imshow(W_QK_full[layer, head].cpu().numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"Head {head}")
        ax.set_xlabel("Key Position (to)")
        ax.set_ylabel("Query Position (from)")
        fig.colorbar(im, ax=ax)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(f"figures/W_QK_full_heatmaps_layer_{layer}.png")
    # plt.close()
# %%
