# %%
import pickle
import json
from pathlib import Path
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

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(BASE_PATH)
BASE_PATH = Path(BASE_PATH)
os.chdir(BASE_PATH)

# from transformer_lens.utils import to_numpy, get_act_name
# from transformer_lens import ActivationCache, HookedTransformer
# from torch import Tensor
# from IPython.display import HTML, display
# from jaxtyping import Bool, Float, Int

import utils.circuits_utils as circuits_utils
from utils.arena_utils import (
    label_to_square,
)
import utils.othello_utils as othello_utils
from utils.probe_utils import (
    # load_probes_and_normalize,
    load_fold_probes_and_normalize,
)
import utils.arena_utils as arena_utils
from utils.helper_fns import (
    get_board_states_and_legal_moves,
)
#     # MIDDLE_SQUARES,
#     neuron_intervention,
#     ALL_SQUARES,
#     
#     calculate_ablation_scores_game_move,
#     calculate_ablation_scores_square,
#     calculate_ablation_scores_square_probability,
#     # plot_probe_outputs,
#     get_w_in,
#     # get_w_out,
#     calculate_neuron_input_weights,
#     calculate_neuron_output_weights,
#     create_feature_names,
#     get_neuron_decision_tree,
#     get_neuron_binary_decision_tree,
#     # visualize_decision_tree,
# )
# from simulate_activations_with_dts import (
#     compute_kl_divergence,
#     compute_top_n_accuracy,
# )

device = "cuda:1" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = circuits_utils.get_model(model_name, device)
n_layers = model.cfg.n_layers

# %% Load the test dataset and process
# test_size = 500
# custom_functions = [
#     # othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
#     # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
#     # othello_utils.games_batch_to_valid_moves_BLRRC, # (legal move)
# ]
# test_data = circuits_utils.construct_othello_dataset(
#     custom_functions=custom_functions,
#     n_inputs=test_size,
#     split="test", 
#     device=device,
# )

# board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)
# board_seqs_id = board_seqs_id[:, 8:30]

# board_seqs_square = t.tensor(test_data["decoded_inputs"]).long().to(device)

# board_states, legal_moves, legal_moves_annotation = get_board_states_and_legal_moves(board_seqs_square)
# legal_moves = legal_moves.to(device=device, dtype=t.float32)

# %%
with open("attention/attention_head_types.json", "r") as f:
    head_type_all = json.load(f)

# %% Loading Probes and model
probes = load_fold_probes_and_normalize(n_layers, device)

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

# %% calculating attention weight circuits
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

# %% heatmap for OV circuits
# for layer in range(n_layers):
#     fig, axs = plt.subplots(2, 4, figsize=(12, 6))
#     fig.suptitle(f"W_OV_full Heatmaps for Layer {layer}", fontsize=16)
#     for head in range(n_heads):
#         ax = axs[head // 4, head % 4]
#         im = ax.imshow(W_OV_full[layer, head].cpu().numpy(), cmap="viridis", aspect="auto")
#         ax.set_title(f"Head {head}, head type: {head_type_all[str(layer)][str(head)]}")
#         ax.set_xlabel("Square token (to)")
#         ax.set_ylabel("Square token (from)")
#         fig.colorbar(im, ax=ax)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     # plt.savefig(f"figures/W_OV_full_heatmaps_layer_{layer}.png")
#     # plt.close()

# %% heatmap for QK circuits
# for layer in range(n_layers):
#     fig, axs = plt.subplots(2, 4, figsize=(12, 6))
#     fig.suptitle(f"W_QK_full Heatmaps for Layer {layer}", fontsize=16)
#     for head in range(n_heads):
#         ax = axs[head // 4, head % 4]
#         im = ax.imshow(W_QK_full[layer, head].cpu().numpy(), cmap="viridis", aspect="auto")
#         ax.set_title(f"Head {head}, head type: {head_type_all[str(layer)][str(head)]}")
#         ax.set_xlabel("Square token (from)")
#         ax.set_ylabel("Square token (to)")
#         fig.colorbar(im, ax=ax)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     # plt.savefig(f"figures/W_QK_full_heatmaps_layer_{layer}.png")
#     # plt.close()

# %% heatmap for probe directions QK
# label = "C1"
# square = label_to_square(label)
# row, col = square // 8, square % 8

# probe_name_list = ["mine", "flipped", "just_played"]
# for layer in range(2):
#     for head in range(n_heads):
#         fig, axs = plt.subplots(3, 3, figsize=(10, 9))
#         fig.suptitle(f"probe dirs for Layer {layer} Head {head} for dst {label}", fontsize=16)
#         for i, probe_name1 in enumerate(probe_name_list):
#             for j, probe_name2 in enumerate(probe_name_list):
#                 ax = axs[i, j]
#                 probe_src = probes[probe_name1][layer]
#                 probe_dst = probes[probe_name2][layer, :, row, col]
#                 scr_QK = einops.einsum(
#                     probe_src,
#                     W_QK[layer, head],
#                     "d_model_src row col, d_model_dst d_model_src -> d_model_dst row col",
#                 )  # [8, 8, 8, 8]
#                 scr_QK_norm = scr_QK / scr_QK.norm(dim=0, keepdim=True)
#                 probe_dst_norm = probe_dst / probe_dst.norm(dim=0, keepdim=True)
#                 cos_sim = einops.einsum(
#                     scr_QK_norm,
#                     probe_dst_norm,
#                     "d_model_dst row col, d_model_dst -> row col",
#                 ).cpu().numpy()
#                 im = ax.imshow(cos_sim, cmap="viridis", aspect="auto")
#                 ax.set_title(f"{probe_name1} to {probe_name2}")
#                 ax.set_xticks(range(8))
#                 ax.set_yticks(range(8))
#                 ax.set_yticklabels(list("ABCDEFGH"))
#                 fig.colorbar(im, ax=ax)
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.show()

# %% heatmap for probe directions OV
label = "C1"
square = label_to_square(label)
row, col = square // 8, square % 8

probe_name_pair = [("flipped", "mine"), ("just_played", "mine"), ("mine", "mine")]
color_map = {
    "Yours head": "red",
    "Mine head": "blue",
    "Other": "gray",
}

for probe_name1, probe_name2 in probe_name_pair:
    n_layer_select = 4
    n_heads = model.cfg.n_heads
    
    # First pass: compute all cos_sim values to find global min/max
    all_cos_sims = []
    for layer in range(n_layer_select):
        for head in range(n_heads):
            probe_src = probes[probe_name1][layer]
            probe_dst = probes[probe_name2][layer]
            src_OV = einops.einsum(
                probe_src, W_OV[layer, head],
                "d_model_src row col, d_model_src d_model_dst -> d_model_dst row col",
            )
            src_OV_norm = src_OV / src_OV.norm(dim=0, keepdim=True)
            probe_dst_norm = probe_dst / probe_dst.norm(dim=0, keepdim=True)
            cos_sim = einops.einsum(
                src_OV_norm, probe_dst_norm,
                "d_model_dst row col, d_model_dst row col -> row col",
            ).cpu().numpy()
            all_cos_sims.append(cos_sim)
    
    # Get global min and max
    vmin = min(np.nanmin(cs) for cs in all_cos_sims)
    vmax = max(np.nanmax(cs) for cs in all_cos_sims)
    v_abs = max(abs(vmin), abs(vmax))
    vmin = -v_abs
    vmax = v_abs
    
    # Create figure with space for colorbar
    fig, axs = plt.subplots(n_layer_select, n_heads, figsize=(3*n_heads, 3*n_layer_select+1.5))
    fig.suptitle(f"probe dirs for {probe_name1} to {probe_name2}", fontsize=16)
    
    # Second pass: plot with consistent colorbar
    idx = 0
    for layer in range(n_layer_select):
        for head in range(n_heads):
            ax = axs[layer, head]
            cos_sim = all_cos_sims[idx]
            idx += 1
            head_type = head_type_all[str(layer)][str(head)]

            im = ax.imshow(cos_sim, cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_title(f"L{layer}H{head} -- {head_type}", color=color_map[head_type])
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_yticklabels(list("ABCDEFGH"))
    
    # Add one large colorbar on the right
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)
    
    plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
    plt.show()

# %%
