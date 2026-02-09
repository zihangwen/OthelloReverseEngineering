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
import utils.othello_utils as othello_utils
from utils.probe_utils import (
    # load_probes_and_normalize,
    load_fold_probes_and_normalize,
    calculate_neuron_input_weights,
    calculate_neuron_output_weights,
)
import utils.arena_utils as arena_utils
from utils.arena_utils import (
    label_to_square,
)
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
n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp
# n_heads = model.cfg.n_heads
# d_head = model.cfg.d_head

# %% Load the test dataset and process
test_size = 500
custom_functions = [
    # othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
    # othello_utils.games_batch_to_valid_moves_BLRRC, # (legal move)
    othello_utils.games_batch_to_flipped_classifier_input_BLC,
]
test_data = circuits_utils.construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=test_size,
    split="test", 
    device=device,
)

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)
# board_seqs_id = board_seqs_id[:, 8:30]

board_seqs_square = t.tensor(test_data["decoded_inputs"]).long().to(device)

# board_states, legal_moves, legal_moves_annotation = get_board_states_and_legal_moves(board_seqs_square)
# legal_moves = legal_moves.to(device=device, dtype=t.float32)

flipped_classifier = test_data["games_batch_to_flipped_classifier_input_BLC"].to(device)

# %% Loading Probes and model
probes = load_fold_probes_and_normalize(n_layers, device)
probe_layer_specific = {
    name: probes[name][5]
    for name in probes.keys()
}
probe_layer_normalized = {
    name: probes[name] / probes[name].norm(dim=1, keepdim=True)
    for name in probes.keys()
}

# %%
w_out = model.W_out.detach().clone() # [layer, neuron, d_model]
b_out = model.b_out.detach().clone() # [layer, d_model]

# W_Q = model.W_Q.detach().clone()  # [layer, head, d_model, d_head]
# W_K = model.W_K.detach().clone()  # [layer, head, d_model, d_head]
# W_O = model.W_O.detach().clone()  # [layer, head, d_head, d_model]
# W_V = model.W_V.detach().clone()  # [layer, head, d_model, d_head]

# W_E = model.W_E[1:].detach().clone()  # [vocab_size, d_model]
# W_U = model.W_U[:, 1:].detach().clone()  # [d_model, 60]

# orginal probe projection calculation
# flipped_attribution = einops.einsum(
#     w_out,
#     probes["flipped"],
#     "layer neuron d_model, layer d_model ... -> layer neuron ...",
# )  # [layer, neuron, probe_dim]

# probe (of one specific layer) projection calculation
flipped_attribution = einops.einsum(
    w_out,
    probe_layer_specific["flipped"],
    "layer neuron d_model, d_model ... -> layer neuron ...",
)  # [layer, neuron, probe_dim]

# %%
# square_idx = 27
square_idx = 18
# token_id = arena_utils.SQUARE_TO_ID[square_idx]
square_label = arena_utils.to_board_label(square_idx)
print(f"Square {square_idx} ({square_label})")

flipped_classifier_mask = flipped_classifier[..., square_idx]  # [game, seq]

neuron_attribution = {}
with t.no_grad(), model.trace(board_seqs_id, scan=False, validate=False):
    for layer in range(n_layers):
        neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output
        flipped_attr_l = flipped_attribution[layer].flatten(start_dim=-2, end_dim=-1)[..., square_idx]

        neuron_attr = einops.einsum(
            neuron_activations_BLD,
            flipped_attr_l,
            "game seq neuron, neuron -> game seq neuron",
        )

        neuron_attribution[layer] = neuron_attr.save()  # [game, seq, neuron]

neuron_attribution = t.stack(
    [neuron_attribution[layer] for layer in range(n_layers)], dim=-2
)  # [game, seq, layer, neuron]

neuron_attribution_selected = neuron_attribution * flipped_classifier_mask[..., None, None]  # [game, seq, layer, neuron]
neuron_attribution_selected = neuron_attribution_selected.sum(dim=(0,1))  # [game, layer, neuron]

# %% TEST
# layer = 0
# with t.no_grad(), model.trace(board_seqs_id, scan=False, validate=False):
#     neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output.save()
#     mlp_out = model.blocks[layer].hook_mlp_out.output.save()

# mlp_out - (einops.einsum(
#     neuron_activations_BLD,
#     w_out[layer, :, :],
#     "batch seq d_mlp, d_mlp d_model -> batch seq d_model",
# ) + b_out[layer, :])

# %% histogram of attributions for a layer
# layer = 0
# plt.figure(figsize=(8, 6))
# plt.hist(
#     neuron_attribution_selected[layer].cpu().numpy().flatten(),
#     bins=100,
#     color="blue",
#     alpha=0.7,
# )
# plt.title(f"Histogram of Neuron Attributions for Layer {layer}", fontsize=16)
# plt.xlabel("Attribution Value", fontsize=14)
# plt.ylabel("Frequency", fontsize=14)
# plt.grid(True)
# plt.show()

# %% index v.s. attribution plot for some layers with shared y-axis (scatter plots)
layers_to_plot = [0, 1]
fig, axs = plt.subplots(1, len(layers_to_plot), figsize=(6 * len(layers_to_plot), 5), sharey=True)
for i, layer in enumerate(layers_to_plot):
    axs[i].scatter(
        t.arange(n_neurons).cpu().numpy(),
        neuron_attribution_selected[layer].cpu().numpy(),
        color="blue",
        alpha=0.7,
        s=10,
    )
    axs[i].set_title(f"Neuron Attributions for Layer {layer}", fontsize=16)
    axs[i].set_xlabel("Neuron Index", fontsize=14)
    if i == 0:
        axs[i].set_ylabel("Attribution Value", fontsize=14)
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# %%
# layer = 0
# topk_neurons_seperate = defaultdict(list)
# topk_neuron_idx = t.topk(neuron_attribution_selected[0], k=2048).indices
# for i_k, idx in enumerate(topk_neuron_idx):
#     topk_neurons_seperate[i_k] = [layer, idx.item()]

topk_neurons_seperate = defaultdict(list)
topk_neuron_idx = t.topk(neuron_attribution_selected[[0,1]].flatten(), k=2048).indices
for i_k, idx in enumerate(topk_neuron_idx):
    layer = idx // n_neurons
    neuron = idx % n_neurons
    topk_neurons_seperate[i_k] = [layer.item(), neuron.item()]

# %%
for i_k, (layer, neuron) in topk_neurons_seperate.items():
    if i_k >= 5:
        break

    temp = [
        [
            calculate_neuron_input_weights(model, probe_layer_normalized["blank"][layer], layer, neuron).numpy(),
            calculate_neuron_input_weights(model, probe_layer_normalized["mine"][layer], layer, neuron).numpy(),
            calculate_neuron_input_weights(model, probe_layer_normalized["flipped"][layer], layer, neuron).numpy(),
            calculate_neuron_input_weights(model, probe_layer_normalized["just_played"][layer], layer, neuron).numpy(),
        ],
        [
            calculate_neuron_output_weights(model, probe_layer_normalized["blank"][layer], layer, neuron).numpy(),
            calculate_neuron_output_weights(model, probe_layer_normalized["mine"][layer], layer, neuron).numpy(),
            calculate_neuron_output_weights(model, probe_layer_normalized["flipped"][layer], layer, neuron).numpy(),
            calculate_neuron_output_weights(model, probe_layer_normalized["just_played"][layer], layer, neuron).numpy(),
        ]
    ]
    vmin = np.nanmin(temp)
    vmax = np.nanmax(temp)
    v_abs = max(abs(vmin), abs(vmax))
    vmin = -v_abs
    vmax = v_abs

    fig, axs = plt.subplots(2, 4, figsize=(3*4, 3*2+1.5))
    fig.suptitle(f"(Rank {i_k}: L{layer}N{neuron}) MLP Weights Projection for square {square_label}", fontsize=16)
    
    for i, weight in enumerate(["w_in", "w_out"]):
        for j, probe_name in enumerate(["blank", "mine", "flipped", "just_played"]):
            ax = axs[i, j]
            im = ax.imshow(temp[i][j], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_title(f"{weight} @ {probe_name} probe ", fontsize=14)
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_yticklabels(list("ABCDEFGH"))
    
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
    plt.show()

# %%
    
