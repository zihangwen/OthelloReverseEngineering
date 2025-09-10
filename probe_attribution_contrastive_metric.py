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

# %% writing neuron
n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp

# %% ----- ----- ----- ----- ----- probe diff ----- ----- ----- ----- ----- %% #
# # %%
# probe_dict = {i : t.load(
#     f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
# )['linear_probe'].squeeze() for i in range(n_layers)}

# probe_t = t.stack([probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]
# blank_probe = probe_t[..., 1] - (probe_t[..., 0] + probe_t[..., 2]) * 0.5  # [layer, d_model, row, col]
# my_probe = probe_t[..., 0] - probe_t[..., 2]  # [layer, d_model, row, col]

# blank_probe_normalized = blank_probe / blank_probe.norm(dim=1, keepdim=True)
# my_probe_normalized = my_probe / my_probe.norm(dim=1, keepdim=True)
# blank_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# # %%
# flipped_probe_dict = {i : t.load(
#     f"linear_probes_flipped/resid_{i}_flipped.pth", map_location=str(device), weights_only="True"
# ).squeeze() for i in range(n_layers)}

# flipped_probe_t = t.stack([flipped_probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]

# flipped_probe = flipped_probe_t[..., 0] - flipped_probe_t[..., 1]  # [layer, d_model, row, col]
# flipped_probe_normalized = flipped_probe / flipped_probe.norm(dim=1, keepdim=True)

# # %%
# just_played_probe_dict = {i : t.load(
#     f"linear_probes_just_played/resid_{i}_played.pth", map_location=str(device), weights_only="True"
# ).squeeze() for i in range(n_layers)
# }

# just_played_probe = t.stack([just_played_probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col]
# just_played_probe_normalized = just_played_probe / just_played_probe.norm(dim=1, keepdim=True)
# just_played_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# # %%
# layer = 5
# neuron = 766

# w_in_LN_blank = calculate_neuron_input_weights(model, blank_probe_normalized[layer-1], layer, neuron)
# w_in_LN_my = calculate_neuron_input_weights(model, my_probe_normalized[layer-1], layer, neuron)
# w_in_LN_flipped = calculate_neuron_input_weights(model, flipped_probe_normalized[layer-1], layer, neuron)
# w_in_LN_just_played = calculate_neuron_input_weights(model, just_played_probe_normalized[layer-1], layer, neuron)

# # %%
# matrices = t.stack(
#     [w_in_LN_blank, w_in_LN_my, w_in_LN_flipped, w_in_LN_just_played], dim=0
# )  # [4, 8, 8]
# titles = [
#     f"Blank In L{layer}N{neuron}", f"My In L{layer}N{neuron}",
#     f"Flipped In L{layer}N{neuron}", f"Just Played In L{layer}N{neuron}",
# ]

# fig = arena_utils.plot_board_values(
#     matrices,
#     title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
#     board_titles=titles,
#     boards_per_row=2,
#     width=650,
#     height=760,
# )

# # %%
# k = 2
# matrices_mean = matrices.mean().item()
# matrices_std = matrices.std().item()

# fig, ax = plt.subplots(figsize=(8, 6))
# plt.hist(matrices.flatten().cpu(), bins=50, alpha=0.5)

# plt.axvline(matrices_mean + k*matrices_std, color='blue', linestyle='dashed', linewidth=1)
# plt.axvline(matrices_mean - k*matrices_std, color='orange', linestyle='dashed', linewidth=1)
# plt.plot()

# # %%
# blank_feature_names = []
# mine_feature_names = []
# flipped_feature_names = []
# just_played_feature_names = []
# for square_idx in range(64):
#     row = square_idx // 8  
#     col = square_idx % 8
#     square = chr(ord('A') + row) + str(col)
    
#     # Add the 3 states for this square
#     blank_feature_names.append(f"{square}_blank")
#     mine_feature_names.append(f"{square}_mine")
#     flipped_feature_names.append(f"{square}_flipped")
#     just_played_feature_names.append(f"{square}_just_played")

# all_feature_names = [
#     blank_feature_names,
#     mine_feature_names,
#     flipped_feature_names,
#     just_played_feature_names
# ]

# # %%
# filtered_feature_names = []
# directional_feature_names = []
# for i in range(len(matrices)):
#     matrix = matrices[i].flatten()
#     feature_names = all_feature_names[i]
#     filtered_feature_names += [feature_names[j] for j in range(64) if abs(matrix[j] - matrices_mean) > k*matrices_std]
#     directional_feature_names += [
#         f"({feature_names[j]})" if matrix[j] > matrices_mean else f"(NOT {feature_names[j]})" 
#         for j in range(64) if abs(matrix[j] - matrices_mean) > k*matrices_std
#     ]

# %% ----- ----- ----- ----- ----- probe single direction ----- ----- ----- ----- ----- %% #
# %%
probe_dict = {i : t.load(
    f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
)['linear_probe'].squeeze() for i in range(n_layers)}

probe_t = t.stack([probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]
mine_probe = probe_t[..., 0]
empty_probe = probe_t[..., 1]
theirs_probe = probe_t[..., 2]  # [layer, d_model, row, col]

mine_probe_normalized = mine_probe / mine_probe.norm(dim=1, keepdim=True)
empty_probe_normalized = empty_probe / empty_probe.norm(dim=1, keepdim=True)
theirs_probe_normalized = theirs_probe / theirs_probe.norm(dim=1, keepdim=True)
empty_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# %%
flipped_probe_dict = {i : t.load(
    f"linear_probes_flipped/resid_{i}_flipped.pth", map_location=str(device), weights_only="True"
).squeeze() for i in range(n_layers)}

flipped_probe_t = t.stack([flipped_probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]

flipped_probe = flipped_probe_t[..., 0]
flipped_probe_normalized = flipped_probe / flipped_probe.norm(dim=1, keepdim=True)

# %%
just_played_probe_dict = {i : t.load(
    f"linear_probes_just_played/resid_{i}_played.pth", map_location=str(device), weights_only="True"
).squeeze() for i in range(n_layers)
}

just_played_probe = t.stack([just_played_probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col]
just_played_probe_normalized = just_played_probe / just_played_probe.norm(dim=1, keepdim=True)
just_played_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# %%
layer = 5
neuron = 766

w_in_LN_mine = calculate_neuron_input_weights(model, mine_probe_normalized[layer-1], layer, neuron)
w_in_LN_empty = calculate_neuron_input_weights(model, empty_probe_normalized[layer-1], layer, neuron)
w_in_LN_theirs = calculate_neuron_input_weights(model, theirs_probe_normalized[layer-1], layer, neuron)
w_in_LN_flipped = calculate_neuron_input_weights(model, flipped_probe_normalized[layer-1], layer, neuron)
w_in_LN_just_played = calculate_neuron_input_weights(model, just_played_probe_normalized[layer-1], layer, neuron)

# %%
matrices = t.stack(
    [w_in_LN_mine, w_in_LN_empty, w_in_LN_theirs, w_in_LN_flipped, w_in_LN_just_played], dim=0
)  # [4, 8, 8]
titles = [
    f"Mine In L{layer}N{neuron}", f"Empty In L{layer}N{neuron}", f"Theirs In L{layer}N{neuron}",
    f"Flipped In L{layer}N{neuron}", f"Just Played In L{layer}N{neuron}",
]

fig = arena_utils.plot_board_values(
    matrices,
    title=f"Input weights cosine similarity with the probe for neuron L{layer}N{neuron}",
    board_titles=titles,
    boards_per_row=3,
    width=975,
    height=760,
)

fig.write_image(f"figures/week8-2/neuron_input_weights_L{layer}N{neuron}.png")

# %%
k = 2
matrices_mean = matrices.mean().item()
matrices_std = matrices.std().item()

fig, ax = plt.subplots(figsize=(8, 6))
plt.hist(matrices.flatten().cpu(), bins=50, alpha=0.5)

plt.axvline(matrices_mean + k*matrices_std, color='blue', linestyle='dashed', linewidth=1, label=f'Mean + {k}*STD')
plt.axvline(matrices_mean - k*matrices_std, color='orange', linestyle='dashed', linewidth=1, label=f'Mean - {k}*STD')
plt.legend()
plt.title(f"Histogram of cosine similarity for neuron L{layer}N{neuron}")
plt.xlabel("Weight value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"figures/week8-2/L{layer}N{neuron}_input_weights_hist.png", dpi=300, bbox_inches='tight')
# plt.plot()

# %%
# mine_feature_names = []
# empty_feature_names = []
# theirs_feature_names = []
# flipped_feature_names = []
# just_played_feature_names = []
# for square_idx in range(64):
#     row = square_idx // 8  
#     col = square_idx % 8
#     square = chr(ord('A') + row) + str(col)
    
#     # Add the 3 states for this square
#     mine_feature_names.append(f"{square}_mine")
#     empty_feature_names.append(f"{square}_empty")
#     theirs_feature_names.append(f"{square}_theirs")
#     flipped_feature_names.append(f"{square}_flipped")
#     just_played_feature_names.append(f"{square}_just_played")

# all_feature_names = [
#     mine_feature_names,
#     empty_feature_names,
#     theirs_feature_names,
#     flipped_feature_names,
#     just_played_feature_names
# ]

# filtered_feature_names2 = []
# directional_feature_names2 = []
# for i in range(len(matrices)):
#     matrix = matrices[i].flatten()
#     feature_names = all_feature_names[i]
#     filtered_feature_names2 += [feature_names[j] for j in range(64) if abs(matrix[j] - matrices_mean) > k*matrices_std]
#     directional_feature_names2 += [
#         f"({feature_names[j]})" if matrix[j] > matrices_mean else f"(NOT {feature_names[j]})" 
#         for j in range(64) if abs(matrix[j] - matrices_mean) > k*matrices_std
#     ]

# %%
def extract_probe_features(matrices, k=2):
    matrices_mean = matrices.mean().item()
    matrices_std = matrices.std().item()

    filtered_feature_names = []
    directional_feature_names = []
    for row in range(8):
        for col in range(8):
            square = chr(ord('A') + row) + str(col)
            mine_weight = matrices[0, row, col].item()
            empty_weight = matrices[1, row, col].item()
            theirs_weight = matrices[2, row, col].item()
            flipped_weight = matrices[3, row, col].item()
            just_played_weight = matrices[4, row, col].item()

            occupied = 0
            if mine_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_mine")
                directional_feature_names.append(f"({square}_mine)")
                # occupied = 1

            if mine_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_mine")
                directional_feature_names.append(f"(NOT {square}_mine)")

            if theirs_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_theirs")
                directional_feature_names.append(f"({square}_theirs)")
                # occupied = 1
            
            if theirs_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_theirs")
                directional_feature_names.append(f"(NOT {square}_theirs)")

            if empty_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_empty")
                directional_feature_names.append(f"({square}_empty)")
            
            if empty_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_empty")
                directional_feature_names.append(f"(NOT {square}_empty)")
                # occupied = 1
            
            if flipped_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_flipped")
                directional_feature_names.append(f"({square}_flipped)")
                # occupied = 1
            
            if flipped_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_flipped")
                directional_feature_names.append(f"(NOT {square}_flipped)")

            if just_played_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_just_played")
                directional_feature_names.append(f"({square}_just_played)")
                # occupied = 1
            
            if just_played_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_just_played")
                directional_feature_names.append(f"(NOT {square}_just_played)")

    return filtered_feature_names, directional_feature_names

# %%
filtered_feature_names, directional_feature_names = extract_probe_features(matrices, k=2)
print("Filtered feature names:")
for name in filtered_feature_names:
    print(name)
print("\nDirectional feature names:")
for name in directional_feature_names:
    print(name)
    
# %%
