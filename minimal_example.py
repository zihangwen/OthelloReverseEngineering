# %%
# Minimal Example for OthelloUnderstanding Codebase
# This example demonstrates how to:
# 1. Load + run the Othello model
# 2. Load + run linear probes
# 3. Load + visualize decision trees
import pickle
from collections import defaultdict
import torch as t
import numpy as np
import einops
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import to_numpy, get_act_name
from torch import Tensor
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
import neuron_simulation.arena_utils as arena_utils

from helper_fns import (
    MIDDLE_SQUARES,
    ALL_SQUARES,
    get_board_states_and_legal_moves,
    plot_probe_outputs,
    get_w_in,
    get_w_out,
    calculate_neuron_input_weights,
    calculate_neuron_output_weights,
    create_feature_names,
)

device = "cuda" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %% Load the OthelloGPT model
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = utils.get_model(model_name, device)

# %% Load probes
probe_dict = {i : t.load(
    f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
)['linear_probe'].squeeze() for i in range(model.cfg.n_layers)}

# %% Load decision trees
dt_name = 'neuron_simulation/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl'
with open(dt_name, "rb") as f:
    decision_trees = pickle.load(f)

function_name = list(decision_trees[0].keys())[0]
n_features = decision_trees[0][function_name]["decision_tree"]["model"].n_features_in_
feature_names = create_feature_names(n_features, function_name)

# %% Load the test dataset and process
test_size = 500
custom_functions = [
    othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
]
# train_data = construct_othello_dataset(
#     custom_functions=custom_functions,
#     n_inputs=dataset_size,
#     split="train",
#     device=device,
# )
test_data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=test_size,
    split="test", 
    device=device,
)

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long()
board_seqs_square = t.tensor(test_data["decoded_inputs"]).long()

# %%
num_games = 50

focus_games_id = board_seqs_id[:num_games]  # shape [50, 59]
focus_games_square = board_seqs_square[:num_games]  # shape [50, 59]
focus_states, focus_legal_moves, focus_legal_moves_annotation = (
    get_board_states_and_legal_moves(focus_games_square)
)

print("focus states:", focus_states.shape)
print("focus_legal_moves", tuple(focus_legal_moves.shape))

# Plot the first 10 moves of the first game
arena_utils.plot_board_values(
    focus_states[0, :10],
    title="Board states",
    width=1000,
    boards_per_row=5,
    board_titles=[
        f"Move {i}, {'white' if i % 2 == 1 else 'black'} to play" for i in range(1, 11)
    ],
    text=np.where(to_numpy(focus_legal_moves[0, :10]), "o", "").tolist(),
)

# %%
focus_logits, focus_cache = model.run_with_cache(focus_games_id.to(device))
W_U_normalized = model.W_U[:, 1:] / model.W_U[:, 1:].norm(dim=0, keepdim=True)

# %%
# game_index = 0
# move = 29

# arena_utils.plot_board_values(
#     focus_states[game_index, move],
#     title="Focus game states",
#     width=400,
#     height=400,
#     text=focus_legal_moves_annotation[game_index][move],
# )

# for layer in range(model.cfg.n_layers):
#     plot_probe_outputs(
#         focus_cache,
#         probe_dict,
#         layer,
#         game_index,
#         move,
#         title=f"Layer {layer} probe outputs after move {move}",
#     )

# %%
layer = 5
blank_probe = probe_dict[layer][..., 1] - (probe_dict[layer][..., 0] + probe_dict[layer][..., 2]) / 2
my_probe = probe_dict[layer][..., 0] - probe_dict[layer][..., 2]
# Scale the probes down to be unit norm per cell

blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)

# Set the center blank probes to 0, since they're never blank so the probe is meaningless
blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

dt_model = decision_trees[layer][function_name]['decision_tree']['model']
r2_scores = decision_trees[layer][function_name]['decision_tree']['r2']

# %%
neuron = 1393

w_in_L5N1393_blank = calculate_neuron_input_weights(model, blank_probe_normalised, layer, neuron)
w_in_L5N1393_my = calculate_neuron_input_weights(model, my_probe_normalised, layer, neuron)
arena_utils.plot_board_values(
    t.stack([w_in_L5N1393_blank, w_in_L5N1393_my]),
    title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
    board_titles=["Blank In", "My In"],
    width=650,
    height=380,
)

w_out_L5N1393_blank = calculate_neuron_output_weights(model, blank_probe_normalised, layer, neuron)
w_out_L5N1393_my = calculate_neuron_output_weights(model, my_probe_normalised, layer, neuron)
arena_utils.plot_board_values(
    t.stack([w_out_L5N1393_blank, w_out_L5N1393_my]),
    title=f"Output weights in terms of the probe for neuron L{layer}N{neuron}",
    board_titles=["Blank Out", "My Out"],
    width=650,
    height=380,
)

neuron_tree = dt_model.estimators_[neuron]
neuron_r2 = r2_scores[neuron].item()
plt.figure(figsize=(20, 12))
plot_tree(
    neuron_tree,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3
)
plt.title(f"Decision Tree for Layer {layer}, Neuron {neuron}\nR² Score: {neuron_r2:.4f}", 
              fontsize=16, pad=20)
plt.show()

# %%
w_in_L5N1393 = get_w_in(model, layer, neuron, normalize=True)
w_out_L5N1393 = get_w_out(model, layer, neuron, normalize=True)

cos_sim = w_out_L5N1393 @ W_U_normalized  # shape (60,)

# Turn into a (rows, cols) tensor, using indexing
cos_sim_rearranged = t.zeros((8, 8), device=device)
cos_sim_rearranged.flatten()[ALL_SQUARES] = cos_sim

# Plot results
arena_utils.plot_board_values(
    cos_sim_rearranged,
    title=f"Cosine sim of neuron L{layer}N{neuron} with W<sub>U</sub> directions",
    width=450,
    height=380,
)

# %%
layers = [_ for _ in range(model.cfg.n_layers)]
neuron_acts = {}
with t.no_grad(), model.trace(focus_games_id.to(device), scan=False, validate=False):
    for layer in layers:
        neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output.save()
        neuron_acts[layer] = neuron_activations_BLD

# %%
# (focus_cache[get_act_name("mlp_post",0)] == neuron_acts[0]).all()
# t.isclose(focus_cache[get_act_name("mlp_post",0)], neuron_acts[0]).all()
