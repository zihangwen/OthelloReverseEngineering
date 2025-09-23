# %%
import os
import sys
import pickle
from collections import defaultdict
import torch as t
import numpy as np
from pathlib import Path

import circuits_utils
import arena_utils
from feature_extraction_utils import (
    create_feature_names,
    extract_probe_features,
    extract_rules_features_from_binary_dt,
    set_overlap_metrics,
)
from probe_utils import (
    load_probes_and_normalize,
    calculate_w_in_cossim_with_probes,
)
# from helper_fns import (
#     # MIDDLE_SQUARES,
#     neuron_intervention,
#     ALL_SQUARES,
#     get_board_states_and_legal_moves,
#     calculate_ablation_scores_game_move,
#     calculate_ablation_scores_square,
#     # plot_probe_outputs,
#     get_w_in,
#     # get_w_out,
#     calculate_neuron_input_weights,
#     calculate_neuron_output_weights,
#     create_feature_names,
#     get_neuron_decision_tree,
#     get_neuron_binary_decision_tree,
#     get_feature_names_cont_dt,
#     # visualize_decision_tree,
# )

# %%
BASE_PATH = Path("/home/zihangw/Algoverse/OthelloReverseEngineering")
os.chdir(BASE_PATH)

# device = "cuda" if t.cuda.is_available() else "cpu"
device = "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = circuits_utils.get_model(model_name, device)

n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp

# %% Binary dt load
binary_dt_name = 'neuron_decision_trees/decision_trees_d8/decision_trees_mlp_neuron_30000.pkl'
with open(binary_dt_name, "rb") as f:
    binary_decision_trees = pickle.load(f)

binary_custom_function_name = list(binary_decision_trees[0].keys())[0]
n_binary_features = binary_decision_trees[0][binary_custom_function_name]["binary_decision_tree"]["model"].n_features_in_
binary_feature_names = create_feature_names(n_binary_features, binary_custom_function_name)

# %% probes load
probes = load_probes_and_normalize(n_layers, device)

# %% binary dt feature extraction
f1_threshold = 0.7
dt_rules = extract_rules_features_from_binary_dt(
    num_layers = n_layers,
    num_neurons = n_neurons,
    binary_decision_trees = binary_decision_trees,
    custom_function_name = "games_batch_to_board_state_flipped_played_BLC",
    binary_feature_names = binary_feature_names,
    f1_threshold=f1_threshold,
)

# %% probe feature extraction
layer = 7
neuron = 255

matrices = calculate_w_in_cossim_with_probes(
    model,
    probes,
    layer,
    neuron,
    layer_offset=0,
)

filtered_feature_names, directional_feature_names = extract_probe_features(matrices, k=2)

# %%
