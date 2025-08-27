# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
import einops
from collections import defaultdict
from typing import Callable, Optional
import os
import importlib
import pickle
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any

import circuits.utils as utils
import circuits.othello_utils as othello_utils
import neuron_simulation.simulation_config as sim_config
import simulate_activations_with_dts as sim_activations

# %%
default_config = sim_config.selected_config
device = "cpu"

# %%
directory = "neuron_decision_trees/decision_trees_0826_features"
# Updated custom function names to match what you trained with
custom_function_names = [
    othello_utils.games_batch_to_board_state_flipped_played_BLC,
    othello_utils.games_batch_to_board_state_flipped_played_valid_move_BLC,
]

test_size = 6000

ablation_method = "dt"
ablate_not_selected = True
add_error = False

desired_layer_tuples = []
# Generate single layer tuples to match your training
for i in range(8):
    desired_layer_tuples.append((i,))

# %%
data = []
for filename in os.listdir(directory):
    if filename.endswith(".pkl") and "ablation" in filename:
        with open(os.path.join(directory, filename), "rb") as f:
            single_data = pickle.load(f)

        data.append(single_data)

plt.figure(figsize=(12, 8))
# function_names = [
#     "games_batch_to_board_state_flipped_played_BLC",
#     "games_batch_to_board_state_flipped_played_valid_move_BLC",
# ]
colors = ['blue', 'red', 'green', 'orange', 'purple']
markers = ['o', 's', '^', 'D', 'v']

metric = "kl"
plot_count = 0
for i, data_entry in enumerate(data):
    hyperparams = data_entry["hyperparameters"]
    results = data_entry["results"]

    ablation_method = hyperparams["ablation_method"]
    # ablate_not_selected = hyperparams["ablate_not_selected"]
    # add_error = hyperparams["add_error"]
    # input_location = hyperparams["input_location"]

    # Get custom function name from results
    layer_key = list(results.keys())[0]
    custom_function_names = list(results[layer_key].keys())

    for custom_function_name in custom_function_names:
        values = []
        layers = []
        for layer_tuple, layer_results in sorted(results.items()):
            func_results = layer_results[custom_function_name]
            if metric == "kl":
                value = func_results["kl"].item()
            elif metric == "patch_accuracy":
                value = func_results["patch_accuracy"]
            else:
                raise ValueError(f"Invalid metric: {metric}")
            values.append(value)
            layers.append(layer_tuple[0])

        if ablation_method == "dt":
            label = f"ablation: {ablation_method} ({custom_function_name})"
        else:
            label = f"ablation: {ablation_method}"
        plt.plot(
            layers,
            values,
            marker=markers[plot_count % len(markers)],
            color=colors[plot_count % len(colors)],
            linestyle='-',
            linewidth=2,
            markersize=6,
            label=label,
        )
        plot_count += 1
    
plt.legend()
plt.xlabel("Layer")
plt.ylabel(metric)
plt.title(f"Ablation Study - {metric}")
