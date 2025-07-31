# %%
import os
import pickle
import matplotlib.pyplot as plt
# Decision Tree Visualization Tool
# This script allows you to visualize decision trees for specific neurons in specific layers
# Based on the approach shown in dt_tutorial.py but adapted for our neuron decision tree data

import pickle
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
import numpy as np
from typing import Optional, List

# %%
with open("/home/zihangw/Algoverse/OthelloReverseEngineering/neuron_simulation/decision_trees/results_mlp_neuron_trainer_0_inputs_6000.pkl", "rb") as f:
    training_result = pickle.load(f)
    
with open("/home/zihangw/Algoverse/OthelloReverseEngineering/neuron_simulation/decision_trees/ablation_results_mlp_neuron_dt_ablate_not_selected_True_add_error_True_trainer_0_inputs_6000.pkl", "rb") as f:
    ablation_dt = pickle.load(f)

with open("/home/zihangw/Algoverse/OthelloReverseEngineering/neuron_simulation/decision_trees/ablation_results_mlp_neuron_mean_ablate_not_selected_True_add_error_True_trainer_0_inputs_6000.pkl", "rb") as f:
    ablation_mean = pickle.load(f)

# %% 
r2_threshold = 0.7
r2_scores = [
    values['games_batch_to_input_tokens_flipped_bs_classifier_input_BLC']['decision_tree']['r2']
    for _, values in training_result["results"].items()
]
num_features = [(r2_score>r2_threshold).sum() for r2_score in r2_scores]

print(num_features)