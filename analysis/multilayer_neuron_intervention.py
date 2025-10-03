# %%
import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path
import gzip

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as t
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(BASE_PATH)
BASE_PATH = Path(BASE_PATH)
os.chdir(BASE_PATH)

import utils.circuits_utils as circuits_utils
from utils.helper_fns import (
    calculate_ablation_scores_square_probability,
)
from decision_trees import dtypes
# import decision_trees

sys.modules['dtypes'] = dtypes

# %%
# device = "cuda" if t.cuda.is_available() else "cpu"
device = "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = circuits_utils.get_model(model_name, device)

n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp

# %%
# binary_decision_tree_dict = defaultdict(dict)
# binary_dt_f1 = defaultdict(dict)
# for layer in range(n_layers):
#     gt_class_path = (
#         BASE_PATH
#         / "decision_trees"
#         / "ground_truth_features"
#         / "classification"
#         / "results"
#         / f"layer_{layer}_trees.pkl.gz"
#     )
#     with gzip.open(gt_class_path, "rb") as f:
#         gt_feature_classifiers = pickle.load(f)
    
#     for gt_binary in gt_feature_classifiers:
#         try:
#             check_is_fitted(gt_binary.tree)
#             binary_decision_tree_dict[gt_binary.layer][gt_binary.neuron] = gt_binary.tree
#             binary_dt_f1[gt_binary.layer][gt_binary.neuron] = gt_binary.test_F1
#         except NotFittedError:
#             print(f"Tree L{gt_binary.layer}N{gt_binary.neuron} is NOT fitted")
#             continue

# f1_threshold = 0.7
# binary_dt_f1_filter_dict = {layer: {neuron: score for neuron, score in scores.items() if score >= f1_threshold} for layer, scores in binary_dt_f1.items()}
# binary_dt_filter_layer_neurons = {layer: list(scores.keys()) for layer, scores in binary_dt_f1_filter_dict.items()}

# %%
reg_decision_tree_dict = defaultdict(dict)
reg_dt_r2 = defaultdict(dict)

for layer in range(n_layers):
    gt_reg_path = (
        BASE_PATH
        / "decision_trees"
        / "ground_truth_features"
        / "regression"
        / "results"
        / f"layer_{layer}_trees.pkl.gz"
    )
    with gzip.open(gt_reg_path, "rb") as f:
        gt_feature_regressors = pickle.load(f)
    
    for gt_reg in gt_feature_regressors:
        try:
            check_is_fitted(gt_reg.tree)
            reg_decision_tree_dict[layer][gt_reg.neuron] = gt_reg.tree
            reg_dt_r2[layer][gt_reg.neuron] = gt_reg.test_R2
        except NotFittedError:
            print(f"Tree L{layer}N{gt_reg.neuron} is NOT fitted")
            continue

r2_threshold = 0.7
reg_dt_r2_filter_dict = {layer: {neuron: score for neuron, score in scores.items() if score >= r2_threshold} for layer, scores in reg_dt_r2.items()}
reg_dt_filter_layer_neurons = {layer: list(scores.keys()) for layer, scores in reg_dt_r2_filter_dict.items()}

# %%
