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

from utils.feature_extraction_utils import (
    aggregate_scores,
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
# model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
# model = circuits_utils.get_model(model_name, device)

# n_layers = model.cfg.n_layers
# n_neurons = model.cfg.d_mlp

n_layers = 8
n_neurons = 2048

# %%
binary_decision_tree_dict = defaultdict(dict)
binary_dt_f1 = defaultdict(dict)
for layer in range(n_layers):
    gt_class_path = (
        BASE_PATH
        / "decision_trees"
        / "ground_truth_features"
        / "classification"
        / "results"
        / f"layer_{layer}_trees.pkl.gz"
    )
    with gzip.open(gt_class_path, "rb") as f:
        gt_feature_classifiers = pickle.load(f)
    
    for gt_binary in gt_feature_classifiers:
        try:
            check_is_fitted(gt_binary.tree)
            binary_decision_tree_dict[gt_binary.layer][gt_binary.neuron] = gt_binary.tree
            binary_dt_f1[gt_binary.layer][gt_binary.neuron] = gt_binary.test_F1
        except NotFittedError:
            print(f"Tree L{gt_binary.layer}N{gt_binary.neuron} is NOT fitted")
            continue

f1_threshold = 0.7
# binary_dt_f1_filter = {layer: [score for score in scores.values() if score >=0] for layer, scores in binary_dt_f1.items()}

binary_dt_f1_filter_dict = {layer: {neuron: score for neuron, score in scores.items() if score >= f1_threshold} for layer, scores in binary_dt_f1.items()}
binary_dt_f1_filter = {layer: list(scores.values()) for layer, scores in binary_dt_f1_filter_dict.items()}

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
# reg_dt_r2_filter = {layer: [score for score in scores.values() if score >=0] for layer, scores in reg_dt_r2.items()}

reg_dt_r2_filter_dict = {layer: {neuron: score for neuron, score in scores.items() if score >= r2_threshold} for layer, scores in reg_dt_r2.items()}
reg_dt_r2_filter = {layer: list(scores.values()) for layer, scores in reg_dt_r2_filter_dict.items()}

# %% ripper load
with open(f"ripper/ripper_all_neurons_analysis.pkl", "rb") as f:
    ripper_all_neurons_analysis = pickle.load(f)

ripper_f1 = aggregate_scores(ripper_all_neurons_analysis, score_key="f1_score")

ripper_features = defaultdict(dict)
for layer in ripper_all_neurons_analysis:
    for info in ripper_all_neurons_analysis[layer]:
        neuron_id = info["neuron_id"]
        # features = info["feature_weights"].keys()
        feature_names = set()
        directional_feature_names = set()
        for feat_name, feat_score in info["top_features"]:
            feature_names.update({f"{feat_name}"})
            if feat_score > 0:
                directional_feature_names.update({f"({feat_name})"})
            else:
                directional_feature_names.update({f"(NOT {feat_name})"})
        ripper_features[layer][neuron_id] = {
            "feature_names": feature_names,
            "directional_feature_names": directional_feature_names,
        }

ripper_f1_filter = {layer: [score for score in scores if score >=0] for layer, scores in ripper_f1.items()}

# %% lasso load
lasso_results = dict()
for layer in range(n_layers):
    with open(f"lasso/layer{layer}_results.pkl", "rb") as f:
        lasso_results[layer] = pickle.load(f)

lasso_r2 = {layer: lasso_results[layer]['per_neuron_r2'] for layer in range(n_layers)}
lasso_r2_filter = {layer: [score for score in scores if score >= 0] for layer, scores in lasso_r2.items()}

lasso_features = defaultdict(dict)
for layer in range(n_layers):
    for neuron, info in lasso_results[layer]['selected_features_by_neuron'].items():
        feature_names = set()
        directional_feature_names = set()
        for id, feat_name, feat_score in info:
            feature_names.update({f"{feat_name}"})
            if feat_score > 0:
                directional_feature_names.update({f"({feat_name})"})
            else:
                directional_feature_names.update({f"(NOT {feat_name})"})
        lasso_features[layer][neuron] = {
            "feature_names": feature_names,
            "directional_feature_names": directional_feature_names,
        }

# %%
x = np.arange(n_layers)
width = 0.35

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)

ax_r2 = fig.add_subplot(gs[0, 0])
ax_f1 = fig.add_subplot(gs[0, 1], sharey=ax_r2)
ax_r2_neuron = fig.add_subplot(gs[1, 0])
ax_f1_neuron = fig.add_subplot(gs[1, 1], sharey=ax_r2_neuron)


# Left: R²
# ax_r2.bar(x - width/2, reg_dt_r2, width, label='Regression DT R²')
# ax_r2.bar(x + width/2, lasso_r2, width, label='Regression lasso R²')
ax_r2.boxplot([reg_dt_r2_filter.get(l, []) for l in range(n_layers)], positions=x - width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="skyblue"), label='Regression DT R²')
ax_r2.boxplot([lasso_r2_filter.get(l, []) for l in range(n_layers)], positions=x + width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="orange"), label='Regression lasso R²')
ax_r2.set_xticks(x)
ax_r2.set_xticklabels([f"layer {layer}" for layer in range(n_layers)], rotation=45)
ax_r2.set_ylabel("R² score")
# ax_r2.set_ylim(0, 1)
ax_r2.set_title("R² across neurons per Layer")
# ax_r2.legend()

# Right: F1
# ax_f1.bar(x - width/2, binary_dt_f1, width, label='Binary DT F1')
# ax_f1.bar(x + width/2, ripper_f1, width, label='RIPPER F1')
ax_f1.boxplot([binary_dt_f1_filter.get(l, []) for l in range(n_layers)], positions=x - width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="lightgreen"), label='Binary DT F1')
ax_f1.boxplot([ripper_f1_filter.get(l, []) for l in range(n_layers)], positions=x + width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="salmon"), label='RIPPER F1')
ax_f1.set_xticks(x)
ax_f1.set_xticklabels([f"layer {layer}" for layer in range(n_layers)], rotation=45)
ax_f1.set_ylabel("F1 score")
# ax_f1.set_ylim(0, 1)
ax_f1.set_title("F1 across neurons per Layer")
# ax_f1.legend()

# Bottom Left: R² Neuron Counts
r2_threshold_list = [0.7, 0.8, 0.9]
for i, threshold in enumerate(r2_threshold_list):
    reg_counts = [len([score for score in scores if score >= threshold]) for _, scores in reg_dt_r2_filter.items()]

    ax_r2_neuron.plot(x, reg_counts, marker='o', label=f'Regression DT R² ≥ {threshold}', color=plt.cm.Blues((3 - i) / len(r2_threshold_list)))

lasso_counts = [len([score for score in scores if score >= 0.7]) for _, scores in lasso_r2_filter.items()]
ax_r2_neuron.plot(x, lasso_counts, marker='o', label=f'Lasso R² ≥ {0.7}', linestyle='--', color="orange")

ax_r2_neuron.legend()
ax_r2_neuron.set_xticks(x)
ax_r2_neuron.set_xticklabels([f"layer {layer}" for layer in range(n_layers)], rotation=45)
ax_r2_neuron.set_ylabel("Number of Neurons")
ax_r2_neuron.set_title("Neurons with High R² per Layer")

# Bottom Right: F1 Neuron Counts
f1_threshold_list = [0.7, 0.8, 0.9]
for i, threshold in enumerate(f1_threshold_list):
    binary_counts = [len([score for score in scores if score >= threshold]) for _, scores in binary_dt_f1_filter.items()]

    ax_f1_neuron.plot(x, binary_counts, marker='o', label=f'Binary DT F1 ≥ {threshold}', color=plt.cm.Greens((3 - i) / len(f1_threshold_list)))

ripper_counts = [len([score for score in scores if score >= 0.7]) for _, scores in ripper_f1_filter.items()]
ax_f1_neuron.plot(x, ripper_counts, marker='o', label=f'RIPPER F1 ≥ {0.7}', linestyle='--', color="salmon")

ax_f1_neuron.legend()
ax_f1_neuron.set_xticks(x)
ax_f1_neuron.set_xticklabels([f"layer {layer}" for layer in range(n_layers)], rotation=45)
ax_f1_neuron.set_ylabel("Number of Neurons")
ax_f1_neuron.set_title("Neurons with High F1 per Layer")
plt.tight_layout()
# plt.show()
plt.savefig(f"figures/contrastive_analysis/scores.pdf", dpi=300, bbox_inches='tight')

# %%
