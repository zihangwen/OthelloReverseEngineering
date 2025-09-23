# %%
import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as t

import utils.circuits_utils as circuits_utils
import utils.arena_utils as arena_utils
from utils.feature_extraction_utils import (
    create_feature_names,
    extract_probe_features,
    extract_rules_features_from_binary_dt,
    extract_rules_features_from_reg_dt,
    aggregate_scores,
    set_overlap_metrics,
)
from utils.probe_utils import (
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

# %% Binary dt
binary_dt_name = 'neuron_decision_trees/decision_trees_d8/decision_trees_mlp_neuron_6000.pkl'
with open(binary_dt_name, "rb") as f:
    binary_decision_trees = pickle.load(f)

binary_custom_function_name = list(binary_decision_trees[0].keys())[0]
n_binary_features = binary_decision_trees[0][binary_custom_function_name]["binary_decision_tree"]["model"].n_features_in_
binary_feature_names = create_feature_names(n_binary_features, binary_custom_function_name)

f1_threshold = 0.7
binary_dt_rules = extract_rules_features_from_binary_dt(
    num_layers = n_layers,
    num_neurons = n_neurons,
    binary_decision_trees = binary_decision_trees,
    custom_function_name = "games_batch_to_board_state_flipped_played_BLC",
    binary_feature_names = binary_feature_names,
    f1_threshold=f1_threshold,
)
binary_dt_f1 = aggregate_scores(binary_dt_rules, score_key="dt_f1")

# %% reg dt
reg_dt_name = 'neuron_decision_trees/decision_trees_0826_features/decision_trees_mlp_neuron_6000.pkl'
with open(reg_dt_name, "rb") as f:
    reg_decision_trees = pickle.load(f)

reg_custom_function_name = list(reg_decision_trees[0].keys())[0]
n_reg_features = reg_decision_trees[0][reg_custom_function_name]["decision_tree"]["model"].n_features_in_
reg_feature_names = create_feature_names(n_reg_features, reg_custom_function_name)

r2_threshold = 0.7
reg_dt_rules = extract_rules_features_from_reg_dt(
    num_layers = n_layers,
    num_neurons = n_neurons,
    reg_decision_trees = reg_decision_trees,
    custom_function_name = "games_batch_to_board_state_flipped_played_BLC",
    reg_feature_names = reg_feature_names,
    r2_threshold=r2_threshold,
)
reg_dt_r2 = aggregate_scores(reg_dt_rules, score_key="dt_r2")

# %% probe feature extraction
probes = load_probes_and_normalize(n_layers, device)

# layer = 7
# neuron = 255
probe_features = defaultdict(dict)
for layer in range(n_layers):
    for neuron in range(n_neurons):
        matrices = calculate_w_in_cossim_with_probes(
            model,
            probes,
            layer,
            neuron,
            layer_offset=0,
        )

        filtered_feature_names, directional_feature_names = extract_probe_features(matrices, k=2)
        probe_features[layer][neuron] = {
            "filtered_feature_names": filtered_feature_names,
            "directional_feature_names": directional_feature_names,
        }

# %% ripper load
with open(f"ripper_all_neurons_analysis.pkl", "rb") as f:
    ripper_all_neurons_analysis = pickle.load(f)

ripper_f1 = aggregate_scores(ripper_all_neurons_analysis, score_key="f1_score")

ripper_features = defaultdict(dict)
for layer in ripper_all_neurons_analysis:
    for neuron, info in enumerate(ripper_all_neurons_analysis[layer]):
        features = info["feature_weights"].keys()
        feature_names = set()
        directional_feature_names = set()
        for feat_name, feat_score in info["feature_weights"].items():
            feature_names.update({f"{feat_name}"})
            if feat_score > 0:
                directional_feature_names.update({f"({feat_name})"})
            else:
                directional_feature_names.update({f"(NOT {feat_name})"})
        ripper_features[layer][neuron] = {
            "feature_names": feature_names,
            "directional_feature_names": directional_feature_names,
        }

# %% lasso load
lasso_results = dict()
for layer in range(n_layers):
    with open(f"lasso_results/layer{layer}_results.pkl", "rb") as f:
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

# %% ----- contrastive scores aggregation ----- %% #
binary_dt_vs_probe_contrastive = defaultdict(dict)
reg_dt_vs_probe_contrastive = defaultdict(dict)
ripper_vs_probe_contrastive = defaultdict(dict)
lasso_vs_probe_contrastive = defaultdict(dict)
for layer in range(n_layers):
    for neuron in range(n_neurons):
        probe_feat = probe_features[layer][neuron]["directional_feature_names"]
        
        binary_dt_feat = binary_dt_rules[layer][neuron]["dt_filtered_directional_features"]
        reg_dt_feat = reg_dt_rules[layer][neuron]["dt_filtered_directional_features"]
        
        ripper_feat = ripper_features[layer][neuron]["directional_feature_names"]
        lasso_feat = lasso_features[layer][neuron]["directional_feature_names"]

        metrics_binary_dt_probe = set_overlap_metrics(binary_dt_feat, probe_feat)
        metrics_reg_dt_probe = set_overlap_metrics(reg_dt_feat, probe_feat)
        metrics_ripper_probe = set_overlap_metrics(ripper_feat, probe_feat)
        metrics_lasso_probe = set_overlap_metrics(lasso_feat, probe_feat) 
    
        binary_dt_vs_probe_contrastive[layer][neuron] = metrics_binary_dt_probe
        reg_dt_vs_probe_contrastive[layer][neuron] = metrics_reg_dt_probe
        ripper_vs_probe_contrastive[layer][neuron] = metrics_ripper_probe
        lasso_vs_probe_contrastive[layer][neuron] = metrics_lasso_probe

# %%
x = np.arange(n_layers)
width = 0.35

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])  # make bottom row a bit taller

ax_r2 = fig.add_subplot(gs[0, 0])
ax_f1 = fig.add_subplot(gs[0, 1])
ax_jac = fig.add_subplot(gs[1, :])  # span both columns


# Left: R²
# ax_r2.bar(x - width/2, reg_dt_r2, width, label='Regression DT R²')
# ax_r2.bar(x + width/2, lasso_r2, width, label='Regression lasso R²')
ax_r2.boxplot([reg_dt_r2.get(l, []) for l in range(n_layers)], positions=x - width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="skyblue"), label='Regression DT R²')
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
ax_f1.boxplot([binary_dt_f1.get(l, []) for l in range(n_layers)], positions=x - width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="lightgreen"), label='Binary DT F1')
ax_f1.boxplot([ripper_f1.get(l, []) for l in range(n_layers)], positions=x + width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="salmon"), label='RIPPER F1')
ax_f1.set_xticks(x)
ax_f1.set_xticklabels([f"layer {layer}" for layer in range(n_layers)], rotation=45)
ax_f1.set_ylabel("F1 score")
# ax_f1.set_ylim(0, 1)
ax_f1.set_title("F1 across neurons per Layer")
# ax_f1.legend()

ax_jac.boxplot(
    [
        [reg_dt_vs_probe_contrastive[layer][neuron]['set2_in_set1'] for neuron in range(n_neurons)]
        for layer in range(n_layers)
    ],
    positions=x - 3* width / 4,
    widths=0.15,
    patch_artist=True,
    boxprops=dict(facecolor="skyblue"),
    # label='Jaccard score (Regression DT vs Probe)',
    label = "Regression DT features in Probe features"
)
ax_jac.boxplot(
    [
        [lasso_vs_probe_contrastive[layer][neuron]['set2_in_set1'] for neuron in range(n_neurons)]
        for layer in range(n_layers)
    ],
    positions=x - width / 4,
    widths=0.15,
    patch_artist=True,
    boxprops=dict(facecolor="orange"),
    # label='Jaccard score (Lasso vs Probe)',
    label = "Lasso features in Probe features"
)
ax_jac.boxplot(
    [
        [binary_dt_vs_probe_contrastive[layer][neuron]['set2_in_set1'] for neuron in range(n_neurons)]
        for layer in range(n_layers)
    ],
    positions=x + width / 4,
    widths=0.15,
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen"),
    # label='Jaccard score (Binary DT vs Probe)',
    label = "Binary DT features in Probe features"
)
ax_jac.boxplot(
    [
        [ripper_vs_probe_contrastive[layer][neuron]['set2_in_set1'] for neuron in range(n_neurons)]
        for layer in range(n_layers)
    ],
    positions=x + 3 * width / 4,
    widths=0.15,
    patch_artist=True,
    boxprops=dict(facecolor="salmon"),
    # label='Jaccard score (RIPPER vs Probe)',
    label = "RIPPER features in Probe features"
)
ax_jac.set_xticks(x)
ax_jac.set_xticklabels([f"layer {layer}" for layer in range(n_layers)], rotation=45)
ax_jac.set_ylabel("Jaccard index")
# ax_jac.set_ylim(0, 1)
ax_jac.set_title("Contrastive metric across neurons per Layer")
ax_jac.legend(loc='upper right', bbox_to_anchor=(1, 1.3))
plt.tight_layout()
# plt.show()
plt.savefig("figures/contrastive_analysis/contrastive_analysis_all_methods.png", dpi=300, bbox_inches='tight')

# %%
