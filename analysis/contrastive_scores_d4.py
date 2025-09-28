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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.circuits_utils as circuits_utils
import utils.arena_utils as arena_utils
from utils.feature_extraction_utils import (
    create_bs_flipped_played_feature_names,
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
from decision_trees.dtypes import DecisionTreeResults, BinaryDecisionTreeResults
from decision_trees import dtypes
# import decision_trees

sys.modules['dtypes'] = dtypes
# sys.modules['ground_truth_dt'] = decision_trees

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
BASE_PATH = Path("/Users/srujanamedicherla/Desktop/Algoverse_project/OthelloReverseEngineering")
# os.chdir(BASE_PATH)

# device = "cuda" if t.cuda.is_available() else "cpu"
device = "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = circuits_utils.get_model(model_name, device)

n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp

# %% Binary dt (d8)
# binary_dt_name = 'neuron_decision_trees/decision_trees_d8/decision_trees_mlp_neuron_6000.pkl'

# with open(binary_dt_name, "rb") as f:
#     binary_decision_trees = pickle.load(f)

# binary_custom_function_name = list(binary_decision_trees[0].keys())[0]
# n_binary_features = binary_decision_trees[0][binary_custom_function_name]["binary_decision_tree"]["model"].n_features_in_
# binary_feature_names = create_feature_names(n_binary_features, binary_custom_function_name)

# binary_decision_tree_dict = defaultdict(dict)
# binary_dt_f1 = defaultdict(dict)
# for layer in range(n_layers):
#     for neuron in range(n_neurons):
#         binary_tree_model = binary_decision_trees[layer][binary_custom_function_name]['binary_decision_tree']['model'].estimators_[neuron]
#         f1 = binary_decision_trees[layer][binary_custom_function_name]['binary_decision_tree']['f1'][neuron].item()
#         binary_decision_tree_dict[layer][neuron] = binary_tree_model
#         binary_dt_f1[layer][neuron] = f1

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

binary_feature_names = create_bs_flipped_played_feature_names(320)

f1_threshold = 0.7
binary_dt_rules = extract_rules_features_from_binary_dt(
    num_layers = n_layers,
    num_neurons = n_neurons,
    binary_decision_trees = binary_decision_tree_dict,
    f1_scores = binary_dt_f1,
    binary_feature_names = binary_feature_names,
    f1_threshold=f1_threshold,
)
binary_dt_f1_filter = {layer: [score for score in scores.values() if score >=0] for layer, scores in binary_dt_f1.items()}

# %% reg dt (d8)
# reg_dt_name = 'neuron_decision_trees/decision_trees_0826_features/decision_trees_mlp_neuron_6000.pkl'

# with open(reg_dt_name, "rb") as f:
#     reg_decision_trees = pickle.load(f)

# reg_custom_function_name = list(reg_decision_trees[0].keys())[0]
# n_reg_features = reg_decision_trees[0][reg_custom_function_name]["decision_tree"]["model"].n_features_in_
# reg_feature_names = create_feature_names(n_reg_features, reg_custom_function_name)

# reg_decision_tree_dict = defaultdict(dict)
# reg_dt_r2 = defaultdict(dict)

# for layer in range(n_layers):
#     for neuron in range(n_neurons):
#         reg_tree_model = reg_decision_trees[layer][reg_custom_function_name]['decision_tree']['model'].estimators_[neuron]
#         r2 = reg_decision_trees[layer][reg_custom_function_name]['decision_tree']['r2'][neuron].item()
#         reg_decision_tree_dict[layer][neuron] = reg_tree_model
#         reg_dt_r2[layer][neuron] = r2

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
            print(f"Tree L{layer}N{gt_binary.neuron} is NOT fitted")
            continue

reg_feature_names = create_bs_flipped_played_feature_names(320)

r2_threshold = 0.7
reg_dt_rules = extract_rules_features_from_reg_dt(
    num_layers = n_layers,
    num_neurons = n_neurons,
    reg_decision_trees = reg_decision_tree_dict,
    r2_scores = reg_dt_r2,
    reg_feature_names = reg_feature_names,
    r2_threshold=r2_threshold,
)
reg_dt_r2_filter = {layer: [score for score in scores.values() if score >=0] for layer, scores in reg_dt_r2.items()}

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
with open(f"ripper/ripper_all_neurons_analysis.pkl", "rb") as f:
    ripper_all_neurons_analysis = pickle.load(f)

ripper_f1 = aggregate_scores(ripper_all_neurons_analysis, score_key="f1_score")

ripper_features = defaultdict(dict)
for layer in ripper_all_neurons_analysis:
    for neuron_index, info in enumerate(ripper_all_neurons_analysis[layer]):
       
        neuron_id = info["neuron_id"]
        top_features = info["top_features"]
        feature_names = set()
        directional_feature_names = set()
        for feat_name, feat_score in top_features:
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

# %% ----- contrastive scores aggregation ----- %% #
binary_dt_vs_probe_contrastive = defaultdict(dict)
reg_dt_vs_probe_contrastive = defaultdict(dict)
ripper_vs_probe_contrastive = defaultdict(dict)
lasso_vs_probe_contrastive = defaultdict(dict)
for layer in range(n_layers):
    for neuron in range(n_neurons):
        probe_feat = probe_features[layer][neuron]["directional_feature_names"]

        try:
            binary_dt_feat = binary_dt_rules[layer][neuron]["dt_filtered_directional_features"]
            metrics_binary_dt_probe = set_overlap_metrics(binary_dt_feat, probe_feat)
            binary_dt_vs_probe_contrastive[layer][neuron] = metrics_binary_dt_probe
        except:
            pass

        reg_dt_feat = reg_dt_rules[layer][neuron]["dt_filtered_directional_features"]
        metrics_reg_dt_probe = set_overlap_metrics(reg_dt_feat, probe_feat)
        reg_dt_vs_probe_contrastive[layer][neuron] = metrics_reg_dt_probe

        ripper_feat = ripper_features[layer][neuron]["directional_feature_names"]
        metrics_ripper_probe = set_overlap_metrics(ripper_feat, probe_feat)
        ripper_vs_probe_contrastive[layer][neuron] = metrics_ripper_probe

        lasso_feat = lasso_features[layer][neuron]["directional_feature_names"]
        metrics_lasso_probe = set_overlap_metrics(lasso_feat, probe_feat) 
        lasso_vs_probe_contrastive[layer][neuron] = metrics_lasso_probe

# %%
jac_metric = "set2_in_set1"
jac_ylabel = "Score"
jac_title = "Containment of model feature in Probe features across neurons per Layer"
jac_output = f"contrastive_analysis_all_methods_containment.pdf"

jac_metric = "jaccard_index"
jac_ylabel = "Jaccard index"
jac_title = "jaccard index of model feature v.s. Probe features across neurons per Layer"
jac_output = f"contrastive_analysis_all_methods_jaccard.pdf"

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
ax_r2.boxplot([reg_dt_r2_filter.get(l, []) for l in range(n_layers)], positions=x - width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="skyblue"))
ax_r2.boxplot([lasso_r2_filter.get(l, []) for l in range(n_layers)], positions=x + width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="orange"))
ax_r2.set_xticks(x)
ax_r2.set_xticklabels([f"layer {layer}" for layer in range(n_layers)], rotation=45)
ax_r2.set_ylabel("R² score")
# ax_r2.set_ylim(0, 1)
ax_r2.set_title("R² across neurons per Layer")
# ax_r2.legend()

# Right: F1
# ax_f1.bar(x - width/2, binary_dt_f1, width, label='Binary DT F1')
# ax_f1.bar(x + width/2, ripper_f1, width, label='RIPPER F1')
ax_f1.boxplot([binary_dt_f1_filter.get(l, []) for l in range(n_layers)], positions=x - width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
ax_f1.boxplot([ripper_f1_filter.get(l, []) for l in range(n_layers)], positions=x + width/2, widths=0.3, patch_artist=True, boxprops=dict(facecolor="salmon"))
ax_f1.set_xticks(x)
ax_f1.set_xticklabels([f"layer {layer}" for layer in range(n_layers)], rotation=45)
ax_f1.set_ylabel("F1 score")
# ax_f1.set_ylim(0, 1)
ax_f1.set_title("F1 across neurons per Layer")
# ax_f1.legend()

ax_jac.boxplot(
    [
        [info[jac_metric] for _, info in reg_dt_vs_probe_contrastive[layer].items()]
        for layer in range(n_layers)
    ],
    positions=x - 3* width / 4,
    widths=0.15,
    patch_artist=True,
    boxprops=dict(facecolor="skyblue"),
    # label='Jaccard score (Regression DT vs Probe)',
    #label = "Regression DT features"
)
ax_jac.boxplot(
    [
        [info[jac_metric] for _, info in lasso_vs_probe_contrastive[layer].items()]
        for layer in range(n_layers)
    ],
    positions=x - width / 4,
    widths=0.15,
    patch_artist=True,
    boxprops=dict(facecolor="orange"),
    # label='Jaccard score (Lasso vs Probe)',
    #label = "Lasso features"
)
ax_jac.boxplot(
    [
        [info[jac_metric] for _, info in binary_dt_vs_probe_contrastive[layer].items()]
        for layer in range(n_layers)
    ],
    positions=x + width / 4,
    widths=0.15,
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen"),
    # label='Jaccard score (Binary DT vs Probe)',
    #label = "Binary DT features"
)
ax_jac.boxplot(
    [
        [info[jac_metric] for _, info in ripper_vs_probe_contrastive[layer].items()]
        for layer in range(n_layers)
    ],
    positions=x + 3 * width / 4,
    widths=0.15,
    patch_artist=True,
    boxprops=dict(facecolor="salmon"),
    # label='Jaccard score (RIPPER vs Probe)',
    #label = "RIPPER features"
)
ax_jac.set_xticks(x)
ax_jac.set_xticklabels([f"layer {layer}" for layer in range(n_layers)], rotation=45)
ax_jac.set_ylabel(jac_ylabel)
# ax_jac.set_ylim(0, 1)
ax_jac.set_title(jac_title)
from matplotlib.patches import Patch
# Add legend for contrastive analysis plot
legend_elements_jac = [Patch(facecolor='skyblue', label='Regression DT features'),
                       Patch(facecolor='orange', label='Lasso features'),
                       Patch(facecolor='lightgreen', label='Binary DT features'),
                       Patch(facecolor='salmon', label='RIPPER features')]
ax_jac.legend(handles=legend_elements_jac, loc='upper right', bbox_to_anchor=(1, 1.3))
plt.tight_layout()
# plt.show()
plt.savefig(f"figures/contrastive_analysis/{jac_output}", dpi=300, bbox_inches='tight')

# %%
