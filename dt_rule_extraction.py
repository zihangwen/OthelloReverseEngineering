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
from tqdm import trange, tqdm

from sklearn.tree import export_graphviz
import graphviz

# from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.tree import plot_tree
from sklearn.tree import _tree

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

device = "cuda" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = utils.get_model(model_name, device)

n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp

# %%
def extract_and_rules(tree, feature_names, target_class=1, min_samples=None, value_threshold=None):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    pred_strengths = []
    samples_per_rule = []
    features_per_rule = []
    used_features = set()
    
    def recurse(node, conditions, features_in_path):
        # if (tree_.feature[node] != _tree.TREE_UNDEFINED) and (tree_.n_node_samples[node] > min_samples):
        recurse_condition = (tree_.feature[node] != _tree.TREE_UNDEFINED)
        if min_samples is not None:
            recurse_condition = recurse_condition and (tree_.n_node_samples[node] > min_samples)
        if value_threshold is not None:
            values = tree_.value[node][0]
            recurse_condition = recurse_condition and (values[target_class].item() < value_threshold)

        if recurse_condition:  # not a leaf
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # left child (feature <= threshold)
            # recurse(tree_.children_left[node],
            #         conditions + [f"({name} <= {threshold:.4f})"],
            #         features_in_path | {name})
            recurse(tree_.children_left[node],
                    conditions + [f"(NOT {name})"],
                    features_in_path | {name})
            
            # right child (feature > threshold)
            # recurse(tree_.children_right[node],
            #         conditions + [f"({name} > {threshold:.4f})"],
            #         features_in_path | {name})
            recurse(tree_.children_right[node],
                    conditions + [f"({name})"],
                    features_in_path | {name})
        else:
            # Leaf node: check predicted class
            values = tree_.value[node][0]
            pred_class = values.argmax()
            if pred_class == target_class:
                rule = " AND ".join(conditions)
                rules.append(rule)
                pred_strengths.append(values[pred_class].item() / values.sum().item())
                samples_per_rule.append(tree_.n_node_samples[node].item())
                features_per_rule.append(features_in_path)
                used_features.update(features_in_path)
    
    recurse(0, [], set())
    return (rules, pred_strengths, samples_per_rule, features_per_rule), used_features

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
# Load decision trees
# dt_name = 'neuron_simulation/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl'
# with open(dt_name, "rb") as f:
#     decision_trees = pickle.load(f)

# function_name = list(decision_trees[0].keys())[0]
# n_features = decision_trees[0][function_name]["decision_tree"]["model"].n_features_in_
# feature_names = create_feature_names(n_features, function_name)

# %%
binary_dt_name = 'neuron_decision_trees/decision_trees_d8/decision_trees_mlp_neuron_30000.pkl'
with open(binary_dt_name, "rb") as f:
    binary_decision_trees = pickle.load(f)

binary_function_name = list(binary_decision_trees[0].keys())[0]
n_binary_features = binary_decision_trees[0][binary_function_name]["binary_decision_tree"]["model"].n_features_in_
binary_feature_names = create_feature_names(n_binary_features, binary_function_name)

# # %%
# layer = 5
# neuron = 766

# binary_tree_model = binary_decision_trees[layer][binary_function_name]['binary_decision_tree']['model'].estimators_[neuron]

# # %%
# (rules, pred_strengths, samples_per_rule, features_per_rule), used_features = extract_and_rules(binary_tree_model, binary_feature_names, target_class=1, value_threshold=0.7)

# # %%
# sorted_rules = sorted(
#     zip(rules, pred_strengths, samples_per_rule, features_per_rule),
#     key=lambda x: (x[2], x[1]),  # sort by samples_per_rule, then pred_strength
#     reverse=True
# )

# filter_min_samples = binary_tree_model.tree_.n_node_samples[0].item() / 59 * .05
# filtered_rules = [(rule, strength, samples, features) for rule, strength, samples, features in sorted_rules if samples >= filter_min_samples]

# filtered_features = set()
# for rule, strength, samples, features in filtered_rules:
#     print(f"Rule: {rule}\n\t(Strength: {strength:.2f}, Samples: {samples})")
#     filtered_features.update(features)

# %%
# plot_tree(
#     binary_tree_model,
#     feature_names=binary_feature_names,
#     filled=True,
#     rounded=True,
#     fontsize=8,
#     max_depth=3
# )
# plt.show()

# %%
# dot_data = export_graphviz(
#     binary_tree_model,
#     out_file=None,
#     feature_names=binary_feature_names,
#     filled=True, rounded=True,
#     special_characters=True,
#     proportion=True,   # scale node size by samples
#     max_depth=3,
# )
# graph = graphviz.Source(dot_data)

# # graph.render("regression_tree")  # saves PDF/PNG
# graph
# graph.render(f"figures/dt_metrics/L{layer}N{neuron}_tree", format="png", cleanup=True)

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

# layer = 5
# neuron = 766

# w_in_LN_mine = calculate_neuron_input_weights(model, mine_probe_normalized[layer-1], layer, neuron)
# w_in_LN_empty = calculate_neuron_input_weights(model, empty_probe_normalized[layer-1], layer, neuron)
# w_in_LN_theirs = calculate_neuron_input_weights(model, theirs_probe_normalized[layer-1], layer, neuron)
# w_in_LN_flipped = calculate_neuron_input_weights(model, flipped_probe_normalized[layer-1], layer, neuron)
# w_in_LN_just_played = calculate_neuron_input_weights(model, just_played_probe_normalized[layer-1], layer, neuron)

# matrices = t.stack(
#     [w_in_LN_mine, w_in_LN_empty, w_in_LN_theirs, w_in_LN_flipped, w_in_LN_just_played], dim=0
# )  # [4, 8, 8]

# %% plotting examples
# titles = [
#     f"Mine In L{layer}N{neuron}", f"Empty In L{layer}N{neuron}", f"Theirs In L{layer}N{neuron}",
#     f"Flipped In L{layer}N{neuron}", f"Just Played In L{layer}N{neuron}",
# ]

# fig = arena_utils.plot_board_values(
#     matrices,
#     title=f"Input weights cosine similarity with the probe for neuron L{layer}N{neuron}",
#     board_titles=titles,
#     boards_per_row=3,
#     width=975,
#     height=760,
# )

# fig.write_image(f"figures/week8-2/neuron_input_weights_L{layer}N{neuron}.png")

# k = 2
# matrices_mean = matrices.mean().item()
# matrices_std = matrices.std().item()

# fig, ax = plt.subplots(figsize=(8, 6))
# plt.hist(matrices.flatten().cpu(), bins=50, alpha=0.5)

# plt.axvline(matrices_mean + k*matrices_std, color='blue', linestyle='dashed', linewidth=1, label=f'Mean + {k}*STD')
# plt.axvline(matrices_mean - k*matrices_std, color='orange', linestyle='dashed', linewidth=1, label=f'Mean - {k}*STD')
# plt.legend()
# plt.title(f"Histogram of cosine similarity for neuron L{layer}N{neuron}")
# plt.xlabel("Weight value")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.savefig(f"figures/week8-2/L{layer}N{neuron}_input_weights_hist.png", dpi=300, bbox_inches='tight')
# plt.plot()


# %%
# filtered_feature_names, directional_feature_names = extract_probe_features(matrices, k=2)
# print("Filtered feature names:")
# for name in filtered_feature_names:
#     print(name)
# print("\nDirectional feature names:")
# for name in directional_feature_names:
#     print(name)


# %%
# layer = 5
# neuron = 766

features_dict = defaultdict(dict)
for layer in trange(1, n_layers):
    for neuron in range(n_neurons):
        binary_tree_model = binary_decision_trees[layer][binary_function_name]['binary_decision_tree']['model'].estimators_[neuron]

        (rules, pred_strengths, samples_per_rule, features_per_rule), used_features = extract_and_rules(binary_tree_model, binary_feature_names, target_class=1, value_threshold=0.7)

        sorted_rules = sorted(
            zip(rules, pred_strengths, samples_per_rule, features_per_rule),
            key=lambda x: (x[2], x[1]),  # sort by samples_per_rule, then pred_strength
            reverse=True
        )

        filter_min_samples = binary_tree_model.tree_.n_node_samples[0].item() / 59 * .05
        filtered_rules = [(rule, strength, samples, features) for rule, strength, samples, features in sorted_rules if samples >= filter_min_samples]

        filtered_features = set()
        for rule, strength, samples, features in filtered_rules:
            # print(f"Rule: {rule}\n\t(Strength: {strength:.2f}, Samples: {samples})")
            filtered_features.update(features)

        w_in_LN_mine = calculate_neuron_input_weights(model, mine_probe_normalized[layer-1], layer, neuron)
        w_in_LN_empty = calculate_neuron_input_weights(model, empty_probe_normalized[layer-1], layer, neuron)
        w_in_LN_theirs = calculate_neuron_input_weights(model, theirs_probe_normalized[layer-1], layer, neuron)
        w_in_LN_flipped = calculate_neuron_input_weights(model, flipped_probe_normalized[layer-1], layer, neuron)
        w_in_LN_just_played = calculate_neuron_input_weights(model, just_played_probe_normalized[layer-1], layer, neuron)

        matrices = t.stack(
            [w_in_LN_mine, w_in_LN_empty, w_in_LN_theirs, w_in_LN_flipped, w_in_LN_just_played], dim=0
        )  # [4, 8, 8]

        filtered_feature_names, directional_feature_names = extract_probe_features(matrices, k=2)
        features_dict[layer][neuron] = {
            "dt_rules": filtered_rules,
            "dt_used_features": used_features,
            "dt_filtered_features": filtered_features,
            "probe_directional_features": directional_feature_names,
            "probe_filtered_feature_names": filtered_feature_names,
        }

# %%
def set_overlap_metrics(set1, set2):
    """
    Compute pure set overlap metrics between two sets of features.

    Parameters:
        set1, set2 : iterable of features (list, set, etc.)

    Returns:
        dict with Jaccard index and Overlap coefficient
    """
    set1 = set(set1)
    set2 = set(set2)
    
    intersection = set1 & set2
    union = set1 | set2
    
    jaccard = len(intersection) / len(union) if union else 1.0
    overlap_coef = len(intersection) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 1.0
    set1_in_set2 = len(intersection) / len(set2) if len(set2) > 0 else 1.0
    set2_in_set1 = len(intersection) / len(set1) if len(set1) > 0 else 1.0
    
    return {
        "jaccard_index": jaccard,
        "overlap_coefficient": overlap_coef,
        "set1_in_set2": set1_in_set2,
        "set2_in_set1": set2_in_set1,
    }

# %%
features_dict_metrics = defaultdict(dict)
for layer in features_dict:
    for neuron in features_dict[layer]:
        # dt_used_features = features_dict[layer][neuron]["dt_used_features"]
        dt_filtered_features = features_dict[layer][neuron]["dt_filtered_features"]
        # probe_directional_features = features_dict[layer][neuron]["probe_directional_features"]
        probe_filtered_feature_names = features_dict[layer][neuron]["probe_filtered_feature_names"]

        metrics_dt_probe = set_overlap_metrics(dt_filtered_features, probe_filtered_feature_names)
        # metrics_probe_dt = set_overlap_metrics(probe_filtered_feature_names, dt_filtered_features)

        features_dict_metrics[layer][neuron] = {
            "metrics_dt_probe_jaccard": metrics_dt_probe["jaccard_index"],
            "metrics_dt_probe_overlap": metrics_dt_probe["overlap_coefficient"],
            "metrics_dt_probe_intersection_over_probe": metrics_dt_probe["set1_in_set2"],
            "metrics_dt_probe_intersection_over_dt": metrics_dt_probe["set2_in_set1"],
        }
        
# %%
jaccard_vals = [
    [features_dict_metrics[l][n]["metrics_dt_probe_jaccard"] for n in range(n_neurons)]
    for l in range(1, n_layers) 
]
# overlap_vals = [
#     [features_dict_metrics[l][n]["metrics_dt_probe_overlap"] for n in range(n_neurons)]
#     for l in range(1, n_layers) 
# ]
intersection_over_probe_vals = [
    [features_dict_metrics[l][n]["metrics_dt_probe_intersection_over_probe"] for n in range(n_neurons)]
    for l in range(1, n_layers) 
]

intersectoin_over_dt_vals = [
    [features_dict_metrics[l][n]["metrics_dt_probe_intersection_over_dt"] for n in range(n_neurons)]
    for l in range(1, n_layers)
]

# plt.figure(figsize=(10,6))
# plt.boxplot(jaccard_vals, positions=[i-0.2 for i in range(1,n_layers)], widths=0.35, patch_artist=True, boxprops=dict(facecolor="skyblue"))
# # plt.boxplot(overlap_vals, positions=[i+0.2 for i in range(1,n_layers)], widths=0.35, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
# plt.boxplot(intersection_over_probe_vals, positions=[i+0.2 for i in range(1,n_layers)], widths=0.35, patch_artist=True, boxprops=dict(facecolor="lightgreen"))

# plt.xticks(range(1,n_layers), [f"Layer {l}" for l in range(1,n_layers)])
# plt.ylabel("Metric Value")
# plt.title("Jaccard vs Overlap per Layer")
# plt.legend(["Jaccard", "dt features in probe features"])
# plt.show()

# %%
layers_list = []
metrics_list = []
values_list = []

for l_idx, l in enumerate(range(1, n_layers)):
    for val in jaccard_vals[l_idx]:
        layers_list.append(f"Layer {l}")
        metrics_list.append("Jaccard")
        values_list.append(val)
    for val in intersection_over_probe_vals[l_idx]:
        layers_list.append(f"Layer {l}")
        metrics_list.append("dt features in probe features")
        values_list.append(val)
    for val in intersectoin_over_dt_vals[l_idx]:
        layers_list.append(f"Layer {l}")
        metrics_list.append("probe features in dt features")
        values_list.append(val)

df = pd.DataFrame({"Layer": layers_list, "Metric": metrics_list, "Value": values_list})

plt.figure(figsize=(10,6))
sns.boxplot(x="Layer", y="Value", hue="Metric", data=df)
plt.title("Feature Overlap Metrics per Layer")
plt.ylabel("Metric Value")
plt.legend(title="Metric", loc='upper right')
plt.show()

# %%
layer_select = 5
neuron_select = 766
print(f"Layer {layer_select} Neuron {neuron_select}:")
print(f"jaccard: {features_dict_metrics[layer_select][neuron_select]['metrics_dt_probe_jaccard']:.2f}")
print(f"dt features in probe features: {features_dict_metrics[layer_select][neuron_select]['metrics_dt_probe_intersection_over_probe']:.2f}")
print(f"probe features in dt features: {features_dict_metrics[layer_select][neuron_select]['metrics_dt_probe_intersection_over_dt']:.2f}")

# %%
