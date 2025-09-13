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
import gzip
import sys
from dataclasses import dataclass
from skimage.filters import threshold_otsu

from sklearn.tree import DecisionTreeRegressor
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
    get_feature_names_cont_dt,
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
def extract_and_rules_cont(
    tree,
    feature_names,
    on_off_threshold=1,
    min_samples=None,
    # operate_thresholds = {"<=": 1, ">":-1}, 
):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    pred_act = []
    samples_per_rule = []
    features_per_rule = []
    used_features = set()
    
    def recurse(node, conditions, features_in_path):
        # if (tree_.feature[node] != _tree.TREE_UNDEFINED) and (tree_.n_node_samples[node] > min_samples):
        recurse_condition = (tree_.feature[node] != _tree.TREE_UNDEFINED)
        if min_samples is not None:
            recurse_condition = recurse_condition and (tree_.n_node_samples[node] > min_samples)
        # if value_threshold is not None:
        #     values = tree_.value[node][0][0]
        #     recurse_condition = recurse_condition and (values[target_class].item() < value_threshold)

        if recurse_condition:  # not a leaf
            name = feature_name[node]
            # threshold = tree_.threshold[node]
            
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
            values = tree_.value[node][0][0].item()
            if values > on_off_threshold:
                rule = " AND ".join(conditions)
                rules.append(rule)
                pred_act.append(values)
                samples_per_rule.append(tree_.n_node_samples[node].item())
                features_per_rule.append(features_in_path)
                used_features.update(features_in_path)
    
    recurse(0, [], set())
    return (rules, pred_act, samples_per_rule, features_per_rule), used_features

def extract_probe_features_cont(matrices, k=2):
    matrices_mean = matrices.mean().item()
    matrices_std = matrices.std().item()

    filtered_feature_names = []
    directional_feature_names = []
    for row in range(8):
        for col in range(8):
            square = chr(ord('A') + row) + str(col)
            blank_weight = matrices[0, row, col].item()
            my_weight = matrices[1, row, col].item()
            flipped_weight = matrices[2, row, col].item()
            just_played_weight = matrices[3, row, col].item()

            # mine_weight = matrices[0, row, col].item()
            # empty_weight = matrices[1, row, col].item()
            # theirs_weight = matrices[2, row, col].item()
            # flipped_weight = matrices[3, row, col].item()
            # just_played_weight = matrices[4, row, col].item()

            occupied = 0
            if blank_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_blank")
                directional_feature_names.append(f"({square}_blank)")
                # occupied = 1

            if blank_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_blank")
                directional_feature_names.append(f"(NOT {square}_blank)")
                # occupied = 1

            if my_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_mine-theirs")
                directional_feature_names.append(f"({square}_mine-theirs)")
                # occupied = 1
            
            if my_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_mine-theirs")
                directional_feature_names.append(f"(NOT {square}_mine-theirs)")
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
def infer_positive_from_negations(features, group, values):
    """
    If all but one value in a group are negated, infer the positive for the last one.
    """
    feats = set(features)
    negs = {v for v in values if f"(NOT {group}_{v})" in feats}
    remaining = set(values) - negs

    if len(remaining) == 1 and len(negs) == len(values) - 1:
        # Replace negations with the positive
        inferred = f"({group}_{remaining.pop()})"
        feats = (feats - {f"(NOT {group}_{v})" for v in negs}) | {inferred}

    return feats

def remove_negations_if_positive_present(features, group, values):
    """
    If a positive feature is present, remove all negations of the same group.
    """
    feats = set(features)
    positives = [f"({group}_{v})" for v in values if f"({group}_{v})" in feats]
    
    if positives:
        # Remove all negations if any positive is present
        feats = feats - {f"(NOT {group}_{v})" for v in values}
    
    return feats

def extract_squares(directional_features):
    squares = set()
    for feat in directional_features:
        squares.add(feat.split("(")[-1].split(" ")[-1].split("_")[0])
    return squares

def direct_feature_infer(directional_features):
    squares = extract_squares(directional_features)
    for square in squares:
        directional_features = infer_positive_from_negations(directional_features, square, ["mine", "theirs", "empty"])
        directional_features = remove_negations_if_positive_present(directional_features, square, ["mine", "theirs", "empty"])
    # filtered = set()
    # for feat in features:
    #     if feat.startswith("NOT "):
    #         continue
    #     if feat.endswith("_flipped") or feat.endswith("_just_played"):
    #         continue
    #     filtered.add(feat)
    return directional_features

def rule_infer(rule):
    directional_features = set(rule.split(" AND "))
    direct_feat_inferred = direct_feature_infer(directional_features)
    rule_inferred = " AND ".join(sorted(direct_feat_inferred))
    return rule_inferred

# %%
# Load decision trees
# dt_name = 'neuron_simulation/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl'
# with open(dt_name, "rb") as f:
#     decision_trees = pickle.load(f)

# function_name = list(decision_trees[0].keys())[0]
# n_features = decision_trees[0][function_name]["decision_tree"]["model"].n_features_in_
# feature_names = create_feature_names(n_features, function_name)

# %%
@dataclass
class DecisionTreeResults:
    """Results for a single square's decision tree."""
    layer: int
    neuron: int
    tree: DecisionTreeRegressor
    train_R2: float
    train_MSE: float
    test_R2: float
    test_MSE: float

sys.modules['__main__'].DecisionTreeResults = DecisionTreeResults

cont_dts = {}
for layer in range(1, n_layers):
    binary_dt_name = f'cont_dt/layer_{layer}_trees.pkl.gz'
    with gzip.open(binary_dt_name, 'rb') as f:
        trees = pickle.load(f)
    
    cont_dts[layer] = trees

cont_dt_features = get_feature_names_cont_dt()
# with open(binary_dt_name, "rb") as f:
#     binary_decision_trees = pickle.load(f)

# binary_function_name = list(binary_decision_trees[0].keys())[0]
# n_binary_features = binary_decision_trees[0][binary_function_name]["binary_decision_tree"]["model"].n_features_in_
# binary_feature_names = create_feature_names(n_binary_features, binary_function_name)

# %%
probe_dict = {i : t.load(
    f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
)['linear_probe'].squeeze() for i in range(n_layers)}

probe_t = t.stack([probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]
blank_probe = probe_t[..., 1] - (probe_t[..., 0] + probe_t[..., 2]) * 0.5  # [layer, d_model, row, col]
my_probe = probe_t[..., 0] - probe_t[..., 2]  # [layer, d_model, row, col]

blank_probe_normalized = blank_probe / blank_probe.norm(dim=1, keepdim=True)
my_probe_normalized = my_probe / my_probe.norm(dim=1, keepdim=True)
blank_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# %%
flipped_probe_dict = {i : t.load(
    f"linear_probes_flipped/resid_{i}_flipped.pth", map_location=str(device), weights_only="True"
).squeeze() for i in range(n_layers)}

flipped_probe_t = t.stack([flipped_probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]

flipped_probe = flipped_probe_t[..., 0] - flipped_probe_t[..., 1]  # [layer, d_model, row, col]
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
r2_threshold = 0.7
features_dict = defaultdict(dict)
for layer in trange(1, n_layers):
    for neuron in range(n_neurons):
        cont_dt = cont_dts[layer][neuron].tree
        r2 = cont_dts[layer][neuron].test_R2
        # binary_tree_model = binary_decision_trees[layer][binary_function_name]['binary_decision_tree']['model'].estimators_[neuron]
        # f1 = binary_decision_trees[layer][binary_function_name]['binary_decision_tree']['f1'][neuron].item()

        if r2 < r2_threshold:
            continue

        leaf_mask = cont_dt.tree_.children_right == -1
        leaf_values = cont_dt.tree_.value[leaf_mask].flatten()
        on_off_threshold = threshold_otsu(leaf_values)

        (rules, pred_act, samples_per_rule, features_per_rule), used_features = extract_and_rules_cont(cont_dt, cont_dt_features, on_off_threshold=on_off_threshold)

        sorted_rules = sorted(
            zip(rules, pred_act, samples_per_rule, features_per_rule),
            key=lambda x: (x[2], x[1]),  # sort by samples_per_rule, then pred_strength
            reverse=True
        )

        filter_min_samples = cont_dt.tree_.n_node_samples[0].item() / 59 * .05
        filtered_rules = [(rule_infer(rule), strength, samples) for rule, strength, samples, features in sorted_rules if samples >= filter_min_samples]

        filtered_features = set()
        filtered_direct_features = set()
        for rule, strength, samples in filtered_rules:
            # print(f"Rule: {rule}\n\t(Strength: {strength:.2f}, Samples: {samples})")
            direct_feat_infered = set(rule.split(" AND "))
            feature_inferred = {feat.split("(")[-1].split(")")[0].split(" ")[-1] for feat in direct_feat_infered}
            
            filtered_features.update(feature_inferred)
            filtered_direct_features.update(direct_feat_infered)

        w_in_LN_blank = calculate_neuron_input_weights(model, blank_probe_normalized[layer-1], layer, neuron)
        w_in_LN_my = calculate_neuron_input_weights(model, my_probe_normalized[layer-1], layer, neuron)
        w_in_LN_flipped = calculate_neuron_input_weights(model, flipped_probe_normalized[layer-1], layer, neuron)
        w_in_LN_just_played = calculate_neuron_input_weights(model, just_played_probe_normalized[layer-1], layer, neuron)

        matrices = t.stack(
            [w_in_LN_blank, w_in_LN_my, w_in_LN_flipped, w_in_LN_just_played], dim=0
        )  # [4, 8, 8]

        filtered_feature_names, directional_feature_names = extract_probe_features_cont(matrices, k=2)
        features_dict[layer][neuron] = {
            "dt_rules": filtered_rules,
            "dt_used_features": used_features,
            "dt_filtered_features": filtered_features,
            "dt_filtered_directional_features": filtered_direct_features,
            "probe_directional_features": directional_feature_names,
            "probe_filtered_feature_names": filtered_feature_names,
            "probe_directional_features_inferred": direct_feature_infer(directional_feature_names.copy()),
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
        # dt_filtered_features = features_dict[layer][neuron]["dt_filtered_features"]
        dt_filtered_features = features_dict[layer][neuron]["dt_filtered_directional_features"]

        # probe_filtered_feature_names = features_dict[layer][neuron]["probe_filtered_feature_names"]
        probe_filtered_feature_names = features_dict[layer][neuron]["probe_directional_features"]
        # probe_filtered_feature_names = features_dict[layer][neuron]["probe_directional_features_inferred"]

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
    [features_dict_metrics[l][n]["metrics_dt_probe_jaccard"] for n in range(n_neurons) if n in features_dict_metrics[l]]
    for l in range(1, n_layers) 
]
# overlap_vals = [
#     [features_dict_metrics[l][n]["metrics_dt_probe_overlap"] for n in range(n_neurons)]
#     for l in range(1, n_layers) 
# ]
intersection_over_probe_vals = [
    [features_dict_metrics[l][n]["metrics_dt_probe_intersection_over_probe"] for n in range(n_neurons) if n in features_dict_metrics[l]]
    for l in range(1, n_layers) 
]

intersectoin_over_dt_vals = [
    [features_dict_metrics[l][n]["metrics_dt_probe_intersection_over_dt"] for n in range(n_neurons) if n in features_dict_metrics[l]]
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
