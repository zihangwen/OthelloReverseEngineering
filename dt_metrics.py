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

from sklearn.tree import export_graphviz
import graphviz

# from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
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
def extract_and_rules(tree, feature_names, target_class=1):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    pred_strengths = []
    samples_per_rule = []
    used_features = set()
    
    def recurse(node, conditions, features_in_path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:  # not a leaf
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
                pred_strengths.append(values[pred_class])
                samples_per_rule.append(tree_.n_node_samples[node])
                used_features.update(features_in_path)
    
    recurse(0, [], set())
    return (rules, pred_strengths, samples_per_rule), used_features

# %%
# Load decision trees
# dt_name = 'neuron_simulation/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl'
# with open(dt_name, "rb") as f:
#     decision_trees = pickle.load(f)

# function_name = list(decision_trees[0].keys())[0]
# n_features = decision_trees[0][function_name]["decision_tree"]["model"].n_features_in_
# feature_names = create_feature_names(n_features, function_name)

# %%
binary_dt_name = 'neuron_decision_trees/decision_trees/decision_trees_mlp_neuron_30000.pkl'
with open(binary_dt_name, "rb") as f:
    binary_decision_trees = pickle.load(f)

binary_function_name = list(binary_decision_trees[0].keys())[0]
n_binary_features = binary_decision_trees[0][binary_function_name]["binary_decision_tree"]["model"].n_features_in_
binary_feature_names = create_feature_names(n_binary_features, binary_function_name)

# %%
layer = 5
neuron = 766

binary_tree_model = binary_decision_trees[layer][binary_function_name]['binary_decision_tree']['model'].estimators_[neuron]

# %%
(rules, pred_strengths, samples_per_rule), used_features = extract_and_rules(binary_tree_model, binary_feature_names, target_class=1)

# %%
sorted_rules = sorted(
    zip(rules, pred_strengths, samples_per_rule),
    key=lambda x: (x[2], x[1]),  # sort by samples_per_rule, then pred_strength
    reverse=True
)


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
dot_data = export_graphviz(
    binary_tree_model,
    out_file=None,
    feature_names=binary_feature_names,
    filled=True, rounded=True,
    special_characters=True,
    proportion=True,   # scale node size by samples
    max_depth=3,
)
graph = graphviz.Source(dot_data)

# graph.render("regression_tree")  # saves PDF/PNG
graph
graph.render(f"figures/dt_metrics/L{layer}N{neuron}_tree", format="png", cleanup=True)
# %%
