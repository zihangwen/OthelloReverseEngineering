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
# Load decision trees
dt_name = 'neuron_simulation/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl'
with open(dt_name, "rb") as f:
    decision_trees = pickle.load(f)

function_name = list(decision_trees[0].keys())[0]
n_features = decision_trees[0][function_name]["decision_tree"]["model"].n_features_in_
feature_names = create_feature_names(n_features, function_name)

# %%
binary_dt_name = 'neuron_simulation/decision_trees_binary/decision_trees_mlp_neuron_6000.pkl'
with open(binary_dt_name, "rb") as f:
    binary_decision_trees = pickle.load(f)

binary_function_name = list(binary_decision_trees[0].keys())[0]
n_binary_features = binary_decision_trees[0][binary_function_name]["binary_decision_tree"]["model"].n_features_in_
binary_feature_names = create_feature_names(n_binary_features, binary_function_name)

# %%
max_depth = 3
layer = 4
neuron = 2046

tree_model = decision_trees[layer][function_name]['decision_tree']['model'].estimators_[neuron]
r2_score = decision_trees[layer][function_name]['decision_tree']['r2'][neuron]
fig, ax = plt.subplots(figsize=(20, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=max_depth
)

ax.set_title(f"Decision Tree (L{layer}N{neuron})\nR² Score: {r2_score:.4f}", fontsize=16, pad=20)

# %%
from sklearn.tree import DecisionTreeRegressor
import graphviz

def export_pruned_tree(tree, cutoff, feature_names=None):
    """
    Export a pruned decision tree to Graphviz DOT format, 
    keeping only leaves with value >= cutoff and their ancestors.
    
    Parameters
    ----------
    tree : DecisionTreeRegressor
        A fitted sklearn decision tree model.
    cutoff : float
        Minimum leaf prediction value to keep.
    feature_names : list of str, optional
        Names for features (default: f0, f1, ...).
    
    Returns
    -------
    dot : str
        Graphviz DOT representation of the pruned tree.
    """
    tree_ = tree.tree_
    n_nodes = tree_.node_count
    
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(tree_.n_features_out_)]
    
    # Build parent links
    parent = [-1] * n_nodes
    for i in range(n_nodes):
        for child in [tree_.children_left[i], tree_.children_right[i]]:
            if child != -1:
                parent[child] = i
    
    # Decide which nodes to keep
    keep = [False] * n_nodes
    for i in range(n_nodes):
        if tree_.children_left[i] == -1:  # leaf
            if tree_.value[i, 0, 0] >= cutoff:
                j = i
                while j != -1 and not keep[j]:
                    keep[j] = True
                    j = parent[j]
    
    # Recursive function to emit DOT nodes/edges
    lines = ["digraph Tree {", "node [shape=box, style=\"rounded\"] ;"]
    
    def recurse(node_id):
        if not keep[node_id]:
            return
        if tree_.children_left[node_id] == -1:  # leaf
            value = tree_.value[node_id, 0, 0]
            lines.append(f'{node_id} [label="value={value:.3f}", shape=ellipse];')
        else:
            feat = feature_names[tree_.feature[node_id]]
            thr = tree_.threshold[node_id]
            lines.append(f'{node_id} [label="{feat} <= {thr:.2f}"];')
            for child, edge_label in [
                (tree_.children_left[node_id], "True"),
                (tree_.children_right[node_id], "False"),
            ]:
                if keep[child]:
                    recurse(child)
                    lines.append(f"{node_id} -> {child} [labeldistance=2.5, labelangle=45, headlabel=\"{edge_label}\"];")
    
    recurse(0)  # start at root
    lines.append("}")
    return "\n".join(lines)

# %%
# dot = export_pruned_tree(tree_model, cutoff=1, feature_names=feature_names)
# graph = graphviz.Source(dot)

dot_data = export_graphviz(
    tree_model,
    out_file=None,
    feature_names=feature_names,
    filled=True, rounded=True,
    special_characters=True,
    proportion=True,   # scale node size by samples
    max_depth=3,
)
graph = graphviz.Source(dot_data)

graph.render("regression_tree")  # saves PDF/PNG
graph
graph.render("tree_3layers", format="png", cleanup=True)

# %%
binary_tree_model = binary_decision_trees[layer][function_name]['binary_decision_tree']['model'].estimators_[neuron]
f1_score = binary_decision_trees[layer][function_name]['binary_decision_tree']['f1'][neuron]
precision = binary_decision_trees[layer][function_name]['binary_decision_tree']['precision'][neuron]
recall = binary_decision_trees[layer][function_name]['binary_decision_tree']['recall'][neuron]
accuracy = binary_decision_trees[layer][function_name]['binary_decision_tree']['accuracy'][neuron]

fig, ax = plt.subplots(figsize=(20, 12))

plot_tree(
    binary_tree_model,
    feature_names=binary_feature_names,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=max_depth
)

ax.set_title(f"Decision Tree (L{layer}N{neuron})\nF1 Score: {f1_score:.4f}. Precision: {precision:.4f}. Recall: {recall:.4f}. Accuracy: {accuracy:.4f}", fontsize=16, pad=20)

# %%
