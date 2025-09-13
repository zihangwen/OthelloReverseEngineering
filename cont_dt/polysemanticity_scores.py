import pickle
import json
import gzip
from functools import lru_cache
from pprint import pprint
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, TypeAlias
from matplotlib.figure import Figure
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from skimage.filters import threshold_otsu
import torch as t
from torch import Tensor
from jaxtyping import Int
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from cont_dt.cont_feature_dt import DecisionTreeResults
from cont_dt.cont_dt_viz import load_all_trees, get_feature_names, visualize_decision_tree


DEPTH = 4
N_LAYERS = 8

def get_leaf_node_values(tree: DecisionTreeRegressor) -> list[float]:
    tree_obj = tree.tree_

    children_left = tree_obj.children_left
    children_right = tree_obj.children_right
    values = tree_obj.value
    thresholds = tree_obj.threshold

    def _traverse(node_id: int) -> list[float]:
        # base case: if leaf node
        if children_left[node_id] == children_right[node_id]:
            return [values[node_id][0][0]]

        return _traverse(children_left[node_id]) + _traverse(children_right[node_id])

    return _traverse(0)


def get_on_indices(leaf_node_values: list[float]) -> list[int]:
    on_off_threshold = threshold_otsu(np.array(leaf_node_values))
    return [i for i in range(len(leaf_node_values)) if leaf_node_values[i] > on_off_threshold]


def get_ancestor_distance(leaf_idx1, leaf_idx2, depth=4):
    """Calculates d_A = N - depth(LCA) for two leaf nodes."""
    if leaf_idx1 == leaf_idx2:
        return 0

    bin1 = format(leaf_idx1, f'0{depth}b')
    bin2 = format(leaf_idx2, f'0{depth}b')

    lca_depth = 0
    for i in range(depth):
        if bin1[i] == bin2[i]:
            lca_depth += 1
        else:
            break
            
    return depth - lca_depth


def get_distance_matrix(depth=4) -> Int[Tensor, "n_leafs n_leafs"]:
    pairwise_distances = t.empty((2**depth, 2**depth))
    for i in range(2**depth):
        for j in range(2**depth):
            pairwise_distances[i][j] = get_ancestor_distance(i, j, depth)

    return pairwise_distances


def get_polysemanticity_score(neuron: DecisionTreeResults) -> Literal[1, 2, 3, 4]:
    decision_tree = neuron.tree

    leaf_node_values = get_leaf_node_values(decision_tree)

    on_indices = get_on_indices(leaf_node_values)

    k = len(on_indices)

    if k < 2:
        return 0

    distance_matrix = get_distance_matrix()

    max_dist = float('-inf')
    for i in range(k):
        for j in range(i + 1, k):
            leaf1_idx = on_indices[i]
            leaf2_idx = on_indices[j]
            dist = distance_matrix[leaf1_idx][leaf2_idx]
            if dist > max_dist:
                max_dist = dist

    return max_dist


if __name__ == "__main__":
    all_trees = load_all_trees()
    feature_names = get_feature_names()

    results = {}
    for layer in range(1, N_LAYERS):
        layer_trees = all_trees[layer]
        results[layer] = [get_polysemanticity_score(neuron) for neuron in layer_trees]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for layer in range(1, N_LAYERS):
        ax = axes[layer - 1]
        
        # Count occurrences of each polysemanticity score
        scores = results[layer]
        unique_scores, counts = np.unique(scores, return_counts=True)
        
        # Create bar plot
        bars = ax.bar(unique_scores, counts)
        ax.set_title(f'Layer {layer}')
        ax.set_xlabel('Polysemanticity Score')
        ax.set_ylabel('Count')
        ax.set_xticks(range(5))  # 0, 1, 2, 3, 4
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()








