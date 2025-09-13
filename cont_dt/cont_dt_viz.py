"""
Visualize decision tree trained on Othello neuron activations
"""

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
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from cont_dt.cont_feature_dt import DecisionTreeResults


FILE_PATH = Path(__file__).resolve()
PARENT_DIR = FILE_PATH.parent
RESULTS_DIR = PARENT_DIR / "results"


N_LAYERS = 8
D_MLP = 2048


def get_feature_names():
    """
    Generate feature names for all projections.
    
    Returns list of feature names in order:
    - 64 projections for mine - theirs (all squares)
    - 60 projections for blank (excluding middle 4: D3, D4, E3, E4)
    - 64 projections for flipped (all squares)
    - 60 projections for placed (excluding middle 4)
    """
    feature_names = []
    
    # Helper to get square name from row/col
    def get_square_name(row, col):
        row_letter = chr(ord('A') + row)
        return f"{row_letter}{col}"
    
    # 1. Mine - Theirs (all 64 squares)
    for row in range(8):
        for col in range(8):
            square = get_square_name(row, col)
            feature_names.append(f"{square} mine-theirs")
    
    # 2. Blank (60 squares, excluding D3, D4, E3, E4)
    middle_squares = {(3, 3), (3, 4), (4, 3), (4, 4)}  # D3, D4, E3, E4
    for row in range(8):
        for col in range(8):
            if (row, col) not in middle_squares:
                square = get_square_name(row, col)
                feature_names.append(f"{square} blank")
    
    # 3. Flipped (all 64 squares)
    for row in range(8):
        for col in range(8):
            square = get_square_name(row, col)
            feature_names.append(f"{square} flipped")
    
    # 4. Placed (60 squares, excluding middle 4)
    for row in range(8):
        for col in range(8):
            if (row, col) not in middle_squares:
                square = get_square_name(row, col)
                feature_names.append(f"{square} placed")
    
    return feature_names


def load_decision_tree_for_layer(
    layer : int, 
) -> list[DecisionTreeResults]:
    """Load decision trees for layer"""
    file_name = f"layer_{layer}_trees.pkl.gz"
    model_path = RESULTS_DIR / file_name

    with gzip.open(model_path, 'rb') as f:
        trees = pickle.load(f)
    
    return trees


def visualize_decision_tree(
    tree: DecisionTreeResults, 
    feature_names: list[str], 
    save_path: str | None = None, 
    figsize: tuple[float, float] = (20, 10),
 ) -> Figure:
    """
    Create a visualization of the decision tree with proper feature labels.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create title with metrics
    title = (f"Decision Tree: Layer {tree.layer}, Neuron {tree.neuron}\n"
             f"Test R² = {tree.test_R2:.3f}\n"
             f"Depth = {tree.tree.max_depth}, "
             f"Leaves = {tree.tree.tree_.n_leaves}")
    
    # Plot the tree
    model = tree.tree
    plot_tree(model, 
              feature_names=feature_names,
              filled=True,
              rounded=True,
              fontsize=10,
              ax=ax,
              impurity=False,  # Don't show impurity (MSE) values
              precision=2)  # Round values to 2 decimal places
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return fig


@dataclass(frozen=True) 
class Condition:
    feature_name: str
    operator: Literal['<=', '>']
    threshold: float


DecisionPath: TypeAlias = list[Condition]


def traverse_tree(tree: DecisionTreeRegressor) -> list[tuple[DecisionPath, float]]:
    tree_obj = tree.tree_

    children_left = tree_obj.children_left
    children_right = tree_obj.children_right
    values = tree_obj.value
    thresholds = tree_obj.threshold
    features = tree_obj.feature
    
    feature_names = get_feature_names()

    def _traverse(node_id: int, path: DecisionPath) -> list[tuple[DecisionPath, float]]:
        # base case: if leaf node
        if children_left[node_id] == children_right[node_id]:
            return [(path, values[node_id][0][0])]

        left_condition = Condition(
            feature_name=feature_names[features[node_id]],
            operator='<=',
            threshold=thresholds[node_id]
        )
        left_path = path + [left_condition]

        right_condition = Condition(
            feature_name=feature_names[features[node_id]],
            operator='>',
            threshold=thresholds[node_id]
        )
        right_path = path + [right_condition]

        return _traverse(children_left[node_id], left_path) + _traverse(children_right[node_id], right_path)

    return _traverse(0, [])


def otsu(leaf_nodes: list[tuple[DecisionPath, float]]) -> list[DecisionPath]:
    leaf_values = np.array([value for _, value in leaf_nodes])
    on_off_threshold = threshold_otsu(leaf_values)

    on_paths = [
        path for path, value in leaf_nodes
        if value > on_off_threshold
    ]

    return on_paths


def process_neuron(tree: DecisionTreeRegressor) -> list[DecisionPath]:
    """Takes in a neuron's tree, returns a list of on decision paths
    representing OR-of-ANDs structure"""
    leaf_nodes = traverse_tree(tree)
    on_paths = otsu(leaf_nodes)
    return on_paths


def is_condition_implied(
    query_cond: Condition, 
    path_conditions_by_feature: dict
) -> bool:
    
    feature = query_cond.feature_name
    
    # Get all the constraints the path places on this specific feature.
    relevant_path_conditions = path_conditions_by_feature.get(feature)

    # If the path says nothing about this feature, it cannot guarantee the query.
    if not relevant_path_conditions:
        return False

    # Now we check the logic based on the operator.
    # CASE 1: Query is "feature > threshold"
    if query_cond.operator == '>':
        # To guarantee 'feature > q_val', the path must contain at least one
        # condition 'feature > p_val' where p_val >= q_val.
        for p_cond in relevant_path_conditions:
            if p_cond.operator == '>' and p_cond.threshold >= query_cond.threshold:
                return True # Found a path condition that guarantees the query condition.

    # CASE 2: Query is "feature <= threshold"
    if query_cond.operator == '<=':
        # To guarantee 'feature <= q_val', the path must contain at least one
        # condition 'feature <= p_val' where p_val <= q_val.
        for p_cond in relevant_path_conditions:
            if p_cond.operator == '<=' and p_cond.threshold <= query_cond.threshold:
                return True # Found a path condition that guarantees the query condition.

    # If we finish the loops without finding a sufficiently strong condition,
    # the implication is false.
    return False


def does_path_imply_query(path: DecisionPath, query: DecisionPath) -> bool:
    # Step 1: Organize the path's conditions by feature for easy lookup.
    path_conditions_by_feature = defaultdict(list)
    for cond in path:
        path_conditions_by_feature[cond.feature_name].append(cond)

    # Step 2: Check every condition in the query.
    # If any query condition is NOT implied by the path, the whole query is not implied.
    for query_condition in query:
        if not is_condition_implied(query_condition, path_conditions_by_feature):
            return False # Found a failing condition, so we can stop early.

    # Step 3: If we looped through all query conditions and they all passed,
    # then the path does imply the query.
    return True


def check_neuron(tree: DecisionTreeResults, query: DecisionPath) -> bool:
    """Takes a neuron's tree and checks if any of its ON conditions
    guarantee the path condition"""
    decision_tree = tree.tree
    on_paths = process_neuron(decision_tree)

    return any(does_path_imply_query(path, query) for path in on_paths)


def check_layer(trees: list[DecisionTreeResults], query: DecisionPath) -> list[int]:
    """Return neuron ids satisfying query"""
    return [neuron_id for neuron_id, neuron_tree in enumerate(trees) if check_neuron(neuron_tree, query)]


@lru_cache(maxsize=1)
def load_all_trees(
    n_layers: int = 8,
) -> dict[int, list[DecisionTreeResults]]:
    """Loads and caches all decision trees from disk."""
    print("Loading all decision trees from disk... (this will happen only once)")
    return {
        layer: load_decision_tree_for_layer(layer=layer)
        for layer in range(1, n_layers)
    }


def check_model(trees: dict[int, list[DecisionTreeResults]], query: DecisionPath) -> dict[int: list[int]]:
    """Returns all neurons satisfying query"""
    return {layer: check_layer(layer_trees, query) for layer, layer_trees in trees.items()}


def find_neurons_for_query(query: DecisionPath) -> dict[int, list[int]]:
    """
    Finds neurons that satisfy the query using a cached tree loader.
    """
    all_trees = load_all_trees() 
    return check_model(all_trees, query)


if __name__ == "__main__":
    all_trees = load_all_trees()

    feature_names = get_feature_names()
    viz = visualize_decision_tree(all_trees[5][1952], feature_names)
    
    process_neuron(all_trees[5][1952].tree)
    # query = [
    #     Condition(feature_name='C0 blank', operator='>', threshold=-1),
    #     Condition(feature_name='D1 mine-theirs', operator='<=', threshold=1),
    #     #Condition(feature_name='E2 mine-theirs', operator='>', threshold=-1),
    # ] 
    # pprint(check_model(all_trees, query))