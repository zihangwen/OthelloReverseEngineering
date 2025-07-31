# %%
import os
import pickle
import matplotlib.pyplot as plt
# Decision Tree Visualization Tool
# This script allows you to visualize decision trees for specific neurons in specific layers
# Based on the approach shown in dt_tutorial.py but adapted for our neuron decision tree data

import pickle
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
import numpy as np
from typing import Optional, List

# %%
with open("/home/zihangw/Algoverse/OthelloReverseEngineering/neuron_simulation/decision_trees/results_mlp_neuron_trainer_0_inputs_6000.pkl", "rb") as f:
    training_result = pickle.load(f)
    
with open("/home/zihangw/Algoverse/OthelloReverseEngineering/neuron_simulation/decision_trees/ablation_results_mlp_neuron_dt_ablate_not_selected_True_add_error_True_trainer_0_inputs_6000.pkl", "rb") as f:
    ablation_dt = pickle.load(f)

with open("/home/zihangw/Algoverse/OthelloReverseEngineering/neuron_simulation/decision_trees/ablation_results_mlp_neuron_mean_ablate_not_selected_True_add_error_True_trainer_0_inputs_6000.pkl", "rb") as f:
    ablation_mean = pickle.load(f)

# %% 
r2_threshold = 0.7
r2_scores = [
    values['games_batch_to_input_tokens_flipped_bs_classifier_input_BLC']['decision_tree']['r2']
    for _, values in training_result["results"].items()
]
num_features = [(r2_score>r2_threshold).sum() for r2_score in r2_scores]

print(num_features)

# %%
# f1_threshold = 0.9
# f1_scores = [
#     values['games_batch_to_input_tokens_flipped_bs_classifier_input_BLC']['binary_decision_tree']['f1']
#     for _, values in training_result["results"].items()
# ]
# num_features = [(f1_score>f1_threshold).sum() for f1_score in f1_scores]

# print(num_features)

# %%
with open("/home/zihangw/Algoverse/OthelloReverseEngineering/neuron_simulation/decision_trees_6000/results_mlp_neuron_trainer_0_inputs_6000.pkl", "rb") as f:
    training_result = pickle.load(f)

r2_threshold = 0.7
r2_scores = [
    values['games_batch_to_input_tokens_flipped_bs_classifier_input_BLC']['decision_tree']['r2']
    for _, values in training_result["results"].items()
]
num_features = [(r2_score>r2_threshold).sum() for r2_score in r2_scores]

print(num_features)

# %%
LAYER = 5
# NEURON_IDX = 1407
DATA_PATH = 'neuron_simulation/decision_trees_6000/decision_trees_mlp_neuron_6000.pkl'
MAX_DEPTH = None  # Set to an integer to limit tree depth in visualization
SAVE_PATH = None  # Set to a file path to save the visualization

with open(DATA_PATH, "rb") as f:
    data  = pickle.load(f)

print(f"Loaded decision tree data from {DATA_PATH}")
print(f"Available layers: {list(data.keys())}")

# %%
function_name = list(data[LAYER].keys())[0]
print(f"Using function: {function_name}")

# %%
# Helper functions
def get_neuron_decision_tree(data: dict, layer: int, neuron_idx: int, function_name: str):
    """Extract the decision tree for a specific neuron."""
    if layer not in data:
        raise ValueError(f"Layer {layer} not found in data. Available layers: {list(data.keys())}")
    
    if function_name not in data[layer]:
        available_funcs = list(data[layer].keys())
        raise ValueError(f"Function {function_name} not found. Available: {available_funcs}")
    
    multi_output_model = data[layer][function_name]['decision_tree']['model']
    
    if neuron_idx >= len(multi_output_model.estimators_):
        raise ValueError(f"Neuron {neuron_idx} not found. Max neuron index: {len(multi_output_model.estimators_) - 1}")
    
    neuron_tree = multi_output_model.estimators_[neuron_idx]
    r2_scores = data[layer][function_name]['decision_tree']['r2']
    neuron_r2 = r2_scores[neuron_idx]
    
    return neuron_tree, neuron_r2

def create_placeholder_feature_names(n_features: int) -> List[str]:
    """Create feature names based on the actual feature structure:
    (64 + 64 + 5) + (192) + (64) = 389 dimensional vector
    - Last move: 64 one-hot move + 64 pre-occupied + 5 coordinates  
    - Board state: 192 one-hot (8x8x3 mine/empty/theirs)
    - Flipped moves: 64 binary encoding of flipped squares
    
    Square notation: A0-H7 where A0 is top-left, H7 is bottom-right
    """
    feature_names = []
    idx = 0
    
    # First 64: Last move one-hot encoding (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + col) + str(row)  # A0, B0, ..., H7
        feature_names.append(f"{square}_just_played")
        idx += 1
    
    # Next 64: Pre-occupied squares (A0-H7)  
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + col) + str(row)
        feature_names.append(f"{square}_pre_occupied")
        idx += 1
    
    # Next 5: Move coordinates and player info
    coord_names = ["move_row", "move_col", "move_number", "black_played", "white_played"]
    for i in range(min(5, n_features - idx)):
        feature_names.append(coord_names[i])
        idx += 1
    
    # Next 192: Board state (8x8x3 = mine/empty/theirs)
    for square_idx in range(min(64, (n_features - idx) // 3)):
        row = square_idx // 8  
        col = square_idx % 8
        square = chr(ord('A') + col) + str(row)
        
        # Add the 3 states for this square
        if idx < n_features:
            feature_names.append(f"{square}_mine")
            idx += 1
        if idx < n_features:
            feature_names.append(f"{square}_empty") 
            idx += 1
        if idx < n_features:
            feature_names.append(f"{square}_theirs")
            idx += 1
    
    # Last 64: Flipped squares (A0-H7)
    for i in range(min(64, n_features - idx)):
        row = i // 8
        col = i % 8
        square = chr(ord('A') + col) + str(row)
        feature_names.append(f"{square}_flipped")
        idx += 1
    
    # Add any remaining features as generic (shouldn't happen with 389 total)
    while idx < n_features:
        feature_names.append(f"Feature_{idx}")
        idx += 1
    
    return feature_names

# %%
# Get layer statistics
def get_neuron_stats(data: dict, layer: int, function_name: str):
    """Get statistics about neurons in a layer."""
    if layer not in data or function_name not in data[layer]:
        return None
    
    r2_scores = np.array(data[layer][function_name]['decision_tree']['r2'])
    
    stats = {
        'total_neurons': len(r2_scores),
        'mean_r2': r2_scores.mean(),
        'median_r2': np.median(r2_scores),
        'max_r2': r2_scores.max(),
        'min_r2': r2_scores.min(),
        'well_explained': (r2_scores > 0.7).sum(),
        'top_10_neurons': np.argsort(r2_scores)[-10:][::-1],  # Top 10 by R²
        'top_10_r2s': r2_scores[np.argsort(r2_scores)[-10:][::-1]]
    }
    
    return stats

# Show layer statistics
stats = get_neuron_stats(data, LAYER, function_name)
if stats:
    print(f"\nLayer {LAYER} Statistics:")
    print(f"Total neurons: {stats['total_neurons']}")
    print(f"Mean R²: {stats['mean_r2']:.4f}")
    print(f"Median R²: {stats['median_r2']:.4f}")
    print(f"Well explained (R² > 0.7): {stats['well_explained']}")
    print(f"Top 10 neurons by R²:")
    for neuron, r2 in zip(stats['top_10_neurons'], stats['top_10_r2s']):
        print(f"  Neuron {neuron}: R² = {r2:.4f}")

#%%
# Create histogram of R² scores
def plot_r2_histogram(data: dict, layer: int, function_name: str, neuron_index: Optional[int] = None, save_path: Optional[str] = None):
    """Plot histogram of R² scores for all neurons in a layer."""
    if layer not in data or function_name not in data[layer]:
        print(f"No data found for layer {layer}, function {function_name}")
        return
    
    r2_scores = np.array(data[layer][function_name]['decision_tree']['r2'])
    
    plt.figure(figsize=(10, 6))
    plt.hist(r2_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('R² Score')
    plt.ylabel('Number of Neurons')
    plt.title(f'Distribution of R² Scores for Layer {layer}\n(Mean: {r2_scores.mean():.3f}, Median: {np.median(r2_scores):.3f})')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line for current neuron's R²
    current_r2 = r2_scores[neuron_index] if neuron_index < len(r2_scores) else None
    if current_r2 is not None:
        plt.axvline(current_r2, color='red', linestyle='--', linewidth=2, 
                   label=f'Neuron {neuron_index}: R² = {current_r2:.3f}')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved R² histogram to {save_path}")
    
    plt.show()

# %%
NEURON_IDX = 1407
# Plot the histogram
plot_r2_histogram(data, LAYER, function_name, neuron_index = NEURON_IDX)

# %%
# Extract the specific neuron's decision tree
tree_model, r2_score = get_neuron_decision_tree(data, LAYER, NEURON_IDX, function_name)
print(f"\nRetrieved decision tree for Layer {LAYER}, Neuron {NEURON_IDX}")
print(f"R² Score: {r2_score:.4f}")

# Debug: Check if trees are actually different
multi_output_model = data[LAYER][function_name]['decision_tree']['model']
print(f"\nDebugging - comparing with other neurons:")
for test_neuron in [0, 1, 2]:
    if test_neuron < len(multi_output_model.estimators_):
        test_tree = multi_output_model.estimators_[test_neuron]
        print(f"Neuron {test_neuron}: tree depth = {test_tree.get_depth()}, n_nodes = {test_tree.tree_.node_count}")

# Create feature names specifically for this tree's features
n_features = tree_model.n_features_in_
feature_names = create_placeholder_feature_names(n_features)
print(f"Number of input features for this neuron's tree: {n_features}")

# Debug: Show tree-specific info
print(f"This tree depth: {tree_model.get_depth()}")
print(f"This tree node count: {tree_model.tree_.node_count}")
print(f"Tree feature importances (top 5): {sorted(enumerate(tree_model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]}")

# %%
# Print decision tree rules in text format
def print_tree_rules(tree_model, neuron_idx: int, layer: int, r2_score: float, 
                    feature_names: List[str], max_depth: Optional[int] = None):
    """Print the decision tree rules in text format."""
    print(f"\n{'='*60}")
    print(f"Decision Tree Rules for Layer {layer}, Neuron {neuron_idx}")
    print(f"R² Score: {r2_score:.4f}")
    print(f"{'='*60}")
    
    # Handle None max_depth for export_text (scikit-learn bug workaround)
    export_max_depth = max_depth if max_depth is not None else tree_model.get_depth()
    
    tree_rules = export_text(
        tree_model, 
        feature_names=feature_names,
        max_depth=export_max_depth
    )
    print(tree_rules)

print_tree_rules(tree_model, NEURON_IDX, LAYER, r2_score, feature_names, max_depth=MAX_DEPTH)

# %%
# Visualize the decision tree
def visualize_decision_tree(tree_model, neuron_idx: int, layer: int, r2_score: float,
                          feature_names: List[str], max_depth: Optional[int] = None,
                          save_path: Optional[str] = None):
    """Visualize a decision tree for a specific neuron."""
    plt.figure(figsize=(20, 12))
    
    plot_tree(
        tree_model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth
    )
    
    plt.title(f"Decision Tree for Layer {layer}, Neuron {neuron_idx}\nR² Score: {r2_score:.4f}", 
              fontsize=16, pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

visualize_decision_tree(tree_model, NEURON_IDX, LAYER, r2_score, feature_names, 
                       max_depth=3, save_path=SAVE_PATH)

# %%