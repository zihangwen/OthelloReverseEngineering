# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
import einops
from collections import defaultdict
from typing import Callable, Optional
import os
import importlib
import pickle
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any

import circuits.utils as utils
import circuits.othello_utils as othello_utils
import neuron_simulation.simulation_config as sim_config
import simulate_activations_with_dts as sim_activations

# %%
default_config = sim_config.selected_config
device = "cpu"

# If desired, you can run simulate_activations_with_dts.py from this cell
# Values from default config will also be used later on in the notebook to filter out saved pickle files

# Example of filtering out certain configurations

# default_config.n_batches = 2

# for combination in default_config.combinations:
#     combination.ablate_not_selected = [True]

# sim_activations.run_simulations(default_config)

# %%
METRICS_NAME_MAPPING = {
    "f1": "F1 Score",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "r2": "R² Score",
}

SAE_METRICS = ["f1", "accuracy", "precision", "recall"]
GROUP_BY_OPTIONS = ["input_location", "custom_function", "decision_tree_file"]

label_lookup = {
    "mlp_neuron": "MLP Neuron",
    "sae_mlp_out_feature": "SAE Trained on MLP out",
    "transcoder": "Transcoder",
    "mlp_neuron_mean_ablate": "MLP Neuron (Mean Ablate)",
    othello_utils.games_batch_to_input_tokens_flipped_classifier_input_BLC.__name__: "Input Tokens and Flipped Squares",
    othello_utils.games_batch_to_board_state_classifier_input_BLC.__name__: "Board State (Mine / Yours / Blank)",
    othello_utils.games_batch_to_input_tokens_classifier_input_BLC.__name__: "Input Tokens",
    othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC.__name__: "Board State, Input Tokens and Flipped Squares",
}

def load_ablation_pickle_files(
    directory: str,
    test_size: int,
    ablation_method: str,
    ablate_not_selected: bool,
    add_error: bool,
):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl") and "ablation" in filename:
            with open(os.path.join(directory, filename), "rb") as f:
                single_data = pickle.load(f)

            hyperparams = single_data["hyperparameters"]
            if hyperparams["ablate_not_selected"] != ablate_not_selected:
                continue

            if hyperparams["input_location"] != "mlp_neuron":
                if hyperparams["add_error"] != add_error:
                    continue

            if hyperparams["ablation_method"] != ablation_method:
                continue

            if hyperparams["test_size"] != test_size:
                continue
            data.append(single_data)
    return data

def load_results_pickle_files(directory: str, test_size: int) -> List[Dict[str, Any]]:
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl') and "ablation" not in filename and "decision_trees" not in filename:
            with open(os.path.join(directory, filename), 'rb') as f:
                single_data = pickle.load(f)

            if single_data['hyperparameters']['test_size'] != test_size:
                continue
            data.append(single_data)
    return data

def calculate_good_features(results: Dict[str, Any], metric: str, threshold: float) -> Tuple[int, int]:
    # dt_r2 = torch.tensor(results["decision_tree"][metric])
    # good_dt_r2 = (dt_r2 > threshold).sum()
    good_dt_r2 = -1

    f1 = torch.tensor(results["binary_decision_tree"][metric])
    good_f1 = (f1 > threshold).sum().item()
    # good_f1 = -1

    return good_dt_r2, good_f1


def extract_other_ablation_results(
    data: List[Dict],
    ablation_method: str,
    desired_metric: str,
    desired_layer_tuples: Optional[List[Tuple[int]]] = None,
) -> Dict:
    allowed_metrics = ["kl", "patch_accuracy"]
    if desired_metric not in allowed_metrics:
        raise ValueError(f"desired_metric must be one of {allowed_metrics}")

    nested_results = {}
    for run in data:
        hyperparams = run["hyperparameters"]
        input_location = hyperparams["input_location"] + f"_{ablation_method}_ablate"
        trainer_id = hyperparams["trainer_id"]

        nested_results[input_location] = {trainer_id: {}}

        for layer_tuple, func_results in run["results"].items():
            if desired_layer_tuples is not None and layer_tuple not in desired_layer_tuples:
                continue
            prev_result = None
            for idx, func_name in enumerate(func_results):

                result = func_results[func_name][desired_metric]
                nested_results[input_location][trainer_id][layer_tuple] = result

                if idx > 0:
                    assert prev_result == result

                prev_result = result

    return nested_results


def extract_ablation_results(
    data: List[Dict],
    custom_function_names: list[str],
    desired_metric: str,
    group_by: str,
    input_location_filter: Optional[str] = None,
    func_name_filter: Optional[str] = None,
    desired_layer_tuples: Optional[List[Tuple[int]]] = None,
    threshold: float = 0.7,
) -> Dict:
    if desired_metric not in SAE_METRICS:
        raise ValueError(f"desired_metric must be one of {SAE_METRICS}")

    if group_by not in GROUP_BY_OPTIONS:
        raise ValueError(f"group_by must be one of {GROUP_BY_OPTIONS}")
    
    if func_name_filter is not None and func_name_filter not in custom_function_names:
        raise ValueError(f"func_name_filter must be one of {custom_function_names}")

    nested_results = {}
    for run in data:
        hyperparams = run["hyperparameters"]
        input_location = hyperparams["input_location"]
        trainer_id = hyperparams["trainer_id"]

        if input_location_filter is not None and input_location != input_location_filter:
            continue

        for custom_function_name in custom_function_names:

            if func_name_filter is not None and custom_function_name != func_name_filter:
                continue
            
            if group_by == "input_location":
                primary_key = input_location
            elif group_by == "custom_function":
                primary_key = custom_function_name
            elif group_by == "decision_tree_file":
                primary_key = os.path.basename(hyperparams["decision_tree_file"])

            if primary_key not in nested_results:
                nested_results[primary_key] = {}
            if trainer_id not in nested_results[primary_key]:
                nested_results[primary_key][trainer_id] = {}

            for layer_tuple, func_results in run["results"].items():
                if desired_layer_tuples is not None and layer_tuple not in desired_layer_tuples:
                    continue
                if custom_function_name in func_results:
                    if desired_metric in ["f1", "accuracy", "precision", "recall", "r2"]:
                        _, good_f1 = calculate_good_features(
                            func_results[custom_function_name], desired_metric, threshold
                        )
                        result = good_f1
                    else:
                        result = func_results[custom_function_name][desired_metric]

                    nested_results[primary_key][trainer_id][layer_tuple] = result

    return nested_results

def extract_ablation_results_mean(
    data: List[Dict],
    custom_function_names: list[str],
    desired_metric: str,
    group_by: str,
    input_location_filter: Optional[str] = None,
    func_name_filter: Optional[str] = None,
    desired_layer_tuples: Optional[List[Tuple[int]]] = None,
    # threshold: float = 0.7,
) -> Dict:
    if desired_metric not in SAE_METRICS:
        raise ValueError(f"desired_metric must be one of {SAE_METRICS}")

    if group_by not in GROUP_BY_OPTIONS:
        raise ValueError(f"group_by must be one of {GROUP_BY_OPTIONS}")
    
    if func_name_filter is not None and func_name_filter not in custom_function_names:
        raise ValueError(f"func_name_filter must be one of {custom_function_names}")

    nested_results = {}
    for run in data:
        hyperparams = run["hyperparameters"]
        input_location = hyperparams["input_location"]
        trainer_id = hyperparams["trainer_id"]

        if input_location_filter is not None and input_location != input_location_filter:
            continue

        for custom_function_name in custom_function_names:

            if func_name_filter is not None and custom_function_name != func_name_filter:
                continue
            
            if group_by == "input_location":
                primary_key = input_location
            elif group_by == "custom_function":
                primary_key = custom_function_name
            elif group_by == "decision_tree_file":
                primary_key = os.path.basename(hyperparams["decision_tree_file"])

            if primary_key not in nested_results:
                nested_results[primary_key] = {}
            if trainer_id not in nested_results[primary_key]:
                nested_results[primary_key][trainer_id] = {}

            for layer_tuple, func_results in run["results"].items():
                if desired_layer_tuples is not None and layer_tuple not in desired_layer_tuples:
                    continue
                if custom_function_name in func_results:
                    if desired_metric in ["f1", "accuracy", "precision", "recall", "r2"]:
                        score = torch.tensor(func_results[custom_function_name]["binary_decision_tree"][desired_metric])
                        result = score.mean().item()
                    else:
                        result = func_results[custom_function_name][desired_metric]

                    nested_results[primary_key][trainer_id][layer_tuple] = result

    return nested_results

# %%
def plot_dataset_size_comparison_binary(metric: str, test_size: int, group_by: str = "decision_tree_file"):
    """Plot overlays comparing the same metric across different dataset sizes"""
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    print("Loading data files...")
    results_data = load_results_pickle_files(directory, test_size)
    print(f"Loaded {len(results_data)} results files")
        
    metric_per_layers = extract_ablation_results(
        results_data,
        custom_function_names,
        metric,
        group_by,
        input_location_filter=None,
        func_name_filter=None,
        desired_layer_tuples=list(range(8)),
    )

    metric_per_layers = dict(sorted(metric_per_layers.items(), key=lambda key: key))
    print("Metric keys:", list(metric_per_layers.keys()))
    
    all_layers = set()
    for _, trainer_ids in metric_per_layers.items():
        for _, layer_results in trainer_ids.items():
            all_layers.update(layer_results.keys())
    
    all_layers = sorted(all_layers)

    # all_layers = all_layers[1:]  # Skip layer 0 for clearer visualization
    
    for i, (individual_label, trainer_ids) in enumerate(metric_per_layers.items()):
        for _, layer_results in trainer_ids.items():
            values = [layer_results.get(layer, np.nan) for layer in all_layers]
            # full_label = f"{display_label} ({ds_size} training games)"
            plt.plot(
                range(len(all_layers)),
                values,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linestyle='-',
                linewidth=2,
                markersize=6,
                label=individual_label,
            )

    # Set up the plot
    title = "Binary Decision Tree Interpretable Neuron Count Comparison Across Dataset Sizes\n(Evaluated on 500-game test set, higher is better)"
    y_label = f"Number of Neurons with {METRICS_NAME_MAPPING[metric]} > 0.7"
    
    plt.xlabel("Layer", fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if all_layers:
        layer_labels = [str(layer) for layer in all_layers]
        plt.xticks(range(len(all_layers)), layer_labels)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/images/{metric}_dataset_size_comparison_mean.png", dpi=300, bbox_inches='tight')
    plt.show()


# def plot_dataset_size_comparison_f1_neurons(
#     layer: int,
#     test_size: int,
#     tree_file_select: list[str],
# ):
#     """Plot overlays comparing the same metric across different dataset sizes"""
#     assert len(tree_file_select) == 2

#     plt.figure(figsize=(12, 8))
    
#     colors = ['blue', 'red', 'green', 'orange', 'purple']
#     markers = ['o', 's', '^', 'D', 'v']
    
#     print("Loading data files...")
#     results_data = load_results_pickle_files(directory, test_size)
#     print(f"Loaded {len(results_data)} results files")

#     dt_file_list = []
#     r2_scores_list = []
#     for data in results_data:
#         hyperparams = data["hyperparameters"]
#         dt_file_list.append(os.path.basename(hyperparams["decision_tree_file"]))

#         r2_scores = np.array(data["results"][layer][custom_function_names[0]]['decision_tree']['r2'])
#         r2_scores_list.append(r2_scores)
    
#     assert all([file in dt_file_list for file in tree_file_select])
#     indices = [dt_file_list.index(file) for file in tree_file_select]
#     r2_diff = r2_scores_list[indices[0]] - r2_scores_list[indices[1]]

#     plt.figure(figsize=(10, 6))
#     plt.plot(r2_diff)
#     plt.axhline(0, color='gray', linestyle='--', linewidth=1)

#     plt.xlabel('Neuron')
#     plt.ylabel(f'R² differences')
    
#     plt.title(f'Per MLP neuron differences for Layer {layer}\nbetween {dt_file_list[indices[0]]} and {dt_file_list[indices[1]]}')
#     plt.grid(True, alpha=0.3)
#     plt.savefig(
#         (
#             f"figures/images/r2_diff_"
#             f"{dt_file_list[indices[0]].split("_")[-1].split(".")[0]}_"
#             f"{dt_file_list[indices[1]].split("_")[-1].split(".")[0]}.png"
#         ),
#         dpi=300, bbox_inches='tight'
#     )
#     plt.show()

#     return r2_diff


def plot_different_f1_threshold(test_size: int, group_by: str = "decision_tree_file", df_select: str = "decision_trees_mlp_neuron_6000.pkl", f1_threshold_list: list[float] = [0.5, 0.7, 0.9]):
    """Plot overlays comparing the same metric across different dataset sizes"""
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    results_data = load_results_pickle_files(directory, test_size)
    print(f"Loaded {len(results_data)} results files")

    metric_per_layers_dict = {}
    print("Loading data files...")
    for f1_threshold in f1_threshold_list:
        print(f"\n--- F1 threshold: {f1_threshold} ---")
        metric_per_layers = extract_ablation_results(
            results_data,
            custom_function_names,
            "f1",
            group_by,
            input_location_filter=None,
            func_name_filter=None,
            desired_layer_tuples=list(range(8)),
            threshold=f1_threshold,
        )
        metric_per_layers_dict[f1_threshold] = metric_per_layers

    # dt_file_list = []
    # for data in results_data:
    #     hyperparams = data["hyperparameters"]
    #     dt_file_list.append(os.path.basename(hyperparams["decision_tree_file"]))
    
    # indices = [idx for idx, df_file in enumerate(dt_file_list) if df_file == df_select]
    # index = indices[0] if indices else None

    all_layers = set()
    for _, trainer_ids in metric_per_layers.items():
        for _, layer_results in trainer_ids.items():
            all_layers.update(layer_results.keys())
    
    all_layers = sorted(all_layers)
    for f1_threshold, metric_per_layers in metric_per_layers_dict.items():
        print(metric_per_layers[df_select])
        for i, (_, layer_results) in enumerate(metric_per_layers[df_select].items()):
            values = [layer_results.get(layer, np.nan) for layer in all_layers]
            # full_label = f"{display_label} ({ds_size} training games)"
            plt.plot(
                range(len(all_layers)),
                values,
                marker=markers[i % len(markers)],
                # color=colors[i % len(colors)],
                linestyle='-',
                linewidth=2,
                markersize=6,
                label=f"F1 > {f1_threshold}",
            )

    
    plt.xlabel("Layer")
    plt.ylabel("Number of Neurons")
    plt.title("Binary Decision Tree Interpretable Neuron Count Comparison Across F1 Thresholds\n(Evaluated on 500-game test set)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if all_layers:
        layer_labels = [str(layer) for layer in all_layers]
        plt.xticks(range(len(all_layers)), layer_labels)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/images/dt_6000/f1_threshold_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_different_metrics(test_size: int, group_by: str = "decision_tree_file", df_select: str = "decision_trees_mlp_neuron_6000.pkl", metrics_list: list[float] = ["f1", "accuracy", "precision", "recall"]):
    """Plot overlays comparing the same metric across different dataset sizes"""
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    results_data = load_results_pickle_files(directory, test_size)
    print(f"Loaded {len(results_data)} results files")

    metric_per_layers_dict = {}
    print("Loading data files...")
    for metric in metrics_list:
        metric_per_layers = extract_ablation_results_mean(
            results_data,
            custom_function_names,
            metric,
            group_by,
            input_location_filter=None,
            func_name_filter=None,
            desired_layer_tuples=list(range(8)),
        )
        metric_per_layers_dict[metric] = metric_per_layers

    all_layers = set()
    for _, trainer_ids in metric_per_layers.items():
        for _, layer_results in trainer_ids.items():
            all_layers.update(layer_results.keys())
    
    all_layers = sorted(all_layers)
    for metric, metric_per_layers in metric_per_layers_dict.items():
        print(metric_per_layers[df_select])
        for i, (_, layer_results) in enumerate(metric_per_layers[df_select].items()):
            values = [layer_results.get(layer, np.nan) for layer in all_layers]
            # full_label = f"{display_label} ({ds_size} training games)"
            plt.plot(
                range(len(all_layers)),
                values,
                marker=markers[i % len(markers)],
                # color=colors[i % len(colors)],
                linestyle='-',
                linewidth=2,
                markersize=6,
                label=f"{METRICS_NAME_MAPPING[metric]}",
            )
    
    plt.xlabel("Layer")
    plt.ylabel("Mean Value")
    plt.title("Binary Decision Tree Metrics\n(Evaluated on 500-game test set)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if all_layers:
        layer_labels = [str(layer) for layer in all_layers]
        plt.xticks(range(len(all_layers)), layer_labels)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/images/dt_6000/metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Create overlay plots comparing different dataset sizes (60, 600, 6000)
directory = "neuron_simulation/decision_trees_binary_eval"
# Updated custom function names to match what you trained with
custom_function_names = [
    othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC.__name__,
]

# default_config = sim_config.selected_config
# 60, 600, or 6000
# dataset_size = 600
test_size = 500

ablation_method = "dt"
ablate_not_selected = True
add_error = False

desired_layer_tuples = []
# Generate single layer tuples to match your training
for i in range(8):
    desired_layer_tuples.append((i,))

group_by = "decision_tree_file"

# Create overlay plots for all three metrics
# dataset_sizes = [60, 600, 6000]  # Add/remove dataset sizes as needed
test_size = 500  # Fixed test size for comparison
metrics_to_compare = ["f1", "accuracy", "precision", "recall"]

# %% ----- ----- ----- ----- ----- dataset size comparison plots ----- ----- ----- ----- ----- %% #
print("Creating dataset size comparison plots...")
for metric in metrics_to_compare:
    print(f"\n=== Creating {metric.upper()} comparison plot ===")
    plot_dataset_size_comparison_binary(metric, test_size, group_by)

print("\nAll comparison plots created successfully!")

# %% ----- ----- ----- ----- ----- metric comparison plots ----- ----- ----- ----- ----- %% #
plot_different_metrics(test_size, group_by, df_select = "decision_trees_mlp_neuron_6000.pkl", metrics_list = metrics_to_compare)

# %% ----- ----- ----- ----- ----- r2 per neuron ----- ----- ----- ----- ----- %% #
# r2_diff = plot_dataset_size_comparison_f1_neurons(
#     layer=5,
#     test_size=test_size,
#     tree_file_select=[
#         "decision_trees_mlp_neuron_6000.pkl",
#         "decision_trees_mlp_neuron_600.pkl",
#     ],
# )

# r2_diff_negative = r2_diff < 0
# r2_diff_negative_indices = np.where(r2_diff_negative)[0]
# r2_diff_negative_scores = r2_diff[r2_diff_negative_indices]

# sorted_pairs = sorted(zip(r2_diff_negative_scores, r2_diff_negative_indices))  # Sort by b
# r2_diff_negative_scores, r2_diff_negative_indices = zip(*sorted_pairs)  # Unzip back to separate lists

# print("number of negative R² differences:", len(r2_diff_negative_scores))
# print("negative neuron indexs:", *r2_diff_negative_indices)
# print("negative neuron scores:", *r2_diff_negative_scores)

# %% ----- ----- ----- ----- ----- f1 with different threshold ----- ----- ----- ----- ----- %% #
plot_different_f1_threshold(test_size, group_by, df_select = "decision_trees_mlp_neuron_6000.pkl", f1_threshold_list=[0.5, 0.7, 0.9])

# %%



