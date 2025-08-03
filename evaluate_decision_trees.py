#!/usr/bin/env python3
"""
Standalone evaluation script for decision trees on Othello data.
Reads decision tree files and evaluates them on 500 test games with interventions.
"""

import argparse
import torch
import pickle
import os
import sys
from typing import Callable, Optional, List
import itertools

# Add the parent directory to the Python path to find circuits module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import circuits.utils as utils
import circuits.othello_utils as othello_utils
import neuron_simulation.simulation_config as sim_config
from simulate_activations_with_dts import (
    construct_dataset_per_layer,
    get_submodule_dict,
    cache_sae_activations,
    calculate_binary_activations,
    calculate_neuron_metrics,
    calculate_binary_metrics,
    perform_interventions,
    update_results_dict,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_decision_trees(decision_tree_file: str) -> dict:
    """Load decision trees from pickle file."""
    if not os.path.exists(decision_tree_file):
        raise FileNotFoundError(f"Decision tree file not found: {decision_tree_file}")
    
    with open(decision_tree_file, "rb") as f:
        decision_trees = pickle.load(f)
    
    return decision_trees


def evaluate_decision_trees(
    decision_tree_file: str,
    input_location: str,
    trainer_id: int,
    ablation_methods: List[str],
    intervention_layers: List[List[int]],
    custom_functions: List[Callable],
    train_dataset_size: int,
    model_name: str = "Othello-GPT-Transformer-Lens",
    test_size: int = 500,
    intervention_threshold: float = 0.2,
    binary_threshold: float = 0.1,
    repo_dir: str = "autoencoders",
    output_location: str = "",
    ablate_not_selected_options: List[bool] = [False],
    add_error_options: List[bool] = [False],
):
    """
    Evaluate decision trees with interventions on test data.
    
    Args:
        decision_tree_file: Path to pickled decision tree file
        input_location: Type of input (e.g. "sae_mlp_out_feature", "mlp_neuron")
        trainer_id: SAE trainer ID to use
        ablation_methods: List of ablation methods to test
        intervention_layers: List of layer combinations to intervene on
        custom_functions: List of custom functions used for input features
        model_name: Name of the model
        test_size: Number of test games to evaluate on
        batch_size: Batch size for processing
        intervention_threshold: Threshold for selecting features to intervene on
        binary_threshold: Threshold for binary activations
        repo_dir: Directory containing autoencoders
        output_location: Directory to save results
        ablate_not_selected_options: Whether to ablate non-selected features
        add_error_options: Whether to add reconstruction error
    """
    
    print(f"Loading decision trees from: {decision_tree_file}")
    decision_trees = load_decision_trees(decision_tree_file)
    
    # Debug: Print structure of decision trees
    print("Decision tree structure:")
    for layer in list(decision_trees.keys())[:2]:  # Just first 2 layers
        print(f"Layer {layer}:")
        for func_name in decision_trees[layer]:
            print(f"  Function: {func_name}")
            print(f"    Keys: {list(decision_trees[layer][func_name].keys())}")
            for key in decision_trees[layer][func_name]:
                print(f"      {key}: {list(decision_trees[layer][func_name][key].keys())}")
        break  # Just show first layer for brevity
    
    print(f"Loading model and test data (test_size={test_size})")
    # Load model directly and only create test data
    model = utils.get_model(model_name, device)
    test_data = construct_dataset_per_layer(
        custom_functions, test_size, "test", device, list(range(8))
    )
    
    # Load SAEs
    print(f"Loading SAEs for input_location={input_location}, trainer_id={trainer_id}")
    ae_list = utils.get_aes(
        node_type=input_location, 
        repo_dir=repo_dir, 
        trainer_id=trainer_id
    )
    
    for i in range(len(ae_list)):
        ae_list[i] = ae_list[i].to(device)
    
    submodule_dict = get_submodule_dict(
        model, model_name, list(range(8)), input_location
    )
    
    # Cache SAE activations on test data to evaluate decision tree performance
    print("Caching SAE activations on test data...")
    test_n_batches = (test_size + 50 - 1) // 50  # Use batch size of 50
    neuron_acts = cache_sae_activations(
        model,
        test_data,
        list(range(8)),
        50,  # batch_size
        test_n_batches,
        input_location,
        ae_list,
        submodule_dict,
    )
    
    binary_acts = calculate_binary_activations(neuron_acts, binary_threshold)
    
    # Move to CPU for sklearn operations
    neuron_acts = utils.to_device(neuron_acts, "cpu")
    binary_acts = utils.to_device(binary_acts, "cpu")
    
    # Evaluate decision trees on test data
    print("Evaluating decision trees on test data...")
    results = {"hyperparameters": {
        "dataset_size": train_dataset_size,
        "test_size": test_size,
        "intervention_threshold": intervention_threshold,
        "binary_threshold": binary_threshold,
        "trainer_id": trainer_id,
        "input_location": input_location,
    }, "results": {}}
    
    for layer in range(8):
        results["results"][layer] = {}
        for custom_function in custom_functions:
            func_name = custom_function.__name__
            
            # Get test data for this layer and function
            games_BLC = test_data[layer][func_name]
            games_BLC = utils.to_device(games_BLC, "cpu")
            
            # Prepare data for sklearn
            from sklearn.model_selection import train_test_split
            import einops
            X = einops.rearrange(games_BLC, "b l c -> (b l) c").cpu().numpy()
            y_regular = einops.rearrange(neuron_acts[layer], "b l d -> (b l) d").cpu().numpy()
            y_binary = einops.rearrange(binary_acts[layer], "b l d -> (b l) d").cpu().numpy()
            
            # Get decision tree model and evaluate on test data
            dt_model = decision_trees[layer][func_name]["decision_tree"]["model"]
            dt_mse, dt_r2 = calculate_neuron_metrics(dt_model, X, y_regular)
            
            results["results"][layer][func_name] = {
                "decision_tree": {
                    "mse": dt_mse,
                    "r2": dt_r2,
                },
                "binary_decision_tree": {
                    "f1": None,
                    "accuracy": None,
                },
            }
    
    # Save decision tree results (like training script does)
    results_filename = f"{output_location}/results_{input_location}_trainer_{trainer_id}_inputs_{train_dataset_size}.pkl"
    print(f"Saving decision tree results to: {results_filename}")
    update_results_dict(results_filename, results)
    
    hyperparameters = {
        "dataset_size": train_dataset_size,  # Use train_dataset_size to match training script
        "test_size": test_size,
        "intervention_threshold": intervention_threshold,
        "binary_threshold": binary_threshold,
        "trainer_id": trainer_id,
        "input_location": input_location,
        "decision_tree_file": decision_tree_file,
    }
    
    # Generate all combinations of ablation parameters
    true_false_combinations = list(itertools.product(ablate_not_selected_options, add_error_options))
    
    for ablation_method in ablation_methods:
        for combo in true_false_combinations:
            ablate_not_selected, add_error = combo
            
            print(f"\nEvaluating with ablation_method={ablation_method}, "
                  f"ablate_not_selected={ablate_not_selected}, add_error={add_error}")
            
            ablations = perform_interventions(
                decision_trees=decision_trees,
                input_location=input_location,
                ablation_method=ablation_method,
                ablate_not_selected=ablate_not_selected,
                add_error=add_error,
                custom_functions=custom_functions,
                model=model,
                intervention_layers=intervention_layers,
                data=test_data,
                threshold=intervention_threshold,
                ae_dict=ae_list,
                submodule_dict=submodule_dict,
                hyperparameters=hyperparameters.copy(),
            )
            
            # Save results
            ablation_filename = (
                f"{output_location}/ablation_results_{input_location}_{ablation_method}_"
                f"ablate_not_selected_{ablate_not_selected}_add_error_{add_error}_"
                f"trainer_{trainer_id}_inputs_{train_dataset_size}.pkl"
            )
            
            print(f"Saving results to: {ablation_filename}")
            update_results_dict(ablation_filename, ablations)


def main():
    # Hardcoded settings - modify these as needed
    decision_tree_file = "neuron_simulation/decision_trees/decision_trees_mlp_neuron_6000.pkl"
    input_location = "mlp_neuron"
    trainer_id = 0
    ablation_methods = ["dt", "mean"]
    intervention_layers = [[0], [1], [2], [3], [4], [5], [6], [7]]
    custom_functions = [othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC]
    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    train_dataset_size = 6000  # Size of training dataset used to train the decision trees

    # Other settings
    test_size = 500
    intervention_threshold = 0.7
    binary_threshold = 0.1
    repo_dir = ""  # Base directory, utils.get_ae will append "autoencoders/"
    output_location = "neuron_simulation/decision_trees"
    ablate_not_selected_options = [True]
    add_error_options = [True]

    print(f"Evaluating decision trees from: {decision_tree_file}")
    print(f"Input location: {input_location}")
    print(f"Trainer ID: {trainer_id}")
    print(f"Ablation methods: {ablation_methods}")
    print(f"Test size: {test_size}")

    evaluate_decision_trees(
        decision_tree_file=decision_tree_file,
        input_location=input_location,
        trainer_id=trainer_id,
        ablation_methods=ablation_methods,
        intervention_layers=intervention_layers,
        custom_functions=custom_functions,
        train_dataset_size=train_dataset_size,
        model_name=model_name,
        test_size=test_size,
        intervention_threshold=intervention_threshold,
        binary_threshold=binary_threshold,
        repo_dir=repo_dir,
        output_location=output_location,
        ablate_not_selected_options=ablate_not_selected_options,
        add_error_options=add_error_options,
    )

    print("Evaluation completed!")


if __name__ == "__main__":
    main()