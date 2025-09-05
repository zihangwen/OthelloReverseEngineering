import torch
import numpy as np
import einops
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from joblib import Parallel, delayed
import multiprocessing
from typing import Callable, Optional
import os
import pickle
import itertools
from importlib import resources
import gc

from datasets import load_dataset

# from xgboost import XGBRegressor, XGBClassifier
# import cuml
import circuits.othello_engine_utils as othello_engine_utils

import circuits.utils as utils
import circuits.othello_utils as othello_utils
# from circuits.eval_sae_as_classifier import print_tensor_memory_usage
import neuron_simulation.simulation_config as sim_config

# Setup
device = "cuda:2" if torch.cuda.is_available() else "cpu"
# device = "cpu"
torch.set_grad_enabled(False)
tracer_kwargs = {"validate": False, "scan": False}
tracer_kwargs = {"validate": True, "scan": True}


def construct_othello_dataset_flex(
    custom_functions: list[Callable],
    inputs_index: list,
    split: str,
    max_str_length: int = 59,
    device: str = "cpu",
    precompute_dataset: bool = True,
) -> dict:
    dataset = load_dataset("adamkarvonen/othello_45MB_games", streaming=False)
    encoded_othello_inputs_bL = []
    decoded_othello_inputs_bL = []
    # for i, example in enumerate(dataset[split]):
    #     if i >= n_inputs:
    #         break
    for i in range(inputs_index[0], inputs_index[1]):
        example = dataset[split][i]
        encoded_input = example["tokens"][:max_str_length]
        decoded_input = othello_engine_utils.to_string(encoded_input)
        encoded_othello_inputs_bL.append(encoded_input)
        decoded_othello_inputs_bL.append(decoded_input)

    data = {}
    data["encoded_inputs"] = encoded_othello_inputs_bL
    data["decoded_inputs"] = decoded_othello_inputs_bL

    if not precompute_dataset:
        return data

    for custom_function in custom_functions:
        print(f"Precomputing {custom_function.__name__}...")
        func_name = custom_function.__name__
        data[func_name] = custom_function(decoded_othello_inputs_bL)

    return utils.to_device(data, device) # changed by zihangw

def construct_dataset_per_layer_flex(
    custom_functions: list[Callable],
    inputs_index: list,
    split: str,
    device: str,
) -> dict:
    """NOTE: By default we use .clone() on tensors, which will increase memory usage with number of layers.
    At current dataset sizes this is not a problem, but keep in mind for larger datasets."""
    custom_functions.append(othello_utils.games_batch_to_valid_moves_BLRRC)
    data = construct_othello_dataset_flex(
        custom_functions=custom_functions,
        inputs_index=inputs_index,
        split=split,
        device=device,
    )

    all_data = {}

    all_data["encoded_inputs"] = data["encoded_inputs"]
    all_data["decoded_inputs"] = data["decoded_inputs"]
    all_data["valid_moves"] = data[othello_utils.games_batch_to_valid_moves_BLRRC.__name__].cpu().numpy().astype(np.int8)

    for custom_function in custom_functions:
        if custom_function == othello_utils.games_batch_to_valid_moves_BLRRC:
            continue
        func_name = custom_function.__name__
        if func_name not in all_data:
            all_data[func_name] = {}

        all_data[func_name] = data[func_name].cpu().numpy().astype(np.int8)

    custom_functions.pop()

    return all_data


def cache_neuron_activations(
    model, data: dict, layers: list, batch_size: int, n_batches: int
) -> dict:
    """Deprecated in favor of using identity autoencoders"""
    neuron_acts = defaultdict(list)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        data_batch = data["encoded_inputs"][batch_start:batch_end]
        data_batch = torch.tensor(data_batch, device=device)

        with torch.no_grad(), model.trace(data_batch, scan=False, validate=False):
            for layer in layers:
                neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output.save()
                neuron_acts[layer].append(neuron_activations_BLD)

    for layer in neuron_acts:
        neuron_acts[layer] = torch.stack(neuron_acts[layer])
        neuron_acts[layer] = einops.rearrange(neuron_acts[layer], "n b l c -> (n b) l c")

    return neuron_acts


def get_submodule_dict(model, model_name: str, layers: list, input_location: str) -> dict:
    submodule_dict = {}

    for layer in layers:
        if input_location in ["sae_feature", "sae_feature_topk"]:
            submodule = utils.get_resid_post_submodule(model_name, layer, model)
        elif input_location == "sae_mlp_feature":
            submodule = utils.get_mlp_activations_submodule(model_name, layer, model)
        elif input_location == "mlp_neuron":
            submodule = utils.get_mlp_activations_submodule(model_name, layer, model)
        elif input_location == "attention_out":
            submodule = model.blocks[layer].hook_attn_out
        elif input_location == "mlp_out" or input_location == "sae_mlp_out_feature":
            submodule = model.blocks[layer].hook_mlp_out
        elif input_location == "transcoder":
            submodule = model.blocks[layer].mlp
        else:
            raise ValueError(f"Invalid input location: {input_location}")
        submodule_dict[layer] = submodule

    return submodule_dict


@torch.no_grad()
def cache_sae_activations(
    model,
    data: dict,
    layers: list,
    batch_size: int,
    n_batches: int,
    input_location: str,
    ae_dict: dict,
    submodule_dict: dict,
) -> dict:
    sae_acts = defaultdict(list)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        data_batch = data["encoded_inputs"][batch_start:batch_end]
        data_batch = torch.tensor(data_batch, device=device)

        ae_dict = utils.to_device(ae_dict, device=device) # changed by zihangw
        acts = {}
        with model.trace(data_batch, **tracer_kwargs):
            for layer in layers:
                submodule = submodule_dict[layer]
                if input_location != "transcoder":
                    x = submodule.output
                else:
                    x = submodule.input[0]
                    if type(submodule.input.shape) == tuple:
                        x = x[0]
                acts[layer] = x.save()

        for layer in layers:
            ae = ae_dict[layer]

            f = ae.encode(acts[layer])
            sae_acts[layer].append(f.detach().cpu())

    for layer in sae_acts:
        sae_acts[layer] = torch.stack(sae_acts[layer])
        sae_acts[layer] = einops.rearrange(sae_acts[layer], "n b l c -> (n b) l c")

    return sae_acts


def get_max_activations(neuron_acts: dict, layer: int) -> torch.Tensor:
    D = neuron_acts[layer].shape[-1]
    max_activations_D = torch.full((D,), float("-inf"), device=neuron_acts[layer].device)

    neuron_acts_BLD = neuron_acts[layer]
    neuron_acts_BD = einops.rearrange(neuron_acts_BLD, "b l d -> (b l) d")

    max_activations_D = torch.max(max_activations_D, neuron_acts_BD.max(dim=0).values)
    return max_activations_D


def calculate_binary_activations(neuron_acts: dict, threshold: float):
    binary_acts = {}

    for layer in neuron_acts:
        max_activations_D = get_max_activations(neuron_acts, layer)

        binary_acts[layer] = (neuron_acts[layer] > (threshold * max_activations_D)).int()
    return binary_acts


# def prepare_data(games_BLC: torch.Tensor, mlp_acts_BLD: torch.Tensor):
#     """sklearn.fit requires 2D input, so we need to flatten the batch and sequence dimensions."""
#     X = einops.rearrange(games_BLC, "b l c -> (b l) c").cpu().numpy()
#     y = einops.rearrange(mlp_acts_BLD, "b l d -> (b l) d").cpu().numpy()
#     return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_data(games_BLC: torch.Tensor, mlp_acts_BLD: torch.Tensor):
    """sklearn.fit requires 2D input, so we need to flatten the batch and sequence dimensions."""
    X = einops.rearrange(games_BLC, "b l c -> (b l) c")
    y = einops.rearrange(mlp_acts_BLD, "b l d -> (b l) d")
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2


def calculate_neuron_metrics(model, X_BF, y_BF):
    y_pred_BF = model.predict(X_BF)

    # Calculate MSE for all neurons at once
    mse_list_F = np.mean((y_BF - y_pred_BF) ** 2, axis=0)

    # Calculate R2 for all neurons at once
    ss_res = np.sum((y_BF - y_pred_BF) ** 2, axis=0)
    ss_tot = np.sum((y_BF - np.mean(y_BF, axis=0)) ** 2, axis=0)

    # Add divide-by-zero protection
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_list_F = 1 - (ss_res / ss_tot)

    # Handle cases where ss_tot is zero
    r2_list_F = np.where(ss_tot == 0, 0, r2_list_F)

    # Clip R2 values to be between 0 and 1
    r2_list_F = np.clip(r2_list_F, 0, 1)

    return mse_list_F, r2_list_F


def calculate_binary_metrics(model, X, y):
    y_pred = model.predict(X)

    # Compute true positives, false positives, true negatives, false negatives
    tp = np.sum((y_pred == 1) & (y == 1), axis=0)
    fp = np.sum((y_pred == 1) & (y == 0), axis=0)
    tn = np.sum((y_pred == 0) & (y == 0), axis=0)
    fn = np.sum((y_pred == 0) & (y == 1), axis=0)

    # Compute metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)

    # Compute F1 score
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )

    return accuracy, precision, recall, f1


def compute_kl_divergence(logits_clean_BLV, logits_patch_BLV):
    # Apply softmax to get probability distributions
    log_probs_clean_BLV = torch.nn.functional.log_softmax(logits_clean_BLV, dim=-1)
    log_probs_patch_BLV = torch.nn.functional.log_softmax(logits_patch_BLV, dim=-1)

    # Compute KL divergence
    kl_div_BLV = torch.nn.functional.kl_div(
        log_probs_patch_BLV, log_probs_clean_BLV.exp(), reduction="none", log_target=False
    )

    # Sum over the vocabulary dimension
    kl_div_BL = kl_div_BLV.sum(dim=-1)

    return kl_div_BL


def compute_top_n_accuracy(
    logits_BLV: torch.Tensor, valid_moves_BLRRC: torch.Tensor
) -> tuple[float, float, float]:
    valid_moves_BLC = einops.rearrange(valid_moves_BLRRC, "b l r1 r2 c -> b l (r1 r2 c)")
    n_BL = einops.reduce(valid_moves_BLC, "B L C -> B L", "sum")

    # Get the shape of the logits tensor
    B, L, V = logits_BLV.shape

    # Create a mask for the top n logits
    top_n_mask = torch.zeros_like(logits_BLV, dtype=torch.bool)

    for b in range(B):
        for l in range(L):
            n = n_BL[b, l].int()
            _, top_n_indices = torch.topk(logits_BLV[b, l], k=n)
            top_n_mask[b, l, top_n_indices] = True

    top_n_mask = top_n_mask.int()
    stoi_top_n_mask = torch.zeros(B, L, (V + 4), dtype=torch.int32, device=top_n_mask.device)

    # This is so cursed. OthelloGPT has D vocab 61 (ignoring center squares, with pass at idx 0)
    stoi_top_n_mask[:, :, :28] = top_n_mask[:, :, :28]
    stoi_top_n_mask[:, :, 30:36] = top_n_mask[:, :, 28:34]
    stoi_top_n_mask[:, :, 38:] = top_n_mask[:, :, 34:]

    pass_BL1 = torch.zeros(B, L, 1, dtype=torch.int32, device=valid_moves_BLC.device)

    valid_moves_with_pass_BLC = torch.cat([pass_BL1, valid_moves_BLC], dim=-1)

    correct_BLC = valid_moves_with_pass_BLC * stoi_top_n_mask

    correct = correct_BLC.sum()
    total = valid_moves_with_pass_BLC.sum()
    accuracy = correct / total

    return correct.item(), total.item(), accuracy.item()


def add_output_folders(base_path):
    # # Get the current working directory
    # current_dir = os.getcwd()

    # # Check if we're already in the neuron_simulation directory
    # if os.path.basename(current_dir) == "neuron_simulation":
    #     base_path = ""
    # else:
    #     base_path = "neuron_simulation"

    # Create the directories
    os.makedirs(os.path.join(base_path, "decision_trees"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "images"), exist_ok=True)


def process_layer_simple(
    layer: int,
    data_layer: dict,
    func_name: str,
    neuron_acts_layer,
    binary_acts_layer,
    max_depth: int,
    binary_dt: bool,
    regular_dt: bool,
    linear_reg: bool = False,
    random_seed: int = 42,
) -> dict:
    print(f"\nLayer {layer}")

    games_BLC = data_layer[func_name]
    games_BLC = utils.to_device(games_BLC, "cpu")

    layer_results = {"layer": layer}

    if regular_dt:
        X_train, X_test, y_train, y_test = prepare_data(games_BLC, neuron_acts_layer)
        # del neuron_acts_layer
        # gc.collect()

        # Decision Tree
        # cmlt_model = cuml.ensemble.RandomForestRegressor(n_estimators=1, bootstrap=False, random_state=random_seed, max_depth=max_depth)
        # cmlt_model, cmlt_mse, cmlt_r2 = train_and_evaluate(
        #     cmlt_model, X_train, X_test, y_train, y_test
        # )
        dt_model, dt_mse, dt_r2 = train_and_evaluate(
            MultiOutputRegressor(
                DecisionTreeRegressor(
                    random_state=random_seed,
                    max_depth=max_depth,  # min_samples_leaf=5, min_samples_split=5
                )
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        )

        dt_mse, dt_r2 = calculate_neuron_metrics(dt_model, X_test, y_test)
        layer_results["regular_dt"] = {"model": dt_model, "mse": dt_mse, "r2": dt_r2}
    else:
        layer_results["regular_dt"] = {"mse": None, "r2": None}

    if binary_dt:
        # Binary Decision Tree
        X_binary_train, X_binary_test, y_binary_train, y_binary_test = prepare_data(
            games_BLC, binary_acts_layer
        )
        del games_BLC, data_layer, binary_acts_layer
        gc.collect()

        dt_binary_model = MultiOutputClassifier(
            DecisionTreeClassifier(
                random_state=random_seed,
                max_depth=max_depth,  # min_samples_leaf=5, min_samples_split=5
            )
        )
        dt_binary_model.fit(X_binary_train, y_binary_train)

        accuracy, precision, recall, f1 = calculate_binary_metrics(
            dt_binary_model, X_binary_test, y_binary_test
        )
        layer_results["binary_dt"] = {
            "model": dt_binary_model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    else:
        layer_results["binary_dt"] = {"accuracy": None, "f1": None}

    if linear_reg:
        lasso_model, lasso_mse, lasso_r2 = train_and_evaluate(
            Lasso(alpha=0.005), X_train, X_test, y_train, y_test
        )
        layer_results["lasso"] = {"model": lasso_model, "mse": lasso_mse, "r2": lasso_r2}

    print(f"Finished Layer {layer}")

    return layer_results


def compute_predictors(
    custom_functions: list[Callable],
    num_cores: int,
    layers: list[int],
    data: dict,
    neuron_acts: dict,
    binary_acts: dict,
    input_location: str,
    dataset_size: int,
    force_recompute: bool,
    save_results: bool,
    max_depth: int,
    output_location: str,
    binary_dt: bool,
    regular_dt: bool,
) -> dict:
    output_filename = (
        f"{output_location}decision_trees/decision_trees_{input_location}_{dataset_size}.pkl"
    )

    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)

    if not force_recompute and os.path.exists(output_filename):
        print(f"Loading decision trees from {output_filename}")
        with open(output_filename, "rb") as f:
            decision_trees = pickle.load(f)
        return decision_trees

    # Use all available cores, but max out at num_cores
    num_cores = min(num_cores, multiprocessing.cpu_count())
    # num_cores = 4

    results = {}

    for layer in layers:
        results[layer] = {}

    for custom_function in custom_functions:
        func_name = custom_function.__name__

        print(f"\n{func_name}")

        layer_results = Parallel(n_jobs=num_cores)(
            delayed(process_layer_simple)(
                layer,
                data,
                func_name,
                neuron_acts[layer],
                binary_acts[layer],
                max_depth,
                binary_dt,
                regular_dt,
            )
            for layer in layers
        )

        for layer_result in layer_results:
            if layer_result is not None:
                layer = layer_result["layer"]
                results[layer][custom_function.__name__] = {
                    "decision_tree": layer_result["regular_dt"],
                    "binary_decision_tree": layer_result["binary_dt"],
                }

        # with ProcessPoolExecutor(max_workers=num_cores) as executor:
        #     future_to_layer = {executor.submit(process_layer, layer, games_BLC, neuron_acts, binary_acts): layer for layer in layers}
        #     for future in concurrent.futures.as_completed(future_to_layer):
        #         layer_result = future.result()
        #         layer = layer_result['layer']
        #         results[layer][custom_function.__name__] = {
        #             'decision_tree': layer_result['regular_dt'],
        #             'binary_decision_tree': layer_result['binary_dt']
        #         }

    if save_results:
        update_results_dict(output_filename, results)
        # with open(output_filename, "wb") as f:
        #     pickle.dump(results, f)
    return results


def simulate_activation(
    data_layer: dict, decision_trees_layer: dict, func_name: str
) -> dict[int, torch.Tensor]:
    board_state_BLC = data_layer[func_name]
    B, L, C = board_state_BLC.shape
    X = einops.rearrange(board_state_BLC, "b l c -> (b l) c").cpu().numpy()

    decision_tree = decision_trees_layer[func_name]["decision_tree"]["model"]
    simulated_activations_BF = decision_tree.predict(X)
    simulated_activations_BF = torch.tensor(
        simulated_activations_BF, dtype=torch.float32
    )
    simulated_activations_BLF = einops.rearrange(
        simulated_activations_BF, "(b l) f -> b l f", b=B, l=L
    )

    return simulated_activations_BLF


def update_results_dict(output_file: str, results: dict):
    if os.path.exists(output_file):
        # If it exists, load the existing results
        with open(output_file, "rb") as f:
            existing_results = pickle.load(f)

        # Update the existing results with new data
        for layer in results["results"]:
            if layer not in existing_results["results"]:
                existing_results["results"][layer] = {}
            for func_name in results["results"][layer]:
                if func_name not in existing_results["results"][layer]:
                    existing_results["results"][layer][func_name] = results["results"][layer][
                        func_name
                    ]

        results = existing_results
    # Write the results (either new or updated) to the file
    with open(output_file, "wb") as f:
        pickle.dump(results, f)


def save_cached_data(config: sim_config.SimulationConfig, inputs_index_list: list[list[int]]):
    add_output_folders(config.output_location)

    model = utils.get_model(config.model_name, device)

    combination = config.combinations[0]
    input_location = combination.input_location
    trainer_ids = combination.trainer_ids
    trainer_id = trainer_ids[0]
    
    for inputs_index in inputs_index_list:
        print("=" * 20 + f" Processing inputs {inputs_index} " + "=" * 20)
        gc.collect()
        torch.cuda.empty_cache()

        train_data = construct_dataset_per_layer_flex(
            config.custom_functions, inputs_index, "train", device,
        )

        with open(f"{config.output_location}cached_data/train_data_{combination.input_location}_trainer_{trainer_id}_inputs_{inputs_index[0]}-{inputs_index[1]}.pkl", "wb") as f:
            pickle.dump(train_data, f)

        
        ae_list = utils.get_aes(
            node_type=input_location, repo_dir=config.repo_dir, trainer_id=trainer_id
        )

        for i in range(len(ae_list)):
            ae_list[i] = ae_list[i].to(device)

        submodule_dict = get_submodule_dict(
            model, config.model_name, config.layers, input_location
        )

        neuron_acts = cache_sae_activations(
            model,
            train_data,
            config.layers,
            config.batch_size,
            config.n_batches,
            input_location,
            ae_list,
            submodule_dict,
        )

        with open(f"{config.output_location}cached_data/neuron_acts_{combination.input_location}_trainer_{trainer_id}_inputs_{inputs_index[0]}-{inputs_index[1]}.pkl", "wb") as f:
            pickle.dump(neuron_acts, f)

        binary_acts = calculate_binary_activations(neuron_acts, config.binary_threshold)
        with open(f"{config.output_location}cached_data/binary_acts_{combination.input_location}_trainer_{trainer_id}_inputs_{inputs_index[0]}-{inputs_index[1]}.pkl", "wb") as f:
            pickle.dump(binary_acts, f)



if __name__ == "__main__":
    import time

    start_time = time.time()
    default_config = sim_config.selected_config
    # default_config = sim_config.test_config
    default_config.save_decision_trees = True # changed by zihangw
    # default_config.binary_dt = True  # changed by zihangw

    # Here you select which functions are going to be used as input to training the decision trees
    # We will iterate over every one
    default_config.custom_functions = [
        othello_utils.games_batch_to_board_state_flipped_played_BLC,
        # othello_utils.games_batch_to_board_state_flipped_played_valid_move_BLC,
        # othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
        # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
    ]

    # Here you select what types of interventions will be performed
    # E.g. decision trees on SAEs on mlp out, mean ablation, decision trees on MLP neurons
    default_config.combinations = [
        # sim_config.selected_sae_mlp_out_feature_config,
        # sim_config.selected_transcoder_config,
        sim_config.MLP_dt_config,
        sim_config.MLP_mean_config,
    ]
    default_config.binary_dt = True  # changed by zihangw
    default_config.regular_dt = False  # changed by zihangw

    default_config.output_location = "neuron_decision_trees/"
    # example config change
    # 6 batches seems to work reasonably well for training decision trees
    # default_config.n_batches = 30
    # default_config.batch_size = 1000

    batch_size = 1000
    n_batches = 60

    inputs_index_list = [
        [i * batch_size, (i + 1) * batch_size]
        for i in range(n_batches)
    ]
    save_cached_data(default_config, inputs_index_list)
    
    print(f"--- {time.time() - start_time} seconds ---")
