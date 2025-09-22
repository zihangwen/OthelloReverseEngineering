# %%
import pickle
from collections import defaultdict
import torch as t
from torch import Tensor
import sys

import numpy as np
import einops
from rich import print as rprint
from rich.table import Column, Table
from rich.console import Console
from rich.terminal_theme import MONOKAI

# from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor

import graphviz
from dataclasses import dataclass
from typing import Literal, TypeAlias, Tuple

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
    calculate_ablation_scores_square_probability,
    calculate_ablation_scores_square_all,
    compute_kl_divergence,
)
# from simulate_activations_with_dts import (
#     compute_kl_divergence,
#     compute_top_n_accuracy,
# )

from utils_feature_extraction import (
    extract_and_rules,
    extract_probe_features,
    rule_infer,
    direct_feature_infer,
    extract_rules_features_from_dt,
)

device = "cuda:1" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = utils.get_model(model_name, device)
n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp

w_out = model.W_out.detach().clone() # [layer, neuron, d_model]
W_U = model.W_U[:, 1:].detach().clone()  # [d_model, 60]

# %%
binary_dt_name = 'neuron_decision_trees/decision_trees_d8/decision_trees_mlp_neuron_30000.pkl'
with open(binary_dt_name, "rb") as f:
    binary_decision_trees = pickle.load(f)

binary_custom_function_name = list(binary_decision_trees[0].keys())[0]
n_binary_features = binary_decision_trees[0][binary_custom_function_name]["binary_decision_tree"]["model"].n_features_in_
binary_feature_names = create_feature_names(n_binary_features, binary_custom_function_name)


# %%
test_size = 500
custom_functions = [
    # othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
    # othello_utils.games_batch_to_valid_moves_BLRRC, # (legal move),
    othello_utils.games_batch_to_state_stack_mine_yours_BLRRC,
]
test_data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=test_size,
    split="test", 
    device=device,
)

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)
board_seqs_square = t.tensor(test_data["decoded_inputs"]).long().to(device)

board_states, legal_moves, legal_moves_annotation = get_board_states_and_legal_moves(board_seqs_square)
legal_moves = legal_moves.to(device=device, dtype=t.float32)
valid_move_number = legal_moves.sum(dim=(-2,-1))

# %%
@dataclass
class ConditionBinary:
    square: str
    feature: Literal["mine", "empty", "theirs", "flipped", "just_played"]
    # feature_name: str
    operator: Literal['', 'NOT']

def get_feature_indices_binary(query: list[ConditionBinary]) -> list[Tuple[int, int, int]]:
    indices = []
    for condition in query:
        square = condition.square
        feature = condition.feature

        # feature_name = condition.feature_name
        operator = condition.operator

        # square, feature = feature_name.split()
        row, col = list(square)
        row = ord(row) - ord('A')
        col = int(col)
        
        if (feature == 'mine'):
            option = 0  # mine
        elif (feature == 'empty'):
            option = 1  # empty
        elif (feature == 'theirs'):
            option = 2  # theirs
        else:
            raise ValueError(f"Unknown feature: {feature}")

        indices.append((row, col, option))
        
        # if operator == 'NOT':
        #     mode = 0
        # else:
        #     mode = 1
        # indices.append((row, col, option, mode))

    return indices

def get_filtered_positions_binary(
    data: dict[str, Tensor],
    intervention_queries: list[list[ConditionBinary]],
    control_queries: list[list[ConditionBinary]],
    feature_function: str = "games_batch_to_state_stack_mine_yours_BLRRC",
):
    features = data[feature_function]
    encoded_inputs = t.tensor(data["encoded_inputs"])
    decoded_inputs = t.tensor(data["decoded_inputs"])
    batch, seq = features.shape[0:2]

    intervention_masks = t.zeros((batch, seq), dtype=t.bool, device=features.device)
    for intervention_query in intervention_queries:
        intervention_indices_binary = get_feature_indices_binary(intervention_query)
        intervention_mask = t.ones((batch, seq), dtype=t.bool, device=features.device)
        for idx in intervention_indices_binary:
            intervention_mask &= (features[..., *idx] == 1)
        
        intervention_masks |= intervention_mask
    
    control_masks = t.zeros((batch, seq), dtype=t.bool, device=features.device)
    for control_query in control_queries:
        control_indices_binary = get_feature_indices_binary(control_query)
        control_mask = t.ones((batch, seq), dtype=t.bool, device=features.device)
        for idx in control_indices_binary:
            control_mask &= (features[..., *idx] == 1)
        
        control_masks |= control_mask
    
    mask = intervention_masks & (~control_masks)
    mask[:,:5] = False
    mask[:,31:] = False  # only consider moves 5-30

    # return mask
    filtered_positions_encoded = []
    filtered_positions_decoded = []
    for game_idx in range(mask.shape[0]):
        indices = t.where(mask[game_idx])[0]
        for idx in indices:
            if 5 <= idx <= 30:
                filtered_positions_encoded.append(encoded_inputs[game_idx, : idx + 1])
                filtered_positions_decoded.append(decoded_inputs[game_idx, : idx + 1])

    return mask, filtered_positions_encoded, filtered_positions_decoded

def filter_neurons(neurons, w_out, W_U, token_id, threshold=0.0):
    neurons_filtered = defaultdict(list)
    write_attribution = einops.einsum(
        w_out,
        W_U,
        "layer neuron d_model, d_model id -> layer neuron id",
    )
    for layer in neurons:
        if layer == 7:
            continue
        for neuron in neurons[layer]:
            write_attr = write_attribution[layer, neuron, token_id-1]
            if write_attr >= threshold:
                neurons_filtered[layer].append(neuron)
    
    return neurons_filtered

# %%
blank_square = 'C0'
intervention_query = [
    ConditionBinary(square='C0', feature='empty', operator=''),
    ConditionBinary(square='D1', feature='theirs', operator=''),
    ConditionBinary(square='E2', feature='mine', operator=''),
]
control_query = [
    ConditionBinary(square='C0', feature='empty', operator=''),
    ConditionBinary(square='C1', feature='theirs', operator=''),
    ConditionBinary(square='C2', feature='mine', operator=''),
]

# intervention_query = [
#     ConditionBinary(square='D1', feature='empty', operator=''),
#     ConditionBinary(square='E2', feature='theirs', operator=''),
#     ConditionBinary(square='F3', feature='mine', operator=''),
# ]
# control_query = [
#     ConditionBinary(square='D1', feature='empty', operator=''),
#     ConditionBinary(square='D2', feature='theirs', operator=''),
#     ConditionBinary(square='D3', feature='mine', operator=''),
# ]
# dt_queries = [
#     [
#         ConditionBinary(square='D1', feature='empty', operator=''),
#         ConditionBinary(square='E2', feature='theirs', operator=''),
#         ConditionBinary(square='F3', feature='mine', operator=''),
#     ],
# ]

# intervention_indices_binary = get_feature_indices_binary(intervention_query)
# control_indices_binary = get_feature_indices_binary(control_query)

# mask, intervention_positions_encoded, intervention_positions_decoded = get_filtered_positions_binary(
mask_i = get_filtered_positions_binary(
    test_data,
    intervention_queries=[intervention_query],
    control_queries=[control_query],
    feature_function="games_batch_to_state_stack_mine_yours_BLRRC",
)
mask_c = get_filtered_positions_binary(
    test_data,
    intervention_queries=[control_query],
    control_queries=[intervention_query],
    feature_function="games_batch_to_state_stack_mine_yours_BLRRC",
)

# %%
# encoded_inputs = t.tensor(test_data["encoded_inputs"])
# decoded_inputs = t.tensor(test_data["decoded_inputs"])

# filtered_positions_encoded = []
# filtered_positions_decoded = []
# count = 0
# for game_idx in range(mask.shape[0]):
#     indices = t.where(mask[game_idx])[0]
#     for idx in indices:
#         if 5 <= idx <= 30:
#             filtered_positions_encoded.append(encoded_inputs[game_idx, : idx + 1])
#             filtered_positions_decoded.append(decoded_inputs[game_idx, : idx + 1])
#             count += 1
#             if count == batch_size:
#                 break
#     if count == batch_size:
#         print("game_idx:", game_idx, "idx:", idx)
#         break

# # %%
# mask_filter = mask[:game_idx+1]
# board_seqs_id_filter = board_seqs_id[:game_idx+1]
# valid_move_number_filter = valid_move_number[:game_idx+1]

# # %%
# positions=intervention_positions_encoded
# query=intervention_query
# # neurons = {layer: [] for layer in range(1, n_layers)}
# # neurons[5].append(1393)
# batch_size = 128

# print(f"Ablating {sum(len(neurons) for neurons in neurons.values())} neurons")
# legal_square_id = neel_utils.to_id(query[0].square)

# clean_logits_square_all = []
# corrupted_logits_square_all = []

# i = 0
# batch = positions[i:i + batch_size]

# legal_moves_batch = get_legal_moves_batch(batch)
# batch_tensor, batch_indices, last_token_indices = right_pad(batch, device=device) 

# clean_logits, clean_logits_square, clean_probs_square = no_ablation(
#     model, 
#     batch_tensor, 
#     batch_indices,
#     last_token_indices,
#     legal_square_id,
# )

# # def zero_ablation(
# #     model,
# #     batch_tensor,
# #     batch_indices,
# #     last_token_indices,
# #     legal_square_id,
# #     neurons,
# # ):
# #     with model.trace(batch_tensor):
# #         for layer in range(1, model.cfg.n_layers):
# #             if neurons[layer]:
# #                 neuron_indices = t.tensor(neurons[layer], device=device)
# #                 n_neurons = len(neurons[layer])
# #                 batch_indices_repeated = einops.repeat(
# #                     batch_indices,
# #                     'batch -> batch neurons',
# #                     neurons=n_neurons,
# #                 )
# #                 last_token_indices_repeated = einops.repeat(
# #                     last_token_indices,
# #                     'batch -> batch neurons',
# #                     neurons=n_neurons,
# #                 )
# #                 neuron_indices_repeated = einops.repeat(
# #                     neuron_indices,
# #                     'neurons -> batch neurons',
# #                     batch=len(batch_tensor),
# #                 )
# #                 model.blocks[layer].mlp.hook_post.output[
# #                     batch_indices_repeated, 
# #                     :, 
# #                     neuron_indices_repeated
# #                 ] = 0
        
# #         logits = model.unembed.output[batch_indices, last_token_indices].save()
# #         probs = t.nn.functional.softmax(logits, dim=-1)
        
# #         logits_square = logits[:, legal_square_id].save()
# #         probs_square = probs[:, legal_square_id].save()

# #         return logits, logits_square, probs_square
    
# corrupted_logits, corrupted_logits_square, corrupted_probs_square = zero_ablation(
#     model, 
#     batch_tensor, 
#     batch_indices,
#     last_token_indices,
#     legal_square_id,
#     neurons,
# )

# clean_logits_square_all.append(clean_logits_square.cpu())
# corrupted_logits_square_all.append(corrupted_logits_square.cpu())

# # clean_logits_square_all = t.cat(clean_logits_square_all, dim=0)
# # corrupted_logits_square_all = t.cat(corrupted_logits_square_all, dim=0)

# # %%
# valid_move_square_mask = mask_filter.to(device)
# token_id=arena_utils.to_id(blank_square)

# logits_clean_BLV, logits_patch_BLV = neuron_intervention(
#     model,
#     layers_neurons=neurons,
#     game_batch_BL=board_seqs_id_filter,
#     ablation_method="zero",
# )

# valid_move_square_mask_bool = valid_move_square_mask.to(dtype=bool)
# kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)

# logits_clean_rank_token = (logits_clean_BLV > logits_clean_BLV[...,token_id].unsqueeze(-1)).sum(-1)
# clean_total = valid_move_square_mask.sum()
# clean_correct = ((logits_clean_rank_token < valid_move_number_filter) * valid_move_square_mask).sum()
# clean_accuracy_topk = clean_correct / clean_total

# logits_patch_rank_token = (logits_patch_BLV > logits_patch_BLV[...,token_id].unsqueeze(-1)).sum(-1)
# patch_total = valid_move_square_mask.sum()
# patch_correct = ((logits_patch_rank_token < valid_move_number_filter) * valid_move_square_mask).sum()
# patch_accuracy_topk = patch_correct / patch_total

# ave_logit_diff = (logits_clean_BLV - logits_patch_BLV)[...,token_id][valid_move_square_mask_bool].mean()

# # (logits_clean_BLV - logits_patch_BLV)[...,token_id][valid_move_square_mask_bool]

# logits_clean_BLV_test = logits_clean_BLV[...,token_id][valid_move_square_mask_bool]
# logits_patch_BLV_test = logits_patch_BLV[...,token_id][valid_move_square_mask_bool]

# %%
game_index_list, move_index_list = t.where(mask_i)
# game_index = 42
# move = 42

idx = 116

game_index = game_index_list[idx].item()
move = move_index_list[idx].item()

focus_games_id = board_seqs_id[game_index].unsqueeze(0)  # [1, 59]
focus_games_square = board_seqs_square[game_index].unsqueeze(0)  # [1, 59]

# focus_board_states = board_states[game_index].unsqueeze(0)  # [1, 59, 8, 8]
focus_legal_moves = legal_moves[game_index].unsqueeze(0)  # [1, 59, 8, 8]
# focus_legal_moves_annotation = legal_moves_annotation[game_index]

focus_legal_moves_weighted = focus_legal_moves / focus_legal_moves.sum(dim=(-2, -1), keepdim=True)  # [1, 59, 1, 1]

fig = arena_utils.plot_board_values(
    board_states[game_index, move],
    width=500,
    title=f"After move {move}, {'white' if move % 2 == 0 else 'black'} to play",
    text=np.where(to_numpy(legal_moves[game_index, move]), "o", "").tolist(),
)

fig.write_image(f"figures/board2/game{game_index}_move{move}.png")

# %%
def find_binary_dt_neurons_for_query(query: list[ConditionBinary]) -> dict[int, list[int]]:
    pass

def rule_to_binary_conditions(rule: str) -> list[ConditionBinary]:
    conditions = rule.split(" AND ")
    binary_conditions = []
    for condition in conditions:
        if "NOT" in condition:
            condition = condition.replace("NOT ", "")
            operator = 'NOT'
        else:
            operator = ''
        
        condition = condition.replace("(", "").replace(")", "")
        square, feature = condition[:2], condition[3:]
        binary_conditions.append(ConditionBinary(square=square, feature=feature, operator=operator))
    
    return binary_conditions

# %%
f1_threshold=0.0
ablation_method = "zero"

# query = intervention_query
dt_rules = extract_rules_features_from_dt(
    n_layers,
    n_neurons, 
    binary_decision_trees,
    binary_custom_function_name,
    binary_feature_names,
    f1_threshold=f1_threshold,
)

# %%
# layer = 5
# neuron = 766

dt_queries = [
    [
        ConditionBinary(square='C0', feature='empty', operator=''),
        ConditionBinary(square='D1', feature='theirs', operator=''),
        ConditionBinary(square='E2', feature='mine', operator=''),
    ],
]

dt_neurons_queries = defaultdict(list)
for i_query, dt_query in enumerate(dt_queries):
    for layer in dt_rules:
        for neuron in dt_rules[layer]:
            rules = dt_rules[layer][neuron]['dt_rules']
            for rule in rules:
                rule_query = rule_to_binary_conditions(rule)
                if all(cond in rule_query for cond in dt_query):
                    dt_neurons_queries[layer].append(neuron)
                    rprint(f"[green]Layer {layer} Neuron {neuron}[/green] matches query {i_query} with rule: {rule}")
                    break

# %%
from cont_dt.fine_grained_intervention.intervention_corrupt import intervene, right_pad, no_ablation, zero_ablation, get_legal_moves_batch, is_accurate_batch, below_threshold, merge_dicts, find_neurons_for_query
from tqdm import tqdm
import arena_utils as neel_utils

@dataclass(frozen=True) 
class Condition:
    feature_name: str
    operator: Literal['<=', '>']
    threshold: float

dt_queries_cont = [
    [
        Condition(feature_name='C0 blank', operator='>', threshold=-1),
        Condition(feature_name='D1 mine-theirs', operator='<=', threshold=1),
        # Condition(feature_name='E2 mine-theirs', operator='>', threshold=-1),
    ],
]

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

neurons = merge_dicts([find_neurons_for_query(query) for query in dt_queries_cont])

# %%
neuron_filtered = filter_neurons(neurons, w_out, W_U, token_id=arena_utils.to_id(blank_square), threshold=0.0)

# %%
metric_i = calculate_ablation_scores_square_all(
    model,
    layers_neurons=neurons,
    board_seqs_id=board_seqs_id.to(device),
    valid_move_square_mask=mask_i.to(device),
    valid_move_number=valid_move_number.to(device),
    token_id=arena_utils.to_id(blank_square),
    ablation_method=ablation_method,
)
metric_c = calculate_ablation_scores_square_all(
    model,
    layers_neurons=neurons,
    board_seqs_id=board_seqs_id.to(device),
    valid_move_square_mask=mask_c.to(device),
    valid_move_number=valid_move_number.to(device),
    token_id=arena_utils.to_id(blank_square),
    ablation_method=ablation_method,
)
from helper_fns import InterventionMetrics
from cont_dt.fine_grained_intervention.intervention_corrupt import print_table
print_table(metric_i, metric_c)


# %%
# token_id=arena_utils.to_id(blank_square)
# threshold=0.1

valid_move_square_mask = mask_i.to(device)
token_id=arena_utils.to_id(blank_square)
layers_neurons={5: [1568, 386, 237, 1422, 466, 1522, 255]}

logits_clean_BLV, logits_patch_BLV = neuron_intervention(
    model,
    layers_neurons=layers_neurons,
    game_batch_BL=board_seqs_id,
    ablation_method="zero",
)

valid_move_square_mask_bool = valid_move_square_mask.to(dtype=bool)
kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)

logits_clean_rank_token = (logits_clean_BLV > logits_clean_BLV[...,token_id].unsqueeze(-1)).sum(-1)
clean_total = valid_move_square_mask.sum()
clean_correct = ((logits_clean_rank_token < valid_move_number) * valid_move_square_mask).sum()
clean_accuracy_topk = clean_correct / clean_total

logits_patch_rank_token = (logits_patch_BLV > logits_patch_BLV[...,token_id].unsqueeze(-1)).sum(-1)
patch_total = valid_move_square_mask.sum()
patch_correct = ((logits_patch_rank_token < valid_move_number) * valid_move_square_mask).sum()
patch_accuracy_topk = patch_correct / patch_total

ave_logit_diff = (logits_clean_BLV - logits_patch_BLV)[...,token_id][valid_move_square_mask_bool].mean()

(logits_clean_BLV - logits_patch_BLV)[...,token_id][valid_move_square_mask_bool]

# logits_clean_BLV_sm = logits_clean_BLV.softmax(dim=-1)[...,token_id]
# logits_patch_BLV_sm = logits_patch_BLV.softmax(dim=-1)[...,token_id]

# clean_flat = logits_clean_BLV_sm[valid_move_square_mask_bool]
# patch_flat = logits_patch_BLV_sm[valid_move_square_mask_bool]
# valid_move_number_flat = valid_move_number[valid_move_square_mask_bool]

# ave_prob_diff = (clean_flat - patch_flat).mean()

# play_total = valid_move_square_mask_bool.sum()

# below_1_percent_corrupted = (patch_flat < 1 / valid_move_number_flat * 0.01).sum() / play_total
# below_5_percent_corrupted = (patch_flat < 1 / valid_move_number_flat * 0.05).sum() / play_total
# below_10_percent_corrupted = (patch_flat < 1 / valid_move_number_flat * 0.1).sum() / play_total

# %%
# positions=intervention_positions_encoded
# query=intervention_query
# neurons = {layer: [] for layer in range(1, n_layers)}
# # neurons[5].append(1393)
# batch_size = 128

# print(f"Ablating {sum(len(neurons) for neurons in neurons.values())} neurons")
# legal_square_id = neel_utils.to_id(query[0].square)

# total_logit_diff = 0
# total_prob_diff = 0
# total_clean_accuracy = 0
# total_corrupted_accuracy = 0
# total_below_1_percent = 0
# total_below_5_percent = 0
# total_below_10_percent = 0

# clean_logits_square_all = []
# corrupted_logits_square_all = []
# for i in tqdm(range(0, len(positions), batch_size), desc="Batches"):
#     batch = positions[i:i + batch_size]

#     legal_moves_batch = get_legal_moves_batch(batch)
#     batch_tensor, batch_indices, last_token_indices = right_pad(batch, device=device) 

#     clean_logits, clean_logits_square, clean_probs_square = no_ablation(
#         model, 
#         batch_tensor, 
#         batch_indices,
#         last_token_indices,
#         legal_square_id,
#     )

#     corrupted_logits, corrupted_logits_square, corrupted_probs_square = zero_ablation(
#         model, 
#         batch_tensor, 
#         batch_indices,
#         last_token_indices,
#         legal_square_id,
#         neurons,
#     )

#     clean_logits_square_all.append(clean_logits_square.cpu())
#     corrupted_logits_square_all.append(corrupted_logits_square.cpu())
    
#     total_logit_diff += (clean_logits_square - corrupted_logits_square).sum().item()
#     total_prob_diff += (clean_probs_square - corrupted_probs_square).sum().item()

# avg_logit_diff = total_logit_diff / len(positions)
# avg_prob_diff = total_prob_diff / len(positions)

# clean_logits_square_all = t.cat(clean_logits_square_all, dim=0)
# corrupted_logits_square_all = t.cat(corrupted_logits_square_all, dim=0)


# %%
topk_neurons_seperate = {}
i_k = 0
for layer in dt_neurons_queries:
    for neuron in dt_neurons_queries[layer]:
        topk_neurons_seperate[i_k] = (layer, neuron)
        i_k += 1

# topk_neurons_seperate = {}
# i_k = 0
# for layer in neurons:
#     for neuron in neurons[layer]:
#         topk_neurons_seperate[i_k] = (layer, neuron)
#         i_k += 1

# topk_neurons_seperate = {0: (5, 766)}

probe_dict = {i : t.load(
    f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
)['linear_probe'].squeeze() for i in range(model.cfg.n_layers)}

probe_t = t.stack([probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]
blank_probe = probe_t[..., 1] - (probe_t[..., 0] + probe_t[..., 2]) * 0.5  # [layer, d_model, row, col]
my_probe = probe_t[..., 0] - probe_t[..., 2]  # [layer, d_model, row, col]

blank_probe_normalized = blank_probe / blank_probe.norm(dim=1, keepdim=True)
my_probe_normalized = my_probe / my_probe.norm(dim=1, keepdim=True)
blank_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

w_in_accu_blank = 0
w_in_accu_my = 0
for i_k, (layer, neuron) in topk_neurons_seperate.items():
    if i_k >= 16:
        break
    print(f"Rank {i_k}: L{layer}N{neuron}")

    w_in_LN_blank = calculate_neuron_input_weights(model, blank_probe_normalized[layer], layer, neuron)
    w_in_LN_my = calculate_neuron_input_weights(model, my_probe_normalized[layer], layer, neuron)

    w_out_LN_blank = calculate_neuron_output_weights(model, blank_probe_normalized[layer], layer, neuron)
    w_out_LN_my = calculate_neuron_output_weights(model, my_probe_normalized[layer], layer, neuron)

    # w_in_accu_blank += w_in_LN_blank
    # w_in_accu_my += w_in_LN_my

    # fig = arena_utils.plot_board_values(
    #     t.stack(
    #         [w_in_LN_blank, w_in_LN_my, w_in_accu_blank/(i_k+1), w_in_accu_my/(i_k+1)],
    #     ),
    #     title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
    #     board_titles=[
    #         f"Blank In (Rank {i_k}: L{layer}N{neuron})", f"My In (Rank {i_k}: L{layer}N{neuron})",
    #         f"Blank In (Mean of top 0 - top {i_k} neurons)", f"My In (Mean of top 0 - top {i_k} neurons)"
    #     ],
    #     boards_per_row=2,
    #     width=650,
    #     height=380*2,
    # )
    # fig.write_image(f"figures/probe/neuron_input_weights_rank_{i_k}_L{layer}N{neuron}.png")

    fig = arena_utils.plot_board_values(
        t.stack(
            [w_in_LN_blank, w_in_LN_my, w_out_LN_blank, w_out_LN_my],
        ),
        title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
        board_titles=[
            f"Blank In (L{layer}N{neuron})", f"My In (L{layer}N{neuron})",
            f"Blank Out (L{layer}N{neuron})", f"My Out (L{layer}N{neuron})",
        ],
        boards_per_row=4,
        width=650*2,
        height=380,
    )
    # fig.write_image(f"figures/binary_dt_neurons_3way/neuron_L{layer}N{neuron}.png")

# %%
topk_neurons_seperate = {}
i_k = 0
for layer in neurons:
    for neuron in neurons[layer]:
        topk_neurons_seperate[i_k] = (layer, neuron)
        i_k += 1

w_out = model.W_out.detach().clone() # [layer, neuron, d_model]
W_U = model.W_U[:, 1:].detach().clone()  # [d_model, 60]

write_attribution = einops.einsum(
    w_out,
    W_U,
    "layer neuron d_model, d_model id -> layer neuron id",
)

write_attribution_square = t.zeros((n_layers, n_neurons, 8, 8), device=device, dtype=t.float32)
write_attribution_square.flatten(start_dim=-2, end_dim=-1)[..., ALL_SQUARES] = write_attribution

for i_k, (layer, neuron) in topk_neurons_seperate.items():
    if i_k >= 16:
        break
    print(f"Rank {i_k}: L{layer}N{neuron}")

    w_out_unembedding = write_attribution_square[layer, neuron]  # [8, 8]

    fig = arena_utils.plot_board_values(
        t.stack(
            [w_out_unembedding],
        ),
        board_titles=[
            f"Out @ unembedding (L{layer}N{neuron})",
        ],
        boards_per_row=1,
        width=650/2,
        height=380,
    )
    fig.write_image(f"figures/cont_dt_neurons_5/neuron_L{layer}N{neuron}.png")

# %%
@dataclass(frozen=True) 
class Condition:
    feature_name: str
    operator: Literal['<=', '>']
    threshold: float

# def get_feature_indices(query: list[Condition]) -> list[Tuple[int, int, int]]:
#     indices = []
#     for condition in query:
#         feature_name = condition.feature_name
#         operator = condition.operator

#         square, feature = feature_name.split()
#         row, col = list(square)
#         row = ord(row) - ord('A')
#         col = int(col)
        
#         mode = None
#         if feature == 'blank':
#             mode = 1
#         else:
#             if operator == '>':
#                 mode = 0
#             else:
#                 mode = 2

#         indices.append((row, col, mode))

#     return indices

# intervention_query = [
#     Condition(feature_name='C0 blank', operator='>', threshold=-1),
#     Condition(feature_name='D1 mine-theirs', operator='<=', threshold=1),
#     Condition(feature_name='E2 mine-theirs', operator='>', threshold=-1),
# ]

dt_queries = [
    [Condition(feature_name='C0 blank', operator='>', threshold=-1), Condition(feature_name='D1 mine-theirs', operator='<=', threshold=1)],
]

# control_query = [
#     Condition(feature_name='C0 blank', operator='>', threshold=-1),
#     Condition(feature_name='C1 mine-theirs', operator='<=', threshold=1),
#     Condition(feature_name='C2 mine-theirs', operator='>', threshold=-1),
# ]

# intervention_indices = get_feature_indices(intervention_query)
# control_indices = get_feature_indices(control_query)

# features = test_data["games_batch_to_state_stack_mine_yours_BLRRC"]

# intervention_mask = (features[..., *(intervention_indices[0])] == 1) & (features[..., *(intervention_indices[1])] == 1) & (features[..., *(intervention_indices[2])] == 1)
# control_mask = (features[..., *(control_indices[0])] == 1) & (features[..., *(control_indices[1])] == 1) & (features[..., *(control_indices[2])] == 1)

# mask2 = intervention_mask & (~control_mask)

# %%
# n_layers = model.cfg.n_layers
# n_neurons = model.cfg.d_mlp

# w_out = model.W_out.detach().clone() # [layer, neuron, d_model]
# # w_out_nomalized = w_out / w_out.norm(dim=-1, keepdim=True)
# W_U = model.W_U[:, 1:].detach().clone()  # [d_model, 60]
# # W_U_normalized = W_U / W_U.norm(dim=0, keepdim=True)

# write_attribution = einops.einsum(
#     w_out,
#     W_U,
#     "layer neuron d_model, d_model id -> layer neuron id",
# )

# write_attribution_square = t.zeros((n_layers, n_neurons, 8, 8), device=device, dtype=t.float32)
# write_attribution_square.flatten(start_dim=-2, end_dim=-1)[..., ALL_SQUARES] = write_attribution
