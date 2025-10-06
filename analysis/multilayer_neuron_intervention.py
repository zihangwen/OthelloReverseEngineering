# %%
import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path
import gzip

from tqdm import tqdm
import einops
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as t
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(BASE_PATH)
BASE_PATH = Path(BASE_PATH)
os.chdir(BASE_PATH)

import utils.circuits_utils as circuits_utils
from utils.helper_fns import (
    dt_simulation_neuron_activation,
    neuron_intervention,
    compute_top_n_accuracy,
    compute_kl_divergence,
)
import utils.othello_utils as othello_utils

from decision_trees import dtypes
# import decision_trees
sys.modules['dtypes'] = dtypes

# %%
# device = "cuda" if t.cuda.is_available() else "cpu"
device = "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = circuits_utils.get_model(model_name, device)

n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp

# %%
# binary_decision_tree_dict = defaultdict(dict)
# binary_dt_f1 = defaultdict(dict)
# for layer in range(n_layers):
#     gt_class_path = (
#         BASE_PATH
#         / "decision_trees"
#         / "ground_truth_features"
#         / "classification"
#         / "results"
#         / f"layer_{layer}_trees.pkl.gz"
#     )
#     with gzip.open(gt_class_path, "rb") as f:
#         gt_feature_classifiers = pickle.load(f)
    
#     for gt_binary in gt_feature_classifiers:
#         try:
#             check_is_fitted(gt_binary.tree)
#             binary_decision_tree_dict[gt_binary.layer][gt_binary.neuron] = gt_binary.tree
#             binary_dt_f1[gt_binary.layer][gt_binary.neuron] = gt_binary.test_F1
#         except NotFittedError:
#             print(f"Tree L{gt_binary.layer}N{gt_binary.neuron} is NOT fitted")
#             continue

# f1_threshold = 0.7
# binary_dt_f1_filter_dict = {layer: {neuron: score for neuron, score in scores.items() if score >= f1_threshold} for layer, scores in binary_dt_f1.items()}
# binary_dt_filter_layer_neurons = {layer: list(scores.keys()) for layer, scores in binary_dt_f1_filter_dict.items()}

# %%
reg_decision_tree_dict = defaultdict(dict)
reg_dt_r2 = defaultdict(dict)

for layer in range(n_layers):
    gt_reg_path = (
        BASE_PATH
        / "decision_trees"
        / "ground_truth_features"
        / "regression"
        / "results"
        / f"layer_{layer}_trees.pkl.gz"
    )
    with gzip.open(gt_reg_path, "rb") as f:
        gt_feature_regressors = pickle.load(f)
    
    for gt_reg in gt_feature_regressors:
        try:
            check_is_fitted(gt_reg.tree)
            reg_decision_tree_dict[layer][gt_reg.neuron] = gt_reg.tree
            reg_dt_r2[layer][gt_reg.neuron] = gt_reg.test_R2
        except NotFittedError:
            print(f"Tree L{layer}N{gt_reg.neuron} is NOT fitted")
            continue

r2_threshold = 0.7
reg_dt_r2_filter_dict = {layer: {neuron: score for neuron, score in scores.items() if score >= r2_threshold} for layer, scores in reg_dt_r2.items()}
reg_dt_filter_layer_neurons = {layer: list(scores.keys()) for layer, scores in reg_dt_r2_filter_dict.items()}

# %%
test_size = 500
custom_functions = [
    # othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
    othello_utils.games_batch_to_board_state_flipped_played_BLC, # (legal move)
    othello_utils.games_batch_to_valid_moves_BLRRC,
]
test_data = circuits_utils.construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=test_size,
    split="test", 
    device=device,
)

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)
valid_moves_BLRRC = t.tensor(test_data["games_batch_to_valid_moves_BLRRC"]).long().to(device)
# board_seqs_square = t.tensor(test_data["decoded_inputs"]).long().to(device)

# %%
func_name = othello_utils.games_batch_to_board_state_flipped_played_BLC.__name__
simulated_acts = dt_simulation_neuron_activation(test_data, reg_decision_tree_dict, n_layers, n_neurons, func_name, device)

# %%
n_layers_ablate = 7
layer_slice_list = []
start = 0
for end in range(start, n_layers_ablate):
    layer_slice = list(range(start, end + 1))
    layer_slice_list.append(layer_slice)

end = 6
for start in range(1, n_layers_ablate):
    layer_slice = list(range(start, end + 1))
    layer_slice_list.append(layer_slice)

# %%
score_all = list()
for i_slice, layer_slice in enumerate(tqdm(layer_slice_list)):
    layer_neurons = {
        layer: neurons
        for layer, neurons in reg_dt_filter_layer_neurons.items() if layer in layer_slice
    }

    logits_clean_BLV, logits_zero_BLV = neuron_intervention(
        model,
        layer_neurons,
        board_seqs_id,
        ablation_method="zero",
    )

    clean_scores = compute_top_n_accuracy(logits_clean_BLV, valid_moves_BLRRC)
    zero_scores = compute_top_n_accuracy(logits_zero_BLV, valid_moves_BLRRC)
    zero_kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_zero_BLV)

    _, logits_dt_BLV = neuron_intervention(
        model,
        layer_neurons,
        board_seqs_id,
        ablation_method="dt",
        simulated_acts=simulated_acts,
    )
    dt_ablation_scores = compute_top_n_accuracy(logits_dt_BLV, valid_moves_BLRRC)
    dt_kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_dt_BLV)

    _, logits_mean_BLV = neuron_intervention(
        model,
        layer_neurons,
        board_seqs_id,
        ablation_method="mean",
    )
    mean_ablation_scores = compute_top_n_accuracy(logits_mean_BLV, valid_moves_BLRRC)
    mean_kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_mean_BLV)

    score_all.append({
        "layer_slice": layer_slice,
        "clean_scores": clean_scores,
        "zero_scores": zero_scores,
        "dt_scores": dt_ablation_scores,
        "mean_scores": mean_ablation_scores,
        "zero_kl_div": zero_kl_div_BL.mean().item(),
        "dt_kl_div": dt_kl_div_BL.mean().item(),
        "mean_kl_div": mean_kl_div_BL.mean().item(),
    })

# %%
# for i, _ in enumerate(score_all):
#     score_all[i]["zero_kl_div"] = score_all[i]["zero_kl_div"].mean().item()
#     score_all[i]["dt_kl_div"] = score_all[i]["dt_kl_div"].mean().item()
#     score_all[i]["mean_kl_div"] = score_all[i]["mean_kl_div"].mean().item()

# %%
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2)
ax_acc = fig.add_subplot(gs[0, 0])
ax_kl = fig.add_subplot(gs[0, 1])

x = np.arange(len(layer_slice_list))
width = 0.2
# Accuracy
ax_acc.plot(x, [s["zero_scores"][-1] for s in score_all], marker="o", label="Zero Ablation")
ax_acc.plot(x, [s["dt_scores"][-1] for s in score_all], marker="o", label="DT Ablation")
ax_acc.plot(x, [s["mean_scores"][-1] for s in score_all], marker="o", label="Mean Ablation")
ax_acc.plot(x, [s["clean_scores"][-1] for s in score_all], color="black", marker="o", linestyle='--', label="Clean")
ax_acc.axvline(x=4, color='gray', linestyle='--')
ax_acc.axvline(x=11, color='gray', linestyle='--')
ax_acc.set_xlabel("Layer Slice")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Ablation Accuracy by Layer Slice")
ax_acc.set_xticks(x, [str(s["layer_slice"]) for s in score_all], rotation=90)
ax_acc.legend()

# KL Divergence
ax_kl.plot(x, [s["zero_kl_div"] for s in score_all], marker="o", label="Zero Ablation")
ax_kl.plot(x, [s["dt_kl_div"] for s in score_all], marker="o", label="DT Ablation")
ax_kl.plot(x, [s["mean_kl_div"] for s in score_all], marker="o", label="Mean Ablation")
ax_kl.plot(x, [0]*len(score_all), color="black", marker="o", linestyle='--', label="Clean")
ax_kl.axvline(x=4, color='gray', linestyle='--')
ax_kl.axvline(x=11, color='gray', linestyle='--')
ax_kl.set_xlabel("Layer Slice")
ax_kl.set_ylabel("KL Divergence")
ax_kl.set_title("Ablation KL Divergence by Layer Slice")
ax_kl.set_xticks(x, [str(s["layer_slice"]) for s in score_all], rotation=90)
# ax_kl.legend()

plt.tight_layout()
# plt.show()
plt.savefig(f"figures/ablation_analysis/ablation_analysis_all_methods.pdf", dpi=300, bbox_inches='tight')



# %%
# board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)
# board_seqs_square = t.tensor(test_data["decoded_inputs"]).long().to(device)

# board_states, legal_moves, legal_moves_annotation = get_board_states_and_legal_moves(board_seqs_square)
# legal_moves = legal_moves.to(device=device, dtype=t.float32)

# valid_move_square_mask = legal_moves.flatten(start_dim=-2, end_dim=-1)[..., square_idx] # [game, seq]
# valid_move_number = legal_moves.sum(dim=(-2,-1))  # [game, seq]

# square_idx = 25
# zero_ablation_scores = calculate_ablation_scores_square_probability(
#     model,
#     layer_neurons,
#     board_seqs_id,
#     valid_move_square_mask,
#     valid_move_number,
#     token_id=square_idx,
#     ablation_method="zero",
#     threshold=0.1,
# )
# dt_ablation_scores = calculate_ablation_scores_square_probability(
#     model,
#     layer_neurons,
#     board_seqs_id,
#     valid_move_square_mask,
#     valid_move_number,
#     token_id=square_idx,
#     ablation_method="dt",
#     threshold=0.1,
#     simulated_acts=simulated_acts,
# )
# dt_ablation_scores = calculate_ablation_scores_square_probability(
#     model,
#     layer_neurons,
#     board_seqs_id,
#     valid_move_square_mask,
#     valid_move_number,
#     token_id=square_idx,
#     ablation_method="dt",
#     threshold=0.1,
#     simulated_acts=simulated_acts,
# )
# %%
