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

# from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import graphviz

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
    calculate_ablation_scores_square_probability,
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
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = utils.get_model(model_name, device)

# %% Load the test dataset and process
test_size = 500
custom_functions = [
    # othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
    othello_utils.games_batch_to_valid_moves_BLRRC, # (legal move)
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

# %% writing neuron
n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp

w_out = model.W_out.detach().clone() # [layer, neuron, d_model]
# w_out_nomalized = w_out / w_out.norm(dim=-1, keepdim=True)
W_U = model.W_U[:, 1:].detach().clone()  # [d_model, 60]
# W_U_normalized = W_U / W_U.norm(dim=0, keepdim=True)

write_attribution = einops.einsum(
    w_out,
    W_U,
    "layer neuron d_model, d_model id -> layer neuron id",
)

write_attribution_square = t.zeros((n_layers, n_neurons, 8, 8), device=device, dtype=t.float32)
write_attribution_square.flatten(start_dim=-2, end_dim=-1)[..., ALL_SQUARES] = write_attribution

# %% ----- ----- ----- ----- ----- ----- specific square ----- ----- ----- ----- ----- ----- %% #
# Calculate neuron attribution for a specific square
square_idx = 25
token_id = arena_utils.SQUARE_TO_ID[square_idx]
print(f"Square {square_idx} ({arena_utils.to_board_label(square_idx)}), Token ID: {token_id}")

valid_move_square_mask = legal_moves.flatten(start_dim=-2, end_dim=-1)[..., square_idx] # [game, seq]
valid_move_number = legal_moves.sum(dim=(-2,-1))  # [game, seq]

neuron_attribution = {}
with t.no_grad(), model.trace(board_seqs_id, scan=False, validate=False):
    for layer in range(n_layers):
        neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output
        write_attr_l = write_attribution_square[layer].flatten(start_dim=-2, end_dim=-1)[..., square_idx] # [neuron]

        neuron_attr = einops.einsum(
            neuron_activations_BLD,
            write_attr_l,
            "game seq neuron, neuron -> game seq neuron",
        )
        # Normalize neuron attribution

        neuron_attribution[layer] = neuron_attr.save()

neuron_attribution = t.stack(
    [neuron_attribution[layer] for layer in range(n_layers)], dim=-2
)  # [game, seq, layer, neuron]

neuron_attribution *= valid_move_square_mask[..., None, None]
neuron_attribution = neuron_attribution.sum(dim=(0, 1))  # [layer, neuron]

# %% ablation with topk neurons from a layer
# layers = [_ for _ in range(n_layers)]
# topk_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# ablation_method = "mean"
# # topk_list = [16]
# # topk_neurons = defaultdict(lambda: defaultdict(list))
# # randk_neurons = defaultdict(lambda: defaultdict(list))
# topk_scores = defaultdict(lambda: defaultdict(dict))
# randk_scores = defaultdict(lambda: defaultdict(dict))
# for topk in topk_list:
#     for layer in layers:
#         # Get the top-k neurons for the current layer
#         topk_neuron_idx = t.topk(neuron_attribution[layer].flatten(), k=topk).indices
#         # topk_neurons[topk][layer] = topk_neuron_idx
        
#         kl_div_BL, clean_accuracy, patch_accuracy = calculate_ablation_scores_square(
#             model,
#             layers_neurons={layer: topk_neuron_idx},
#             board_seqs_id=board_seqs_id.to(device),
#             valid_move_square_mask=valid_move_square_mask.to(device),
#             valid_move_number=valid_move_number.to(device),
#             token_id=token_id,
#             ablation_method=ablation_method,
#         )
#         topk_scores[topk][layer] = {
#             "kl_div_BL": kl_div_BL,
#             "clean_accuracy": clean_accuracy,
#             "patch_accuracy": patch_accuracy,
#         }

#         t.manual_seed(layer)
#         randk_neuron_idx = t.randperm(n_neurons)[:topk]
#         # randk_neurons[topk][layer] = randk_neuron_idx

#         randk_kl_div_BL, randk_clean_accuracy, randk_patch_accuracy = calculate_ablation_scores_square(
#             model,
#             layers_neurons={layer: randk_neuron_idx},
#             board_seqs_id=board_seqs_id.to(device),
#             valid_move_square_mask=valid_move_square_mask.to(device),
#             valid_move_number=valid_move_number.to(device),
#             token_id=token_id,
#             ablation_method=ablation_method,
#         )
#         randk_scores[topk][layer] = {
#             "kl_div_BL": randk_kl_div_BL,
#             "clean_accuracy": randk_clean_accuracy,
#             "patch_accuracy": randk_patch_accuracy,
#         }

# fig, ax = plt.subplots(2,4, figsize=(20, 8))
# axes = ax.flatten()
# for layer in layers:
#     kl_list_topk = [layer_info[layer]["kl_div_BL"] for _, layer_info in topk_scores.items()]
#     kl_list_randk = [layer_info[layer]["kl_div_BL"] for _, layer_info in randk_scores.items()]
#     axes[layer].plot(topk_list, kl_list_topk, label="Top-k", marker='o')
#     axes[layer].plot(topk_list, kl_list_randk, label="Random-k", marker='x')
#     axes[layer].set_title(f"Layer {layer}")
#     axes[layer].set_xlabel("Top-k Neurons")
#     axes[layer].set_ylabel("KL Divergence")
#     axes[layer].set_xscale("log", base=2)
# axes[-1].legend()
# fig.suptitle(f"KL Divergence for Top-k and Random-k Neurons\nSquare {square_idx} ({arena_utils.to_board_label(square_idx)}), Token ID: {token_id}")
# fig.tight_layout()
# fig.savefig(f"figures/topk_kl_divergence_layer_level_square{square_idx}_token{token_id}.png", dpi=600)

# fig, ax = plt.subplots(2,4, figsize=(20, 8))
# axes = ax.flatten()
# for layer in layers:
#     patch_acc_list_topk = [layer_info[layer]["patch_accuracy"] for _, layer_info in topk_scores.items()]
#     patch_acc_list_randk = [layer_info[layer]["patch_accuracy"] for _, layer_info in randk_scores.items()]
#     axes[layer].plot(topk_list, patch_acc_list_topk, label="Top-k", marker='o')
#     axes[layer].plot(topk_list, patch_acc_list_randk, label="Random-k", marker='x')
#     axes[layer].set_title(f"Layer {layer}")
#     axes[layer].set_xlabel("Top-k Neurons")
#     axes[layer].set_ylabel("Patch Accuracy")
#     axes[layer].set_xscale("log", base=2)
# axes[-1].legend()
# fig.suptitle(f"Patch Accuracy for Top-k and Random-k Neurons\nSquare {square_idx} ({arena_utils.to_board_label(square_idx)}), Token ID: {token_id}")
# fig.tight_layout()
# fig.savefig(f"figures/topk_patch_accuracy_layer_level_square{square_idx}_token{token_id}.png", dpi=600)

# %% topk neurons across layers
topk_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

topk_neurons = defaultdict(lambda: defaultdict(list))
for topk in topk_list:
    topk_neuron_idx = t.topk(neuron_attribution.flatten(), k=topk).indices
    for idx in topk_neuron_idx:
        layer = idx // n_neurons
        neuron = idx % n_neurons
        topk_neurons[topk][layer.item()].append(neuron.item())

    topk_temp = sorted(topk_neurons[topk].items(), key=lambda kv: kv[0])
    topk_neurons[topk] = defaultdict(list, topk_temp)

# %% ----- ----- ----- ----- ----- distribution of logits (table) ----- ----- ----- ----- ----- %% #
# logits_clean_BLV, logits_patch_BLV = neuron_intervention(
#     model,
#     layers_neurons=topk_neurons[32],
#     game_batch_BL=board_seqs_id.to(device),
#     ablation_method="zero",
# )

# descending_clean_order = t.argsort(logits_clean_BLV, dim=-1, descending=True)
# descending_patch_order = t.argsort(logits_patch_BLV, dim=-1, descending=True)

# row_coord, cols_coord = t.where(valid_move_square_mask)
# t.manual_seed(42)  # For reproducibility
# rdm_idx = t.randint(int(valid_move_square_mask.sum().item()), (10,))
# # game = 0
# # seq = 4
# for i_row, i_col in zip(row_coord[rdm_idx], cols_coord[rdm_idx]):
#     game = i_row.item()
#     seq = i_col.item()

#     logits_clean_gs = logits_clean_BLV[game, seq]
#     logits_patch_gs = logits_patch_BLV[game, seq]
#     vm_num_gs = valid_move_number[game, seq]
#     descending_clean_order_gs = descending_clean_order[game, seq]
#     descending_patch_order_gs = descending_patch_order[game, seq]

#     table = Table("rank", "clean logits", "square label", "patch logits", "square label", title=f"Logits for Game {game}, Seq {seq}, target square {arena_utils.to_board_label(square_idx)} (Valid Move Number: {vm_num_gs:.0f})")
#     for i in range(int(vm_num_gs)+10):
#         clean_token = descending_clean_order_gs[i].item()
#         patch_token = descending_patch_order_gs[i].item()
#         if clean_token == token_id:
#             color_clean = "[green]"
#         else:
#             color_clean = ""
#         if patch_token == token_id:
#             color_patch = "[cyan]"
#         else:
#             color_patch = ""
#         table.add_row(
#             str(i+1),
#             color_clean+f"{logits_clean_gs[clean_token]:.4f}",
#             color_clean+arena_utils.to_board_label(arena_utils.ID_TO_SQUARE[clean_token]),
#             color_patch+f"{logits_patch_gs[patch_token]:.4f}",
#             color_patch+arena_utils.to_board_label(arena_utils.ID_TO_SQUARE[patch_token]),
#         )
#     console = Console(record=True)
#     console.print(table)
#     console.save_svg(f"figures/table/game{game}_seq{seq}_token{token_id}_patch.svg", theme=MONOKAI)


# %% ablation experiments
ablation_method="zero"
accu_threshold=0.5
topk_scores = defaultdict(dict)
randk_scores = defaultdict(lambda: defaultdict(dict))
for topk in topk_list:
    kl_div_BL, clean_accuracy, patch_accuracy = calculate_ablation_scores_square_probability(
        model,
        layers_neurons=topk_neurons[topk],
        board_seqs_id=board_seqs_id.to(device),
        valid_move_square_mask=valid_move_square_mask.to(device),
        valid_move_number=valid_move_number.to(device),
        token_id=token_id,
        ablation_method=ablation_method,
        threshold=accu_threshold,
    )
    topk_scores[topk] = {
        "kl_div_BL": kl_div_BL,
        "clean_accuracy": clean_accuracy,
        "patch_accuracy": patch_accuracy,
    }

    n_random_trial = 100
    for i_trial in range(n_random_trial):
        randk_neurons_topk = defaultdict(list)
        randk_neuron_idx = t.randperm(n_layers * n_neurons)[:topk]
        for idx in randk_neuron_idx:
            layer = idx // n_neurons
            neuron = idx % n_neurons
            randk_neurons_topk[layer.item()].append(neuron.item())
        
        randk_temp = sorted(randk_neurons_topk.items(), key=lambda kv: kv[0])
        randk_neurons_topk = defaultdict(list, randk_temp)

        randk_kl_div_BL, randk_clean_accuracy, randk_patch_accuracy = calculate_ablation_scores_square_probability(
            model,
            layers_neurons=randk_neurons_topk,
            board_seqs_id=board_seqs_id.to(device),
            valid_move_square_mask=valid_move_square_mask.to(device),
            valid_move_number=valid_move_number.to(device),
            token_id=token_id,
            ablation_method=ablation_method,
            threshold=accu_threshold,
        )
        randk_scores[topk][i_trial] = {
            "kl_div_BL": randk_kl_div_BL,
            "clean_accuracy": randk_clean_accuracy,
            "patch_accuracy": randk_patch_accuracy,
        }

# %%
randk_scores_analysis_score = defaultdict(dict)
randk_scores_analysis_score_std = defaultdict(dict)
randk_scores_analysis_score_ci95 = defaultdict(dict)
for topk in topk_list:
    randk_kl_div_BL_list = [layer_info["kl_div_BL"] for _, layer_info in randk_scores[topk].items()]
    randk_clean_accuracy_list = [layer_info["clean_accuracy"] for _, layer_info in randk_scores[topk].items()]
    randk_patch_accuracy_list = [layer_info["patch_accuracy"] for _, layer_info in randk_scores[topk].items()]

    randk_scores_analysis_score[topk] = {
        "kl_div_BL": np.mean(randk_kl_div_BL_list),
        "clean_accuracy": np.mean(randk_clean_accuracy_list),
        "patch_accuracy": np.mean(randk_patch_accuracy_list),
    }
    randk_scores_analysis_score_std[topk] = {
        "kl_div_BL": np.std(randk_kl_div_BL_list),
        "clean_accuracy": np.std(randk_clean_accuracy_list),
        "patch_accuracy": np.std(randk_patch_accuracy_list),
    }
    randk_scores_analysis_score_ci95[topk] = {
        "kl_div_BL": 1.96 * np.std(randk_kl_div_BL_list) / np.sqrt(n_random_trial),
        "clean_accuracy": 1.96 * np.std(randk_clean_accuracy_list) / np.sqrt(n_random_trial),
        "patch_accuracy": 1.96 * np.std(randk_patch_accuracy_list) / np.sqrt(n_random_trial),
    }

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
plt.rcParams['font.size'] = 12
axes = ax.flatten()
kl_list_topk = [layer_info["kl_div_BL"] for _, layer_info in topk_scores.items()]
kl_list_randk = [layer_info["kl_div_BL"] for _, layer_info in randk_scores_analysis_score.items()]
kl_list_rank_ci = [layer_info["kl_div_BL"] for _, layer_info in randk_scores_analysis_score_ci95.items()]
axes[0].plot(topk_list, kl_list_topk, label="Top-k", marker='o')
axes[0].plot(topk_list, kl_list_randk, label="Random-k", marker='x')
axes[0].fill_between(
    topk_list,
    np.array(kl_list_randk)-np.array(kl_list_rank_ci),
    np.array(kl_list_randk)+np.array(kl_list_rank_ci),
    alpha=0.2, color='gray', label="Random-k 95% CI"
)
axes[0].set_title("KL Divergence for Top-k and Random-k Neurons", fontsize=18)
axes[0].set_xlabel("Number of Neurons", fontsize=16)
axes[0].set_ylabel("KL Divergence", fontsize=16)
axes[0].set_xscale("log", base=2)
plt.xticks(fontsize=16)

patch_acc_list_topk = [layer_info["patch_accuracy"] for _, layer_info in topk_scores.items()]
patch_acc_list_randk = [layer_info["patch_accuracy"] for _, layer_info in randk_scores_analysis_score.items()]
patch_acc_list_rank_ci = [layer_info["patch_accuracy"] for _, layer_info in randk_scores_analysis_score_ci95.items()]
axes[1].plot(topk_list, patch_acc_list_topk, label="Top-k", marker='o')
axes[1].plot(topk_list, patch_acc_list_randk, label="Random-k", marker='x')
axes[1].fill_between(
    topk_list,
    np.array(patch_acc_list_randk)-np.array(patch_acc_list_rank_ci),
    np.array(patch_acc_list_randk)+np.array(patch_acc_list_rank_ci),
    alpha=0.2, color='gray', label="Random-k 95% CI"
)
axes[1].set_title("Patch Accuracy for Top-k and Random-k Neurons", fontsize=18)
axes[1].set_xlabel("Number of Neurons", fontsize=16)
axes[1].set_ylabel("Patch Accuracy", fontsize=16)
axes[1].set_xscale("log", base=2)
axes[1].legend()
plt.xticks(fontsize=16)

fig.suptitle(f"Square {square_idx} ({arena_utils.to_board_label(square_idx)}), Token ID: {token_id}", fontsize=20)
fig.tight_layout()
fig.savefig(f"figures/topk_cross_layer_square{square_idx}_token{token_id}_{ablation_method}_ablation_ci95_probability_{accu_threshold:.1f}.png", dpi=600)

# %% print topk neurons for specific square (cross-layer)
topk = 32
print(f"Top {topk} neurons for square {square_idx} ({arena_utils.to_board_label(square_idx)}), token ID {token_id}:")
for layer, neurons in topk_neurons[topk].items():
    print(f"Layer {layer}: Neurons {neurons}")

# %% ----- ----- ----- ----- ----- ----- topk neurons (layer-neuron pairs) ----- ----- ----- ----- ----- ----- %% #
topk_neurons_seperate = defaultdict(list)
topk_neuron_idx = t.topk(neuron_attribution.flatten(), k=2048).indices
for i_k, idx in enumerate(topk_neuron_idx):
    layer = idx // n_neurons
    neuron = idx % n_neurons
    topk_neurons_seperate[i_k] = [layer.item(), neuron.item()]

# %% ----- ----- ----- ----- ----- ----- probe directions ----- ----- ----- ----- ----- ----- %% #
# Load probes (probe directions (Mine, Empty, Yours))
probe_dict = {i : t.load(
    f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
)['linear_probe'].squeeze() for i in range(model.cfg.n_layers)}

probe_t = t.stack([probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]
blank_probe = probe_t[..., 1] - (probe_t[..., 0] + probe_t[..., 2]) * 0.5  # [layer, d_model, row, col]
my_probe = probe_t[..., 0] - probe_t[..., 2]  # [layer, d_model, row, col]

blank_probe_normalized = blank_probe / blank_probe.norm(dim=1, keepdim=True)
my_probe_normalized = my_probe / my_probe.norm(dim=1, keepdim=True)
blank_probe_normalized[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# w_in = model.W_in.detach().clone()  # [layer, d_model, neuron]
# w_in_nomalized = w_in / w_in.norm(dim=0, keepdim=True)

# probe_cos_sim = einops.einsum(
#     w_in,
#     blank_probe_normalized,
#     "layer d_model neuron, layer d_model row col -> layer neuron row col",
# )

# %% main linear probe
# full_linear_probe = t.load("linear_probes/main_linear_probe.pth", map_location=str(device), weights_only=True)

# black_to_play, white_to_play, _ = (0, 1, 2)
# empty, white, black = (0, 1, 2)

# linear_probe = t.stack(
#     [
#         # "Empty" direction = average of empty direction across probe modes
#         full_linear_probe[[black_to_play, white_to_play], ..., [empty, empty]].mean(0),
#         # "Theirs" direction = average of {x to play, classification != x} across probe modes
#         full_linear_probe[[black_to_play, white_to_play], ..., [white, black]].mean(0),
#         # "Mine" direction = average of {x to play, classification == x} across probe modes
#         full_linear_probe[[black_to_play, white_to_play], ..., [black, white]].mean(0),
#     ],
#     dim=-1,
# )

# blank_probe = linear_probe[..., 0] - linear_probe[..., 1] * 0.5 - linear_probe[..., 2] * 0.5
# # mine(2) - theirs(1)
# my_probe = linear_probe[..., 2] - linear_probe[..., 1]

# blank_probe_normalized = blank_probe / blank_probe.norm(dim=0, keepdim=True)
# my_probe_normalized = my_probe / my_probe.norm(dim=0, keepdim=True)

# # Set the center blank probes to 0, since they're never blank so the probe is meaningless
# blank_probe_normalized[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# %%
w_in_accu_blank = 0
w_in_accu_my = 0
for i_k, (layer, neuron) in topk_neurons_seperate.items():
    if i_k >= 16:
        break
    print(f"Rank {i_k}: L{layer}N{neuron}")

    w_in_LN_blank = calculate_neuron_input_weights(model, blank_probe_normalized[layer], layer, neuron)
    w_in_LN_my = calculate_neuron_input_weights(model, my_probe_normalized[layer], layer, neuron)

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
            [w_in_LN_blank, w_in_LN_my],
        ),
        title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
        board_titles=[
            f"Blank In (Rank {i_k}: L{layer}N{neuron})", f"My In (Rank {i_k}: L{layer}N{neuron})",
        ],
        boards_per_row=2,
        width=650,
        height=380,
    )
    fig.write_image(f"figures/probe_separate/neuron_input_weights_rank_{i_k}_L{layer}N{neuron}.png")

# %% ----- ----- ----- ----- ----- ----- decision trees ----- ----- ----- ----- ----- ----- %% #
# Load decision trees
dt_name = 'neuron_simulation/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl'
with open(dt_name, "rb") as f:
    decision_trees = pickle.load(f)

function_name = list(decision_trees[0].keys())[0]
n_features = decision_trees[0][function_name]["decision_tree"]["model"].n_features_in_
feature_names = create_feature_names(n_features, function_name)

# %%
max_depth = 3
import re
for i_k, (layer, neuron) in topk_neurons_seperate.items():
    if i_k >= 16:
        break
    tree_model, r2_score = get_neuron_decision_tree(decision_trees, layer, neuron, function_name)
    # fig, ax = plt.subplots(figsize=(20, 12))

    # plot_tree(
    #     tree_model,
    #     feature_names=feature_names,
    #     filled=True,
    #     rounded=True,
    #     fontsize=8,
    #     max_depth=max_depth
    # )
    
    # ax.set_title(f"Decision Tree (Rank {i_k}: L{layer}N{neuron})\nR² Score: {r2_score:.4f}", 
    #           fontsize=16, pad=20)
    # fig.savefig(f"figures/decision_tree/dt_layer_rank_{i_k}_L{layer}N{neuron}.png", dpi=300, bbox_inches='tight')

    dot_data = export_graphviz(
        tree_model,
        out_file=None,
        feature_names=feature_names,
        filled=True, rounded=True,
        special_characters=True,
        proportion=False,   # scale node size by samples
        max_depth=3,
        impurity=False,   # removes "mse" or "variance"
    )
    dot_data = re.sub(r'samples = \d+', '', dot_data)
    graph = graphviz.Source(dot_data)
    # graph.graph_attr.update(label=f"Decision Tree (Rank {i_k}: L{layer}N{neuron})\nR² Score: {r2_score:.4f}", labelloc='top', fontsize='16')
    # graph.render("regression_tree")  # saves PDF/PNG
    # graph
    graph.render(f"figures/decision_tree/dt_layer_rank_{i_k}_L{layer}N{neuron}", format="png", cleanup=True)
    # print(f"Saved visualization to {save_path}")
    # plt.show()

# %% ----- ----- ----- ----- ----- ----- binary decision trees ----- ----- ----- ----- ----- ----- %% #
binary_dt_name = 'neuron_simulation/decision_trees_binary/decision_trees_mlp_neuron_6000.pkl'
with open(binary_dt_name, "rb") as f:
    binary_decision_trees = pickle.load(f)

binary_function_name = list(binary_decision_trees[0].keys())[0]
n_binary_features = binary_decision_trees[0][binary_function_name]["binary_decision_tree"]["model"].n_features_in_
binary_feature_names = create_feature_names(n_binary_features, binary_function_name)

# %%
max_depth = 3
for i_k, (layer, neuron) in topk_neurons_seperate.items():
    if i_k >= 16:
        break
    tree_model, f1_score = get_neuron_binary_decision_tree(binary_decision_trees, layer, neuron, binary_function_name)
    fig, ax = plt.subplots(figsize=(20, 12))

    plot_tree(
        tree_model,
        feature_names=binary_feature_names,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth
    )
    
    ax.set_title(f"Decision Tree (Rank {i_k}: L{layer}N{neuron})\nF1 Score: {f1_score:.4f}", 
              fontsize=16, pad=20)
    fig.savefig(f"figures/decision_tree_binary/dt_layer_rank_{i_k}_L{layer}N{neuron}.png", dpi=300, bbox_inches='tight')
    # print(f"Saved visualization to {save_path}")
    # plt.show()

# %% ----- ----- ----- ----- top-k neurons ablation experiments (reproducible) ----- ----- ----- ----- %% #
# topk_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# topk_neurons = defaultdict(lambda: defaultdict(list))
# randk_neurons = defaultdict(lambda: defaultdict(list))
# for topk in topk_list:
#     topk_neuron_idx = t.topk(neuron_attribution.flatten(), k=topk).indices
#     for idx in topk_neuron_idx:
#         layer = idx // n_neurons
#         neuron = idx % n_neurons
#         topk_neurons[topk][layer.item()].append(neuron.item())

#     topk_temp = sorted(topk_neurons[topk].items(), key=lambda kv: kv[0])
#     topk_neurons[topk] = defaultdict(list, topk_temp)
    
#     t.manual_seed(topk)  # For reproducibility
#     randk_neuron_idx = t.randperm(n_layers * n_neurons)[:topk]
#     for idx in randk_neuron_idx:
#         layer = idx // n_neurons
#         neuron = idx % n_neurons
#         randk_neurons[topk][layer.item()].append(neuron.item())
    
#     randk_temp = sorted(randk_neurons[topk].items(), key=lambda kv: kv[0])
#     randk_neurons[topk] = defaultdict(list, randk_temp)

# ablation_method="mean"
# topk_scores = defaultdict(dict)
# randk_scores = defaultdict(dict)
# for topk in topk_list:
#     kl_div_BL, clean_accuracy, patch_accuracy = calculate_ablation_scores_square(
#         model,
#         layers_neurons=topk_neurons[topk],
#         board_seqs_id=board_seqs_id.to(device),
#         valid_move_square_mask=valid_move_square_mask.to(device),
#         valid_move_number=valid_move_number.to(device),
#         token_id=token_id,
#         ablation_method=ablation_method,
#     )
#     topk_scores[topk] = {
#         "kl_div_BL": kl_div_BL,
#         "clean_accuracy": clean_accuracy,
#         "patch_accuracy": patch_accuracy,
#     }

#     randk_kl_div_BL, randk_clean_accuracy, randk_patch_accuracy = calculate_ablation_scores_square(
#         model,
#         layers_neurons=randk_neurons[topk],
#         board_seqs_id=board_seqs_id.to(device),
#         valid_move_square_mask=valid_move_square_mask.to(device),
#         valid_move_number=valid_move_number.to(device),
#         token_id=token_id,
#         ablation_method=ablation_method,
#     )
#     randk_scores[topk] = {
#         "kl_div_BL": randk_kl_div_BL,
#         "clean_accuracy": randk_clean_accuracy,
#         "patch_accuracy": randk_patch_accuracy,
#     }

# fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# axes = ax.flatten()
# kl_list_topk = [layer_info["kl_div_BL"] for _, layer_info in topk_scores.items()]
# kl_list_randk = [layer_info["kl_div_BL"] for _, layer_info in randk_scores.items()]
# axes[0].plot(topk_list, kl_list_topk, label="Top-k", marker='o')
# axes[0].plot(topk_list, kl_list_randk, label="Random-k", marker='x')
# axes[0].set_title("KL Divergence for Top-k and Random-k Neurons")
# axes[0].set_xlabel("Top-k Neurons")
# axes[0].set_ylabel("KL Divergence")
# axes[0].set_xscale("log", base=2)
# patch_acc_list_topk = [layer_info["patch_accuracy"] for _, layer_info in topk_scores.items()]
# patch_acc_list_randk = [layer_info["patch_accuracy"] for _, layer_info in randk_scores.items()]
# axes[1].plot(topk_list, patch_acc_list_topk, label="Top-k", marker='o')
# axes[1].plot(topk_list, patch_acc_list_randk, label="Random-k", marker='x')
# axes[1].set_title("Patch Accuracy for Top-k and Random-k Neurons")
# axes[1].set_xlabel("Top-k Neurons")
# axes[1].set_ylabel("Patch Accuracy")
# axes[1].set_xscale("log", base=2)
# axes[1].legend()
# fig.suptitle(f"Square {square_idx} ({arena_utils.to_board_label(square_idx)}), Token ID: {token_id}")
# fig.tight_layout()
# fig.savefig(f"figures/topk_cross_layer_square{square_idx}_token{token_id}_mean_ablation.png", dpi=600)

# %% ----- ----- ----- ----- ----- ----- board plot ----- ----- ----- ----- %% #
# game_index_list, move_index_list = t.where(valid_move_square_mask)
# # game_index = 42
# # move = 42

# idx = 116

# game_index = game_index_list[idx].item()
# move = move_index_list[idx].item()

# focus_games_id = board_seqs_id[game_index].unsqueeze(0)  # [1, 59]
# focus_games_square = board_seqs_square[game_index].unsqueeze(0)  # [1, 59]

# # focus_board_states = board_states[game_index].unsqueeze(0)  # [1, 59, 8, 8]
# focus_legal_moves = legal_moves[game_index].unsqueeze(0)  # [1, 59, 8, 8]
# # focus_legal_moves_annotation = legal_moves_annotation[game_index]

# focus_legal_moves_weighted = focus_legal_moves / focus_legal_moves.sum(dim=(-2, -1), keepdim=True)  # [1, 59, 1, 1]

# fig = arena_utils.plot_board_values(
#     board_states[game_index, move],
#     width=500,
#     title=f"After move {move}, {'white' if move % 2 == 0 else 'black'} to play",
#     text=np.where(to_numpy(legal_moves[game_index, move]), "o", "").tolist(),
# )

# fig.write_image(f"figures/board/game{game_index}_move{move}.png")

# %% test
# logits_clean_BLV, logits_patch_BLV = neuron_intervention(
#     model,
#     layers_neurons=topk_neurons[32],
#     game_batch_BL=board_seqs_id.to(device),
#     ablation_method="zero",
# )

# # kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)

# logits_clean_BLV_sm = logits_clean_BLV.softmax(dim=-1)[...,token_id]
# logits_patch_BLV_sm = logits_patch_BLV.softmax(dim=-1)[...,token_id]

# clean_flat = logits_clean_BLV_sm[valid_move_square_mask.to(dtype=bool)]
# patch_flat = logits_patch_BLV_sm[valid_move_square_mask.to(dtype=bool)]
# valid_move_number_flat = valid_move_number[valid_move_square_mask.to(dtype=bool)]

# play_total = valid_move_square_mask.sum()
# clean_accuracy = (clean_flat > 1 / valid_move_number_flat * 0.1).sum() / play_total
# patch_accuracy = (patch_flat > 1 / valid_move_number_flat * 0.1).sum() / play_total