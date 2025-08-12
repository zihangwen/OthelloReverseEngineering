# %%
import pickle
from collections import defaultdict
import torch as t
import numpy as np
import einops
# from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
# from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import to_numpy, get_act_name
# from torch import Tensor
# from IPython.display import HTML, display
# from jaxtyping import Bool, Float, Int
import neuron_simulation.arena_utils as arena_utils

from helper_fns import (
    # MIDDLE_SQUARES,
    ALL_SQUARES,
    get_board_states_and_legal_moves,
    calculate_ablation_scores_game_move,
    calculate_ablation_scores_square,
    # plot_probe_outputs,
    # get_w_in,
    # get_w_out,
    # calculate_neuron_input_weights,
    # calculate_neuron_output_weights,
    # create_feature_names,
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

# %% Load probes
# probe_dict = {i : t.load(
#     f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
# )['linear_probe'].squeeze() for i in range(model.cfg.n_layers)}

# %% Load decision trees
# dt_name = 'neuron_simulation/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl'
# with open(dt_name, "rb") as f:
#     decision_trees = pickle.load(f)

# function_name = list(decision_trees[0].keys())[0]
# n_features = decision_trees[0][function_name]["decision_tree"]["model"].n_features_in_
# feature_names = create_feature_names(n_features, function_name)

# %% Load the test dataset and process
test_size = 500
custom_functions = [
    othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
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

# %% ----- ----- ----- ----- ----- ----- specific game and move ----- ----- ----- ----- ----- ----- %% #
game_index = 42
move = 42

focus_games_id = board_seqs_id[game_index].unsqueeze(0)  # [1, 59]
focus_games_square = board_seqs_square[game_index].unsqueeze(0)  # [1, 59]

# focus_board_states = board_states[game_index].unsqueeze(0)  # [1, 59, 8, 8]
focus_legal_moves = legal_moves[game_index].unsqueeze(0)  # [1, 59, 8, 8]
# focus_legal_moves_annotation = legal_moves_annotation[game_index]

focus_legal_moves_weighted = focus_legal_moves / focus_legal_moves.sum(dim=(-2, -1), keepdim=True)  # [1, 59, 1, 1]

arena_utils.plot_board_values(
    board_states[game_index, move],
    width=500,
    title=f"After move {move}, {'white' if move % 2 == 0 else 'black'} to play",
    text=np.where(to_numpy(legal_moves[game_index, move]), "o", "").tolist(),
)

neuron_attribution = {}
with t.no_grad(), model.trace(focus_games_id.to(device), scan=False, validate=False):
    for layer in range(n_layers):
        neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output
        write_attr_l = write_attribution_square[layer] # [neuron, row, col]

        neuron_attr = einops.einsum(
            neuron_activations_BLD,
            write_attr_l,
            focus_legal_moves_weighted,
            "game seq neuron, neuron row col, game seq row col -> game seq neuron",
        )
        # Normalize neuron attribution

        neuron_attribution[layer] = neuron_attr.save()

neuron_attribution = t.stack(
    [neuron_attribution[layer] for layer in range(n_layers)], dim=-2
)  # [game, seq, layer, neuron]

neuron_attribution = neuron_attribution[0, move]  # [layer, neuron]

# %% ablation with topk neurons from a layer
layers = [_ for _ in range(n_layers)]

topk_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# topk_list = [1, 2]
# topk_neurons = defaultdict(lambda: defaultdict(list))
# randk_neurons = defaultdict(lambda: defaultdict(list))

topk_scores = defaultdict(lambda: defaultdict(dict))
randk_scores = defaultdict(lambda: defaultdict(dict))
for topk in topk_list:
    # t.cuda.empty_cache()
    for layer in layers:
        # Get the top-k neurons for the current layer
        
        topk_neuron_idx = t.topk(neuron_attribution[layer].flatten(), k=topk).indices
        # topk_neurons[topk][layer] = topk_neuron_idx
        
        kl_div_BL, clean_accuracy, patch_accuracy = calculate_ablation_scores_game_move(
            model,
            layers_neurons={layer: topk_neuron_idx},
            focus_games_id=focus_games_id.to(device),
            focus_legal_moves=focus_legal_moves.to(device),
            move=move,
            ablation_method="zero",
        )
        topk_scores[topk][layer] = {
            "kl_div_BL": kl_div_BL,
            "clean_accuracy": clean_accuracy,
            "patch_accuracy": patch_accuracy,
        }

        t.manual_seed(layer)  # For reproducibility
        randk_neuron_idx = t.randperm(n_neurons)[:topk]
        # randk_neurons[topk][layer] = randk_neuron_idx

        randk_kl_div_BL, randk_clean_accuracy, randk_patch_accuracy = calculate_ablation_scores_game_move(
            model,
            layers_neurons={layer: randk_neuron_idx},
            focus_games_id=focus_games_id.to(device),
            focus_legal_moves=focus_legal_moves.to(device),
            move=move,
            ablation_method="zero",
        )
        randk_scores[topk][layer] = {
            "kl_div_BL": randk_kl_div_BL,
            "clean_accuracy": randk_clean_accuracy,
            "patch_accuracy": randk_patch_accuracy,
        }

fig, ax = plt.subplots(2,4, figsize=(20, 8))
axes = ax.flatten()
for layer in layers:
    kl_list_topk = [layer_info[layer]["kl_div_BL"] for _, layer_info in topk_scores.items()]
    kl_list_randk = [layer_info[layer]["kl_div_BL"] for _, layer_info in randk_scores.items()]
    axes[layer].plot(topk_list, kl_list_topk, label="Top-k", marker='o')
    axes[layer].plot(topk_list, kl_list_randk, label="Random-k", marker='x')
    axes[layer].set_title(f"Layer {layer}")
    axes[layer].set_xlabel("Top-k Neurons")
    axes[layer].set_ylabel("KL Divergence")
    axes[layer].set_xscale("log", base=2)
axes[-1].legend()
fig.suptitle(f"KL Divergence for Top-k and Random-k Neurons\nGame {game_index}, Move {move}")
fig.tight_layout()
fig.savefig(f"figures/topk_kl_divergence_layer_level_game{game_index}_move{move}.png", dpi=600)

fig, ax = plt.subplots(2,4, figsize=(20, 8))
axes = ax.flatten()
for layer in layers:
    patch_acc_list_topk = [layer_info[layer]["patch_accuracy"] for _, layer_info in topk_scores.items()]
    patch_acc_list_randk = [layer_info[layer]["patch_accuracy"] for _, layer_info in randk_scores.items()]
    axes[layer].plot(topk_list, patch_acc_list_topk, label="Top-k", marker='o')
    axes[layer].plot(topk_list, patch_acc_list_randk, label="Random-k", marker='x')
    axes[layer].set_title(f"Layer {layer}")
    axes[layer].set_xlabel("Top-k Neurons")
    axes[layer].set_ylabel("Patch Accuracy")
    axes[layer].set_xscale("log", base=2)
axes[-1].legend()
fig.suptitle(f"Patch Accuracy for Top-k and Random-k Neurons\nGame {game_index}, Move {move}")
fig.tight_layout()
fig.savefig(f"figures/topk_patch_accuracy_layer_level_game{game_index}_move{move}.png", dpi=600)

# %% ablation with topk neurons across layers
topk_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
topk_neurons = defaultdict(lambda: defaultdict(list))
randk_neurons = defaultdict(lambda: defaultdict(list))
for topk in topk_list:
    topk_neuron_idx = t.topk(neuron_attribution.flatten(), k=topk).indices
    for idx in topk_neuron_idx:
        layer = idx // n_neurons
        neuron = idx % n_neurons
        topk_neurons[topk][layer.item()].append(neuron.item())

    topk_temp = sorted(topk_neurons[topk].items(), key=lambda kv: kv[0])
    topk_neurons[topk] = defaultdict(list, topk_temp)
    
    t.manual_seed(topk)  # For reproducibility
    randk_neuron_idx = t.randperm(n_layers * n_neurons)[:topk]
    for idx in randk_neuron_idx:
        layer = idx // n_neurons
        neuron = idx % n_neurons
        randk_neurons[topk][layer.item()].append(neuron.item())
    
    randk_temp = sorted(randk_neurons[topk].items(), key=lambda kv: kv[0])
    randk_neurons[topk] = defaultdict(list, randk_temp)

topk_scores = defaultdict(dict)
randk_scores = defaultdict(dict)
for topk in topk_list:
    kl_div_BL, clean_accuracy, patch_accuracy = calculate_ablation_scores_game_move(
        model,
        layers_neurons=topk_neurons[topk],
        focus_games_id=focus_games_id.to(device),
        focus_legal_moves=focus_legal_moves.to(device),
        move=move,
        ablation_method="zero",
    )
    topk_scores[topk] = {
        "kl_div_BL": kl_div_BL,
        "clean_accuracy": clean_accuracy,
        "patch_accuracy": patch_accuracy,
    }

    randk_kl_div_BL, randk_clean_accuracy, randk_patch_accuracy = calculate_ablation_scores_game_move(
        model,
        layers_neurons=randk_neurons[topk],
        focus_games_id=focus_games_id.to(device),
        focus_legal_moves=focus_legal_moves.to(device),
        move=move,
        ablation_method="zero",
    )
    randk_scores[topk] = {
        "kl_div_BL": randk_kl_div_BL,
        "clean_accuracy": randk_clean_accuracy,
        "patch_accuracy": randk_patch_accuracy,
    }

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
axes = ax.flatten()
kl_list_topk = [layer_info["kl_div_BL"] for _, layer_info in topk_scores.items()]
kl_list_randk = [layer_info["kl_div_BL"] for _, layer_info in randk_scores.items()]
axes[0].plot(topk_list, kl_list_topk, label="Top-k", marker='o')
axes[0].plot(topk_list, kl_list_randk, label="Random-k", marker='x')
axes[0].set_title("KL Divergence for Top-k and Random-k Neurons")
axes[0].set_xlabel("Top-k Neurons")
axes[0].set_ylabel("KL Divergence")
axes[0].set_xscale("log", base=2)

patch_acc_list_topk = [layer_info["patch_accuracy"] for _, layer_info in topk_scores.items()]
patch_acc_list_randk = [layer_info["patch_accuracy"] for _, layer_info in randk_scores.items()]
axes[1].plot(topk_list, patch_acc_list_topk, label="Top-k", marker='o')
axes[1].plot(topk_list, patch_acc_list_randk, label="Random-k", marker='x')
axes[1].set_title("Patch Accuracy for Top-k and Random-k Neurons")
axes[1].set_xlabel("Top-k Neurons")
axes[1].set_ylabel("Patch Accuracy")
axes[1].set_xscale("log", base=2)
axes[1].legend()
fig.suptitle(f"Game {game_index}, Move {move}")
fig.tight_layout()
fig.savefig(f"figures/topk_cross_layer_game{game_index}_move{move}.png", dpi=600)

# %% ----- ----- ----- ----- ----- ----- specific square ----- ----- ----- ----- ----- ----- %% #
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
layers = [_ for _ in range(n_layers)]
topk_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# topk_list = [16]
# topk_neurons = defaultdict(lambda: defaultdict(list))
# randk_neurons = defaultdict(lambda: defaultdict(list))
topk_scores = defaultdict(lambda: defaultdict(dict))
randk_scores = defaultdict(lambda: defaultdict(dict))
for topk in topk_list:
    for layer in layers:
        # Get the top-k neurons for the current layer
        topk_neuron_idx = t.topk(neuron_attribution[layer].flatten(), k=topk).indices
        # topk_neurons[topk][layer] = topk_neuron_idx
        
        kl_div_BL, clean_accuracy, patch_accuracy = calculate_ablation_scores_square(
            model,
            layers_neurons={layer: topk_neuron_idx},
            board_seqs_id=board_seqs_id.to(device),
            valid_move_square_mask=valid_move_square_mask.to(device),
            valid_move_number=valid_move_number.to(device),
            token_id=token_id,
            ablation_method="zero",
        )
        topk_scores[topk][layer] = {
            "kl_div_BL": kl_div_BL,
            "clean_accuracy": clean_accuracy,
            "patch_accuracy": patch_accuracy,
        }

        t.manual_seed(layer)
        randk_neuron_idx = t.randperm(n_neurons)[:topk]
        # randk_neurons[topk][layer] = randk_neuron_idx

        randk_kl_div_BL, randk_clean_accuracy, randk_patch_accuracy = calculate_ablation_scores_square(
            model,
            layers_neurons={layer: randk_neuron_idx},
            board_seqs_id=board_seqs_id.to(device),
            valid_move_square_mask=valid_move_square_mask.to(device),
            valid_move_number=valid_move_number.to(device),
            token_id=token_id,
            ablation_method="zero",
        )
        randk_scores[topk][layer] = {
            "kl_div_BL": randk_kl_div_BL,
            "clean_accuracy": randk_clean_accuracy,
            "patch_accuracy": randk_patch_accuracy,
        }

fig, ax = plt.subplots(2,4, figsize=(20, 8))
axes = ax.flatten()
for layer in layers:
    kl_list_topk = [layer_info[layer]["kl_div_BL"] for _, layer_info in topk_scores.items()]
    kl_list_randk = [layer_info[layer]["kl_div_BL"] for _, layer_info in randk_scores.items()]
    axes[layer].plot(topk_list, kl_list_topk, label="Top-k", marker='o')
    axes[layer].plot(topk_list, kl_list_randk, label="Random-k", marker='x')
    axes[layer].set_title(f"Layer {layer}")
    axes[layer].set_xlabel("Top-k Neurons")
    axes[layer].set_ylabel("KL Divergence")
    axes[layer].set_xscale("log", base=2)
axes[-1].legend()
fig.suptitle(f"KL Divergence for Top-k and Random-k Neurons\nSquare {square_idx} ({arena_utils.to_board_label(square_idx)}), Token ID: {token_id}")
fig.tight_layout()
fig.savefig(f"figures/topk_kl_divergence_layer_level_square{square_idx}_token{token_id}.png", dpi=600)

fig, ax = plt.subplots(2,4, figsize=(20, 8))
axes = ax.flatten()
for layer in layers:
    patch_acc_list_topk = [layer_info[layer]["patch_accuracy"] for _, layer_info in topk_scores.items()]
    patch_acc_list_randk = [layer_info[layer]["patch_accuracy"] for _, layer_info in randk_scores.items()]
    axes[layer].plot(topk_list, patch_acc_list_topk, label="Top-k", marker='o')
    axes[layer].plot(topk_list, patch_acc_list_randk, label="Random-k", marker='x')
    axes[layer].set_title(f"Layer {layer}")
    axes[layer].set_xlabel("Top-k Neurons")
    axes[layer].set_ylabel("Patch Accuracy")
    axes[layer].set_xscale("log", base=2)
axes[-1].legend()
fig.suptitle(f"Patch Accuracy for Top-k and Random-k Neurons\nSquare {square_idx} ({arena_utils.to_board_label(square_idx)}), Token ID: {token_id}")
fig.tight_layout()
fig.savefig(f"figures/topk_patch_accuracy_layer_level_square{square_idx}_token{token_id}.png", dpi=600)

# %% ablation with topk neurons across layers
topk_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
topk_neurons = defaultdict(lambda: defaultdict(list))
randk_neurons = defaultdict(lambda: defaultdict(list))
for topk in topk_list:
    topk_neuron_idx = t.topk(neuron_attribution.flatten(), k=topk).indices
    for idx in topk_neuron_idx:
        layer = idx // n_neurons
        neuron = idx % n_neurons
        topk_neurons[topk][layer.item()].append(neuron.item())

    topk_temp = sorted(topk_neurons[topk].items(), key=lambda kv: kv[0])
    topk_neurons[topk] = defaultdict(list, topk_temp)
    
    t.manual_seed(topk)  # For reproducibility
    randk_neuron_idx = t.randperm(n_layers * n_neurons)[:topk]
    for idx in randk_neuron_idx:
        layer = idx // n_neurons
        neuron = idx % n_neurons
        randk_neurons[topk][layer.item()].append(neuron.item())
    
    randk_temp = sorted(randk_neurons[topk].items(), key=lambda kv: kv[0])
    randk_neurons[topk] = defaultdict(list, randk_temp)

topk_scores = defaultdict(dict)
randk_scores = defaultdict(dict)
for topk in topk_list:
    kl_div_BL, clean_accuracy, patch_accuracy = calculate_ablation_scores_square(
        model,
        layers_neurons=topk_neurons[topk],
        board_seqs_id=board_seqs_id.to(device),
        valid_move_square_mask=valid_move_square_mask.to(device),
        valid_move_number=valid_move_number.to(device),
        token_id=token_id,
        ablation_method="zero",
    )
    topk_scores[topk] = {
        "kl_div_BL": kl_div_BL,
        "clean_accuracy": clean_accuracy,
        "patch_accuracy": patch_accuracy,
    }

    randk_kl_div_BL, randk_clean_accuracy, randk_patch_accuracy = calculate_ablation_scores_square(
        model,
        layers_neurons=randk_neurons[topk],
        board_seqs_id=board_seqs_id.to(device),
        valid_move_square_mask=valid_move_square_mask.to(device),
        valid_move_number=valid_move_number.to(device),
        token_id=token_id,
        ablation_method="zero",
    )
    randk_scores[topk] = {
        "kl_div_BL": randk_kl_div_BL,
        "clean_accuracy": randk_clean_accuracy,
        "patch_accuracy": randk_patch_accuracy,
    }

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
axes = ax.flatten()
kl_list_topk = [layer_info["kl_div_BL"] for _, layer_info in topk_scores.items()]
kl_list_randk = [layer_info["kl_div_BL"] for _, layer_info in randk_scores.items()]
axes[0].plot(topk_list, kl_list_topk, label="Top-k", marker='o')
axes[0].plot(topk_list, kl_list_randk, label="Random-k", marker='x')
axes[0].set_title("KL Divergence for Top-k and Random-k Neurons")
axes[0].set_xlabel("Top-k Neurons")
axes[0].set_ylabel("KL Divergence")
axes[0].set_xscale("log", base=2)
patch_acc_list_topk = [layer_info["patch_accuracy"] for _, layer_info in topk_scores.items()]
patch_acc_list_randk = [layer_info["patch_accuracy"] for _, layer_info in randk_scores.items()]
axes[1].plot(topk_list, patch_acc_list_topk, label="Top-k", marker='o')
axes[1].plot(topk_list, patch_acc_list_randk, label="Random-k", marker='x')
axes[1].set_title("Patch Accuracy for Top-k and Random-k Neurons")
axes[1].set_xlabel("Top-k Neurons")
axes[1].set_ylabel("Patch Accuracy")
axes[1].set_xscale("log", base=2)
axes[1].legend()
fig.suptitle(f"Square {square_idx} ({arena_utils.to_board_label(square_idx)}), Token ID: {token_id}")
fig.tight_layout()
fig.savefig(f"figures/topk_cross_layer_square{square_idx}_token{token_id}.png", dpi=600)

# %% print topk neurons for specific square (cross-layer)
topk = 32
print(f"Top {topk} neurons for square {square_idx} ({arena_utils.to_board_label(square_idx)}), token ID {token_id}:")
for layer, neurons in topk_neurons[topk].items():
    print(f"Layer {layer}: Neurons {neurons}")

# %%
# neuron_acts = {}
# with t.no_grad(), model.trace(board_seqs_id.to(device), scan=False, validate=False):
#     for layer in range(n_layers):
#         neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output.save()
#         neuron_acts[layer] = neuron_activations_BLD

# neuron_acts = t.stack([neuron_acts[layer] for layer in range(n_layers)], dim=-2) # [batch, seq, layer, neuron]

# layer = 5
# neuron_idx = 1407

# neuron_attribution = einops.einsum(
#     neuron_acts[:,:,layer,neuron_idx],
#     write_attribution_square[layer,neuron_idx],
#     "batch seq, row col -> batch seq row col",
# )

# neuron_attribution_sum = einops.einsum(
#     neuron_attribution,
#     legal_moves.to(device=device, dtype=t.bool),
#     "batch seq row col, batch seq row col -> batch seq",
# )

# %%
# logits, cache = model.run_with_cache(board_seqs_id.to(device))
# cache[get_act_name("mlp_post", layer)]

# %% ablation with all neurons from some layers
# layers = [4,5,6]

# logits_clean_BLV, logits_patch_BLV = neuron_intervention(
#     model,
#     layers_neurons={layer: [_ for _ in range(n_neurons)] for layer in layers},
#     game_batch_BL=board_seqs_id.to(device),
#     ablation_method="zero",
# )

# kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)

# clean_correct, clean_total, clean_accuracy = compute_top_n_accuracy(
#     logits_clean_BLV, legal_moves.unsqueeze(-1)
# )

# patch_correct, patch_total, patch_accuracy = compute_top_n_accuracy(
#     logits_patch_BLV, legal_moves.unsqueeze(-1)
# )

# print(f"Layer {layer}, Top {topk} Neurons:")
# print(f"KL Divergence: {kl_div_BL.mean().item():.4f}")
# print(f"Clean Accuracy: {clean_accuracy:.4f} ({clean_correct}/{clean_total})")
# print(f"Patch Accuracy: {patch_accuracy:.4f} ({patch_correct}/{patch_total})")

# %% test of ablation across neurons
# topk = 32

# logits_clean_BLV, logits_patch_BLV = neuron_intervention(
#     model,
#     layers_neurons=topk_neurons[topk],
#     game_batch_BL=focus_games_id.to(device),
#     ablation_method="zero",
# )

# logits_clean_BLV_move = logits_clean_BLV[:, move].unsqueeze(1) # [1, 1, ids]
# logits_patch_BLV_move = logits_patch_BLV[:, move].unsqueeze(1) # [1, 1, ids]
# focus_legal_moves_move = focus_legal_moves[:, move].unsqueeze(1).unsqueeze(-1)  # [1, 1, row, col, 1]

# kl_div_BL = compute_kl_divergence(logits_clean_BLV_move, logits_patch_BLV_move)

# clean_correct, clean_total, clean_accuracy = compute_top_n_accuracy(
#     logits_clean_BLV_move, focus_legal_moves_move
# )

# patch_correct, patch_total, patch_accuracy = compute_top_n_accuracy(
#     logits_patch_BLV_move, focus_legal_moves_move
# )

# print(f"Top {topk} Neurons:")
# print(f"KL Divergence: {kl_div_BL.item():.4f}")
# print(f"Clean Accuracy: {clean_accuracy:.4f} ({clean_correct}/{clean_total})")
# print(f"Patch Accuracy: {patch_accuracy:.4f} ({patch_correct}/{patch_total})")

# %% another way to calculate rank
# logits_patch_order = t.argsort(logits_patch_BLV, dim=-1, descending=True)
# logits_patch_rank = t.argsort(logits_patch_order, dim=-1)
# logits_patch_rank_token = logits_patch_rank[..., token_id]  # [game, seq]
