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
    get_w_in,
    # get_w_out,
    calculate_neuron_input_weights,
    calculate_neuron_output_weights,
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

# %% ----- ----- ----- ----- ----- ----- specific square ----- ----- ----- ----- ----- ----- %% #
square_idx = 25
token_id = arena_utils.SQUARE_TO_ID[square_idx]
print(f"Square {square_idx} ({arena_utils.to_board_label(square_idx)}), Token ID: {token_id}")

valid_move_square_mask = legal_moves.flatten(start_dim=-2, end_dim=-1)[..., square_idx] # [game, seq]
valid_move_number = legal_moves.sum(dim=(-2,-1))  # [game, seq]

neuron_attribution = defaultdict(list)
batch_size = 5
for start in range(0, board_seqs_id.shape[0], batch_size):
    end = min(start + batch_size, board_seqs_id.shape[0])
    board_seqs_id_batch = board_seqs_id[start:end]

    with t.no_grad(), model.trace(board_seqs_id_batch, scan=False, validate=False):
        for layer in range(n_layers):
            neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output
            neuron_activations_resid = einops.einsum(
                neuron_activations_BLD,
                model.W_out[layer],
                "game seq neuron, neuron d_model -> game seq neuron d_model",
            )
            neuron_activations_resid_ln = model.ln_final(neuron_activations_resid)  # [game, seq, neuron, d_model]

            neuron_attr = einops.einsum(
                neuron_activations_resid_ln,
                model.W_U[...,token_id],
                "game seq neuron d_model, d_model -> game seq neuron",
            )
            neuron_attribution[layer].append(neuron_attr.save())
    
    t.cuda.empty_cache()

for layer in range(n_layers):
    neuron_attribution[layer] = t.cat(neuron_attribution[layer], dim=0)  # [game, seq, neuron]

neuron_attribution = t.stack(
    [neuron_attribution[layer] for layer in range(n_layers)], dim=-2
)  # [game, seq, layer, neuron]

neuron_attribution *= valid_move_square_mask[..., None, None]
neuron_attribution = neuron_attribution.sum(dim=(0, 1))  # [layer, neuron]

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
fig.savefig(f"figures_ln/topk_cross_layer_square{square_idx}_token{token_id}.png", dpi=600)

# %% ----- ----- ----- ----- ----- ----- read probe directions ----- ----- ----- ----- ----- ----- %% #
# print topk neurons for specific square (cross-layer)
topk = 32
print(f"Top {topk} neurons for square {square_idx} ({arena_utils.to_board_label(square_idx)}), token ID {token_id}:")
for layer, neurons in topk_neurons[topk].items():
    print(f"Layer {layer}: Neurons {neurons}")

# %% Load probes (probe directions (Mine, Empty, Yours))
# probe_dict = {i : t.load(
#     f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
# )['linear_probe'].squeeze() for i in range(model.cfg.n_layers)}

# probe_t = t.stack([probe_dict[i] for i in range(n_layers)], dim=0)  # [layer, d_model, row, col, options]
# blank_probe = probe_t[..., 1] - (probe_t[..., 0] + probe_t[..., 2]) * 0.5  # [layer, d_model, row, col]
# my_probe = probe_t[..., 0] - probe_t[..., 2]  # [layer, d_model, row, col]

# blank_probe_normalized = blank_probe / blank_probe.norm(dim=1, keepdim=True)
# my_probe_normalized = my_probe / my_probe.norm(dim=1, keepdim=True)
# blank_probe_normalised[..., [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# w_in = model.W_in.detach().clone()  # [layer, d_model, neuron]
# w_in_nomalized = w_in / w_in.norm(dim=0, keepdim=True)

# probe_cos_sim = einops.einsum(
#     w_in,
#     blank_probe_normalized,
#     "layer d_model neuron, layer d_model row col -> layer neuron row col",
# )

# %% main linear probe
full_linear_probe = t.load("linear_probes/main_linear_probe.pth", map_location=str(device), weights_only=True)

black_to_play, white_to_play, _ = (0, 1, 2)
empty, white, black = (0, 1, 2)

linear_probe = t.stack(
    [
        # "Empty" direction = average of empty direction across probe modes
        full_linear_probe[[black_to_play, white_to_play], ..., [empty, empty]].mean(0),
        # "Theirs" direction = average of {x to play, classification != x} across probe modes
        full_linear_probe[[black_to_play, white_to_play], ..., [white, black]].mean(0),
        # "Mine" direction = average of {x to play, classification == x} across probe modes
        full_linear_probe[[black_to_play, white_to_play], ..., [black, white]].mean(0),
    ],
    dim=-1,
)

blank_probe = linear_probe[..., 0] - linear_probe[..., 1] * 0.5 - linear_probe[..., 2] * 0.5
# mine(2) - theirs(1)
my_probe = linear_probe[..., 2] - linear_probe[..., 1]

blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)

# Set the center blank probes to 0, since they're never blank so the probe is meaningless
blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# %%
topk_neurons_seperate = defaultdict(list)
topk_neuron_idx = t.topk(neuron_attribution.flatten(), k=2048).indices
for i_k, idx in enumerate(topk_neuron_idx):
    layer = idx // n_neurons
    neuron = idx % n_neurons
    topk_neurons_seperate[i_k] = [layer.item(), neuron.item()]

# %%
w_in_accu_blank = 0
w_in_accu_my = 0
for i_k, (layer, neuron) in topk_neurons_seperate.items():
    if i_k >= 16:
        break
    print(f"Rank {i_k}: L{layer}N{neuron}")

    w_in_LN_blank = calculate_neuron_input_weights(model, blank_probe_normalised, layer, neuron)
    w_in_LN_my = calculate_neuron_input_weights(model, my_probe_normalised, layer, neuron)

    w_in_accu_blank += w_in_LN_blank
    w_in_accu_my += w_in_LN_my

    fig = arena_utils.plot_board_values(
        t.stack(
            [w_in_LN_blank, w_in_LN_my, w_in_accu_blank/(i_k+1), w_in_accu_my/(i_k+1)],
        ),
        title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
        board_titles=[
            f"Blank In (Rank {i_k}: L{layer}N{neuron})", f"My In (Rank {i_k}: L{layer}N{neuron})",
            f"Blank In (Mean of top 0 - top {i_k} neurons)", f"My In (Mean of top 0 - top {i_k} neurons)"
        ],
        boards_per_row=2,
        width=650,
        height=380*2,
    )
    fig.write_image(f"figures_ln/probe/neuron_input_weights_rank_{i_k}_L{layer}N{neuron}.png")

    # fig = arena_utils.plot_board_values(
    #     t.stack([w_in_accu_blank, w_in_accu_my]),
    #     title=f"Accumulated input weights in terms of the probe for top 0-{i_k} neurons",
    #     board_titles=["Blank In", "My In"],
    #     # boards_per_row=5,
    #     width=650,
    #     height=380,
    # )
    # fig.write_image(f"figures_ln/probe/accu_neuron_input_weights_rank_{i_k}.png")


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
