# %%
import pickle
from collections import defaultdict
import torch as t
# import numpy as np
import einops
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
# from transformer_lens import ActivationCache, HookedTransformer
# from transformer_lens.utils import to_numpy, get_act_name
# from torch import Tensor
# from IPython.display import HTML, display
# from jaxtyping import Bool, Float, Int
# import neuron_simulation.arena_utils as arena_utils

from helper_fns import (
    # MIDDLE_SQUARES,
    ALL_SQUARES,
    get_board_states_and_legal_moves,
    # plot_probe_outputs,
    # get_w_in,
    # get_w_out,
    # calculate_neuron_input_weights,
    # calculate_neuron_output_weights,
    # create_feature_names,
)
from simulate_activations_with_dts import (
    compute_kl_divergence,
    compute_top_n_accuracy,
)

device = "cuda" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)
tracer_kwargs = {"validate": True, "scan": True}

print(f"Using device: {device}")

# %%

model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = utils.get_model(model_name, device)

# %% Load probes
probe_dict = {i : t.load(
    f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
)['linear_probe'].squeeze() for i in range(model.cfg.n_layers)}

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

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long()
board_seqs_square = t.tensor(test_data["decoded_inputs"]).long()

# %%
def neuron_intervention(
    model,
    layers_neurons: dict[list],
    game_batch_BL: t.Tensor,
    ablation_method: str = "zero",
):
    # allowed_methods = ["mean", "zero", "max"]
    allowed_methods = ["zero"]
    assert ablation_method in allowed_methods, (
        f"Invalid ablation method. Must be one of {allowed_methods}"
    )

    mean_activations = {}
    max_activations = {}

    # Get clean logits and mean submodule activations
    with t.no_grad(), model.trace(game_batch_BL, **tracer_kwargs):
        # for layer in layers:
        #     original_input_BLD = model.blocks[layer].mlp.hook_post.output
        #     mean_activations[layer] = original_input_BLD.mean(dim=(0, 1)).save()
        #     max_activations = original_input_BLD.max(dim=0).values
        #     max_activations[layer] = max_activations.max(dim=0).values.save()
        logits_clean_BLV = model.unembed.output.save()
    
    with t.no_grad(), model.trace(game_batch_BL, **tracer_kwargs):
        for layer, neuron_indices in layers_neurons.items():
            original_input_BLD = model.blocks[layer].mlp.hook_post.output
            if ablation_method == "zero":
                original_input_BLD[:, :, neuron_indices] = 0.0
            elif ablation_method == "mean":
                raise NotImplementedError("Mean ablation not implemented yet.")
            elif ablation_method == "max":
                raise NotImplementedError("Max ablation not implemented yet.")
        
        logits_patch_BLV = model.unembed.output.save()
    
    return logits_clean_BLV, logits_patch_BLV

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


# %%
game_index = 42
move = 42

board_states, legal_moves, legal_moves_annotation = get_board_states_and_legal_moves(board_seqs_square)
legal_moves = legal_moves.to(device=device, dtype=t.float32)

focus_games_id = board_seqs_id[game_index].unsqueeze(0)  # [1, 59]
focus_games_square = board_seqs_square[game_index].unsqueeze(0)  # [1, 59]
focus_legal_moves = legal_moves[game_index].unsqueeze(0)  # [1, 59, 8, 8]


# %%
neuron_attribution = {}
with t.no_grad(), model.trace(focus_games_id.to(device), scan=False, validate=False):
    for layer in range(n_layers):
        neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output
        write_attr_l = write_attribution_square[layer] # [neuron, row, col]

        neuron_attr = einops.einsum(
            neuron_activations_BLD,
            write_attr_l,
            legal_moves,
            "game seq neuron, neuron row col, game seq row col -> game seq neuron",
        )

        neuron_attribution[layer] = neuron_attr.save()

neuron_attribution = t.stack(
    [neuron_attribution[layer] for layer in range(n_layers)], dim=-2
)  # [game, seq, layer, neuron]

# %%
topk_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
topk_neurons = defaultdict(lambda: defaultdict(list))
randk_neurons = defaultdict(lambda: defaultdict(list))

for topk in topk_list:
    topk_neuron_idx = t.topk(neuron_attribution[0, move].flatten(), k=topk).indices
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

# %%
topk = 32

logits_clean_BLV, logits_patch_BLV = neuron_intervention(
    model,
    # layers_neurons=topk_neurons[topk],
    layers_neurons={4:[_ for _ in range(2048)], 5:[_ for _ in range(2048)], 6:[_ for _ in range(2048)]},
    game_batch_BL=focus_games_id.to(device),
    ablation_method="zero",
)

logits_clean_BLV_move = logits_clean_BLV[:, move].unsqueeze(1) # [1, 1, ids]
logits_patch_BLV_move = logits_patch_BLV[:, move].unsqueeze(1) # [1, 1, ids]
focus_legal_moves_move = focus_legal_moves[:, move].unsqueeze(1).unsqueeze(-1)  # [1, 1, row, col, 1]

kl_div_BL = compute_kl_divergence(logits_clean_BLV_move, logits_patch_BLV_move)

clean_correct, clean_total, clean_accuracy = compute_top_n_accuracy(
    logits_clean_BLV_move, focus_legal_moves_move
)

patch_correct, patch_total, patch_accuracy = compute_top_n_accuracy(
    logits_patch_BLV_move, focus_legal_moves_move
)

print(f"Top {topk} Neurons:")
print(f"KL Divergence: {kl_div_BL.item():.4f}")
print(f"Clean Accuracy: {clean_accuracy:.4f} ({clean_correct}/{clean_total})")
print(f"Patch Accuracy: {patch_accuracy:.4f} ({patch_correct}/{patch_total})")

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

# %% pure intervention study
layers = [4,5,6]

logits_clean_BLV, logits_patch_BLV = neuron_intervention(
    model,
    layers_neurons={layer: [_ for _ in range(n_neurons)] for layer in layers},
    game_batch_BL=board_seqs_id.to(device),
    ablation_method="zero",
)

kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)

clean_correct, clean_total, clean_accuracy = compute_top_n_accuracy(
    logits_clean_BLV, legal_moves.unsqueeze(-1)
)

patch_correct, patch_total, patch_accuracy = compute_top_n_accuracy(
    logits_patch_BLV, legal_moves.unsqueeze(-1)
)

print(f"Layer {layer}, Top {topk} Neurons:")
print(f"KL Divergence: {kl_div_BL.mean().item():.4f}")
print(f"Clean Accuracy: {clean_accuracy:.4f} ({clean_correct}/{clean_total})")
print(f"Patch Accuracy: {patch_accuracy:.4f} ({patch_correct}/{patch_total})")


# %%
