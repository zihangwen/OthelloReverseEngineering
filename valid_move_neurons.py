# %%
import pickle
from collections import defaultdict
import torch as t
import numpy as np
import einops
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import to_numpy, get_act_name
from torch import Tensor
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
import neuron_simulation.arena_utils as arena_utils

from helper_fns import (
    MIDDLE_SQUARES,
    ALL_SQUARES,
    get_board_states_and_legal_moves,
    plot_probe_outputs,
    get_w_in,
    get_w_out,
    calculate_neuron_input_weights,
    calculate_neuron_output_weights,
    create_feature_names,
)

device = "cuda" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%

model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = utils.get_model(model_name, device)

# %% Load probes
probe_dict = {i : t.load(
    f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
)['linear_probe'].squeeze() for i in range(model.cfg.n_layers)}

# %% Load decision trees
dt_name = 'neuron_simulation/decision_trees_bs/decision_trees_mlp_neuron_6000.pkl'
with open(dt_name, "rb") as f:
    decision_trees = pickle.load(f)

function_name = list(decision_trees[0].keys())[0]
n_features = decision_trees[0][function_name]["decision_tree"]["model"].n_features_in_
feature_names = create_feature_names(n_features, function_name)

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
board_states, legal_moves, legal_moves_annotation = get_board_states_and_legal_moves(board_seqs_square)
legal_moves = legal_moves.to(device=device, dtype=t.float32)

# %% writing neuron
n_layers = model.cfg.n_layers
n_neurons = model.cfg.d_mlp

W_U_normalized = model.W_U[:, 1:] / model.W_U[:, 1:].norm(dim=0, keepdim=True) # [d_model, 60]
w_out = model.W_out.detach().clone()
w_out_nomalized = w_out / w_out.norm(dim=-1, keepdim=True) # [layer, neuron, d_model]

write_attribution = einops.einsum(
    w_out_nomalized,
    W_U_normalized,
    "layer neuron d_model, d_model id -> layer neuron id",
)

write_attribution_square = t.zeros((n_layers, n_neurons, 8, 8), device=device, dtype=t.float32)
write_attribution_square.flatten(start_dim=-2, end_dim=-1)[..., ALL_SQUARES] = write_attribution

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
neuron_attribution = {}
with t.no_grad(), model.trace(board_seqs_id.to(device), scan=False, validate=False):
    for layer in range(n_layers):
        neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output
        write_attr_l = write_attribution_square[layer] # [neuron, x, y]

        neuron_attr = einops.einsum(
            neuron_activations_BLD,
            write_attr_l,
            legal_moves,
            "batch seq neuron, neuron row col, batch seq row col -> batch seq neuron",
        ).save()

        neuron_attribution[layer] = neuron_attr

# %%
topk = 100
topk_neurons = {}
for layer in range(n_layers):
    # neuron_attr_score = einops.reduce(
    #     neuron_attribution[layer].detach(),
    #     "batch seq neuron -> neuron",
    #     "sum",
    # )
    neuron_attr_score = neuron_attribution[layer].sum(dim=(0,1))
    topk_neuron_idx = t.topk(neuron_attr_score, k=topk).indices
    topk_neurons[layer] = topk_neuron_idx

# %%
