# %%
import pickle
import json
from pathlib import Path
from collections import defaultdict
import torch as t
import numpy as np
import einops
from rich import print as rprint
from rich.table import Column, Table
from rich.console import Console
from rich.terminal_theme import MONOKAI
import os

from IPython.display import HTML, display
# from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from skimage.filters import threshold_otsu
import graphviz

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(BASE_PATH)
BASE_PATH = Path(BASE_PATH)
os.chdir(BASE_PATH)

from transformer_lens.utils import to_numpy
import transformer_lens
import circuitsvis as cv
# from transformer_lens.utils import to_numpy, get_act_name
# from transformer_lens import ActivationCache, HookedTransformer
# from torch import Tensor
# from IPython.display import HTML, display
# from jaxtyping import Bool, Float, Int

import utils.circuits_utils as circuits_utils
from utils.arena_utils import (
    label_to_square,
)
import utils.othello_utils as othello_utils
from utils.probe_utils import (
    # load_probes_and_normalize,
    load_fold_probes_and_normalize,
)
import utils.arena_utils as arena_utils
from utils.helper_fns import (
    get_board_states_and_legal_moves,
    compute_top_n_accuracy,
    compute_kl_divergence,
)
#     # MIDDLE_SQUARES,
#     neuron_intervention,
#     ALL_SQUARES,
#     
#     calculate_ablation_scores_game_move,
#     calculate_ablation_scores_square,
#     calculate_ablation_scores_square_probability,
#     # plot_probe_outputs,
#     get_w_in,
#     # get_w_out,
#     calculate_neuron_input_weights,
#     calculate_neuron_output_weights,
#     create_feature_names,
#     get_neuron_decision_tree,
#     get_neuron_binary_decision_tree,
#     # visualize_decision_tree,
# )
# from simulate_activations_with_dts import (
#     compute_kl_divergence,
#     compute_top_n_accuracy,
# )

device = "cuda:1" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = circuits_utils.get_model(model_name, device)
n_layers = model.cfg.n_layers

# W_Q = model.W_Q.detach().clone()  # [layer, head, d_model, d_head]
# W_K = model.W_K.detach().clone()  # [layer, head, d_model, d_head]
W_O = model.W_O.detach().clone()  # [layer, head, d_head, d_model]
W_V = model.W_V.detach().clone()  # [layer, head, d_model, d_head]

# W_E = model.W_E[1:].detach().clone()  # [vocab_size, d_model]
# W_U = model.W_U[:, 1:].detach().clone()  # [d_model, 60]

# %%
probes = load_fold_probes_and_normalize(n_layers, device)

probe_layer_specific = {
    name: probes[name][5]
    for name in probes.keys()
}
probe_layer_normalized = {
    name: probes[name] / probes[name].norm(dim=1, keepdim=True)
    for name in probes.keys()
}

# %%
# with open("attention/attention_head_types.json", "r") as f:
#     head_type_all = json.load(f)

# %% Load the test dataset and process
test_size = 500
custom_functions = [
    # othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
    othello_utils.games_batch_to_valid_moves_BLRRC, # (legal move)
]
test_data = circuits_utils.construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=test_size,
    split="test", 
    device=device,
)

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)

# game_idx = 0
# n_moves = 9
# # n_layers_selected = 4
# # move_idx = 8
# # board_seqs_id = board_seqs_id[0, :20]
# board_seqs_id = board_seqs_id[game_idx, :n_moves]

# %%
def intervention_Direction(D_space, layers, tag):
    with t.no_grad(), model.trace(board_seqs_id):
        # for layer in range(1, model.cfg.n_layers):
        for layer in layers:
            if isinstance(D_space, dict):
                # layer-specific D
                D = D_space[layer]
            else:
                D = D_space  # (d_model, x) where x is the number of directions stacked together
            
            Q, R = t.linalg.qr(D)
            # pattern = model.blocks[layer].attn.hook_pattern.output  # (batch, heads, seq_len, seq_len)
            hook_norm = model.blocks[layer].ln1.output  # (batch, seq_len, d_model)

            x = hook_norm

            # Columns of D are zero iff the corresponding diagonal of R is zero
            valid = R.diag().abs() > 1e-6          # [n_vectors] boolean mask
            Q_valid = Q[:, valid]                  # keep only non-degenerate basis vectors

            # Reconstruct projected x
            if Q_valid.shape[1] == 0:
                x_proj = t.zeros_like(x)
            else:
                x_proj = x @ Q_valid @ Q_valid.T

            # # Solve min || D c − x ||²
            # # torch.linalg.lstsq expects (..., m, n) @ (..., n, k)
            # c = t.linalg.lstsq(D, x.reshape(-1, x.shape[-1]).T).solution
            # # c: (128, batch*seq)

            # # Reconstruct projected x
            # x_proj = (D @ c).T.reshape_as(x)

            # remove x_proj components in the direction of the original x
            if tag == "remove_proj":
                new_v = einops.einsum(
                    x - x_proj, W_V[layer],
                    "batch seq d_model, head d_model d_head -> batch seq head d_head"
                ) + model.b_V[layer]
            else:
                # keep only x_proj components in the direction of the original x
                new_v = einops.einsum(
                    x_proj, W_V[layer],
                    "batch seq d_model, head d_model d_head -> batch seq head d_head"
                ) + model.b_V[layer]

            new_v = t.nan_to_num(new_v)

            v = model.blocks[layer].attn.hook_v.output
            v[:] = new_v

        logits_patch_BLV = model.unembed.output.save()

    return logits_patch_BLV

# %%
layers_chosen = [
    (0,),
    (0, 1, 2, 3, 4, 5, 6, 7),
    (1, 2, 3, 4, 5,),
    (1, 2, 3, 4, 5, 6, 7),
]
tag = "remove_proj"
# tag = "keep_proj"
# v_values_list = []
# hook_norm_list = []
valid_moves_BLRRC = test_data["games_batch_to_valid_moves_BLRRC"]  # (seq_len, 60)

with t.no_grad(), model.trace(board_seqs_id):
    logits_clean_BLV = model.unembed.output.save()
clean_accuracy = compute_top_n_accuracy(logits_clean_BLV, valid_moves_BLRRC)


# %%
# # Normalize and collect specific layer (e.g., layer 5) for all probes
dirs = []
for key in ["flipped", "just_played", "mine"]:
    d = probe_layer_specific[key]                      # (d_model, 8, 8)
    d = d / d.norm(dim=0, keepdim=True)                # normalize each vector
    dirs.append(d)

# Stack: (num_probs, d_model, 8, 8)
D = t.stack(dirs, dim=1)

# Flatten everything except d_model
D = D.reshape(D.shape[0], -1)                        # (d_model, 128)
D = t.nan_to_num(D)

# Normalize and collect
# dir_dict = defaultdict(list)
# for key in ["flipped", "just_played", "mine"]:
#     d = probes[key]                      # (head, d_model, 8, 8)
#     d = d / d.norm(dim=1, keepdim=True)                # normalize each vector
#     for layer in range(n_layers):
#         dir_dict[layer].append(d[layer])  # (d_model, 8, 8)

# D_dict = {}
# for layer in range(n_layers):
# # Stack: (num_probs, d_model, 8, 8)
#     D = t.stack(dir_dict[layer], dim=0)

#     # Flatten everything except d_model
#     D = D.reshape(-1, D.shape[1]).T                        # (d_model, 128)
#     D = t.nan_to_num(D)
#     D_dict[layer] = D

patch_accuracy_dict = {}
kl_div_BL_dict = {}
for layers in layers_chosen:
    
    logits_patch_BLV = intervention_Direction(D, layers, tag)

    patch_accuracy = compute_top_n_accuracy(logits_patch_BLV, valid_moves_BLRRC)
    kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)

    patch_accuracy_dict[layers] = patch_accuracy
    kl_div_BL_dict[layers] = kl_div_BL

# %% random direction control
# seed 42
t.manual_seed(42)
random_D = t.randn_like(D)
random_D = random_D / random_D.norm(dim=0, keepdim=True)

random_patch_accuracy_dict = {}
random_kl_div_BL_dict = {}
for layers in layers_chosen:
    logits_random_patch_BLV = intervention_Direction(random_D, layers, tag)

    random_patch_accuracy = compute_top_n_accuracy(logits_random_patch_BLV, valid_moves_BLRRC)
    random_kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_random_patch_BLV)

    random_patch_accuracy_dict[layers] = random_patch_accuracy
    random_kl_div_BL_dict[layers] = random_kl_div_BL

# %% zero direction control
zero_D = t.zeros_like(D)
zero_patch_accuracy_dict = {}
zero_kl_div_BL_dict = {}
for layers in layers_chosen:
    logits_zero_patch_BLV = intervention_Direction(zero_D, layers, tag)

    zero_patch_accuracy = compute_top_n_accuracy(logits_zero_patch_BLV, valid_moves_BLRRC)
    zero_kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_zero_patch_BLV)

    zero_patch_accuracy_dict[layers] = zero_patch_accuracy
    zero_kl_div_BL_dict[layers] = zero_kl_div_BL

# %% Draw a table
table = Table(title=f"Attention Head Intervention Results\n(Accuracy before intervention: {clean_accuracy[-1]*100:.2f}%)", show_lines=True)
table.add_column("Intervention layers", style="bold cyan", no_wrap=True)
table.add_column("Accu.", style="light_green", justify="right")
table.add_column("KL", style="green", justify="right")
table.add_column("Accu. (Rand. )", style="red", justify="right")
table.add_column("KL (Rand.)", style="red", justify="right")
table.add_column("Accu. (Zero)", style="yellow", justify="right")
table.add_column("KL (Zero)", style="yellow", justify="right")
for layers in layers_chosen:
    patch_accuracy = patch_accuracy_dict[layers]
    kl_div_BL = kl_div_BL_dict[layers]
    random_patch_accuracy = random_patch_accuracy_dict[layers]
    random_kl_div_BL = random_kl_div_BL_dict[layers]

    # layer_str = " ".join(f"L{l}" for l in layers)
    table.add_row(
        f"{" ".join(f"L{l}" for l in layers)}",
        f"{patch_accuracy[-1]*100:.2f}%", f"{kl_div_BL.mean():.4f}",
        f"{random_patch_accuracy[-1]*100:.2f}%", f"{random_kl_div_BL.mean():.4f}",
        f"{zero_patch_accuracy_dict[layers][-1]*100:.2f}%", f"{zero_kl_div_BL_dict[layers].mean():.4f}",
    )

console = Console(record=True)
console.print(table)

# %%
