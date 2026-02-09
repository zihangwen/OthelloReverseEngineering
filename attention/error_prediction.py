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
    ALL_SQUARES,
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
    # compute_top_n_accuracy,
    # compute_kl_divergence,
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

# %%
with t.no_grad(), model.trace(board_seqs_id):
    logits_clean_BLV = model.unembed.output.save()

# %%
def compute_top_n_accuracy_out(
    logits_BLV: t.Tensor, valid_moves_BLRR: t.Tensor
) -> tuple[float, float, float]:
    B, L, r1, r2 = valid_moves_BLRR.shape
    n_BL = einops.reduce(valid_moves_BLRR, "B L r1 r2 -> B L", "sum")

    # Create a mask for the top n logits
    top_n_mask = t.zeros_like(logits_BLV, dtype=t.bool)

    for b in range(B):
        for l in range(L):
            n = n_BL[b, l].int()
            _, top_n_indices = t.topk(logits_BLV[b, l], k=n)
            top_n_mask[b, l, top_n_indices] = True

    top_n_mask = top_n_mask.int()
    top_n_square = t.zeros(B, L, r1, r2, dtype =t.int32, device=top_n_mask.device)
    top_n_square.flatten(2, 3)[...,ALL_SQUARES] = top_n_mask[...,1:]

    return top_n_square
    # correct_BLC = valid_moves_with_pass_BLC * stoi_top_n_mask

    # correct = correct_BLC.sum()
    # total = valid_moves_with_pass_BLC.sum()
    # accuracy = correct / total

    # return correct.item(), total.item(), accuracy.item()

# %%
valid_moves_BLRR = test_data["games_batch_to_valid_moves_BLRRC"].squeeze(-1)  # (seq_len, 60)
top_n_square = compute_top_n_accuracy_out(logits_clean_BLV, valid_moves_BLRR)

error_index = t.nonzero(
    top_n_square != valid_moves_BLRR
)

# %% Plot pairs of games and moves with errors
game_move_pairs = [
    (6, 50),
    (25, 34),
    (44, 49),
    (182, 51),
]
n_moves = len(game_move_pairs)

fig, axs = plt.subplots(2, n_moves, figsize=(5*n_moves, 5*(1)+1.5))
# fig.suptitle(f"Attention attribution per key move to query {n_moves-1} move @ Mine Probe", fontsize=16)
from matplotlib.colors import ListedColormap
cmap_board = ListedColormap(["white", "gray", "black"])

# we need boundaries around -1,0,1
bounds_board = [-1.5, -0.5, 0.5, 1.5]
board_seqs_square = t.tensor(test_data["decoded_inputs"])
# board_seqs_square = board_seqs_square[game_idx].unsqueeze(0) # [move, 8, 8]
board_states, _, legal_moves_annotation = get_board_states_and_legal_moves(board_seqs_square)

top_n_square_mask = top_n_square.cpu().numpy()
legal_moves_annotation_predict = np.where(top_n_square_mask, "o", "")

for i_move in range(n_moves):
    game_idx, move = game_move_pairs[i_move]

    ax = axs[0, i_move]
    im = ax.imshow(board_states[game_idx, move], cmap=cmap_board, vmin=-1.5, vmax=1.5)

    sqaure_label = arena_utils.to_board_label(board_seqs_square[game_idx, move])
    ax.set_title(f"Game {game_idx} Move {move} ({sqaure_label}), {'white' if move % 2 == 0 else 'black'} (valid moves)")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_yticklabels(list("ABCDEFGH"))

    # add legal moves annotations
    for r in range(8):
        for c in range(8):
            if legal_moves_annotation[game_idx, move, r, c] != "":
                ax.text(c, r, legal_moves_annotation[game_idx, move, r, c], color = "k", fontsize=12, ha="center", va="center")
    
    # top_n_square
    ax = axs[1, i_move]
    im = ax.imshow(board_states[game_idx, move], cmap=cmap_board, vmin=-1.5, vmax=1.5)
    sqaure_label = arena_utils.to_board_label(board_seqs_square[game_idx, move])
    ax.set_title(f"Game {game_idx} Move {move} ({sqaure_label}), {'white' if move % 2 == 0 else 'black'} (prediction)")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_yticklabels(list("ABCDEFGH"))

    # add legal moves annotations
    for r in range(8):
        for c in range(8):
            if legal_moves_annotation_predict[game_idx, move, r, c] != "":
                ax.text(c, r, legal_moves_annotation_predict[game_idx, move, r, c], color = "r", fontsize=12, ha="center", va="center")

# %%
