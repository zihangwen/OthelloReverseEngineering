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
# W_O = model.W_O.detach().clone()  # [layer, head, d_head, d_model]
# W_V = model.W_V.detach().clone()  # [layer, head, d_model, d_head]

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
with open("attention/attention_head_types.json", "r") as f:
    head_type_all = json.load(f)

# %% Load the test dataset and process
test_size = 500
custom_functions = [
    othello_utils.games_batch_to_flipped_classifier_input_BLC,
    othello_utils.games_batch_to_just_played_BLC,
    # othello_utils.games_batch_to_board_state_classifier_input_BLC,
]
test_data = circuits_utils.construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=test_size,
    split="test", 
    device=device,
)

start_move = 8
n_moves = 30

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)
board_seqs_id = board_seqs_id[:, :n_moves]

flipped_squares = einops.rearrange(
    test_data["games_batch_to_flipped_classifier_input_BLC"][:, :n_moves],
    "batch seq (row col) -> batch seq row col", row = 8, col = 8
).bool().cpu()
flipped_squares_expanded = flipped_squares.cpu().numpy()[:,start_move:,np.newaxis, np.newaxis,:,:]  # (batch, seq_len, 1, 1, row, col)

just_played_squares = einops.rearrange(
    test_data["games_batch_to_just_played_BLC"][:, :n_moves],
    "batch seq (row col) -> batch seq row col", row = 8, col = 8
).bool().cpu()

played_and_flipped_squares = flipped_squares + just_played_squares
played_and_flipped_squares_expanded = played_and_flipped_squares.cpu().numpy()[:,start_move:,np.newaxis, np.newaxis,:,:]  # (batch, seq_len, 1, 1, row, col)


# %%
# keys = [transformer_lens.utils.get_act_name("result", i) for i in range(model.cfg.n_layers)]
logits, cache = model.run_with_cache(
    board_seqs_id,
)

# %%
def topk_accuracy(pred, gt):
    p = pred.flatten()
    g = gt.flatten()
    k = int(g.sum())
    top_k_idx = np.argsort(p)[-k:]
    binarized = np.zeros_like(g)
    binarized[top_k_idx] = 1
    threshold = p[top_k_idx[0]]  # minimum value among top-k
    acc = (binarized == g).mean()
    return threshold, acc

def botk_accuracy(pred, gt):
    p = pred.flatten()
    g = gt.flatten()
    k = int(g.sum())
    bottom_k_idx = np.argsort(p)[:k]
    binarized = np.ones_like(g)
    binarized[bottom_k_idx] = 0
    threshold = p[bottom_k_idx[-1]]  # maximum value among bottom-k
    acc = (binarized == g).mean()
    return threshold, acc

# %%
# resid_pre = t.stack([
#     cache["resid_pre", layer] for layer in range(model.cfg.n_layers)
# ], dim = 2)
attn_out = t.stack([
    cache["attn_out", layer] for layer in range(model.cfg.n_layers)
], dim = 2)
# resid_mid = t.stack([
#     cache["resid_mid", layer] for layer in range(model.cfg.n_layers)
# ], dim = 2)
mlp_out = t.stack([
    cache["mlp_out", layer] for layer in range(model.cfg.n_layers)
], dim = 2)
resid_post = t.stack([
    cache["resid_post", layer] for layer in range(model.cfg.n_layers)
], dim = 2)

# %%
func_all = t.stack([
    attn_out, mlp_out,
    # resid_post,
], dim = -1)  # (batch, seq_len, n_layers, d_model, 2)

func_all_normalized = func_all / func_all.norm(dim=3, keepdim=True)
probe_layer_specific_normalized = {
    name: probe_layer_specific[name] / probe_layer_specific[name].norm(dim=0, keepdim=True)
    for name in probe_layer_specific.keys()
}

# %% write to flip direction
func_flipped = einops.einsum(
    func_all_normalized,
    probe_layer_specific_normalized["flipped"],
    "batch seq layer d_model func, d_model row col-> batch seq layer func row col",
).cpu().numpy()

best_thresholds_topk = np.stack([
    topk_accuracy(
        func_flipped[game, seq, layer, func],
        flipped_squares.cpu().numpy()[game, seq]
    )[0]
    for game in range(test_size)
    for seq in range(start_move, n_moves)
    for layer in range(n_layers)
    for func in range(func_all.shape[-1])
]).reshape(test_size, -1, n_layers, func_all.shape[-1])

binarized_func_flipped_topk = (func_flipped[:,start_move:] >= best_thresholds_topk[..., np.newaxis, np.newaxis]).astype(int)
accuracy_flipped_topk = (binarized_func_flipped_topk == flipped_squares_expanded).mean(axis=(4,5))

# F1 score for topk
tp_flipped_topk = ((binarized_func_flipped_topk == 1) & (flipped_squares_expanded == 1)).sum(axis=(4,5))
tn_flipped_topk = ((binarized_func_flipped_topk == 0) & (flipped_squares_expanded == 0)).sum(axis=(4,5))
fp_flipped_topk = ((binarized_func_flipped_topk == 1) & (flipped_squares_expanded == 0)).sum(axis=(4,5))
fn_flipped_topk = ((binarized_func_flipped_topk == 0) & (flipped_squares_expanded == 1)).sum(axis=(4,5))
precision_flipped_topk = tp_flipped_topk / (tp_flipped_topk + fp_flipped_topk + 1e-10)
recall_flipped_topk = tp_flipped_topk / (tp_flipped_topk + fn_flipped_topk + 1e-10)
f1_flipped_topk = 2 * (precision_flipped_topk * recall_flipped_topk) / (precision_flipped_topk + recall_flipped_topk + 1e-10)

fig, axs = plt.subplots(2, 8, figsize=(3*8, 3*2+1.5))
fig.suptitle(f"F1 Score of Writing to Flipped Dir Across Layers", fontsize=16)

# Second pass: plot with consistent colorbar
idx = 0

for func in range(func_all.shape[-1]):
    for layer in range(n_layers):
        ax = axs.flatten()[idx]
        im = ax.hist2d(
            x=np.arange(start_move, n_moves).repeat(test_size),
            y=f1_flipped_topk[:, :, layer, func].flatten(),
            bins=[n_moves - start_move, 20],
            range=[[start_move, n_moves], [0, 1]],
            cmap='Blues'
        )[3]

        ax.set_title(f"Layer {layer} - {'Attention' if func == 0 else 'MLP'}", fontsize=14)
        ax.set_xlabel("Move Index", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        idx += 1

# Add one large colorbar on the right
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Number of Games', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
plt.show()

# %% write to mine (for just flipped)
func_mine = einops.einsum(
    func_all_normalized,
    probe_layer_specific_normalized["mine"],
    "batch seq layer d_model func, d_model row col-> batch seq layer func row col",
).cpu().numpy()

best_thresholds_botk = np.stack([
    botk_accuracy(
        func_mine[game, seq, layer, func],
        played_and_flipped_squares.cpu().numpy()[game, seq]
    )[0]
    for game in range(test_size)
    for seq in range(start_move, n_moves)
    for layer in range(n_layers)
    for func in range(func_all.shape[-1])
]).reshape(test_size, -1, n_layers, func_all.shape[-1])

binarized_func_mine_botk = (func_mine[:,start_move:] <= best_thresholds_botk[..., np.newaxis, np.newaxis]).astype(int)
accuracy_mine_botk = (binarized_func_mine_botk == played_and_flipped_squares_expanded).mean(axis=(4,5))

# F1 score for botk
tp_mine_botk = ((binarized_func_mine_botk == 1) & (played_and_flipped_squares_expanded == 1)).sum(axis=(4,5))
tn_mine_botk = ((binarized_func_mine_botk == 0) & (played_and_flipped_squares_expanded == 0)).sum(axis=(4,5))
fp_mine_botk = ((binarized_func_mine_botk == 1) & (played_and_flipped_squares_expanded == 0)).sum(axis=(4,5))
fn_mine_botk = ((binarized_func_mine_botk == 0) & (played_and_flipped_squares_expanded == 1)).sum(axis=(4,5))
precision_mine_botk = tp_mine_botk / (tp_mine_botk + fp_mine_botk + 1e-10)
recall_mine_botk = tp_mine_botk / (tp_mine_botk + fn_mine_botk + 1e-10)
f1_mine_botk = 2 * (precision_mine_botk * recall_mine_botk) / (precision_mine_botk + recall_mine_botk + 1e-10)

fig, axs = plt.subplots(2, 8, figsize=(3*8, 3*2+1.5))
fig.suptitle(f"F1 Score of Writing to Mine Dir (for Just Played and Flipped Tiles) Across Layers", fontsize=16)
# Second pass: plot with consistent colorbar
idx = 0

for func in range(func_all.shape[-1]):
    for layer in range(n_layers):
        ax = axs.flatten()[idx]
        im = ax.hist2d(
            x=np.arange(start_move, n_moves).repeat(test_size),
            y=f1_mine_botk[:, :, layer, func].flatten(),
            bins=[n_moves - start_move, 20],
            range=[[start_move, n_moves], [0, 1]],
            cmap='Reds'
        )[3]

        ax.set_title(f"Layer {layer} - {'Attention' if func == 0 else 'MLP'}", fontsize=14)
        ax.set_xlabel("Move Index", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        idx += 1
# Add one large colorbar on the right
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Number of Games', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
plt.show()

# %%
