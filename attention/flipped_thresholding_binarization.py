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
    # othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
    # othello_utils.games_batch_to_valid_moves_BLRRC, # (legal move)
]
test_data = circuits_utils.construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=test_size,
    split="test", 
    device=device,
)

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)
board_seqs_id = board_seqs_id[:, :30]

flipped_squares = test_data["games_batch_to_flipped_classifier_input_BLC"][:, :30]
flipped_squares = einops.rearrange(flipped_squares, "batch seq (row col) -> batch seq row col", row = 8, col = 8)

# %%
# keys = [transformer_lens.utils.get_act_name("result", i) for i in range(model.cfg.n_layers)]
logits, cache = model.run_with_cache(
    board_seqs_id,
)

# %%
# pattern_list = dict()
# with t.no_grad(), model.trace(board_seqs_id):
#     for layer in range(model.cfg.n_layers):
#         pattern = model.blocks[layer].attn.hook_pattern.output  # (batch, heads, seq_len, seq_len)
#         pattern_list[layer] = pattern.cpu().save()

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

func_flipped = einops.einsum(
    func_all_normalized,
    probe_layer_specific_normalized["flipped"],
    "batch seq layer d_model func, d_model row col-> batch seq layer func row col",
).cpu().numpy()

flipped_squares_expanded = flipped_squares.cpu().numpy()[:,8:,np.newaxis, np.newaxis,:,:]  # (batch, seq_len, 1, 1, row, col)

# %% otsu for each game seq layer func
threshold_otsus = np.stack([
    threshold_otsu(func_flipped[game, seq, layer, func])
    for game in range(test_size)
    for seq in range(8,30)
    for layer in range(n_layers)
    for func in range(func_all.shape[-1])
]).reshape(test_size, -1, n_layers, func_all.shape[-1])  # (n_layers, 2)

binarized_func_flipped = (func_flipped[:,8:] >= threshold_otsus[..., np.newaxis, np.newaxis]).astype(int)

accuracy_flipped = (binarized_func_flipped == flipped_squares_expanded).mean(axis=(4,5))
tp_flipped = ((binarized_func_flipped == 1) & (flipped_squares_expanded == 1)).sum(axis=(4,5))
tn_flipped = ((binarized_func_flipped == 0) & (flipped_squares_expanded == 0)).sum(axis=(4,5))
fp_flipped = ((binarized_func_flipped == 1) & (flipped_squares_expanded == 0)).sum(axis=(4,5))
fn_flipped = ((binarized_func_flipped == 0) & (flipped_squares_expanded == 1)).sum(axis=(4,5))

tp_flipped_rate = tp_flipped / (tp_flipped + fn_flipped + 1e-10)
tn_flipped_rate = tn_flipped / (tn_flipped + fp_flipped + 1e-10)
fp_flipped_rate = fp_flipped / (fp_flipped + tn_flipped + 1e-10)
fn_flipped_rate = fn_flipped / (fn_flipped + tp_flipped + 1e-10)

# %% otsu for each layer func (aggregated across games and seq)
threshold_otsus_all = np.stack([
    threshold_otsu(func_flipped[:, 8:, layer, func])
    for layer in range(n_layers)
    for func in range(func_all.shape[-1])
]).reshape(n_layers, func_all.shape[-1])  # (n_layers, n_funcs)

binarized_func_flipped_all = (func_flipped[:,8:] >= threshold_otsus_all[np.newaxis, np.newaxis, ..., np.newaxis, np.newaxis]).astype(int)

accuracy_flipped_all = (binarized_func_flipped_all == flipped_squares_expanded).mean(axis=(4,5))
tp_flipped_all = ((binarized_func_flipped_all == 1) & (flipped_squares_expanded == 1)).sum(axis=(4,5))
tn_flipped_all = ((binarized_func_flipped_all == 0) & (flipped_squares_expanded == 0)).sum(axis=(4,5))
fp_flipped_all = ((binarized_func_flipped_all == 1) & (flipped_squares_expanded == 0)).sum(axis=(4,5))
fn_flipped_all = ((binarized_func_flipped_all == 0) & (flipped_squares_expanded == 1)).sum(axis=(4,5))


# %% best accuracy threshold
def scan_thresholds(pred, gt):
    p = pred.flatten()
    g = gt.flatten()
    thresholds = np.unique(p)
    best_acc = 0
    best_threshold = thresholds[0]
    for threshold in thresholds:
        binarized = (p >= threshold).astype(int)
        acc = (binarized == g).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            # tp = ((binarized == 1) & (g == 1)).sum()
            # tn = ((binarized == 0) & (g == 0)).sum()
            # fp = ((binarized == 1) & (g == 0)).sum()
            # fn = ((binarized == 0) & (g == 1)).sum()
    return best_threshold, best_acc

# best threshold for each game seq layer func
best_thresholds = np.stack([
    scan_thresholds(
        func_flipped[game, seq, layer, func],
        flipped_squares.cpu().numpy()[game, seq]
    )[0]
    for game in range(test_size)
    for seq in range(8,30)
    for layer in range(n_layers)
    for func in range(func_all.shape[-1])
]).reshape(test_size, -1, n_layers, func_all.shape[-1])

binarized_func_flipped_best = (func_flipped[:,8:] >= best_thresholds[..., np.newaxis, np.newaxis]).astype(int)

accuracy_flipped_best = (binarized_func_flipped_best == flipped_squares_expanded).mean(axis=(4,5))
# tp_flipped_best = ((binarized_func_flipped_best == 1) & (flipped_squares_expanded == 1)).sum(axis=(4,5))
# tn_flipped_best = ((binarized_func_flipped_best == 0) & (flipped_squares_expanded == 0)).sum(axis=(4,5))
# fp_flipped_best = ((binarized_func_flipped_best == 1) & (flipped_squares_expanded == 0)).sum(axis=(4,5))
# fn_flipped_best = ((binarized_func_flipped_best == 0) & (flipped_squares_expanded == 1)).sum(axis=(4,5))

# tp_flipped_best_rate = tp_flipped_best / (tp_flipped_best + fn_flipped_best + 1e-10)
# tn_flipped_best_rate = tn_flipped_best / (tn_flipped_best + fp_flipped_best + 1e-10)
# fp_flipped_best_rate = fp_flipped_best / (fp_flipped_best + tn_flipped_best + 1e-10)
# fn_flipped_best_rate = fn_flipped_best / (fn_flipped_best + tp_flipped_best + 1e-10)

# # find game and seq for fp not zero is layer 5 mlp
# layer = 5
# func = 0
# fp_nonzero_indices = np.argwhere(fp_flipped_best[:, :, layer, func] > 0)
# fp_nonzero_list = []
# for game_idx, seq_idx in fp_nonzero_indices:
#     fp_count = fp_flipped_best[game_idx, seq_idx, layer, func]
#     if fp_count > 0:
#         # print(f"Game {game_idx}, Seq {seq_idx + 8}, FP Count: {fp_count}")
#         fp_nonzero_list.append((game_idx.item(), seq_idx.item() + 8, fp_count.item()))

# print(max(fp_nonzero_list, key=lambda x: x[2]))

# %% top k thresholding accuracy
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

best_thresholds_topk = np.stack([
    topk_accuracy(
        func_flipped[game, seq, layer, func],
        flipped_squares.cpu().numpy()[game, seq]
    )[0]
    for game in range(test_size)
    for seq in range(8,30)
    for layer in range(n_layers)
    for func in range(func_all.shape[-1])
]).reshape(test_size, -1, n_layers, func_all.shape[-1])

binarized_func_flipped_topk = (func_flipped[:,8:] >= best_thresholds_topk[..., np.newaxis, np.newaxis]).astype(int)
accuracy_flipped_topk = (binarized_func_flipped_topk == flipped_squares_expanded).mean(axis=(4,5))

# F1 score for topk
tp_flipped_topk = ((binarized_func_flipped_topk == 1) & (flipped_squares_expanded == 1)).sum(axis=(4,5))
tn_flipped_topk = ((binarized_func_flipped_topk == 0) & (flipped_squares_expanded == 0)).sum(axis=(4,5))
fp_flipped_topk = ((binarized_func_flipped_topk == 1) & (flipped_squares_expanded == 0)).sum(axis=(4,5))
fn_flipped_topk = ((binarized_func_flipped_topk == 0) & (flipped_squares_expanded == 1)).sum(axis=(4,5))
precision_flipped_topk = tp_flipped_topk / (tp_flipped_topk + fp_flipped_topk + 1e-10)
recall_flipped_topk = tp_flipped_topk / (tp_flipped_topk + fn_flipped_topk + 1e-10)
f1_flipped_topk = 2 * (precision_flipped_topk * recall_flipped_topk) / (precision_flipped_topk + recall_flipped_topk + 1e-10)

# another way to calculate topk accuracy without thresholding
# best_thresholds_topk = []
# accuracy_flipped_topk = []
# for game in range(test_size):
#     for seq in range(8,30):
#         for layer in range(n_layers):
#             for func in range(func_all.shape[-1]):
#                 threshold, acc = topk_accuracy(
#                     func_flipped[game, seq, layer, func],
#                     flipped_squares.cpu().numpy()[game, seq]
#                 )
#                 best_thresholds_topk.append(threshold)
#                 accuracy_flipped_topk.append(acc)

# best_thresholds_topk = np.array(best_thresholds_topk).reshape(test_size, -1, n_layers, func_all.shape[-1])
# accuracy_flipped_topk = np.array(accuracy_flipped_topk).reshape(test_size, -1, n_layers, func_all.shape[-1])

# %% plot accuracy distribution (hist 1D)
# fig, axs = plt.subplots(2, 8, figsize=(3*8, 3*2+1.5))
# fig.suptitle(f"Accuracy of Neuron Flipped Square Detectors Across Layers", fontsize=16)

# # Second pass: plot with consistent colorbar
# idx = 0

# for func in range(func_all.shape[-1]):
#     for layer in range(n_layers):
#         ax = axs.flatten()[idx]
#         # hist
#         im = ax.hist(
#             accuracy_flipped[:, :, layer, func].flatten(),
#             bins=20,
#             range=(0, 1),
#             color='skyblue',
#             edgecolor='black'
#         )
#         ax.set_title(f"Layer {layer} - {'Attention Out' if func == 0 else 'MLP Out'}", fontsize=14)
#         ax.set_xlabel("Accuracy", fontsize=12)
#         ax.set_ylabel("Number of moves in games", fontsize=12)
#         idx += 1
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# %% plot F1 distribution (hist 2D)
fig, axs = plt.subplots(2, 8, figsize=(3*8, 3*2+1.5))
fig.suptitle(f"F1 Score of Neuron Flipped Square Detectors Across Layers", fontsize=16)

# Second pass: plot with consistent colorbar
idx = 0

for func in range(func_all.shape[-1]):
    for layer in range(n_layers):
        ax = axs.flatten()[idx]
        im = ax.hist2d(
            x=np.arange(8,30).repeat(test_size),
            y=f1_flipped_topk[:, :, layer, func].flatten(),
            bins=[22, 20],
            range=[[8, 30], [0, 1]],
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

# %% plot accuracy distribution (hist 2D)
fig, axs = plt.subplots(2, 8, figsize=(3*8, 3*2+1.5))
fig.suptitle(f"Accuracy of Neuron Flipped Square Detectors Across Layers", fontsize=16)

# Second pass: plot with consistent colorbar
idx = 0

for func in range(func_all.shape[-1]):
    for layer in range(n_layers):
        ax = axs.flatten()[idx]
        im = ax.hist2d(
            x=np.arange(8,30).repeat(test_size),
            y=accuracy_flipped[:, :, layer, func].flatten(),
            bins=[22, 20],
            range=[[8, 30], [0, 1]],
            cmap='Blues'
        )[3]

        ax.set_title(f"Layer {layer} - {'Attention' if func == 0 else 'MLP'}", fontsize=14)
        ax.set_xlabel("Move Index", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        idx += 1

# Add one large colorbar on the right
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Number of Games', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
plt.show()


# %%
# min_accuracy = np.argmin(tp_flipped_rate.sum(axis=(-1,-2)))
# game = min_accuracy // 22
# move = min_accuracy % 22 + 8

# print(f"Game with lowest tp rate: Game {game}, Move {move}")

# %%
# game = 0
# move = 20 + 8

# %%
