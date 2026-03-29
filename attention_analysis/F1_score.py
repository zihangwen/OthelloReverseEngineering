# %%
"""
F1 score analysis for residual stream → probe direction projections.

Evaluates how well attn_out and mlp_out write to the flipped and mine probe
directions, using topk (for flipped) and botk (for mine/played) thresholding.
Outputs 2D histograms of F1 scores across moves and layers.
"""
from pathlib import Path
import os
import torch as t
import numpy as np
import einops
import matplotlib.pyplot as plt

BASE_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_PATH)

import utils.othello_utils as othello_utils
from attention_analysis.attention_utils import (
    setup_model_and_probes,
    load_test_dataset,
    stack_residual_streams,
    topk_accuracy,
    botk_accuracy,
    compute_f1,
)

device = "cuda:1" if t.cuda.is_available() else "cpu"
print(f"Using device: {device}")

FIG_DIR = BASE_PATH / "attention_analysis" / "fig" / "F1_score"
os.makedirs(FIG_DIR, exist_ok=True)

# %%
model, n_layers, probes, probe_layer_specific = setup_model_and_probes(device=device)

test_size = 500
start_move, n_moves = 8, 30
test_data, board_seqs_id, _ = load_test_dataset(
    [
        othello_utils.games_batch_to_flipped_classifier_input_BLC,
        othello_utils.games_batch_to_just_played_BLC,
    ],
    n_games=test_size,
    n_moves=n_moves,
    device=device,
)

# %% Ground-truth board labels
flipped_squares = einops.rearrange(
    test_data["games_batch_to_flipped_classifier_input_BLC"][:, :n_moves],
    "batch seq (row col) -> batch seq row col", row=8, col=8,
).bool().cpu()

just_played_squares = einops.rearrange(
    test_data["games_batch_to_just_played_BLC"][:, :n_moves],
    "batch seq (row col) -> batch seq row col", row=8, col=8,
).bool().cpu()

played_and_flipped = (flipped_squares + just_played_squares)
flipped_exp       = flipped_squares.numpy()[:, start_move:, np.newaxis, np.newaxis, :, :]
played_flipped_exp = played_and_flipped.numpy()[:, start_move:, np.newaxis, np.newaxis, :, :]

# %%
_, cache = model.run_with_cache(board_seqs_id)
streams = stack_residual_streams(cache, n_layers, streams=("attn_out", "mlp_out"))

# func_all: [batch, seq, layer, d_model, 2]  (dim 4: 0=attn_out, 1=mlp_out)
func_all = t.stack([streams["attn_out"], streams["mlp_out"]], dim=-1)
n_funcs = func_all.shape[-1]

func_all_norm = func_all / func_all.norm(dim=3, keepdim=True)

# %% Flipped: topk thresholding
func_flipped = einops.einsum(
    func_all_norm, probes["flipped"],
    "batch seq layer d_model func, layer d_model row col -> batch seq layer func row col",
).cpu().numpy()

best_thresh_topk = np.stack([
    topk_accuracy(func_flipped[g, s, l, f], flipped_squares.numpy()[g, s])[0]
    for g in range(test_size)
    for s in range(start_move, n_moves)
    for l in range(n_layers)
    for f in range(n_funcs)
]).reshape(test_size, n_moves - start_move, n_layers, n_funcs)

binarized_flipped = (func_flipped[:, start_move:] >= best_thresh_topk[..., np.newaxis, np.newaxis]).astype(int)
tp = ((binarized_flipped == 1) & (flipped_exp == 1)).sum(axis=(4, 5))
fp = ((binarized_flipped == 1) & (flipped_exp == 0)).sum(axis=(4, 5))
fn = ((binarized_flipped == 0) & (flipped_exp == 1)).sum(axis=(4, 5))
f1_flipped = compute_f1(tp, fp, fn)["f1"]

fig, axs = plt.subplots(n_funcs, n_layers, figsize=(3 * n_layers, 3 * n_funcs + 1.5))
fig.suptitle("F1 Score: Writing to Flipped Direction Across Layers", fontsize=16)
for f in range(n_funcs):
    for l in range(n_layers):
        ax = axs[f, l] if n_funcs > 1 else axs[l]
        im = ax.hist2d(
            x=np.arange(start_move, n_moves).repeat(test_size),
            y=f1_flipped[:, :, l, f].flatten(),
            bins=[n_moves - start_move, 20],
            range=[[start_move, n_moves], [0, 1]],
            cmap="Blues",
        )[3]
        ax.set_title(f"L{l} — {'Attn' if f == 0 else 'MLP'}", fontsize=14)
        ax.set_xlabel("Move"); ax.set_ylabel("F1")

fig.subplots_adjust(right=0.92)
cb = fig.colorbar(im, cax=fig.add_axes([0.94, 0.15, 0.02, 0.7]))
cb.set_label("Games", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
fig.savefig(FIG_DIR / "F1_flipped.jpg", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "F1_flipped.pdf", bbox_inches="tight")
plt.show()

# %% Mine/played+flipped: botk thresholding
func_mine = einops.einsum(
    func_all_norm, probes["mine"],
    "batch seq layer d_model func, layer d_model row col -> batch seq layer func row col",
).cpu().numpy()

best_thresh_botk = np.stack([
    botk_accuracy(func_mine[g, s, l, f], played_and_flipped.numpy()[g, s])[0]
    for g in range(test_size)
    for s in range(start_move, n_moves)
    for l in range(n_layers)
    for f in range(n_funcs)
]).reshape(test_size, n_moves - start_move, n_layers, n_funcs)

binarized_mine = (func_mine[:, start_move:] <= best_thresh_botk[..., np.newaxis, np.newaxis]).astype(int)
tp = ((binarized_mine == 1) & (played_flipped_exp == 1)).sum(axis=(4, 5))
fp = ((binarized_mine == 1) & (played_flipped_exp == 0)).sum(axis=(4, 5))
fn = ((binarized_mine == 0) & (played_flipped_exp == 1)).sum(axis=(4, 5))
f1_mine = compute_f1(tp, fp, fn)["f1"]

fig, axs = plt.subplots(n_funcs, n_layers, figsize=(3 * n_layers, 3 * n_funcs + 1.5))
fig.suptitle("F1 Score: Writing to Mine Direction (Played+Flipped) Across Layers", fontsize=16)
for f in range(n_funcs):
    for l in range(n_layers):
        ax = axs[f, l] if n_funcs > 1 else axs[l]
        im = ax.hist2d(
            x=np.arange(start_move, n_moves).repeat(test_size),
            y=f1_mine[:, :, l, f].flatten(),
            bins=[n_moves - start_move, 20],
            range=[[start_move, n_moves], [0, 1]],
            cmap="Reds",
        )[3]
        ax.set_title(f"L{l} — {'Attn' if f == 0 else 'MLP'}", fontsize=14)
        ax.set_xlabel("Move"); ax.set_ylabel("F1")

fig.subplots_adjust(right=0.92)
cb = fig.colorbar(im, cax=fig.add_axes([0.94, 0.15, 0.02, 0.7]))
cb.set_label("Games", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
fig.savefig(FIG_DIR / "F1_mine_played_flipped.jpg", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "F1_mine_played_flipped.pdf", bbox_inches="tight")
plt.show()

# %% Mine/flipped: botk thresholding
func_mine = einops.einsum(
    func_all_norm, probes["mine"],
    "batch seq layer d_model func, layer d_model row col -> batch seq layer func row col",
).cpu().numpy()

best_thresh_botk = np.stack([
    botk_accuracy(func_mine[g, s, l, f], flipped_squares.numpy()[g, s])[0]
    for g in range(test_size)
    for s in range(start_move, n_moves)
    for l in range(n_layers)
    for f in range(n_funcs)
]).reshape(test_size, n_moves - start_move, n_layers, n_funcs)

binarized_mine = (func_mine[:, start_move:] <= best_thresh_botk[..., np.newaxis, np.newaxis]).astype(int)
tp = ((binarized_mine == 1) & (flipped_exp == 1)).sum(axis=(4, 5))
fp = ((binarized_mine == 1) & (flipped_exp == 0)).sum(axis=(4, 5))
fn = ((binarized_mine == 0) & (flipped_exp == 1)).sum(axis=(4, 5))
f1_mine = compute_f1(tp, fp, fn)["f1"]

fig, axs = plt.subplots(n_funcs, n_layers, figsize=(3 * n_layers, 3 * n_funcs + 1.5))
fig.suptitle("F1 Score: Writing to Mine Direction (Played+Flipped) Across Layers", fontsize=16)
for f in range(n_funcs):
    for l in range(n_layers):
        ax = axs[f, l] if n_funcs > 1 else axs[l]
        im = ax.hist2d(
            x=np.arange(start_move, n_moves).repeat(test_size),
            y=f1_mine[:, :, l, f].flatten(),
            bins=[n_moves - start_move, 20],
            range=[[start_move, n_moves], [0, 1]],
            cmap="Reds",
        )[3]
        ax.set_title(f"L{l} — {'Attn' if f == 0 else 'MLP'}", fontsize=14)
        ax.set_xlabel("Move"); ax.set_ylabel("F1")

fig.subplots_adjust(right=0.92)
cb = fig.colorbar(im, cax=fig.add_axes([0.94, 0.15, 0.02, 0.7]))
cb.set_label("Games", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
fig.savefig(FIG_DIR / "F1_mine_flipped.jpg", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "F1_mine_flipped.pdf", bbox_inches="tight")
plt.show()

# %%