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
from utils.plot_utils import (
    plot_board_states,
)
from utils.arena_plotly_utils import (
    imshow,
)

device = "cuda:1" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = circuits_utils.get_model(model_name, device)
n_layers = model.cfg.n_layers

W_Q = model.W_Q.detach().clone()  # [layer, head, d_model, d_head]
W_K = model.W_K.detach().clone()  # [layer, head, d_model, d_head]
W_O = model.W_O.detach().clone()  # [layer, head, d_head, d_model]
W_V = model.W_V.detach().clone()  # [layer, head, d_model, d_head]

W_E = model.W_E[1:].detach().clone()  # [vocab_size, d_model]
W_U = model.W_U[:, 1:].detach().clone()  # [d_model, 60]

# %%
probes = load_fold_probes_and_normalize(n_layers, device)

# %%
with open("attention/attention_head_types.json", "r") as f:
    head_type_all = json.load(f)

# %% Load the test dataset and process
test_size = 500
custom_functions = [
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

game_idx = 25
n_moves = 9
n_layers_selected = 4
# move_idx = 8
# board_seqs_id = board_seqs_id[0, :20]
board_seqs_id = board_seqs_id[game_idx, :n_moves]

# %%
# pattern_list = dict()
# attn_scores_list = dict()
# with t.no_grad(), model.trace(board_seqs_id):
#     for layer in range(model.cfg.n_layers):
#         pattern = model.blocks[layer].attn.hook_pattern.output  # (batch, heads, seq_len, seq_len)
#         attn_scores = model.blocks[layer].attn.hook_attn_scores.output  # (batch, heads, seq_len, seq_len)
#         pattern_list[layer] = pattern.cpu().save()
#         attn_scores_list[layer] = attn_scores.cpu().save()

# keys = [transformer_lens.utils.get_act_name("result", i) for i in range(model.cfg.n_layers)]
logits, cache = model.run_with_cache(
    board_seqs_id,
)

# %%
probe_layer_specific = {
    name: probes[name][5]
    for name in probes.keys()
}
probe_layer_normalized = {
    name: probes[name] / probes[name].norm(dim=1, keepdim=True)
    for name in probes.keys()
}

# %% test
# einops.einsum(
#     cache["v", 0],
#     cache["pattern", 0],
#     "batch seq_k head d_head, batch head seq_q seq_k -> batch seq_q head d_head",
# ) == cache["z", 0]

# einops.einsum(
#     cache["v", 0],
#     cache["pattern", 0],
#     W_O[0],
#     "batch seq_k head d_head, batch head seq_q seq_k, head d_head d_model -> batch seq_q d_model",
# ) + model.b_O[0] == cache["attn_out", 0]

# (einops.einsum(
#     cache["v", 0],
#     cache["pattern", 0],
#     W_O[0],
#     "batch seq_k head d_head, batch head seq_q seq_k, head d_head d_model -> batch seq_q seq_k d_model",
# ) + model.b_O[0] / n_moves).sum(-2) - cache["attn_out", 0]

# temp = einops.einsum(
#     cache["blocks.0.ln1.hook_normalized"],
#     W_V[0],
#     "batch seq d_model, head d_model d_head -> batch seq head d_head",
# ) + model.b_V[0]

# (einops.einsum(
#     temp,
#     cache["pattern", 0],
#     W_O[0],
#     "batch seq_k head d_head, batch head seq_q seq_k, head d_head d_model -> batch seq_q seq_k d_model",
# ) + model.b_O[0] / n_moves).sum(-2) - cache["attn_out", 0]


# %% plot per k for last move
fig, axs = plt.subplots(n_layers_selected+1, n_moves, figsize=(3*n_moves, 3*(n_layers_selected+1)+1.5))
fig.suptitle(f"Attention attribution per key move to query {n_moves-1} move @ Mine Probe", fontsize=16)

from matplotlib.colors import ListedColormap
cmap_board = ListedColormap(["white", "gray", "black"])

# we need boundaries around -1,0,1
bounds_board = [-1.5, -0.5, 0.5, 1.5]
board_seqs_square = t.tensor(test_data["decoded_inputs"])
board_seqs_square = board_seqs_square[game_idx, :n_moves].unsqueeze(0) # [move, 8, 8]
board_states, legal_moves, _ = get_board_states_and_legal_moves(board_seqs_square)

for move in range(n_moves):
    ax = axs[0, move]
    im = ax.imshow(board_states[0, move], cmap=cmap_board, vmin=-1.5, vmax=1.5)

    sqaure_label = arena_utils.to_board_label(board_seqs_square[0, move])
    ax.set_title(f"Move {move} ({sqaure_label}), {'white' if move % 2 == 0 else 'black'} to play")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_yticklabels(list("ABCDEFGH"))

attn_out_qk = t.stack([
    einops.einsum(
        cache["v", layer],
        cache["pattern", layer],
        W_O[layer],
        "batch seq_k head d_head, batch head seq_q seq_k, head d_head d_model -> batch seq_q seq_k d_model",
    ) + model.b_O[layer] / n_moves
    for layer in range(model.cfg.n_layers)
], dim = 3)

attn_out_qk_last_probe_dir = einops.einsum(
    attn_out_qk[0, -1],
    probe_layer_specific["mine"],
    "seq_k layer d_model, d_model ... -> seq_k layer ...",
).cpu().numpy()

vmin = np.nanmin(attn_out_qk_last_probe_dir[:,1:1+n_layers_selected])
vmax = np.nanmax(attn_out_qk_last_probe_dir[:,1:1+n_layers_selected])
v_abs = max(abs(vmin), abs(vmax))
vmin = -v_abs
vmax = v_abs

for layer in range(1, 1+n_layers_selected):
    for move in range(n_moves):
        ax = axs[layer, move]
        im = ax.imshow(attn_out_qk_last_probe_dir[move, layer], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"L{layer} - Key Move {move}")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_yticklabels(list("ABCDEFGH"))

# Add one large colorbar on the right
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
plt.show()

# %%
def orthonormalize_dirs_qr(*dirs: t.Tensor, eps: float = 1e-8) -> list[t.Tensor]:
    """
    Orthonormalize k direction tensors of shape (d_model, ...).
    Returns k tensors q_i with same shape, orthonormal per each '...' index.
    """
    # Stack into (d_model, ..., k)
    D = t.stack(list(dirs), dim=-1)  # last dim is k
    D = t.nan_to_num(D)

    d_model = D.shape[0]
    extra_shape = D.shape[1:-1]
    k = D.shape[-1]
    m = int(t.tensor(extra_shape).prod().item()) if len(extra_shape) else 1

    # Reshape to a batch of matrices: (m, d_model, k)
    Dm = D.reshape(d_model, m, k).permute(1, 0, 2).contiguous()

    # QR per batch item
    # Q: (m, d_model, k) with orthonormal columns (when directions are independent)
    Qm, _ = t.linalg.qr(Dm, mode="reduced")

    # Back to (d_model, ..., k)
    Q = Qm.permute(1, 0, 2).reshape(d_model, *extra_shape, k)

    # Split into list of (d_model, ...)
    return [Q[..., i] for i in range(k)]

# %% plot per k for last move (flipped probe dir + just played probe dir)
fig, axs = plt.subplots(n_layers_selected+1, n_moves, figsize=(3*n_moves, 3*(n_layers_selected+1)+1.5))
fig.suptitle(f"Attention attribution per key move to query {n_moves-1} move @ Mine Probe", fontsize=16)

from matplotlib.colors import ListedColormap
cmap_board = ListedColormap(["white", "gray", "black"])

# we need boundaries around -1,0,1
bounds_board = [-1.5, -0.5, 0.5, 1.5]
board_seqs_square = t.tensor(test_data["decoded_inputs"])
board_seqs_square = board_seqs_square[game_idx, :n_moves].unsqueeze(0) # [move, 8, 8]
board_states, legal_moves, _ = get_board_states_and_legal_moves(board_seqs_square)

for move in range(n_moves):
    ax = axs[0, move]
    im = ax.imshow(board_states[0, move], cmap=cmap_board, vmin=-1.5, vmax=1.5)

    sqaure_label = arena_utils.to_board_label(board_seqs_square[0, move])
    ax.set_title(f"Move {move} ({sqaure_label}), {'white' if move % 2 == 0 else 'black'} to play")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_yticklabels(list("ABCDEFGH"))

attn_out_qk_list = []
for layer in range(model.cfg.n_layers):
    hook_norm = cache[f"blocks.{layer}.ln1.hook_normalized"]  # (batch, seq, d_model)

    # ---- collect & normalize probe directions ----
    dirs = []
    for key in ["flipped", "just_played"]:
        d = probe_layer_specific[key]
        d = d / d.norm(dim=0, keepdim=True)  # normalize over d_model
        dirs.append(d)

    # Stack into D: (k, d_model, ...)
    D = t.stack(dirs, dim=0)
    D = t.nan_to_num(D)

    k = D.shape[0]
    d_model = D.shape[1]
    extra_shape = D.shape[2:]                 # "..."
    m = int(t.tensor(extra_shape).prod().item()) if len(extra_shape) > 0 else 1

    # Flatten "..." so we can solve per-index
    D_flat = D.reshape(k, d_model, m)         # (k, d_model, m)

    # ---- compute s = D^T x, where x = hook_norm ----
    # s: (batch, seq, k, m)
    s = einops.einsum(
        hook_norm, D_flat,
        "batch seq d_model, k d_model m -> batch seq k m"
    )

    # ---- Gram matrix G = D^T D ----
    # G: (m, k, k)
    G = einops.einsum(
        D_flat, D_flat,
        "k d_model m, j d_model m -> m k j"
    )

    # ---- solve (G + eps I) c = s for c ----
    eps = 1e-6
    I = t.eye(k, device=G.device, dtype=G.dtype)[None, :, :]  # (1,k,k)
    G_reg = G + eps * I                                       # (m,k,k)

    # reshape to use torch.linalg.solve: (m,k,k) and (m, batch*seq, k)
    bs = hook_norm.shape[0] * hook_norm.shape[1]
    s_mbk = s.reshape(bs, k, m).permute(2, 0, 1).contiguous()  # (m, bs, k)

    # Solve: (m,k,k) @ (m,bs,k,1) = (m,bs,k,1)
    c_mbk = t.linalg.solve(G_reg, s_mbk.transpose(1, 2)).transpose(1, 2)  # (m, bs, k)

    # ---- reconstruct projected hook_norm: x_proj = sum_i c_i * d_i ----
    # x_proj_flat: (bs, d_model, m)
    x_proj_flat = einops.einsum(
        c_mbk, D_flat.permute(2, 0, 1),   # (m,bs,k) and (m,k,d_model)
        "m bs k, m k d_model -> bs d_model m"
    )

    # unflatten back to (batch, seq, d_model, ...)
    x_proj = x_proj_flat.reshape(hook_norm.shape[0], hook_norm.shape[1], d_model, *extra_shape)

    # ---- push through W_V to get projected v = p ----
    new_v = einops.einsum(
        x_proj, W_V[layer],
        "batch seq d_model ..., head d_model d_head -> batch seq head d_head ..."
    ) + model.b_V[layer, ..., None, None]

    new_v = t.nan_to_num(new_v)
    
    # ----- orthogonalize probe directions and compute separately ----- #
    # hook_norm = cache[f"blocks.{layer}.ln1.hook_normalized"]
    # flipped_probe_dir = probe_layer_specific["flipped"]
    # flipped_probe_dir = flipped_probe_dir / flipped_probe_dir.norm(dim=0, keepdim=True)
    
    # just_played_probe_dir = probe_layer_specific["just_played"]
    # just_played_probe_dir = just_played_probe_dir / just_played_probe_dir.norm(dim=0, keepdim=True)

    # q_flipped, q_just_played = orthonormalize_dirs_qr(flipped_probe_dir, just_played_probe_dir)

    # flipped_v = einops.einsum(
    #     hook_norm,
    #     q_flipped,
    #     q_flipped,
    #     W_V[layer],
    #     "batch seq d_model, d_model ..., d_model2 ..., head d_model2 d_head -> batch seq head d_head ...",
    # ) + model.b_V[layer,...,None,None]

    # just_played_v = einops.einsum(
    #     hook_norm,
    #     q_just_played,
    #     q_just_played,
    #     W_V[layer],
    #     "batch seq d_model, d_model ..., d_model2 ..., head d_model2 d_head -> batch seq head d_head ...",
    # ) + model.b_V[layer,...,None,None]
    # just_played_v = t.nan_to_num(just_played_v)

    # new_v = flipped_v + just_played_v
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #

    attn_out_one = einops.einsum(
        new_v,
        cache["pattern", layer],
        W_O[layer],
        "batch seq_k head d_head ..., batch head seq_q seq_k, head d_head d_model -> batch seq_q seq_k d_model ...",
    ) + (model.b_O[layer,...,None,None] / n_moves)

    attn_out_qk_list.append(attn_out_one)

attn_out_qk = t.stack(attn_out_qk_list, dim=3)

attn_out_qk_flipped_probe_dir = einops.einsum(
    attn_out_qk[0, -1],
    probe_layer_specific["mine"],
    "seq_k layer d_model ..., d_model ... -> seq_k layer ...",
).cpu().numpy()

vmin = np.nanmin(attn_out_qk_flipped_probe_dir[:,1:1+n_layers_selected])
vmax = np.nanmax(attn_out_qk_flipped_probe_dir[:,1:1+n_layers_selected])
v_abs = max(abs(vmin), abs(vmax))
vmin = -v_abs
vmax = v_abs

for layer in range(1, 1+n_layers_selected):
    for move in range(n_moves):
        ax = axs[layer, move]
        im = ax.imshow(attn_out_qk_flipped_probe_dir[move, layer], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"L {layer} - Key Move {move}")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_yticklabels(list("ABCDEFGH"))

# Add one large colorbar on the right
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
plt.show()

# %%
attn_z = t.stack([
    cache["z", layer] for layer in range(model.cfg.n_layers)
], dim = 2)

attn_results = einops.einsum(
    attn_z,
    W_O,
    "... layer head d_head, layer head d_head d_model -> ... layer head d_model",
)
resid_pre = t.stack([
    cache["resid_pre", layer] for layer in range(model.cfg.n_layers)
], dim = 2)
attn_out = t.stack([
    cache["attn_out", layer] for layer in range(model.cfg.n_layers)
], dim = 2)
resid_mid = t.stack([
    cache["resid_mid", layer] for layer in range(model.cfg.n_layers)
], dim = 2)
mlp_out = t.stack([
    cache["mlp_out", layer] for layer in range(model.cfg.n_layers)
], dim = 2)
resid_post = t.stack([
    cache["resid_post", layer] for layer in range(model.cfg.n_layers)
], dim = 2)

# resid_pre_norm = resid_pre / resid_pre.norm(dim=-1, keepdim=True)
# attn_out_norm = attn_out / attn_out.norm(dim=-1, keepdim=True)
# resid_mid_norm = resid_mid / resid_mid.norm(dim=-1, keepdim=True)
# mlp_out_norm = mlp_out / mlp_out.norm(dim=-1, keepdim=True)
# resid_post_norm = resid_post / resid_post.norm(dim=-1, keepdim=True)

# attn_head_labels = [f"L{layer}H{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]

# %% probe directions for all layers (attn out, resid mid, mlp out, resid post)
probe_name = "flipped"

resid_post_probe_dir = einops.einsum(
    resid_post[0, :n_moves, :n_layers_selected, :],
    # probes[probe_name],
    # resid_post_norm[0, move, :, :],
    # probe_layer_normalized[probe_name],
    # "layer d_model, layer d_model ... -> layer ...",
    probe_layer_specific[probe_name],
    "n_move layer d_model, d_model ... -> n_move layer ...",
).cpu().numpy()

fig, axs = plt.subplots(n_layers_selected+1, n_moves, figsize=(3*n_moves, 3*(n_layers_selected+1)+1.5))
fig.suptitle(f"resid_post @ probe dirs {probe_name}", fontsize=16)

from matplotlib.colors import ListedColormap
cmap_board = ListedColormap(["white", "gray", "black"])

# we need boundaries around -1,0,1
bounds_board = [-1.5, -0.5, 0.5, 1.5]
board_seqs_square = t.tensor(test_data["decoded_inputs"])
board_seqs_square = board_seqs_square[game_idx, :n_moves].unsqueeze(0) # [move, 8, 8]
board_states, legal_moves, _ = get_board_states_and_legal_moves(board_seqs_square)

for move in range(n_moves):
    ax = axs[0, move]
    im = ax.imshow(board_states[0, move], cmap=cmap_board, vmin=-1.5, vmax=1.5)

    sqaure_label = arena_utils.to_board_label(board_seqs_square[0, move])
    ax.set_title(f"Move {move} ({sqaure_label}), {'white' if move % 2 == 0 else 'black'} to play")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_yticklabels(list("ABCDEFGH"))

vmin = np.nanmin(resid_post_probe_dir)
vmax = np.nanmax(resid_post_probe_dir)
v_abs = abs(vmax)
vmin = -v_abs
vmax = v_abs

for layer in range(n_layers_selected):
    for move in range(n_moves):
        ax = axs[layer+1, move]
        im = ax.imshow(resid_post_probe_dir[move, layer], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"L{layer} - Move {move}")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_yticklabels(list("ABCDEFGH"))

# Add one large colorbar on the right
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
plt.show()

# %%
resid_pre_probe_dir = einops.einsum(
    resid_pre[0, :n_moves, :n_layers_selected, :],
    # probes[probe_name],
    # resid_pre_norm[0, move, :, :],
    # probe_layer_normalized[probe_name],
    # "layer d_model, layer d_model ... -> layer ...",
    probe_layer_specific[probe_name],
    "n_move layer d_model, d_model ... -> n_move layer ...",
).cpu().numpy()

fig, axs = plt.subplots(n_layers_selected+1, n_moves, figsize=(3*n_moves, 3*(n_layers_selected+1)+1.5))
fig.suptitle(f"resid_pre @ probe dirs {probe_name}", fontsize=16)

from matplotlib.colors import ListedColormap
cmap_board = ListedColormap(["white", "gray", "black"])

# we need boundaries around -1,0,1
bounds_board = [-1.5, -0.5, 0.5, 1.5]
board_seqs_square = t.tensor(test_data["decoded_inputs"])
board_seqs_square = board_seqs_square[game_idx, :n_moves].unsqueeze(0) # [move, 8, 8]
board_states, legal_moves, _ = get_board_states_and_legal_moves(board_seqs_square)

for move in range(n_moves):
    ax = axs[0, move]
    im = ax.imshow(board_states[0, move], cmap=cmap_board, vmin=-1.5, vmax=1.5)

    sqaure_label = arena_utils.to_board_label(board_seqs_square[0, move])
    ax.set_title(f"Move {move} ({sqaure_label}), {'white' if move % 2 == 0 else 'black'} to play")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_yticklabels(list("ABCDEFGH"))

vmin = np.nanmin(resid_pre_probe_dir)
vmax = np.nanmax(resid_pre_probe_dir)
v_abs = abs(vmax)
vmin = -v_abs
vmax = v_abs

for layer in range(n_layers_selected):
    for move in range(n_moves):
        ax = axs[layer+1, move]
        im = ax.imshow(resid_pre_probe_dir[move, layer], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"L{layer} - Move {move}")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_yticklabels(list("ABCDEFGH"))

# Add one large colorbar on the right
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
plt.show()

# %% probe directions for attention heads only (0-3 layers)
# move = 20
# n_layer_select = 4
# n_heads = model.cfg.n_heads
# probe_name_list = ["mine", "flipped", "just_played"]
# # probe_name = "mine"

# color_map = {
#     "Yours head": "red",
#     "Mine head": "blue",
#     "Other": "gray",
# }

# for probe_name in probe_name_list:
#     attn_probe_dir = einops.einsum(
#         attn_results[0, move, :, :],
#         probes[probe_name],
#         "layer head d_model, layer d_model ... -> layer head ...",
#     ).cpu().numpy()[:n_layer_select]

#     fig, axs = plt.subplots(n_layer_select, n_heads, figsize=(3*n_heads, 3*n_layer_select+1.5))
#     fig.suptitle(f"probe dirs {probe_name}", fontsize=16)

#     # Second pass: plot with consistent colorbar
#     idx = 0

#     vmin = np.nanmin(attn_probe_dir)
#     vmax = np.nanmax(attn_probe_dir)
#     v_abs = max(abs(vmin), abs(vmax))
#     vmin = -v_abs
#     vmax = v_abs

#     for layer in range(n_layer_select):
#         for head in range(n_heads):
#             ax = axs[layer, head]
#             head_type = head_type_all[str(layer)][str(head)]
#             im = ax.imshow(attn_probe_dir[layer, head], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
#             ax.set_title(f"L{layer}H{head} -- {head_type}", color=color_map[head_type])
#             ax.set_xticks(range(8))
#             ax.set_yticks(range(8))
#             ax.set_yticklabels(list("ABCDEFGH"))

#     # Add one large colorbar on the right
#     fig.subplots_adjust(right=0.92)
#     cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
#     fig.colorbar(im, cax=cbar_ax)

#     plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
#     plt.show()


# %% attribution to a specific square
# probe_name = "mine"
# check_square_label = "B5"
# check_square_id = arena_utils.to_square(check_square_label)
# check_row = check_square_id // 8
# check_col = check_square_id % 8

# attn_probe_dir = einops.einsum(
#     attn_results[0, move, :, :],
#     probes[probe_name][:],
#     "layer head d_model, layer d_model ... -> layer head ...",
# ).cpu()

# # attn_out_probe_dir = einops.einsum(
# #     attn_out[0, move, :, :],
# #     probes[probe_name],
# #     "layer d_model, layer d_model ... -> layer ...",
# # ).cpu().numpy()

# # resid_mid_probe_dir = einops.einsum(
# #     resid_mid[0, move, :, :],
# #     probes[probe_name],
# #     "layer d_model, layer d_model ... -> layer ...",
# # ).cpu().numpy()

# mlp_out_probe_dir = einops.einsum(
#     mlp_out[0, move, :, :],
#     probes[probe_name][:],
#     "layer d_model, layer d_model ... -> layer ...",
# ).cpu()

# # resid_post_probe_dir = einops.einsum(
# #     resid_post[0, move, :, :],
# #     probes[probe_name],
# #     "layer d_model, layer d_model ... -> layer ...",
# # ).cpu().numpy()

# attr_probe_dir = t.cat([
#     attn_probe_dir,
#     mlp_out_probe_dir.unsqueeze(1),
# ], dim=1)[..., check_row, check_col].numpy()  # (layer, head+1)

# vmin = np.nanmin(attr_probe_dir)
# vmax = np.nanmax(attr_probe_dir)
# v_abs = max(abs(vmin), abs(vmax))
# vmin = -v_abs
# vmax = v_abs

# fig, ax = plt.subplots(figsize=(12, 6))
# im = ax.imshow(attr_probe_dir, cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
# ax.set_title(f"Attribution to {probe_name} probe dir at {check_square_label} (row {check_row}, col {check_col})", fontsize=16)
# ax.set_xlabel("Head & MLP")
# ax.set_ylabel("Layer")
# ax.set_xticks(range(attr_probe_dir.shape[1]))
# ax.set_yticks(range(attr_probe_dir.shape[0]))
# ax.set_yticklabels([f"L{layer}" for layer in range(model.cfg.n_layers)])
# ax.set_xticklabels([f"H{head}" for head in range(model.cfg.n_heads)] + ["MLP"])
# fig.colorbar(im, ax=ax)
# plt.show()

# %%
# for move in range(27, 29):
#     plot_board_states(
#         data=test_data,
#         game_index=game_idx,
#         move=move,
#         save_path=BASE_PATH / "figures" / "attention_plots" / "attention_one_game",
#         figure_type="png",
#     )
