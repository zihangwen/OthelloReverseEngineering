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

game_idx = 0
# move_idx = 8
# board_seqs_id = board_seqs_id[0, :20]
board_seqs_id = board_seqs_id[game_idx]

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
attn_z = t.stack([
    cache["z", layer] for layer in range(model.cfg.n_layers)
], dim = 2)

attn_results = einops.einsum(
    attn_z,
    W_O,
    "... layer head d_head, layer head d_head d_model -> ... layer head d_model",
) + model.b_O.unsqueeze(0).unsqueeze(0).unsqueeze(-2) / model.cfg.n_heads  # [batch, seq, layer, head, d_model]
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

# %% probe directions for all layers (attn out, resid mid, mlp out, resid post)
move = 8
n_layers = model.cfg.n_layers
probe_name_list = ["mine", "flipped", "just_played"]
# probe_name = "mine"
probe_layer_specific = {
    name: probes[name][5]
    for name in probes.keys()
}
probe_layer_normalized = {
    name: probes[name] / probes[name].norm(dim=1, keepdim=True)
    for name in probes.keys()
}

for probe_name in probe_name_list:
    resid_pre_probe_dir = einops.einsum(
        resid_pre[0, move, :, :],
        # probes[probe_name],
        # resid_pre_norm[0, move, :, :],
        # probe_layer_normalized[probe_name],
        # "layer d_model, layer d_model ... -> layer ...",
        probe_layer_specific[probe_name],
        "layer d_model, d_model ... -> layer ...",
    ).cpu().numpy()

    attn_out_probe_dir = einops.einsum(
        attn_out[0, move, :, :],
        # probes[probe_name],
        # attn_out_norm[0, move, :, :],
        # probe_layer_normalized[probe_name],
        # "layer d_model, layer d_model ... -> layer ...",
        probe_layer_specific[probe_name],
        "layer d_model, d_model ... -> layer ...",
    ).cpu().numpy()

    resid_mid_probe_dir = einops.einsum(
        resid_mid[0, move, :, :],
        # probes[probe_name],
        # resid_mid_norm[0, move, :, :],
        # probe_layer_normalized[probe_name],
        # "layer d_model, layer d_model ... -> layer ...",
        probe_layer_specific[probe_name],
        "layer d_model, d_model ... -> layer ...",
    ).cpu().numpy()

    mlp_out_probe_dir = einops.einsum(
        mlp_out[0, move, :, :],
        # probes[probe_name],
        # mlp_out_norm[0, move, :, :],
        # probe_layer_normalized[probe_name],
        # "layer d_model, layer d_model ... -> layer ...",
        probe_layer_specific[probe_name],
        "layer d_model, d_model ... -> layer ...",
    ).cpu().numpy()

    resid_post_probe_dir = einops.einsum(
        resid_post[0, move, :, :],
        # probes[probe_name],
        # resid_post_norm[0, move, :, :],
        # probe_layer_normalized[probe_name],
        # "layer d_model, layer d_model ... -> layer ...",
        probe_layer_specific[probe_name],
        "layer d_model, d_model ... -> layer ...",
    ).cpu().numpy()

    fig, axs = plt.subplots(4, n_layers, figsize=(3*n_layers, 3*4+1.5))
    fig.suptitle(f"probe dirs {probe_name}", fontsize=16)

    # Second pass: plot with consistent colorbar
    idx = 0

    vmin = np.nanmin(
        np.concatenate([attn_out_probe_dir, resid_mid_probe_dir, mlp_out_probe_dir, resid_post_probe_dir], axis=0)
    )
    vmax = np.nanmax(
        np.concatenate([attn_out_probe_dir, resid_mid_probe_dir, mlp_out_probe_dir, resid_post_probe_dir], axis=0)
    )
    if probe_name in ["just_played", "flipped"]:
        v_abs = abs(vmax)
    else:
        v_abs = max(abs(vmin), abs(vmax))
    vmin = -v_abs
    vmax = v_abs

    for layer in range(n_layers):
        
        ax = axs[0, layer]
        im = ax.imshow(attn_out_probe_dir[layer], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"attn out (L{layer})")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_yticklabels(list("ABCDEFGH"))

        ax = axs[1, layer]
        im = ax.imshow(resid_mid_probe_dir[layer], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"resid mid (L{layer})")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_yticklabels(list("ABCDEFGH"))

        ax = axs[2, layer]
        im = ax.imshow(mlp_out_probe_dir[layer], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"mlp out (L{layer})")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_yticklabels(list("ABCDEFGH"))

        ax = axs[3, layer]
        im = ax.imshow(resid_post_probe_dir[layer], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"resid post (L{layer})")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_yticklabels(list("ABCDEFGH"))

    # Add one large colorbar on the right
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
    plt.show()

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

# %% check just played with embedding directions
# probe_name = "just_played"
# W_E_all_squares = t.zeros((8, 8, model.cfg.d_model), device=device)
# W_E_all_squares.flatten(start_dim=0, end_dim=1)[arena_utils.ALL_SQUARES] = W_E
# W_E_all_squares_norm = W_E_all_squares / W_E_all_squares.norm(dim=-1, keepdim=True)
# probe_just_played = probes[probe_name]
# probe_just_played_norm = probe_just_played / probe_just_played.norm(dim=1, keepdim=True)

# fig, axs = plt.subplots(2, 4, figsize=(3*4, 3*2+1.5))
# for layer in range(model.cfg.n_layers):
#     proj_scores = einops.einsum(
#         # W_E_all_squares_norm,
#         # probe_just_played_norm[layer],
#         W_E_all_squares,
#         probe_just_played[layer],
#         "row col d_model, d_model row col -> row col",
#     ).cpu().numpy()

#     vmin = np.nanmin(proj_scores)
#     vmax = np.nanmax(proj_scores)
#     v_abs = max(abs(vmin), abs(vmax))
#     vmin = -v_abs
#     vmax = v_abs

#     ax = axs[layer // 4, layer % 4]
#     im = ax.imshow(proj_scores, cmap="RdBu", vmin=vmin, vmax=vmax)
#     ax.set_title(f"L{layer}")
#     ax.set_xticks(range(8))
#     ax.set_yticks(range(8))
#     ax.set_yticklabels(list("ABCDEFGH"))
# # fig.suptitle("Cosine similarity between W_E and just played probes", fontsize=16)
# fig.suptitle("Projection between W_E and just played probes", fontsize=16)
# fig.subplots_adjust(right=0.92)
# cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# fig.colorbar(im, cax=cbar_ax)
# plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
# plt.show()

# %%
# per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
# per_head_residual = einops.rearrange(
#     per_head_residual, "(layer head) ... -> layer head ...", layer=model.cfg.n_layers
# )

# imshow(
#     per_head_logit_diffs,
#     labels={"x": "Head", "y": "Layer"},
#     title="Logit Difference From Each Head",
#     width=600,
# )



# %%
for move in range(22, 24):
    plot_board_states(
        data=test_data,
        game_index=game_idx,
        move=move,
        save_path=BASE_PATH / "figures" / "attention_plots" / "attention_one_game",
        figure_type="png",
    )

# %%
