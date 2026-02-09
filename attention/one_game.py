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
n_moves = 9
n_layers_selected = 4
# move_idx = 8
# board_seqs_id = board_seqs_id[0, :20]
board_seqs_id = board_seqs_id[game_idx, :n_moves]

# %%
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

# %%
probe_list = ["mine", "flipped", "just_played"]
probe_projs = {}
for probe_name in probe_list:
# probe_name = "mine"

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

    resid_pre_probe_dir = einops.einsum(
        resid_pre[0, -1, :, :],
        probe_layer_specific[probe_name],
        "layer d_model, d_model ... -> layer ...",
    ).cpu().numpy()

    attn_out_probe_dir = einops.einsum(
        attn_out[0, -1, :, :],
        probe_layer_specific[probe_name],
        "layer d_model, d_model ... -> layer ...",
    ).cpu().numpy()

    resid_mid_probe_dir = einops.einsum(
        resid_mid[0, -1, :, :],
        probe_layer_specific[probe_name],
        "layer d_model, d_model ... -> layer ...",
    ).cpu().numpy()

    mlp_out_probe_dir = einops.einsum(
        mlp_out[0, -1, :, :],
        probe_layer_specific[probe_name],
        "layer d_model, d_model ... -> layer ...",
    ).cpu().numpy()

    resid_post_probe_dir = einops.einsum(
        resid_post[0, -1, :, :],
        probe_layer_specific[probe_name],
        "layer d_model, d_model ... -> layer ...",
    ).cpu().numpy()

    attn_out_qk = t.stack([
        einops.einsum(
            cache["v", layer],
            cache["pattern", layer],
            W_O[layer],
            "batch seq_k head d_head, batch head seq_q seq_k, head d_head d_model -> batch seq_q seq_k head d_model",
        ) + model.b_O[layer] / n_moves / model.cfg.n_heads  # evenly distribute bias
        for layer in range(model.cfg.n_layers)
    ], dim = 3)  # [batch, seq_q, seq_k, layer, head, d_model]

    attn_out_qk_last_probe_dir = einops.einsum(
        attn_out_qk[0, -1],
        probe_layer_specific[probe_name],
        "seq_k layer head d_model, d_model ... -> seq_k layer head ...",
    ).cpu().numpy()

    one_probe_proj = defaultdict(dict)
    for layer in range(n_layers):
        mine_heads = []
        yours_heads = []
        other_heads = []
        for head in range(model.cfg.n_heads):
            head_type = head_type_all[str(layer)][str(head)]
            if head_type == "Mine head":
                mine_heads.append(head)
            elif head_type == "Yours head":
                yours_heads.append(head)
            else:
                other_heads.append(head)
        one_probe_proj[layer]["mine_heads"] = attn_out_qk_last_probe_dir[:, layer, mine_heads] # [move, mine_heads, row, col]
        one_probe_proj[layer]["yours_heads"] = attn_out_qk_last_probe_dir[:, layer, yours_heads] # [move, yours_heads, row, col]
        one_probe_proj[layer]["other_heads"] = attn_out_qk_last_probe_dir[:, layer, other_heads] # [move, other_heads, row, col]
        one_probe_proj[layer]["resid_pre"] = resid_pre_probe_dir[layer]
        one_probe_proj[layer]["attn_out"] = attn_out_probe_dir[layer]
        one_probe_proj[layer]["resid_mid"] = resid_mid_probe_dir[layer]
        one_probe_proj[layer]["mlp_out"] = mlp_out_probe_dir[layer]
        one_probe_proj[layer]["resid_post"] = resid_post_probe_dir[layer]
    
    probe_projs[probe_name] = one_probe_proj

# %%
head_types = ["mine_heads", "yours_heads", "other_heads"]
color_map = {
    "yours_heads": "red",
    "mine_heads": "blue",
    "other_heads": "gray",
}
num_heads_types = len(head_types)

# %% Plot attention attribution per src move to dst move @ mine probe for a particular layer and all head types
mine_probe_proj = probe_projs["mine"]
layer_chosen = 1

fig, axs = plt.subplots(num_heads_types+1, n_moves, figsize=(3*n_moves, 3*(num_heads_types+1)+1.5))
fig.suptitle(f"Attention attribution per src move to dst {n_moves-1} move @ Mine Probe", fontsize=16)
from matplotlib.colors import ListedColormap
cmap_board = ListedColormap(["white", "gray", "black"])
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

temp = [
    mine_probe_proj[layer_chosen][head_type].sum(axis=1)
    for head_type in head_types
]

vmin = np.nanmin(temp)
vmax = np.nanmax(temp)
v_abs = max(abs(vmin), abs(vmax))
vmin = -v_abs
vmax = v_abs

for i, head_type in enumerate(head_types):
    for move in range(n_moves):
        ax = axs[i+1, move]
        im = ax.imshow(mine_probe_proj[layer_chosen][head_type][move].sum(axis=0), cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"L{layer_chosen} - {head_type} - Src Move {move}", color=color_map[head_type])
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_yticklabels(list("ABCDEFGH"))
# Add one large colorbar on the right
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
plt.show()

# %% head type aggregated over all moves and heads (inside type) per layer
import copy 
probe_projs_blocks = copy.deepcopy(probe_projs)
for probe_name in probe_projs_blocks.keys():
    for layer in range(n_layers):
        for block in head_types:
            probe_projs_blocks[probe_name][layer][block] = einops.reduce(
                probe_projs_blocks[probe_name][layer][block],
                "move head row col -> row col",
                "sum",
            )
block_types = ['mine_heads', 'yours_heads'] + ["mlp_out", "resid_post"]
num_block_types = len(block_types)

# %% Plot attribution for all blocks to @ all probes for all layers
for probe_name in probe_projs_blocks.keys():
    fig, axs = plt.subplots(num_block_types, n_layers, figsize=(3*n_layers, 3*num_block_types+1.5))
    fig.suptitle(f"Attention attribution aggregated over all moves and heads @ {probe_name.capitalize()} Probe", fontsize=16)

    temp = [
        probe_projs_blocks[probe_name][layer][block]
        for layer in range(n_layers)
        for block in block_types
    ]
    vmin = np.nanmin(temp)
    vmax = np.nanmax(temp)
    v_abs = max(abs(vmin), abs(vmax))
    vmin = -v_abs
    vmax = v_abs

    for i, block in enumerate(block_types):
        for layer in range(n_layers):
            ax = axs[i, layer]
            im = ax.imshow(probe_projs_blocks[probe_name][layer][block], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_title(f"L{layer} - {block}", color=color_map.get(block, "black"))
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_yticklabels(list("ABCDEFGH"))
    # Add one large colorbar on the right
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
    plt.show()

# %% TEST
probe_test_chosen = "flipped"
layer_test_chosen = 3
print(
    f"maximize difference between resid_post and sum of other blocks @ {probe_test_chosen} probe at layer {layer_test_chosen}",
    (probe_projs[probe_test_chosen][layer_test_chosen]["resid_post"] - (
        probe_projs[probe_test_chosen][layer_test_chosen]["mlp_out"] + 
        probe_projs_blocks[probe_test_chosen][layer_test_chosen]["mine_heads"] + 
        probe_projs_blocks[probe_test_chosen][layer_test_chosen]["yours_heads"] + 
        probe_projs_blocks[probe_test_chosen][layer_test_chosen]["other_heads"] + 
        probe_projs[probe_test_chosen][layer_test_chosen]["resid_pre"]
    )).max()
)

# %%
    # temp = [
    #     probe_projs_blocks[probe_name][layer][head_type]
    #     for layer in range(n_layers)
    #     for head_type in head_types
    # ]
    # vmin = np.nanmin(temp)
    # vmax = np.nanmax(temp)
    # v_abs = max(abs(vmin), abs(vmax))
    # vmin = -v_abs
    # vmax = v_abs

    # for i, head_type in enumerate(head_types):
    #     for layer in range(n_layers):
    #         ax = axs[i, layer]
    #         im = ax.imshow(probe_projs_blocks[probe_name][layer][head_type], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
    #         ax.set_title(f"L{layer} - {head_type}", color=color_map[head_type])
    #         ax.set_xticks(range(8))
    #         ax.set_yticks(range(8))
    #         ax.set_yticklabels(list("ABCDEFGH"))
    # # Add one large colorbar on the right
    # fig.subplots_adjust(right=0.92)
    # cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
    # plt.show()

# %%
# fig, axs = plt.subplots(n_layers_selected+1, n_moves, figsize=(3*n_moves, 3*(n_layers_selected+1)+1.5))
# fig.suptitle(f"Attention attribution per key move to query {n_moves-1} move @ Mine Probe", fontsize=16)

# from matplotlib.colors import ListedColormap
# cmap_board = ListedColormap(["white", "gray", "black"])

# # we need boundaries around -1,0,1
# bounds_board = [-1.5, -0.5, 0.5, 1.5]
# board_seqs_square = t.tensor(test_data["decoded_inputs"])
# board_seqs_square = board_seqs_square[game_idx, :n_moves].unsqueeze(0) # [move, 8, 8]
# board_states, legal_moves, _ = get_board_states_and_legal_moves(board_seqs_square)

# for move in range(n_moves):
#     ax = axs[0, move]
#     im = ax.imshow(board_states[0, move], cmap=cmap_board, vmin=-1.5, vmax=1.5)

#     sqaure_label = arena_utils.to_board_label(board_seqs_square[0, move])
#     ax.set_title(f"Move {move} ({sqaure_label}), {'white' if move % 2 == 0 else 'black'} to play")
#     ax.set_xticks(range(8))
#     ax.set_yticks(range(8))
#     ax.set_yticklabels(list("ABCDEFGH"))


# vmin = np.nanmin(attn_out_qk_last_probe_dir[:,1:1+n_layers_selected])
# vmax = np.nanmax(attn_out_qk_last_probe_dir[:,1:1+n_layers_selected])
# v_abs = max(abs(vmin), abs(vmax))
# vmin = -v_abs
# vmax = v_abs

# for layer in range(1, 1+n_layers_selected):
#     for move in range(n_moves):
#         ax = axs[layer, move]
#         im = ax.imshow(attn_out_qk_last_probe_dir[move, layer], cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax)
#         ax.set_title(f"L{layer} - Key Move {move}")
#         ax.set_xticks(range(8))
#         ax.set_yticks(range(8))
#         ax.set_yticklabels(list("ABCDEFGH"))

# # Add one large colorbar on the right
# fig.subplots_adjust(right=0.92)
# cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# fig.colorbar(im, cax=cbar_ax)

# plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
# plt.show()

# %%