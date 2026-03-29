# %%
"""
Attention attribution split by Mine / Yours / Other head types.

For a single game of n_moves, computes per-key, per-head attn_out_qk projections
onto probe directions, then aggregates and plots by head type and block type.
"""
import copy
from pathlib import Path
import os
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_PATH)

from board_state_analysis.board_state_utils import (
    setup_model_and_probes,
    load_test_dataset,
    load_head_types,
    compute_probe_projections,
    plot_probe_heatmap_grid,
)
from utils.helper_fns import get_board_states_and_legal_moves
import utils.arena_utils as arena_utils

device = "cuda:1" if t.cuda.is_available() else "cpu"
print(f"Using device: {device}")

FIG_DIR = BASE_PATH / "board_state_analysis" / "fig" / "attention_attr_blocks_w_heads"
os.makedirs(FIG_DIR, exist_ok=True)

# %%
model, n_layers, probes, probe_layer_specific = setup_model_and_probes(device=device)
test_data, board_seqs_id_full, _ = load_test_dataset([], n_games=500, device=device)
head_type_all = load_head_types()

game_idx = 0
n_moves = 9
move = n_moves - 1  # which move position to inspect in Plot 0
board_seqs_id = board_seqs_id_full[game_idx, :n_moves]

# %% Compute per-key, per-head attn_out_qk for each layer
# probe_projs[probe_name][layer]["mine_heads"] -> [batch, seq_q, seq_k, 8, 8] (summed over heads)
# probe_projs[probe_name][layer][stream_key]   -> [batch, seq, 8, 8]
probe_projs = compute_probe_projections(
    model=model,
    board_seqs_id=board_seqs_id,
    probes=probe_layer_specific,
    head_type_all=head_type_all,
    probe_name_list=["mine", "flipped", "just_played"],
    stream_keys=["resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"],
)

# %% Plot 0: residual stream projections [stream_keys × n_layers] at chosen move
stream_keys_plot = ["attn_out", "resid_mid", "mlp_out", "resid_post"]
probe_name_list = ["mine", "flipped", "just_played"]

for probe_name in probe_name_list:
    data = [
        probe_projs[probe_name][layer][stream_key][0, move]
        for stream_key in stream_keys_plot
        for layer in range(n_layers)
    ]
    cell_titles = [
        f"{stream_key} L{l}"
        for stream_key in stream_keys_plot
        for l in range(n_layers)
    ]
    fig = plot_probe_heatmap_grid(
        data=data,
        n_rows=len(stream_keys_plot),
        n_cols=n_layers,
        title=f"Residual stream projections @ {probe_name} probe (move {move})",
        cell_titles=cell_titles,
        scale="symmetric" if probe_name not in ["just_played", "flipped"] else "positive",
    )
    stem = f"stream_proj_{probe_name}_move{move}"
    fig.savefig(FIG_DIR / f"{stem}.jpg", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.show()

# %% Plot 1: single layer, all head types, per source move
head_types = ["mine_heads", "yours_heads", "other_heads"]
color_map = {"yours_heads": "red", "mine_heads": "blue", "other_heads": "gray"}
layer_chosen = 5

board_seqs_square = t.tensor(test_data["decoded_inputs"])[game_idx, :n_moves].unsqueeze(0)
board_states, _, _ = get_board_states_and_legal_moves(board_seqs_square)
cmap_board = ListedColormap(["white", "gray", "black"])

mine_probe_proj = probe_projs["mine"]
fig, axs = plt.subplots(len(head_types) + 1, n_moves, figsize=(3 * n_moves, 3 * (len(head_types) + 1) + 1.5))
fig.suptitle(f"Attention attribution per src move to dst {n_moves-1} @ Mine Probe (L{layer_chosen})", fontsize=16)

for move in range(n_moves):
    ax = axs[0, move]
    ax.imshow(board_states[0, move], cmap=cmap_board, vmin=-1.5, vmax=1.5)
    label = arena_utils.to_board_label(board_seqs_square[0, move])
    ax.set_title(f"Move {move} ({label})")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_yticklabels(list("ABCDEFGH"))

# [0, -1]: batch=0, last query position -> [seq_k, 8, 8] (heads already summed)
temp = [mine_probe_proj[layer_chosen][ht][0, -1] for ht in head_types]
v_abs = max(abs(np.nanmin(temp)), abs(np.nanmax(temp)))

for i, head_type in enumerate(head_types):
    for move in range(n_moves):
        ax = axs[i + 1, move]
        im = ax.imshow(mine_probe_proj[layer_chosen][head_type][0, -1, move],
                       cmap="RdBu", aspect="auto", vmin=-v_abs, vmax=v_abs)
        ax.set_title(f"L{layer_chosen} {head_type} Move {move}", color=color_map[head_type])
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_yticklabels(list("ABCDEFGH"))

fig.subplots_adjust(right=0.92)
fig.colorbar(im, cax=fig.add_axes([0.94, 0.15, 0.02, 0.7]))
plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
stem = f"head_attr_L{layer_chosen}"
fig.savefig(FIG_DIR / f"{stem}.jpg", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
plt.show()

# %% Aggregate over all moves per block type
# For head types: select batch=0, last query pos -> [seq_k, 8, 8], then sum over seq_k
# For stream keys: select batch=0, last seq pos  -> [8, 8]
probe_projs_blocks = copy.deepcopy(probe_projs)
for probe_name in probe_projs_blocks:
    for layer in range(n_layers):
        for block in head_types:
            probe_projs_blocks[probe_name][layer][block] = (
                probe_projs_blocks[probe_name][layer][block][0, -1].sum(axis=0)
            )  # [seq_k, 8, 8] -> [8, 8]
        for key in ["resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"]:
            probe_projs_blocks[probe_name][layer][key] = (
                probe_projs_blocks[probe_name][layer][key][0, -1]
            )  # [8, 8]

block_types = ["mine_heads", "yours_heads", "mlp_out", "resid_post"]

# %% Plot 2: all probes, all layers, all block types
for probe_name in probe_projs_blocks:
    data = [
        probe_projs_blocks[probe_name][layer][block]
        for block in block_types
        for layer in range(n_layers)
    ]
    cell_titles = [
        f"L{layer} - {block}"
        for block in block_types
        for layer in range(n_layers)
    ]
    cell_colors = [
        color_map.get(block, "black")
        for block in block_types
        for _ in range(n_layers)
    ]
    fig = plot_probe_heatmap_grid(
        data=data,
        n_rows=len(block_types),
        n_cols=n_layers,
        title=f"Attribution aggregated over all moves @ {probe_name.capitalize()} Probe",
        cell_titles=cell_titles,
        cell_title_colors=cell_colors,
        scale="symmetric" if probe_name not in ["just_played", "flipped"] else "positive",
    )
    stem = f"head_attr_blocks_{probe_name}"
    fig.savefig(FIG_DIR / f"{stem}.jpg", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.show()

# %% Save board states for reference
from utils.plot_utils import plot_board_states
for m in range(9):
    plot_board_states(
        data=test_data,
        game_index=game_idx,
        move=m,
        save_path=FIG_DIR / "boards",
        figure_type="pdf",
    )

# %%