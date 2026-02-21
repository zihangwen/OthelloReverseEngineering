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
board_seqs_id = board_seqs_id[:, :30]

# %%
pattern_list = dict()
with t.no_grad(), model.trace(board_seqs_id):
    for layer in range(model.cfg.n_layers):
        pattern = model.blocks[layer].attn.hook_pattern.output  # (batch, heads, seq_len, seq_len)
        pattern_list[layer] = pattern.cpu().save()

# %%
# keys = [transformer_lens.utils.get_act_name("pattern", i) for i in range(model.cfg.n_layers)]
# logits, cache = model.run_with_cache(
#     board_seqs_id,
#     names_filter=lambda name: name in keys
# )

# %% plot the attention patterns for each layer (average over games) (display with circuitsvis)
# for layer in range(model.cfg.n_layers):
#     # attention_pattern = cache["pattern", layer]
#     attention_pattern = pattern_list[layer].value
#     mean_attention_pattern = einops.reduce(attention_pattern, "n_games head row col -> head row col", "mean")
#     display(
#         cv.attention.attention_patterns(tokens=["_"]*30, attention=mean_attention_pattern)
#     )

# %%
def diagonal_offsets_mean(A, exclude_first_col = True):
    if exclude_first_col:
        first_col = A[...,0]
        first_col_mean = first_col.mean(dim=-1)
        A = A[...,1:,1:]
    else:
        first_col_mean = None
    n_rows, _ = A.shape[-2:]
    even_diags = []
    odd_diags = []
    for offset in range(-n_rows + 1, 1):
        diag = A.diagonal(offset=offset, dim1=-2, dim2=-1)
        if offset % 2 == 0:
            even_diags.append(diag)
        else:
            odd_diags.append(diag)
    even_diags = t.cat(even_diags, dim=-1)
    odd_diags = t.cat(odd_diags, dim=-1)
    even_mean = even_diags.mean(dim=-1)
    odd_mean = odd_diags.mean(dim=-1)
    
    return even_mean, odd_mean, first_col_mean

# %%
# head_type_all = defaultdict(dict)
# for layer in range(model.cfg.n_layers):
#     attention_pattern = pattern_list[layer].value
#     mean_attention_pattern = einops.reduce(
#         attention_pattern, "n_games head row col -> head row col", "mean"
#     )

#     even_mean, odd_mean, first_col_mean = diagonal_offsets_mean(mean_attention_pattern)
#     for head in range(model.cfg.n_heads):
#         even = even_mean[head].item()
#         odd = odd_mean[head].item()
#         if even > 2 * odd:
#             head_type_all[layer][head] = "Yours head"
#         elif odd > 2 * even:
#             head_type_all[layer][head] = "Mine head"
#         else:
#             head_type_all[layer][head] = "Other"

# with open("temp/attention_head_types.json", "w") as f:
#     json.dump(head_type_all, f, indent=4, sort_keys=True)

# %%
with open("attention/attention_head_types.json", "r") as f:
    head_type_all = json.load(f)

color_map = {
    "Yours head": "red",
    "Mine head": "blue",
    "Other": "gray",
}

# %% per layer plot of attention patterns for each head (heatmap)
# n_heads = model.cfg.n_heads
# for layer in range(model.cfg.n_layers):
#     attention_pattern = pattern_list[layer].value
#     mean_attention_pattern = einops.reduce(
#         attention_pattern, "n_games head row col -> head row col", "mean"
#     ).numpy()
#     # mean_attention_pattern = attention_pattern[0]

#     # Create figure with subplots for each head
#     fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # Adjust based on n_heads
#     axes = axes.flatten()

#     for head in range(n_heads):
#         ax = axes[head]
#         im = ax.imshow(mean_attention_pattern[head], cmap="Blues", vmin=0, vmax=1)
#         ax.set_title(f"Head {head}")
#         ax.set_xlabel("src Position")
#         ax.set_ylabel("dst Position")

#     #plt.colorbar(im, ax=axes)
#     plt.suptitle(f"Layer {layer} Attention Patterns")
#     plt.tight_layout()

#     # Save as PNG
#     # plt.savefig(f"attention_layer_{layer}.png", dpi=150, bbox_inches="tight")
#     plt.show()
#     # plt.close()

# %% plot all layers and heads together
fig, axes = plt.subplots(4, 8, figsize=(16, 8), sharex=True, sharey=True)  # Adjust based on n_heads
# axes = axes.flatten()
n_heads = model.cfg.n_heads
for layer in range(4):
    attention_pattern = pattern_list[layer].value
    mean_attention_pattern = einops.reduce(
        attention_pattern, "n_games head row col -> head row col", "mean"
    ).numpy()
    for head in range(n_heads):
        ax = axes[layer, head]
        head_type = head_type_all[str(layer)][str(head)]
        im = ax.imshow(mean_attention_pattern[head], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"L{layer}H{head} -- {head_type}", color=color_map[head_type])
        ax.set_xlabel("src Position")
        ax.set_ylabel("dst Position")

for ax in axes.flat:
    ax.label_outer()
    #plt.colorbar(im, ax=axes)
    # plt.suptitle(f"Layer {layer} Attention Patterns")

# color bar 0 to 1
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
plt.show()

# %% plot of standard deviation across games for all layers and heads together
fig, axes = plt.subplots(4, 8, figsize=(16, 8), sharex=True, sharey=True)  # Adjust based on n_heads
# axes = axes.flatten()
n_heads = model.cfg.n_heads
for layer in range(4):
    attention_pattern = pattern_list[layer].value
    std_attention_pattern = attention_pattern.std(dim=0).numpy()
    for head in range(n_heads):
        ax = axes[layer, head]
        head_type = head_type_all[str(layer)][str(head)]
        im = ax.imshow(std_attention_pattern[head], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"L{layer}H{head} -- {head_type}", color=color_map[head_type])
        ax.set_xlabel("src Position")
        ax.set_ylabel("dst Position")

for ax in axes.flat:
    ax.label_outer()
    #plt.colorbar(im, ax=axes)
    # plt.suptitle(f"Layer {layer} Attention Patterns")

plt.tight_layout()
plt.show()

# %% table of standard deviation across games for all layers and heads together
# n_heads = model.cfg.n_heads
# n_layer_select = 4
# std_all = dict()
# for layer in range(n_layer_select):
#     attention_pattern = pattern_list[layer].value
#     std_all[layer] = attention_pattern.std(dim=0).mean(dim=(1,2)).numpy()

# from rich.theme import Theme
# light_theme = Theme({
#     "header": "bold black",
#     "layer": "bold black",
#     "blue": "blue",
#     "gray": "dim black",
#     "red": "red",
# })
# console = Console(theme=light_theme, record=True)
# table = Table(title="Attention Pattern Standard Deviation Across Games", show_lines=True, show_header=False)
# for layer in range(n_layer_select):
#     # row = [f"L{layer}"]
#     row = []
#     for head in range(n_heads):
#         head_type = head_type_all[str(layer)][str(head)]
#         head_color = color_map[head_type]
#         row.append(f"[{head_color}]L{layer}H{head}: {std_all[layer][head]:.3f}[/{head_color}]")
        
#     table.add_row(*row)

# console.print(table)

# %% table of standard deviation (separate offsets) across games for all layers and heads together
n_heads = model.cfg.n_heads
n_layer_select = 4
mean_all = dict()
std_all = dict()
for layer in range(n_layer_select):
    attention_pattern = pattern_list[layer].value
    mean_all[layer] = diagonal_offsets_mean(attention_pattern.mean(dim=0))
    std_all[layer] = diagonal_offsets_mean(attention_pattern.std(dim=0))

from rich.theme import Theme
light_theme = Theme({
    "header": "bold black",
    "layer": "bold black",
    "blue": "blue",
    "gray": "dim black",
    "red": "red",
})
console = Console(theme=light_theme, record=True)
table = Table(title="Attention Pattern Standard Deviation Across Games", show_lines=True, show_header=False)
for layer in range(n_layer_select):
    # row = [f"L{layer}"]
    row = []
    yours_mean, mine_mean, _ = mean_all[layer]
    yours_std, mine_std, _ = std_all[layer]
    for head in range(n_heads):
        head_type = head_type_all[str(layer)][str(head)]
        head_color = color_map[head_type]
        
        # mean \u00B1 std
        cell = (
            f"[{head_color}]L{layer}H{head}[/{head_color}]:\n"
            f"  [blue]\u03BC={mine_mean[head]:.3f}\n  \u03C3={mine_std[head]:.3f}[/blue]\n"
            f"  [red]\u03BC={yours_mean[head]:.3f}\n  \u03C3={yours_std[head]:.3f}[/red]"
        )
        row.append(cell)
        
    table.add_row(*row)

console.print(table)

# %%
