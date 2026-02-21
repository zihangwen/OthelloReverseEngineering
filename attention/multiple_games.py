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

# %% Load the test dataset and process
test_size = 2
custom_functions = [
    othello_utils.games_batch_to_flipped_classifier_input_BLC,
    othello_utils.games_batch_to_just_played_BLC,
    othello_utils.games_batch_to_board_state_classifier_input_BLC,
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

# game_idx = 0
n_moves = 30
n_layers_selected = 4
# move_idx = 8
# board_seqs_id = board_seqs_id[0, :20]
# board_seqs_id = board_seqs_id[game_idx, :n_moves]

board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)[:, :n_moves]

flipped_squares = einops.rearrange(
    test_data["games_batch_to_flipped_classifier_input_BLC"][:, :n_moves],
    "batch seq (row col) -> batch seq row col", row = 8, col = 8
).bool().cpu()

just_played_squares = einops.rearrange(
    test_data["games_batch_to_just_played_BLC"][:, :n_moves],
    "batch seq (row col) -> batch seq row col", row = 8, col = 8
).bool().cpu()

board_states = einops.rearrange(
    test_data["games_batch_to_board_state_classifier_input_BLC"][:, :n_moves],
    "batch seq (row col c) -> batch seq row col c",
    row = 8,
    col = 8,
).bool().cpu() # c=0 for mine, c=1 for empty, c=2 for yours

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

b_O = model.b_O.detach().clone()  # [layer, head, d_model]

# %% b_O casting and masking
total_moves = n_moves
b_O_cast = b_O[None, None].expand(total_moves, total_moves, 8, 512).clone() # [1, 1, layer, d_model]
b_O_mask = t.tril(t.ones(total_moves, total_moves, dtype=t.bool))
b_O_divisor = t.arange(1, total_moves + 1).view(total_moves, 1, 1, 1).to(device)  # [total_moves, 1, 1, 1]

b_O_cast[~b_O_mask] = 0.0
b_O_cast /= b_O_divisor  # [seq_q, seq_k, layer, d_model]

# %%
probes = load_fold_probes_and_normalize(n_layers, device)

# %%
with open("attention/attention_head_types.json", "r") as f:
    head_type_all = json.load(f)

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

v_matrices = t.stack([
    cache["v", layer] for layer in range(model.cfg.n_layers)
], dim = 2)
pattern_matrices = t.stack([
    cache["pattern", layer] for layer in range(model.cfg.n_layers)
], dim = 2)

del cache
t.cuda.empty_cache()

# %%
# probe_list = ["mine", "flipped", "just_played"]
# probe_list = ["mine"]
probe_projs = {}
# for probe_name in probe_list:
probe_name = "mine"
resid_pre_probe_dir = einops.einsum(
    resid_pre,
    probe_layer_specific[probe_name],
    "batch seq layer d_model, d_model ... -> batch seq layer ...",
).cpu()
del resid_pre
t.cuda.empty_cache()

attn_out_probe_dir = einops.einsum(
    attn_out,
    probe_layer_specific[probe_name],
    "batch seq layer d_model, d_model ... -> batch seq layer ...",
).cpu()
del attn_out
t.cuda.empty_cache()

resid_mid_probe_dir = einops.einsum(
    resid_mid,
    probe_layer_specific[probe_name],
    "batch seq layer d_model, d_model ... -> batch seq layer ...",
).cpu()
del resid_mid
t.cuda.empty_cache()

mlp_out_probe_dir = einops.einsum(
    mlp_out,
    probe_layer_specific[probe_name],
    "batch seq layer d_model, d_model ... -> batch seq layer ...",
).cpu()
del mlp_out
t.cuda.empty_cache()

resid_post_probe_dir = einops.einsum(
    resid_post,
    probe_layer_specific[probe_name],
    "batch seq layer d_model, d_model ... -> batch seq layer ...",
).cpu()
del resid_post
t.cuda.empty_cache()

# out of cuda memory
attn_out_qk = einops.einsum(
    v_matrices,
    pattern_matrices,
    W_O,
    "batch seq_k layer head d_head, batch head layer seq_q seq_k, layer head d_head d_model -> batch seq_q seq_k layer head d_model",
) # [batch, seq_q, seq_k, layer, head, d_model]
del v_matrices
del pattern_matrices
t.cuda.empty_cache()

attn_out_qk_last_probe_dir = einops.einsum(
    attn_out_qk + b_O_cast[None, :, :, :, None, :] / model.cfg.n_heads,
    probe_layer_specific[probe_name],
    "batch seq_q seq_k layer head d_model, d_model ... -> batch seq_q seq_k layer head ...",
).cpu()

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
    probe_projs[(probe_name, layer, "mine_heads")] = attn_out_qk_last_probe_dir[:, :, :, layer, mine_heads].sum(-3) # [batch, seq_q, seq_k, layer, row, col]
    probe_projs[(probe_name, layer, "yours_heads")] = attn_out_qk_last_probe_dir[:, :, :, layer, yours_heads].sum(-3) # [batch, seq_q, seq_k, layer, row, col]
    probe_projs[(probe_name, layer, "other_heads")] = attn_out_qk_last_probe_dir[:, :, :, layer, other_heads].sum(-3) # [batch, seq_q, seq_k, layer, row, col]
    probe_projs[(probe_name, layer, "resid_pre")] = resid_pre_probe_dir[:, :, layer] # [batch, seq, layer, row, col]
    probe_projs[(probe_name, layer, "attn_out")] = attn_out_probe_dir[:, :, layer] # [batch, seq, layer, row, col]
    probe_projs[(probe_name, layer, "resid_mid")] = resid_mid_probe_dir[:, :, layer] # [batch, seq, layer, row, col]
    probe_projs[(probe_name, layer, "mlp_out")] = mlp_out_probe_dir[:, :, layer] # [batch, seq, layer, row, col]
    probe_projs[(probe_name, layer, "resid_post")] = resid_post_probe_dir[:, :, layer] # [batch, seq, layer, row, col]

# # %% first do mean+-2std to cluster the board output (per game, play, layer, type, across board)
# positive_clusters = {}
# for probe_name in probe_projs.keys():
#     for layer in range(n_layers):
#         for block_type in ["mine_heads", "yours_heads", "other_heads", "attn_out", "mlp_out"]:
#             proj = probe_projs[probe_name][layer][block_type] # [batch, seqq, seqk, head, row, col]
#             mean = proj.mean(dim=(-1, -2))
#             std = proj.std(dim=(-1, -2))
#             threshold = mean + 2 * std
#             positive_cluster = proj > threshold[..., None, None]
#             positive_clusters[(probe_name, layer, block_type)] = positive_cluster

# %% first do mean+-2std to cluster the board output (per game, play, layer, across type and board)
positive_clusters = {}
negative_clusters = {}
# for probe_name in probe_list:
probe_name = "mine"
for layer in range(n_layers):
    # proj_all = []
    # for block_type in ["mine_heads", "yours_heads", "other_heads", "mlp_out"]:
    #     proj = probe_projs[(probe_name, layer, block_type)]
    #     proj_all.append(proj.flatten(start_dim=2)) # flatten batch, seqk, seqq, head together
    # proj_all = t.cat(proj_all, dim=-1) # [batch, seq_k, board]
    # mean = proj_all.mean(dim=-1)
    # std = proj_all.std(dim=-1)
    # threshold_pos = mean + 2 * std
    # threshold_neg = mean - 2 * std
    
    proj_move = defaultdict(list)
    for block_type in ["mine_heads", "yours_heads", "other_heads"]:
        for move in range(n_moves):
            proj = probe_projs[(probe_name, layer, block_type)][:, move, :move+1] # [batch, seq_q, seq_k, row, col]
            proj_move[move].append(proj.flatten(start_dim=1)) # [batch, seq_k*row*col]
    
    for block_type in ["mlp_out"]:
        for move in range(n_moves):
            proj = probe_projs[(probe_name, layer, block_type)][:, move] # [batch, seq, row, col]
            proj_move[move].append(proj.flatten(start_dim=1)) # [batch, row*col]
    
    mean = []
    std = []
    for move in range(n_moves):
        # proj_all.append(t.cat(proj_move[move], dim=-1)) # [batch, seq_k*row*col + row*col]
        mean.append(t.cat(proj_move[move], dim=-1).mean(dim=-1)) # [batch]
        std.append(t.cat(proj_move[move], dim=-1).std(dim=-1)) # [batch]
    mean = t.stack(mean, dim=1) # [batch, seq]
    std = t.stack(std, dim=1) # [batch, seq]
    threshold_pos = mean + 2 * std
    threshold_neg = mean - 2 * std

    for block_type in ["mine_heads", "yours_heads", "other_heads", "mlp_out"]:
        proj = probe_projs[(probe_name, layer, block_type)] # [batch, seq_q, seq_k, row, col]
        broad_cast_times = len(proj.shape) - len(threshold_pos.shape)
        # broadcast threshold to match proj shape
        positive_cluster = (proj > threshold_pos[(...,) + (None,) * broad_cast_times]) & (proj != 0)
        negative_cluster = (proj < threshold_neg[(...,) + (None,) * broad_cast_times]) & (proj != 0)
        positive_clusters[(probe_name, layer, block_type)] = positive_cluster
        negative_clusters[(probe_name, layer, block_type)] = negative_cluster

# %% first do otsu to cluster the board output
# pass

# %% flipped and just_played board states match with
scores = {}
# for probe_name in probe_list:
probe_name = "mine"
for layer in range(n_layers):
    for block_type in ["attn_out", "mlp_out"]:
        positive_cluster = positive_clusters[(probe_name, layer, block_type)]
        # negative_cluster = negative_clusters[(probe_name, layer, block_type)]
        flipped_match = (positive_cluster & flipped_squares).sum(dim=[-1,-2])
        just_played_match = (positive_cluster & just_played_squares).sum(dim=[-1,-2])
        others = positive_cluster.sum(dim=[-1,-2]) - flipped_match - just_played_match
        scores[(probe_name, layer, block_type)] = {
            "flipped_match": flipped_match,
            "just_played_match": just_played_match,
            "others": others,
        }
    
    for block_type in ["mine_heads", "yours_heads", "other_heads"]:
        # [batch, seq_q, seq_k, head, row, col]
        positive_cluster = positive_clusters[(probe_name, layer, block_type)]
        negative_cluster = negative_clusters[(probe_name, layer, block_type)]

        for seqq in range(n_moves):
            flipped_pos = (positive_cluster[:, seqq, :seqq+1] & flipped_squares[:, :seqq+1]).sum(dim=[-2,-1])
            flipped_neg = (negative_cluster[:, seqq, :seqq+1] & flipped_squares[:, :seqq+1]).sum(dim=[-2,-1])
            just_played_pos = (positive_cluster[:, seqq, :seqq+1] & just_played_squares[:, :seqq+1]).sum(dim=[-2,-1])
            just_played_neg = (negative_cluster[:, seqq, :seqq+1] & just_played_squares[:, :seqq+1]).sum(dim=[-2,-1])

            # board state match
            # board_pos = (negative_cluster[:, seqq, :seqq+1]) & board_states ...
            
            # others
            # others_pos = ...
            # others_neg = ...

            # scores[(probe_name, layer, block_type, seqq)] = {}
