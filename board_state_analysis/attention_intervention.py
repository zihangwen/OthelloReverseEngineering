# %%
"""
Causal intervention on attention layer V projections.

Tests whether probe directions (flipped, just_played, mine) carry task-relevant
information by ablating them from the layer-norm output before V projection.
Compares probe directions vs random vs zero controls across multiple layer configs.

Sweeps over:
  - probe sets : ["flipped", "just_played", "mine"] and ["flipped", "just_played"]
  - tags       : "keep_proj" and "remove_proj"
"""
from pathlib import Path
import os
import torch as t
import einops
from rich.table import Table
from rich.console import Console

BASE_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_PATH)

import utils.othello_utils as othello_utils
from board_state_analysis.board_state_utils import (
    setup_model_and_probes,
    load_test_dataset,
)
from utils.helper_fns import compute_top_n_accuracy, compute_kl_divergence

device = "cuda:1" if t.cuda.is_available() else "cpu"
print(f"Using device: {device}")

FIG_DIR = BASE_PATH / "board_state_analysis" / "fig" / "attention_intervention"
os.makedirs(FIG_DIR, exist_ok=True)

# %%
model, n_layers, probes, probe_layer_specific = setup_model_and_probes(device=device)
W_V = model.W_V.detach().clone()

test_data, board_seqs_id, _ = load_test_dataset(
    [othello_utils.games_batch_to_valid_moves_BLRRC],
    n_games=500,
    device=device,
)
valid_moves_BLRRC = test_data["games_batch_to_valid_moves_BLRRC"]


# %%
def intervention_Direction(D_space, layers, tag):
    """
    Modify V values by projecting layer-norm output onto D_space, then either
    removing ('remove_proj') or keeping ('keep_proj') that projection.

    D_space : Tensor [d_model, n_dirs] or dict {layer: Tensor}
    layers  : tuple of layer indices to intervene on
    tag     : "remove_proj" | "keep_proj"
    """
    with t.no_grad(), model.trace(board_seqs_id):
        for layer in layers:
            D = D_space[layer] if isinstance(D_space, dict) else D_space
            Q, R = t.linalg.qr(D)
            hook_norm = model.blocks[layer].ln1.output
            x = hook_norm

            valid = R.diag().abs() > 1e-6
            Q_valid = Q[:, valid]
            x_proj = x @ Q_valid @ Q_valid.T if Q_valid.shape[1] > 0 else t.zeros_like(x)

            residual = x - x_proj if tag == "remove_proj" else x_proj
            new_v = einops.einsum(
                residual, W_V[layer],
                "batch seq d_model, head d_model d_head -> batch seq head d_head",
            ) + model.b_V[layer]
            new_v = t.nan_to_num(new_v)
            model.blocks[layer].attn.hook_v.output[:] = new_v

        logits_patch_BLV = model.unembed.output.save()
    return logits_patch_BLV


def build_probe_D(probe_name_list):
    """Stack and normalise probe directions → [d_model, n_dirs]."""
    dirs = []
    for key in probe_name_list:
        d = probe_layer_specific[key]
        d = d / d.norm(dim=0, keepdim=True)
        dirs.append(d)
    
    return t.nan_to_num(t.stack(dirs, dim=1).reshape(dirs[0].shape[0], -1))

def build_probe_D_dict(probe_name_list):
    """Build dict of probe direction stacks for each layer → {layer: Tensor}."""
    D_dict = {}
    for layer in range(n_layers):
        dirs = []
        for key in probe_name_list:
            d = probes[key][layer]
            d = d / d.norm(dim=0, keepdim=True)
            dirs.append(d)
        D_dict[layer] = t.nan_to_num(t.stack(dirs, dim=1).reshape(dirs[0].shape[0], -1))
    
    return D_dict

# %% Clean baseline (shared across all sweeps)
with t.no_grad(), model.trace(board_seqs_id):
    logits_clean = model.unembed.output.save()

clean_accuracy = compute_top_n_accuracy(logits_clean, valid_moves_BLRRC)
print(f"Clean accuracy: {clean_accuracy[-1]*100:.2f}%")

layers_chosen = [
    (0,),
    (0, 1, 2, 3, 4, 5, 6, 7),
    (1, 2, 3, 4, 5),
    (1, 2, 3, 4, 5, 6, 7),
]

probe_sets = {
    "flipped_played_mine": ["flipped", "just_played", "mine"],
    "flipped_played":      ["flipped", "just_played"],
}

# %% Sweep over probe sets and tags
for probe_set_name, probe_name_list in probe_sets.items():
    D = build_probe_D(probe_name_list)
    D_dict = build_probe_D_dict(probe_name_list)

    t.manual_seed(42)
    random_D = t.randn_like(D)
    random_D = random_D / random_D.norm(dim=0, keepdim=True)
    zero_D = t.zeros_like(D)

    for tag in ["keep_proj", "remove_proj"]:
        results = {}
        for layers in layers_chosen:
            logits_patch = intervention_Direction(D_dict,   layers, tag)
            logits_rand  = intervention_Direction(random_D, layers, tag)
            logits_zero  = intervention_Direction(zero_D,   layers, tag)
            results[layers] = {
                "patch_acc": compute_top_n_accuracy(logits_patch, valid_moves_BLRRC),
                "patch_kl":  compute_kl_divergence(logits_clean, logits_patch),
                "rand_acc":  compute_top_n_accuracy(logits_rand,  valid_moves_BLRRC),
                "rand_kl":   compute_kl_divergence(logits_clean, logits_rand),
                "zero_acc":  compute_top_n_accuracy(logits_zero,  valid_moves_BLRRC),
                "zero_kl":   compute_kl_divergence(logits_clean, logits_zero),
            }

        table = Table(
            title=f"V-ablation: {tag}  probes: {probe_name_list}"
                  f"  (clean: {clean_accuracy[-1]*100:.2f}%)",
            show_lines=True,
        )
        table.add_column("Layers",         style="bold cyan",   no_wrap=True)
        table.add_column("Accu.",          style="light_green", justify="right")
        table.add_column("KL",             style="green",       justify="right")
        table.add_column("Accu. (Random)", style="red",         justify="right")
        table.add_column("KL (Random)",    style="red",         justify="right")
        table.add_column("Accu. (Zero)",   style="yellow",      justify="right")
        table.add_column("KL (Zero)",      style="yellow",      justify="right")

        for layers, res in results.items():
            table.add_row(
                " ".join(f"L{l}" for l in layers),
                f"{res['patch_acc'][-1]*100:.2f}%", f"{res['patch_kl'].mean():.4f}",
                f"{res['rand_acc'][-1]*100:.2f}%",  f"{res['rand_kl'].mean():.4f}",
                f"{res['zero_acc'][-1]*100:.2f}%",  f"{res['zero_kl'].mean():.4f}",
            )

        console = Console(record=True)
        console.print(table)

        stem = f"V_ablation_{probe_set_name}_{tag}"
        (FIG_DIR / f"{stem}.txt").write_text(console.export_text())
        print(f"Saved → {FIG_DIR / stem}.txt")

# %%