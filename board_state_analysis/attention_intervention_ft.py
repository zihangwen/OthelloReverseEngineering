# %%
"""
Probe-direction intervention sweep using GPTWithProbeIntervention.

Uses HF pretrained weights + the finetuning model's built-in probe intervention
(not nnsight hooks) to test how much probe directions carry task-relevant
information across different layer configurations.

Compares:
  - clean:     no intervention
  - probe:     intervention with actual probe directions (HF weights)
  - random:    intervention with random orthonormal directions (control)
  - finetuned: finetuned checkpoint, only for the layers it was trained on
"""
from pathlib import Path
import os
import torch as t

BASE_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_PATH)

from rich.table import Table
from rich.console import Console

from board_state_analysis.board_state_utils import load_test_dataset
import utils.othello_utils as othello_utils
from utils.helper_fns import compute_top_n_accuracy, compute_kl_divergence

from finetuning.mingpt.model import GPTConfig
from finetuning.model_probe import GPTWithProbeIntervention
from finetuning.utils_probe import load_probe_dirs_per_layer, to_device

device = "cuda:1" if t.cuda.is_available() else "cpu"
print(f"Using device: {device}")

FIG_DIR = BASE_PATH / "board_state_analysis" / "fig" / "attention_intervention_ft"
os.makedirs(FIG_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
HF_MODEL_NAME  = "Baidicoot/Othello-GPT-Transformer-Lens"
CKPT_PATH      = "finetuning/checkpoints/probe_ft_1_7_3probe_Lp.ckpt"
CKPT_LAYERS    = (1, 2, 3, 4, 5, 6, 7)   # layers the checkpoint was trained on
PROBE_KEYS     = ["flipped", "just_played", "mine"]

GPT_CONFIG = GPTConfig(
    vocab_size = 61,
    block_size = 59,
    n_layer    = 8,
    n_head     = 8,
    n_embd     = 512,
)

LAYERS_SWEEP = [
    (),
    (0,),
    (0, 1, 2, 3, 4, 5, 6, 7),
    (1, 2, 3, 4, 5),
    (1, 2, 3, 4, 5, 6, 7),
]

# %%
test_data, board_seqs_id, _ = load_test_dataset(
    [othello_utils.games_batch_to_valid_moves_BLRRC],
    n_games=500,
    device=device,
)
valid_moves_BLRRC = test_data["games_batch_to_valid_moves_BLRRC"]

# %% Build probe and random direction dicts for all relevant layers
all_layers = sorted({l for config in LAYERS_SWEEP for l in config})
probe_dirs_all = load_probe_dirs_per_layer(PROBE_KEYS, all_layers, device)

t.manual_seed(42)
random_D = t.randn_like(probe_dirs_all[0])
random_D = random_D / random_D.norm(dim=0, keepdim=True)
random_dirs_all = {
    layer: random_D
    for layer in probe_dirs_all.keys()
}
zero_D = t.zeros_like(probe_dirs_all[0])
zero_dirs_all = {
    layer: zero_D
    for layer in probe_dirs_all.keys()
}

def build_model(intervention_layers, probe_dirs):
    """HF pretrained weights with the given intervention layers and directions."""
    model = GPTWithProbeIntervention(
        GPT_CONFIG,
        to_device(probe_dirs, device),
        intervention_layers=list(intervention_layers),
    )
    model.load_pretrained_from_hf(HF_MODEL_NAME)
    return model.to(device).eval()


def build_finetuned_model(probe_dirs):
    """Load finetuned checkpoint (only valid for CKPT_LAYERS)."""
    model = GPTWithProbeIntervention(
        GPT_CONFIG,
        to_device(probe_dirs, device),
        intervention_layers=list(CKPT_LAYERS),
    )
    sd = t.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def run(model):
    with t.no_grad():
        logits, _ = model(board_seqs_id)
    return logits


# %% Clean baseline (no intervention)
logits_clean = run(build_model((), {}))
clean_acc = compute_top_n_accuracy(logits_clean, valid_moves_BLRRC)[-1]
print(f"Clean accuracy: {clean_acc * 100:.2f}%")

# %% Finetuned model (run once, only applies to CKPT_LAYERS)
print(f"Loading finetuned checkpoint: {CKPT_PATH}")
logits_ft = run(build_finetuned_model(probe_dirs_all))
ft_acc = compute_top_n_accuracy(logits_ft, valid_moves_BLRRC)[-1]
ft_kl  = compute_kl_divergence(logits_clean, logits_ft).mean().item()

# %% Sweep
results = {}
for layers in LAYERS_SWEEP:
    if not layers:
        results[layers] = {
            "probe_acc": clean_acc, "probe_kl": 0.0,
            "rand_acc":  clean_acc, "rand_kl":  0.0,
            "zero_acc":  clean_acc, "zero_kl":  0.0,
            "ft_acc": None, "ft_kl": None,
        }
        continue

    logits_probe = run(build_model(layers, probe_dirs_all))
    logits_rand  = run(build_model(layers, random_dirs_all))
    logits_zero = run(build_model(layers, zero_dirs_all))

    results[layers] = {
        "probe_acc": compute_top_n_accuracy(logits_probe, valid_moves_BLRRC)[-1],
        "probe_kl":  compute_kl_divergence(logits_clean, logits_probe).mean().item(),
        "rand_acc":  compute_top_n_accuracy(logits_rand,  valid_moves_BLRRC)[-1],
        "rand_kl":   compute_kl_divergence(logits_clean, logits_rand).mean().item(),
        "zero_acc": compute_top_n_accuracy(logits_zero,  valid_moves_BLRRC)[-1],
        "zero_kl":  compute_kl_divergence(logits_clean, logits_zero).mean().item(),
        "ft_acc":    ft_acc if tuple(layers) == CKPT_LAYERS else None,
        "ft_kl":     ft_kl  if tuple(layers) == CKPT_LAYERS else None,
    }

# %% Results table
table = Table(
    title=f"Probe intervention (GPTWithProbeIntervention, HF weights)  "
          f"— clean accuracy: {clean_acc * 100:.2f}%",
    show_lines=True,
)
table.add_column("Layers",           style="bold cyan",   no_wrap=True)
table.add_column("Probe Accu.",      style="light_green", justify="right")
table.add_column("Probe KL",         style="green",       justify="right")
table.add_column("Random Accu.",     style="red",         justify="right")
table.add_column("Random KL",        style="red",         justify="right")
table.add_column("Zero Accu.",       style="yellow",      justify="right")
table.add_column("Zero KL",          style="yellow",      justify="right")
table.add_column("FT Accu.",         style="magenta",     justify="right")
table.add_column("FT KL",            style="magenta",     justify="right")

for layers, res in results.items():
    label = "clean (no interv.)" if not layers else " ".join(f"L{l}" for l in layers)
    ft_acc_str = f"{res['ft_acc'] * 100:.2f}%" if res["ft_acc"] is not None else "—"
    ft_kl_str  = f"{res['ft_kl']:.4f}"         if res["ft_kl"]  is not None else "—"
    table.add_row(
        label,
        f"{res['probe_acc'] * 100:.2f}%", f"{res['probe_kl']:.4f}",
        f"{res['rand_acc']  * 100:.2f}%", f"{res['rand_kl']:.4f}",
        f"{res['zero_acc'] * 100:.2f}%", f"{res['zero_kl']:.4f}",
        ft_acc_str, ft_kl_str,
    )

console = Console(record=True)
console.print(table)

probe_set_name = "_".join(PROBE_KEYS)
stem = f"probe_intervention_ft_{probe_set_name}"
(FIG_DIR / f"{stem}.txt").write_text(console.export_text())
print(f"Saved → {FIG_DIR / stem}.txt")

# %%
