"""
Intervention experiment using GPTWithProbeIntervention.

Replicates the experiment in attention/attention_intervention.py but uses the
mingpt-based GPTWithProbeIntervention model so that the probe projection is
baked into forward() rather than post-hoc patched.

Comparison structure (mirrors attention_intervention.py):
  - clean:      GPTWithProbeIntervention with no intervention layers
  - intervened: GPTWithProbeIntervention with chosen intervention layers
  - random:     same architecture but probe_dirs replaced with random directions
  - zero:       same architecture but probe_dirs replaced with zeros

Usage (from the finetuning/ directory):
    python test.py
"""

# %%
import os
import sys
import torch

# ── path setup ────────────────────────────────────────────────────────────────
# Make repo root importable for utils.*
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
# sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from utils.circuits_utils import construct_othello_dataset
from utils.othello_utils import games_batch_to_valid_moves_BLRRC
from utils.helper_fns import compute_top_n_accuracy, compute_kl_divergence
from utils.probe_utils import load_fold_probes_and_normalize

from finetuning.mingpt.model import GPTConfig
from finetuning.model2 import GPTWithProbeIntervention

# ── config ────────────────────────────────────────────────────────────────────
HF_MODEL_NAME       = "Baidicoot/Othello-GPT-Transformer-Lens"
TEST_SIZE           = 500
PROBE_KEYS          = ["flipped", "just_played", "mine"]
PROBE_LAYER         = 5
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

# Layer combinations to test — mirrors layers_chosen in attention_intervention.py
LAYERS_CHOSEN = [
    [],                        # no intervention (clean baseline)
    [0],
    # list(range(8)),            # all layers
    # list(range(1, 6)),         # layers 1–5
    # list(range(1, 8)),         # layers 1–7
]


# ── helpers ───────────────────────────────────────────────────────────────────

def build_probe_dirs(probe_keys, probe_layer, device):
    """Stack and QR-prepare probe directions (same logic as train.py)."""
    probes = load_fold_probes_and_normalize(n_layers=8, device=device)
    dirs = []
    for key in probe_keys:
        d = probes[key][probe_layer]                    # (d_model, 8, 8)
        d = d / d.norm(dim=0, keepdim=True)
        dirs.append(d)
    D = torch.stack(dirs, dim=1)                        # (d_model, n_probes, 8, 8)
    D = D.reshape(D.shape[0], -1)                       # (d_model, n_dirs)
    D = torch.nan_to_num(D)
    return D


def build_model(gpt_config, probe_dirs, intervention_layers, device):
    """Instantiate GPTWithProbeIntervention and load pretrained weights."""
    model = GPTWithProbeIntervention(
        gpt_config,
        probe_dirs.to(device),
        intervention_layers=intervention_layers,
    )
    model.load_pretrained_from_hf(HF_MODEL_NAME)
    model = model.to(device)
    model.eval()
    return model


def run_model(model, board_seqs_id):
    """Forward pass; returns logits shaped (B, L, 61)."""
    with torch.no_grad():
        logits, _ = model(board_seqs_id)
    return logits


# ── main ──────────────────────────────────────────────────────────────────────
# %%

print(f"Device: {DEVICE}")

# 1. Load test dataset
print(f"\nLoading {TEST_SIZE} test games ...")
test_data = construct_othello_dataset(
    custom_functions=[games_batch_to_valid_moves_BLRRC],
    n_inputs=TEST_SIZE,
    split="test",
    device=DEVICE,
)
board_seqs_id      = torch.tensor(test_data["encoded_inputs"]).long().to(DEVICE)
valid_moves_BLRRC  = test_data["games_batch_to_valid_moves_BLRRC"]

# 2. Build probe directions (and random / zero controls)
print("Loading probe directions ...")
probe_dirs = build_probe_dirs(PROBE_KEYS, PROBE_LAYER, DEVICE)
print(f"  Probe direction matrix: {tuple(probe_dirs.shape)}")

# torch.manual_seed(42)
# random_dirs = torch.randn_like(probe_dirs)
# random_dirs = random_dirs / random_dirs.norm(dim=0, keepdim=True)
# random_dirs = torch.nan_to_num(random_dirs)

zero_dirs = torch.zeros_like(probe_dirs)

# 3. GPT config (must match HF checkpoint)
# vocab_size=61: 60 valid board positions + 1 pass/padding token (index 0)
# CharDataset maps -100 → index 0, which occupies the same slot as HF's pass token.
gpt_config = GPTConfig(
    vocab_size  = 61,
    block_size  = 59,
    n_layer     = 8,
    n_head      = 8,
    n_embd      = 512,
)

# %%
import utils.circuits_utils as circuits_utils
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model_old = circuits_utils.get_model(model_name, DEVICE)
with torch.no_grad():
    logits_old = model_old(board_seqs_id)

# %%
# probes = load_fold_probes_and_normalize(8, DEVICE)

# probe_layer_specific = {
#     name: probes[name][5]
#     for name in probes.keys()
# }

# dirs = []
# for key in ["flipped", "just_played", "mine"]:
#     d = probe_layer_specific[key]                      # (d_model, 8, 8)
#     d = d / d.norm(dim=0, keepdim=True)                # normalize each vector
#     dirs.append(d)

# # Stack: (num_probs, d_model, 8, 8)
# D = torch.stack(dirs, dim=1)

# # Flatten everything except d_model
# D = D.reshape(D.shape[0], -1)                        # (d_model, 128)
# D = torch.nan_to_num(D)


# %%
# 4. Clean baseline (no intervention)
print("\nRunning clean baseline (no intervention) ...")
clean_model = build_model(gpt_config, probe_dirs, [], DEVICE)
logits_clean = run_model(clean_model, board_seqs_id)
_, _, clean_acc = compute_top_n_accuracy(logits_clean, valid_moves_BLRRC)
print(f"  Clean accuracy: {clean_acc * 100:.2f}%")
# del clean_model

# %%

# 5. Run experiments for each layer combination
results = []

for layers in LAYERS_CHOSEN:
    label = f"L{','.join(str(l) for l in layers)}" if layers else "none"
    print(f"\nInterventions: [{label}]")

    # probe dirs
    model   = build_model(gpt_config, probe_dirs, layers, DEVICE)
    logits  = run_model(model, board_seqs_id)
    _, _, acc = compute_top_n_accuracy(logits, valid_moves_BLRRC)
    kl        = compute_kl_divergence(logits_clean, logits).mean().item()
    # del model

    # # random dirs (control)
    # rmodel   = build_model(gpt_config, random_dirs, layers, DEVICE)
    # rlogits  = run_model(rmodel, board_seqs_id)
    # _, _, racc = compute_top_n_accuracy(rlogits, valid_moves_BLRRC)
    # rkl        = compute_kl_divergence(logits_clean, rlogits).mean().item()
    # del rmodel

    # # zero dirs (control)
    # zmodel   = build_model(gpt_config, zero_dirs,  layers, DEVICE)
    # zlogits  = run_model(zmodel, board_seqs_id)
    # _, _, zacc = compute_top_n_accuracy(zlogits, valid_moves_BLRRC)
    # zkl        = compute_kl_divergence(logits_clean, zlogits).mean().item()
    # del zmodel

    print(f"  Probe   — acc: {acc*100:.2f}%  KL: {kl:.4f}")

    results.append((label, acc, kl))

# 6. Summary table
print(f"\n{'─'*50}")
print(f"Intervention results  (clean acc: {clean_acc*100:.2f}%)")
print(f"{'─'*50}")
print(f"{'Layers':<16} {'Accu.':>8} {'KL':>8}")
print(f"{'─'*50}")
for label, acc, kl in results:
    print(f"{label:<16} {acc*100:>7.2f}% {kl:>8.4f}")
print(f"{'─'*50}")


# %%
