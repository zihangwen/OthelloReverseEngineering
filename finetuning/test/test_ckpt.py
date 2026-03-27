"""
Test probe_ft.ckpt (finetuned GPTWithProbeIntervention) against the clean
HF baseline on 500 test games.

Prints accuracy and KL divergence for:
  - clean:    HF pretrained, no intervention
  - finetuned: checkpoint, intervention_layers=[1..7]

Usage (from repo root):
    python finetuning/test/test_ckpt.py
"""

# %%
import os
import torch

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO_ROOT)

from utils.circuits_utils import construct_othello_dataset
from utils.othello_utils import games_batch_to_valid_moves_BLRRC
from utils.helper_fns import compute_top_n_accuracy, compute_kl_divergence

from finetuning.mingpt.model import GPTConfig
from finetuning.model_probe import GPTWithProbeIntervention
from finetuning.utils_probe import load_probe_dirs, load_probe_dirs_per_layer, to_device

# ── config ────────────────────────────────────────────────────────────────────
HF_MODEL_NAME       = "Baidicoot/Othello-GPT-Transformer-Lens"
CKPT_PATH           = "finetuning/checkpoints/probe_ft_1_7_3probe_Lp.ckpt"
TEST_SIZE           = 500
PROBE_KEYS          = ["mine", "flipped", "just_played"]
PROBE_LAYER         = 5
INTERVENTION_LAYERS = [1, 2, 3, 4, 5, 6, 7]
PER_LAYER_PROBE     = True   # True: layer i uses its own layer-i probe dirs
DEVICE              = "cuda:1" if torch.cuda.is_available() else "cpu"

# ── GPT config (must match training) ─────────────────────────────────────────
GPT_CONFIG = GPTConfig(
    vocab_size = 61,
    block_size = 59,
    n_layer    = 8,
    n_head     = 8,
    n_embd     = 512,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def build_clean_model(gpt_config, probe_dirs, intervention_layers, device):
    """HF pretrained, no intervention (clean baseline)."""
    model = GPTWithProbeIntervention(
        gpt_config,
        to_device(probe_dirs, device),
        intervention_layers=intervention_layers,
    )
    model.load_pretrained_from_hf(HF_MODEL_NAME)
    return model.to(device).eval()


def build_finetuned_model(gpt_config, probe_dirs, intervention_layers, ckpt_path, device):
    """Load finetuned checkpoint."""
    model = GPTWithProbeIntervention(
        gpt_config,
        to_device(probe_dirs, device),
        intervention_layers=intervention_layers,
    )
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        print(f"  Warning — missing keys: {missing}")
    if unexpected:
        print(f"  Warning — unexpected keys: {unexpected}")
    return model.to(device).eval()


def run_model(model, board_seqs_id):
    with torch.no_grad():
        logits, _ = model(board_seqs_id)
    return logits


# ── main ──────────────────────────────────────────────────────────────────────
# %%
print(f"Device: {DEVICE}")

# 1. Load test data
print(f"\nLoading {TEST_SIZE} test games ...")
test_data = construct_othello_dataset(
    custom_functions=[games_batch_to_valid_moves_BLRRC],
    n_inputs=TEST_SIZE,
    split="test",
    device=DEVICE,
)
board_seqs_id     = torch.tensor(test_data["encoded_inputs"]).long().to(DEVICE)
valid_moves_BLRRC = test_data["games_batch_to_valid_moves_BLRRC"]

# %%
# 2. Probe directions
print("Loading probe directions ...")
if PER_LAYER_PROBE:
    probe_dirs = load_probe_dirs_per_layer(PROBE_KEYS, INTERVENTION_LAYERS, DEVICE)
    print(f"  Per-layer probe matrices: { {k: tuple(v.shape) for k, v in probe_dirs.items()} }")
else:
    probe_dirs = load_probe_dirs(PROBE_KEYS, PROBE_LAYER, DEVICE)
    print(f"  Probe direction matrix: {tuple(probe_dirs.shape)}")

# 3. Clean baseline
print("\nRunning clean baseline (no intervention, HF weights) ...")
clean_model  = build_clean_model(GPT_CONFIG, probe_dirs, [], DEVICE)
logits_clean = run_model(clean_model, board_seqs_id)
_, _, clean_acc = compute_top_n_accuracy(logits_clean, valid_moves_BLRRC)
print(f"  Clean accuracy: {clean_acc * 100:.2f}%")

interv_model  = build_clean_model(GPT_CONFIG, probe_dirs, INTERVENTION_LAYERS, DEVICE)
logits_interv = run_model(interv_model, board_seqs_id)
_, _, interv_acc = compute_top_n_accuracy(logits_interv, valid_moves_BLRRC)
interv_kl        = compute_kl_divergence(logits_clean, logits_interv).mean().item()
print(f"  Ref accuracy: {interv_acc * 100:.2f}%")

# 4. Finetuned checkpoint
print(f"\nLoading finetuned checkpoint: {CKPT_PATH}")
print(f"  Intervention layers: {INTERVENTION_LAYERS}")
ft_model  = build_finetuned_model(GPT_CONFIG, probe_dirs, INTERVENTION_LAYERS, CKPT_PATH, DEVICE)
logits_ft = run_model(ft_model, board_seqs_id)
_, _, ft_acc = compute_top_n_accuracy(logits_ft, valid_moves_BLRRC)
ft_kl        = compute_kl_divergence(logits_clean, logits_ft).mean().item()
print(f"  Finetuned accuracy: {ft_acc * 100:.2f}%  KL from clean: {ft_kl:.4f}")

# 5. Summary
print(f"\n{'─'*55}")
print(f"Results  (test size: {TEST_SIZE})")
print(f"{'─'*55}")
print(f"{'Model':<24} {'Accuracy':>10} {'KL (vs clean)':>14}")
print(f"{'─'*55}")
print(f"{'clean (HF, no interv.)':<24} {clean_acc*100:>9.2f}% {'—':>14}")
print(f"{'ref (HF, interv.)':<24} {interv_acc*100:>9.2f}% {interv_kl:>14.4f}")
print(f"{'finetuned ckpt':<24} {ft_acc*100:>9.2f}% {ft_kl:>14.4f}")
print(f"{'─'*55}")

# %%
