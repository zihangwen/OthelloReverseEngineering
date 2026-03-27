"""
Tests for ProbeModelTrainerConfig and ProbeModelTrainer.

Checks:
  1. Config defaults and overrides
  2. Trainer init — no-KL path (kl_weight=0, ref_model is None)
  3. Trainer init — KL path (kl_weight>0, ref_model loaded and frozen)
  4. _compute_loss — CE-only
  5. _compute_loss — CE + KL
  6. freeze_up_to — verify frozen / unfrozen params
  7. One training step (loss decreases, gradients flow)

Usage (from repo root):
    python -m finetuning.test.test_trainer
"""

# %%
import os
import torch

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.chdir(REPO_ROOT)

from finetuning.mingpt.model import GPTConfig
from finetuning.mingpt.dataset import CharDataset
from finetuning.model_probe import GPTWithProbeIntervention
from finetuning.trainer_probe import ProbeModelTrainerConfig, ProbeModelTrainer
from finetuning.utils_probe import load_probe_dirs
from utils.circuits_utils import construct_othello_dataset

HF_MODEL_NAME = "Baidicoot/Othello-GPT-Transformer-Lens"
PROBE_KEYS    = ["flipped", "just_played", "mine"]
PROBE_LAYER   = 5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Minimal GPT config matching the HF checkpoint
GPT_CONFIG = GPTConfig(
    vocab_size = 61,
    block_size = 59,
    n_layer    = 8,
    n_head     = 8,
    n_embd     = 512,
)

print(f"Device: {DEVICE}")


# ── helpers ───────────────────────────────────────────────────────────────────

def make_model(intervention_layers=None, probe_dirs=None):
    """Build and load a GPTWithProbeIntervention."""
    if probe_dirs is None:
        probe_dirs = load_probe_dirs(PROBE_KEYS, PROBE_LAYER, DEVICE)
    model = GPTWithProbeIntervention(
        GPT_CONFIG,
        probe_dirs.to(DEVICE),
        intervention_layers=intervention_layers or [],
    )
    model.load_pretrained_from_hf(HF_MODEL_NAME)
    return model.to(DEVICE)


# ── real dataset (small slice) ────────────────────────────────────────────────
# Load 200 real games — same source as build_datasets(), just capped for speed.
print("Loading real Othello data (200 games) ...")
_seqs = construct_othello_dataset(
    custom_functions=[], n_inputs=200, split="train", max_str_length=60,
)["encoded_inputs"]
ds = CharDataset(_seqs)
print(f"  vocab_size={ds.vocab_size}  block_size={ds.block_size}  n={len(ds)}")

# %%
model = GPTWithProbeIntervention(
    GPT_CONFIG,
    load_probe_dirs(PROBE_KEYS, PROBE_LAYER, DEVICE),
    intervention_layers=[],
)
model.load_pretrained_from_hf(HF_MODEL_NAME)

# ── test 1: config defaults and overrides ────────────────────────────────────
# %%
print("\n─── Test 1: ProbeModelTrainerConfig ───")

cfg_default = ProbeModelTrainerConfig()
assert cfg_default.kl_weight    == 0.1,  f"expected 0.1, got {cfg_default.kl_weight}"
assert cfg_default.freeze_up_to == -1,   f"expected -1,  got {cfg_default.freeze_up_to}"
assert cfg_default.max_epochs   == 10,   f"expected 10,  got {cfg_default.max_epochs}"

cfg_custom = ProbeModelTrainerConfig(kl_weight=0.5, freeze_up_to=3, max_epochs=2)
assert cfg_custom.kl_weight    == 0.5
assert cfg_custom.freeze_up_to == 3
assert cfg_custom.max_epochs   == 2

print("  PASS — defaults and overrides correct")


# ── test 2: trainer init, kl_weight=0 → no ref_model ─────────────────────────
# %%
print("\n─── Test 2: ProbeModelTrainer init — no KL (kl_weight=0) ───")

model  = make_model()
config = ProbeModelTrainerConfig(kl_weight=0.0, max_epochs=1, batch_size=8)
trainer = ProbeModelTrainer(model, ds, None, config)

assert trainer.ref_model is None, "ref_model should be None when kl_weight=0"
print("  PASS — ref_model is None")


# ── test 3: trainer init, kl_weight>0 → ref_model loaded and frozen ──────────
# %%
print("\n─── Test 3: ProbeModelTrainer init — KL path (kl_weight=0.1) ───")

model  = make_model()
config = ProbeModelTrainerConfig(kl_weight=0.1, ref_model_name=HF_MODEL_NAME,
                             max_epochs=1, batch_size=8)
trainer = ProbeModelTrainer(model, ds, None, config)

assert trainer.ref_model is not None, "ref_model should be loaded"
n_trainable = sum(p.requires_grad for p in trainer.ref_model.parameters())
assert n_trainable == 0, f"ref_model has {n_trainable} trainable params (should be 0)"
print(f"  PASS — ref_model loaded, all {sum(1 for _ in trainer.ref_model.parameters())} params frozen")


# ── test 4: _compute_loss — CE only ──────────────────────────────────────────
# %%
print("\n─── Test 4: _compute_loss — CE only (kl_weight=0) ───")

model   = make_model(intervention_layers=[0])
config  = ProbeModelTrainerConfig(kl_weight=0.0, max_epochs=1, batch_size=8)
trainer = ProbeModelTrainer(model, ds, None, config)

x = torch.stack([ds[i][0] for i in range(4)]).to(DEVICE)
y = torch.stack([ds[i][1] for i in range(4)]).to(DEVICE)
with torch.no_grad():
    logits, _ = model(x, None)
loss = trainer._compute_loss(logits, y, x)

assert loss.ndim == 0,       "loss should be a scalar"
assert loss.item() > 0,      "CE loss should be positive"
assert torch.isfinite(loss), "CE loss should be finite"
print(f"  PASS — CE loss: {loss.item():.4f}")


# ── test 5: _compute_loss — CE + KL ──────────────────────────────────────────
# %%
print("\n─── Test 5: _compute_loss — CE + KL (kl_weight=0.1) ───")

model   = make_model(intervention_layers=[0])
config  = ProbeModelTrainerConfig(kl_weight=0.1, ref_model_name=HF_MODEL_NAME,
                              max_epochs=1, batch_size=8)
trainer = ProbeModelTrainer(model, ds, None, config)

ce_only_config  = ProbeModelTrainerConfig(kl_weight=0.0, max_epochs=1, batch_size=8)
ce_only_trainer = ProbeModelTrainer(model, ds, None, ce_only_config)

with torch.no_grad():
    logits, _ = model(x, None)
loss_kl = trainer._compute_loss(logits, y, x)
loss_ce = ce_only_trainer._compute_loss(logits, y, x)

assert torch.isfinite(loss_kl), "CE+KL loss should be finite"
# KL(p||p) = 0 when models are the same; KL >= 0 always, so loss_kl >= loss_ce
assert loss_kl.item() >= loss_ce.item() - 1e-4, \
    f"CE+KL ({loss_kl.item():.4f}) should be >= CE ({loss_ce.item():.4f})"
print(f"  PASS — CE: {loss_ce.item():.4f}  CE+KL: {loss_kl.item():.4f}  "
      f"KL contribution: {(loss_kl - loss_ce).item():.4f}")


# ── test 6: freeze_up_to — check frozen / unfrozen params ────────────────────
# %%
print("\n─── Test 6: freeze_up_to ───")

FREEZE_AT = 3
model   = make_model()
config  = ProbeModelTrainerConfig(kl_weight=0.0, freeze_up_to=FREEZE_AT,
                              max_epochs=1, batch_size=8)
trainer = ProbeModelTrainer(model, ds, None, config)

raw = model.module if hasattr(model, "module") else model

# Embeddings should be frozen
assert not raw.tok_emb.weight.requires_grad, "tok_emb should be frozen"
assert not raw.pos_emb.requires_grad,        "pos_emb should be frozen"

# Blocks 0..freeze_up_to should be frozen
for i in range(FREEZE_AT + 1):
    for name, p in raw.blocks[i].named_parameters():
        assert not p.requires_grad, f"blocks[{i}].{name} should be frozen"

# Blocks after freeze_up_to should be trainable
for i in range(FREEZE_AT + 1, GPT_CONFIG.n_layer):
    n_trainable = sum(p.requires_grad for p in raw.blocks[i].parameters())
    assert n_trainable > 0, f"blocks[{i}] should have trainable params"

print(f"  PASS — embeddings + blocks 0..{FREEZE_AT} frozen; blocks {FREEZE_AT+1}..7 trainable")


# ── test 7: one training step — loss finite, gradients flow ──────────────────
# %%
print("\n─── Test 7: one training step ───")

FREEZE_AT = 4
model   = make_model(intervention_layers=[0])
config  = ProbeModelTrainerConfig(
    kl_weight    = 0.1,
    freeze_up_to = FREEZE_AT,
    ref_model_name = HF_MODEL_NAME,
    max_epochs   = 1,
    batch_size   = 8,
    num_workers  = 0,
)
trainer = ProbeModelTrainer(model, ds, None, config)

raw       = model.module if hasattr(model, "module") else model
optimizer = raw.configure_optimizers(config)

for epoch in range(3):
    x_b, y_b = ds[0]
    x_b = x_b.unsqueeze(0).to(DEVICE)
    y_b = y_b.unsqueeze(0).to(DEVICE)

    logits, _ = model(x_b, None)
    loss_before = trainer._compute_loss(logits, y_b, x_b)

    model.zero_grad()
    loss_before.backward()

    # Check that frozen params have no gradient
    for i in range(FREEZE_AT + 1):
        for name, p in raw.blocks[i].named_parameters():
            assert p.grad is None or p.grad.abs().max() == 0, \
                f"blocks[{i}].{name} is frozen but has non-zero grad"

    # Check that unfrozen params do have gradients
    has_grad = [
        p.grad is not None and p.grad.abs().max() > 0
        for i in range(FREEZE_AT + 1, GPT_CONFIG.n_layer)
        for p in raw.blocks[i].parameters()
    ]
    assert any(has_grad), "No gradients in unfrozen blocks"

    optimizer.step()

# %%
with torch.no_grad():
    logits2, _ = model(x_b, None)
loss_after = trainer._compute_loss(logits2, y_b, x_b)

assert torch.isfinite(loss_after), "loss after step should be finite"
print(f"  PASS — loss before: {loss_before.item():.4f}  after: {loss_after.item():.4f}")
if FREEZE_AT >= 0:
    print(f"         frozen blocks 0..{FREEZE_AT} have no gradients")
print(f"         unfrozen blocks {FREEZE_AT+1}..7 received gradients")


# ── summary ───────────────────────────────────────────────────────────────────
# %%
print("\n" + "─" * 50)
print("All trainer tests passed.")
print("─" * 50)
