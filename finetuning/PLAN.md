# Finetuning Plan: OthelloGPT with Probe-Direction V-Vector Intervention

## Goal

Finetune the original OthelloGPT model so that the value vectors (`v`) in each
attention layer are computed only from the subspace spanned by the **MINE**,
**FLIPPED**, and **PLACED** (just_played) probe directions.  Rather than
post-hoc patching (as in `attention_intervention.py`), we bake the projection
directly into the model's `forward()` pass and then finetune the remaining
weights so the model learns to function well under this constraint.

---

## Source files (imported, not copied)

| Source | What we use |
|---|---|
| `finetuning/mingpt/model.py` | `GPTConfig`, `GPT`, `Block` — imported as `finetuning.mingpt.model` |
| `finetuning/mingpt/trainer.py` | `TrainerConfig`, `Trainer` — imported as `finetuning.mingpt.trainer` |
| `finetuning/mingpt/dataset.py` | `CharDataset` — imported as `finetuning.mingpt.dataset` |
| `utils/circuits_utils.py` | `construct_othello_dataset`, `to_device` |
| `utils/probe_utils.py` | `load_fold_probes_and_normalize` |

---

## File Structure

```
finetuning/
├── PLAN.md              ← this document
├── __init__.py          ← makes finetuning a proper package
├── mingpt/              ← original mingpt source (unmodified)
│   ├── model.py
│   ├── trainer.py
│   └── dataset.py
├── data/                ← Othello game data utilities
│   └── othello.py
├── model_probe.py       ← probe-constrained model (ProbeBlock, GPTWithProbeIntervention)
├── trainer_probe.py     ← CE + KL training loop (ProbeModelTrainerConfig, ProbeModelTrainer)
├── utils_probe.py       ← probe/data loading utilities
├── run.py               ← CLI entry point + run_finetuning orchestration
├── checkpoints/         ← saved .ckpt and .log files
└── test/
    └── test_ckpt.py     ← evaluate a saved checkpoint against the HF baseline
```

---

## File 1: `model_probe.py`

### Purpose
Extends `mingpt/model.py` with classes that bake the probe-subspace projection
into the value computation. Also includes the HF weight loader.

### Intervention logic (from `attention_intervention.py`)

```
Q, R     = qr(D)                        # D: (d_model, n_dirs)
Q_valid  = Q[:, |diag(R)| > 1e-6]
x_raw    = (x - mean) / std             # raw-normalised (no LN weight/bias)
x_proj   = x_raw @ Q_valid @ Q_valid.T  # project onto probe subspace
x_v      = x_proj * ln_weight + ln_bias # re-apply LN scale (fold_ln match)
v        = W_V(x_v) + b_V               # value from probe-projected input
# K, Q computed from full LN output unchanged
```

### Classes

#### `ProbeProjectedAttention(nn.Module)`
- K, Q from full LN-normalised `x_ln`
- V from `x_v` passed in pre-computed by `ProbeBlock`
- No `Q_valid` here — projection lives in `ProbeBlock`

#### `ProbeBlock(nn.Module)`
- Holds `Q_valid` as a registered buffer (fixed, not trained)
- Computes `x_raw = F.layer_norm(x, [C])` (no weight/bias)
- Computes `x_v = (x_raw @ Q_valid @ Q_valid.T) * ln1.weight + ln1.bias`
- Passes `(x_ln, x_v)` to `ProbeProjectedAttention`

#### `GPTWithProbeIntervention(GPT)`
- `intervention_layers`: list of layer indices to replace with `ProbeBlock`
- `probe_dirs`: `(d_model, n_dirs)` tensor **or** `dict[int → Tensor]` for per-layer dirs
- `load_pretrained_from_hf(hf_model_name)`: loads `final.pth` directly with `strict=False`
  (Q_valid buffers absent from HF checkpoint are kept from `__init__`)

---

## File 2: `trainer_probe.py`

### Purpose
Training loop for `GPTWithProbeIntervention` with a combined
**cross-entropy + KL divergence** loss.

### Design

**Loss:**
```
L = CE(logits, targets) + β * KL(p_finetuned || p_original)
```
- CE is the standard next-move prediction loss (`ignore_index=0` for padding token)
- KL anchors the output distribution to the original unintervened OthelloGPT
- `kl_weight=0` recovers pure CE

**Reference model:**
- Built as `GPTWithProbeIntervention(..., intervention_layers=[])` — no intervention
- Loaded from HF at trainer init; all parameters frozen
- Used only inside `torch.no_grad()` during the KL computation

**Freezing strategy:**
- `freeze_up_to: int` — freeze embeddings + blocks `0..freeze_up_to` (-1 = freeze nothing)
- Only blocks with index > `freeze_up_to` and the unembedding are trained

### Classes

#### `ProbeModelTrainerConfig(TrainerConfig)`
Extends `TrainerConfig` with:
- `kl_weight: float = 0.1` — β for the KL term
- `freeze_up_to: int = -1` — freeze blocks 0..N (-1 = freeze nothing)
- `ref_model_name: str` — HF repo name for the reference model

#### `ProbeModelTrainer(Trainer)`
Extends `Trainer` with:
- Loads reference model at init for KL computation
- Freezes parameters according to `freeze_up_to`
- Overrides `_compute_loss` to add `β * KL(p_finetuned || p_ref)`
- Saves checkpoint only when test loss improves (if test set provided)

---

## File 3: `utils_probe.py`

### Purpose
Probe direction and dataset loading utilities. Imports `to_device` from
`utils.circuits_utils` and re-exports it so callers only need one import.

### Functions

#### `to_device(data, device)` *(re-exported from `utils.circuits_utils`)*
Recursively moves tensors in a nested structure (dict, list, Tensor) to device.

#### `load_probe_dirs(probe_keys, probe_layer, device) -> Tensor`
- Returns a single `(d_model, n_dirs)` tensor using the same `probe_layer` for all
  intervention layers
- Column-normalised, NaN-cleaned

#### `load_probe_dirs_per_layer(probe_keys, intervention_layers, device) -> dict[int, Tensor]`
- Returns `{layer_i: (d_model, n_dirs)}` — each layer uses its own layer-i probe vectors
- Calls the shared `_stack_probe_dirs` helper internally (same as `load_probe_dirs`)

#### `build_datasets(n_train, n_test) -> (CharDataset, CharDataset | None)`
- Loads from the HF Othello dataset via `construct_othello_dataset`
- `n_test=0` returns `None` for the test dataset

---

## File 4: `run.py`

### Purpose
CLI entry point and top-level orchestration. Logging goes to both stdout and
a `.log` file co-located with `--ckpt_path` (e.g. `checkpoints/probe_ft.log`).

### Flags

| Flag | Default | Description |
|---|---|---|
| `--hf_model_name` | `"Baidicoot/Othello-GPT-Transformer-Lens"` | HF repo to load weights from |
| `--n_train` | `20_000_000` | Max training sequences (capped at ~792k available) |
| `--n_test` | `0` | Test sequences; 0 = no test set |
| `--probe_keys` | `mine flipped just_played` | Probe names for the V subspace |
| `--probe_layer` | `5` | Single layer's probes to use; ignored when `--per_layer_probe` is set |
| `--per_layer_probe` | flag | Use layer-i probe dirs for layer-i intervention |
| `--intervention_layers` | `None` | Layer indices for ProbeBlock (0-indexed); empty = no intervention |
| `--freeze_up_to` | `-1` | Freeze blocks 0..N during finetuning (-1 = freeze nothing) |
| `--kl_weight` | `0.1` | β for the KL divergence term |
| `--max_epochs` | `10` | |
| `--batch_size` | `64` | |
| `--learning_rate` | `1e-4` | Lower than pretraining 3e-4 |
| `--weight_decay` | `0.1` | |
| `--lr_decay` | flag | Enable cosine LR decay with linear warmup |
| `--num_workers` | `0` | DataLoader workers |
| `--ckpt_path` | `None` | Path to save best checkpoint; `.log` written alongside it |
| `--device` | `cuda` | `cuda`, `cuda:0`, `cpu`, etc. |

---

## File 5: `test/test_ckpt.py`

### Purpose
Evaluate a saved checkpoint against two baselines on `TEST_SIZE=500` test games.

### Models compared

| Label | Weights | Intervention |
|---|---|---|
| `clean` | HF pretrained | none |
| `ref (HF, interv.)` | HF pretrained | `INTERVENTION_LAYERS` with probe dirs |
| `finetuned ckpt` | `CKPT_PATH` | `INTERVENTION_LAYERS` with probe dirs |

### Key config constants
- `CKPT_PATH` — path to the `.ckpt` file to evaluate
- `INTERVENTION_LAYERS` — must match what was used during training
- `PER_LAYER_PROBE` — set `True` to use per-layer probe dirs (calls `load_probe_dirs_per_layer`);
  `False` uses a single `PROBE_LAYER`

### Metrics reported
- **Accuracy**: top-n valid-move accuracy via `compute_top_n_accuracy`
- **KL**: mean KL divergence from the clean baseline logits

---

## Key Design Decisions

1. **Import from `mingpt/`, do not copy.** `finetuning/` is a proper package
   (has `__init__.py`); all mingpt classes are imported directly.

2. **`probe_dirs` accepts tensor or dict.** A flat tensor uses one fixed probe layer
   for all intervened layers; a `dict[int → Tensor]` uses each layer's own probe
   directions. The model handles both in `__init__` — callers choose via
   `--per_layer_probe` in `run.py` or `PER_LAYER_PROBE` in `test_ckpt.py`.

3. **CE + KL loss.** Cross-entropy trains next-move prediction under the
   constraint; KL prevents unbounded drift from the original model's output
   distribution. `kl_weight=0` recovers pure CE.

4. **Freezing via `freeze_up_to`.** Cleaner than a soft KL penalty for
   controlling early-layer drift.

5. **Only V is constrained.** K and Q use the full LN-normalised input,
   matching the intervention in `attention_intervention.py`.

6. **`Q_valid` as `register_buffer`.** Not a trainable parameter — excluded
   from `configure_optimizers` automatically. Saved in the checkpoint and
   restored by `load_state_dict(strict=True)` in `test_ckpt.py`.

7. **`strict=False` in `load_pretrained_from_hf`.** `Q_valid` buffers are not in
   the HF checkpoint; they are pre-computed from the probe dirs in `__init__`.
   In `test_ckpt.py`, loading from our own checkpoint uses `strict=True` since
   all keys including `Q_valid` are present.
