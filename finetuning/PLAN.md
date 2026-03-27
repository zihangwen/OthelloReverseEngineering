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
| `finetuning/data/othello.py` | `get` — imported at module level after sys.path setup |

We do **not** reuse anything from `finetuning_archive/load_model.py` or
`finetuning_archive/finetuning_trainer.py`.

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
├── trainer_probe.py     ← CE + KL training loop (ProbeTrainerConfig, ProbeTrainer)
├── utils_probe.py       ← data loading utilities (load_probe_dirs, build_datasets)
└── run.py               ← CLI entry point + run_finetuning orchestration
```

---

## File 1: `model_probe.py`

### Purpose
Extends `mingpt/model.py` with classes that bake the probe-subspace projection
into the value computation. Also includes the HF → mingpt weight loader.

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
- Holds `Q_valid` as a registered buffer
- Computes `x_raw = F.layer_norm(x, [C])` (no weight/bias)
- Computes `x_v = (x_raw @ Q_valid @ Q_valid.T) * ln1.weight + ln1.bias`
- Passes `(x_ln, x_v)` to `ProbeProjectedAttention`

#### `GPTWithProbeIntervention(GPT)`
- `intervention_layers`: list of layer indices to replace with `ProbeBlock`
- `probe_dirs`: `(d_model, n_dirs)` tensor or `dict[int → Tensor]`
- `load_pretrained_from_hf(hf_model_name)`: maps HF weights to mingpt keys

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
- CE is the standard next-move prediction loss
- KL anchors the output distribution to the original unintervened OthelloGPT,
  preventing the downstream weights from drifting too far
- Implemented as `F.kl_div(log_p_ref, p_finetuned)` since
  `F.kl_div(input, target) = KL(target || input)`

**Freezing strategy:**
- `freeze_up_to: int` — freeze all blocks with index ≤ this value (including
  embeddings and the ProbeBlock itself)
- Only downstream blocks (index > `freeze_up_to`) and the unembedding are trained
- This isolates the question: can downstream layers decode probe-projected V well?

### Classes

#### `ProbeModelTrainerConfig(TrainerConfig)`
Extends `TrainerConfig` with:
- `kl_weight: float = 0.1` — β for the KL term
- `freeze_up_to: int = -1` — freeze blocks 0..freeze_up_to (-1 = freeze nothing)
- `ref_model_name: str` — HF name of the reference model for KL

#### `ProbeModelTrainer(Trainer)`
Extends `Trainer` with:
- Loads reference model at init for KL computation
- Freezes parameters according to `freeze_up_to`
- Overrides loss computation to add `β * KL(p_finetuned || p_ref)`

---

## File 3: `utils_probe.py`

### Purpose
Data loading utilities. Kept separate from the training loop so
`trainer_probe.py` stays focused on the loss and training.

### Functions

#### `load_probe_dirs(probe_keys, probe_layer, device) -> Tensor`
- Calls `utils.probe_utils.load_fold_probes_and_normalize`
- Selects `probes[key][probe_layer]` for each key, normalises, stacks
- Returns `(d_model, n_dirs)` tensor, NaN-cleaned

#### `build_datasets(data_path, test_fraction=0.05) -> (CharDataset, CharDataset)`
- Loads from `.pickle`/`.pkl` or from `data/othello.py` for directory inputs
- Returns `(train_dataset, test_dataset)`

---

## File 4: `run.py`

### Purpose
CLI entry point and top-level orchestration. Contains both `parse_args` and
`run_finetuning` so all pipeline logic lives in one place alongside the flags
that configure it.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--hf_model_name` | `"Baidicoot/Othello-GPT-Transformer-Lens"` | HF repo to load weights from |
| `--data_path` | (required) | Path to pickled Othello games |
| `--probe_keys` | `mine flipped just_played` | Space-separated probe names |
| `--probe_layer` | `5` | Which layer's probes to use for projection |
| `--intervention_layers` | all layers | Which model layers get ProbeBlock |
| `--freeze_up_to` | `-1` | Freeze blocks 0..N during finetuning |
| `--kl_weight` | `0.1` | β for the KL divergence term |
| `--max_epochs` | `10` | |
| `--batch_size` | `64` | |
| `--learning_rate` | `1e-4` | Lower than pretraining |
| `--weight_decay` | `0.1` | |
| `--lr_decay` | flag | Enable cosine LR decay |
| `--ckpt_path` | `None` | Where to save checkpoints |
| `--device` | `cuda` | |

---

## Key Design Decisions

1. **Import from `mingpt/`, do not copy.** `finetuning/` is a proper package
   (has `__init__.py`); all mingpt classes are imported directly.

2. **`trainer_probe.py` is training-only.** Utility functions live in `utils.py`.
   This keeps the training loop focused and testable independently.

3. **CE + KL loss.** Cross-entropy trains next-move prediction under the
   constraint; KL prevents unbounded drift from the original model's output
   distribution. `kl_weight=0` recovers pure CE; `kl_weight→∞` freezes output
   distribution entirely.

4. **Freezing via `freeze_up_to`.** Cleaner than a soft KL penalty for
   controlling early-layer drift. Freezing blocks 0..N and training N+1 onward
   isolates whether downstream layers can adapt to probe-projected V.

5. **Only V is constrained.** K and Q use the full LN-normalised input,
   matching the intervention in `attention_intervention.py`.

6. **`Q_valid` as `register_buffer`.** Not a trainable parameter — excluded
   from `configure_optimizers` automatically. Stays fixed at QR-factorised
   probe directions throughout training.

7. **`strict=False` in `load_state_dict`.** `Q_valid` buffers are not in the
   HF checkpoint; they are pre-computed and left at their QR values.
