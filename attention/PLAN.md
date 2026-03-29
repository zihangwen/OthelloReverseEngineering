# Attention Directory Refactoring Plan

## 1. Current File Inventory

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `error_prediction.py` | Evaluates model top-N accuracy on legal moves; plots predicted vs actual board | 143 | Active |
| `plot_board_states.py` | Minimal wrapper — plots board states from test data | 40 | Active (thin) |
| `F1_score.py` | Computes precision/recall/F1 for flipped & mine squares; topk vs botk thresholding | 319 | Active (canonical) |
| `multiple_games.py` | Large-scale projection + threshold clustering across games | 359 | Active |
| `attention_weights.py` | Analyzes QK and OV circuits; cosine similarity between probe dirs through weight matrices | 385 | Active |
| `attention_intervention.py` | Causal ablation of V projections; accuracy/KL vs probe/random/zero baselines | 308 | Active |
| `attention_attr_blocks.py` | MLP neuron attribution to flipped squares; top-K neuron w_in/w_out heatmaps | 278 | Active |
| `attention_attr_blocks_w_heads.py` | Attention attribution split by Mine/Yours/Other head types | 404 | Active |
| `attention_attn_patterns.py` | Visualizes attention patterns; diagonal offset analysis for Mine vs Yours heads | 331 | Active |
| `attention_source_seq_to_dst.py` | Per-key attribution to destination; projects attn_out_qk onto probe directions | 682 | Active (most complex) |
| `mlp_weights.py` | MLP w_in/w_out projections onto probe dirs for top neurons | 165 | **Redundant** — overlaps heavily with `attention_attr_blocks.py` |
| `archive/flipped_thresholding_binarization.py` | Early topk-only version of F1 analysis | ~426 | **Superseded** by `F1_score.py` |

---

## 2. Duplicated Code Patterns

### 2a. Already available in `utils/` — just not being imported

These patterns are re-implemented inline across attention scripts but already exist in `utils/`:

| Pattern | Where it exists in `utils/` | Duplicated in |
|---------|------------------------------|---------------|
| Model loading | `circuits_utils.get_model()` | Every file (~9×) |
| Dataset construction | `circuits_utils.construct_othello_dataset()` | Every file (~9×) |
| Probe loading | `probe_utils.load_fold_probes_and_normalize()` | Every file (~9×) |
| `calculate_neuron_input_weights()` | `probe_utils` | `attention_attr_blocks.py`, `mlp_weights.py` |
| `calculate_neuron_output_weights()` | `probe_utils` | `attention_attr_blocks.py`, `mlp_weights.py` |

The biggest issue is that each script re-writes the same 15–25 line setup block (model + probes + dataset) instead of calling these shared functions.

### 2b. Missing from `utils/` — needs to be extracted

| Pattern | Files | Estimated duplicated lines |
|---------|-------|---------------------------|
| Residual stream stacking from cache | `attention_source_seq_to_dst.py`, `attention_attr_blocks_w_heads.py`, `multiple_games.py`, `F1_score.py` | ~60–80 |
| Probe projection einsum | 6+ files (2–5 instances each) | ~40–60 |
| Symmetric heatmap grid plotting (8×8 boards) | 7+ files (15–25 lines each) | ~120–175 |
| Head type loading + stratification | `attention_attr_blocks_w_heads.py`, `attention_attn_patterns.py`, `attention_weights.py`, `attention_intervention.py` | ~50–60 |
| QK/OV circuit composition | `attention_weights.py` | ~30 |
| topk/botk binarization + F1 metrics | `F1_score.py`, partially in `multiple_games.py` | ~40 |
| Intervention direction projection | `attention_intervention.py` | isolated, ~80 |

---

## 3. Proposed Utility Module: `attention/attention_utils.py`

Create a single new file `attention/attention_utils.py` containing attention-experiment-specific shared code. This keeps it separate from the general-purpose `utils/` package (which has broader project scope).

### Functions to add:

```python
# --- Setup ---
def setup_model_and_probes(model_name, device):
    """Load model and fold probes; return (model, n_layers, probes, probe_layer_specific)."""

def load_test_dataset(custom_functions, n_games, device):
    """Construct test dataset; return (test_data, board_seqs_id, board_seqs_square)."""

# --- Cache utilities ---
def stack_residual_streams(cache, n_layers, streams=None):
    """Stack cache tensors by layer; returns dict of [batch, seq, layer, d_model] tensors.
    Default streams: resid_pre, attn_out, resid_mid, mlp_out, resid_post."""

def run_model_and_cache(model, board_seqs_id, names_filter=None):
    """Run model with NNsight trace; return cache dict."""

# --- Probe projection ---
def project_onto_probe(activations, probe, probe_name=None):
    """einsum activations [..., d_model] × probe [d_model, row, col] → [..., row, col]."""

def project_all_streams(streams_dict, probe_layer_specific):
    """Project each residual stream tensor onto all probes; return nested dict."""

# --- Head type utilities ---
def load_head_types(json_path="attention/attention_head_types.json"):
    """Load head type dict from JSON; return {layer: {head: type_str}}."""

def stratify_heads(head_type_all, n_layers, n_heads):
    """Return {type_name: [(layer, head), ...]} grouping."""

# --- Circuit utilities ---
def compute_W_OV(model, layer, head):
    """Return W_V @ W_O for a single head."""

def compute_W_QK(model, layer, head):
    """Return W_Q @ W_K^T for a single head."""

# --- Metrics ---
def topk_accuracy(pred, gt):
    """Select top-K predictions matching |gt| positives; return (threshold, accuracy)."""

def botk_accuracy(pred, gt):
    """Select bottom-K predictions matching |gt| positives; return (threshold, accuracy)."""

def compute_f1(tp, fp, tn, fn):
    """Return dict with precision, recall, f1."""

# --- Plotting ---
def plot_probe_heatmap_grid(data, row_labels, col_labels, title,
                             cmap="RdBu", symmetric=True, figsize_per_cell=(3, 3)):
    """Generic [rows × cols] grid of 8×8 heatmaps with shared colorbar."""

def plot_board_comparison(board_states, predictions, game_idx, moves,
                           pred_title="Predicted", gt_title="Ground Truth"):
    """Side-by-side board rows: one row for GT, one for predictions."""
```

### Functions to add to `utils/plot_utils.py` (project-wide):

- `plot_board_heatmap_grid()` — extend the existing file since board visualization is used broadly

---

## 4. File-by-File Refactoring Actions

### `mlp_weights.py` → **Delete or merge into `attention_attr_blocks.py`**
- `attention_attr_blocks.py` already computes top-K neuron attributions and plots w_in/w_out
- `mlp_weights.py` does the same with slightly different top-K selection logic
- Action: keep the best version in `attention_attr_blocks.py`, delete `mlp_weights.py`

### `plot_board_states.py` → **Delete**
- 40 lines that call `plot_board_states()` already in `utils/plot_utils.py`
- Can be replaced by a one-liner in a notebook or merged into another script

### `archive/flipped_thresholding_binarization.py` → **Already archived; leave as is**

### All other files → **Refactor setup blocks**
Each of the remaining 8 files should replace their setup boilerplate with:
```python
from attention.attention_utils import setup_model_and_probes, load_test_dataset
model, n_layers, probes, probe_layer_specific = setup_model_and_probes(model_name, device)
test_data, board_seqs_id, board_seqs_square = load_test_dataset(custom_functions, 500, device)
```

---

## 5. Experiment Categorization

The experiments fall into 4 categories:

### A. Weight-space analysis (no data needed)
- `attention_weights.py` — QK/OV circuits × probes
- `mlp_weights.py` / `attention_attr_blocks.py` — MLP w_in/w_out × probes

### B. Activation-space analysis (needs model forward pass)
- `attention_attn_patterns.py` — attention pattern statistics
- `attention_source_seq_to_dst.py` — per-key attribution to destination token
- `attention_attr_blocks_w_heads.py` — attention attribution by head type
- `multiple_games.py` — residual stream clustering

### C. Evaluation metrics
- `error_prediction.py` — top-N accuracy on legal moves
- `F1_score.py` — flipped/mine detection F1

### D. Causal intervention
- `attention_intervention.py` — V-direction ablation

---

## 6. Phased Refactoring Plan

### Phase 1 — Extract `attention_utils.py` (quick wins, high impact)
1. Write `attention/attention_utils.py` with `setup_model_and_probes()` and `load_test_dataset()`
2. Replace setup boilerplate in all 8 active experiment files
3. Add `stack_residual_streams()` and replace in the 4 files that do it inline
4. Add `load_head_types()` and `stratify_heads()`, use in 4 files

**Expected reduction: ~200–300 lines of duplicated setup code removed**

### Phase 2 — Consolidate plotting
1. Add `plot_probe_heatmap_grid()` to `attention_utils.py`
2. Replace the ad-hoc plotting blocks in `attention_source_seq_to_dst.py`, `attention_attr_blocks_w_heads.py`, `F1_score.py`, `attention_attn_patterns.py`
3. Extend `utils/plot_utils.py` with `plot_board_comparison()` for side-by-side boards

**Expected reduction: ~120–175 lines removed**

### Phase 3 — Consolidate metrics + delete redundant files
1. Add `topk_accuracy()`, `botk_accuracy()`, `compute_f1()` to `attention_utils.py` (canonical from `F1_score.py`)
2. Add `compute_W_OV()`, `compute_W_QK()` for `attention_weights.py`
3. Delete `mlp_weights.py` (merge any unique logic into `attention_attr_blocks.py`)
4. Delete `plot_board_states.py`

**Expected reduction: ~50–80 lines + 2 files deleted**

### Phase 3 result: each experiment file becomes a thin script that:
- Calls setup helpers (5 lines)
- Runs the specific experiment logic (the unique part)
- Calls shared plotting utilities (3–5 lines)

---

## 7. File Structure After Refactoring

```
attention/
├── attention_utils.py          ← NEW: all shared setup, cache, projection, plot, metrics
├── attention_weights.py        ← weight-space circuit analysis
├── attention_attn_patterns.py  ← attention pattern visualization
├── attention_attr_blocks.py    ← MLP neuron attribution (absorbs mlp_weights.py)
├── attention_attr_blocks_w_heads.py  ← attention attribution by head type
├── attention_source_seq_to_dst.py    ← per-key attribution analysis
├── attention_intervention.py   ← causal V-ablation
├── F1_score.py                 ← flipped/mine detection metrics
├── error_prediction.py         ← top-N legal move accuracy
├── multiple_games.py           ← clustering across games
├── attention_head_types.json   ← head type labels
└── archive/
    └── flipped_thresholding_binarization.py
```

Files to delete: `mlp_weights.py`, `plot_board_states.py`
