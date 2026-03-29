# %%
"""
Shared utilities for attention analysis experiments.

Consolidates the repeated boilerplate found across:
  error_prediction.py, F1_score.py, attention_weights.py,
  attention_intervention.py, attention_attr_blocks.py,
  attention_attr_blocks_w_heads.py, attention_attn_patterns.py,
  attention_source_seq_to_dst.py, multiple_games.py

Usage (at the top of any experiment script):
    from attention_analysis.attention_utils import (
        setup_model_and_probes,
        load_test_dataset,
        stack_residual_streams,
        load_head_types,
        stratify_heads,
        get_head_color,
        HEAD_COLOR_MAP,
        extract_weight_matrices,
        compute_W_OV,
        compute_W_QK,
        compute_W_OV_full,
        compute_W_QK_full,
        topk_accuracy,
        botk_accuracy,
        compute_f1,
        plot_probe_heatmap_grid,
        plot_board_comparison,
    )
"""

import json
from collections import defaultdict
from typing import Callable, Sequence

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch as t


import utils.circuits_utils as circuits_utils
from utils.probe_utils import (
    load_fold_probes_and_normalize,
    calculate_neuron_input_weights,
    calculate_neuron_output_weights,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Baidicoot/Othello-GPT-Transformer-Lens"
DEFAULT_PROBE_LAYER = 5
DEFAULT_STREAMS = ("resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post")
HEAD_COLOR_MAP: dict[str, str] = {
    "Yours head": "red",
    "Mine head":  "blue",
    "Other":      "gray",
}

ROW_LABELS = list("ABCDEFGH")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_model_and_probes(
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    probe_layer: int = DEFAULT_PROBE_LAYER,
):
    """
    Load model and fold probes.

    Returns
    -------
    model, n_layers, probes, probe_layer_specific
        probes              : dict[str, Tensor [n_layers, d_model, 8, 8]]
        probe_layer_specific: dict[str, Tensor [d_model, 8, 8]]  (at probe_layer)
    """
    if device is None:
        device = "cuda:1" if t.cuda.is_available() else "cpu"
    t.set_grad_enabled(False)

    model = circuits_utils.get_model(model_name, device)
    n_layers = model.cfg.n_layers
    probes = load_fold_probes_and_normalize(n_layers, device)
    probe_layer_specific = {name: probes[name][probe_layer] for name in probes}

    return model, n_layers, probes, probe_layer_specific


def load_test_dataset(
    custom_functions: list[Callable],
    n_games: int = 500,
    n_moves: int | None = None,
    device: str | None = None,
    split: str = "test",
):
    """
    Build an OthelloGPT dataset and return the most-used derived tensors.

    Returns
    -------
    test_data, board_seqs_id, board_seqs_square
        board_seqs_id    : LongTensor [n_games, seq_len]  (token ids)
        board_seqs_square: LongTensor [n_games, seq_len]  (decoded square indices)
    """
    if device is None:
        device = "cuda:1" if t.cuda.is_available() else "cpu"

    test_data = circuits_utils.construct_othello_dataset(
        custom_functions=custom_functions,
        n_inputs=n_games,
        split=split,
        device=device,
    )
    board_seqs_id = t.tensor(test_data["encoded_inputs"]).long().to(device)
    board_seqs_square = t.tensor(test_data["decoded_inputs"]).long()

    if n_moves is not None:
        board_seqs_id = board_seqs_id[:, :n_moves]

    return test_data, board_seqs_id, board_seqs_square


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def stack_residual_streams(
    cache,
    n_layers: int,
    streams: Sequence[str] = DEFAULT_STREAMS,
) -> dict[str, t.Tensor]:
    """
    Stack per-layer cache tensors along a new layer dim.

    Returns dict: stream_name -> Tensor [batch, seq, layer, d_model]
    Missing streams (e.g. resid_mid when not cached) are silently skipped.
    """
    result = {}
    for stream in streams:
        try:
            result[stream] = t.stack(
                [cache[stream, layer] for layer in range(n_layers)], dim=2
            )
        except KeyError:
            pass
    return result


# ---------------------------------------------------------------------------
# Attribution projection
# ---------------------------------------------------------------------------

def compute_probe_projections(
    model,
    board_seqs_id: t.Tensor,
    probes: dict[str, t.Tensor],
    head_type_all: dict,
    probe_name_list: list[str] | None = None,
    stream_keys: list[str] | None = None,
) -> dict:
    """
    Run the model, stack residual streams, and compute per-probe projections.

    Parameters
    ----------
    model
        TransformerLens model.
    board_seqs_id
        Token-id sequence, shape ``[seq]`` or ``[1, seq]``.
    probes
        Either ``{name: Tensor[d_model, 8, 8]}`` (layer-specific, when
        ``probe_layer`` is None) or ``{name: Tensor[n_layers, d_model, 8, 8]}``
        (per-layer, slice with ``probe_layer``).
    head_type_all
        ``{str(layer): {str(head): type_str}}`` from ``load_head_types``.
    probe_name_list
        Subset of probe names to iterate; defaults to all keys in ``probes``.
    stream_keys
        Stream names to project; defaults to all streams in the cache.

    Returns
    -------
    probe_projs : dict
        ``probe_projs[probe_name][layer]`` contains:
        - ``"mine_heads"``, ``"yours_heads"``, ``"other_heads"``:
          arrays ``[batch, seq_q, seq_k, 8, 8]`` — summed over selected heads
        - one key per stream in ``stream_keys``:
          array ``[batch, seq, 8, 8]``
        (batch=1 for a single game)
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    d_model = model.cfg.d_model
    W_O = model.W_O.detach()   # [layer, head, d_head, d_model]
    b_O = model.b_O.detach()   # [layer, d_model]

    _, cache = model.run_with_cache(board_seqs_id)
    streams = stack_residual_streams(cache, n_layers)

    if probe_name_list is None:
        probe_name_list = list(probes.keys())
    if stream_keys is None:
        stream_keys = list(streams.keys())

    n_moves = cache["hook_embed"].shape[1]
    device = model.device

    # Causal b_O_cast: [seq_q, seq_k, n_layers, d_model]
    # For query at position q, the bias is split equally among its q+1 attended keys.
    b_O_cast = b_O[None, None].expand(n_moves, n_moves, n_layers, d_model).clone()
    b_O_mask = t.tril(t.ones(n_moves, n_moves, dtype=t.bool, device=device))
    b_O_divisor = t.arange(1, n_moves + 1, device=device).float().view(n_moves, 1, 1, 1)
    b_O_cast[~b_O_mask] = 0.0
    b_O_cast = b_O_cast / b_O_divisor  # [seq_q, seq_k, n_layers, d_model]

    probe_projs: dict = {}
    for probe_name in probe_name_list:
        probe = probes[probe_name]
        if probe.ndim == 3:
            probe = probe[None]  # add layer dim for uniformity: [1, d_model, 8, 8]

        # Stream projections at last query position for all games: [batch, layer, d_model]
        stream_projs = {
            key: einops.einsum(
                streams[key],
                probe,
                "batch seq layer d_model, layer d_model row col -> batch seq layer row col",
            ).cpu().numpy()
            for key in stream_keys
        }

        # [batch, seq_q, seq_k, n_layers, head, d_model]
        attn_out_qk = t.stack([
            einops.einsum(
                cache["v", layer],
                cache["pattern", layer],
                W_O[layer],
                "batch seq_k head d_head, batch head seq_q seq_k, head d_head d_model"
                " -> batch seq_q seq_k head d_model",
            )
            for layer in range(n_layers)
        ], dim=3) + b_O_cast[None, :, :, :, None, :] / model.cfg.n_heads

        # Last query position: [batch, seq_k, n_layers, head, row, col]
        attn_out_qk_last = einops.einsum(
            attn_out_qk,
            probe,
            "batch seq_q seq_k layer head d_model, layer d_model row col -> batch seq_q seq_k layer head row col",
        ).cpu().numpy()

        one_probe_proj: dict = defaultdict(dict)
        for layer in range(n_layers):
            mine_heads  = [h for h in range(n_heads) if head_type_all[str(layer)][str(h)] == "Mine head"]
            yours_heads = [h for h in range(n_heads) if head_type_all[str(layer)][str(h)] == "Yours head"]
            other_heads = [h for h in range(n_heads) if head_type_all[str(layer)][str(h)] == "Other"]

            one_probe_proj[layer]["mine_heads"]  = attn_out_qk_last[:, :, :, layer, mine_heads].sum(-3) # [batch, seq_q, seq_k, layer, head, row, col] -> [batch, seq_q, seq_k, row, col]
            one_probe_proj[layer]["yours_heads"] = attn_out_qk_last[:, :, :, layer, yours_heads].sum(-3) # [batch, seq_q, seq_k, layer, head, row, col] -> [batch, seq_q, seq_k, row, col]
            one_probe_proj[layer]["other_heads"] = attn_out_qk_last[:, :, :, layer, other_heads].sum(-3) # [batch, seq_q, seq_k, layer, head, row, col] -> [batch, seq_q, seq_k, row, col]
            for key in stream_keys:
                one_probe_proj[layer][key] = stream_projs[key][:, :, layer]

        probe_projs[probe_name] = one_probe_proj

    return probe_projs


# ---------------------------------------------------------------------------
# Head type utilities
# ---------------------------------------------------------------------------

def load_head_types(
    json_path: str = "attention/attention_head_types.json",
) -> dict[str, dict[str, str]]:
    """Load head type labels; returns {str(layer): {str(head): type_str}}."""
    with open(json_path, "r") as f:
        return json.load(f)


def stratify_heads(
    head_type_all: dict,
    n_layers: int,
    n_heads: int,
) -> dict[str, list[tuple[int, int]]]:
    """
    Group (layer, head) pairs by type.

    Returns dict: type_str -> [(layer, head), ...]
    """
    groups: dict[str, list] = defaultdict(list)
    for layer in range(n_layers):
        for head in range(n_heads):
            groups[head_type_all[str(layer)][str(head)]].append((layer, head))
    return dict(groups)


def get_head_color(head_type_all: dict, layer: int, head: int) -> str:
    """Return the matplotlib color string for a single head."""
    return HEAD_COLOR_MAP[head_type_all[str(layer)][str(head)]]


# ---------------------------------------------------------------------------
# Circuit weight utilities
# ---------------------------------------------------------------------------

def extract_weight_matrices(model) -> dict[str, t.Tensor]:
    """
    Clone all attention weight matrices from the model in one call.

    Returns dict with keys: W_Q, W_K, W_V, W_O, W_E, W_U
    """
    return {
        "W_Q": model.W_Q.detach().clone(),        # [layer, head, d_model, d_head]
        "W_K": model.W_K.detach().clone(),        # [layer, head, d_model, d_head]
        "W_V": model.W_V.detach().clone(),        # [layer, head, d_model, d_head]
        "W_O": model.W_O.detach().clone(),        # [layer, head, d_head, d_model]
        "W_E": model.W_E[1:].detach().clone(),    # [vocab, d_model]  (skip "pass")
        "W_U": model.W_U[:, 1:].detach().clone(), # [d_model, vocab]  (skip "pass")
    }


def compute_W_OV(W_V: t.Tensor, W_O: t.Tensor) -> t.Tensor:
    """
    W_OV = W_V @ W_O  [layer, head, d_model, d_model]

    Describes what information gets moved from source to destination residual stream.
    """
    return einops.einsum(
        W_V, W_O,
        "layer head d_model1 d_head, layer head d_head d_model2 -> layer head d_model1 d_model2",
    )


def compute_W_QK(W_Q: t.Tensor, W_K: t.Tensor) -> t.Tensor:
    """
    W_QK = W_Q @ W_K^T  [layer, head, d_model, d_model]

    Bilinear form describing which residual stream vectors attend to which.
    """
    return einops.einsum(
        W_Q, W_K,
        "layer head d_model1 d_head, layer head d_model2 d_head -> layer head d_model1 d_model2",
    )


def compute_W_OV_full(W_E: t.Tensor, W_OV: t.Tensor, W_U: t.Tensor) -> t.Tensor:
    """W_E @ W_OV @ W_U  [layer, head, vocab, vocab]  (full token-space OV circuit)."""
    return einops.einsum(
        W_E, W_OV, W_U,
        "vocab1 d_model1, layer head d_model1 d_model2, d_model2 vocab2 -> layer head vocab1 vocab2",
    )


def compute_W_QK_full(W_E: t.Tensor, W_QK: t.Tensor) -> t.Tensor:
    """W_E @ W_QK @ W_E^T  [layer, head, vocab, vocab]  (full token-space QK circuit)."""
    return einops.einsum(
        W_E, W_QK, W_E,
        "vocab1 d_model1, layer head d_model1 d_model2, vocab2 d_model2 -> layer head vocab1 vocab2",
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def topk_accuracy(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """
    Binarise by selecting the top-K values, where K = |positives in gt|.
    Used for *flipped* square detection (highest projection = flipped).

    Returns (threshold, accuracy).
    """
    p = np.asarray(pred).flatten().astype(float)
    g = np.asarray(gt).flatten().astype(float)
    k = int(g.sum())
    if k == 0:
        return float("nan"), float("nan")
    top_k_idx = np.argsort(p)[-k:]
    binarized = np.zeros_like(g)
    binarized[top_k_idx] = 1
    return p[top_k_idx[0]], (binarized == g).mean()


def botk_accuracy(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """
    Binarise by selecting the bottom-K values, where K = |positives in gt|.
    Used for *mine/empty* square detection (lowest projection = mine).

    Returns (threshold, accuracy).
    """
    p = np.asarray(pred).flatten().astype(float)
    g = np.asarray(gt).flatten().astype(float)
    k = int(g.sum())
    if k == 0:
        return float("nan"), float("nan")
    bottom_k_idx = np.argsort(p)[:k]
    binarized = np.ones_like(g)
    binarized[bottom_k_idx] = 0
    return p[bottom_k_idx[-1]], (binarized == g).mean()


def compute_f1(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    """
    Compute precision, recall, and F1 from confusion matrix counts.
    All inputs can be scalars or same-shape arrays.
    """
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_probe_heatmap_grid(
    data: Sequence[np.ndarray],
    n_rows: int,
    n_cols: int,
    title: str,
    cell_titles: Sequence[str],
    cell_title_colors: Sequence[str] | None = None,
    cmap: str = "RdBu",
    scale: str = "symmetric",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize_per_cell: tuple[float, float] = (3.0, 3.0),
    colorbar_label: str = "",
) -> plt.Figure:
    """
    [n_rows × n_cols] grid of 8×8 heatmaps with a single shared colorbar.

    Parameters
    ----------
    data              : list of n_rows*n_cols arrays each [8, 8], row-major order
    n_rows, n_cols    : grid dimensions
    title             : figure suptitle
    cell_titles       : per-cell subplot title, length must equal n_rows * n_cols
    cell_title_colors : per-cell title color (defaults to "black")
    scale             : "symmetric" — v_abs = max(|vmin|, |vmax|), vmin=-v_abs, vmax=v_abs
                        "positive"  — v_abs = |vmax|,              vmin=-v_abs, vmax=v_abs
    """
    assert len(data) == n_rows * n_cols
    assert len(cell_titles) == n_rows * n_cols

    # Colour limits
    if vmin is None or vmax is None:
        flat = np.concatenate([d.flatten() for d in data])
        flat = flat[~np.isnan(flat)]
        data_min = flat.min() if len(flat) else 0.0
        data_max = flat.max() if len(flat) else 1.0
        vmin = vmin if vmin is not None else data_min
        vmax = vmax if vmax is not None else data_max

    if scale == "symmetric":
        v_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -v_abs, v_abs
    elif scale == "positive":
        v_abs = abs(vmax)
        vmin, vmax = -v_abs, v_abs

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows + 1.5),
    )
    fig.suptitle(title, fontsize=16)

    # Normalise axs to always be 2-D
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs[np.newaxis, :]
    elif n_cols == 1:
        axs = axs[:, np.newaxis]

    im = None
    for idx, arr in enumerate(data):
        r, c = divmod(idx, n_cols)
        ax = axs[r, c]
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(cell_titles[idx], color=(cell_title_colors[idx] if cell_title_colors else "black"))
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_yticklabels(ROW_LABELS)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    if colorbar_label:
        cb.set_label(colorbar_label, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
    return fig


def plot_neuron_weight_projections(
    model,
    layer: int,
    neuron: int,
    probes: dict[str, t.Tensor],
    probe_names: Sequence[str],
    title: str = "",
    probe_layer: int | None = None,
    figsize_per_cell: tuple[float, float] = (3.0, 3.0),
) -> plt.Figure:
    """
    Compute and plot w_in / w_out projections onto probe directions for a single neuron.

    Produces a [2 × n_probes] grid: top row = w_in, bottom row = w_out.

    Parameters
    ----------
    model
        TransformerLens model.
    layer, neuron
        Which MLP neuron to inspect.
    probes
        ``{name: Tensor[d_model, 8, 8]}`` (layer-specific, ``probe_layer=None``)
        or ``{name: Tensor[n_layers, d_model, 8, 8]}`` (per-layer, slice with ``probe_layer``).
    probe_names
        Ordered list of probe keys to show as columns.
    title
        Figure suptitle.
    probe_layer
        Layer index to slice when probes are per-layer tensors.
    """
    w_in_projs, w_out_projs = [], []
    for name in probe_names:
        p = probes[name]
        if probe_layer is not None:
            p = p[probe_layer]  # [d_model, 8, 8]
        assert p.ndim == 3, f"Probe {name} has unexpected shape {p.shape}"
        w_in_projs.append(
            calculate_neuron_input_weights(model, p, layer, neuron).cpu().numpy()
        )
        w_out_projs.append(
            calculate_neuron_output_weights(model, p, layer, neuron).cpu().numpy()
        )

    data = w_in_projs + w_out_projs
    cell_titles = (
        [f"w_in @ {p}" for p in probe_names]
        + [f"w_out @ {p}" for p in probe_names]
    )
    return plot_probe_heatmap_grid(
        data=data,
        n_rows=2,
        n_cols=len(probe_names),
        title=title,
        cell_titles=cell_titles,
        scale="symmetric",
        figsize_per_cell=figsize_per_cell,
    )


