# %%
from pathlib import Path
import os
import torch as t
import numpy as np
import einops
from rich.table import Table
from rich.console import Console
from rich.theme import Theme
import matplotlib.pyplot as plt

BASE_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_PATH)

from board_state_analysis.board_state_utils import (
    setup_model_and_probes,
    load_test_dataset,
    load_head_types,
    HEAD_COLOR_MAP,
    get_head_color,
)

device = "cuda:1" if t.cuda.is_available() else "cpu"
print(f"Using device: {device}")

FIG_DIR = BASE_PATH / "board_state_analysis" / "fig" / "attention_attn_patterns"
os.makedirs(FIG_DIR, exist_ok=True)

# %%
model, n_layers, probes, probe_layer_specific = setup_model_and_probes(device=device)
test_data, board_seqs_id, _ = load_test_dataset([], n_games=500, n_moves=30, device=device)
head_type_all = load_head_types()
n_heads = model.cfg.n_heads

# %%
def diagonal_offsets_mean(A, exclude_first_col=True):
    """Mean of even- and odd-offset diagonals; optionally exclude col 0."""
    if exclude_first_col:
        first_col_mean = A[..., 0].mean(dim=-1)
        A = A[..., 1:, 1:]
    else:
        first_col_mean = None
    n_rows = A.shape[-2]
    even_diags, odd_diags = [], []
    for offset in range(-n_rows + 1, 1):
        diag = A.diagonal(offset=offset, dim1=-2, dim2=-1)
        (even_diags if offset % 2 == 0 else odd_diags).append(diag)
    even_mean = t.cat(even_diags, dim=-1).mean(dim=-1)
    odd_mean  = t.cat(odd_diags,  dim=-1).mean(dim=-1)
    return even_mean, odd_mean, first_col_mean

# %% Collect attention patterns via NNsight trace
pattern_list = {}
with t.no_grad(), model.trace(board_seqs_id):
    for layer in range(n_layers):
        pattern_list[layer] = model.blocks[layer].attn.hook_pattern.output.cpu().save()

# %% calculate head types based on attention patterns, save to JSON for later use in other analyses (e.g. attention attribution blocks)
# attention_attn_patterns.py
# head_type_all = defaultdict(dict)
# for layer in range(model.cfg.n_layers):
#     attention_pattern = pattern_list[layer].value
#     mean_attention_pattern = einops.reduce(
#         attention_pattern, "n_games head row col -> head row col", "mean"
#     )

#     even_mean, odd_mean, first_col_mean = diagonal_offsets_mean(mean_attention_pattern)
#     for head in range(model.cfg.n_heads):
#         even = even_mean[head].item()
#         odd = odd_mean[head].item()
#         if even > 2 * odd:
#             head_type_all[layer][head] = "Yours head"
#         elif odd > 2 * even:
#             head_type_all[layer][head] = "Mine head"
#         else:
#             head_type_all[layer][head] = "Other"

# with open("board_state_analysis/attention_head_types.json", "w") as f:
#     json.dump(head_type_all, f, indent=4, sort_keys=True)


# %% Mean attention pattern: all layers × heads
n_layer_select = 4
fig, axes = plt.subplots(n_layer_select, n_heads, figsize=(16, 8), sharex=True, sharey=True)
for layer in range(n_layer_select):
    mean_pat = einops.reduce(
        pattern_list[layer].value, "n_games head row col -> head row col", "mean"
    ).numpy()
    for head in range(n_heads):
        ax = axes[layer, head]
        im = ax.imshow(mean_pat[head], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(
            f"L{layer}H{head} -- {head_type_all[str(layer)][str(head)]}",
            color=get_head_color(head_type_all, layer, head),
        )
for ax in axes.flat:
    ax.label_outer()
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
fig.savefig(FIG_DIR / "attn_pattern_mean.jpg", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "attn_pattern_mean.pdf", bbox_inches="tight")
plt.show()

# %% Std attention pattern
fig, axes = plt.subplots(n_layer_select, n_heads, figsize=(16, 8), sharex=True, sharey=True)
for layer in range(n_layer_select):
    std_pat = pattern_list[layer].value.std(dim=0).numpy()
    for head in range(n_heads):
        ax = axes[layer, head]
        im = ax.imshow(std_pat[head], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(
            f"L{layer}H{head} -- {head_type_all[str(layer)][str(head)]}",
            color=get_head_color(head_type_all, layer, head),
        )
for ax in axes.flat:
    ax.label_outer()
plt.tight_layout()
fig.savefig(FIG_DIR / "attn_pattern_std.jpg", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "attn_pattern_std.pdf", bbox_inches="tight")
plt.show()

# %% Diagonal offset table (Mine vs Yours mean ± std)
mean_all, std_all = {}, {}
for layer in range(n_layer_select):
    pat = pattern_list[layer].value
    mean_all[layer] = diagonal_offsets_mean(pat.mean(dim=0))
    std_all[layer]  = diagonal_offsets_mean(pat.std(dim=0))

light_theme = Theme({"blue": "blue", "gray": "dim black", "red": "red"})
console = Console(theme=light_theme, record=True)
table = Table(
    title="Attention Pattern Diagonal Offset Stats (Mine=blue/odd, Yours=red/even)",
    show_lines=True, show_header=False,
)
for layer in range(n_layer_select):
    row = []
    yours_mean, mine_mean, _ = mean_all[layer]
    yours_std,  mine_std,  _ = std_all[layer]
    for head in range(n_heads):
        hc = get_head_color(head_type_all, layer, head)
        cell = (
            f"[{hc}]L{layer}H{head}[/{hc}]:\n"
            f"  [blue]μ={mine_mean[head]:.3f}\n  σ={mine_std[head]:.3f}[/blue]\n"
            f"  [red]μ={yours_mean[head]:.3f}\n  σ={yours_std[head]:.3f}[/red]"
        )
        row.append(cell)
    table.add_row(*row)
console.print(table)

# %% Export table as LaTeX
# Mirrors Rich table: rows = layers, columns = heads.
# Each cell = head label (coloured) + Mine diag stats (blue) + Yours diag stats (red).
# Requires in preamble: \usepackage{booktabs, makecell, xcolor}
# \definecolor{minehead}{rgb}{0,0,1}
# \definecolor{yourshead}{rgb}{1,0,0}
# \definecolor{otherhead}{rgb}{0.5,0.5,0.5}
TEX_COLOR = {"Mine head": "minehead", "Yours head": "yourshead", "Other": "otherhead"}

col_spec = "c" * n_heads
lines = []
lines.append(r"% Required packages: booktabs, makecell, xcolor")
lines.append(r"\begin{table}[ht]")
lines.append(r"  \centering")
lines.append(
    r"  \caption{Attention pattern diagonal offset statistics. "
    r"Each cell: head label, "
    r"\textcolor{minehead}{Mine diagonal ($\mu \pm \sigma$)}, "
    r"\textcolor{yourshead}{Yours diagonal ($\mu \pm \sigma$)}.}"
)
lines.append(r"  \label{tab:attn_pattern_diag_offsets}")
lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
lines.append(r"    \toprule")
lines.append("    " + " & ".join(f"H{h}" for h in range(n_heads)) + r" \\")
lines.append(r"    \midrule")

for layer in range(n_layer_select):
    yours_mean, mine_mean, _ = mean_all[layer]
    yours_std,  mine_std,  _ = std_all[layer]
    cells = []
    for head in range(n_heads):
        head_type = head_type_all[str(layer)][str(head)]
        tc = TEX_COLOR[head_type]
        cell = (
            r"\makecell{"
            + f"\\textcolor{{{tc}}}{{L{layer}H{head}}} \\\\ "
            + f"\\textcolor{{minehead}}{{$\\mu$={mine_mean[head]:.3f}, $\\sigma$={mine_std[head]:.3f}}} \\\\ "
            + f"\\textcolor{{yourshead}}{{$\\mu$={yours_mean[head]:.3f}, $\\sigma$={yours_std[head]:.3f}}}"
            + r"}"
        )
        cells.append(cell)
    lines.append("    " + " & ".join(cells) + r" \\")
    if layer < n_layer_select - 1:
        lines.append(r"    \midrule")

lines.append(r"    \bottomrule")
lines.append(r"  \end{tabular}")
lines.append(r"\end{table}")

tex_path = FIG_DIR / "table_diag_offset_stats.tex"
tex_path.write_text("\n".join(lines) + "\n")
print(f"Saved LaTeX table → {tex_path}")

# %%
