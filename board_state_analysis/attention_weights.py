# %%
from pathlib import Path
import os
import torch as t
import numpy as np
import einops
from rich.table import Table
from rich.console import Console
from rich.theme import Theme

BASE_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_PATH)

from board_state_analysis.board_state_utils import (
    setup_model_and_probes,
    load_head_types,
    extract_weight_matrices,
    compute_W_OV,
    compute_W_QK,
    plot_probe_heatmap_grid,
    HEAD_COLOR_MAP,
    get_head_color,
)
from utils.arena_utils import label_to_square

device = "cuda:1" if t.cuda.is_available() else "cpu"
print(f"Using device: {device}")

FIG_DIR = BASE_PATH / "board_state_analysis" / "fig" / "attention_weights"
os.makedirs(FIG_DIR, exist_ok=True)

# %%
model, n_layers, probes, _ = setup_model_and_probes(device=device)
head_type_all = load_head_types()
weights = extract_weight_matrices(model)
W_Q, W_K, W_V, W_O = weights["W_Q"], weights["W_K"], weights["W_V"], weights["W_O"]
W_E, W_U = weights["W_E"], weights["W_U"]

n_heads = model.cfg.n_heads

# %%
W_OV = compute_W_OV(W_V, W_O)   # [layer, head, d_model, d_model]
W_QK = compute_W_QK(W_Q, W_K)   # [layer, head, d_model, d_model]

# %% OV cosine-similarity heatmaps: probe_src -> W_OV -> probe_dst
label = "C1"
square = label_to_square(label)
row, col = square // 8, square % 8

probe_name_pair = [("flipped", "mine"), ("just_played", "mine"), ("mine", "mine")]
n_layer_select = 4

all_cos_sims_dict = {}
for probe_name1, probe_name2 in probe_name_pair:
    all_cos_sims = []
    for layer in range(n_layer_select):
        for head in range(n_heads):
            probe_src = probes[probe_name1][layer]
            probe_dst = probes[probe_name2][layer]
            src_OV = einops.einsum(
                probe_src, W_OV[layer, head],
                "d_model_src row col, d_model_src d_model_dst -> d_model_dst row col",
            )
            src_OV_norm = src_OV / src_OV.norm(dim=0, keepdim=True)
            cos_sim = einops.einsum(
                src_OV_norm, probe_dst,
                "d_model_dst row col, d_model_dst row col -> row col",
            ).cpu().numpy()
            all_cos_sims.append(cos_sim)
    all_cos_sims_dict[f"{probe_name1}_to_{probe_name2}"] = all_cos_sims

    cell_titles = [
        f"L{layer}H{head} -- {head_type_all[str(layer)][str(head)]}"
        for layer in range(n_layer_select)
        for head in range(n_heads)
    ]
    cell_colors = [
        get_head_color(head_type_all, layer, head)
        for layer in range(n_layer_select)
        for head in range(n_heads)
    ]

    fig = plot_probe_heatmap_grid(
        data=all_cos_sims,
        n_rows=n_layer_select,
        n_cols=n_heads,
        title=f"OV cosine sim: {probe_name1} → {probe_name2}",
        cell_titles=cell_titles,
        cell_title_colors=cell_colors,
    )
    stem = f"OV_cossim_{probe_name1}_to_{probe_name2}"
    fig.savefig(FIG_DIR / f"{stem}.jpg", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.show()

# %% Fixed-threshold scoring table
tau = 0.1
scores_dict = {}
for probe_name1, probe_name2 in probe_name_pair:
    all_cos_sims = all_cos_sims_dict[f"{probe_name1}_to_{probe_name2}"]
    scores_values = []
    for cos_sim in all_cos_sims:
        flat = cos_sim.flatten()
        scores_values.append([
            (flat > tau).mean().item(),
            (np.abs(flat) <= tau).mean().item(),
            (flat < -tau).mean().item(),
        ])
    scores_dict[f"{probe_name1}_to_{probe_name2}"] = scores_values

# %% Rich table (terminal display)
color_map = HEAD_COLOR_MAP
light_theme = Theme({"blue": "blue", "gray": "dim black", "red": "red"})

probe_name1, probe_name2 = probe_name_pair[0]
scores_values = scores_dict[f"{probe_name1}_to_{probe_name2}"]
console = Console(theme=light_theme, record=True)
table = Table(
    title=f"Cosine Similarity Score for {probe_name1} → {probe_name2} (τ={tau})",
    show_lines=True, show_header=False,
)
for layer in range(n_layer_select):
    row = []
    for head in range(n_heads):
        idx = layer * n_heads + head
        s0, s1, s2 = scores_values[idx]
        head_type = head_type_all[str(layer)][str(head)]
        hc = color_map[head_type]
        cell = (
            f"[{hc}]L{layer}H{head}[/{hc}]:\n"
            f"  [blue]{s0*100:.1f}%[/blue]\n"
            f"  [gray]{s1*100:.1f}%[/gray]\n"
            f"  [red]{s2*100:.1f}%[/red]"
        )
        row.append(cell)
    table.add_row(*row)
console.print(table)

# %% Export table as LaTeX (one .tex per probe pair)
# Mirrors the Rich table: rows = layers, columns = heads.
# Each cell = head label (coloured by type) + 3 stacked percentages.
# Requires in preamble: \usepackage{booktabs, makecell, xcolor}
# Colour aliases expected:  \definecolor{minehead}{rgb}{0,0,1}   (blue)
#                           \definecolor{yourshead}{rgb}{1,0,0}  (red)
#                           \definecolor{otherhead}{rgb}{0.5,0.5,0.5} (gray)
TEX_COLOR = {"Mine head": "minehead", "Yours head": "yourshead", "Other": "otherhead"}

for p1, p2 in probe_name_pair:
    key = f"{p1}_to_{p2}"
    sv  = scores_dict[key]

    col_spec = "c" * n_heads  # one column per head, no extra columns

    lines = []
    lines.append(r"% Required packages: booktabs, makecell, xcolor")
    lines.append(r"% Define colours:  \definecolor{minehead}{rgb}{0,0,1}")
    lines.append(r"%                  \definecolor{yourshead}{rgb}{1,0,0}")
    lines.append(r"%                  \definecolor{otherhead}{rgb}{0.5,0.5,0.5}")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{OV cosine similarity scores: \texttt{"
        + p1.replace("_", r"\_") + r"} $\to$ \texttt{" + p2.replace("_", r"\_")
        + r"} ($\tau=" + str(tau) + r"$). "
        + r"Each cell: \textcolor{minehead}{head label}, "
        + r"\textcolor{blue}{$>$$\tau$}, \textcolor{otherhead}{$|\cdot|\le\tau$}, \textcolor{yourshead}{$<$$-\tau$}.}"
    )
    lines.append(r"  \label{tab:OV_cossim_" + key + r"}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # Column header: H0 … H{n_heads-1}
    lines.append("    " + " & ".join(f"H{h}" for h in range(n_heads)) + r" \\")
    lines.append(r"    \midrule")

    for layer in range(n_layer_select):
        cells = []
        for head in range(n_heads):
            idx = layer * n_heads + head
            s0, s1, s2 = sv[idx]
            head_type = head_type_all[str(layer)][str(head)]
            tc = TEX_COLOR[head_type]
            # \makecell stacks lines inside a single cell
            cell = (
                r"\makecell{"
                + f"\\textcolor{{{tc}}}{{L{layer}H{head}}} \\\\ "
                + f"\\textcolor{{blue}}{{{s0*100:.1f}\\%}} \\\\ "
                + f"\\textcolor{{otherhead}}{{{s1*100:.1f}\\%}} \\\\ "
                + f"\\textcolor{{yourshead}}{{{s2*100:.1f}\\%}}"
                + r"}"
            )
            cells.append(cell)
        lines.append("    " + " & ".join(cells) + r" \\")
        if layer < n_layer_select - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    tex_path = FIG_DIR / f"table_OV_cossim_{key}.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table → {tex_path}")

# %%