# %%
"""
Per-neuron MLP attribution analysis.

Projects W_out onto probe directions to get per-neuron attribution scores,
runs the model to collect actual neuron activations, identifies top-k neurons
by cumulative contribution to a target probe direction at a chosen square,
then plots w_in / w_out probe projections for each top neuron.
"""
from pathlib import Path
import os
import torch as t
import numpy as np
import einops
import matplotlib.pyplot as plt

BASE_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_PATH)

from attention_analysis.attention_utils import (
    setup_model_and_probes,
    load_test_dataset,
    plot_neuron_weight_projections,
)
import utils.arena_utils as arena_utils
import utils.othello_utils as othello_utils

device = "cuda:1" if t.cuda.is_available() else "cpu"
print(f"Using device: {device}")

FIG_DIR = BASE_PATH / "attention_analysis" / "fig" / "mlp_neuron_attribution"
os.makedirs(FIG_DIR, exist_ok=True)

# %%
model, n_layers, probes, probe_layer_specific = setup_model_and_probes(device=device)
test_data, board_seqs_id, _ = load_test_dataset(
    [othello_utils.games_batch_to_flipped_classifier_input_BLC],
    n_games=500,
    device=device,
)

n_neurons = model.cfg.d_mlp

# %%
square_idx = 18
square_label = arena_utils.to_board_label(square_idx)
print(f"Square {square_idx} ({square_label})")

flipped_classifier = t.tensor(
    test_data["games_batch_to_flipped_classifier_input_BLC"]
).to(device)
flipped_classifier_mask = flipped_classifier[..., square_idx]  # [game, seq]

# %% Project W_out onto the flipped probe for the target square
w_out = model.W_out.detach()  # [layer, neuron, d_model]

flipped_attribution = einops.einsum(
    w_out,
    probes["flipped"],
    "layer neuron d_model, layer d_model row col -> layer neuron row col",
)  # [layer, neuron, 8, 8]

# %% Collect actual neuron activations via nnsight tracing
neuron_attribution = {}
with t.no_grad(), model.trace(board_seqs_id, scan=False, validate=False):
    for layer in range(n_layers):
        neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output
        flipped_attr_l = flipped_attribution[layer].flatten(start_dim=-2)[..., square_idx]  # [neuron]

        neuron_attr = einops.einsum(
            neuron_activations_BLD,
            flipped_attr_l,
            "game seq neuron, neuron -> game seq neuron",
        )
        neuron_attribution[layer] = neuron_attr.save()  # [game, seq, neuron]

neuron_attribution = t.stack(
    [neuron_attribution[layer] for layer in range(n_layers)], dim=-2
)  # [game, seq, layer, neuron]

# Sum attributions only at positions where the square was flipped
neuron_attribution_selected = (
    neuron_attribution * flipped_classifier_mask[..., None, None]
).sum(dim=(0, 1))  # [layer, neuron]

# %% Scatter plot: neuron index vs attribution for selected layers
layers_to_plot = [0, 1]
fig, axs = plt.subplots(1, len(layers_to_plot), figsize=(6 * len(layers_to_plot), 5), sharey=True)
for i, layer in enumerate(layers_to_plot):
    axs[i].scatter(
        t.arange(n_neurons).cpu().numpy(),
        neuron_attribution_selected[layer].cpu().numpy(),
        color="blue", alpha=0.7, s=10,
    )
    axs[i].set_title(f"Neuron Attributions for Layer {layer}", fontsize=16)
    axs[i].set_xlabel("Neuron Index", fontsize=14)
    if i == 0:
        axs[i].set_ylabel("Attribution Value", fontsize=14)
    axs[i].grid(True)
plt.tight_layout()
stem = f"neuron_attr_scatter_{square_label}"
fig.savefig(FIG_DIR / f"{stem}.jpg", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
plt.show()

# %% Identify top-k neurons (across layers 0 and 1) and plot w_in/w_out projections
topk_neurons = {}
topk_neuron_idx = t.topk(neuron_attribution_selected[[0, 1]].flatten(), k=2048).indices
for i_k, idx in enumerate(topk_neuron_idx):
    topk_neurons[i_k] = [idx.item() // n_neurons, idx.item() % n_neurons]

probe_names = ["blank", "mine", "flipped", "just_played"]
n_top_plot = 5

for i_k in range(n_top_plot):
    layer, neuron = topk_neurons[i_k]
    fig = plot_neuron_weight_projections(
        model=model,
        layer=layer,
        neuron=neuron,
        probes=probes,
        probe_names=probe_names,
        probe_layer=layer,
        title=f"(Rank {i_k}: L{layer}N{neuron}) MLP Weight Projections — square {square_label}",
    )
    stem = f"neuron_weights_rank{i_k}_L{layer}N{neuron}_{square_label}"
    fig.savefig(FIG_DIR / f"{stem}.jpg", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.show()

# %%