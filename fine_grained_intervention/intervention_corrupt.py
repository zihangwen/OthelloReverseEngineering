"""Fine-grained intervention for length 3 legal move conditions"""

import torch as t
from torch import Tensor
import numpy as np
import einops
from nnsight.models import NNsightModel

import neel_utils as neel_utils
import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from cont_dt.cont_dt_viz import Condition, DecisionTreeResults, find_neurons_for_query

import json
import sys
from jaxtyping import Bool, Float, Int
from typing import Tuple
from tqdm import tqdm
from rich import print as rprint
from rich.table import Table
from pprint import pprint
from dataclasses import dataclass

from fine_grained_intervention.utils import (
    load_model,
    load_data,
    get_filtered_positions,
    sanity_check,
    find_neurons_for_query_DLA,
    merge_dicts,
    get_legal_moves_batch,
    right_pad,
    no_ablation,
)


@dataclass(frozen=True)
class InterventionMetrics:
    logit_diff: float
    prob_diff: float
    clean_accuracy: float
    corrupted_accuracy: float
    accuracy_diff: float
    below_1_percent: float
    below_5_percent: float
    below_10_percent: float


def zero_ablation(
    model: NNsightModel,
    batch_tensor: Int[Tensor, "batch seq"],
    batch_indices: Int[Tensor, "batch"],
    last_token_indices: Int[Tensor, "batch"],
    legal_square_id: int,
    neurons: dict[int, list[int]],
) -> Tuple[Float[Tensor, "batch d_vocab"], Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    with model.trace(batch_tensor):
        for layer in range(1, model.cfg.n_layers):
            if neurons[layer]:
                neuron_indices = t.tensor(neurons[layer], device=device)
                n_neurons = len(neurons[layer])
                batch_indices_repeated = einops.repeat(
                    batch_indices,
                    'batch -> batch neurons',
                    neurons=n_neurons,
                )
                last_token_indices_repeated = einops.repeat(
                    last_token_indices,
                    'batch -> batch neurons',
                    neurons=n_neurons,
                )
                neuron_indices_repeated = einops.repeat(
                    neuron_indices,
                    'neurons -> batch neurons',
                    batch=len(batch_tensor),
                )
                model.blocks[layer].mlp.hook_post.output[
                    batch_indices_repeated, 
                    last_token_indices_repeated, 
                    neuron_indices_repeated
                ] = 0
        
        logits = model.unembed.output[batch_indices, last_token_indices].save()
        probs = t.nn.functional.softmax(logits, dim=-1)
        
        logits_square = logits[:, legal_square_id].save()
        probs_square = probs[:, legal_square_id].save()

        return logits, logits_square, probs_square


def is_accurate_batch(
    logits: Float[Tensor, "batch d_vocab"],
    legal_moves_batch: list[list[int]],
    legal_square_id,
) -> list[bool]:
    """If there are K legal moves, we say accurate if the legal square is in the
    top K logits"""
    accurate = []

    for j, legal_moves in enumerate(legal_moves_batch):
        k = len(legal_moves)

        top_k_tokens = logits[j].topk(k=k).indices.tolist()
        accurate.append(legal_square_id in top_k_tokens)
        
    return accurate


def below_threshold(
    probs_square: Float[Tensor, "batch"],
    legal_moves_batch: list[list[int]],
    alpha: float = 0.01,
) -> list[bool]:
    return [(prob < alpha * 1 / len(legal_moves)).item() for prob, legal_moves in zip(probs_square, legal_moves_batch)]


def intervene(
    model: NNsightModel, 
    positions: list[Int[Tensor, "n_moves"]], 
    query: list[Condition],
    dt_queries: list[list[Condition]] | None = None, 
    dla_positions: list[Int[Tensor, "n_moves"]] | None = None,
    dla: bool = False,
    k: int | None = None,
    batch_size: int = 1024,
    device = "cuda",
) -> InterventionMetrics:
    if dla:
        neurons = find_neurons_for_query_DLA(model, dla_positions, query, k=k, device=device)
    else:
        neurons = merge_dicts([find_neurons_for_query(query) for query in dt_queries])

    print(f"Ablating {sum(len(neurons) for neurons in neurons.values())} neurons")
    legal_square_id = neel_utils.to_id(query[0].feature_name.split()[0])

    total_logit_diff = 0
    total_prob_diff = 0
    total_clean_accuracy = 0
    total_corrupted_accuracy = 0
    total_below_1_percent = 0
    total_below_5_percent = 0
    total_below_10_percent = 0

    for i in tqdm(range(0, len(positions), batch_size), desc="Batches"):
        batch = positions[i:i + batch_size]

        legal_moves_batch = get_legal_moves_batch(batch)
        batch_tensor, batch_indices, last_token_indices = right_pad(batch, device=device) 

        clean_logits, clean_logits_square, clean_probs_square = no_ablation(
            model, 
            batch_tensor, 
            batch_indices,
            last_token_indices,
            legal_square_id,
        )

        corrupted_logits, corrupted_logits_square, corrupted_probs_square = zero_ablation(
            model, 
            batch_tensor, 
            batch_indices,
            last_token_indices,
            legal_square_id,
            neurons,
        )

        is_accurate_clean = is_accurate_batch(clean_logits, legal_moves_batch, legal_square_id)
        total_clean_accuracy += sum(is_accurate_clean)

        is_accurate_corrupted = is_accurate_batch(corrupted_logits, legal_moves_batch, legal_square_id)
        total_corrupted_accuracy += sum(is_accurate_corrupted)

        below_1_percent_corrupted = below_threshold(corrupted_probs_square, legal_moves_batch, alpha=0.01)
        total_below_1_percent += sum(below_1_percent_corrupted)

        below_5_percent_corrupted = below_threshold(corrupted_probs_square, legal_moves_batch, alpha=0.05)
        total_below_5_percent += sum(below_5_percent_corrupted)

        below_10_percent_corrupted = below_threshold(corrupted_probs_square, legal_moves_batch, alpha=0.1)
        total_below_10_percent += sum(below_10_percent_corrupted)
        
        total_logit_diff += (clean_logits_square - corrupted_logits_square).sum().item()
        total_prob_diff += (clean_probs_square - corrupted_probs_square).sum().item()
    
    avg_logit_diff = total_logit_diff / len(positions)
    avg_prob_diff = total_prob_diff / len(positions)
    avg_clean_accuracy = total_clean_accuracy / len(positions)
    avg_corrupted_accuracy = total_corrupted_accuracy / len(positions)
    avg_num_below_1_percent = total_below_1_percent / len(positions)
    avg_num_below_5_percent = total_below_5_percent / len(positions)
    avg_num_below_10_percent = total_below_10_percent / len(positions)

    return InterventionMetrics(
        logit_diff=avg_logit_diff,
        prob_diff=avg_prob_diff,
        clean_accuracy=avg_clean_accuracy,
        corrupted_accuracy=avg_corrupted_accuracy,
        accuracy_diff=avg_clean_accuracy - avg_corrupted_accuracy,
        below_1_percent=avg_num_below_1_percent,
        below_5_percent=avg_num_below_5_percent,
        below_10_percent=avg_num_below_10_percent,
    )
        

def print_table(intervened_metrics: InterventionMetrics, control_metrics: InterventionMetrics) -> None:
    table = Table(title="Intervention Results")
    
    table.add_column("Metric", style="", no_wrap=True)
    table.add_column("Intervention", justify="right", style="")
    table.add_column("Control", justify="right", style="")
    
    table.add_row(
        "Logit Diff",
        f"{intervened_metrics.logit_diff:.4f}",
        f"{control_metrics.logit_diff:.4f}"
    )
    table.add_row(
        "Prob Diff",
        f"{intervened_metrics.prob_diff:.4f}",
        f"{control_metrics.prob_diff:.4f}"
    )
    table.add_row(
        "Clean Accuracy",
        f"{intervened_metrics.clean_accuracy:.2%}",
        f"{control_metrics.clean_accuracy:.2%}"
    )
    table.add_row(
        "Corrupted Accuracy",
        f"{intervened_metrics.corrupted_accuracy:.2%}",
        f"{control_metrics.corrupted_accuracy:.2%}"
    )
    table.add_row(
        "Accuracy Diff",
        f"{intervened_metrics.accuracy_diff:.2%}",
        f"{control_metrics.accuracy_diff:.2%}"
    )
    table.add_row(
        "Below 1 Percent Original",
        f"{intervened_metrics.below_1_percent:.2%}",
        f"{control_metrics.below_1_percent:.2%}"
    )
    table.add_row(
        "Below 5 Percent",
        f"{intervened_metrics.below_5_percent:.2%}",
        f"{control_metrics.below_5_percent:.2%}"
    )
    table.add_row(
        "Below 10 Percent",
        f"{intervened_metrics.below_10_percent:.2%}",
        f"{control_metrics.below_10_percent:.2%}"
    )
    rprint(table)

if __name__ == "__main__":
    # Hack: messed up .pkl b/c I put dataclass definition in same pickling script
    sys.modules['__main__'].DecisionTreeResults = DecisionTreeResults

    device = "cuda" if t.cuda.is_available() else "cpu"

    model = load_model(device=device)
    data = load_data(device=device)

    intervention_query = [
        Condition(feature_name='C0 blank', operator='>', threshold=-1),
        Condition(feature_name='D1 mine-theirs', operator='<=', threshold=1),
        Condition(feature_name='E2 mine-theirs', operator='>', threshold=-1),
    ]

    dt_queries = [
        [Condition(feature_name='C0 blank', operator='>', threshold=-1), Condition(feature_name='D1 mine-theirs', operator='<=', threshold=1)],
    ]

    control_query = [
        Condition(feature_name='C0 blank', operator='>', threshold=-1),
        Condition(feature_name='C1 mine-theirs', operator='<=', threshold=1),
        Condition(feature_name='C2 mine-theirs', operator='>', threshold=-1),
    ] 

    intervention_positions_encoded, intervention_positions_decoded = get_filtered_positions(data, intervention_query, control_query, intervention=True)
    control_positions_encoded, control_positions_decoded = get_filtered_positions(data, intervention_query, control_query, intervention=False)
    #sanity_check(intervention_positions_decoded, control_positions_decoded)

    rprint(f"\n[bold]Number of intervention positions:[/bold] {len(intervention_positions_encoded)}")
    rprint(f"[bold]Number of control positions:[/bold] {len(control_positions_encoded)}")

    intervened_metrics = intervene(
        model=model,
        positions=intervention_positions_encoded,
        dt_queries=dt_queries,
        query=intervention_query,
        # dla_positions=intervention_positions_encoded,
        # dla=True,
        # k=k,
        device=device,
    )
    control_metrics = intervene(
        model=model,
        positions=control_positions_encoded,
        dt_queries=dt_queries,
        query=intervention_query,
        # dla_positions=intervention_positions_encoded,
        # dla=True,
        # k=k,
    )

    print_table(intervened_metrics, control_metrics)