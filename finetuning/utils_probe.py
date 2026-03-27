"""
Data loading utilities for probe-intervention finetuning.
"""

import logging

import torch

from finetuning.mingpt.dataset import CharDataset
from utils.circuits_utils import construct_othello_dataset, to_device
from utils.probe_utils import load_fold_probes_and_normalize

logger = logging.getLogger(__name__)


def _stack_probe_dirs(probes, probe_keys, probe_layer):
    """Stack and normalise probe directions for one layer. Internal helper."""
    dirs = []
    for key in probe_keys:
        d = probes[key][probe_layer]                    # (d_model, 8, 8)
        d = d / d.norm(dim=0, keepdim=True)
        dirs.append(d)
    D = torch.stack(dirs, dim=1)                        # (d_model, n_probes, 8, 8)
    D = D.reshape(D.shape[0], -1)                       # (d_model, n_dirs)
    return torch.nan_to_num(D)


def load_probe_dirs(probe_keys, probe_layer, device):
    """
    Load and stack probe directions for the given keys at the given layer.

    Args:
        probe_keys:  list of str, e.g. ["mine", "flipped", "just_played"]
        probe_layer: int — which layer's probe vectors to use (e.g. 5)
        device:      str

    Returns:
        D: (d_model, n_dirs) tensor, column-normalised, NaN-cleaned.
    """
    probes = load_fold_probes_and_normalize(n_layers=8, device=device)
    for key in probe_keys:
        if key not in probes:
            raise KeyError(
                f"Probe key '{key}' not found. Available: {list(probes.keys())}"
            )
    return _stack_probe_dirs(probes, probe_keys, probe_layer)


def load_probe_dirs_per_layer(probe_keys, intervention_layers, device):
    """
    Load per-layer probe directions: layer i uses its own layer-i probe vectors.

    Args:
        probe_keys:          list of str, e.g. ["mine", "flipped", "just_played"]
        intervention_layers: list of int — layers that will receive the intervention
        device:              str

    Returns:
        dict[int, Tensor] — maps each layer index to its (d_model, n_dirs) matrix,
        column-normalised and NaN-cleaned.
    """
    probes = load_fold_probes_and_normalize(n_layers=8, device=device)
    for key in probe_keys:
        if key not in probes:
            raise KeyError(
                f"Probe key '{key}' not found. Available: {list(probes.keys())}"
            )
    return {layer: _stack_probe_dirs(probes, probe_keys, layer)
            for layer in intervention_layers}


def build_datasets(n_train=20_000_000, n_test=0):
    """
    Build train (and optionally test) CharDatasets from the Othello HF dataset.

    Sequences have tokens 1-60, so CharDataset produces vocab_size=61 with
    stoi[k]=k for k=1-60 and stoi[-100]=0 for padding.

    Args:
        n_train: Maximum number of training sequences to load (capped by
                 available data — 792 498 in the train split).
        n_test:  Number of test sequences to load; 0 = no test set.

    Returns:
        (train_dataset, test_dataset) — test_dataset is None when n_test=0.
    """
    train_seqs = construct_othello_dataset(
        custom_functions=[], n_inputs=n_train, split="train", max_str_length=60,
    )["encoded_inputs"]
    train_dataset = CharDataset(train_seqs)
    logger.info(
        "Train dataset: %d sequences, vocab_size=%d, block_size=%d",
        len(train_seqs), train_dataset.vocab_size, train_dataset.block_size,
    )

    test_dataset = None
    if n_test > 0:
        test_seqs = construct_othello_dataset(
            custom_functions=[], n_inputs=n_test, split="test", max_str_length=60,
        )["encoded_inputs"]
        test_dataset = CharDataset(test_seqs)
        logger.info("Test  dataset: %d sequences", len(test_seqs))

    return train_dataset, test_dataset
