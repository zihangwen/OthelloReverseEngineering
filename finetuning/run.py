"""
CLI entry point for finetuning GPTWithProbeIntervention.

Example:
    python run.py \\
        --probe_keys mine flipped just_played \\
        --probe_layer 5 \\
        --max_epochs 10 \\
        --learning_rate 1e-4 \\
        --ckpt_path checkpoints/probe_ft.ckpt
"""

import os
import sys
import logging
import argparse

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import torch

from finetuning.mingpt.model import GPTConfig
from finetuning.model_probe import GPTWithProbeIntervention
from finetuning.trainer_probe import ProbeModelTrainerConfig, ProbeModelTrainer
from finetuning.utils_probe import load_probe_dirs, load_probe_dirs_per_layer, to_device, build_datasets


def run_finetuning(args):
    """Full finetuning pipeline."""
    fmt      = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    datefmt  = "%m/%d/%Y %H:%M:%S"
    handlers = [logging.StreamHandler()]
    if args.ckpt_path:
        log_path = os.path.splitext(os.path.abspath(args.ckpt_path))[0] + ".log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(format=fmt, datefmt=datefmt, level=logging.INFO, handlers=handlers)
    logger = logging.getLogger(__name__)
    if args.ckpt_path:
        logger.info("Logging to %s", log_path)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"

    # 1. Probe directions
    if args.per_layer_probe:
        logger.info("Loading per-layer probe directions: %s", args.probe_keys)
        probe_dirs = load_probe_dirs_per_layer(
            args.probe_keys, args.intervention_layers or [], device
        )
        logger.info("Per-layer probe matrices: %s",
                    {k: tuple(v.shape) for k, v in probe_dirs.items()})
    else:
        logger.info("Loading probe directions: %s at layer %d", args.probe_keys, args.probe_layer)
        probe_dirs = load_probe_dirs(args.probe_keys, args.probe_layer, device)
        logger.info("Probe direction matrix shape: %s", tuple(probe_dirs.shape))

    # 2. Datasets
    logger.info("Loading datasets (n_train=%d, n_test=%d)", args.n_train, args.n_test)
    train_dataset, test_dataset = build_datasets(
        n_train=args.n_train, n_test=args.n_test,
    )

    # 3. Model
    gpt_config = GPTConfig(
        vocab_size  = train_dataset.vocab_size,
        block_size  = train_dataset.block_size,
        n_layer     = 8,
        n_head      = 8,
        n_embd      = 512,
    )
    intervention_layers = getattr(args, "intervention_layers", None)
    logger.info("Building GPTWithProbeIntervention (intervention layers: %s) ...",
                intervention_layers if intervention_layers else "none")
    model = GPTWithProbeIntervention(gpt_config, to_device(probe_dirs, device),
                                     intervention_layers=intervention_layers)
    model.load_pretrained_from_hf(args.hf_model_name)
    model = model.to(device)

    # 4. Train
    if args.ckpt_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.ckpt_path)), exist_ok=True)
    trainer_config = ProbeModelTrainerConfig(
        max_epochs     = args.max_epochs,
        batch_size     = args.batch_size,
        learning_rate  = args.learning_rate,
        weight_decay   = args.weight_decay,
        lr_decay       = args.lr_decay,
        ckpt_path      = args.ckpt_path,
        num_workers    = args.num_workers,
        kl_weight      = args.kl_weight,
        freeze_up_to   = args.freeze_up_to,
        ref_model_name = args.hf_model_name,
    )
    trainer = ProbeModelTrainer(model, train_dataset, test_dataset, trainer_config)
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune OthelloGPT with probe-direction constrained V vectors."
    )

    # Data
    parser.add_argument(
        "--n_train", type=int, default=20_000_000,
        help="Max training sequences to load (capped at ~792k available, default: all).",
    )
    parser.add_argument(
        "--n_test", type=int, default=0,
        help="Test sequences to load; 0 = no test set (default: 0).",
    )

    # Model
    parser.add_argument(
        "--hf_model_name", default="Baidicoot/Othello-GPT-Transformer-Lens",
        help="HuggingFace repo to load pretrained weights from.",
    )

    # Probe directions
    parser.add_argument(
        "--probe_keys", nargs="+", default=["flipped", "just_played", "mine"],
        help="Which probe directions to use for the V-vector subspace "
             "(default: mine flipped just_played).",
    )
    parser.add_argument(
        "--probe_layer", type=int, default=5,
        help="Which layer's trained probe vectors define the subspace (default: 5). "
             "Ignored when --per_layer_probe is set.",
    )
    parser.add_argument(
        "--per_layer_probe", action="store_true",
        help="Use layer-i probe directions for layer-i intervention instead of a "
             "single fixed probe_layer.",
    )

    # Intervention layer selection
    parser.add_argument(
        "--intervention_layers", nargs="*", type=int, default=None,
        help="Layer indices to apply probe intervention (0-indexed). "
             "If omitted or empty, no layers are intervened on. "
             "Example: --intervention_layers 1 2 3 4 5",
    )

    # Training
    parser.add_argument("--max_epochs",    type=int,   default=10)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate (default: 1e-4; lower than pretraining 3e-4).",
    )
    parser.add_argument("--weight_decay",  type=float, default=0.1)
    parser.add_argument(
        "--lr_decay", action="store_true",
        help="Enable cosine learning rate decay with linear warmup.",
    )
    parser.add_argument("--num_workers",   type=int,   default=0)
    parser.add_argument(
        "--kl_weight", type=float, default=0.1,
        help="Weight β for the KL divergence term in the loss (default: 0.1). "
             "Set to 0 for pure cross-entropy.",
    )
    parser.add_argument(
        "--freeze_up_to", type=int, default=-1,
        help="Freeze all blocks with index ≤ this value during training "
             "(default: -1, freeze nothing).",
    )

    # Checkpoint
    parser.add_argument(
        "--ckpt_path", default=None,
        help="File path to save the best checkpoint, e.g. checkpoints/probe_ft.ckpt.",
    )

    # Device
    parser.add_argument(
        "--device", default="cuda",
        help="Device to train on: 'cuda', 'cuda:0', 'cpu', etc. (default: cuda).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_finetuning(args)


if __name__ == "__main__":
    main()