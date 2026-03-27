"""
Training loop for GPTWithProbeIntervention.

Loss: CE(logits, targets) + β * KL(p_finetuned || p_original)

ProbeTrainerConfig extends TrainerConfig with kl_weight and freeze_up_to.
ProbeModelTrainer extends Trainer with the combined loss and parameter freezing.
"""

import logging

import torch
import torch.nn.functional as F

from finetuning.mingpt.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


class ProbeModelTrainerConfig(TrainerConfig):
    kl_weight      = 0.1   # β — weight of the KL term; 0 = pure CE
    freeze_up_to   = -1    # freeze blocks 0..N (-1 = freeze nothing)
    ref_model_name = "Baidicoot/Othello-GPT-Transformer-Lens"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ProbeModelTrainer(Trainer):
    """
    Extends Trainer with:
      - a reference model loaded at init for KL computation
      - parameter freezing according to freeze_up_to
      - combined CE + β * KL loss
    """

    def __init__(self, model, train_dataset, test_dataset, config: ProbeModelTrainerConfig):
        # Skip Trainer's DataParallel wrapping — use a single GPU (cuda:0) directly.
        self.model        = model
        self.train_dataset = train_dataset
        self.test_dataset  = test_dataset
        self.config        = config
        self.device        = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model         = self.model.to(self.device)

        # Load reference model (original OthelloGPT, no intervention) for KL term.
        # Use a plain GPTWithProbeIntervention (intervention_layers=[]) instead of
        # HookedTransformer to avoid nnsight graph-retention issues during backward.
        if config.kl_weight > 0:
            from finetuning.model_probe import GPTWithProbeIntervention
            raw = self.model
            dummy_dirs = torch.zeros(raw.gpt_config.n_embd, 1)
            ref = GPTWithProbeIntervention(raw.gpt_config, probe_dirs=dummy_dirs,
                                           intervention_layers=[])
            ref.load_pretrained_from_hf(config.ref_model_name)
            self.ref_model = ref.to(self.device).eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        else:
            self.ref_model = None

        # Freeze early blocks
        if config.freeze_up_to >= 0:
            raw_model = model.module if hasattr(model, "module") else model
            # Freeze embeddings
            for p in raw_model.tok_emb.parameters():
                p.requires_grad_(False)
            raw_model.pos_emb.requires_grad_(False)
            # Freeze blocks 0 .. freeze_up_to
            for i in range(config.freeze_up_to + 1):
                for p in raw_model.blocks[i].parameters():
                    p.requires_grad_(False)
            logger.info("Frozen: embeddings + blocks 0..%d", config.freeze_up_to)

    def _compute_loss(self, logits, targets, x):
        """CE + β * KL against the reference model."""
        # Cross-entropy — CharDataset encodes the -100 padding sentinel as token
        # index 0 (it sorts to the front of the vocabulary), so ignore_index=0.
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0,
        )

        if self.ref_model is None or self.config.kl_weight == 0:
            return ce_loss

        # KL divergence: KL(p_finetuned || p_ref)
        # F.kl_div(input, target) = KL(target || input), so pass log_p_ref as
        # input and p_finetuned as target to get KL(p_finetuned || p_ref).
        with torch.no_grad():
            ref_logits, _ = self.ref_model(x, None)
            ref_logits = ref_logits[:, :, :logits.size(-1)]

        log_p_ft  = F.log_softmax(logits,     dim=-1)   # finetuned (log)
        log_p_ref = F.log_softmax(ref_logits, dim=-1)   # reference (log)
        # KL(p_ft || p_ref) using log_target=True: both inputs are log-probs
        kl        = F.kl_div(log_p_ref, log_p_ft, reduction="batchmean", log_target=True)

        return ce_loss + self.config.kl_weight * kl

    def train(self):
        """Override to use _compute_loss instead of the model's built-in loss."""
        import math
        import numpy as np
        from tqdm import tqdm
        from torch.utils.data.dataloader import DataLoader

        model, config = self.model, self.config
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data   = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data, shuffle=is_train, pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    logits, _ = model(x, None)          # get logits only
                    loss = self._compute_loss(logits, y, x)
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / \
                                       float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: loss {loss.item():.5f}. lr {lr:e}"
                    )

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss   = float("inf")
        self.tokens = 0
        for epoch in range(config.max_epochs):
            run_epoch("train")
            if self.test_dataset is not None:
                test_loss = run_epoch("test")

            if self.config.ckpt_path is not None:
                if self.test_dataset is None:
                    self.save_checkpoint()
                    continue
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_checkpoint()
