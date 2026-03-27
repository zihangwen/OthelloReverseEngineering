"""
Probe-direction constrained GPT model.

Imports GPTConfig and GPT from model.py (the original mingpt model, unchanged).
Adds three classes that bake the QR-based probe-subspace projection into the
attention forward pass so that every layer's V vector is computed only from
the component of the LN-normalised input that lies in the MINE / FLIPPED /
PLACED probe subspace. K and Q use the full LN-normalised input unchanged.
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from finetuning.mingpt.model import GPT  # original mingpt model; import GPTConfig from model directly
from finetuning.mingpt.model import Block

logger = logging.getLogger(__name__)


class ProbeProjectedAttention(nn.Module):
    """
    Drop-in replacement for CausalSelfAttention.

    K and Q are computed from the full LN-normalised input (x_ln).
    V is computed from x_v, passed in pre-computed by ProbeBlock:
        x_raw  = (x - mean) / std           (raw norm, no LN weight/bias)
        x_proj = x_raw @ Q_valid @ Q_valid.T (project in raw-normalised space)
        x_v    = x_proj * ln_weight + ln_bias (re-apply LN weight/bias)
        v      = W_V(x_v) + b_V

    This matches fold_ln=True in attention_intervention.py where hook_normalized
    captures raw-normalised activations and W_V has absorbed the LN scale.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key        = nn.Linear(config.n_embd, config.n_embd)
        self.query      = nn.Linear(config.n_embd, config.n_embd)
        self.value      = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj       = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                  .view(1, 1, config.block_size, config.block_size),
        )
        self.n_head = config.n_head

    def forward(self, x_ln, x_v, only_last=-1):
        """
        Args:
            x_ln: full LN-normalised hidden state (B, T, d_model) — for K and Q.
            x_v:  probe-projected input (B, T, d_model) — for V.
        Returns:
            (y, att) — same signature as CausalSelfAttention.
        """
        B, T, C = x_ln.size()

        # K and Q from the full LN-normalised input
        k = self.key(x_ln).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x_ln).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # V from the probe-projected input (computed by ProbeBlock)
        v = self.value(x_v).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        if only_last != -1:
            att[:, :, -only_last:, :-only_last] = float('-inf')
        att = F.softmax(att, dim=-1)

        y = self.attn_drop(att) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, att


class ProbeBlock(nn.Module):
    """
    Transformer block that constrains V to the probe subspace.

    Replicates the fold_ln=True projection from attention_intervention.py:
      1. x_raw  = (x − mean) / std              — raw norm (= hook_normalized in fold_ln model)
      2. x_proj = x_raw @ Q_valid @ Q_valid.T   — project in raw-normalised space
      3. x_v    = x_proj * ln1.weight + ln1.bias — re-apply LN scale (≡ folded W_V)
      4. V      = W_V(x_v)

    K and Q use the full LN output (weight/bias applied), unchanged from vanilla GPT.
    Q_valid is the orthonormal basis from QR-factorising the probe direction matrix.
    """

    def __init__(self, config, Q_valid: torch.Tensor):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.attn = ProbeProjectedAttention(config)
        # Orthonormal probe subspace basis — fixed, not a trainable parameter
        self.register_buffer("Q_valid", Q_valid.detach())  # (d_model, k)
        self.mlp  = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, return_att=False):
        x_ln = self.ln1(x)                                          # full LN for K, Q

        # Raw normalisation (no weight/bias) — matches hook_normalized in fold_ln model
        x_raw = F.layer_norm(x, [x.shape[-1]])                      # (x − mean) / std

        # Project in raw-normalised space, then re-apply LN weight/bias
        x_v = (x_raw @ self.Q_valid @ self.Q_valid.T) * self.ln1.weight + self.ln1.bias

        updt, att = self.attn(x_ln, x_v)
        x = x + updt
        x = x + self.mlp(self.ln2(x))
        if return_att:
            return x, att
        else:
            return x


class GPTWithProbeIntervention(GPT):
    """
    Full OthelloGPT whose attention value vectors are constrained to the
    subspace spanned by MINE / FLIPPED / PLACED probe directions.

    The intervention is baked directly into forward() — no post-hoc patching.
    forward() is inherited from GPT unchanged since self.blocks is nn.Sequential.

    Usage:
        gpt_config = GPTConfig(vocab_size=62, block_size=59,
                               n_layer=8, n_head=8, n_embd=512)

        # No intervention (all standard blocks — same as vanilla GPT):
        model = GPTWithProbeIntervention(gpt_config, probe_dirs)

        # Intervene on layers 1-5, leave 0, 6, 7 as standard Block:
        model = GPTWithProbeIntervention(gpt_config, probe_dirs,
                                         intervention_layers=[1, 2, 3, 4, 5])

        model.load_pretrained_from_hf("Baidicoot/Othello-GPT-Transformer-Lens")
    """

    def __init__(self, config, probe_dirs, intervention_layers=None):
        """
        Args:
            config:               GPTConfig
            probe_dirs:           (d_model, n_dirs) tensor — stacked, normalised probe
                                  directions (MINE, FLIPPED, PLACED), NaN-cleaned.
                                  Alternatively a dict[int -> Tensor] for layer-specific dirs.
            intervention_layers:  List of layer indices that should use ProbeBlock.
                                  None / [] (default) means no intervention — all layers
                                  keep the standard Block (useful as a control baseline).
        """
        super().__init__(config)   # builds standard blocks + applies _init_weights
        self.gpt_config = config   # store for downstream use (e.g. building ref model)

        # Which layers get the probe intervention; empty set = no intervention
        intervene_on = set(intervention_layers) if intervention_layers else set()

        # Pre-compute the orthonormal basis (Q_valid) for the probe subspace.
        # Only computed for layers that actually need it.
        if isinstance(probe_dirs, dict):
            Q_valid_per_layer = {
                layer: self._compute_Q_valid(d)
                for layer, d in probe_dirs.items()
                if layer in intervene_on
            }
        else:
            _Q = self._compute_Q_valid(probe_dirs)
            Q_valid_per_layer = {i: _Q for i in intervene_on}

        # Build the block list: ProbeBlock for intervened layers, Block for the rest.
        # nn.Sequential so GPT.forward() works unchanged (x = self.blocks(x)).
        # Weights will be overwritten by load_pretrained_from_hf().
        self.blocks = nn.Sequential(*[
            ProbeBlock(config, Q_valid_per_layer[i]) if i in intervene_on
            else Block(config)
            for i in range(self.n_layer)
        ])

    @staticmethod
    def _compute_Q_valid(probe_dirs: torch.Tensor) -> torch.Tensor:
        """
        QR-factorise probe_dirs (d_model, n_dirs) and return the orthonormal
        columns corresponding to non-degenerate directions.

        Returns Q_valid of shape (d_model, k) where k <= n_dirs.
        """
        Q, R = torch.linalg.qr(probe_dirs)
        valid = R.diag().abs() > 1e-6
        return Q[:, valid]

    def load_pretrained_from_hf(self, hf_model_name: str):
        """
        Download final.pth from hf_model_name and load it directly.
        Keys in final.pth already match this model's state dict, so no
        transformation is needed. strict=False keeps the pre-computed
        Q_valid buffers in ProbeBlock layers.

        """
        from huggingface_hub import hf_hub_download

        logger.info("Downloading %s from HuggingFace ...", hf_model_name)
        path = hf_hub_download(hf_model_name, "final.pth")
        hf_sd = torch.load(path, map_location="cpu", weights_only=True)

        # Keys match directly — no transformation needed.
        # strict=False: Q_valid buffers absent from checkpoint; kept from __init__.
        missing, unexpected = self.load_state_dict(hf_sd, strict=False)
        probe_keys    = [k for k in missing if "Q_valid" in k]
        other_missing = [k for k in missing if "Q_valid" not in k]
        logger.info("Pretrained weights loaded from %s", hf_model_name)
        logger.info("  Q_valid buffers kept (pre-computed): %d", len(probe_keys))
        if other_missing:
            logger.warning("  Unexpected missing keys: %s", other_missing)
        if unexpected:
            logger.warning("  Unexpected extra keys: %s", unexpected)

    # def load_pretrained_from_hf(self, hf_model_name: str):
    #     """
    #     Download hf_model_name from HuggingFace (as a HookedTransformer) and
    #     load its weights into this model.

    #     Written from scratch based on the documented HF → mingpt key mapping.
    #     Uses strict=False because Q_valid buffers in ProbeProjectedAttention
    #     are not present in the HF checkpoint (they are pre-computed).

    #     Key mapping (HookedTransformer → GPTWithProbeIntervention):
    #     ─────────────────────────────────────────────────────────────────────
    #     embed.W_E         [V_hf, D]     → tok_emb.weight   [V, D]
    #     pos_embed.W_pos   [T, D]        → pos_emb           [1, T, D]
    #     blocks.i.ln1.w/b               → blocks.i.ln1.weight/bias
    #     blocks.i.ln2.w/b               → blocks.i.ln2.weight/bias
    #     blocks.i.attn.W_Q [nh, D, dh]  → blocks.i.attn.query.weight [D, D]
    #     blocks.i.attn.b_Q [nh, dh]     → blocks.i.attn.query.bias   [D]
    #     (same pattern for W_K/b_K, W_V/b_V)
    #     blocks.i.attn.W_O [nh, dh, D]  → blocks.i.attn.proj.weight  [D, D]
    #     blocks.i.attn.b_O [D]          → blocks.i.attn.proj.bias    [D]
    #     blocks.i.mlp.W_in [D, 4D]      → blocks.i.mlp.0.weight      [4D, D]
    #     blocks.i.mlp.b_in [4D]         → blocks.i.mlp.0.bias        [4D]
    #     blocks.i.mlp.W_out[4D, D]      → blocks.i.mlp.2.weight      [D, 4D]
    #     blocks.i.mlp.b_out[D]          → blocks.i.mlp.2.bias        [D]
    #     ln_final.w/b                   → ln_f.weight/bias
    #     unembed.W_U       [D, V_hf]    → head.weight       [V, D]
    #     ─────────────────────────────────────────────────────────────────────
    #     """
    #     from transformer_lens import HookedTransformer  # for loading HF checkpoint as a HookedTransformer

    #     logger.info("Downloading %s from HuggingFace ...", hf_model_name)
    #     hf_model = HookedTransformer.from_pretrained(
    #         hf_model_name,
    #         center_writing_weights=False,
    #         center_unembed=False,
    #         fold_ln=False,
    #     )
    #     hf_sd = {k: v.detach().cpu() for k, v in hf_model.state_dict().items()}
    #     del hf_model

    #     # Start from the current state dict (includes pre-computed Q_valid buffers)
    #     sd = self.state_dict()
    #     D  = self.tok_emb.embedding_dim

    #     # Token embedding: vocab sizes match (both 61), direct copy
    #     sd["tok_emb.weight"] = hf_sd["embed.W_E"]

    #     # Positional embedding: [T, D] → [1, T, D]
    #     sd["pos_emb"] = hf_sd["pos_embed.W_pos"].unsqueeze(0)

    #     for i in range(self.n_layer):
    #         # LayerNorms
    #         sd[f"blocks.{i}.ln1.weight"] = hf_sd[f"blocks.{i}.ln1.w"]
    #         sd[f"blocks.{i}.ln1.bias"]   = hf_sd[f"blocks.{i}.ln1.b"]
    #         sd[f"blocks.{i}.ln2.weight"] = hf_sd[f"blocks.{i}.ln2.w"]
    #         sd[f"blocks.{i}.ln2.bias"]   = hf_sd[f"blocks.{i}.ln2.b"]

    #         # Q, K, V projections
    #         # HF: W [nh, D, dh]  →  mingpt linear.weight [D, D]
    #         #   weight = W.permute(0, 2, 1).reshape(D, D)
    #         # HF: b [nh, dh]     →  mingpt linear.bias [D]
    #         #   bias = b.reshape(D)
    #         for proj, wk, bk in [("query", "W_Q", "b_Q"),
    #                               ("key",   "W_K", "b_K"),
    #                               ("value", "W_V", "b_V")]:
    #             W = hf_sd[f"blocks.{i}.attn.{wk}"]           # [nh, D, dh]
    #             sd[f"blocks.{i}.attn.{proj}.weight"] = W.permute(0, 2, 1).reshape(D, D)
    #             b = hf_sd[f"blocks.{i}.attn.{bk}"]           # [nh, dh]
    #             sd[f"blocks.{i}.attn.{proj}.bias"]   = b.reshape(D)

    #         # Output projection
    #         # HF: W_O [nh, dh, D]  →  mingpt proj.weight [D, D]
    #         #   weight = W_O.permute(2, 0, 1).reshape(D, D)
    #         W_O = hf_sd[f"blocks.{i}.attn.W_O"]              # [nh, dh, D]
    #         sd[f"blocks.{i}.attn.proj.weight"] = W_O.permute(2, 0, 1).reshape(D, D)
    #         sd[f"blocks.{i}.attn.proj.bias"]   = hf_sd[f"blocks.{i}.attn.b_O"]

    #         # MLP: HF stores [in, out]; nn.Linear.weight is [out, in] → transpose
    #         sd[f"blocks.{i}.mlp.0.weight"] = hf_sd[f"blocks.{i}.mlp.W_in"].T
    #         sd[f"blocks.{i}.mlp.0.bias"]   = hf_sd[f"blocks.{i}.mlp.b_in"]
    #         sd[f"blocks.{i}.mlp.2.weight"] = hf_sd[f"blocks.{i}.mlp.W_out"].T
    #         sd[f"blocks.{i}.mlp.2.bias"]   = hf_sd[f"blocks.{i}.mlp.b_out"]

    #     # Final LayerNorm
    #     sd["ln_f.weight"] = hf_sd["ln_final.w"]
    #     sd["ln_f.bias"]   = hf_sd["ln_final.b"]

    #     # Unembedding: [D, V_hf] → [V, D], vocab sizes match
    #     sd["head.weight"] = hf_sd["unembed.W_U"].T

    #     # strict=False: Q_valid buffers are absent from the HF checkpoint;
    #     # they remain at their pre-computed QR values from __init__.
    #     missing, unexpected = self.load_state_dict(sd, strict=False)
    #     probe_keys    = [k for k in missing if "Q_valid" in k]
    #     other_missing = [k for k in missing if "Q_valid" not in k]
    #     logger.info("Pretrained weights loaded from %s", hf_model_name)
    #     logger.info("  Q_valid buffers kept (pre-computed): %d", len(probe_keys))
    #     if other_missing:
    #         logger.warning("  Unexpected missing keys: %s", other_missing)
    #     if unexpected:
    #         logger.warning("  Unexpected extra keys: %s", unexpected)

    # # forward() is inherited from GPT unchanged — self.blocks is nn.Sequential
    # # so x = self.blocks(x) works as expected.
