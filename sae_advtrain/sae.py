"""Sparse autoencoders operating on backbone activations.

Two flavors, selectable via SAEConfig.kind:

  - TopKSAE (default): hard top-k sparsity, trained with plain MSE
    reconstruction loss -- no sparsity-coefficient tuning needed (Gao et al.
    2024, "Scaling and Evaluating Sparse Autoencoders"). This is the
    lower-friction option and what SAE-FT-style methods build on.
  - L1SAE: classic ReLU + L1-penalty SAE (Anthropic's "Towards
    Monosemanticity" recipe). Included because L1 sparsity is "soft" and
    behaves differently under distribution shift than hard top-k -- worth
    ablating since the project is explicitly about testing ambiguities.

Both use an untied (or optionally tied) encoder/decoder and a pre-encoder
bias that is subtracted before encoding and added back after decoding
(standard trick, centers the data the SAE sees).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSAE(nn.Module):
    d_in: int
    dict_size: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        codes = self.encode(x)
        recon = self.decode(codes)
        return recon, codes

    @torch.no_grad()
    def renormalize_decoder_(self):
        if getattr(self, "normalize_decoder", False) and not self.tied_weights:
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1, eps=1e-8)


class TopKSAE(BaseSAE):
    def __init__(self, d_in: int, dict_size: int, k: int, tied_weights: bool = False,
                 normalize_decoder: bool = True):
        super().__init__()
        self.d_in = d_in
        self.dict_size = dict_size
        self.k = min(k, dict_size)
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder

        W_dec = torch.randn(dict_size, d_in)
        W_dec = F.normalize(W_dec, dim=1)
        self.W_dec = nn.Parameter(W_dec)
        if tied_weights:
            self.register_parameter("W_enc", None)
        else:
            self.W_enc = nn.Parameter(W_dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_pre = nn.Parameter(torch.zeros(d_in))

    @property
    def encoder_weight(self) -> torch.Tensor:
        return self.W_dec.t() if self.tied_weights else self.W_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_pre
        pre_acts = x_centered @ self.encoder_weight + self.b_enc
        pre_acts = F.relu(pre_acts)
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        codes = torch.zeros_like(pre_acts)
        codes.scatter_(-1, topk_idx, topk_vals)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes @ self.W_dec + self.b_pre


class L1SAE(BaseSAE):
    def __init__(self, d_in: int, dict_size: int, l1_coef: float, tied_weights: bool = False,
                 normalize_decoder: bool = True):
        super().__init__()
        self.d_in = d_in
        self.dict_size = dict_size
        self.l1_coef = l1_coef
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder

        W_dec = torch.randn(dict_size, d_in)
        W_dec = F.normalize(W_dec, dim=1)
        self.W_dec = nn.Parameter(W_dec)
        if tied_weights:
            self.register_parameter("W_enc", None)
        else:
            self.W_enc = nn.Parameter(W_dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_pre = nn.Parameter(torch.zeros(d_in))

    @property
    def encoder_weight(self) -> torch.Tensor:
        return self.W_dec.t() if self.tied_weights else self.W_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_pre
        pre_acts = x_centered @ self.encoder_weight + self.b_enc
        return F.relu(pre_acts)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes @ self.W_dec + self.b_pre

    def sparsity_penalty(self, codes: torch.Tensor) -> torch.Tensor:
        return self.l1_coef * codes.abs().sum(dim=-1).mean()


def build_sae(sae_cfg, d_in: int) -> BaseSAE:
    if sae_cfg.kind == "topk":
        return TopKSAE(
            d_in=d_in,
            dict_size=sae_cfg.dict_size,
            k=sae_cfg.k,
            tied_weights=sae_cfg.tied_weights,
            normalize_decoder=sae_cfg.normalize_decoder,
        )
    elif sae_cfg.kind == "l1":
        return L1SAE(
            d_in=d_in,
            dict_size=sae_cfg.dict_size,
            l1_coef=sae_cfg.l1_coef,
            tied_weights=sae_cfg.tied_weights,
            normalize_decoder=sae_cfg.normalize_decoder,
        )
    else:
        raise ValueError(f"Unknown SAE kind '{sae_cfg.kind}'")


def fraction_of_variance_unexplained(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """FVU = ||x - x_hat||^2 / ||x - mean(x)||^2. FVU=0 is perfect
    reconstruction, FVU=1 is as good as predicting the per-batch mean,
    FVU>1 means the SAE is actively worse than that (a sign the SAE is
    stale relative to the activations it is being asked to reconstruct --
    see README.md's discussion of the SAE-drift problem)."""
    with torch.no_grad():
        resid_var = (x - x_hat).pow(2).sum()
        total_var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-8)
        return (resid_var / total_var).item()


def dead_feature_fraction(codes: torch.Tensor) -> float:
    """Fraction of latents that never fired in this batch. High values mean
    the dictionary is under-utilized (common failure mode, especially for
    TopK SAEs with too-small k or too-large dict_size)."""
    with torch.no_grad():
        active = (codes.abs() > 0).any(dim=0)
        return 1.0 - active.float().mean().item()
