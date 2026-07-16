"""Loss building blocks. Kept as small pure functions so trainer.py can
compose them differently per `method` / `drift.mode` without duplicating
math anywhere.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Clean/adversarial representation alignment (the core of the proposed method,
# and also used by the feature_align_raw ablation baseline with align.space="raw")
# --------------------------------------------------------------------------


def alignment_loss(feat_clean: torch.Tensor, feat_adv: torch.Tensor, kind: str = "l2",
                    stop_grad_clean: bool = True) -> torch.Tensor:
    target = feat_clean.detach() if stop_grad_clean else feat_clean

    if kind == "l2":
        return F.mse_loss(feat_adv, target)
    elif kind == "l1":
        return F.l1_loss(feat_adv, target)
    elif kind == "cosine":
        cos = F.cosine_similarity(feat_adv, target, dim=-1, eps=1e-8)
        return (1.0 - cos).mean()
    elif kind == "feature_add":
        # SAE-FT-inspired: only penalize latent mass the adversarial example
        # activates that the clean example did not use at all. Intended for
        # non-negative sparse codes (SAE latents); more surgical than a
        # blanket distance because it leaves legitimate shared/robust
        # features completely unconstrained and only discourages the
        # adversary from recruiting *new* directions.
        clean_active = (target > 0).float()
        new_mass = feat_adv * (1.0 - clean_active)
        return new_mass.clamp_min(0).sum(dim=-1).mean()
    else:
        raise ValueError(f"Unknown alignment kind '{kind}'")


# --------------------------------------------------------------------------
# SAE training / validity losses
# --------------------------------------------------------------------------


def sae_reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def residual_absolute_loss(sae, h_clean: torch.Tensor) -> torch.Tensor:
    """Keeps the CURRENT backbone's clean-input activations reconstructable
    by the (possibly frozen) SAE. Simplest, reference-free drift control:
    no snapshot of the pre-adversarial-training backbone is required."""
    recon, _ = sae(h_clean)
    return F.mse_loss(recon, h_clean)


def residual_delta_loss(sae, h_clean_current: torch.Tensor, h_clean_initial: torch.Tensor) -> torch.Tensor:
    """SAE-FT-inspired variant: instead of constraining the absolute
    activation, constrain the *drift* (current backbone vs. a frozen
    snapshot of the backbone taken before adversarial training started) to
    be expressible through the frozen SAE's decoder. More permissive than
    residual_absolute_loss -- it allows the represented content to change a
    lot, as long as the change decomposes into the known dictionary rather
    than introducing entirely new, uninterpreted directions.
    Requires trainer.py to keep a frozen copy of the initial backbone around
    (see Trainer._maybe_snapshot_initial_backbone)."""
    delta = h_clean_current - h_clean_initial.detach()
    recon_delta, _ = sae(delta)
    return F.mse_loss(recon_delta, delta)


def sae_dead_feature_penalty(codes: torch.Tensor, target_frac_active: float = 0.0) -> torch.Tensor:
    """Optional: not used by default, provided for experimentation. Encourages
    a minimum firing rate per latent within a batch to counter dead features."""
    frac_active = (codes.abs() > 0).float().mean(dim=0)
    return F.relu(target_frac_active - frac_active).mean()


# --------------------------------------------------------------------------
# TRADES
# --------------------------------------------------------------------------


def trades_kl(logits_adv: torch.Tensor, logits_clean: torch.Tensor) -> torch.Tensor:
    """KL(softmax(adv) || softmax(clean)), matching the official TRADES
    implementation's outer-loss term (gradient is allowed to flow through
    both logits_adv and logits_clean, i.e. logits_clean is NOT detached
    here -- this is intentional and matches Zhang et al. 2019's released
    code, not an oversight)."""
    log_probs_adv = F.log_softmax(logits_adv, dim=1)
    probs_clean = F.softmax(logits_clean, dim=1)
    return F.kl_div(log_probs_adv, probs_clean, reduction="batchmean")
