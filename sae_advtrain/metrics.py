"""Diagnostics used to monitor training health and to report the same kind
of representation-stability numbers the literature review leaned on (FVU is
in sae.py; CKA lives here since it compares two different tensors rather
than an SAE and its own reconstruction)."""
from __future__ import annotations

import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear Centered Kernel Alignment between two (N, D) feature matrices
    with the same N (same samples). 1.0 = identical representational
    geometry, 0.0 = unrelated. Used to track how far the backbone's
    representation at the SAE's layer has drifted from where it started
    (see README's discussion of Do-SAEs-transfer-under-fine-tuning)."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.t() @ y
    numerator = (xty ** 2).sum()
    xtx_norm = torch.sqrt((x.t() @ x).pow(2).sum())
    yty_norm = torch.sqrt((y.t() @ y).pow(2).sum())
    denom = (xtx_norm * yty_norm).clamp_min(1e-12)
    return (numerator / denom).item()


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()
