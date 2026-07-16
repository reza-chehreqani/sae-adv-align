"""Evaluation utilities.

`evaluate_pgd` always attacks plain cross-entropy w.r.t. the true label,
regardless of which `base` recipe trained the model -- this is the standard
robustness metric reported across the AT literature (even TRADES papers
report PGD/AutoAttack-CE robust accuracy, not a TRADES-specific attack), so
every method in this project is judged by the same yardstick.

`evaluate_step_sweep` and `evaluate_autoattack` exist specifically to guard
against the obfuscated-gradient failure mode flagged for the SAE-alignment
method: if robust accuracy keeps dropping substantially as PGD step count
grows far past convergence, or AutoAttack (which includes gradient-free and
black-box components) finds many more misclassifications than white-box
PGD does, that is a red flag that apparent robustness is partly a gradient
artifact rather than real. Always run these on the SAE-aligned checkpoints,
not just plain PGD-20.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from . import attacks, metrics
from .hooks import ActivationExtractor
from .sae import BaseSAE, dead_feature_fraction, fraction_of_variance_unexplained


@torch.no_grad()
def evaluate_clean(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def evaluate_pgd(model, loader, epsilon, alpha, steps, device, norm="linf", restarts: int = 1) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        still_correct = torch.ones(x.size(0), dtype=torch.bool, device=device)
        for _r in range(restarts):
            loss_fn = attacks.make_ce_loss_fn(model, y)
            x_adv = attacks.pgd_attack(
                x, loss_fn, epsilon=epsilon, alpha=alpha, steps=steps, norm=norm,
                random_start=True,
            )
            with torch.no_grad():
                pred = model(x_adv).argmax(1)
            still_correct &= (pred == y)
        correct += still_correct.sum().item()
        total += y.size(0)
    return correct / total


def evaluate_step_sweep(model, loader, epsilon, alpha, device, steps_list: List[int] = (10, 20, 50, 100),
                         norm: str = "linf", max_batches: Optional[int] = 10) -> Dict[int, float]:
    """Robust accuracy at increasing PGD step counts, on a subset of
    `loader` for speed. A genuinely robust model's accuracy should mostly
    plateau; if it keeps falling sharply as steps grow, suspect obfuscated
    gradients (Athalye et al. 2018) rather than real robustness."""
    results = {}
    for steps in steps_list:
        model.eval()
        correct, total = 0, 0
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            loss_fn = attacks.make_ce_loss_fn(model, y)
            x_adv = attacks.pgd_attack(x, loss_fn, epsilon=epsilon, alpha=alpha, steps=steps, norm=norm,
                                        random_start=True)
            with torch.no_grad():
                pred = model(x_adv).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        results[steps] = correct / total
    return results


def evaluate_autoattack(model, loader, epsilon, device, norm="Linf",
                         n_examples: Optional[int] = 1000) -> Optional[float]:
    """Runs the standard AutoAttack (Croce & Hein 2020) ensemble if the
    `autoattack` package is installed (`pip install autoattack`). This is
    the field-standard "did we actually get robust" check -- it mixes
    white-box (APGD-CE, APGD-T) and black-box/gradient-free (FAB-T, Square)
    attacks, so it is far less susceptible to gradient masking than PGD
    alone. Returns None (and prints instructions) if unavailable."""
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("autoattack is not installed; run `pip install autoattack` to enable this check. Skipping.")
        return None

    model.eval()
    xs, ys, n = [], [], 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        n += x.size(0)
        if n_examples is not None and n >= n_examples:
            break
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    if n_examples is not None:
        x_all, y_all = x_all[:n_examples], y_all[:n_examples]
    x_all, y_all = x_all.to(device), y_all.to(device)

    adversary = AutoAttack(model, norm=norm, eps=epsilon, version="standard", device=device)
    x_adv = adversary.run_standard_evaluation(x_all, y_all, bs=128)
    with torch.no_grad():
        pred = model(x_adv).argmax(1)
    return (pred == y_all).float().mean().item()


@torch.no_grad()
def compute_representation_diagnostics(
    model,
    extractor: ActivationExtractor,
    loader,
    device,
    sae: Optional[BaseSAE] = None,
    initial_model=None,
    initial_extractor: Optional[ActivationExtractor] = None,
    max_batches: int = 10,
) -> Dict[str, float]:
    """FVU / dead-feature-fraction of the (possibly frozen) SAE against the
    CURRENT backbone's clean activations, plus linear CKA between the
    current and initial (pre-adversarial-training) backbone's
    representations at the hooked layer, if `initial_model` is given. This
    is the diagnostic bundle referenced throughout README.md's discussion
    of whether the SAE is still valid."""
    model.eval()
    feats, feats_initial = [], []
    for i, (x, _y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        model(x)
        feats.append(extractor.pop().detach().cpu())
        if initial_model is not None:
            initial_model(x)
            feats_initial.append(initial_extractor.pop().detach().cpu())

    h = torch.cat(feats, dim=0)
    out: Dict[str, float] = {}

    if sae is not None:
        recon, codes = sae(h.to(device))
        out["sae_fvu"] = fraction_of_variance_unexplained(h.to(device), recon)
        out["sae_dead_frac"] = dead_feature_fraction(codes)

    if initial_model is not None:
        h_initial = torch.cat(feats_initial, dim=0)
        out["cka_vs_initial"] = metrics.linear_cka(h, h_initial)

    return out
