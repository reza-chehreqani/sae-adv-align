"""Full evaluation suite for a trained checkpoint.

Reports, in order of increasing attack strength / diagnostic depth:
  1. clean accuracy
  2. PGD-20 accuracy (single restart) -- fast, standard "during-development" number
  3. PGD-50 accuracy with 5 random restarts -- a stronger, still-white-box number
  4. a PGD step-count sweep (10/20/50/100) -- if accuracy keeps dropping well past
     what plateaus for the baseline, treat the result with suspicion (possible
     obfuscated gradients, see Athalye et al. 2018)
  5. AutoAttack (if `pip install autoattack` has been done) -- the field-standard
     check, mixes white-box and gradient-free/black-box attacks
  6. for align.space=="sae" checkpoints: SAE FVU / dead-feature-fraction on the
     CURRENT backbone's test-set activations, i.e. "is the SAE (still) valid"

Always run this (not just quick in-training PGD-10 numbers) before drawing any
conclusion about which drift-control strategy "won".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from sae_advtrain.config import load_config
from sae_advtrain.data import build_dataloaders
from sae_advtrain.evaluate import (
    compute_representation_diagnostics,
    evaluate_autoattack,
    evaluate_clean,
    evaluate_pgd,
    evaluate_step_sweep,
)
from sae_advtrain.hooks import ActivationExtractor
from sae_advtrain.models import build_model, resolve_feature_layer
from sae_advtrain.sae import build_sae
from sae_advtrain.utils import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt file saved by scripts/train.py")
    parser.add_argument("--config", default=None,
                         help="Override config; defaults to the one embedded in the checkpoint.")
    parser.add_argument("--n_autoattack", type=int, default=1000,
                         help="Number of test examples to run AutoAttack on (it is slow).")
    parser.add_argument("--step_sweep", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--pgd50_restarts", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip_autoattack", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(args.config) if args.config else ckpt.get("cfg")
    if cfg is None:
        raise ValueError("No config found in checkpoint and none provided via --config")

    device = resolve_device(args.device)
    _train_loader, _val_loader, test_loader, num_classes = build_dataloaders(cfg.data)

    model = build_model(cfg.model.arch, num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = {"checkpoint": args.checkpoint, "name": cfg.name, "base": cfg.base,
               "align_enabled": cfg.align.enabled,
               "align_space": cfg.align.space if cfg.align.enabled else None,
               "drift_mode": cfg.drift.mode if (cfg.align.enabled and cfg.align.space == "sae") else None}

    results["clean_acc"] = evaluate_clean(model, test_loader, device)
    print(f"Clean accuracy:                {results['clean_acc']:.4f}")

    results["pgd20_acc"] = evaluate_pgd(
        model, test_loader, epsilon=cfg.attack.epsilon, alpha=cfg.attack.alpha,
        steps=20, device=device, norm=cfg.attack.norm,
    )
    print(f"PGD-20 accuracy:                {results['pgd20_acc']:.4f}")

    results["pgd50_multirestart_acc"] = evaluate_pgd(
        model, test_loader, epsilon=cfg.attack.epsilon, alpha=cfg.attack.alpha,
        steps=50, device=device, norm=cfg.attack.norm, restarts=args.pgd50_restarts,
    )
    print(f"PGD-50 x{args.pgd50_restarts} restarts accuracy: {results['pgd50_multirestart_acc']:.4f}")

    step_sweep = evaluate_step_sweep(
        model, test_loader, epsilon=cfg.attack.epsilon, alpha=cfg.attack.alpha,
        device=device, steps_list=args.step_sweep,
    )
    results["step_sweep"] = step_sweep
    print("Step-count sweep (a large continued drop suggests possible gradient masking):")
    for steps, acc in step_sweep.items():
        print(f"  PGD-{steps:<4d} {acc:.4f}")

    if not args.skip_autoattack:
        aa_acc = evaluate_autoattack(
            model, test_loader, epsilon=cfg.attack.epsilon, device=device, n_examples=args.n_autoattack,
        )
        if aa_acc is not None:
            results["autoattack_acc"] = aa_acc
            print(f"AutoAttack accuracy ({args.n_autoattack} examples): {aa_acc:.4f}")

    if cfg.align.enabled and cfg.align.space == "sae":
        layer_name, feat_width = resolve_feature_layer(model, cfg.model.feature_layer)
        extractor = ActivationExtractor(model, layer_name, reduce=cfg.sae.reduce)
        sae = build_sae(cfg.sae, d_in=feat_width).to(device)
        if "sae" in ckpt:
            sae.load_state_dict(ckpt["sae"])
        else:
            print("Checkpoint has no SAE state; loading from cfg.sae.pretrained_path instead.")
            sae_ckpt = torch.load(cfg.sae.pretrained_path, map_location=str(device), weights_only=False)
            sae.load_state_dict(sae_ckpt["sae"])

        diag = compute_representation_diagnostics(model, extractor, test_loader, device, sae=sae, max_batches=20)
        results.update(diag)
        print(f"SAE FVU on current backbone (test set):   {diag['sae_fvu']:.4f}  "
              f"(>{cfg.drift.fvu_alarm_threshold} means the SAE is stale)")
        print(f"SAE dead-feature fraction:                {diag['sae_dead_frac']:.4f}")

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    out_path = os.path.join(run_dir, f"eval_results_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
