"""Main training entry point.

Every experiment in this project -- the clean-pretraining phase, both AT
baselines, the raw-feature-alignment ablation, and the proposed SAE-latent
alignment method under any of the four drift-control strategies -- is
launched through this one script. The config file (plus optional --set
overrides) fully determines behavior; see configs/*.yaml for ready-made
starting points and README.md for what each one tests.

Usage:
    python scripts/train.py --config configs/cifar10_baseline_madry.yaml
    python scripts/train.py --config configs/cifar10_sae_align_frozen_residual.yaml \
        --set resume_from=runs/clean_pretrain/checkpoints/final.pt \
        --set sae.pretrained_path=runs/sae_pretrain/sae.pt
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sae_advtrain.cli import parse_config_args
from sae_advtrain.data import build_dataloaders
from sae_advtrain.trainer import Trainer


def main():
    cfg = parse_config_args(
        "Train a classifier with clean / Madry-PGD / TRADES adversarial training, "
        "optionally with raw-activation or SAE-latent representation alignment."
    )
    train_loader, val_loader, _test_loader, num_classes = build_dataloaders(cfg.data)
    cfg.model.num_classes = num_classes

    print(f"=== {cfg.name} === base={cfg.base} align.enabled={cfg.align.enabled} "
          f"align.space={cfg.align.space if cfg.align.enabled else '-'} "
          f"drift.mode={cfg.drift.mode if (cfg.align.enabled and cfg.align.space == 'sae') else '-'}")

    trainer = Trainer(cfg, train_loader, val_loader)
    final_ckpt = trainer.train()
    print(f"Training complete. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
