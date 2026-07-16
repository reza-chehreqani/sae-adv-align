"""Pretrain a sparse autoencoder on a frozen backbone's clean activations.

This has to run *before* any align.space=="sae" experiment: those configs
point at sae.pretrained_path, which is exactly the checkpoint this script
produces.

Typical workflow:
    1. python scripts/train.py     --config configs/cifar10_clean_pretrain.yaml
       (produces runs/clean_pretrain/checkpoints/final.pt -- a plain,
       non-adversarially-trained classifier)
    2. python scripts/pretrain_sae.py --config configs/cifar10_sae_pretrain.yaml \
           --set resume_from=runs/clean_pretrain/checkpoints/final.pt
       (produces runs/sae_pretrain/sae.pt)
    3. python scripts/train.py --config configs/cifar10_sae_align_frozen_residual.yaml \
           --set resume_from=runs/clean_pretrain/checkpoints/final.pt \
           --set sae.pretrained_path=runs/sae_pretrain/sae.pt

Reuses ExperimentConfig for convenience (data/model/sae/optim sections);
`resume_from` here means "backbone to extract activations from", and the
backbone is always frozen -- this script only ever trains the SAE.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from sae_advtrain.cli import parse_config_args
from sae_advtrain.data import build_dataloaders
from sae_advtrain.hooks import ActivationExtractor
from sae_advtrain.losses import sae_reconstruction_loss
from sae_advtrain.models import build_model, resolve_feature_layer
from sae_advtrain.sae import L1SAE, build_sae, dead_feature_fraction, fraction_of_variance_unexplained
from sae_advtrain.utils import Timer, resolve_device, save_checkpoint, set_seed


def main():
    cfg = parse_config_args("Pretrain a sparse autoencoder on frozen backbone activations.")
    if not cfg.resume_from:
        raise ValueError("Set resume_from to the backbone checkpoint to extract activations from.")

    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    train_loader, val_loader, _test_loader, num_classes = build_dataloaders(cfg.data)

    model = build_model(cfg.model.arch, num_classes).to(device)
    ckpt = torch.load(cfg.resume_from, map_location=str(device), weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    layer_name, feat_width = resolve_feature_layer(model, cfg.model.feature_layer)
    extractor = ActivationExtractor(model, layer_name, reduce=cfg.sae.reduce)
    print(f"Hooking '{layer_name}' (width={feat_width}, reduce={cfg.sae.reduce})")

    sae = build_sae(cfg.sae, d_in=feat_width).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.optim.lr)

    os.makedirs(cfg.out_dir, exist_ok=True)
    step = 0
    for epoch in range(cfg.optim.epochs):
        timer = Timer()
        running_loss, running_fvu, running_dead, n = 0.0, 0.0, 0.0, 0
        for x, _y in train_loader:
            x = x.to(device)
            with torch.no_grad():
                model(x)
                h = extractor.pop()

            optimizer.zero_grad(set_to_none=True)
            recon, codes = sae(h)
            loss = sae_reconstruction_loss(h, recon)
            if isinstance(sae, L1SAE):
                loss = loss + sae.sparsity_penalty(codes)
            loss.backward()
            optimizer.step()
            sae.renormalize_decoder_()

            with torch.no_grad():
                fvu = fraction_of_variance_unexplained(h, recon)
                dead = dead_feature_fraction(codes)
            running_loss += loss.item()
            running_fvu += fvu
            running_dead += dead
            n += 1
            step += 1

            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step} loss={loss.item():.4f} fvu={fvu:.4f} dead_frac={dead:.4f}")

        print(f"epoch {epoch} done in {timer.elapsed():.1f}s | mean loss={running_loss/n:.4f} "
              f"mean fvu={running_fvu/n:.4f} mean dead_frac={running_dead/n:.4f}")

    # final validation-set FVU, the number you should sanity-check before trusting the SAE
    val_fvu, val_n = 0.0, 0
    with torch.no_grad():
        for x, _y in val_loader:
            x = x.to(device)
            model(x)
            h = extractor.pop()
            recon, _codes = sae(h)
            val_fvu += fraction_of_variance_unexplained(h, recon) * x.size(0)
            val_n += x.size(0)
    print(f"Final held-out val FVU: {val_fvu / val_n:.4f} (0=perfect, 1=as good as predicting the mean)")

    out_path = os.path.join(cfg.out_dir, "sae.pt")
    save_checkpoint(
        out_path,
        sae=sae.state_dict(),
        cfg=cfg,
        layer_name=layer_name,
        feat_width=feat_width,
        val_fvu=val_fvu / val_n,
    )
    print(f"Saved pretrained SAE to {out_path}")


if __name__ == "__main__":
    main()
