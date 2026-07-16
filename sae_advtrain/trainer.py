"""Trainer: composes the base recipe (madry/trades) x alignment (none/raw/sae)
x drift-control mode (frozen/frozen_residual/joint/periodic_refresh) into a
single training loop, entirely driven by ExperimentConfig. No method-specific
code paths exist outside of this file and drift_control.py -- every
combination discussed in the design conversation is one config away.
"""
from __future__ import annotations

import copy
import os
from typing import Optional

import torch
import torch.nn.functional as F

from . import attacks, evaluate, losses, metrics
from .config import ExperimentConfig, save_config
from .drift_control import SAEController
from .hooks import ActivationExtractor
from .models import build_model, resolve_feature_layer
from .sae import build_sae
from .utils import (CSVLogger, MeterBag, Timer, build_lr_scheduler, count_parameters,
                     load_checkpoint, resolve_device, save_checkpoint, set_seed)


class Trainer:
    def __init__(self, cfg: ExperimentConfig, train_loader, val_loader):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = resolve_device(cfg.device)
        set_seed(cfg.seed)

        # -- backbone -----------------------------------------------------
        self.model = build_model(cfg.model.arch, cfg.model.num_classes).to(self.device)
        if cfg.resume_from:
            ckpt = load_checkpoint(cfg.resume_from, map_location=str(self.device))
            self.model.load_state_dict(ckpt["model"])
            print(f"Resumed backbone weights from {cfg.resume_from}")

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=cfg.optim.lr, momentum=cfg.optim.momentum,
            weight_decay=cfg.optim.weight_decay,
        )
        self.lr_step = build_lr_scheduler(self.optimizer, cfg.optim, steps_per_epoch=len(train_loader))

        # -- feature hook / SAE / drift control ----------------------------
        self.needs_features = cfg.align.enabled
        self.feature_layer_name: Optional[str] = None
        self.extractor: Optional[ActivationExtractor] = None
        self.sae = None
        self.sae_controller: Optional[SAEController] = None
        self.initial_backbone = None
        self.initial_extractor: Optional[ActivationExtractor] = None

        if self.needs_features:
            self.feature_layer_name, feat_width = resolve_feature_layer(self.model, cfg.model.feature_layer)
            self.extractor = ActivationExtractor(self.model, self.feature_layer_name, reduce=cfg.sae.reduce)

            if cfg.align.space == "sae":
                self.sae = build_sae(cfg.sae, d_in=feat_width).to(self.device)
                if not cfg.sae.pretrained_path:
                    raise ValueError(
                        "align.enabled=True with align.space='sae' requires sae.pretrained_path "
                        "(see scripts/pretrain_sae.py)."
                    )
                sae_ckpt = load_checkpoint(cfg.sae.pretrained_path, map_location=str(self.device))
                self.sae.load_state_dict(sae_ckpt["sae"])
                self.sae_controller = SAEController(self.sae, cfg.drift)
                print(f"Loaded pretrained SAE from {cfg.sae.pretrained_path}, drift mode = {cfg.drift.mode}")

                if cfg.drift.residual_kind == "delta" and cfg.drift.mode in ("frozen_residual", "periodic_refresh"):
                    self._snapshot_initial_backbone()
            elif cfg.align.space == "raw":
                pass  # alignment_loss operates directly on extractor output, no SAE needed
            else:
                raise ValueError(f"Unknown align.space '{cfg.align.space}'")

        # -- bookkeeping ----------------------------------------------------
        os.makedirs(cfg.out_dir, exist_ok=True)
        save_config(cfg, os.path.join(cfg.out_dir, "config.yaml"))
        self.csv_logger = CSVLogger(os.path.join(cfg.out_dir, "train_log.csv"))
        self.eval_csv_logger = CSVLogger(os.path.join(cfg.out_dir, "eval_log.csv"))
        self.ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_robust_acc = -1.0

        print(f"Backbone params: {count_parameters(self.model):,}")
        if self.sae is not None:
            print(f"SAE params: {count_parameters(self.sae):,} (dict_size={cfg.sae.dict_size}, "
                  f"kind={cfg.sae.kind}, layer={self.feature_layer_name}, reduce={cfg.sae.reduce})")

    # -----------------------------------------------------------------------
    # setup helpers
    # -----------------------------------------------------------------------

    def _snapshot_initial_backbone(self):
        self.initial_backbone = copy.deepcopy(self.model).to(self.device)
        self.initial_backbone.eval()
        for p in self.initial_backbone.parameters():
            p.requires_grad_(False)
        self.initial_extractor = ActivationExtractor(
            self.initial_backbone, self.feature_layer_name, reduce=self.cfg.sae.reduce
        )
        print("Snapshot of the initial (pre-adversarial-training) backbone taken for "
              "drift.residual_kind='delta'.")

    # -----------------------------------------------------------------------
    # adversarial example generation
    # -----------------------------------------------------------------------

    def _build_inner_loss_fn(self, x, y, base_kind, clean_logits=None):
        cfg = self.cfg
        base_fn = (attacks.make_ce_loss_fn(self.model, y) if base_kind == "ce"
                   else attacks.make_trades_kl_loss_fn(self.model, clean_logits))

        if not (cfg.align.enabled and cfg.align.inner_max_uses_alignment):
            return base_fn

        # EXPERIMENTAL ablation: also reward moving away from the clean
        # representation while crafting the adversarial example. Off by
        # default -- see AlignConfig / README for the rationale.
        with torch.no_grad():
            self.model(x)
            feat_clean = self.extractor.pop()
            s_clean = self.sae_controller.encode(feat_clean) if cfg.align.space == "sae" else feat_clean

        def _fn(x_adv):
            base_loss = base_fn(x_adv)
            feat_adv = self.extractor.pop()
            s_adv = self.sae_controller.encode(feat_adv) if cfg.align.space == "sae" else feat_adv
            align = losses.alignment_loss(s_clean, s_adv, kind=cfg.align.kind, stop_grad_clean=True)
            return base_loss + cfg.align.inner_max_align_weight * align

        return _fn

    def _generate_adversarial(self, x, y):
        cfg = self.cfg
        if cfg.base == "clean":
            return x
        with attacks.eval_mode_for_attack(self.model):
            if cfg.base == "madry":
                loss_fn = self._build_inner_loss_fn(x, y, base_kind="ce")
            elif cfg.base == "trades":
                with torch.no_grad():
                    clean_logits = self.model(x)
                loss_fn = self._build_inner_loss_fn(x, y, base_kind="kl", clean_logits=clean_logits)
            else:
                raise ValueError(f"Unknown base '{cfg.base}'")

            x_adv = attacks.pgd_attack(
                x, loss_fn, epsilon=cfg.attack.epsilon, alpha=cfg.attack.alpha,
                steps=cfg.attack.steps, norm=cfg.attack.norm, random_start=cfg.attack.random_start,
            )
        return x_adv

    # -----------------------------------------------------------------------
    # one optimization step
    # -----------------------------------------------------------------------

    def _step(self, x, y) -> dict:
        cfg = self.cfg
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        x_adv = self._generate_adversarial(x, y)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logits_clean = self.model(x)
        feat_clean = self.extractor.pop() if self.needs_features else None

        logits_adv = self.model(x_adv)
        feat_adv = self.extractor.pop() if self.needs_features else None

        logs: dict = {}
        total_loss = torch.zeros((), device=self.device)

        if cfg.base == "madry":
            ce_adv = F.cross_entropy(logits_adv, y)
            total_loss = total_loss + ce_adv
            logs["ce_adv"] = ce_adv.item()
        elif cfg.base == "clean":
            ce = F.cross_entropy(logits_adv, y)  # logits_adv == logits_clean here, x_adv == x
            total_loss = total_loss + ce
            logs["ce_clean"] = ce.item()
        elif cfg.base == "trades":
            ce_clean = F.cross_entropy(logits_clean, y)
            kl = losses.trades_kl(logits_adv, logits_clean)
            total_loss = total_loss + ce_clean + cfg.trades.beta * kl
            logs["ce_clean"] = ce_clean.item()
            logs["trades_kl"] = kl.item()
        else:
            raise ValueError(f"Unknown base '{cfg.base}'")

        if cfg.align.enabled:
            if cfg.align.space == "sae":
                s_clean = self.sae_controller.encode(feat_clean)
                s_adv = self.sae_controller.encode(feat_adv)
                align_loss = losses.alignment_loss(
                    s_clean, s_adv, kind=cfg.align.kind, stop_grad_clean=cfg.align.stop_grad_clean
                )

                h_clean_initial = None
                if self.initial_backbone is not None:
                    with torch.no_grad():
                        self.initial_backbone(x)
                        h_clean_initial = self.initial_extractor.pop()

                aux = self.sae_controller.pre_backward_aux_losses(feat_clean, h_clean_initial)
                for k, v in aux.items():
                    logs[k] = v.item() if torch.is_tensor(v) else v
                    if k == "residual_loss":
                        total_loss = total_loss + v
            else:  # raw
                align_loss = losses.alignment_loss(
                    feat_clean, feat_adv, kind=cfg.align.kind, stop_grad_clean=cfg.align.stop_grad_clean
                )

            total_loss = total_loss + cfg.align.weight * align_loss
            logs["align_loss"] = align_loss.item()

        total_loss.backward()
        self.optimizer.step()

        if cfg.align.enabled and cfg.align.space == "sae":
            # Must run AFTER the backbone's optimizer.step(): 'joint' mode
            # mutates the SAE's own parameters in place here, and doing that
            # before total_loss.backward() has finished using those same
            # parameters (e.g. inside align_loss) corrupts the backward pass.
            post_logs = self.sae_controller.post_backward_update(feat_clean.detach())
            logs.update(post_logs)

            diag = self.sae_controller.diagnostics(feat_clean.detach())
            logs.update(diag)
            if diag["sae_fvu"] > cfg.drift.fvu_alarm_threshold:
                logs["sae_stale_warning"] = 1.0

        logs["clean_acc"] = metrics.accuracy(logits_clean.detach(), y)
        logs["adv_acc"] = metrics.accuracy(logits_adv.detach(), y)
        logs["total_loss"] = total_loss.item()
        return logs

    # -----------------------------------------------------------------------
    # SAE periodic refresh
    # -----------------------------------------------------------------------

    def _refresh_sae_if_needed(self, epoch: int):
        cfg = self.cfg
        if not (cfg.align.enabled and cfg.align.space == "sae" and cfg.drift.mode == "periodic_refresh"):
            return
        if (epoch + 1) % cfg.drift.refresh_every != 0:
            return

        def batches():
            it = iter(self.train_loader)
            for _ in range(cfg.drift.refresh_steps):
                try:
                    x, _y = next(it)
                except StopIteration:
                    it = iter(self.train_loader)
                    x, _y = next(it)
                x = x.to(self.device)
                was_training = self.model.training
                self.model.eval()
                with torch.no_grad():
                    self.model(x)
                    h = self.extractor.pop()
                self.model.train(was_training)
                yield h

        mean_loss = self.sae_controller.refresh(batches())
        print(f"[epoch {epoch}] periodic SAE refresh done, mean reconstruction loss = {mean_loss:.4f}")

    # -----------------------------------------------------------------------
    # epoch / full training loops
    # -----------------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        bag = MeterBag()
        timer = Timer()
        for step, (x, y) in enumerate(self.train_loader):
            lr = self.lr_step(epoch, step)
            logs = self._step(x, y)
            logs["lr"] = lr
            bag.update(n=x.size(0), **{k: v for k, v in logs.items() if k != "sae_stale_warning"})

            if step % self.cfg.log_every == 0:
                avgs = bag.averages()
                msg = " ".join(f"{k}={v:.4f}" for k, v in avgs.items())
                print(f"epoch {epoch} step {step}/{len(self.train_loader)} {msg}")
                self.csv_logger.log({"epoch": epoch, "step": step, **avgs})

        self._refresh_sae_if_needed(epoch)
        print(f"epoch {epoch} done in {timer.elapsed():.1f}s")
        return bag.averages()

    def evaluate_quick(self, epoch: int) -> dict:
        """Cheap PGD-10 evaluation on the small held-out val slice, used to
        track progress during training and to pick the best checkpoint. Use
        scripts/evaluate_checkpoint.py for the full evaluation suite
        (AutoAttack, step sweep, representation diagnostics) at the end."""
        clean_acc = evaluate.evaluate_clean(self.model, self.val_loader, self.device)
        robust_acc = evaluate.evaluate_pgd(
            self.model, self.val_loader, epsilon=self.cfg.attack.epsilon, alpha=self.cfg.attack.alpha,
            steps=10, device=self.device, norm=self.cfg.attack.norm,
        )
        row = {"epoch": epoch, "val_clean_acc": clean_acc, "val_robust_acc_pgd10": robust_acc}

        if self.cfg.align.enabled and self.cfg.align.space == "sae":
            diag = evaluate.compute_representation_diagnostics(
                self.model, self.extractor, self.val_loader, self.device, sae=self.sae,
                initial_model=self.initial_backbone, initial_extractor=self.initial_extractor,
            )
            row.update(diag)

        print(f"[eval] epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))
        self.eval_csv_logger.log(row)
        return row

    def save(self, epoch: int, tag: str):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
        }
        if self.sae is not None:
            state["sae"] = self.sae.state_dict()
        path = os.path.join(self.ckpt_dir, f"{tag}.pt")
        save_checkpoint(path, **state)
        return path

    def train(self):
        cfg = self.cfg
        for epoch in range(cfg.optim.epochs):
            self.train_one_epoch(epoch)

            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.optim.epochs - 1:
                row = self.evaluate_quick(epoch)
                if row["val_robust_acc_pgd10"] > self.best_robust_acc:
                    self.best_robust_acc = row["val_robust_acc_pgd10"]
                    self.save(epoch, "best")

            if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.optim.epochs - 1:
                self.save(epoch, "last")

        return self.save(cfg.optim.epochs - 1, "final")
