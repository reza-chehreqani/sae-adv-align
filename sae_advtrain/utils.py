"""Small, dependency-free utilities shared across the project."""
from __future__ import annotations

import csv
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class AverageMeter:
    """Tracks a running mean of a scalar."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        if value is None:
            return
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


class MeterBag:
    """A dict of AverageMeters, created lazily on first use."""

    def __init__(self):
        self._meters: Dict[str, AverageMeter] = {}

    def update(self, n: int = 1, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if k not in self._meters:
                self._meters[k] = AverageMeter()
            self._meters[k].update(v, n)

    def averages(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self._meters.items()}

    def reset(self):
        for m in self._meters.values():
            m.reset()


class CSVLogger:
    """Appends dict rows to a CSV, writing the header the first time a new
    key set is seen (extends the header if new columns show up)."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fieldnames = []
        if os.path.exists(path):
            with open(path, "r", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                if header:
                    self._fieldnames = header

    def log(self, row: Dict) -> None:
        row = {**row, "timestamp": time.time()}
        new_keys = [k for k in row.keys() if k not in self._fieldnames]
        if new_keys or not self._fieldnames:
            self._fieldnames = self._fieldnames + new_keys
            # rewrite header (rare path; only happens when schema grows)
            rows = []
            if os.path.exists(self.path):
                with open(self.path, "r", newline="") as fh:
                    rows = list(csv.DictReader(fh))
            with open(self.path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        with open(self.path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            writer.writerow(row)


def save_checkpoint(path: str, **state) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> dict:
    # weights_only=False: our checkpoints embed an ExperimentConfig dataclass,
    # not just tensors, and PyTorch >= 2.6 defaults to weights_only=True which
    # would refuse to unpickle it. Only ever load checkpoints you created
    # yourself with this codebase -- this is not safe for untrusted files.
    return torch.load(path, map_location=map_location, weights_only=False)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, optim_cfg, steps_per_epoch: int):
    """Returns a callable step(epoch, step_in_epoch) -> lr that mutates
    optimizer.param_groups[*]['lr'] in place. Kept simple/explicit instead of
    using torch's scheduler classes so warmup + multistep/cosine compose
    trivially and are easy to unit-test."""

    base_lr = optim_cfg.lr
    warmup_steps = optim_cfg.warmup_epochs * steps_per_epoch

    def lr_at(epoch: int, step_in_epoch: int) -> float:
        global_step = epoch * steps_per_epoch + step_in_epoch
        if warmup_steps > 0 and global_step < warmup_steps:
            return base_lr * (global_step + 1) / warmup_steps
        if optim_cfg.lr_schedule == "multistep":
            lr = base_lr
            for m in optim_cfg.milestones:
                if epoch >= m:
                    lr *= optim_cfg.lr_gamma
            return lr
        elif optim_cfg.lr_schedule == "cosine":
            import math

            progress = (global_step - warmup_steps) / max(1, optim_cfg.epochs * steps_per_epoch - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * base_lr * (1 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown lr_schedule {optim_cfg.lr_schedule}")

    def step(epoch: int, step_in_epoch: int) -> float:
        lr = lr_at(epoch, step_in_epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr

    return step


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
