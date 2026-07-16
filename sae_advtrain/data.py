"""Dataloaders.

Important: images are handed out in raw [0, 1] range with NO Normalize()
transform. Normalization is applied inside the model (see models/build.py's
NormalizedModel) so that PGD perturbations are always crafted and clipped in
the standard, human-interpretable [0, 1] pixel space -- this matches Madry
et al. / TRADES / RobustBench conventions and makes epsilon values directly
comparable to published numbers.

`dataset: synthetic` generates random tensors entirely offline (no network
access) and exists so the whole pipeline can be smoke-tested and CI-checked
without ever touching the internet -- swap back to cifar10/cifar100 for
real experiments.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T


class SyntheticImageDataset(Dataset):
    """Deterministic random 32x32x3 images in [0, 1] with random labels.
    Only for smoke-testing the training/eval pipeline end to end without a
    real dataset -- do not draw research conclusions from it."""

    def __init__(self, size: int, num_classes: int = 10, seed: int = 0, image_size: int = 32):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, 3, image_size, image_size, generator=g)
        self.labels = torch.randint(0, num_classes, (size,), generator=g)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


def _split_indices(n: int, val_fraction: float, seed: int = 0) -> Tuple[list, list]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_fraction)) if val_fraction > 0 else 0
    return perm[n_val:], perm[:n_val]


def build_dataloaders(data_cfg) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Returns (train_loader, val_loader, test_loader, num_classes).
    val_loader is a small held-out slice of the *training* set (no
    augmentation) used for cheap monitoring during training; test_loader is
    the real held-out test split, used only in scripts/evaluate_checkpoint.py.
    """
    if data_cfg.dataset == "synthetic":
        num_classes = 10
        train_full = SyntheticImageDataset(data_cfg.synthetic_size, num_classes, seed=0)
        test_ds = SyntheticImageDataset(max(256, data_cfg.synthetic_size // 4), num_classes, seed=1)
        train_idx, val_idx = _split_indices(len(train_full), data_cfg.val_fraction)
        train_ds = Subset(train_full, train_idx)
        val_ds = Subset(train_full, val_idx)

    elif data_cfg.dataset in ("cifar10", "cifar100"):
        cls = torchvision.datasets.CIFAR10 if data_cfg.dataset == "cifar10" else torchvision.datasets.CIFAR100
        num_classes = 10 if data_cfg.dataset == "cifar10" else 100

        train_tf = T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        eval_tf = T.Compose([T.ToTensor()])

        train_full_aug = cls(root=data_cfg.data_root, train=True, download=True, transform=train_tf)
        train_full_noaug = cls(root=data_cfg.data_root, train=True, download=False, transform=eval_tf)
        test_ds = cls(root=data_cfg.data_root, train=False, download=True, transform=eval_tf)

        train_idx, val_idx = _split_indices(len(train_full_aug), data_cfg.val_fraction)
        train_ds = Subset(train_full_aug, train_idx)
        val_ds = Subset(train_full_noaug, val_idx)
    else:
        raise ValueError(f"Unknown dataset '{data_cfg.dataset}'")

    train_loader = DataLoader(
        train_ds, batch_size=data_cfg.batch_size, shuffle=True,
        num_workers=data_cfg.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg.eval_batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=data_cfg.eval_batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, num_classes
