"""Forward-hook based activation extraction.

Grabs the output of an arbitrary named submodule during a normal forward
pass (no architecture changes needed) and reduces it to a flat feature
matrix that the SAE / raw-alignment loss can consume:

  - 'gap'     : global-average-pool each (C, H, W) map to a (C,) vector.
                Matches how SAE-FT operates on pooled CLIP embeddings; fast,
                low-dimensional, a reasonable default.
  - 'spatial' : keep every spatial location as its own "token", i.e. reshape
                (B, C, H, W) -> (B*H*W, C). More expressive (closer to how
                Feature Denoising and per-patch vision SAEs operate) but
                more expensive and the alignment loss must be reduced over
                positions. Clean and adversarial feature maps are always
                spatially aligned here (PGD does not change image geometry),
                so position-wise pairing is valid.
  - 'none'    : return the raw (B, C, H, W) tensor untouched (advanced use).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def get_module_by_name(model: nn.Module, dotted_name: str) -> nn.Module:
    module = model
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


class ActivationExtractor:
    def __init__(self, model: nn.Module, layer_name: str, reduce: str = "gap"):
        self.reduce = reduce
        self.layer_name = layer_name
        self._activation: Optional[torch.Tensor] = None
        module = get_module_by_name(model, layer_name)
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._activation = output

    def pop(self) -> torch.Tensor:
        """Return the reduced activation captured by the most recent forward
        pass. Raises if no forward pass has happened since construction/last pop."""
        if self._activation is None:
            raise RuntimeError(
                f"ActivationExtractor for '{self.layer_name}' has no captured activation. "
                "Did you run a forward pass through the model first?"
            )
        act = self._activation
        self._activation = None
        return self._reduce(act)

    def _reduce(self, act: torch.Tensor) -> torch.Tensor:
        if self.reduce == "gap":
            if act.dim() == 4:
                return act.mean(dim=(2, 3))
            return act
        elif self.reduce == "spatial":
            if act.dim() == 4:
                b, c, h, w = act.shape
                return act.permute(0, 2, 3, 1).reshape(b * h * w, c)
            return act
        elif self.reduce == "none":
            return act
        else:
            raise ValueError(f"Unknown reduce mode '{self.reduce}'")

    def remove(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
