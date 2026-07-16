"""SAEController: the single place that implements all four answers to
"what do we do about the SAE becoming stale as the backbone is trained?"

  - frozen            : SAE parameters never change. Cheapest, riskiest --
                         used mainly as a naive baseline to demonstrate the
                         drift problem is real in your setting.
  - frozen_residual    : SAE parameters never change, but we add a loss term
                         (on the BACKBONE's parameters) that keeps clean-input
                         activations reconstructable by the frozen SAE. This
                         mirrors SAE-FT's regularize-the-model-not-the-SAE
                         strategy and is the recommended starting point.
  - joint              : SAE parameters are updated every step, on a
                         detached copy of the activations, via their own
                         Adam optimizer and a pure reconstruction objective.
                         Alternating optimization: the alignment loss's
                         gradient w.r.t. SAE parameters is never applied.
  - periodic_refresh   : behaves like frozen_residual between refreshes;
                         every `refresh_every` epochs, unfreezes the SAE and
                         runs `refresh_steps` reconstruction-loss updates on
                         fresh clean activations, then re-freezes it. A
                         middle ground between "never update" and "update
                         every step".

All four expose the same interface so trainer.py doesn't need to know which
one is active beyond passing the right config.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from . import losses
from .sae import BaseSAE, L1SAE, dead_feature_fraction, fraction_of_variance_unexplained


class SAEController:
    def __init__(self, sae: BaseSAE, drift_cfg):
        self.sae = sae
        self.cfg = drift_cfg
        self.sae_optimizer: Optional[torch.optim.Optimizer] = None

        if drift_cfg.mode in ("frozen", "frozen_residual", "periodic_refresh"):
            self._freeze()
        elif drift_cfg.mode == "joint":
            self._unfreeze()
            self.sae_optimizer = torch.optim.Adam(self.sae.parameters(), lr=drift_cfg.sae_lr)
        else:
            raise ValueError(f"Unknown drift.mode '{drift_cfg.mode}'")

    # -- freezing helpers ----------------------------------------------------

    def _freeze(self):
        for p in self.sae.parameters():
            p.requires_grad_(False)

    def _unfreeze(self):
        for p in self.sae.parameters():
            p.requires_grad_(True)

    @property
    def is_frozen(self) -> bool:
        return not next(self.sae.parameters()).requires_grad

    # -- per-step API used by trainer.py -------------------------------------

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Differentiable w.r.t. h regardless of mode -- freezing only stops
        gradient from being *applied* to the SAE's own parameters, it never
        blocks gradient from flowing through to the backbone."""
        return self.sae.encode(h)

    def pre_backward_aux_losses(
        self,
        h_clean: torch.Tensor,
        h_clean_initial: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Loss terms that must be ADDED to the backbone's total loss and
        backpropagated together with it (safe to call before
        total_loss.backward()). Returns {} for modes with nothing to add."""
        if self.cfg.mode in ("frozen_residual", "periodic_refresh"):
            return {"residual_loss": self._residual_loss(h_clean, h_clean_initial) * self.cfg.residual_weight}
        return {}

    def post_backward_update(self, h_clean_detached: torch.Tensor) -> Dict[str, float]:
        """Anything that mutates the SAE's OWN parameters in place. Must only
        be called AFTER the backbone's optimizer.step() has already consumed
        the graph that used the SAE's pre-update parameters (e.g. the
        alignment loss) -- calling this first corrupts that graph's
        backward pass (PyTorch's autograd versioning will raise a
        "modified by an inplace operation" RuntimeError; this bit us during
        development and is exactly why the split exists). 'frozen' /
        'frozen_residual' / 'periodic_refresh' are no-ops here between
        refreshes; only 'joint' does anything on every step."""
        if self.cfg.mode == "joint":
            return {"sae_recon_loss": self._sae_reconstruction_step(h_clean_detached)}
        return {}

    def _residual_loss(self, h_clean: torch.Tensor, h_clean_initial: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cfg.residual_kind == "absolute":
            return losses.residual_absolute_loss(self.sae, h_clean)
        elif self.cfg.residual_kind == "delta":
            if h_clean_initial is None:
                raise ValueError(
                    "drift.residual_kind='delta' requires an initial-backbone activation snapshot; "
                    "trainer.py must be configured to keep one (see Trainer._snapshot_initial_backbone)."
                )
            return losses.residual_delta_loss(self.sae, h_clean, h_clean_initial)
        else:
            raise ValueError(f"Unknown drift.residual_kind '{self.cfg.residual_kind}'")

    def _sae_reconstruction_step(self, h_detached: torch.Tensor) -> float:
        self.sae_optimizer.zero_grad(set_to_none=True)
        recon, codes = self.sae(h_detached)
        loss = losses.sae_reconstruction_loss(h_detached, recon)
        if isinstance(self.sae, L1SAE):
            loss = loss + self.sae.sparsity_penalty(codes)
        loss.backward()
        self.sae_optimizer.step()
        self.sae.renormalize_decoder_()
        return loss.item()

    # -- periodic refresh -----------------------------------------------------

    def refresh(self, activation_batches) -> float:
        """`activation_batches` is an iterable yielding already-computed,
        detached clean-activation tensors (trainer.py is responsible for
        running the backbone forward pass, since only it knows how to build
        clean batches / apply the hook). Temporarily unfreezes the SAE, runs
        one reconstruction step per yielded batch, then re-freezes it.
        Returns the mean reconstruction loss observed during the refresh."""
        was_frozen = self.is_frozen
        self._unfreeze()
        if self.sae_optimizer is None:
            self.sae_optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.cfg.sae_lr)
        total, count = 0.0, 0
        for h in activation_batches:
            total += self._sae_reconstruction_step(h)
            count += 1
        if was_frozen:
            self._freeze()
        return total / max(1, count)

    # -- diagnostics ------------------------------------------------------------

    def diagnostics(self, h: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            recon, codes = self.sae(h)
            return {
                "sae_fvu": fraction_of_variance_unexplained(h, recon),
                "sae_dead_frac": dead_feature_fraction(codes),
            }
