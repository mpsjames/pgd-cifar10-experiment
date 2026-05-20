"""Implement untargeted APGD-CE for CIFAR-10 L-infinity perturbations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.attacks.apgd_core import run_restart
from src.attacks.base import BaseAttack


class APGDAttack(BaseAttack):
    """Apply APGD-CE under the project's CIFAR-10 Linf threat model."""

    def __init__(self, config) -> None:
        super().__init__(config)
        if config.rho is None or config.n_restarts is None:
            raise ValueError("APGD-CE requires rho and n_restarts on AttackConfig")
        if not 0.0 < float(config.rho) <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {config.rho}")
        if int(config.n_restarts) < 1:
            raise ValueError(f"n_restarts must be >= 1, got {config.n_restarts}")
        if int(config.n_restarts) > 100:
            raise ValueError(f"n_restarts must be <= 100, got {config.n_restarts}")
        self._debug_history: dict[str, list[list[Tensor]]] = {
            "best_loss": [],
            "step_size": [],
        }

    def perturb(self, model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        """Generate APGD-CE adversarial examples for one raw input batch."""
        was_training, x_float = self._prepare(model, x, y)
        if float(self.config.epsilon) <= 0.0 or int(self.config.num_steps) == 0:
            self._reset_debug_history()
            self._restore(model, was_training)
            return x.clone()

        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                global_best = x_float.clone()
                global_loss = torch.full(
                    (x_float.size(0),),
                    float("-inf"),
                    dtype=x_float.dtype,
                    device=x_float.device,
                )
                self._reset_debug_history()
                for restart in range(int(self.config.n_restarts)):
                    x_best, best_loss = run_restart(self, model, x_float, y, restart)
                    improved = best_loss > global_loss
                    if improved.any():
                        global_best[improved] = x_best[improved]
                        global_loss[improved] = best_loss[improved]
                return (
                    self._project_linf(x_float, global_best, float(self.config.epsilon))
                    .detach()
                    .to(dtype=x.dtype)
                )
        finally:
            self._restore(model, was_training)

    def _reset_debug_history(self) -> None:
        """Reset APGD introspection state; this is not a public metrics contract."""
        self._debug_history = {"best_loss": [], "step_size": []}
