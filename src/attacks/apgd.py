"""Implement untargeted APGD-CE for CIFAR-10 L-infinity perturbations.

Reference:
    Croce and Hein, "Reliable evaluation of adversarial robustness with an
    ensemble of diverse parameter-free attacks", ICML 2020.

The implementation stays inside the project contract from principles §4.8:
attacks consume raw `[0, 1]` tensors, run their gradient-sensitive math in
fp32, and project with delta clipping before pixel clipping.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.attacks.base import BaseAttack
from src.data.validation import validate_input_batch


class APGDAttack(BaseAttack):
    """Apply APGD-CE under the project's CIFAR-10 Linf threat model."""

    def __init__(self, config) -> None:
        super().__init__(config)
        if config.norm != "Linf":
            raise ValueError("APGD-CE supports only Linf in this project")
        if config.rho is None or config.n_restarts is None:
            raise ValueError("APGD-CE requires rho and n_restarts on AttackConfig")
        if not 0.0 < float(config.rho) <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {config.rho}")
        if int(config.n_restarts) < 1:
            raise ValueError(f"n_restarts must be >= 1, got {config.n_restarts}")
        if int(config.n_restarts) > 100:
            raise ValueError(f"n_restarts must be <= 100, got {config.n_restarts}")
        self.debug_best_loss_history: list[Tensor] = []
        self.debug_step_size_history: list[Tensor] = []

    def perturb(self, model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        """Generate APGD-CE adversarial examples for one raw input batch."""
        validate_input_batch(x, y)
        if self.config.epsilon == 0.0 or self.config.num_steps == 0:
            self.debug_best_loss_history = []
            self.debug_step_size_history = []
            return x.clone()

        was_training = model.training
        model.eval()
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                x_float = x.detach().float()
                global_best = x_float.clone()
                global_loss = torch.full(
                    (x_float.size(0),),
                    float("-inf"),
                    dtype=x_float.dtype,
                    device=x_float.device,
                )
                self.debug_best_loss_history = []
                self.debug_step_size_history = []
                for restart in range(int(self.config.n_restarts)):
                    x_best, best_loss = self._run_restart(model, x_float, y, restart)
                    improved = best_loss > global_loss
                    if improved.any():
                        global_best[improved] = x_best[improved]
                        global_loss[improved] = best_loss[improved]
                return self._project(x_float, global_best).detach().to(dtype=x.dtype)
        finally:
            if was_training:
                model.train()

    def _run_restart(
        self, model: nn.Module, x_orig: Tensor, y: Tensor, restart: int
    ) -> tuple[Tensor, Tensor]:
        device = x_orig.device
        generator = torch.Generator(device=device).manual_seed(
            int(self.config.seed or 42) + restart
        )
        if self.config.random_start:
            noise = (
                torch.rand(
                    x_orig.shape,
                    generator=generator,
                    device=device,
                    dtype=x_orig.dtype,
                )
                * 2.0
                - 1.0
            ) * float(self.config.epsilon)
            x_adv = self._project(x_orig, x_orig + noise)
        else:
            x_adv = x_orig.clone()

        loss = self._loss(model, x_adv, y)
        x_best = x_adv.clone()
        best_loss = loss.detach().clone()
        x_prev = x_adv.clone()
        step_size = torch.full(
            (x_orig.size(0), 1, 1, 1),
            2.0 * float(self.config.epsilon),
            dtype=x_orig.dtype,
            device=device,
        )
        best_at_checkpoint = best_loss.clone()
        step_at_checkpoint = step_size.clone()
        improvements_since_checkpoint: list[Tensor] = []
        checkpoints = _checkpoints(int(self.config.num_steps))
        rho = float(self.config.rho)

        self.debug_best_loss_history.append(best_loss.detach().cpu().clone())
        self.debug_step_size_history.append(
            step_size.flatten(1)[:, 0].detach().cpu().clone()
        )

        for step in range(1, int(self.config.num_steps) + 1):
            x_current = x_adv.detach().requires_grad_(True)
            ce = F.cross_entropy(model(x_current), y, reduction="sum")
            grad = torch.autograd.grad(ce, x_current, only_inputs=True)[0]
            momentum = 0.75 * (x_adv.detach() - x_prev.detach())
            x_next = x_adv.detach() + step_size * grad.sign() + momentum
            x_next = self._project(x_orig, x_next)
            x_prev = x_adv.detach()
            x_adv = x_next.detach()

            loss = self._loss(model, x_adv, y)
            improved = loss > best_loss
            improvements_since_checkpoint.append(improved.detach())
            if improved.any():
                x_best[improved] = x_adv[improved]
                best_loss[improved] = loss.detach()[improved]

            if step in checkpoints:
                stacked = torch.stack(improvements_since_checkpoint)
                no_improve_fraction = (~stacked).float().mean(dim=0)
                no_progress = no_improve_fraction >= rho
                no_best_gain = best_loss <= best_at_checkpoint + 1e-12
                step_unchanged = torch.isclose(
                    step_size.flatten(), step_at_checkpoint.flatten()
                )
                halve = no_progress | (step_unchanged & no_best_gain)
                if halve.any():
                    step_size[halve] = step_size[halve] * 0.5
                    x_adv[halve] = x_best[halve]
                    x_prev[halve] = x_best[halve]
                best_at_checkpoint = best_loss.clone()
                step_at_checkpoint = step_size.clone()
                improvements_since_checkpoint = []

            self.debug_best_loss_history.append(best_loss.detach().cpu().clone())
            self.debug_step_size_history.append(
                step_size.flatten(1)[:, 0].detach().cpu().clone()
            )

        return self._project(x_orig, x_best), best_loss.detach()

    def _loss(self, model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(model(x), y, reduction="none").detach()

    def _project(self, x_orig: Tensor, candidate: Tensor) -> Tensor:
        delta = torch.clamp(
            candidate - x_orig,
            min=-float(self.config.epsilon),
            max=float(self.config.epsilon),
        )
        return torch.clamp(x_orig + delta, min=0.0, max=1.0)


def _checkpoints(num_steps: int) -> set[int]:
    points = {math.ceil(p * num_steps) for p in (0.22, 0.42, 0.62, 0.82)}
    return {point for point in points if 1 <= point <= num_steps}
