"""Core APGD restart routine kept separate from the public attack class."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def run_restart(attack, model: nn.Module, x_orig: Tensor, y: Tensor, restart: int):
    config = attack.config
    device = x_orig.device
    generator = torch.Generator(device=device).manual_seed(int(config.seed or 42) + restart)
    if config.random_start:
        noise = (
            torch.rand(x_orig.shape, generator=generator, device=device, dtype=x_orig.dtype) * 2.0
            - 1.0
        ) * float(config.epsilon)
        x_adv = attack._project_linf(x_orig, x_orig + noise, float(config.epsilon))
    else:
        x_adv = x_orig.clone()

    loss = attack._loss(model, x_adv, y, reduction="none").detach()
    x_best = x_adv.clone()
    best_loss = loss.detach().clone()
    x_prev = x_adv.clone()
    step_size = torch.full(
        (x_orig.size(0), 1, 1, 1),
        2.0 * float(config.epsilon),
        dtype=x_orig.dtype,
        device=device,
    )
    best_at_checkpoint = best_loss.clone()
    step_at_checkpoint = step_size.clone()
    improvements_since_checkpoint: list[Tensor] = []
    checkpoints = _checkpoints(int(config.num_steps))
    rho = float(config.rho)

    attack._debug_history["best_loss"].append([best_loss.detach().cpu().clone()])
    attack._debug_history["step_size"].append([step_size.flatten(1)[:, 0].detach().cpu().clone()])

    for step in range(1, int(config.num_steps) + 1):
        x_current = x_adv.detach().requires_grad_(True)
        ce = attack._loss(model, x_current, y, reduction="sum")
        grad = torch.autograd.grad(ce, x_current, only_inputs=True)[0]
        momentum = 0.75 * (x_adv.detach() - x_prev.detach())
        x_next = x_adv.detach() + step_size * grad.sign() + momentum
        x_next = attack._project_linf(x_orig, x_next, float(config.epsilon))
        x_prev = x_adv.detach()
        x_adv = x_next.detach()

        loss = attack._loss(model, x_adv, y, reduction="none").detach()
        improved = loss > best_loss
        improvements_since_checkpoint.append(improved.detach())
        if improved.any():
            x_best[improved] = x_adv[improved]
            best_loss[improved] = loss.detach()[improved]

        if step in checkpoints:
            step_size, halve = _adapt_step_size(
                step_size,
                best_loss,
                best_at_checkpoint,
                step_at_checkpoint,
                improvements_since_checkpoint,
                rho,
            )
            if halve.any():
                x_adv[halve] = x_best[halve]
                x_prev[halve] = x_best[halve]
            best_at_checkpoint = best_loss.clone()
            step_at_checkpoint = step_size.clone()
            improvements_since_checkpoint = []

        attack._debug_history["best_loss"][-1].append(best_loss.detach().cpu().clone())
        attack._debug_history["step_size"][-1].append(
            step_size.flatten(1)[:, 0].detach().cpu().clone()
        )

    return attack._project_linf(x_orig, x_best, float(config.epsilon)), best_loss.detach()


def _adapt_step_size(
    step_size: Tensor,
    best_loss: Tensor,
    best_at_checkpoint: Tensor,
    step_at_checkpoint: Tensor,
    improvements: list[Tensor],
    rho: float,
) -> tuple[Tensor, Tensor]:
    stacked = torch.stack(improvements)
    no_progress = (~stacked).float().mean(dim=0) >= rho
    no_best_gain = best_loss <= best_at_checkpoint + 1e-12
    step_unchanged = torch.isclose(step_size.flatten(), step_at_checkpoint.flatten())
    halve = no_progress | (step_unchanged & no_best_gain)
    step_size = step_size.clone()
    if halve.any():
        step_size[halve] = step_size[halve] * 0.5
    return step_size, halve


def _checkpoints(num_steps: int) -> set[int]:
    points = {math.ceil(p * num_steps) for p in (0.22, 0.42, 0.62, 0.82)}
    return {point for point in points if 1 <= point <= num_steps}
