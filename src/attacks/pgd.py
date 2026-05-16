"""Implement PGD and BIM for CIFAR-10 L-infinity perturbations.

This module covers both random-start PGD and deterministic BIM via
`AttackConfig.random_start`. The update order matches the project contract:
clip to the delta ball first, then clip to the `[0, 1]` pixel domain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.attacks.base import BaseAttack
from src.data.validation import validate_input_batch


class PGDAttack(BaseAttack):
    """Apply Projected Gradient Descent or BIM under an L-infinity budget."""

    def perturb(self, model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial examples with iterative sign-gradient updates.

        Args:
            model: Victim model. Mode does not matter on entry; the original
                `model.training` value is restored before returning.
            x: Clean inputs, shape `(B, 3, 32, 32)`, dtype `float32`, range
                `[0, 1]`, on the same device as `model`.
            y: True labels, shape `(B,)`, dtype `long`.

        Returns:
            `x_adv` with the same shape and dtype as `x`, projected into the
            configured Linf ball and clipped to `[0, 1]`.

        Raises:
            ValueError: When `self.config.norm != "Linf"`.
            AssertionError: When `validate_input_batch` rejects `x` or `y`.

        Notes:
            - `random_start=False` yields BIM behavior.
            - Attack gradients run in fp32 even under outer AMP.
            - Delta clip must happen before pixel clip; reversing the order can
            move `x_adv` outside the epsilon ball after clipping.
        """
        if self.config.norm != "Linf":
            raise ValueError("PGD supports only Linf in this project")
        validate_input_batch(x, y)
        was_training = model.training
        model.eval()

        x_orig = x.detach()
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_float = x_orig.float()
            if self.config.epsilon == 0 or self.config.num_steps == 0:
                x_adv = x_float.clone()
            elif self.config.random_start:
                noise = torch.empty_like(x_float).uniform_(
                    -self.config.epsilon, self.config.epsilon
                )
                x_adv = (x_float + noise).clamp(0.0, 1.0)
            else:
                x_adv = x_float.clone()

            for _ in range(self.config.num_steps):
                x_adv = x_adv.detach().requires_grad_(True)
                loss = F.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
                step = x_adv + self.config.alpha * grad.sign()
                # Delta clip must run before pixel clip to preserve the Linf budget.
                delta = torch.clamp(
                    step - x_float, min=-self.config.epsilon, max=self.config.epsilon
                )
                x_adv = torch.clamp(x_float + delta, min=0.0, max=1.0)

        if was_training:
            model.train()
        return x_adv.detach().to(dtype=x.dtype)
