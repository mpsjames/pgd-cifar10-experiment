"""Implement FGSM for CIFAR-10 L-infinity attacks.

The implementation follows the single-step sign update from Goodfellow-style
FGSM, constrained to the project's Linf-only scope. Inputs stay in raw
pixel space; normalization happens inside `NormalizedModel`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.attacks.base import BaseAttack
from src.data.validation import validate_input_batch


class FGSMAttack(BaseAttack):
    """Apply Fast Gradient Sign Method for L-infinity perturbations."""

    def perturb(self, model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        """Generate a single-step adversarial batch with FGSM.

        Args:
            model: Victim model. Mode does not matter on entry; the original
                `model.training` state is restored before returning.
            x: Clean inputs, shape `(B, 3, 32, 32)`, dtype `float32`, range
                `[0, 1]`, on the same device as `model`.
            y: True labels, shape `(B,)`, dtype `long`, on the same device as
                `x`.

        Returns:
            `x_adv` with the same shape and dtype as `x`, clipped to `[0, 1]`.

        Raises:
            ValueError: When `self.config.norm != "Linf"`.
            AssertionError: When `validate_input_batch` rejects `x` or `y`.

        Notes:
            Gradient computation is forced to fp32 because outer autocast can
            silently degrade attack gradients on flat loss surfaces.
        """
        if self.config.norm != "Linf":
            raise ValueError("FGSM supports only Linf in this project")
        validate_input_batch(x, y)
        was_training = model.training
        model.eval()
        x_orig = x.detach()
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_adv = x_orig.float().clone().detach().requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            adv = x_orig + self.config.epsilon * grad.sign()
            adv = adv.clamp(0.0, 1.0)
        if was_training:
            model.train()
        return adv.detach().to(dtype=x.dtype)
