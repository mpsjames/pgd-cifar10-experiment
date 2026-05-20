"""Define the public adversarial-attack interface used across the project.

Attacks operate on raw `[0, 1]` image tensors and rely on `Normalizer`
to perform dataset normalization inside `forward` (plan §4.7). Subclasses are
responsible for generating candidate perturbations; callers remain responsible
for validating the result with `verify_perturbation` (plan §4.8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn.functional as F
from torch import Tensor, nn

from src.data.validation import validate_input_batch
from src.experiments.config import AttackConfig


class BaseAttack(ABC):
    """Define the shared contract for adversarial attacks.

    Subclass contract:
        - Call `super().__init__(config)` before validating family-specific
          hyperparameters.
        - Accept raw image batches in `[0, 1]` rather than pre-normalized
          tensors.
        - Return adversarial samples with the same shape as the input batch.
        - Keep gradient-sensitive attack math in fp32 when precision matters.
        - Restore the incoming model training mode before returning.

    Attributes:
        config: Immutable attack hyperparameters loaded from YAML.
    """

    SUPPORTED_NORMS: tuple[str, ...] = ("Linf",)

    def __init__(self, config: AttackConfig) -> None:
        if config.norm not in self.SUPPORTED_NORMS:
            raise ValueError(
                f"{type(self).__name__} only supports {self.SUPPORTED_NORMS}, "
                f"got norm={config.norm!r}"
            )
        self.config = config

    @abstractmethod
    def perturb(self, model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial examples for a raw `[0, 1]` input batch.

        Args:
            model: Victim model. Implementations may switch it to eval mode
                temporarily and must restore the incoming training state if
                they do so.
            x: Clean inputs, shape `(B, 3, 32, 32)`, dtype `float32`, range
                `[0, 1]`.
            y: True labels, shape `(B,)`, dtype `long`.

        Returns:
            Adversarial samples with the same shape as `x`. The caller must run
            `verify_perturbation` before reporting any derived metric.
        """

    @staticmethod
    def _loss(model: nn.Module, x: Tensor, y: Tensor, *, reduction: str = "none") -> Tensor:
        """Compute cross-entropy with an explicit reduction contract."""
        return F.cross_entropy(model(x), y, reduction=reduction)

    @staticmethod
    def _project_linf(x_orig: Tensor, x_adv: Tensor, epsilon: float) -> Tensor:
        """Project `x_adv` into the Linf epsilon-ball around `x_orig`."""
        delta = (x_adv - x_orig).clamp(-epsilon, epsilon)
        return (x_orig + delta).clamp(0.0, 1.0)

    @staticmethod
    def _prepare(model: nn.Module, x: Tensor, y: Tensor) -> tuple[bool, Tensor]:
        """Validate inputs, switch to eval mode, and return fp32 clean inputs."""
        validate_input_batch(x, y)
        was_training = model.training
        model.eval()
        return was_training, x.detach().float()

    @staticmethod
    def _restore(model: nn.Module, was_training: bool) -> None:
        """Restore the model's incoming training state."""
        if was_training:
            model.train()
