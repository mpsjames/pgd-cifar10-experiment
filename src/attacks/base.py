"""Define the public adversarial-attack interface used across the project.

Attacks operate on raw `[0, 1]` image tensors and rely on `NormalizedModel`
to perform dataset normalization inside `forward` (plan §4.7). Subclasses are
responsible for generating candidate perturbations; callers remain responsible
for validating the result with `verify_perturbation` (plan §4.8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from src.experiments.config import AttackConfig


class BaseAttack(ABC):
    """Define the shared contract for adversarial attacks.

    Subclass contract:
        - Store the immutable `AttackConfig` passed to `__init__`.
        - Accept raw image batches in `[0, 1]` rather than pre-normalized
        tensors.
        - Return adversarial samples with the same shape as the input batch.
        - Keep gradient-sensitive attack math in fp32 when precision matters.

    Attributes:
        config: Immutable attack hyperparameters loaded from YAML.
    """

    def __init__(self, config: AttackConfig):
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
