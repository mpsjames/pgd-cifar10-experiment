"""CIFAR-10 ResNet adapter."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn


class CIFARResNetAdapter(nn.Module):
    """Adapt torchvision ResNet stems to 32x32 CIFAR-10 inputs."""

    def __init__(self, factory: Callable[..., nn.Module], num_classes: int) -> None:
        super().__init__()
        self.model = factory(weights=None, num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
