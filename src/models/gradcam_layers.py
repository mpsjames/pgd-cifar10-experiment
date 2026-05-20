"""Resolve architecture-specific Grad-CAM target layers."""

from __future__ import annotations

from torch import nn

ARCH_TO_GRADCAM_LAYER: dict[str, str] = {
    "resnet18": "model.model.layer3.1.conv2",
    "wrn_34_10": "model.block3.layer.4.conv2",
    "vit_tiny": "model.blocks.11.norm1",
}


def get_gradcam_target(model: nn.Module, arch: str) -> nn.Module:
    """Resolve the configured Grad-CAM target layer for an architecture.

    Args:
        model: Wrapped model whose submodule tree matches the configured
            architecture registry.
        arch: Architecture key from `ARCH_TO_GRADCAM_LAYER`.

    Returns:
        Concrete `nn.Module` used as the Grad-CAM activation target.

    Raises:
        KeyError: When `arch` has no registered layer path.
        AttributeError: When the stored path does not match the model
            structure.
    """
    if arch not in ARCH_TO_GRADCAM_LAYER:
        raise KeyError(f"Unknown architecture: {arch}")

    return model.get_submodule(ARCH_TO_GRADCAM_LAYER[arch])
