"""Resolve architecture-specific Grad-CAM target layers."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

# Values are either a dotted-path string (for nn.Module.get_submodule) or a
# callable that takes the wrapped model and returns the target nn.Module.
# Callables allow depth-independent resolution (e.g. blocks[-1]) so the
# registry stays correct when ViT depth is overridden in the YAML.
_LayerSpec = str | Callable[[nn.Module], nn.Module]

ARCH_TO_GRADCAM_LAYER: dict[str, _LayerSpec] = {
    "resnet18": "model.model.layer3.1.conv2",
    "vit_tiny": lambda m: m.get_submodule("model").blocks[-1].norm1,  # type: ignore[union-attr]
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
    spec = ARCH_TO_GRADCAM_LAYER[arch]
    if callable(spec):
        return spec(model)
    return model.get_submodule(spec)
