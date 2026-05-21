from __future__ import annotations

from torch import nn

from src.experiments.config import ModelConfig, ViTConfig
from src.models.builders import ARCH_BUILDERS, wrap_with_normalization
from src.models.gradcam_layers import ARCH_TO_GRADCAM_LAYER, get_gradcam_target


def test_every_arch_has_gradcam_target_layer() -> None:
    assert set(ARCH_TO_GRADCAM_LAYER) == set(ARCH_BUILDERS)


def test_target_layer_path_resolves() -> None:
    for arch, builder in ARCH_BUILDERS.items():
        vit = ViTConfig() if arch == "vit_tiny" else None
        model_config = ModelConfig(arch=arch, checkpoint_path=None, vit=vit)  # type: ignore[arg-type]
        model = wrap_with_normalization(builder(model_config), model_config)
        target = get_gradcam_target(model, arch)
        assert isinstance(target, nn.Module)
