from __future__ import annotations

from torch import nn

from src.experiments.config import ModelConfig
from src.models.builders import ARCH_BUILDERS, wrap_with_normalization
from src.models.gradcam_layers import ARCH_TO_GRADCAM_LAYER, get_gradcam_target


def test_every_arch_has_gradcam_target_layer() -> None:
    assert set(ARCH_TO_GRADCAM_LAYER) == set(ARCH_BUILDERS)


def test_target_layer_path_resolves() -> None:
    for arch, builder in ARCH_BUILDERS.items():
        model = wrap_with_normalization(
            builder(10), ModelConfig(arch=arch, checkpoint_path=None)
        )
        target = get_gradcam_target(model, arch)
        assert isinstance(target, nn.Module)
