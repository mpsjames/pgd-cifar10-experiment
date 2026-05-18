from __future__ import annotations

import matplotlib
import torch

matplotlib.use("Agg")

from src.experiments.config import ModelConfig
from src.models.builders import ARCH_BUILDERS, wrap_with_normalization
from src.viz.gradcam import compute_gradcam
from src.viz.perturbation_panels import make_perturbation_panel


def test_perturbation_panel_no_crash() -> None:
    x = torch.rand(1, 3, 32, 32)
    fig = make_perturbation_panel(x, x)
    assert fig is not None


def test_gradcam_renders_for_resnet18_smoke() -> None:
    model_config = ModelConfig("resnet18", None)
    model = wrap_with_normalization(ARCH_BUILDERS["resnet18"](model_config), model_config)
    x = torch.rand(1, 3, 32, 32)
    heatmap = compute_gradcam(model, "resnet18", x)
    assert heatmap.shape == (1, 32, 32)
