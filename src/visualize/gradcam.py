"""Compute Grad-CAM heatmaps for the supported CIFAR-10 architectures."""

from __future__ import annotations

import torch
from torch import Tensor

from src.models.gradcam_layers import get_gradcam_target
from src.models.normalizer import Normalizer


def compute_gradcam(
    model: Normalizer, arch: str, x: Tensor, class_idx: int | None = None
) -> Tensor:
    """Compute a normalized Grad-CAM heatmap for an input batch.

    Args:
        model: Wrapped classifier to explain.
        arch: Architecture key used to resolve the target layer.
        x: Input images, shape `(B, 3, 32, 32)`, dtype float, range `[0, 1]`.
        class_idx: Optional class index to explain. When `None`, use the top-1
            prediction of the first sample.

    Returns:
        Heatmap tensor of shape `(B, 32, 32)` normalized to `[0, 1]`.

    Notes:
        Forward/backward hooks are always removed in a `finally` block so
        repeated notebook runs do not accumulate stale hooks.
    """
    model.eval()
    target_layer = get_gradcam_target(model, arch)
    activations: list[Tensor] = []
    gradients: list[Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    try:
        logits = model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1)[0].item())
        score = logits[:, class_idx].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        activation = activations[-1]
        gradient = gradients[-1]
        if activation.ndim == 4:
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1))
            cam = torch.nn.functional.interpolate(
                cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
        elif activation.ndim == 3:
            # ViT tokens are `(B, CLS + patches, D)`; Grad-CAM uses patch tokens.
            patch_activation = activation[:, 1:, :]
            patch_gradient = gradient[:, 1:, :]
            weights = patch_gradient.mean(dim=1, keepdim=True)
            cam_tokens = torch.relu((weights * patch_activation).sum(dim=2))
            side = int(cam_tokens.size(1) ** 0.5)
            if side * side != cam_tokens.size(1):
                raise ValueError(
                    f"N_patches={cam_tokens.size(1)} is not a perfect square; "
                    "Grad-CAM reshape requires image_size divisible by patch_size"
                )
            cam = cam_tokens.reshape(-1, side, side)
            cam = torch.nn.functional.interpolate(
                cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
        else:
            raise ValueError(f"Unsupported Grad-CAM activation rank for {arch}: {activation.ndim}")
        cam_min = cam.flatten(1).min(dim=1).values[:, None, None]
        cam_max = cam.flatten(1).max(dim=1).values[:, None, None]
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)
    finally:
        handle_fwd.remove()
        handle_bwd.remove()
