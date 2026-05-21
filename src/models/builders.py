"""Factory functions and checkpoint loader for CIFAR-10 models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from src.experiments.config import ModelConfig
from src.models.normalizer import Normalizer
from src.models.resnet import CIFARResNetAdapter
from src.models.vit import VisionTransformerTiny


def build_resnet18(model_config: ModelConfig) -> nn.Module:
    """Build the CIFAR-10 ResNet-18 variant used by this project."""
    return CIFARResNetAdapter(resnet18, model_config.num_classes)


def build_vit_tiny(model_config: ModelConfig) -> nn.Module:
    """Build the native-CIFAR ViT-Tiny variant declared by `model_config.vit`."""
    if model_config.vit is None:
        raise ValueError("vit_tiny requires model_config.vit (ViTConfig) to be set")
    vit = model_config.vit
    return VisionTransformerTiny(
        image_size=vit.image_size,
        patch_size=vit.patch_size,
        embed_dim=vit.embed_dim,
        depth=vit.depth,
        num_heads=vit.num_heads,
        mlp_ratio=vit.mlp_ratio,
        dropout=vit.dropout,
        attn_dropout=vit.attn_dropout,
        drop_path=vit.drop_path,
        num_classes=model_config.num_classes,
    )


def build_model(model_config: ModelConfig) -> nn.Module:
    """Build the architecture declared by `ModelConfig`."""
    return ARCH_BUILDERS[model_config.arch](model_config)


def wrap_with_normalization(model: nn.Module, model_config: ModelConfig) -> Normalizer:
    """Wrap a raw classifier with CIFAR-10 normalization buffers."""
    return Normalizer(model, model_config.cifar_mean, model_config.cifar_std)


ARCH_BUILDERS: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "resnet18": build_resnet18,
    "vit_tiny": build_vit_tiny,
}


def build_normalized_model(model_config: ModelConfig) -> Normalizer:
    """Build and wrap a model with CIFAR-10 normalization in one call."""
    return wrap_with_normalization(build_model(model_config), model_config)


def load_model_from_checkpoint(model_config: ModelConfig, path: Path) -> Normalizer:
    """Load a NormalizedModel from a saved checkpoint file."""
    model = build_normalized_model(model_config).eval()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model
