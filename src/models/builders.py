"""Construct CIFAR-10 model variants and wrap them with normalization."""

from __future__ import annotations

from typing import Callable

from torch import Tensor, nn
from torchvision.models import resnet18

from src.experiments.config import ModelConfig
from src.models.normalize_wrapper import NormalizedModel
from src.models.vit import VisionTransformerTiny


class CIFARResNetAdapter(nn.Module):
    """Adapt torchvision ResNet stems to 32x32 CIFAR-10 inputs."""

    def __init__(self, factory: Callable[..., nn.Module], num_classes: int) -> None:
        super().__init__()
        self.model = factory(weights=None, num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Run the adapted ResNet on an already normalized input batch."""
        return self.model(x)


class WideBasicBlock(nn.Module):
    """Implement the residual block used by the custom WideResNet."""

    def __init__(
        self, in_planes: int, out_planes: int, stride: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, padding=1, bias=False
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_planes == out_planes
            else nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the two-convolution residual block."""
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(self.dropout(out))))
        return out + self.shortcut(x)


class WideLayer(nn.Module):
    """Stack multiple `WideBasicBlock` instances into one stage."""

    def __init__(
        self,
        n_blocks: int,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for idx in range(n_blocks):
            layers.append(
                WideBasicBlock(
                    in_planes if idx == 0 else out_planes,
                    out_planes,
                    stride if idx == 0 else 1,
                    dropout,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Run the sequential stage over one activation tensor."""
        return self.layer(x)


class WideResNet(nn.Module):
    """WideResNet for CIFAR-10. WRN-34-10 uses n=(34-4)/6=5 blocks per group."""

    def __init__(
        self,
        depth: int = 34,
        widen_factor: int = 10,
        dropout: float = 0.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy (depth - 4) % 6 == 0")
        n_blocks = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.block1 = WideLayer(
            n_blocks, widths[0], widths[1], stride=1, dropout=dropout
        )
        self.block2 = WideLayer(
            n_blocks, widths[1], widths[2], stride=2, dropout=dropout
        )
        self.block3 = WideLayer(
            n_blocks, widths[2], widths[3], stride=2, dropout=dropout
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Run the WRN trunk and return logits."""
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.pool(out).flatten(1)
        return self.fc(out)


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """Build the CIFAR-10 ResNet-18 variant used by this project."""
    return CIFARResNetAdapter(resnet18, num_classes)


def build_wrn_34_10(num_classes: int = 10) -> nn.Module:
    """Build the WRN-34-10 variant used in clean and adversarial runs."""
    return WideResNet(depth=34, widen_factor=10, dropout=0.0, num_classes=num_classes)


def build_vit_tiny(
    num_classes: int = 10,
    image_size: int = 32,
    patch_size: int = 4,
    embed_dim: int = 192,
    depth: int = 12,
    num_heads: int = 3,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
) -> nn.Module:
    """Build the native-CIFAR ViT-Tiny variant."""
    return VisionTransformerTiny(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_dropout=attn_dropout,
        num_classes=num_classes,
    )


def build_model(model_config: ModelConfig) -> nn.Module:
    """Build the architecture declared by `ModelConfig`."""
    if model_config.arch == "vit_tiny":
        if model_config.vit is None:
            raise ValueError("vit_tiny requires model_config.vit (ViTConfig) to be set")
        vit = model_config.vit
        return build_vit_tiny(
            num_classes=model_config.num_classes,
            image_size=vit.image_size,
            patch_size=vit.patch_size,
            embed_dim=vit.embed_dim,
            depth=vit.depth,
            num_heads=vit.num_heads,
            mlp_ratio=vit.mlp_ratio,
            dropout=vit.dropout,
            attn_dropout=vit.attn_dropout,
        )
    return ARCH_BUILDERS[model_config.arch](model_config.num_classes)


def wrap_with_normalization(
    model: nn.Module, model_config: ModelConfig
) -> NormalizedModel:
    """Wrap a raw classifier with CIFAR-10 normalization buffers.

    Args:
        model: Classifier that expects normalized inputs.
        model_config: Model config supplying `cifar_mean` and `cifar_std`.

    Returns:
        `NormalizedModel` that accepts raw `[0, 1]` CIFAR-10 images.
    """
    return NormalizedModel(model, model_config.cifar_mean, model_config.cifar_std)


ARCH_BUILDERS: dict[str, Callable[[int], nn.Module]] = {
    "resnet18": build_resnet18,
    "wrn_34_10": build_wrn_34_10,
    "vit_tiny": build_vit_tiny,
}
