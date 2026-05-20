"""WideResNet for CIFAR-10."""

from __future__ import annotations

from torch import Tensor, nn


class WideBasicBlock(nn.Module):
    """Residual block used by WideResNet."""

    def __init__(self, in_planes: int, out_planes: int, stride: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_planes == out_planes
            else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(self.dropout(out))))
        return out + self.shortcut(x)


class WideLayer(nn.Module):
    """Stack of WideBasicBlock instances forming one stage."""

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
        self.block1 = WideLayer(n_blocks, widths[0], widths[1], stride=1, dropout=dropout)
        self.block2 = WideLayer(n_blocks, widths[1], widths[2], stride=2, dropout=dropout)
        self.block3 = WideLayer(n_blocks, widths[2], widths[3], stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.pool(out).flatten(1)
        return self.fc(out)
