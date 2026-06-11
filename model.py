"""
ResNet implementation following:
  "Deep Residual Learning for Image Recognition", He et al., 2015.
  https://arxiv.org/abs/1512.03385

Blocks
- BasicBlock : 2 x 3x3 conv                 (ResNet-18/34 and all CIFAR nets)
- Bottleneck : 1x1 -> 3x3 -> 1x1            (ResNet-50/101/152)

Shortcuts (paper Sec. 3.3 / 4.1)
- 'A'    : identity; zero-padding when channels increase (parameter-free)
- 'B'    : identity; 1x1 conv projection only when dims change
- 'C'    : 1x1 conv projection on every shortcut
- 'none' : no skip connection at all -> the "plain" baseline networks the
           paper compares against (Fig. 4 left / Fig. 6 left)

Architectures
- ImageNet (Table 1): 7x7 s2 stem + maxpool, 4 stages of 64/128/256/512.
- CIFAR-10 (Sec. 4.2): 3x3 stem, 3 stages of 16/32/64, depth 6n+2,
  option A shortcuts -> resnet20/32/44/56/110/1202.
  Plain (non-residual) counterparts -> plain20/32/44/56/110.

Weight init: He (Kaiming) normal for conv layers, BN weight=1, bias=0.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)


class PadShortcut(nn.Module):
    """Option A: parameter-free shortcut.

    Subsamples spatially when stride > 1 and zero-pads extra channels
    (paper Sec. 3.3: "extra zero entries padded for increasing dimensions").
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.extra_channels = out_planes - in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.extra_channels > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.extra_channels))
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: nn.Module | None = None, residual: bool = True):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = shortcut
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + (x if self.shortcut is None else self.shortcut(x))
        return F.relu(out, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: nn.Module | None = None, residual: bool = True):
        super().__init__()
        # The paper does not pin down where the stride goes; the original
        # Caffe code strided the first 1x1. We stride the 3x3 ("v1.5",
        # torchvision default), which avoids dropping activations.
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = shortcut
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.residual:
            out = out + (x if self.shortcut is None else self.shortcut(x))
        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    """Generic ResNet: stage widths double from `base_planes`, one stage per
    entry in `layers`. Covers both the ImageNet nets (4 stages from 64) and
    the CIFAR nets (3 stages from 16)."""

    def __init__(self, block: type, layers: list[int], num_classes: int,
                 stem: str = "imagenet", base_planes: int = 64,
                 shortcut: str = "B"):
        super().__init__()
        if stem not in ("imagenet", "cifar"):
            raise ValueError(f"unknown stem {stem!r}")
        if shortcut not in ("A", "B", "C", "none"):
            raise ValueError(f"unknown shortcut option {shortcut!r}")
        self.shortcut_option = shortcut
        self.residual = shortcut != "none"
        self.in_planes = base_planes

        if stem == "imagenet":
            self.stem = nn.Sequential(
                nn.Conv2d(3, base_planes, kernel_size=7, stride=2, padding=3,
                          bias=False),
                nn.BatchNorm2d(base_planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:  # cifar: 3x3 stride-1 conv, no maxpool (32x32 input)
            self.stem = nn.Sequential(
                conv3x3(3, base_planes),
                nn.BatchNorm2d(base_planes),
                nn.ReLU(inplace=True),
            )

        stages = []
        for i, blocks in enumerate(layers):
            planes = base_planes * 2 ** i
            stages.append(self._make_stage(block, planes, blocks,
                                           stride=1 if i == 0 else 2))
        self.stages = nn.Sequential(*stages)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_planes, num_classes)

        self._init_weights()

    def _make_shortcut(self, in_planes: int, out_planes: int,
                       stride: int) -> nn.Module | None:
        if self.shortcut_option == "none":
            return None
        dims_change = stride != 1 or in_planes != out_planes
        if self.shortcut_option == "A":
            return PadShortcut(in_planes, out_planes, stride) if dims_change else None
        if self.shortcut_option == "B" and not dims_change:
            return None
        # B with changed dims, or C always: 1x1 projection
        return nn.Sequential(
            conv1x1(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
        )

    def _make_stage(self, block: type, planes: int, blocks: int,
                    stride: int) -> nn.Sequential:
        out_planes = planes * block.expansion
        layers = []
        for i in range(blocks):
            block_stride = stride if i == 0 else 1
            shortcut = self._make_shortcut(self.in_planes, out_planes,
                                           block_stride)
            layers.append(block(self.in_planes, planes, block_stride,
                                shortcut, residual=self.residual))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He init, paper Sec. 3.4
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# --------------------------------------------------------------------------
# ImageNet architectures (Table 1)
# --------------------------------------------------------------------------

def _imagenet_resnet(block, layers):
    def factory(num_classes: int = 1000, stem: str = "imagenet",
                shortcut: str = "B") -> ResNet:
        return ResNet(block, layers, num_classes, stem=stem,
                      base_planes=64, shortcut=shortcut)
    return factory


resnet18  = _imagenet_resnet(BasicBlock, [2, 2, 2, 2])
resnet34  = _imagenet_resnet(BasicBlock, [3, 4, 6, 3])
resnet50  = _imagenet_resnet(Bottleneck, [3, 4, 6, 3])
resnet101 = _imagenet_resnet(Bottleneck, [3, 4, 23, 3])
resnet152 = _imagenet_resnet(Bottleneck, [3, 8, 36, 3])


# --------------------------------------------------------------------------
# CIFAR-10 architectures (Sec. 4.2): depth = 6n + 2, option A shortcuts.
# Plain nets are the same topology with no skip connections (Fig. 6 left):
# the baseline whose error *increases* with depth, motivating ResNets.
# --------------------------------------------------------------------------

def _cifar_layers(depth: int) -> list[int]:
    if (depth - 2) % 6 != 0:
        raise ValueError("CIFAR net depth must be 6n + 2")
    n = (depth - 2) // 6
    return [n, n, n]


def _cifar_resnet(depth: int):
    def factory(num_classes: int = 10, shortcut: str = "A") -> ResNet:
        return ResNet(BasicBlock, _cifar_layers(depth), num_classes,
                      stem="cifar", base_planes=16, shortcut=shortcut)
    return factory


def _cifar_plain(depth: int):
    def factory(num_classes: int = 10) -> ResNet:
        return ResNet(BasicBlock, _cifar_layers(depth), num_classes,
                      stem="cifar", base_planes=16, shortcut="none")
    return factory


resnet20   = _cifar_resnet(20)
resnet32   = _cifar_resnet(32)
resnet44   = _cifar_resnet(44)
resnet56   = _cifar_resnet(56)
resnet110  = _cifar_resnet(110)
resnet1202 = _cifar_resnet(1202)

plain20  = _cifar_plain(20)
plain32  = _cifar_plain(32)
plain44  = _cifar_plain(44)
plain56  = _cifar_plain(56)
plain110 = _cifar_plain(110)


IMAGENET_MODELS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}

CIFAR_MODELS = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet1202": resnet1202,
}

PLAIN_MODELS = {
    "plain20": plain20,
    "plain32": plain32,
    "plain44": plain44,
    "plain56": plain56,
    "plain110": plain110,
}

MODELS = {**IMAGENET_MODELS, **CIFAR_MODELS, **PLAIN_MODELS}

_CIFAR_NAME = re.compile(r"^(resnet|plain)(\d+)$")


def resolve_model(name: str):
    """Return a model factory for `name`, raising ValueError if invalid.

    Besides the predefined names, any CIFAR-style name with a valid 6n+2
    depth works: resolve_model("resnet38"), resolve_model("plain26"), ...
    """
    if name in MODELS:
        return MODELS[name]
    m = _CIFAR_NAME.match(name)
    if m:
        family, depth = m.group(1), int(m.group(2))
        rem = (depth - 2) % 6
        if rem != 0:
            lo, hi = depth - rem, depth - rem + 6
            raise ValueError(
                f"{name}: CIFAR {family} depth must be 6n+2 "
                f"(nearest valid: {family}{lo}, {family}{hi})")
        return _cifar_plain(depth) if family == "plain" else _cifar_resnet(depth)
    raise ValueError(f"unknown model {name!r}; use one of {list(MODELS)} "
                     f"or resnet/plain with a 6n+2 depth (e.g. resnet38)")


def build_model(name: str, num_classes: int = 10, **kwargs) -> ResNet:
    """kwargs: `shortcut` for resnet models, `stem` for ImageNet models only."""
    return resolve_model(name)(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    for name in MODELS:
        kwargs = {"stem": "cifar"} if name in IMAGENET_MODELS else {}
        m = build_model(name, num_classes=10, **kwargs)
        y = m(x)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{name:10s}  out={tuple(y.shape)}  params={params:.2f}M")
