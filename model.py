"""
ResNet implementation following:
  "Deep Residual Learning for Image Recognition", He et al., 2015.
  https://arxiv.org/abs/1512.03385

- BasicBlock: 2 x 3x3 conv (ResNet-18/34)
- Bottleneck: 1x1 -> 3x3 -> 1x1 (ResNet-50/101/152)
- Shortcut: identity when shapes match, 1x1 conv projection (option B) when
  dimensions / spatial size change.
- Weight init: He (Kaiming) normal for conv layers (Sec. 3.4 / He 2015).
  BatchNorm weight=1, bias=0.
- Stem variants:
    'imagenet' : 7x7 stride-2 conv + 3x3 stride-2 maxpool  (224x224 input)
    'cifar'    : 3x3 stride-1 conv, no maxpool             (32x32 input)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        # Original paper places stride on the 3x3 (v1). Torchvision's "v1.5"
        # places it on the 3x3 as well — we follow the paper exactly.
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = out + identity
        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    def __init__(self, block, layers: list[int], num_classes: int = 1000,
                 stem: str = "imagenet"):
        super().__init__()
        assert stem in ("imagenet", "cifar")
        self.in_planes = 64

        if stem == "imagenet":
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:  # cifar
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes: int, blocks: int, stride: int):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            # Option B: 1x1 projection shortcut when dims change.
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes=1000, stem="imagenet"):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, stem)

def resnet34(num_classes=1000, stem="imagenet"):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, stem)

def resnet50(num_classes=1000, stem="imagenet"):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, stem)

def resnet101(num_classes=1000, stem="imagenet"):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, stem)

def resnet152(num_classes=1000, stem="imagenet"):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, stem)


MODELS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


def build_model(name: str, num_classes: int = 10, stem: str = "cifar"):
    if name not in MODELS:
        raise ValueError(f"unknown model {name}, choose from {list(MODELS)}")
    return MODELS[name](num_classes=num_classes, stem=stem)


if __name__ == "__main__":
    for n in MODELS:
        m = build_model(n, num_classes=10, stem="cifar")
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{n:10s}  out={tuple(y.shape)}  params={params:.2f}M")
