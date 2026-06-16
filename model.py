"""
ResNet — "Deep Residual Learning for Image Recognition", He et al., 2015.
https://arxiv.org/abs/1512.03385

Two building blocks (paper Fig. 5), defined separately and explicitly:
  BasicBlock : 3x3 -> 3x3                 (ResNet-18/34 and all CIFAR nets)
  Bottleneck : 1x1 -> 3x3 -> 1x1          (ResNet-50/101/152)

The shortcut matches the block INPUT to the block OUTPUT when they differ:
  CIFAR  (Sec. 4.2) : zero-padding, parameter-free   (option A)
  ImageNet (Table 1): 1x1 conv projection            (option B)

A `plain` net is the same network with the skip connections removed; it is
the non-residual baseline whose error grows with depth (paper Fig. 6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


class ZeroPadShortcut(nn.Module):
    """Option A shortcut (paper Sec. 3.3) — parameter-free.

    Matches the block's input to its output when dimensions change, with no
    learned weights:
      1. if the block downsamples (stride > 1), subsample the input the same way;
      2. append `added_channels` all-zero channels so the channel count matches.
    """

    def __init__(self, stride, added_channels):
        super().__init__()
        self.stride = stride
        self.added_channels = added_channels

    def forward(self, x):
        # 1) spatial downsample: keep every `stride`-th pixel along H and W
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        # 2) concatenate zero-filled channels so the channels match the output
        batch, _, height, width = x.shape
        zeros = x.new_zeros(batch, self.added_channels, height, width)
        return torch.cat([x, zeros], dim=1)


def make_shortcut(in_ch, out_ch, stride, projection):
    """Skip path matching the block INPUT (in_ch) to its OUTPUT (out_ch).
    None means identity (dimensions already match)."""
    if in_ch == out_ch and stride == 1:
        return None         # Identical shortcut
    if projection:                                   
        return nn.Sequential(conv1x1(in_ch, out_ch, stride),
                             nn.BatchNorm2d(out_ch))
    return ZeroPadShortcut(stride, out_ch - in_ch)    # option A: zero-pad


class BasicBlock(nn.Module):
    """Two 3x3 convolutions: in_ch -> out_ch -> out_ch."""

    def __init__(self, in_ch, out_ch, stride=1, projection=False, residual=True):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = residual

        if residual:
            self.shortcut = make_shortcut(in_ch, out_ch, stride, projection)
        else:   # plain network: no skip connection
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # added before activation function
        if self.residual:
            out = out + (x if self.shortcut is None else self.shortcut(x))
        return F.relu(out)


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1"""
    def __init__(self, in_ch, mid_ch, out_ch, stride=1, projection=True,
                 residual=True):
        super().__init__()
        self.conv1 = conv1x1(in_ch, mid_ch)               # reduce
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = conv3x3(mid_ch, mid_ch, stride)      # bottleneck 3x3
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = conv1x1(mid_ch, out_ch)              # restore
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.residual = residual
        if residual:
            self.shortcut = make_shortcut(in_ch, out_ch, stride, projection)
        else:   # plain network: no skip connection
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.residual:
            out = out + (x if self.shortcut is None else self.shortcut(x))
        return F.relu(out)


class ResNet(nn.Module):
    """Assembles stem + stages + classifier. `stages` is a list of specs,
    one per resolution, with the channel numbers written out explicitly:
        ("basic", out_ch, n_blocks, stride)
        ("bottleneck", mid_ch, out_ch, n_blocks, stride)"""

    def __init__(self, stem, stages, num_classes, residual=True):
        super().__init__()
        self.residual = residual # If residual is false then its plain network
        self.projection = (stem == "imagenet")   # B for ImageNet, A for CIFAR

        if stem == "imagenet":   # 224x224 input
            base = 64
            self.stem = nn.Sequential(
                nn.Conv2d(3, base, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(base),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:     # cifar: 32x32, no maxpool
            base = 16
            self.stem = nn.Sequential(
                conv3x3(3, base),
                nn.BatchNorm2d(base),
                nn.ReLU(),
            )
        self.in_ch = base
                #   ("basic",      out_ch,         n_blocks, stride)   # example spec
                #   ("bottleneck", mid_ch, out_ch, n_blocks, stride)   # example spec
        self.stages = nn.Sequential(*[self._make_stage(spec) for spec in stages])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_ch, num_classes)
        self._init_weights()

    def _make_stage(self, spec):
        """Stack one resolution's blocks. Only the first block downsamples
        (uses `stride`) and changes the channel count; the rest keep
        dimensions, so their shortcut is a plain identity. `self.in_ch` chains
        each block's input to the previous block's output."""
        kind = spec[0]
        n_blocks, stride = spec[-2], spec[-1]
        blocks = []
        for i in range(n_blocks):
            s = stride if i == 0 else 1
            if kind == "basic":
                out_ch = spec[1]
                blocks.append(BasicBlock(self.in_ch, out_ch, s,
                                         self.projection, self.residual))
            else:
                mid_ch, out_ch = spec[1], spec[2]
                blocks.append(Bottleneck(self.in_ch, mid_ch, out_ch, s,
                                         self.projection, self.residual))
            self.in_ch = out_ch
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# --- CIFAR models (Sec. 4.2): 3 stages of {16, 32, 64}, depth 6n+2 --------

def _cifar(n, num_classes, residual):
    stages = [
        ("basic", 16, n, 1),     # 32x32, no downsample
        ("basic", 32, n, 2),     # -> 16x16
        ("basic", 64, n, 2),     # -> 8x8
    ]
    return ResNet("cifar", stages, num_classes, residual=residual)


def resnet20(num_classes=10):   return _cifar(3,  num_classes, residual=True)
def resnet32(num_classes=10):   return _cifar(5,  num_classes, residual=True)
def resnet44(num_classes=10):   return _cifar(7,  num_classes, residual=True)
def resnet56(num_classes=10):   return _cifar(9,  num_classes, residual=True)
def resnet110(num_classes=10):  return _cifar(18, num_classes, residual=True)
def resnet1202(num_classes=10): return _cifar(200, num_classes, residual=True)

def plain20(num_classes=10):    return _cifar(3,  num_classes, residual=False)
def plain32(num_classes=10):    return _cifar(5,  num_classes, residual=False)
def plain44(num_classes=10):    return _cifar(7,  num_classes, residual=False)
def plain56(num_classes=10):    return _cifar(9,  num_classes, residual=False)
def plain110(num_classes=10):   return _cifar(18, num_classes, residual=False)


# --- ImageNet models (Table 1) --------------------------------------------

def resnet18(num_classes=1000, stem="imagenet"):
    stages = [("basic", 64, 2, 1), ("basic", 128, 2, 2),
              ("basic", 256, 2, 2), ("basic", 512, 2, 2)]
    return ResNet(stem, stages, num_classes)


def resnet34(num_classes=1000, stem="imagenet"):
    stages = [("basic", 64, 3, 1), ("basic", 128, 4, 2),
              ("basic", 256, 6, 2), ("basic", 512, 3, 2)]
    return ResNet(stem, stages, num_classes)


def resnet50(num_classes=1000, stem="imagenet"):
    # ("bottleneck", mid_ch, out_ch, n_blocks, stride) — out is 4x mid (Table 1)
    stages = [("bottleneck", 64, 256, 3, 1), ("bottleneck", 128, 512, 4, 2),
              ("bottleneck", 256, 1024, 6, 2), ("bottleneck", 512, 2048, 3, 2)]
    return ResNet(stem, stages, num_classes)


def resnet101(num_classes=1000, stem="imagenet"):
    stages = [("bottleneck", 64, 256, 3, 1), ("bottleneck", 128, 512, 4, 2),
              ("bottleneck", 256, 1024, 23, 2), ("bottleneck", 512, 2048, 3, 2)]
    return ResNet(stem, stages, num_classes)


def resnet152(num_classes=1000, stem="imagenet"):
    stages = [("bottleneck", 64, 256, 3, 1), ("bottleneck", 128, 512, 8, 2),
              ("bottleneck", 256, 1024, 36, 2), ("bottleneck", 512, 2048, 3, 2)]
    return ResNet(stem, stages, num_classes)


IMAGENET_MODELS = {"resnet18": resnet18, "resnet34": resnet34,
                   "resnet50": resnet50, "resnet101": resnet101,
                   "resnet152": resnet152}
CIFAR_MODELS = {"resnet20": resnet20, "resnet32": resnet32,
                "resnet44": resnet44, "resnet56": resnet56,
                "resnet110": resnet110, "resnet1202": resnet1202}
PLAIN_MODELS = {"plain20": plain20, "plain32": plain32, "plain44": plain44,
                "plain56": plain56, "plain110": plain110}
MODELS = {**IMAGENET_MODELS, **CIFAR_MODELS, **PLAIN_MODELS}


def build_model(name, num_classes=10, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown model {name}, choose from {list(MODELS)}")
    return MODELS[name](num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    for name in MODELS:
        kwargs = {"stem": "cifar"} if name in IMAGENET_MODELS else {}
        m = build_model(name, num_classes=10, **kwargs)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{name:10s}  out={tuple(m(x).shape)}  params={params:.2f}M")
