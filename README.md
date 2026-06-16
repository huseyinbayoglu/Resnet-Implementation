# ResNet in PyTorch

A faithful implementation of *Deep Residual Learning for Image Recognition*
(He, Zhang, Ren, Sun, 2015) with a training script for CIFAR-10. The aim is to
reproduce the architecture from the paper and its central finding: plain
networks get worse as they grow deeper, while residual networks do not.

## What is in the repository

`model.py` defines the network. It contains the two building blocks from the
paper (BasicBlock and Bottleneck), the shortcut connection that matches
dimensions when the channel count changes, and the `ResNet` class that puts
together a stem, a sequence of stages, and a classifier. Factory functions
build every standard variant.

`train.py` trains a model on CIFAR-10 with the paper's recipe and writes a log
for every epoch.

`scripts/` holds two helpers: `runpod_setup.sh` prepares a fresh GPU machine,
and `train_all.sh` trains the full set of models for the degradation comparison.

`Resnet.pdf` is the original paper, kept for reference.

## Models

ImageNet family (paper Table 1): `resnet18`, `resnet34`, `resnet50`,
`resnet101`, `resnet152`.

CIFAR family (paper Section 4.2): `resnet20`, `resnet32`, `resnet44`,
`resnet56`, `resnet110`, `resnet1202`.

Plain baselines: `plain20`, `plain32`, `plain44`, `plain56`, `plain110`. These
are the same CIFAR networks with the skip connections removed. They exist to
show the degradation problem.

## Implementation notes

BasicBlock is two 3x3 convolutions, each followed by batch normalization. The
block adds its input back before the final activation, which is the residual
idea from the paper.

Bottleneck is a 1x1, 3x3, 1x1 stack used in the deeper ImageNet networks. The
1x1 layers reduce and then restore the channel count, so the 3x3 runs on a
narrow tensor and stays cheap.

The shortcut matches the block input to its output when the dimensions change.
CIFAR networks use parameter free zero padding (option A in the paper) so a
residual network has exactly the same parameter count as its plain
counterpart. ImageNet networks use a 1x1 convolution projection (option B).
When the dimensions already match, the shortcut is the identity.

Weights are initialized with Kaiming (He) normal for convolutions, and with one
and zero for batch normalization, as described in Section 3.4.

## Setup

```bash
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically the first time you train.

## Sanity check

```bash
python model.py
```

This builds every variant on a small input, then prints the output shapes and
parameter counts. The CIFAR numbers match the paper (resnet20 has 0.27M
parameters, resnet56 has 0.85M, resnet110 has 1.7M).

## Training

The model name is a positional argument.

```bash
python train.py resnet20
python train.py plain20
python train.py resnet110 --warmup-epochs 1
python train.py resnet50 --stem cifar
```

The defaults follow the paper's CIFAR recipe: SGD with momentum 0.9 and weight
decay 1e-4, learning rate 0.1 divided by ten at 50 and 75 percent of training,
batch size 128, 164 epochs, and four pixel padding with random crop and
horizontal flip. Pass `--amp` for bf16 mixed precision on Ampere or newer GPUs.

Each run writes three files: `logs/<name>.csv` with per epoch loss and error,
`logs/<name>.json` with a summary including the best test error, and a
checkpoint at `checkpoints/<name>_best.pt`.

| flag | default | meaning |
|------|---------|---------|
| `model` | `resnet20` | positional; any CIFAR, plain, or ImageNet variant |
| `--epochs` | `164` | total epochs |
| `--milestones` | `82 123` | epochs where the learning rate drops |
| `--warmup-epochs` | `0` | linear warmup; use `1` for resnet110 and resnet1202 |
| `--amp` | off | bf16 mixed precision |
| `--stem` | `cifar` | stem for the ImageNet family models only |
| `--batch-size` | `128` | |
| `--lr` | `0.1` | |

## Reproducing the degradation experiment

```bash
bash scripts/train_all.sh
```

This trains plain and residual networks at depths 20, 56, and 110. As the depth
grows, the plain networks get worse while the residual networks stay low. That
is the paper's Figure 6 reproduced on your own machine. You can shorten a run
with `EPOCHS=80` and speed it up on an A100 with `AMP=1 NUM_WORKERS=8`.

## Expected results

Paper Table 6, CIFAR-10 test error.

| model | test error |
|-------|------------|
| resnet20 | 8.75% |
| resnet32 | 7.51% |
| resnet44 | 7.17% |
| resnet56 | 6.97% |
| resnet110 | 6.43% |
| resnet1202 | 7.93% |

The plain networks have no published table, only the Figure 6 curves. The
expected behaviour is that their error climbs with depth, the opposite of the
residual networks.

## Running on a GPU host

For a fresh machine such as a RunPod pod:

```bash
bash scripts/runpod_setup.sh
tmux new -s train
bash scripts/train_all.sh
```

## Reference

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for
Image Recognition. arXiv:1512.03385, 2015.
