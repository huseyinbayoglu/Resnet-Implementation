# ResNet (PyTorch) — Paper-Faithful Implementation

Implementation of *Deep Residual Learning for Image Recognition*
(He et al., 2015) — ResNet-18 / 34 / 50 / 101 / 152 — with a CIFAR-10
training script.

## What's in the model
- `BasicBlock` (2× 3×3) and `Bottleneck` (1×1 → 3×3 → 1×1)
- Option B projection shortcut (1×1 conv) only when dims change; otherwise identity
- Kaiming/He normal init for conv & linear weights, BN γ=1, β=0
- Two stems: `imagenet` (7×7 s2 + maxpool) and `cifar` (3×3 s1, no pool)

## Run on Colab

```bash
!git clone <your-repo-url> resnet
%cd resnet
!pip install -q -r requirements.txt
!python train.py --model resnet34 --epochs 100 --batch-size 128
```

GPU runtime önerilir (Runtime → Change runtime type → GPU).

## Useful flags

| flag | default | notes |
|------|---------|-------|
| `--model` | `resnet34` | `resnet18/34/50/101/152` |
| `--epochs` | `100` | |
| `--batch-size` | `128` | paper kullanıyor |
| `--lr` | `0.1` | SGD, momentum 0.9, wd 1e-4 |
| `--milestones` | `50 75` | LR /= 10 |
| `--warmup-epochs` | `1` | linear warmup |
| `--stem` | `cifar` | 32×32 için |

Sanity check:

```bash
python model.py
```

Tüm modellerin parametre sayısını ve forward shape'ini basar.
