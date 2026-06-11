"""
Predict random CIFAR-10 test images with a trained checkpoint.

Prints true label, predicted label and softmax confidence for each image,
plus sample accuracy, and saves an image grid (green title = correct,
red = wrong).

Usage:
    python predict.py checkpoints/resnet20_best.pt
    python predict.py checkpoints/resnet20_best.pt --num-images 16 --seed 42
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import build_model

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["model"], num_classes=10,
                        **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"loaded {ckpt['model']} (epoch {ckpt['epoch']}, "
          f"test acc {ckpt['test_acc']:.4f}) from {path}")
    return model


def save_grid(images, labels, preds, confs, path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping image grid")
        return
    n = len(images)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows))
    for ax in fig.axes:
        ax.axis("off")
    for i, ax in enumerate(fig.axes[:n]):
        ax.imshow(images[i])
        correct = preds[i] == labels[i]
        ax.set_title(f"{CLASSES[preds[i]]} {confs[i]:.0%}\n"
                     f"(true: {CLASSES[labels[i]]})",
                     fontsize=9, color="green" if correct else "red")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"image grid saved to {path}")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Predict random CIFAR-10 "
                                            "test images")
    p.add_argument("checkpoint", help="path to a *_best.pt from train.py")
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--grid-path", default="predictions.png",
                   help="output image grid ('' to skip)")
    p.add_argument("--seed", type=int, default=None,
                   help="fix which images are sampled")
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = load_checkpoint(args.checkpoint, device)

    # No transform here: keep PIL images for display, normalize separately.
    test_set = datasets.CIFAR10(args.data_dir, train=False, download=True)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(test_set)), args.num_images)
    images = [test_set[i][0] for i in indices]
    labels = [test_set[i][1] for i in indices]

    x = torch.stack([normalize(img) for img in images]).to(device)
    probs = F.softmax(model(x), dim=1)
    confs, preds = probs.max(dim=1)
    confs, preds = confs.tolist(), preds.tolist()

    print(f"\n{'idx':>5}  {'true':<11} {'pred':<11} {'conf':>6}")
    print("-" * 38)
    correct = 0
    for idx, label, pred, conf in zip(indices, labels, preds, confs):
        ok = pred == label
        correct += ok
        print(f"{idx:>5}  {CLASSES[label]:<11} {CLASSES[pred]:<11} "
              f"{conf:>6.1%}  {'OK' if ok else 'X'}")
    print(f"\nsample accuracy: {correct}/{args.num_images}")

    if args.grid_path:
        save_grid(images, labels, preds, confs, args.grid_path)


if __name__ == "__main__":
    main()
