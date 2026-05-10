"""
CIFAR-10 training script for the ResNet implementation in model.py.

Follows the paper's CIFAR recipe (Sec. 4.2) where applicable:
- SGD with momentum=0.9, weight_decay=1e-4
- LR 0.1, divided by 10 at the standard milestones
- Per-pixel mean subtraction + 4-pixel pad random crop + horizontal flip

Usage (Colab):
    !git clone <repo>
    %cd <repo>
    !pip install -q torch torchvision
    !python train.py --model resnet34 --epochs 100 --batch-size 128
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_loaders(data_dir: str, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # paper: 4-pixel pad + crop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optimizer, criterion, device, epoch: int,
                    log_interval: int = 100):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

        if (i + 1) % log_interval == 0:
            print(f"  epoch {epoch} step {i+1}/{len(loader)}  "
                  f"loss={running_loss/total:.4f}  acc={correct/total:.4f}")
    dt = time.time() - t0
    return running_loss / total, correct / total, dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet34",
                   choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[50, 75],
                   help="LR is divided by 10 at these epochs (paper: ~32k/48k iters).")
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=1,
                   help="Linear LR warmup (paper Sec. 4.2 uses 0.01 warmup for ResNet-110).")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", default="./checkpoints")
    p.add_argument("--stem", default="cifar", choices=["cifar", "imagenet"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers)

    model = build_model(args.model, num_classes=10, stem=args.stem).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {args.model}  params={n_params:.2f}M  stem={args.stem}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.milestones,
                                               gamma=args.gamma)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Linear warmup over first `warmup_epochs`
        if epoch <= args.warmup_epochs:
            warm_lr = args.lr * epoch / max(1, args.warmup_epochs)
            for g in optimizer.param_groups:
                g["lr"] = warm_lr

        tr_loss, tr_acc, dt = train_one_epoch(model, train_loader, optimizer,
                                              criterion, device, epoch)
        te_loss, te_acc = evaluate(model, test_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:3d}/{args.epochs}] lr={cur_lr:.4f}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  | "
              f"test loss={te_loss:.4f} acc={te_acc:.4f}  ({dt:.1f}s)")

        if epoch > args.warmup_epochs:
            scheduler.step()

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "model": args.model,
                "stem": args.stem,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "test_acc": te_acc,
            }, os.path.join(args.out_dir, f"{args.model}_best.pt"))

    print(f"best test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
