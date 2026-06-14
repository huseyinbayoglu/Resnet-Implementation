"""
CIFAR-10 training script for the ResNet implementation in model.py.

Follows the paper's CIFAR-10 recipe (Sec. 4.2):
- SGD, momentum 0.9, weight decay 1e-4, batch size 128
- LR 0.1, divided by 10 at 32k and 48k iterations, stop at 64k
  (with batch 128 that is ~epochs 82 / 123, 164 total)
- Per-image augmentation: 4-pixel pad + 32x32 random crop + horizontal flip
- For resnet110/1202 the paper warms up with LR 0.01 until training error
  drops; use --warmup-epochs 1 to approximate this.

Usage:
    python train.py resnet20
    python train.py plain20                       # non-residual baseline
    python train.py resnet110 --warmup-epochs 1
    python train.py resnet34 --stem cifar         # ImageNet net on CIFAR
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import IMAGENET_MODELS, MODELS, build_model

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_loaders(data_dir: str, batch_size: int,
                num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True, download=True,
                                 transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optimizer, criterion, device, epoch: int,
                    amp: bool = False,
                    log_interval: int = 100) -> tuple[float, float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # bf16 autocast: faster on Ampere+ GPUs (A100), no loss scaling needed.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

        if (i + 1) % log_interval == 0:
            print(f"  epoch {epoch} step {i + 1}/{len(loader)}  "
                  f"loss={running_loss / total:.4f}  acc={correct / total:.4f}")
    return running_loss / total, correct / total, time.time() - t0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
    p.add_argument("model", nargs="?", default="resnet20",
                   choices=sorted(MODELS),
                   help="resnet20/32/44/56/110 (CIFAR), plain20/.../110 "
                        "(non-residual baseline), or resnet18/34/50/101/152 "
                        "(ImageNet family). Default: resnet20")
    p.add_argument("--stem", default="cifar", choices=["cifar", "imagenet"],
                   help="Stem for ImageNet-family models (resnet18/34/...). "
                        "CIFAR-family models always use the cifar stem.")
    p.add_argument("--epochs", type=int, default=164,
                   help="Paper: 64k iterations at batch 128 ~ 164 epochs.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[82, 123],
                   help="LR is divided by 10 at these epochs "
                        "(paper: 32k/48k iterations).")
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear LR warmup; paper uses a 0.01 warmup for "
                        "resnet110/1202.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true",
                   help="bf16 mixed precision (faster on A100 / Ampere+).")
    p.add_argument("--out-dir", default="./checkpoints")
    p.add_argument("--log-dir", default="./logs",
                   help="Per-epoch CSV and final summary JSON go here.")
    p.add_argument("--run-name", default=None,
                   help="Log/checkpoint file prefix (default: model name).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or args.model
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size,
                                            args.num_workers)

    model_kwargs = {"stem": args.stem} if args.model in IMAGENET_MODELS else {}
    model = build_model(args.model, num_classes=10, **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {args.model}  params={n_params:.2f}M  {model_kwargs}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.milestones,
                                               gamma=args.gamma)

    csv_path = os.path.join(args.log_dir, f"{run_name}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc",
                                "test_loss", "test_acc", "test_err",
                                "seconds"])

    best_acc, best_epoch = 0.0, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            warm_lr = args.lr * epoch / (args.warmup_epochs + 1)
            for g in optimizer.param_groups:
                g["lr"] = warm_lr
        elif epoch == args.warmup_epochs + 1:
            for g in optimizer.param_groups:
                g["lr"] = args.lr

        tr_loss, tr_acc, dt = train_one_epoch(model, train_loader, optimizer,
                                              criterion, device, epoch,
                                              amp=args.amp)
        te_loss, te_acc = evaluate(model, test_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:3d}/{args.epochs}] lr={cur_lr:.4f}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  | "
              f"test loss={te_loss:.4f} acc={te_acc:.4f}  ({dt:.1f}s)")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{cur_lr:.5f}",
                                    f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                                    f"{te_loss:.4f}", f"{te_acc:.4f}",
                                    f"{(1 - te_acc) * 100:.2f}",
                                    f"{dt:.1f}"])

        if epoch > args.warmup_epochs:
            scheduler.step()

        if te_acc > best_acc:
            best_acc, best_epoch = te_acc, epoch
            torch.save({
                "model": args.model,
                "model_kwargs": model_kwargs,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "test_acc": te_acc,
            }, os.path.join(args.out_dir, f"{run_name}_best.pt"))

    summary = {
        "run_name": run_name,
        "model": args.model,
        "model_kwargs": model_kwargs,
        "params_m": round(n_params, 3),
        "epochs": args.epochs,
        "seed": args.seed,
        "best_test_acc": round(best_acc, 4),
        "best_test_err_pct": round((1 - best_acc) * 100, 2),
        "best_epoch": best_epoch,
        "final_test_acc": round(te_acc, 4),
        "final_test_err_pct": round((1 - te_acc) * 100, 2),
        "total_hours": round((time.time() - t_start) / 3600, 2),
    }
    with open(os.path.join(args.log_dir, f"{run_name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"best test acc: {best_acc:.4f} "
          f"(err {(1 - best_acc) * 100:.2f}%, epoch {best_epoch})")


if __name__ == "__main__":
    main()
