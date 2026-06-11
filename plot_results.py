"""
Plot training curves from train.py's CSV logs.

Single CSV  -> detailed 2-panel figure: error curves + loss curves, with
               LR drops marked and the best test error annotated.
Multiple    -> overlay of test error curves (paper Fig. 6 style), solid
               lines for resnets, dashed for plain nets.

Usage:
    python plot_results.py logs/resnet20.csv
    python plot_results.py logs/*.csv -o comparison.png
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt

PAPER_BEST = {"resnet20": 8.75, "resnet32": 7.51, "resnet44": 7.17,
              "resnet56": 6.97, "resnet110": 6.43, "resnet1202": 7.93}


def read_log(path: str) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for key, val in row.items():
                cols.setdefault(key, []).append(float(val))
    return cols


def run_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def lr_drop_epochs(log: dict[str, list[float]]) -> list[float]:
    lrs, epochs = log["lr"], log["epoch"]
    return [epochs[i] for i in range(1, len(lrs)) if lrs[i] < lrs[i - 1]]


def mark_lr_drops(ax, log):
    for i, e in enumerate(lr_drop_epochs(log)):
        ax.axvline(e, color="gray", ls=":", lw=1,
                   label="LR ÷10" if i == 0 else None)


def plot_single(log: dict[str, list[float]], name: str, out: str):
    fig, (ax_err, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = log["epoch"]
    train_err = [(1 - a) * 100 for a in log["train_acc"]]

    ax_err.plot(epochs, train_err, color="tab:blue", lw=1.2,
                alpha=0.7, label="train error")
    ax_err.plot(epochs, log["test_err"], color="tab:red", lw=1.6,
                label="test error")
    best_i = min(range(len(log["test_err"])), key=log["test_err"].__getitem__)
    best_e, best = epochs[best_i], log["test_err"][best_i]
    ax_err.scatter([best_e], [best], color="tab:red", zorder=5, s=25)
    ax_err.annotate(f"best {best:.2f}%", (best_e, best),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)
    if name in PAPER_BEST:
        ax_err.axhline(PAPER_BEST[name], color="green", ls="--", lw=1,
                       label=f"paper {PAPER_BEST[name]:.2f}%")
    mark_lr_drops(ax_err, log)
    ax_err.set_xlabel("epoch")
    ax_err.set_ylabel("error (%)")
    ax_err.set_ylim(bottom=0)
    ax_err.set_title(f"{name} — error")
    ax_err.legend()
    ax_err.grid(alpha=0.3)

    ax_loss.plot(epochs, log["train_loss"], color="tab:blue", lw=1.2,
                 alpha=0.7, label="train loss")
    ax_loss.plot(epochs, log["test_loss"], color="tab:red", lw=1.6,
                 label="test loss")
    mark_lr_drops(ax_loss, log)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title(f"{name} — loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}  (best test error {best:.2f}% @ epoch {best_e:.0f})")


def plot_overlay(logs: dict[str, dict[str, list[float]]], out: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, log in sorted(logs.items()):
        ls = "--" if name.startswith("plain") else "-"
        line, = ax.plot(log["epoch"], log["test_err"], ls=ls, lw=1.5,
                        label=f"{name} ({min(log['test_err']):.2f}%)")
        if name in PAPER_BEST:
            ax.axhline(PAPER_BEST[name], color=line.get_color(),
                       ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test error (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("CIFAR-10 test error (dotted = paper Table 6)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser(description="Plot train.py CSV logs")
    p.add_argument("csv", nargs="+", help="one or more CSV logs")
    p.add_argument("-o", "--out", default=None,
                   help="output PNG (default: <run>_curves.png or "
                        "comparison.png)")
    args = p.parse_args()

    logs = {run_name(path): read_log(path) for path in args.csv}
    if len(logs) == 1:
        name, log = next(iter(logs.items()))
        plot_single(log, name, args.out or f"{name}_curves.png")
    else:
        plot_overlay(logs, args.out or "comparison.png")


if __name__ == "__main__":
    main()
