"""
Paper Fig. 6 style figure: plain nets (left) vs ResNets (right), test error
by depth. Shows the degradation problem — plain error climbs with depth while
ResNet error stays low.

Usage:
    python plot_degradation.py --log-dir results/logs -o degradation.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt


def read(path):
    epochs, test_err = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            test_err.append(float(row["test_err"]))
    return epochs, test_err


def depth(filename: str) -> int:
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else 0


def panel(ax, files, title):
    for path in sorted(files, key=lambda f: depth(os.path.basename(f))):
        name = os.path.splitext(os.path.basename(path))[0]
        epochs, test_err = read(path)
        line, = ax.plot(epochs, test_err, lw=1.8,
                        label=f"{name}  (best {min(test_err):.2f}%)")
        ax.scatter([epochs[test_err.index(min(test_err))]], [min(test_err)],
                   color=line.get_color(), s=20, zorder=5)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="results/logs")
    p.add_argument("-o", "--out", default="degradation.png")
    args = p.parse_args()

    plain = glob.glob(os.path.join(args.log_dir, "plain*.csv"))
    resnet = glob.glob(os.path.join(args.log_dir, "resnet*.csv"))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))
    panel(ax_left, plain, "Plain nets — no skip connections")
    panel(ax_right, resnet, "ResNets")
    ax_left.set_ylabel("test error (%)")
    fig.suptitle("Degradation problem: deeper plain nets get worse, "
                 "ResNets do not", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
