"""
Compare training results in logs/*.json against the paper's CIFAR-10
numbers (He et al., 2015, Table 6 — test error %).

Usage:
    python compare_results.py [--log-dir ./logs]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# Table 6, CIFAR-10 test error (%). ResNet-110: best of 5 runs is 6.43,
# mean 6.61 +/- 0.16. The paper gives no table for the plain nets (only the
# Fig. 6 curves), so plain runs show "-"; the expected trend is error
# INCREASING with depth, the opposite of the ResNets.
PAPER_ERR = {
    "resnet20": 8.75,
    "resnet32": 7.51,
    "resnet44": 7.17,
    "resnet56": 6.97,
    "resnet110": 6.43,
    "resnet1202": 7.93,
}


def sort_key(result: dict):
    name = result["model"]
    family = name.rstrip("0123456789")
    depth = int(name[len(family):] or 0)
    return family, depth, result["run_name"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="./logs")
    args = p.parse_args()

    paths = glob.glob(os.path.join(args.log_dir, "*.json"))
    if not paths:
        print(f"no result JSONs found in {args.log_dir}")
        return
    results = []
    for path in paths:
        with open(path) as f:
            results.append(json.load(f))

    header = (f"{'run':<12} {'model':<12} {'params':>7} "
              f"{'best err%':>9} {'paper err%':>10} {'diff':>6}")
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=sort_key):
        err = r["best_test_err_pct"]
        paper = PAPER_ERR.get(r["model"])
        paper_s = f"{paper:.2f}" if paper is not None else "-"
        diff_s = f"{err - paper:+.2f}" if paper is not None else "-"
        print(f"{r['run_name']:<12} {r['model']:<12} {r['params_m']:>6.2f}M "
              f"{err:>9.2f} {paper_s:>10} {diff_s:>6}")


if __name__ == "__main__":
    main()
