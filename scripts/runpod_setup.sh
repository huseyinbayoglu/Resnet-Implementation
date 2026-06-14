#!/usr/bin/env bash
# One-time setup inside a RunPod pod (template: "RunPod PyTorch 2.x").
# Run from the repo root:  bash scripts/runpod_setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# The RunPod PyTorch template already ships torch+torchvision with CUDA;
# this is a no-op there but makes the script work on a bare image too.
pip install -q -r requirements.txt

# tmux keeps training alive if the SSH/web terminal disconnects.
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)

python - <<'EOF'
import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", end="")
print(f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
EOF

# Sanity check: forward pass + param counts for every model variant.
python model.py

# Pre-download CIFAR-10 (~170MB) so training runs start instantly.
python - <<'EOF'
from torchvision import datasets
datasets.CIFAR10("./data", train=True, download=True)
datasets.CIFAR10("./data", train=False, download=True)
print("CIFAR-10 ready in ./data")
EOF

echo
echo "Setup OK. Start training with:"
echo "  tmux new -s train"
echo "  bash scripts/train_all.sh"


