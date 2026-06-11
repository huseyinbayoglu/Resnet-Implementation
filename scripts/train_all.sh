#!/usr/bin/env bash
# Train the paper's CIFAR-10 ResNets back to back and compare with Table 6.
# Run from the repo root (ideally inside tmux):  bash scripts/train_all.sh
#
# Override the model list or epochs via env vars:
#   MODELS="resnet20 resnet56" bash scripts/train_all.sh
#   EPOCHS=82 bash scripts/train_all.sh        # quick half-schedule run
set -euo pipefail

cd "$(dirname "$0")/.."

# Plain (non-residual) baselines first, then the ResNets: with depth, plain
# error goes UP while ResNet error goes DOWN (paper Fig. 6).
MODELS=${MODELS:-"plain20 plain32 plain44 plain56 resnet20 resnet32 resnet44 resnet56 resnet110"}
EPOCHS=${EPOCHS:-164}
SEED=${SEED:-0}

mkdir -p logs

for model in $MODELS; do
    # Paper warms up resnet110/1202 with a lower LR (Sec. 4.2).
    warmup=0
    case "$model" in resnet110|resnet1202) warmup=1 ;; esac

    echo "=== training $model (epochs=$EPOCHS warmup=$warmup seed=$SEED) ==="
    python train.py "$model" \
        --epochs "$EPOCHS" \
        --warmup-epochs "$warmup" \
        --seed "$SEED" \
        --num-workers 4 \
        2>&1 | tee "logs/${model}.train.log"
done

echo
echo "=== results vs paper (Table 6) ==="
python compare_results.py
