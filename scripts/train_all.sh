#!/usr/bin/env bash
# Train the paper's CIFAR-10 ResNets back to back and compare with Table 6.
# Run from the repo root (ideally inside tmux):  bash scripts/train_all.sh
#
# Override via env vars:
#   MODELS="resnet20 resnet56" bash scripts/train_all.sh
#   EPOCHS=82 bash scripts/train_all.sh        # quick half-schedule run
#   AMP=1 NUM_WORKERS=8 bash scripts/train_all.sh   # fast on A100
set -euo pipefail

cd "$(dirname "$0")/.."

# Depth comparison for the degradation problem (paper Fig. 6): plain error
# goes UP with depth, ResNet error goes DOWN. Default = 3 depths x 2 types.
MODELS=${MODELS:-"plain20 plain56 plain110 resnet20 resnet56 resnet110"}
EPOCHS=${EPOCHS:-164}
# LR drops at 50% and 75% of training (paper: 32k/48k of 64k iters), so the
# schedule scales automatically when you shorten EPOCHS for a quick run.
MILESTONES=${MILESTONES:-"$((EPOCHS / 2)) $((EPOCHS * 3 / 4))"}
SEED=${SEED:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
AMP=${AMP:-0}

amp_flag=""
[ "$AMP" = "1" ] && amp_flag="--amp"

mkdir -p logs

for model in $MODELS; do
    # Very deep nets need a lower-LR warmup to start converging (Sec. 4.2);
    # this matters for the 110-layer plain net especially.
    warmup=0
    case "$model" in resnet110|resnet1202|plain110|plain1202) warmup=1 ;; esac

    echo "=== training $model (epochs=$EPOCHS milestones=$MILESTONES warmup=$warmup amp=$AMP) ==="
    python train.py "$model" \
        --epochs "$EPOCHS" \
        --milestones $MILESTONES \
        --warmup-epochs "$warmup" \
        --seed "$SEED" \
        --num-workers "$NUM_WORKERS" \
        $amp_flag \
        2>&1 | tee "logs/${model}.train.log"
done

echo
echo "Done. Each model's best test error is in logs/<model>.json"
echo "and the per-epoch curve is in logs/<model>.csv"
