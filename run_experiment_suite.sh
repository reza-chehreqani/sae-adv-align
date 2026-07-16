#!/usr/bin/env bash
# Runs the full pipeline: clean pretraining -> SAE pretraining -> every
# baseline and every drift-control variant of the proposed method -> full
# evaluation of every checkpoint -> a single comparison table.
#
# This is a long-running script (7-8 training runs at the default "fast
# fine-tuning" schedule, ~30 epochs each). Comment out lines for methods you
# don't want yet, or run each `python scripts/...` line by hand -- nothing
# here is special, it's just the configs in configs/ in a sensible order.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Phase 1: clean pretraining ==="
python scripts/train.py --config configs/cifar10_clean_pretrain.yaml

echo "=== Phase 2: SAE pretraining ==="
python scripts/pretrain_sae.py --config configs/cifar10_sae_pretrain.yaml

echo "=== Phase 3: baselines ==="
python scripts/train.py --config configs/cifar10_baseline_madry.yaml
python scripts/train.py --config configs/cifar10_baseline_trades.yaml
python scripts/train.py --config configs/cifar10_feature_align_raw.yaml

echo "=== Phase 4: proposed method, all four drift-control strategies ==="
python scripts/train.py --config configs/cifar10_sae_align_frozen.yaml
python scripts/train.py --config configs/cifar10_sae_align_frozen_residual.yaml
python scripts/train.py --config configs/cifar10_sae_align_joint.yaml
python scripts/train.py --config configs/cifar10_sae_align_periodic_refresh.yaml

echo "=== Phase 5 (optional): TRADES + SAE alignment, shows the two axes compose ==="
python scripts/train.py --config configs/cifar10_trades_sae_align_frozen_residual.yaml

echo "=== Phase 6: full evaluation of every checkpoint ==="
for run in baseline_madry baseline_trades feature_align_raw \
           sae_align_frozen sae_align_frozen_residual sae_align_joint sae_align_periodic_refresh \
           trades_sae_align_frozen_residual; do
    echo "--- evaluating $run ---"
    python scripts/evaluate_checkpoint.py --checkpoint "runs/$run/checkpoints/best.pt" --skip_autoattack
done

echo "=== Phase 7: comparison table ==="
python scripts/aggregate_results.py runs/* --out results_summary.csv
echo "Done. See results_summary.csv"
