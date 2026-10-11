#!/bin/bash
# Secondary official-architecture experiment (U-TAE fine-tuning, official split).
# Runs sequentially (memory), resumable per variant. Order = priority (main comparison first).
cd "$(dirname "$0")/.."
for v in B C A E D; do
  [ -f results/predictions/pixel_utae/${v}_seed0.json ] || python3 -m src.experiments.pixel_utae_finetune --variant $v --seed 0 --epochs 6
done
echo PIXEL ALL DONE
