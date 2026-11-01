#!/bin/bash
# 学習ステップ数をそろえた比較のため、v2 の 60000step 時点を同条件で評価する
set -e
cd "$(dirname "$0")"
echo "########## eval v2@60k (temporal ensemble 0.01) $(date +%H:%M) ##########"
./.venv/bin/python scripts/eval_policy.py \
  --checkpoint outputs/checkpoints/act_v2/060000/pretrained_model \
  --episodes 18 --seed 777 \
  --camera-preset tight --image-size 160 \
  --temporal-ensemble 0.01 \
  --video outputs/video/eval_v2_60k_te.mp4 \
  --json-out results/eval_act_v2_60k_temporal_ensemble.json
echo "########## DONE eval v2@60k $(date +%H:%M) ##########"
