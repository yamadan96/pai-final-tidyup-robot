#!/bin/bash
# v3 チェックポイントの評価（時間アンサンブルあり → なし の順）
set -e
cd "$(dirname "$0")"
echo "########## eval v3 (temporal ensemble 0.01) $(date +%H:%M) ##########"
./.venv/bin/python scripts/eval_policy.py \
  --checkpoint outputs/checkpoints/act_v3/last/pretrained_model \
  --episodes 18 --seed 777 \
  --camera-preset tight --image-size 160 \
  --temporal-ensemble 0.01 --goal-cue \
  --video outputs/video/eval_v3_te.mp4 \
  --json-out results/eval_act_v3_temporal_ensemble.json
echo "########## eval v3 (plain) $(date +%H:%M) ##########"
./.venv/bin/python scripts/eval_policy.py \
  --checkpoint outputs/checkpoints/act_v3/last/pretrained_model \
  --episodes 18 --seed 777 \
  --camera-preset tight --image-size 160 \
  --goal-cue \
  --video outputs/video/eval_v3_plain.mp4 \
  --json-out results/eval_act_v3_plain.json
echo "########## DONE eval v3 $(date +%H:%M) ##########"
