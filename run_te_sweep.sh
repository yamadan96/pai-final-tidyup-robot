#!/bin/bash
# 時間アンサンブル係数のスイープ（v2@100k、同一シード・同一本数で係数のみを振る）
set -e
cd "$(dirname "$0")"
for COEF in 0.001 0.005 0.02 0.05 0.1; do
  TAG="${COEF/./p}"
  echo "########## eval v2@100k te=${COEF} $(date +%H:%M) ##########"
  ./.venv/bin/python scripts/eval_policy.py \
    --checkpoint outputs/checkpoints/act_v2/last/pretrained_model \
    --episodes 18 --seed 777 \
    --camera-preset tight --image-size 160 \
    --temporal-ensemble "${COEF}" \
    --json-out "results/eval_act_v2_te_${TAG}.json"
done
echo "########## DONE te sweep $(date +%H:%M) ##########"
