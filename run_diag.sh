#!/bin/bash
# v2 / v3 の把持対象診断を 18 本ずつ、同じシードで回して比較する
set -e
cd "$(dirname "$0")"
echo "########## diag v3 $(date +%H:%M) ##########"
./.venv/bin/python scripts/diagnose_grasp_target.py \
  --checkpoint outputs/checkpoints/act_v3/last/pretrained_model \
  --episodes 18 --seed 777 --image-size 160 --camera-preset tight \
  --temporal-ensemble 0.01 --goal-cue \
  --json-out results/diagnose_grasp_target_v3.json
echo "########## diag v2 (18本に揃えて再測定) $(date +%H:%M) ##########"
./.venv/bin/python scripts/diagnose_grasp_target.py \
  --checkpoint outputs/checkpoints/act_v2/100000/pretrained_model \
  --episodes 18 --seed 777 --image-size 160 --camera-preset tight \
  --temporal-ensemble 0.01 \
  --json-out results/diagnose_grasp_target_v2_18.json
echo "########## DONE diag $(date +%H:%M) ##########"
