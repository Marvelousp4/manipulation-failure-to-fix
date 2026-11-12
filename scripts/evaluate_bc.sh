#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
f2f evaluate --env PickCube-v1 --policy official-bc \
  --checkpoint artifacts/bc-pickcube.pt \
  --provenance artifacts/checkpoint-provenance.json \
  --upstream "${F2F_UPSTREAM:-.cache/ManiSkill}" \
  --device cuda --sim-backend cpu --render-backend cpu \
  --control-mode pd_ee_delta_pos --episodes 50 --seed 10000 --max-steps 100 \
  --obs-noise 0.01 --action-delay 2 --video-episodes 1 --output runs/pickcube-v02
