#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
root="${F2F_CLEAN_RUN:-runs/clean-reproduce}"
upstream="${F2F_CLEAN_UPSTREAM:-.cache/clean-ManiSkill}"
prefix="${F2F_CLEAN_PREFIX:-clean-reproduce}"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required for this training budget"'
dataset="$root/data/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
reference="$root/reference/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
python scripts/train_clean_bc.py --upstream "$upstream" --dataset "$dataset" \
  --reference "$reference" --audit "$root/corrected-replay/replay.json" \
  --output "$root/bc-validation" --run-name "$prefix-bc" --updates 100000 \
  > "$root/bc-training.log" 2>&1
if python - "$root/bc-validation" <<'PY'
import json,sys
from pathlib import Path
rows=[json.loads(p.read_text())['summary'] for p in Path(sys.argv[1]).glob('[0-9]*.json')]
best=max(rows,key=lambda s:(s['success_once'],s['success_at_end'],-s['updates']))
sys.exit(0 if best['success_once']<.8 or best['success_at_end']<.7 else 1)
PY
then
  patch_path="$PWD/patches/dp-clean-training.patch"
  if ! git -C "$upstream" apply --reverse --check "$patch_path" 2>/dev/null; then
    git -C "$upstream" apply --check "$patch_path"
    git -C "$upstream" apply "$patch_path"
  fi
  python scripts/train_clean_dp.py --upstream "$upstream" --dataset "$dataset" \
    --output "$root/dp-validation" --run-name "$prefix-dp" --updates 30000 \
    > "$root/dp-training.log" 2>&1
fi
python scripts/select_clean.py --bc-validation "$root/bc-validation" \
  --bc-checkpoints "runs/$prefix-bc/checkpoints" --dp-validation "$root/dp-validation" \
  --dp-checkpoints "runs/$prefix-dp/checkpoints" --output "$root/selection.json"
python scripts/test_clean.py --upstream "$upstream" --selection "$root/selection.json" \
  --output "$root/test" > "$root/test.log" 2>&1
