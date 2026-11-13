#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
root="${F2F_CLEAN_RUN:-runs/clean-review}"
upstream="${F2F_CLEAN_UPSTREAM:-.cache/clean-ManiSkill}"
revision=62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3
if [ ! -d "$upstream/.git" ]; then
  git clone https://github.com/mani-skill/ManiSkill.git "$upstream"
fi
git -C "$upstream" checkout "$revision"
for patch in bc-batched-transfer bc-clean-training; do
  patch_path="$PWD/patches/$patch.patch"
  if ! git -C "$upstream" apply --reverse --check "$patch_path" 2>/dev/null; then
    git -C "$upstream" apply --check "$patch_path"
    git -C "$upstream" apply "$patch_path"
  fi
done
if [ -e "$root/data/trajectory.h5" ]; then
  echo "Refusing to replace existing data; choose a fresh F2F_CLEAN_RUN" >&2
  exit 1
fi
python scripts/download_demo.py
mkdir -p "$root/data"
cp "$HOME/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5" "$root/data/trajectory.h5"
cp "$HOME/.maniskill/demos/PickCube-v1/motionplanning/trajectory.json" "$root/data/trajectory.json"
python scripts/run_official.py --reconfiguration-freq 1 --module mani_skill.trajectory.replay_trajectory \
  --traj-path "$root/data/trajectory.h5" --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-envs 1 -b cpu
python -m failure2fix.clean audit --upstream "$upstream" \
  --dataset "$root/data/trajectory.state.pd_ee_delta_pos.physx_cpu.h5" \
  --output "$root/corrected-replay" --episodes 50
# Isolate the official planner's native NumPy dependency from the evaluation environment.
python -m pip install --target .cache/planner-deps --upgrade --no-deps numpy==1.26.4 scipy==1.14.1
PYTHONPATH="$PWD/.cache/planner-deps" python scripts/generate_clean_reference.py \
  --output "$root/reference" --seed 11000 --episodes 50
python scripts/run_official.py --reconfiguration-freq 1 --module mani_skill.trajectory.replay_trajectory \
  --traj-path "$root/reference/PickCube-v1/motionplanning/trajectory.h5" \
  --use-first-env-state -c pd_ee_delta_pos -o state --save-traj --num-envs 1 -b cpu
