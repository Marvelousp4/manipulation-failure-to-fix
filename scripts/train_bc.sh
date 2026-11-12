#!/usr/bin/env bash
set -euo pipefail
# Run from repository root after installing requirements-sim.txt.
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
upstream="${F2F_UPSTREAM:-.cache/ManiSkill}"
revision=62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3
if [ ! -d "$upstream/.git" ]; then
  git clone https://github.com/mani-skill/ManiSkill.git "$upstream"
fi
git -C "$upstream" checkout "$revision"
patch_path="$PWD/patches/bc-batched-transfer.patch"
if ! git -C "$upstream" apply --reverse --check "$patch_path" 2>/dev/null; then
  git -C "$upstream" apply --check "$patch_path"
  git -C "$upstream" apply "$patch_path"
fi
python scripts/download_demo.py
dataset="$HOME/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
python scripts/run_official.py --module mani_skill.trajectory.replay_trajectory \
  --traj-path "$HOME/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5" \
  --use-first-env-state -c pd_ee_delta_pos -o state --save-traj --num-envs 1 -b cpu
mkdir -p artifacts
command=(python scripts/run_official.py --script "$upstream/examples/baselines/bc/bc.py" --env-id PickCube-v1
  --demo-path "$dataset" --control-mode pd_ee_delta_pos --sim-backend cpu
  --max-episode-steps 100 --total-iters 10000 --seed 1 --num-eval-envs 1
  --num-eval-episodes 10 --no-capture-video --save-freq 9999 --exp-name f2f-bc-v02)
printf '%q ' "${command[@]}" > artifacts/training-command.txt
"${command[@]}" 2>&1 | tee artifacts/training.log
cp runs/f2f-bc-v02/checkpoints/9999.pt artifacts/bc-pickcube.pt
python scripts/provenance.py --dataset "$dataset" --upstream "$upstream" \
  --checkpoint artifacts/bc-pickcube.pt
