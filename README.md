# Manipulation Failure-to-Fix

v0.2: official ManiSkill state BC, paired clean/noise/delay evaluation, provenance and videos.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-sim.txt
pip install -e '.[dev]'
pytest -q
F2F_SIM_TESTS=1 pytest -q tests/test_maniskill.py
bash scripts/train_bc.sh
bash scripts/evaluate_bc.sh
```

Uses GPU training, CPU physics and CPU Vulkan rendering on WSL. Training: all 1,000 official
motion-planning demonstrations, seed 1, 10,000 updates. Evaluation: 50 paired seeds per condition.
`runs/` contains JSONL traces, metrics, configuration and MP4 videos. `artifacts/` records provenance.
The included BC patch batches GPU transfers without changing training samples.
Gymnasium 0.29.1 is required by the pinned official baseline's evaluation interface.

[Protocol](docs/PROTOCOL.md) · [v0.2 evidence](docs/V02.md).
ToyReach remains a smoke test. License pending IP/open-source review.
