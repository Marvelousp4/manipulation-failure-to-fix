# Manipulation Failure-to-Fix

Reproducible manipulation-policy failure research. Robustness experiments are paused.

Clean PickCube reference: official Diffusion Policy, selected at 20k updates, **100/100**
untouched test successes and **100/100** successes retained at step 200.
BC trained through 100k improves substantially but still has poor goal retention.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-clean.txt
pip install -e '.[dev]'
pytest -q
F2F_CLEAN_RUN=runs/clean-reproduce bash scripts/prepare_clean.sh
F2F_CLEAN_RUN=runs/clean-reproduce bash scripts/run_clean_training.sh
```

[Clean review](docs/CLEAN_BASELINE.md) · [Evidence](artifacts/clean-review/) ·
[Trained DP weights](artifacts/clean-review/dp-20000-ema.pt) · [Preserved v0.2](docs/V02.md).

Replay gate (requires prepared WSL environment):
```bash
python scripts/probe_replay_gate.py
```
[Research gate](docs/RESEARCH_GAP.md). Gate is **not yet run on ManiSkill**.

License pending IP/open-source review.
