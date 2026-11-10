# Manipulation Failure-to-Fix

Reproducible experiments on manipulation-policy failures under deployment shifts.

**Stage 1:** paired-seed clean/shifted rollouts, observation noise, action delay, JSONL/CSV logs, tests. `ToyReach` is a smoke test, **not** manipulation evidence.

## Quickstart
```bash
python -m pip install -e '.[dev]'
pytest -q
f2f evaluate --env toy --policy toy-oracle --episodes 20 --output runs/toy
```

## Next
1. Reproduce a trained ManiSkill `PickCube-v1` policy.
2. Measure failures under controlled shifts.
3. Compare repairs under equal budgets.

[Protocol](docs/PROTOCOL.md) · [Roadmap](docs/ROADMAP.md)

No scientific claims yet. License pending IP/open-source review.
