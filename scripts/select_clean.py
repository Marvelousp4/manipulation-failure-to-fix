"""Freeze the clean policy before the one-shot test evaluation."""

import argparse
from pathlib import Path

from failure2fix.clean import write_json
from failure2fix.clean_selection import best_candidate, candidate

p = argparse.ArgumentParser()
p.add_argument("--bc-validation", default="runs/clean-review/bc-validation")
p.add_argument("--bc-checkpoints", default="runs/clean-bc-100k/checkpoints")
p.add_argument("--dp-validation", default="runs/clean-review/dp-validation")
p.add_argument("--dp-checkpoints", default="runs/clean-dp-30k/checkpoints")
p.add_argument("--output", default="artifacts/clean-review/selection.json")
a = p.parse_args()
if Path(a.output).exists():
    raise FileExistsError("Selection already frozen")
rows = []
for update in range(10000, 100001, 10000):
    rows.append(candidate(Path(a.bc_validation) / f"{update:06d}.json", "bc", a.bc_checkpoints))
bc = best_candidate(rows)
if bc["validation"]["success_once"] < 0.8 or bc["validation"]["success_at_end"] < 0.7:
    for update in [10000, 20000, 30000]:
        rows.append(candidate(Path(a.dp_validation) / f"{update:06d}.json", "dp", a.dp_checkpoints))
selected = best_candidate(rows)
write_json(
    a.output,
    dict(
        selected=selected,
        candidates=rows,
        selection_rule="validation success_once, then success_at_end, then earliest update",
        test_seeds=list(range(40000, 40100)),
        horizon=200,
        credibility_targets=dict(success_once=0.8, success_at_end=0.7),
    ),
)
print(selected)
