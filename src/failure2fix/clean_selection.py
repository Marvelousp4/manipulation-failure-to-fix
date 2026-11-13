"""Select by fixed validation success, never by offline loss or test outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from failure2fix.clean import sha256


def candidate(path, method, checkpoints):
    data = json.loads(Path(path).read_text())
    summary, episodes = data["summary"], data["episodes"]
    if len(episodes) != 50 or {r["seed"] for r in episodes} != set(range(20000, 20050)):
        raise ValueError("Checkpoint selection requires exactly the reserved validation seeds")
    for key in ["success_once", "success_at_end"]:
        measured = sum(bool(r[key]) for r in episodes) / 50
        if abs(summary[key] - measured) > 1e-10:
            raise ValueError("Validation summary does not match its episodes")
    checkpoint = Path(checkpoints) / f"{summary['updates']}.pt"
    return dict(
        method=method,
        updates=summary["updates"],
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        validation=summary,
    )


def best_candidate(rows):
    if not rows:
        raise ValueError("No validated checkpoints")
    return max(
        rows,
        key=lambda r: (
            r["validation"]["success_once"],
            r["validation"]["success_at_end"],
            -r["updates"],
        ),
    )
