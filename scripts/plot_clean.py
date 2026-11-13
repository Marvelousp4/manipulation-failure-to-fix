"""Plot recorded BC validation and offline errors; never infer missing results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--validation", default="runs/clean-review/bc-validation")
p.add_argument("--output", default="artifacts/clean-review/bc-learning-curve.png")
a = p.parse_args()
rows = [
    json.loads(f.read_text())["summary"] for f in sorted(Path(a.validation).glob("[0-9]*.json"))
]
x = [r["updates"] / 1000 for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
for key, label in [("success_once", "Success once"), ("success_at_end", "Success at step 200")]:
    axes[0].plot(x, [100 * r[key] for r in rows], "-o", label=label)
axes[0].set(xlabel="BC updates (thousands)", ylabel="Validation success (%)", ylim=(0, 100))
for key, label in [
    ("training_mse", "Training demonstrations"),
    ("heldout_mse", "Held-out references"),
]:
    axes[1].semilogy(x, [r[key]["mse"] for r in rows], "-o", label=label)
axes[1].set(xlabel="BC updates (thousands)", ylabel="Offline action MSE")
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.savefig(a.output, dpi=180)
