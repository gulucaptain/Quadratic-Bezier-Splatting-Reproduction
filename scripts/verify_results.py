"""Verify published paired statistics against raw completed-run records."""

import json
from pathlib import Path

import numpy as np

root = Path("results/pilot")
runs = {}
for path in root.glob("*/metrics.json"):
    config = json.loads((path.parent / "config.json").read_text())
    metrics = json.loads(path.read_text())
    key = (Path(config["image"]).stem, config["seed"], config["audit"])
    if key in runs:
        raise ValueError(f"Duplicate run: {key}")
    trace = [json.loads(line) for line in (path.parent / "trace.jsonl").read_text().splitlines()]
    assert trace[-1]["step"] == config["steps"] == 500
    np.testing.assert_allclose(metrics["best_psnr"], max(row["psnr"] for row in trace), atol=1e-9)
    assert metrics["lpips"] is None
    runs[key] = metrics
assert len(runs) == 18
summary = json.loads(Path("results/pilot_summary.json").read_text())
for mode in ["none", "full"]:
    for source, dest in [("best_psnr", "psnr"), ("ssim", "ssim"), ("seconds", "seconds")]:
        values = [m[source] for key, m in runs.items() if key[2] == mode]
        np.testing.assert_allclose(np.mean(values), summary[mode][dest]["mean"], atol=1e-10)
deltas = [
    runs[(im, seed, "full")]["best_psnr"] - runs[(im, seed, "none")]["best_psnr"]
    for im in ["A0", "L0", "P0"]
    for seed in [0, 1, 2]
]
np.testing.assert_allclose(np.mean(deltas), summary["mean_delta_psnr"], atol=1e-10)
assert sum(d > 0 for d in deltas) == summary["positive_pairs"] == 8
assert sum(d < 0 for d in deltas) == summary["negative_pairs"] == 1
regenerated = Path("runs/report/pilot_summary.json")
if regenerated.exists():
    assert json.loads(regenerated.read_text()) == summary
print("Verified 18 runs, 9 pairs, and published summary values.")
