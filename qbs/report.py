"""Build figures and paired statistics from saved raw outputs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/pilot"))
    parser.add_argument("--out", type=Path, default=Path("runs/report"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    root = args.root
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(root.glob("*/metrics.json")):
        config = json.loads((path.parent / "config.json").read_text())
        metrics = json.loads(path.read_text())
        records.append(
            dict(
                image=Path(config["image"]).stem,
                seed=config["seed"],
                audit=config["audit"],
                psnr=metrics["best_psnr"],
                final_psnr=metrics["final_psnr"],
                ssim=metrics["ssim"],
                seconds=metrics["seconds"],
                best_step=metrics["best_step"],
            )
        )
    if not records:
        raise ValueError(f"No completed runs found in {root}")
    with (output / "pilot_summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    paired = []
    for a in records:
        if a["audit"] != "none":
            continue
        b = next(
            (
                b
                for b in records
                if b["image"] == a["image"] and b["seed"] == a["seed"] and b["audit"] == "full"
            ),
            None,
        )
        if b:
            paired.append(
                dict(
                    image=a["image"],
                    seed=a["seed"],
                    delta_psnr=b["psnr"] - a["psnr"],
                    delta_ssim=b["ssim"] - a["ssim"],
                )
            )
    summary = dict(runs=len(records), pairs=len(paired), paired=paired)
    for mode in ["none", "full"]:
        rows = [r for r in records if r["audit"] == mode]
        summary[mode] = {
            key: dict(
                mean=float(np.mean([r[key] for r in rows])),
                sample_sd=float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else None,
            )
            for key in ["psnr", "ssim", "seconds"]
        }
    if paired:
        ds = [r["delta_psnr"] for r in paired]
        summary.update(
            mean_delta_psnr=float(np.mean(ds)),
            positive_pairs=sum(d > 0 for d in ds),
            negative_pairs=sum(d < 0 for d in ds),
        )
    (output / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    if args.no_figures:
        print(json.dumps(summary, indent=2))
        return
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for row, name in enumerate(["A0", "L0", "P0"]):
        for col, (mode, filename, label) in enumerate(
            [
                ("none", "target.png", "Target"),
                ("none", "initial.png", "Initial templates"),
                ("none", "best.png", "Optimization only"),
                ("full", "best.png", "Optimization + audit"),
            ]
        ):
            folder = root / f"{name}_s0_{mode}"
            if (folder / filename).exists():
                picture = Image.open(folder / filename)
            elif filename == "target.png" and (folder / "config.json").exists():
                cfg = json.loads((folder / "config.json").read_text())
                picture = (
                    Image.open(cfg["image"])
                    .convert("RGB")
                    .resize((cfg["width"], cfg["height"]), Image.Resampling.LANCZOS)
                )
            else:
                axes[row, col].axis("off")
                continue
            axes[row, col].imshow(picture)
            axes[row, col].axis("off")
            title = label
            if col >= 2 and (folder / "metrics.json").exists():
                m = json.loads((folder / "metrics.json").read_text())
                title += f"\nPSNR {m['best_psnr']:.2f} dB"
            axes[row, col].set_title(title, fontsize=10)
    fig.suptitle(
        "Mac pilot: 128 x 128, 128 strokes, 500 steps, seed 0\nNo trained initializer; not paper-level reproduction",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, name in zip(axes, ["A0", "L0", "P0"]):
        for mode, color in [("none", "#2376aa"), ("full", "#dc7036")]:
            runs = []
            for seed in [0, 1, 2]:
                trace = root / f"{name}_s{seed}_{mode}" / "trace.jsonl"
                if trace.exists():
                    runs.append([json.loads(line) for line in trace.read_text().splitlines()])
            if runs and len({len(r) for r in runs}) == 1:
                steps = [r["step"] for r in runs[0]]
                values = np.array([[r["psnr"] for r in run] for run in runs])
                mean = values.mean(0)
                sd = values.std(0, ddof=1) if len(values) > 1 else 0
                ax.plot(
                    steps,
                    mean,
                    color=color,
                    label="With audit" if mode == "full" else "Without audit",
                )
                ax.fill_between(steps, mean - sd, mean + sd, color=color, alpha=0.15)
        ax.set_title(name)
        ax.set_xlabel("Adam steps")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Mean +/- sample SD over 3 seeds; fixed tau=0.3 evaluation")
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
