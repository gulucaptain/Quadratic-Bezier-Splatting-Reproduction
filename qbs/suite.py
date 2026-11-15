"""Run a configured, matched-seed experiment suite on one device."""

import argparse
import hashlib
import json
from pathlib import Path

from .experiment import fit, parser


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    cli.add_argument("--root", type=Path, default=Path("runs/pilot"))
    cli.add_argument("--device", choices=["auto", "cpu", "mps"], default="mps")
    cli.add_argument("--force", action="store_true", help="Recompute completed runs")
    args = cli.parse_args()
    config = json.loads(args.config.read_text())
    for image in config["images"]:
        for seed in config["seeds"]:
            for mode in config["audit_modes"]:
                out = args.root / f"{Path(image).stem}_s{seed}_{mode}"
                argv = [
                    "--image",
                    image,
                    "--out",
                    str(out),
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--audit",
                    mode,
                ]
                for key, value in config["fit_options"].items():
                    flag = "--" + key.replace("_", "-")
                    if isinstance(value, bool):
                        if value:
                            argv.append(flag)
                    else:
                        argv.extend([flag, str(value)])
                if args.force:
                    argv.append("--overwrite")
                opts = parser().parse_args(argv)
                if (out / "metrics.json").exists() and not args.force:
                    saved = json.loads((out / "config.json").read_text())
                    changed = [
                        k for k, v in vars(opts).items() if k != "overwrite" and saved.get(k) != v
                    ]
                    for name in ["renderer.py", "strokes.py", "experiment.py", "model.py"]:
                        digest = hashlib.sha256(
                            (Path(__file__).parent / name).read_bytes()
                        ).hexdigest()
                        if saved.get("source_sha256", {}).get(name) != digest:
                            changed.append(name)
                    if changed:
                        raise ValueError(
                            f"Cache mismatch in {out}: {changed}. Select a new root or --force."
                        )
                    print(f"CACHED {out}", flush=True)
                    continue
                print(f"RUN {out}", flush=True)
                fit(opts)


if __name__ == "__main__":
    main()
