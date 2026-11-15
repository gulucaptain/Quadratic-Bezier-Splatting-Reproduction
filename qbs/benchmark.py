"""Measure Mac cost at increasing canvas/budget, with conservative local priors."""

import argparse
import json
import time
from pathlib import Path

import torch

from .experiment import sync
from .renderer import render
from .strokes import templates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/scaling.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows = []
    for size, n in [(128, 128), (256, 250), (600, 2000), (600, 5000)]:
        target = torch.full((size, size, 3), 0.5, device="mps")
        s = templates(target, n)
        # Simulate local 256px patch geometry: keep centers, scale extent/width.
        scale = min(1.0, 256 / size)
        controls = s[:, :6].reshape(-1, 3, 2)
        center = controls.mean(1, keepdim=True)
        s[:, :6] = (center + (controls - center) * scale).flatten(1)
        s[:, 9] *= scale
        s.requires_grad_()
        samples = []
        for run in range(3):
            s.grad = None
            sync("mps")
            start = time.perf_counter()
            image = render(s, size, size, tile=32, recompute=True)
            sync("mps")
            forward = time.perf_counter() - start
            start = time.perf_counter()
            image.square().mean().backward()
            sync("mps")
            backward = time.perf_counter() - start
            samples.append(
                dict(
                    forward_s=forward,
                    backward_s=backward,
                    driver_allocated_gib=torch.mps.driver_allocated_memory() / 2**30,
                )
            )
            del image
        row = dict(
            size=size,
            strokes=n,
            recompute=True,
            samples=samples,
            mean_step_seconds=sum(r["forward_s"] + r["backward_s"] for r in samples[1:]) / 2,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps(rows, indent=2))
        del s, target
        torch.mps.empty_cache()
