"""Render saved floating-point strokes at a new resolution (no re-fitting)."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .experiment import save_image
from .renderer import render

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strokes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--snapshot", choices=["best", "final"], default="best")
    p.add_argument("--device", default="mps")
    p.add_argument("--primitive", default="quadratic", choices=["quadratic", "line"])
    args = p.parse_args()
    torch.set_num_threads(4)
    s = torch.tensor(np.load(args.strokes)[args.snapshot], device=args.device)
    with torch.no_grad():
        im = render(s, args.height, args.width, primitive=args.primitive)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(im, args.out)
