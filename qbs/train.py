"""Initializer training entry point. Supply Flickr2K directory for paper training.
--smoke uses synthetic data ONLY to verify end-to-end gradients, not performance.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .experiment import save_image, sync
from .model import Initializer, model_loss
from .renderer import render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--strokes", type=int, default=250)
    p.add_argument("--render-size", type=int, default=256)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", help="Resume model/optimizer, RNG and dataset order from checkpoint")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.microbatch, args.strokes, args.render_size) < 1:
        p.error("Training sizes and epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        p.error("--max-steps must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        p.error("Output directory is nonempty; select a new directory or resume its checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        yy, xx = np.mgrid[:256, :256] / 256
        array = np.stack([xx, yy, ((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.12).astype(float)], -1)
        images = [torch.tensor(array, dtype=torch.float32).permute(2, 0, 1)]
        files = []
    else:
        if not args.data:
            p.error("--data required unless --smoke")
        files = sorted(
            f for f in Path(args.data).rglob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not files:
            p.error("No training images found")
        images = None
    model = Initializer(args.strokes, args.pretrained).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total = args.max_steps or args.epochs * math.ceil((len(files) or 1) / args.batch_size)
    gen = np.random.default_rng(args.seed)
    config = vars(args) | dict(
        parameter_count=sum(x.numel() for x in model.parameters()),
        source_files=[str(f) for f in files],
        training_policy="one random 256px crop per sampled image; cycling shuffled epochs",
        tau=1.0,
        status="synthetic diagnostic" if args.smoke else "training",
    )
    if not args.resume:
        (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    step = 0
    start = time.perf_counter()
    first_step = 0
    order = []
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        if saved["count"] != args.strokes:
            raise ValueError("Resume stroke count mismatch")
        for key in (
            "epochs",
            "batch_size",
            "microbatch",
            "render_size",
            "max_steps",
            "seed",
            "smoke",
            "data",
        ):
            if saved["config"].get(key) != config.get(key):
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first_step = saved["step"]
        order = saved["order"]
        gen.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if "mps_rng" in saved and args.device == "mps":
            torch.mps.set_rng_state(saved["mps_rng"].cpu())
    if first_step >= total:
        print("Checkpoint already reached the configured total steps")
        return
    for step in range(first_step, total):
        warm = max(1, int(0.05 * total))
        factor = min(1, (step + 1) / warm) * (
            0.5 * (1 + math.cos(math.pi * max(0, step - warm) / max(1, total - warm)))
        )
        for group in optimizer.param_groups:
            group["lr"] = 1e-4 * factor
        optimizer.zero_grad(set_to_none=True)
        value = 0.0
        for begin in range(0, args.batch_size, args.microbatch):
            batch = []
            for _ in range(min(args.microbatch, args.batch_size - begin)):
                if images is not None:
                    crop = images[0]
                else:
                    if not order:
                        order = gen.permutation(len(files)).tolist()
                    im = Image.open(files[order.pop()]).convert("RGB")
                    if min(im.size) < 256:
                        scale = 256 / min(im.size)
                        im = im.resize(
                            tuple(math.ceil(v * scale) for v in im.size), Image.Resampling.LANCZOS
                        )
                    x = int(gen.integers(im.width - 256 + 1))
                    y = int(gen.integers(im.height - 256 + 1))
                    crop = torch.tensor(
                        np.array(im.crop((x, y, x + 256, y + 256))) / 255, dtype=torch.float32
                    ).permute(2, 0, 1)
                batch.append(crop)
            target = torch.stack(batch).to(args.device)
            strokes = model(target, args.seed + step * args.batch_size + begin)
            pred = torch.stack(
                [
                    render(
                        s.flip(0),
                        args.render_size,
                        args.render_size,
                        tau=1.0,
                        tile=32,
                        recompute=True,
                    ).permute(2, 0, 1)
                    for s in strokes
                ]
            )
            resized = torch.nn.functional.interpolate(
                target, (args.render_size, args.render_size), mode="bilinear", align_corners=False
            )
            loss = model_loss(pred, resized) * len(batch) / args.batch_size
            loss.backward()
            value += float(loss.detach())
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("Nonfinite model gradient")
        optimizer.step()
        sync(args.device)
        row = dict(
            step=step + 1,
            loss=value,
            elapsed=time.perf_counter() - start,
            backbone_grad=float(model.backbone.patch_embed.proj.weight.grad.norm()),
        )
        with (out / "trace.jsonl").open("a" if step else "w") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if step % 100 == 0 or step == total - 1:
            state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                step=step + 1,
                count=args.strokes,
                config=config,
                order=order,
                numpy_rng=gen.bit_generator.state,
                torch_rng=torch.get_rng_state(),
            )
            if args.device == "mps":
                state["mps_rng"] = torch.mps.get_rng_state()
            torch.save(state, out / "checkpoint.pt")
    save_image(resized[0].permute(1, 2, 0), out / "target.png")
    save_image(pred[0].permute(1, 2, 0), out / "prediction.png")


if __name__ == "__main__":
    main()
