"""Reproducible local fitting and matched-seed ablations (not pretrained Full)."""

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from .renderer import render
from .strokes import AuditConfig, adapt, direction_field, direction_loss, templates


def save_image(tensor, path):
    Image.fromarray((tensor.detach().cpu().numpy().clip(0, 1) * 255).round().astype("uint8")).save(
        path
    )


def sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def fit(args):
    if min(args.width, args.height) < 7:
        raise ValueError("Image dimensions must be at least 7 for SSIM evaluation")
    if min(args.strokes, args.steps, args.tile, args.log_every, args.threads) < 1:
        raise ValueError(
            "Stroke count, steps, tile size, log interval and threads must be positive"
        )
    if args.lr <= 0 or args.audit_tail < 0:
        raise ValueError("Learning rate must be positive and audit tail must be nonnegative")
    if args.checkpoint and max(args.width, args.height) > 256:
        raise ValueError("Learned initialization currently supports one patch (at most 256 x 256)")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not getattr(args, "overwrite", False):
        raise FileExistsError(
            f"Output directory is not empty: {out}. Use a new directory or --overwrite."
        )
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS unavailable. Run with GPU access or explicitly select --device cpu."
        )
    image = (
        Image.open(args.image)
        .convert("RGB")
        .resize((args.width, args.height), Image.Resampling.LANCZOS)
    )
    target = torch.tensor(np.array(image) / 255, dtype=torch.float32, device=device)
    field = direction_field(target)
    s = templates(target, args.strokes, args.seed, direction=field if args.init == "edge" else None)
    # Fine strokes in front of coarse strokes; index within groups is fixed.
    s = s.flip(0)
    if args.checkpoint:
        from .model import Initializer

        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = Initializer(count=ckpt["count"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            s = model(target.permute(2, 0, 1)[None], args.seed)[0].flip(0)
    strokes = torch.nn.Parameter(s)
    optimizer = torch.optim.Adam([strokes], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    born = torch.full((len(strokes),), -1000, device=device, dtype=torch.long)
    cfg = AuditConfig(purity_min=args.purity_min, lambda_bad=args.lambda_bad)
    metadata = vars(args) | dict(
        resolved_device=device,
        torch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        input_sha256=hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
        audit_config=asdict(cfg),
        method_label="learned-initializer" if args.checkpoint else "untrained-template-initializer",
        background="black",
        ordering="fine-front, new births front",
        coordinate_metric="unit square",
    )
    metadata["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    (out / "config.json").write_text(json.dumps(metadata, indent=2))
    save_image(target, out / "target.png")
    with torch.no_grad():
        initial = render(strokes, args.height, args.width, 0.3, args.tile, primitive=args.primitive)
    save_image(initial, out / "initial.png")
    best = float("inf")
    best_s = strokes.detach().clone()
    best_step = 0
    history = []
    events = []
    start = time.perf_counter()
    for step in range(args.steps + 1):
        # Fixed final softness for all snapshot evaluation: values are comparable.
        tau = 1.0 if step < 100 else (0.6 if step < 200 else 0.3)
        if step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                pred = render(
                    strokes, args.height, args.width, 0.3, args.tile, primitive=args.primitive
                )
                mse = float((pred - target).square().mean())
                if mse < best:
                    best = mse
                    best_s = strokes.detach().clone()
                    best_step = step
            sync(device)
            row = dict(
                step=step,
                mse=mse,
                psnr=-10 * math.log10(max(mse, 1e-15)),
                strokes=len(strokes),
                elapsed=time.perf_counter() - start,
            )
            history.append(row)
            with (out / "trace.jsonl").open("a" if step else "w") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        pred = render(
            strokes,
            args.height,
            args.width,
            tau,
            args.tile,
            primitive=args.primitive,
            recompute=args.recompute,
        )
        loss = (pred - target).square().mean()
        if step >= 200 and step % 5 == 0 and args.direction_weight:
            loss = loss + args.direction_weight * direction_loss(strokes, field)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at {step}")
        loss.backward()
        if not torch.isfinite(strokes.grad).all():
            raise FloatingPointError(f"nonfinite gradient at {step}")
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            strokes[:, :9].clamp_(0, 1)
            strokes[:, 9].clamp_(0.0005, 0.5)
            strokes[:, 10].clamp_(0, 1)
        completed = step + 1
        if (
            args.audit != "none"
            and completed >= 100
            and completed % 50 == 0
            and completed <= args.steps - args.audit_tail
        ):
            with torch.no_grad():
                canvas, stats = render(
                    strokes,
                    args.height,
                    args.width,
                    tau,
                    args.tile,
                    primitive=args.primitive,
                    target=target,
                    lambda_bad=cfg.lambda_bad,
                )
                edited, born, mapping, event = adapt(
                    strokes,
                    stats,
                    target,
                    canvas,
                    field,
                    born,
                    completed,
                    args.seed + completed,
                    cfg,
                    args.audit,
                    capacity=args.strokes,
                )
            # Keep Adam moments for surviving strokes; reset only changed slots.
            old = optimizer.state[strokes]
            parameter = torch.nn.Parameter(edited)
            state = {"step": old["step"]}
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = torch.zeros_like(edited)
                for new_idx, old_idx in enumerate(mapping):
                    if old_idx >= 0:
                        state[key][new_idx] = old[key][old_idx]
            optimizer.state.pop(strokes)
            optimizer.param_groups[0]["params"] = [parameter]
            optimizer.state[parameter] = state
            strokes = parameter
            events.append(dict(step=completed, **event))
    with torch.no_grad():
        final = render(strokes, args.height, args.width, 0.3, args.tile, primitive=args.primitive)
        selected, stats = render(
            best_s,
            args.height,
            args.width,
            0.3,
            args.tile,
            primitive=args.primitive,
            target=target,
            lambda_bad=cfg.lambda_bad,
        )
    sync(device)
    save_image(final, out / "final.png")
    save_image(selected, out / "best.png")
    np.savez_compressed(
        out / "strokes.npz", best=best_s.cpu().numpy(), final=strokes.detach().cpu().numpy()
    )
    np.savez_compressed(out / "audit.npz", **{k: v.cpu().numpy() for k, v in stats.items()})
    selected_np = selected.cpu().numpy()
    target_np = target.cpu().numpy()
    metrics = dict(
        initial_psnr=history[0]["psnr"],
        best_psnr=-10 * math.log10(best),
        best_step=best_step,
        final_psnr=-10 * math.log10(float((final - target).square().mean())),
        ssim=float(structural_similarity(target_np, selected_np, data_range=1.0, channel_axis=2)),
        seconds=time.perf_counter() - start,
        final_strokes=len(strokes),
        best_strokes=len(best_s),
        audit_mean={k: float(v.mean()) for k, v in stats.items()},
        events=events,
        lpips=None,
        comparison_to_paper="not comparable: local pilot resolution, templates, dataset subset",
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)
    return metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--strokes", type=int, default=128)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tile", type=int, default=32)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--init", choices=["random", "edge"], default="random")
    p.add_argument("--audit", choices=["none", "full", "prune", "split", "respawn"], default="full")
    p.add_argument("--audit-tail", type=int, default=100, help="Paper: 500. Pilot: 100.")
    p.add_argument("--direction-weight", type=float, default=0.04)
    p.add_argument("--primitive", choices=["quadratic", "line"], default="quadratic")
    p.add_argument("--purity-min", type=float, default=0.5)
    p.add_argument("--lambda-bad", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing run")
    p.add_argument("--checkpoint")
    p.add_argument("--recompute", action="store_true")
    return p


def main():
    fit(parser().parse_args())


if __name__ == "__main__":
    main()
