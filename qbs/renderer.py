"""Equations (1),(2),(6)-(10), Appendix A.1. Black background, front first.

Closest-root selection is detached; the envelope theorem gives the distance
gradient by differentiating B(t*) at fixed t*, away from ties/degeneracies.
Tile bounding boxes are conservative convex-hull bounds expanded by half-width.
"""

import math

import torch
from torch.utils.checkpoint import checkpoint


@torch.no_grad()
def closest_parameter(control, pixels, eps=1e-6):
    p0, p1, p2 = control.unbind(1)
    u, v = p0 - 2 * p1 + p2, p1 - p0
    w = p0[:, None] - pixels[None]
    a = (u * u).sum(-1)[:, None]
    b = 3 * (u * v).sum(-1)[:, None]
    c = 2 * (v * v).sum(-1)[:, None] + (u[:, None] * w).sum(-1)
    d = (v[:, None] * w).sum(-1)
    aa = a.clamp_min(eps)
    shift = b / (3 * aa)
    p = c / aa - b * b / (3 * aa * aa)
    q = 2 * b**3 / (27 * aa**3) - b * c / (3 * aa * aa) + d / aa
    disc = (q / 2) ** 2 + (p / 3) ** 3
    sqrt_disc = disc.clamp_min(0).sqrt()

    def cbrt(x):
        return x.sign() * x.abs().pow(1 / 3)

    one = cbrt(-q / 2 + sqrt_disc) + cbrt(-q / 2 - sqrt_disc) - shift
    radius = (-p / 3).clamp_min(0).sqrt()
    angle = torch.acos((-q / (2 * radius**3).clamp_min(1e-30)).clamp(-1, 1))
    roots = torch.stack(
        [2 * radius * torch.cos((angle + 2 * math.pi * k) / 3) - shift for k in range(3)], -1
    )
    roots = torch.where((disc > 0)[..., None], one[..., None], roots)
    # Clipped out-of-domain roots merely repeat an endpoint, already a candidate.
    roots = roots.clamp(0, 1)
    t = torch.cat([torch.zeros_like(d)[..., None], torch.ones_like(d)[..., None], roots], -1)
    delta = (
        w[:, :, None] + 2 * v[:, None, None] * t[..., None] + u[:, None, None] * t[..., None] ** 2
    )
    chosen = (delta * delta).sum(-1).argmin(-1, keepdim=True)
    best = t.gather(-1, chosen).squeeze(-1)
    seg = p2 - p0
    linear = (-w * seg[:, None]).sum(-1) / (seg * seg).sum(-1)[:, None].clamp_min(eps)
    return torch.where(a <= eps, linear.clamp(0, 1), best)


def distance(control, pixels, primitive="quadratic"):
    if primitive == "line":
        control = torch.stack(
            [control[:, 0], (control[:, 0] + control[:, 2]) / 2, control[:, 2]], 1
        )
    else:
        # Appendix A.1 explicitly reduces near-degenerate curves to segments,
        # not just their root solver. This also removes the P1 gradient there.
        linear = torch.stack([control[:, 0], (control[:, 0] + control[:, 2]) / 2, control[:, 2]], 1)
        degenerate = (control[:, 0] - 2 * control[:, 1] + control[:, 2]).detach().square().sum(
            -1
        ) <= 1e-6
        control = torch.where(degenerate[:, None, None], linear, control)
    t = closest_parameter(control, pixels)
    p0, p1, p2 = control.unbind(1)
    curve = (
        (1 - t[..., None]) ** 2 * p0[:, None]
        + 2 * (1 - t[..., None]) * t[..., None] * p1[:, None]
        + t[..., None] ** 2 * p2[:, None]
    )
    return torch.linalg.vector_norm(curve - pixels[None], dim=-1)


def contributions(strokes, pixels, tau, primitive="quadratic"):
    d = distance(strokes[:, :6].reshape(-1, 3, 2), pixels, primitive)
    width = strokes[:, 9:10].clamp_min(1e-6)
    kernel = (
        (torch.sigmoid((width - d) / (tau * width)) - 0.5) / (1 / (1 + math.exp(-1 / tau)) - 0.5)
    ).clamp_min(0)
    alpha = strokes[:, 10:11] * kernel
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1 - alpha], 0), 0)[:-1]
    return alpha * trans


def render(
    strokes,
    height,
    width,
    tau=0.3,
    tile=32,
    primitive="quadratic",
    recompute=False,
    target=None,
    lambda_bad=1.0,
):
    """Return HWC image and optional per-stroke audit arrays. 11 values/stroke:
    P0xy,P1xy,P2xy,RGB,half_width,opacity. Order is front-to-back.
    Audit is for diagnostics/scheduling (call under no_grad), not a loss.
    """
    device, dtype = strokes.device, strokes.dtype
    xy = strokes[:, :6].detach().reshape(-1, 3, 2)
    lo = xy.amin(1) - strokes[:, 9:10].detach()
    hi = xy.amax(1) + strokes[:, 9:10].detach()
    stats = torch.zeros(len(strokes), 4, device=device, dtype=dtype) if target is not None else None
    rows = []
    for y in range(0, height, tile):
        tiles = []
        for x in range(0, width, tile):
            h, w = min(tile, height - y), min(tile, width - x)
            ys = (torch.arange(y, y + h, device=device, dtype=dtype) + 0.5) / height
            xs = (torch.arange(x, x + w, device=device, dtype=dtype) + 0.5) / width
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            pixels = torch.stack([gx, gy], -1).reshape(-1, 2)
            mask = (
                (hi[:, 0] >= (x + 0.5) / width)
                & (lo[:, 0] <= (x + w - 0.5) / width)
                & (hi[:, 1] >= (y + 0.5) / height)
                & (lo[:, 1] <= (y + h - 0.5) / height)
            )
            ids = mask.nonzero().flatten()
            if not len(ids):
                tiles.append(torch.zeros(h, w, 3, device=device, dtype=dtype) + strokes.sum() * 0)
                continue
            local = strokes[ids]

            def raster(s, pix):
                phi = contributions(s, pix, tau, primitive)
                return phi.T @ s[:, 6:9]

            if recompute and torch.is_grad_enabled():
                canvas = checkpoint(raster, local, pixels, use_reentrant=False)
            else:
                canvas = raster(local, pixels)
            tiles.append(canvas.reshape(h, w, 3))
            if target is not None:
                phi = contributions(local, pixels, tau, primitive)
                partial = (phi[..., None] * local[:, None, 6:9]).cumsum(0)
                before = torch.cat([torch.zeros_like(partial[:1]), partial[:-1]], 0)
                gt = target[y : y + h, x : x + w].reshape(-1, 3)
                improvement = ((before - gt) ** 2).sum(-1) - ((partial - gt) ** 2).sum(-1)
                values = torch.stack(
                    [
                        phi.sum(-1),
                        improvement.clamp_min(0).sum(-1),
                        (-improvement).clamp_min(0).sum(-1),
                        (phi * (gt - canvas).abs().sum(-1)).sum(-1),
                    ],
                    -1,
                )
                stats.index_add_(0, ids, values)
        rows.append(torch.cat(tiles, 1))
    result = torch.cat(rows, 0)
    if stats is None:
        return result
    area, good, bad, residual = stats.unbind(-1)
    return result, dict(
        area=area,
        good=good,
        bad=bad,
        purity=good / (good + bad + 1e-6),
        benefit=(good - lambda_bad * bad) / (area + 1e-6),
        residual=residual / (area + 1e-6),
    )
