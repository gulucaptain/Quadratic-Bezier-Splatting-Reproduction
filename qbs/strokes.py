"""Templates, structure-tensor field and adaptive scheduling.

Thresholds not specified by the paper are centralized in AuditConfig.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def sample_image(image, xy):
    shape = xy.shape[:-1]
    grid = xy.reshape(1, 1, -1, 2) * 2 - 1
    out = F.grid_sample(
        image.permute(2, 0, 1)[None], grid, align_corners=False, padding_mode="border"
    )
    return out[0, :, 0].T.reshape(*shape, image.shape[-1])


def templates(target, count, seed=0, ratio=(0.2, 0.3, 0.5), centers=None, direction=None):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=gen).to(target.device)

    sizes = [int(count * ratio[0]), int(count * ratio[1])]
    sizes.append(count - sum(sizes))
    lengths = []
    widths = []
    for n, (l0, l1, w0, w1) in zip(
        sizes, [(0.24, 0.56, 0.12, 0.30), (0.08, 0.22, 0.04, 0.12), (0.008, 0.06, 0.006, 0.03)]
    ):
        lengths.append(l0 + (l1 - l0) * rand(n))
        widths.append(w0 + (w1 - w0) * rand(n))
    length = torch.cat(lengths)
    width = torch.cat(widths)
    center = rand(count, 2) if centers is None else centers
    angle = rand(count) * 2 * math.pi
    tangent = torch.stack([angle.cos(), angle.sin()], -1)
    if direction is not None:
        field = sample_image(direction, center)
        tangent = torch.where((field[:, 2] > 0.2)[:, None], field[:, :2], tangent)
    normal = torch.stack([-tangent[:, 1], tangent[:, 0]], -1)
    # Undisclosed curvature prior: uniform +/- 0.15 of curve length.
    curvature = (rand(count) - 0.5) * 0.3 * length
    control = torch.stack(
        [
            center - length[:, None] * tangent / 2,
            center + curvature[:, None] * normal,
            center + length[:, None] * tangent / 2,
        ],
        1,
    ).clamp(0, 1)
    color = sample_image(target, center)
    return torch.cat(
        [control.flatten(1), color, width[:, None], torch.full_like(width[:, None], 0.7)], -1
    )


def direction_field(target, scales=(1.0, 3.0)):
    """Multi-scale tensor directions; chosen scales/blending are assumptions."""
    gray = target.detach().cpu().numpy() @ np.array([0.299, 0.587, 0.114])
    fields = []
    for scale in scales:
        gx = gaussian_filter(gray, scale, order=(0, 1))
        gy = gaussian_filter(gray, scale, order=(1, 0))
        xx = gaussian_filter(gx * gx, 2 * scale)
        yy = gaussian_filter(gy * gy, 2 * scale)
        xy = gaussian_filter(gx * gy, 2 * scale)
        coherence = np.sqrt((xx - yy) ** 2 + 4 * xy**2) / (xx + yy + 1e-12)
        angle = 0.5 * np.arctan2(2 * xy, xx - yy) + np.pi / 2
        fields.append((angle, coherence))
    a, c = fields[0]
    b, d = fields[1]
    # Blend doubled angles to respect the sign ambiguity of tensor eigenvectors.
    vx = c * np.cos(2 * a) + (1 - c) * d * np.cos(2 * b)
    vy = c * np.sin(2 * a) + (1 - c) * d * np.sin(2 * b)
    angle = 0.5 * np.arctan2(vy, vx)
    field = np.stack([np.cos(angle), np.sin(angle), np.clip(c * c + (1 - c) * d, 0, 1)], -1)
    return torch.tensor(field, dtype=target.dtype, device=target.device)


def direction_loss(strokes, field):
    p0, p1, p2 = strokes[:, :6].reshape(-1, 3, 2).unbind(1)
    t = torch.tensor([0.25, 0.5, 0.75], device=strokes.device)[None, :, None]
    xy = (1 - t) ** 2 * p0[:, None] + 2 * (1 - t) * t * p1[:, None] + t * t * p2[:, None]
    tangent = 2 * ((1 - t) * (p1 - p0)[:, None] + t * (p2 - p1)[:, None])
    tangent = F.normalize(tangent, dim=-1, eps=1e-6)
    sampled = sample_image(field, xy)
    direction = F.normalize(sampled[:, :, :2], dim=-1, eps=1e-6)
    weights = sampled[:, :, 2] * torch.tensor([0.25, 0.5, 0.25], device=strokes.device)[None]
    return (
        (weights * (1 - (tangent * direction).sum(-1).square())).sum(-1) / (weights.sum(-1) + 1e-6)
    ).mean()


@dataclass
class AuditConfig:
    purity_min: float = 0.5
    split_purity_min: float = 0.8
    split_purity_max: float = 0.9999
    support_min: float = 1.0
    opacity_min: float = 0.01
    radius_max: float = 0.3
    max_edit_fraction: float = 0.1
    protection_steps: int = 100
    lambda_bad: float = 1.0
    nms_radius: int = 3


def split_stroke(stroke):
    p0, p1, p2 = stroke[:6].reshape(3, 2)
    a, b = (p0 + p1) / 2, (p1 + p2) / 2
    mid = (a + b) / 2
    children = stroke.repeat(2, 1)
    children[0, :6] = torch.stack([p0, a, mid]).flatten()
    children[1, :6] = torch.stack([mid, b, p2]).flatten()
    return children


@torch.no_grad()
def adapt(strokes, stats, target, canvas, field, born, step, seed, cfg, mode="full", capacity=None):
    n = len(strokes)
    capacity = capacity or n
    eligible = (step - born) >= cfg.protection_steps
    bad = (
        (
            (stats["benefit"] <= 0)
            & ((stats["purity"] < cfg.purity_min) | (stats["area"] < cfg.support_min))
        )
        | (strokes[:, 10] < cfg.opacity_min)
        | (strokes[:, 9] > cfg.radius_max)
    )
    candidates = (bad & eligible).nonzero().flatten()
    limit = max(1, int(n * cfg.max_edit_fraction))
    candidates = candidates[torch.argsort(stats["benefit"][candidates])[:limit]]
    removed = set(candidates.cpu().tolist()) if mode in ("full", "prune", "respawn") else set()
    # Preserve at least one primitive, so prune-only remains renderable.
    if len(removed) >= n:
        removed.remove(n - 1)
    room = capacity - (n - len(removed))
    split = (
        (stats["benefit"] > 0)
        & (stats["area"] >= cfg.support_min)
        & (stats["purity"] >= cfg.split_purity_min)
        & (stats["purity"] < cfg.split_purity_max)
        & eligible
    )
    valid_length = torch.linalg.vector_norm(strokes[:, 4:6] - strokes[:, :2], dim=-1) > 1e-3
    ids = (split & valid_length).nonzero().flatten()
    ids = ids[torch.argsort(stats["residual"][ids], descending=True)[: min(room // 2, limit)]]
    split_ids = set(ids.cpu().tolist()) if mode in ("full", "split") else set()
    out = []
    birth = []
    mapping = []
    actual_splits = 0
    for i in range(n):
        if i in removed:
            continue
        if i in split_ids:
            actual_splits += 1
            out.extend(split_stroke(strokes[i]).unbind(0))
            birth.extend([step, step])
            mapping.extend([-1, -1])
        else:
            out.append(strokes[i])
            birth.append(int(born[i]))
            mapping.append(i)
    spawn = capacity - len(out) if mode in ("full", "respawn") else 0
    if spawn:
        residual = ((target - canvas) ** 2).sum(-1).clone()
        centers = []
        h, w = residual.shape
        for _ in range(spawn):
            flat = int(residual.argmax())
            yy, xx = divmod(flat, w)
            centers.append([(xx + 0.5) / w, (yy + 0.5) / h])
            r = cfg.nms_radius
            residual[max(0, yy - r) : yy + r + 1, max(0, xx - r) : xx + r + 1] = -1
        center = torch.tensor(centers, device=target.device, dtype=target.dtype)
        new = templates(target, spawn, seed, ratio=(0, 0, 1), centers=center, direction=field)
        # Birth-time assignment: new fine strokes in front; survivors keep order.
        out = list(new.unbind(0)) + out
        birth = [step] * spawn + birth
        mapping = [-1] * spawn + mapping
    return (
        torch.stack(out),
        torch.tensor(birth, device=strokes.device),
        mapping,
        dict(pruned=len(removed), split=actual_splits, spawned=spawn),
    )
