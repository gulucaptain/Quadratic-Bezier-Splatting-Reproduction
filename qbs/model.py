"""Appendix A.2: ViT-small and cross-attention-first cascaded decoder.

No downloaded checkpoint by default: paper does not specify backbone pretraining.
The residual formula is followed literally; it does NOT enforce group width bounds.
"""

import math

import timm
import torch
import torch.nn.functional as F
from torch import nn

from .strokes import templates


class Stage(nn.Module):
    def __init__(self, dim=384, heads=6):
        super().__init__()
        self.ca = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.sa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, q, memory):
        n = self.norms[0](q)
        q = q + self.ca(n, memory, memory, need_weights=False)[0]
        n = self.norms[1](q)
        q = q + self.sa(n, n, n, need_weights=False)[0]
        return q + self.ff(self.norms[2](q))


class Initializer(nn.Module):
    def __init__(self, count=250, pretrained=False, ratio=(0.2, 0.3, 0.5), cross_attention=True):
        super().__init__()
        self.count = count
        self.ratio = ratio
        self.cross_attention = cross_attention
        self.backbone = timm.create_model(
            "vit_small_patch16_224", pretrained=pretrained, num_classes=0
        )
        self.projection = nn.Linear(384, 384)
        self.sizes = [int(count * ratio[0]), int(count * ratio[1])]
        self.sizes.append(count - sum(self.sizes))
        self.queries = nn.ParameterList(
            [nn.Parameter(torch.randn(n, 384) * 0.02) for n in self.sizes]
        )
        self.stages = nn.ModuleList([Stage() for _ in range(3)])
        self.head = nn.Sequential(
            nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 11)
        )
        # Small output residuals start close to the supplied geometric prior.
        nn.init.normal_(self.head[-1].weight, std=1e-4)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, image, seed=0):
        x = F.interpolate(image, (224, 224), mode="bilinear", align_corners=False)
        features = self.projection(
            self.backbone.forward_features((x - self.mean) / self.std)[:, 1:]
        )
        groups = []
        for query, stage in zip(self.queries, self.stages):
            n = len(query)
            if not n:
                continue
            pos = torch.arange(n, device=image.device)[:, None]
            freq = torch.exp(
                torch.arange(0, 384, 2, device=image.device) * (-math.log(10000) / 384)
            )
            pe = torch.stack([torch.sin(pos * freq), torch.cos(pos * freq)], -1).flatten(1)
            q = (query + pe)[None].expand(len(image), -1, -1)
            memory = torch.cat([features] + groups, 1)
            if self.cross_attention:
                q = stage(q, memory)
            else:
                norm = stage.norms[1](q)
                q = q + stage.sa(norm, norm, norm, need_weights=False)[0]
                q = q + stage.ff(stage.norms[2](q))
            groups.append(q)
        residual = 0.5 * torch.tanh(self.head(torch.cat(groups, 1)))
        template = torch.stack(
            [
                templates(im.permute(1, 2, 0), self.count, seed + i, ratio=self.ratio)
                for i, im in enumerate(image)
            ]
        )
        return (template + residual).clamp(0, 1)


def model_loss(pred, target):
    """Mean reductions; 4-neighbor Laplacian with replicate boundary assumption."""
    kernel = pred.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None].repeat(3, 1, 1, 1)

    def lap(x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=3)

    return (
        (pred - target).abs().mean()
        + (pred - target).square().mean()
        + 0.5 * (lap(pred) - lap(target)).abs().mean()
    )
