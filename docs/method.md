# Method correspondence and implementation assumptions

Reference: [arXiv:2611.10713v1](https://arxiv.org/html/2611.10713v1). Equation and appendix numbers refer to that version. This implementation was reconstructed from the manuscript; exact equivalence to the authors' unavailable implementation is not established.

## Equation-to-code map

| Paper | Code | Validation |
|---|---|---|
| Eq. 1; Appendix A.1, Eq. 12–23 | [`closest_parameter`, `distance`](../qbs/renderer.py) | Independent NumPy polynomial-root oracle; double-precision gradient check; line/point and near-linear cases |
| Eq. 2 | [`contributions`, `render`](../qbs/renderer.py) | Full-canvas vs tiled images and gradients; recomputation equivalence |
| Eq. 3–5; Appendix A.2 | [`Initializer`, `Stage`](../qbs/model.py) | Architecture reconstruction; synthetic forward/backward diagnostics, not trained-model validation |
| Eq. 6–10 | [`render(target=...)`](../qbs/renderer.py) | Incremental benefits telescope to the total reconstruction-error change |
| Section 3.3 adaptive operations | [`adapt`, `split_stroke`](../qbs/strokes.py) | Capacity/protection rules, de Casteljau geometry, pruning precedence |
| Eq. 11 and Eq. 24 | [`fit`](../qbs/experiment.py), [`direction_loss`](../qbs/strokes.py) | Three sample locations and orientation invariance |
| Table 4 | [`templates`](../qbs/strokes.py) | 20/30/50% query groups, listed length/width intervals, target-sampled colors, opacity 0.7 |

## Stroke representation and renderer

Each row has 11 floating-point values:

```text
[P0.x, P0.y, P1.x, P1.y, P2.x, P2.y, R, G, B, half_width, opacity]
```

Control points are normalized to the unit square. Rows are ordered front to back. The curve is

\[
B(t)=(1-t)^2P_0+2(1-t)tP_1+t^2P_2,\quad 0\le t\le1.
\]

With `u=P0-2P1+P2`, `v=P1-P0`, and `w=P0-p`, stationary points satisfy

\[
(u^Tu)t^3+3(u^Tv)t^2+(2v^Tv+u^Tw)t+v^Tw=0.
\]

Real roots and endpoints determine the closest point. Roots outside the interval are clamped to endpoints, which are already candidates. Root selection is detached; the distance is differentiated with the selected parameter held fixed. Away from ties, the envelope theorem gives the distance gradient. Near-degenerate curves use the paper's `||u||² <= 1e-6` line-segment branch.

Opacity follows Eq. 2, including its finite-support clamping and softness normalization. The background is black. Tile bounds use the control-point convex hull expanded by half-width; culling does not replace the analytic curve with a polyline. Optional checkpointing recomputes tiles during backward to reduce retained memory. CUDA-specific warp reductions and fused kernels are not implemented.

## Initializer architecture

The implementation uses `vit_small_patch16_224`, discards the CLS token, and projects spatial tokens to dimension 384. Three query groups use six-head, pre-normalized cross-attention followed by within-group self-attention and a feed-forward block. Later groups attend to image features and previous groups. A shared MLP with two 256-dimensional hidden layers predicts `0.5*tanh` residuals added to templates, followed by clamping to `[0,1]`.

The default 250-query network has 29,176,459 parameters, compared with the paper's approximate 28M description. A 30-query diagnostic has 29,091,979 parameters. No exact architectural identity is claimed. The network has only undergone short synthetic gradient diagnostics; it was not used for the released image-fitting results.

## Explicit assumptions and unresolved details

| Topic | Release behavior | Reproduction consequence |
|---|---|---|
| Audit thresholds | `lambda_bad=1`, purity cutoff 0.5, split purity `[0.8,0.9999)`, minimum visible support 1 pixel, opacity cutoff 0.01, maximum radius 0.3 | Values are implementation choices where numerical settings are not disclosed |
| Editing schedule | Protection 100 steps; at most 10% pruning proposals per event; NMS radius 3 pixels | Logged in every run's `audit_config` |
| Curvature | Uniform offset within ±0.15 of curve length | Not uniquely specified by the manuscript |
| Direction field | Derivative scales 1 and 3 pixels; tensor smoothing twice those scales; coherence-weighted doubled-angle blending | Eq. 24 is implemented; its input field is independently specified |
| Fixed depth | Fine groups precede coarse groups; new births are inserted in front; survivors keep relative order; children occupy their parent's position | Numeric depth-key assignments are not reproduced exactly |
| Adam state | Surviving strokes retain moments; edited strokes receive zero moments | State migration is not specified by the paper |
| Reductions and edges | Mean losses; four-neighbor Laplacian; replicated boundaries | Alternative reduction/edge conventions can change training behavior |
| Backbone initialization | Random weights by default; `--pretrained` is opt-in | The manuscript does not unambiguously identify pretrained weights |
| Bounded residuals | Eq. 5 is followed literally | A ±0.5 residual does not enforce the original fine-scale width interval; no additional hard scale bounds are imposed |
| Geometry bounds | Fit-time coordinates/colors in `[0,1]`, half-width in `[0.0005,0.5]`, opacity in `[0,1]` | Bounds are implementation settings rather than a fully disclosed author configuration |
| Snapshot selection | Evaluate at fixed `tau=0.3` every 50 steps in the pilot and every 100 steps in the long run; minimize target-image MSE | Selection frequency is independently specified; final output is also retained |
| Budget | Fixed initial capacity in published fitting experiments | The paper's complete growth-to-5K/6K pipeline is missing |
| Patch integration | Learned fitting entry point supports one patch, at most 256×256 | Full-image patch stitching and global refinement are not reproduced |
| Rectangular canvases | Unit-square coordinate metric | Aspect-ratio-aware width conversion remains unvalidated; published fitting tests are square |
| Training epochs | Shuffled image sampling with one random crop per image per nominal epoch; the last effective batch may cross the epoch boundary | The manuscript's patch count per epoch is not disclosed |

Pruning takes precedence when a stroke is proposed for both pruning and splitting. Split-only mode has no room to create children when initialized at full capacity; this mode must not be presented as the paper's Split Only ablation. Direction loss is activated after 200 steps at five-step intervals with weight 0.04.

## Paper-level completion matrix

| Paper result | Release status | Remaining work |
|---|---|---|
| Table 1: main image benchmarks | Not reproduced | Trained initializer, full datasets, native budgets/resolutions, six baselines, LPIPS |
| Table 2: renderer comparison | Formula checks and local scaling only | Comparable baseline renderers, hardware, kernels, and timing protocol |
| Table 3: ablations | Small matched-seed audit comparison only | Learned initialization, all architecture/edit ablations, original evaluation regime |
| Table 4: priors | Implemented with stated assumptions | Sensitivity to undisclosed curvature/ordering choices |
| Table 5: ordering | Not reproduced | Original/random/reversed within-layer evaluation |
| Appendix A.4.2: line primitives | Line renderer implemented; experiment incomplete | Same trained initializer and full evaluation |
| Figure 7: cross-patch behavior | Not reproduced | Global patch pipeline |
| Figure 10: material appearance | Not implemented | Brush/canvas textures, paint height, lighting, supersampling |
| User study | Not conducted | Independent perceptual evaluation |

Passing numerical tests establishes properties of the implementation. It does not establish algorithmic equivalence or replication of the paper's experimental claims.
