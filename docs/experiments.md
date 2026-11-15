# Experimental protocol

## Published fitting suite

The release contains 18 complete fitting runs: three public Im2Oil inputs (`A0`, `L0`, `P0`), seeds 0/1/2, and auditing enabled/disabled. The immutable input locations and SHA-256 digests are in [data/manifest.json](../data/manifest.json). These inputs are a selected subset of the upstream repository, not an identified copy of the paper's complete Gallery evaluation split.

| Setting | Value |
|---|---|
| Resize | RGB, 128×128, Pillow Lanczos; direct square resize |
| Initial strokes / maximum capacity | 128 / 128 |
| Initializer | Random multiscale templates; target colors sampled at centers |
| Optimizer | Adam, initial learning rate 0.02, cosine decay over 500 steps |
| Softness | 1.0 before step 100, 0.6 before step 200, 0.3 thereafter |
| Audit events | Every 50 completed steps starting at 100, through step 400 inclusive |
| Direction regularization | Weight 0.04, after 200 steps, every 5 steps |
| Tile size | 32 pixels |
| Raster sampling | 1×; no supersampling |
| Snapshot selection | Lowest recorded MSE at fixed softness 0.3 |
| SSIM | scikit-image, RGB, `data_range=1`, default window/settings |
| PSNR | `-10 log10(mean squared RGB error)` on floating-point arrays in `[0,1]` |
| LPIPS | Not measured; represented by JSON `null` |

Each pair uses the same seed and identical initial geometry/color parameters. The methods share maximum capacity; pruning and splitting can change the stroke population. Results are descriptive means and sample standard deviations over the three seeds. They are not estimates of population-wide significance.

The complete executable configuration is [configs/pilot.json](../configs/pilot.json). Stored [run configurations](../results/pilot) are authoritative for the archived experiments.

## Re-running and cache behavior

```bash
qbs-prepare-data
qbs-suite --config configs/pilot.json --root runs/pilot --device mps
qbs-report --root runs/pilot --out runs/report
```

Completed compatible runs are skipped. A cache with changed arguments or core source digests raises an error. `--force` recomputes runs explicitly. A single-image fit rejects a nonempty output directory unless `--overwrite` is supplied. Neither behavior is a checkpoint-resume mechanism for interrupted image fitting; an incomplete fit must be rerun.

Original inputs are not committed as dataset files. The figure builder can regenerate missing `target.png` files from downloaded inputs. Numerical aggregation with `--no-figures` requires only the published JSON/JSONL records.

## Long-run evaluation

One additional run uses A0, seed 0, 128×128 pixels, 128 strokes, 5,000 steps, and an audit tail of 500 steps. It logs every 100 steps. The cosine schedule therefore differs from the 500-step suite. Comparing the two runs does not isolate iteration count alone.

```bash
qbs-fit --image data/gallery/A0.jpg --out runs/long_A0_s0_full \
  --device mps --steps 5000 --audit-tail 500 --log-every 100
```

## Renderer scaling

```bash
python -m qbs.benchmark --out runs/scaling.json
```

The benchmark measures synthetic targets at 128/256/600 pixels and up to 5,000 strokes. Control-point extents and widths are scaled to approximate local 256-pixel patch priors. Each configuration performs three forward/backward passes with recomputation enabled; the first is warm-up and the last two determine the reported average. MPS synchronization brackets timing regions. Driver-allocated memory is an allocator observation, not a measured peak-memory bound.

These timings exclude optimizer updates, audit scheduling, data preparation, and full model inference. The 5,000-step time extrapolation is not an end-to-end measurement or a comparison with the paper's CUDA implementation.

## Initializer diagnostics and training

The published synthetic diagnostics test 30 queries at 64×64 and 250 queries at 256×256, batch size 1, for five steps. Templates change with the step seed. Nonzero finite backbone gradients establish that the computation graph is connected; losses across changing templates do not establish convergence. Diagnostic checkpoints are not released as pretrained models.

For full training, obtain Flickr2K through its [upstream project](https://github.com/LimBee/NTIRE2017), respecting its data terms. The training command uses:

- AdamW, base learning rate `1e-4`, weight decay `1e-4`.
- Effective batch size 32 with configurable microbatch accumulation.
- Linear warm-up for 5% of steps followed by cosine decay.
- 256-pixel random crops; images smaller than 256 are upscaled first.
- L1 + MSE + 0.5 × Laplacian L1 loss; training softness is fixed to 1.0.
- A nominal 200-epoch schedule; exact training-patch accounting is an assumption.

The full schedule has not been executed. Resume requires the same dataset path and schedule-related arguments; incompatible arguments are rejected. A checkpoint already at the requested total steps exits without further training. Load only trusted local checkpoints: training state includes Python/NumPy optimizer and RNG objects.

## Numerical validation

The reference CPU tests include an independent NumPy root solver and double-precision finite differences. MPS validation compares image outputs and stroke gradients against CPU float32. Tests do not cover every optimizer trajectory, image, or undisclosed manuscript setting.

Release checks additionally exercise installation, a CPU fit, an MPS fit, cache validation, invalid-input handling, and regeneration of the published summary. See the [GitHub Actions workflow](../.github/workflows/tests.yml) for automated CPU checks.

## Provenance

Archived runs retain their original input and source digests, runtime metadata, and traces. They were generated before the release packaging change from `src` to `qbs`; historical source digests are therefore not asserted to match the formatted release files. Packaging and CLI checks are validated separately. Scientific metrics are preserved rather than silently replaced with new timings.

An event-count correction is recorded in the run metrics: `split_proposals` retains the originally reported proposal count; `split` counts executed splits after pruning precedence. For the fixed-capacity full-audit runs, executed splits equal `pruned - spawned`. This correction does not change stroke parameters, images, or image metrics. Earlier incomplete exploratory runs are excluded from the published suite.
