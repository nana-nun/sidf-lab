# SIDF Research Notes: v0.1 to v0.3 and Next Architecture

Date: 2026-05-13

## 1. Current Definition

SIDF is an experimental image description format that stores reconstruction conditions rather than final pixels.

The current research direction is:

> Store a low-information STATIC guide, seed, and physical parameters, then reconstruct an image through a deterministic stochastic relaxation process.

At this stage, SIDF should be described as an experimental reconstruction model, not yet as a proven high-compression image format.

## 2. Important Positioning

Current SIDF can reasonably claim:

- Low-resolution guides can steer deterministic image reconstruction.
- Data fidelity terms prevent brightness drift and preserve dark regions.
- Edge-aware interactions reduce leakage across structural boundaries.
- Seeded stochastic terms can add reproducible texture-like variation.

Current SIDF should not yet claim:

- Better compression than PNG, JPEG, AVIF, JPEG XL, or neural codecs.
- True semantic super-resolution.
- Recovery of lost high-frequency details from natural images.
- Environment-independent bit-perfect reproducibility across implementations.
- Practical decode speed for large images.

## 3. Model History

### Model A: External-Field Noise Relaxation

Energy:

```text
E = sum J(v_i - v_j)^2 - sum h_i v_i
```

Observation:

- The cross structure appeared, but bright regions expanded too much.
- Dark background became gray.
- Small positive noise in the STATIC layer acted as a one-way force toward brightness.

Interpretation:

This model behaves more like external-field-driven noise relaxation than image reconstruction.

Main lesson:

`-h_i v_i` is not a good image fidelity term because it only rewards larger values when `h_i > 0`.

### Model C: Edge-Preserving Stochastic Relaxation

Energy:

```text
E = lambda_data * sum (v_i - s_i)^2
  + sum J_ij(v_i - v_j)^2

J_ij = J_base * exp(-gamma * (s_i - s_j)^2)
```

Observation:

- Dark background remains dark.
- Cross mean stays close to the target value.
- Edge leakage is strongly reduced.
- The output becomes stable and predictable.

Example metrics:

```text
MAD                 : 0.0195
Cross Mean          : 0.5014 target 0.5000
Background Mean     : 0.0085 target 0.0000
Edge Leakage        : 0.0097
Cross Variance      : 0.0016
Background Variance : 0.0004
```

Interpretation:

Model C is a solid base model for controlled reconstruction. It is less "emergent", but much more suitable as a file-format foundation.

### Model D: Confidence-Aware Multi-Resolution Reconstruction

Energy:

```text
E = lambda_data * sum c_i(v_i - s_i)^2
  + sum J_ij(v_i - v_j)^2
  + texture_strength * sum texture(i, j) * v_i
```

Pipeline:

```text
16x16 low-res guide
-> bilinear upscaled guide
-> confidence map from guide gradient
-> edge-aware stochastic relaxation
-> rendered 64x64 output
```

Observation:

- The project finally moved from same-resolution filtering to low-resolution guide reconstruction.
- Compared with bilinear upscaling, the cross edge appears visually tighter.
- The confidence map highlights edge regions and controls fidelity strength.
- Seeded texture is visible, but still mostly granular noise.

Interpretation:

Model D is promising, but it is better described as confidence-aware multi-resolution reconstruction than proven super-resolution.

## 4. Image Result Interpretation

The attached images show the research trajectory clearly:

- Model A demonstrates why the original external-field formulation fails: it lets brightness spread into the background.
- Model C demonstrates that data fidelity and edge-aware coupling produce stable structure-preserving reconstruction.
- Model D demonstrates the first useful low-resolution-to-higher-resolution pipeline.

The most important current result is not "high-detail emergence" yet. The important result is that SIDF has moved from an unstable physical metaphor into a controllable reconstruction model.

## 5. Recommended Project Structure

Start with Python for research speed, but keep module boundaries close to a future Rust implementation.

```text
sidf-lab/
  README.md
  docs/
    sidf-research-notes.md
    sidf-v0.3-draft.md
    experiment-log-template.md
  python/
    sidf/
      __init__.py
      prng.py
      guides.py
      confidence.py
      texture.py
      energy.py
      anneal.py
      metrics.py
      visualize.py
    experiments/
      exp_001_model_a_cross.py
      exp_002_model_c_cross.py
      exp_003_model_d_multires_cross.py
      exp_004_shapes_benchmark.py
    outputs/
      .gitkeep
  rust/
    sidf-core/
      Cargo.toml
      src/
        lib.rs
        prng.rs
        fixed.rs
        guide.rs
        confidence.rs
        energy.rs
        anneal.rs
        metrics.rs
  specs/
    sidf-v0.2.1.md
    sidf-v0.3.0-draft.md
```

## 6. Python Module Responsibilities

### `guides.py`

Guide generation and loading.

- Synthetic cross, circle, diagonal line, gradient, and patch guides.
- Downscale and upscale helpers.
- Later: RGB/YCbCr conversion helpers.

### `confidence.py`

Confidence map generation.

- Gradient-based confidence.
- Edge confidence.
- Flat-region confidence reduction.
- Future: confidence from compression artifacts or segmentation.

### `texture.py`

Deterministic texture priors.

- White noise baseline.
- Perlin or fractal noise.
- Directional noise along edges.
- Future: region-specific texture presets.

### `energy.py`

Energy functions.

- Model A external field.
- Model C edge-preserving fidelity.
- Model D confidence-aware reconstruction.

Keep this module pure and explicit. It should be the easiest module to port to Rust.

### `anneal.py`

Annealing loop and update schedule.

- Metropolis update.
- Greedy/ICM update for faster comparison.
- Multi-scale schedule.
- Deterministic pixel traversal options.

### `metrics.py`

Evaluation.

- MAD.
- Background mean and variance.
- Foreground mean and variance.
- Edge leakage.
- Edge width.
- SSIM and PSNR later.
- Decode time.

### `visualize.py`

Plot helpers only.

- Static guide.
- Upscaled guide.
- Confidence map.
- Rendered image.
- Difference maps.
- Metric overlays.

## 7. Rust Migration Strategy

Do not port the whole Python prototype at once.

Recommended order:

1. Define fixed data structures in Python.
2. Freeze one model: likely Model C first.
3. Port only the core annealing kernel to Rust.
4. Keep visualization and experiment orchestration in Python.
5. Add Python bindings later with PyO3 or exchange `.npy`/raw buffers first.

Rust should eventually own:

- PRNG.
- Fixed-point arithmetic.
- Energy calculation.
- Annealing update loop.
- SIDF binary parse/write.

Python should continue to own:

- Experiment scripting.
- Plotting.
- Rapid parameter search.
- Research notebooks or batch comparisons.

## 8. Next Experiments

### Experiment 004: Shape Benchmark

Goal:

Test whether Model D works beyond horizontal and vertical cross structures.

Inputs:

- Cross.
- Diagonal line.
- Circle.
- Thin line.
- Soft gradient.
- Checker edge.

Compare:

- Bilinear.
- Bicubic.
- Model C same-resolution baseline.
- Model D multi-resolution reconstruction.

Metrics:

- Edge leakage.
- Edge width.
- Foreground/background mean.
- Variance.
- Decode time.

### Experiment 005: Texture Prior Comparison

Goal:

Replace white-noise texture with structured noise.

Compare:

- White noise.
- Smoothed white noise.
- Fractal noise.
- Perlin/Simplex-like noise.
- Edge-aligned anisotropic noise.

Expected result:

The output should move from granular noise toward controlled material-like texture.

### Experiment 006: Real Patch Test

Goal:

Test failure modes on small real or illustration patches.

Patch classes:

- Anime eye.
- Hair strand.
- Cloth shadow.
- Soft skin gradient.
- Natural cloud/water texture.

Main question:

Does the confidence map harden soft gradients too much?

## 9. Proposed Spec Direction

### SIDF v0.2.1

Name:

```text
Edge-Preserving Annealing Reconstruction
```

Scope:

- Same-resolution reconstruction.
- Data fidelity.
- Edge-aware interaction.
- Deterministic seed.

### SIDF v0.3.0

Name:

```text
Confidence-Aware Multi-Resolution Reconstruction
```

Scope:

- Low-resolution STATIC guide.
- Target output shape.
- Confidence map.
- Texture prior.
- Multi-resolution reconstruction pipeline.

## 10. Immediate Decision Points

Before writing more code, decide these items:

1. Python package style: plain scripts first, or installable package.
2. Image domain: grayscale only for now, or Y channel with color later.
3. Determinism target: NumPy reproducible prototype first, or fixed-point deterministic prototype early.
4. Experiment outputs: PNG only, or PNG plus JSON metrics.
5. Rust timing: after Model C is frozen, or after Model D is benchmarked.

Recommended choices for now:

- Use grayscale only.
- Use a small Python package plus experiment scripts.
- Save every experiment as PNG plus JSON metrics.
- Freeze Model C as the first Rust target.
- Keep Model D in Python until texture and confidence behavior are better understood.

