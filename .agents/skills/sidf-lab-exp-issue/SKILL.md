---
name: sidf-lab-exp-issue
description: Handle sidf-lab GitHub Issues labeled t:exp. Use for reproducible SIDF experiments, baseline comparisons, metrics, saved result artifacts, experiment notes, conservative interpretation, and limitations.
---

# SIDF Lab Experiment Issue

## Overview

Use this skill for `t:exp` Issues after `sidf-issue-runner` has selected the Issue, branch, Project status, and GitHub comment flow. The goal is a reproducible experiment with a simple baseline, saved artifacts, metrics, and cautious interpretation.

## Required Context

Read:

- `AGENTS.md`
- `docs/research-state.md`
- `docs/research-plan.md`
- `docs/repository-architecture.md`
- `docs/experiment-log-template.md` when present
- relevant `results/*/notes.md`
- relevant `references/notes/*`
- the target Issue

## Workflow

1. Restate the experiment question and hypothesis before implementation.
2. Identify the baseline before building the SIDF model path.
3. Start with small images and minimal sweeps.
4. Save results under `results/<date>-<short-name>/`.
5. Save generated images; never rely on console output or `plt.show()` alone.
6. Record commands, seeds, sizes, config, metrics, decode time, date, and dependency versions when practical.
7. Keep `Question`, `Hypothesis`, `Setup`, `Baseline`, `Result`, `Interpretation`, `Limitations`, and `Next` separate.

## Required Artifacts

Include when applicable:

- `config.json`
- `metrics.json`
- `notes.md`
- input or STATIC guide image
- upscaled guide or baseline image
- rendered image
- confidence map when used
- difference map when useful

## Baselines

Prefer simple baselines first:

- nearest upscaling
- bilinear upscaling
- bicubic upscaling
- static guide direct display
- deterministic smoothing filter

Compare SIDF output against baselines before interpreting results.

## Verification

Run the experiment command and inspect that expected files exist. For Python module changes, also run targeted tests or:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

## Claim Policy

Do not claim compression superiority, super-resolution performance, emergence, or general image quality from a single experiment. State measured results, interpretation, limitations, and next questions separately.
