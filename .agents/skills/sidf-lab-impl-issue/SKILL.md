---
name: sidf-lab-impl-issue
description: Handle sidf-lab GitHub Issues labeled t:impl. Use for scoped Python package, CLI, test, metrics, experiment tooling, or future Rust implementation changes with appropriate tests or CLI verification.
---

# SIDF Lab Implementation Issue

## Overview

Use this skill for `t:impl` Issues after `sidf-issue-runner` has selected the Issue, branch, Project status, and GitHub comment flow. The goal is a small implementation change with direct verification and no unsupported research claims.

## Required Context

Read:

- `AGENTS.md`
- `docs/repository-architecture.md`
- `docs/research-state.md`
- relevant files in `src/sidf_lab/`
- relevant tests in `tests/`
- relevant experiments in `experiments/` when present
- the target Issue

## Workflow

1. Map the Issue to existing module boundaries before editing.
2. Keep orchestration, metrics, visualization, IO, and model kernels in their existing areas when possible.
3. Add or update targeted tests when behavior changes.
4. If the implementation generates images, save the images to a documented path.
5. Keep broad refactors out of the PR unless required by the Issue.
6. Document public behavior changes when needed.

## Python Verification

Use the project `.venv` when available. If it is missing and Python work is required, create it and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set the module path before running repository imports:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

For CLI checks:

```powershell
$env:PYTHONPATH = "src"
python -m sidf_lab.cli
```

## Done Criteria

Implementation Issues are not done until at least one relevant test, script, or CLI check has run, or the reason it could not run is recorded.

## Claim Policy

Implementation work may enable experiments, but it does not by itself prove compression, super-resolution, or model quality claims.
