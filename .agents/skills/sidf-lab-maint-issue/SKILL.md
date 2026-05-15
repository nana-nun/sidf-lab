---
name: sidf-lab-maint-issue
description: Handle sidf-lab GitHub Issues labeled t:maint. Use for repository maintenance, workflow changes, templates, environment setup, CI, housekeeping, and direct verification of operational changes.
---

# SIDF Lab Maintenance Issue

## Overview

Use this skill for `t:maint` Issues after `sidf-issue-runner` has selected the Issue, branch, Project status, and GitHub comment flow. The goal is repository upkeep that preserves research workflow and stays easy to review.

## Required Context

Read:

- `AGENTS.md`
- `README.md`
- `docs/repository-architecture.md`
- `.github/*` when workflow or templates are affected
- dependency or environment files when setup is affected
- the target Issue

## Workflow

1. Identify whether the change affects workflow, environment, repository layout, templates, CI, or housekeeping.
2. Keep maintenance changes separate from behavior changes when possible.
3. Preserve existing project conventions and file locations.
4. Update human-facing docs in Japanese when repository operation changes.
5. Record a verification command or inspection result.

## Verification

Use the smallest direct check that proves the maintenance change works:

- template or docs change: inspect generated Markdown structure or template fields
- dependency or environment change: run install, import, or test command when practical
- workflow change: run the relevant command or dry run when available
- repository cleanup: inspect `git status --short` and changed paths

## Done Criteria

Maintenance Issues are not done until the operational effect is documented and at least one verification result is recorded.

## Claim Policy

Do not mix research interpretation into maintenance PRs. Open a follow-up Issue when upkeep reveals research or implementation work.
