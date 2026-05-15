---
name: sidf-lab-docs-issue
description: Handle sidf-lab GitHub Issues labeled t:docs. Use for Japanese human-facing documentation, English AI-facing workflow or skill documentation, research state updates, specs drafts, and careful separation of claims, hypotheses, results, interpretation, and limitations.
---

# SIDF Lab Documentation Issue

## Overview

Use this skill for `t:docs` Issues after `sidf-issue-runner` has selected the Issue, branch, Project status, and GitHub comment flow. The goal is documentation that helps humans or future agents work safely without overstating research status.

## Required Context

Read:

- `AGENTS.md`
- `README.md`
- `docs/research-state.md`
- `docs/research-plan.md`
- `docs/repository-architecture.md`
- relevant `docs/*`, `specs/*`, `results/*/notes.md`, or `.agents/*`
- the target Issue

## Language Policy

Write human-facing docs in Japanese:

- `README.md`
- `docs/*`
- `references/*`
- `results/*/notes.md`

AI-facing docs may be English:

- `.agents/*`
- skill files
- workflow or policy files intended primarily for agents

## Workflow

1. Identify the reader: human researcher, future agent, implementer, or reviewer.
2. Keep claims, hypotheses, results, interpretations, limitations, and next steps separate.
3. Put draft specifications in `specs/`; do not mix them into experiment results.
4. Update `docs/research-state.md` only when a result or interpretation actually changes the current research state.
5. Link related docs instead of duplicating long policy blocks.

## Verification

Inspect Markdown structure, links, headings, and terminology. For skill documentation, validate frontmatter shape and check that no template placeholder remains.

## Claim Policy

Use cautious wording around `compression`, `super-resolution`, and `emergence`. Mention measurement status and limitations whenever research results are discussed.
