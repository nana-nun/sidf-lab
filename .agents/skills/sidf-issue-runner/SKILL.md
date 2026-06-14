---
name: sidf-issue-runner
description: Orchestrate GitHub Issue work in the nana-nun/sidf-lab SIDF research repository. Use when Codex is asked to work from a sidf-lab Issue URL or number, select the correct tag-specific skill, create an Issue branch, update GitHub Project status, leave Issue comments, verify changes, create a PR, or apply SIDF Issue policies for t:exp, t:ref, t:impl, t:docs, or t:maint work.
---

# SIDF Issue Runner

## Overview

Use this skill as the standard procedure for handling SIDF research repository Issues. It coordinates Issue selection, the GitHub lifecycle, and delegation to a tag skill. Process one sidf-lab GitHub Issue as one small, reviewable unit of work; treat the Issue as the required source of scope, keep claims conservative, and preserve the distinction between hypotheses, measured results, interpretations, and limitations.

## Issue Selection

When the user names a specific Issue number or URL, use that Issue even if other open Issues have higher priority. Mention the explicit selection in the start comment.

When the user asks to work on "one remaining Issue", "the next Issue", or otherwise does not specify an Issue, select from open Issues by priority before convenience:

1. List open Issues with labels and Project status.
2. Exclude Issues whose Project status is `In Progress`, `Review`, `Done`, or `Blocked`, unless the user explicitly asks to resume one of them.
3. Select the highest priority label in this order: `p:0`, then `p:1`, then `p:2`.
4. Within the same priority, prefer `Ready` over `Todo`.
5. If several Issues are still tied, choose the smallest safe, reviewable Issue and state why.

Do not choose a lower-priority Issue merely because it looks easier. If a lower-priority Issue is selected despite a higher-priority open Issue, record the reason in the start comment and final response. Acceptable reasons include an explicit user request, a higher-priority Issue being blocked, or missing permissions or context that make the higher-priority Issue impossible to start.

## Start Checklist

Before editing files:

1. Select the target Issue according to the Issue Selection rules above, unless the user already specified the Issue.
2. Read the target Issue, including labels, priority, Project status, tasks, acceptance criteria, and references.
3. Check the working tree with `git status --short --branch`; do not overwrite unrelated user changes.
4. Read these repository files when present:
   - `AGENTS.md`
   - `README.md`
   - `docs/research-state.md`
   - `docs/research-plan.md`
   - `docs/repository-architecture.md`
   - Issue-specific references listed in the Issue body
5. Identify exactly one primary Issue type label: `t:exp`, `t:ref`, `t:impl`, `t:docs`, or `t:maint`.
6. If the Issue type is missing or contradictory, infer only when the title and tasks are unambiguous; otherwise ask for clarification.
7. Select and use the matching tag-specific skill before making type-specific decisions.

## Tag Skill Dispatch

Use the common runner for lifecycle and the tag skill for work rules:

- `t:exp` -> `sidf-lab-exp-issue`
- `t:ref` -> `sidf-lab-ref-issue`
- `t:impl` -> `sidf-lab-impl-issue`
- `t:docs` -> `sidf-lab-docs-issue`
- `t:maint` -> `sidf-lab-maint-issue`

If a matching tag skill is missing, keep working from this runner and note the missing skill in the final response or Issue comment.

## GitHub Lifecycle

At work start:

1. Create or switch to the Issue branch.
2. Move the Issue's GitHub Project status to `In Progress`.
3. Re-read or inspect the Issue/Project item enough to confirm that status is now `In Progress`.
4. Add an Issue comment stating that work has started, the branch name, priority, selected scope, and, when the Issue was selected from a list, why it was selected under the priority rules.

Do not edit repository files before the Project status has been updated to `In Progress`, unless GitHub tooling is unavailable. If the status update cannot be completed because credentials, permissions, Project fields, network, or GitHub tooling are unavailable, stop before file edits when practical and either ask for help or leave a start comment saying that the status update is blocked. If the user explicitly asks to continue despite the blocked status update, record the limitation in the final response and PR.

During work:

- If blocked, move the Project status to `Blocked` and comment with the blocker and next required action.
- If scope changes, comment on the Issue before expanding the PR beyond the Issue's acceptance criteria.
- Keep the PR small; open a follow-up Issue for unrelated discoveries.

At work completion:

1. Run the verification required by the tag skill.
2. Review limitations, skipped checks, TODOs, new research questions, and out-of-scope discoveries from the work.
3. For completed experiment Issues, create at least one actionable Next Issue based on the measured result. For other Issue types, create or propose follow-up Issues when the remaining work is concrete and not already covered by an open Issue.
4. Commit and push the branch when the user asked for a complete Issue workflow or PR.
5. Create a PR with the standard template.
6. Move the Issue's GitHub Project status to `Review`.
7. Add an Issue comment with the PR URL, verification summary, saved results if any, limitations, follow-up Issue URLs, and any remaining follow-ups that were not created.
8. Do not merge the PR unless the user explicitly asks.

Prefer available GitHub tooling for these actions. With GitHub CLI, use `gh issue comment`, `gh pr create`, and `gh project` commands after discovering the relevant project item and Status field options. If project automation cannot be completed because credentials, permissions, or field IDs are unavailable, leave an Issue comment with the status change that should happen and state the limitation in the final response.

## Follow-up Issue Creation

At completion, decide whether new Issues are needed before opening or finalizing the PR. The goal is to preserve useful next work without expanding the current PR beyond its scope.

For `t:exp`, a completed experiment must create at least one Next Issue after results are available. Base it on a measured limitation, comparison gap, implementation requirement, or specification decision. Do not preselect the Next Issue before seeing the result, and do not create a vague continuation merely to satisfy the rule.

Create or propose a follow-up Issue when:

- A limitation or skipped verification is concrete enough to be worked independently.
- An experiment result creates a clear next experiment, baseline, metric, or saved artifact requirement.
- A documentation change identifies an unresolved specification decision or missing reference.
- An implementation change exposes a test gap, refactor, performance concern, or reproducibility requirement.
- A literature review identifies a specific paper, method, or comparison needed for SIDF positioning.

Do not create a follow-up Issue when:

- The item is vague, speculative, or not actionable yet.
- An open Issue already covers the same work; link the existing Issue instead.
- The item is an acceptance criterion that should be completed in the current Issue.
- The follow-up would mix unrelated work types; split it by `t:exp`, `t:ref`, `t:impl`, `t:docs`, or `t:maint`.

Before creating a follow-up Issue:

1. Search open Issues for duplicates.
2. Choose the correct `t:*` label and priority label.
3. Keep the Issue body small but actionable, with `Goal`, `Context`, `Tasks`, `Acceptance Criteria`, and `References` when useful.
4. Add it to the GitHub Project when practical.
5. Mention the new Issue URL in the parent Issue comment and PR `Limitations` or `Results` section as appropriate.

If GitHub access is unavailable, list the proposed follow-up Issue title, labels, and body in the parent Issue comment when possible, or in the final response if commenting is also unavailable.

## Branching

Use one branch per Issue unless the user asks for a different workflow.

Branch format:

```text
<type>/issue-<number>-<short-name>
```

Map labels to branch prefixes:

- `t:exp` -> `exp`
- `t:ref` -> `ref`
- `t:impl` -> `impl`
- `t:docs` -> `docs`
- `t:maint` -> `maint`

Examples:

```text
docs/issue-10-sidf-issue-runner
impl/issue-3-python-package-skeleton
exp/issue-4-model-c-cross-baseline
```

## Shared Issue Type Policies

The tag-specific skills are authoritative for detailed work rules. These shared policies are fallback reminders.

### `t:exp`

Run reproducible experiments. Start with small inputs and simple sweeps. Always compare against a simple baseline before interpreting SIDF model output.

Save results under `results/YYYY-MM-DD-issue-<number>-<short-title>/` and include, when applicable. Use the GitHub Issue number in `<number>` and a short ASCII slug in `<short-title>`. If an experiment starts without an Issue number, create or identify the Issue before finalizing the PR, then rename the result directory to the issue-numbered form:

- `config.json`
- `metrics.json`
- `notes.md`
- input or STATIC guide image
- upscaled guide or baseline image
- rendered image
- confidence map when used
- difference map when useful

Write `notes.md` with separated sections: `Question`, `Hypothesis`, `Setup`, `Baseline`, `Result`, `Interpretation`, `Limitations`, and `Next`. Record commands, seeds, input/output sizes, model config, metrics, decode time, date, and Python/dependency versions as far as practical.

### `t:ref`

Add literature, links, or reading notes. Put BibTeX-manageable papers in `references/papers.bib`, web links in `references/links.md`, and reading notes in `references/notes/`.

Record URL, title, authors, year, and DOI when available. Keep summaries separate from interpretations and note relevance to SIDF reconstruction explicitly.

### `t:impl`

Implement Python or tooling changes in the smallest useful scope. Prefer existing module boundaries in `src/sidf_lab/`.

For Python verification, use the project `.venv` when available or create it if needed:

```powershell
$runtimePython = Get-ChildItem -LiteralPath "$env:USERPROFILE\.cache\codex-runtimes" -Recurse -Filter python.exe |
  Where-Object { $_.FullName -notmatch "WindowsApps" } |
  Select-Object -First 1
& $runtimePython.FullName -m venv --system-site-packages --without-pip .venv
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

For CLI-level checks:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m sidf_lab.cli
```

If implementation generates images, save the images rather than relying on `plt.show()`.

### `t:docs`

Update human-facing documentation in Japanese. Keep AI-facing workflow or skill files in English when that improves future agent reliability.

Separate claims, hypotheses, results, interpretations, limitations, and next steps. Be especially careful with wording around `compression`, `super-resolution`, and `emergence`.

### `t:maint`

Adjust repository structure, workflow, templates, CI, dependency files, or housekeeping. Keep behavior changes separate from cleanup when possible, and record at least one verification command or inspection result.

## Research Claim Policy

Do not present SIDF as a completed practical compression format.

Do not claim super-resolution performance without Ground Truth comparison, metrics, and limitations.

Do not treat an implementation as a formal specification unless deterministic behavior and environment assumptions are documented.

Prefer wording such as:

- "hypothesis"
- "current result"
- "interpretation"
- "limitation"
- "candidate model"
- "draft specification"

Avoid wording that implies validated generality from a single synthetic experiment.

## Verification

Run verification that matches the Issue type and changed files:

- Documentation-only changes: inspect rendered Markdown structure and run lightweight repository checks when available.
- Python implementation: run targeted tests first, then broader tests when shared behavior changed.
- Experiments: run the experiment command, confirm `config.json`, `metrics.json`, `notes.md`, and generated images exist.
- Maintenance: run the workflow, command, or validation most directly affected by the change.

If a requested verification cannot run, state exactly why and what remains unverified.

## PR Preparation

Create a PR after the work is verified and pushed. Prepare PR text with these sections:

```markdown
## Summary

## Verification

## Results

## Limitations

## Related Issue
```

In `Verification`, include commands actually run and whether they passed. In `Results`, summarize saved experiment artifacts or say "Not applicable" for non-experiment work. In `Limitations`, include remaining uncertainty, skipped checks, or scope boundaries. In `Related Issue`, use `Closes #<number>` when the work is intended to complete the Issue.

After creating the PR, comment on the Issue with:

- PR URL
- verification commands and outcomes
- result artifact paths for experiment work
- limitations or skipped checks
- statement that the PR is ready for review and has not been merged
