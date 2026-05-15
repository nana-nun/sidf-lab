---
name: sidf-lab-ref-issue
description: Handle sidf-lab GitHub Issues labeled t:ref. Use for literature review, reference collection, BibTeX entries, web links, reading notes, citation metadata, and cautious relevance notes for SIDF research.
---

# SIDF Lab Reference Issue

## Overview

Use this skill for `t:ref` Issues after `sidf-issue-runner` has selected the Issue, branch, Project status, and GitHub comment flow. The goal is to add useful references without overstating what they prove for SIDF.

## Required Context

Read:

- `AGENTS.md`
- `docs/research-state.md`
- `docs/research-plan.md`
- `references/README.md` when present
- `references/links.md` when present
- `references/papers.bib` when present
- relevant `references/notes/*`
- the target Issue

## Workflow

1. Identify the requested source type: paper, web article, book, dataset, or background topic.
2. Record metadata as far as available: URL, title, authors, year, venue, DOI.
3. Put BibTeX-manageable papers in `references/papers.bib`.
4. Put web links and non-BibTeX sources in `references/links.md`.
5. Add a Markdown reading note in `references/notes/` when the Issue asks for synthesis or when the source affects research direction.
6. Separate source summary, SIDF relevance, limitations, and follow-up questions.

## Reading Note Shape

Prefer:

```markdown
# Title

## Source

## Summary

## Relevance to SIDF

## Limitations

## Follow-up
```

## Verification

Check Markdown links, BibTeX syntax by inspection or available tooling, and that references are reachable or clearly marked if not checked. Avoid adding unverifiable citation details.

## Claim Policy

Do not turn a cited method into a SIDF result. Say what the source reports, why it may matter, and what remains untested in this repository.
