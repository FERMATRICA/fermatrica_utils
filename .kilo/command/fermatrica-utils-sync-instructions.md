---
description: Review and sync AGENTS.md with current project structure
---

# Sync FERMATRICA_UTILS Instructions

Review AGENTS.md and `.kilo/command/*.md` against the actual project structure and code conventions. Detect and fix any drift.

## Checklist for this review

1. **Directory structure** — read `fermatrica_utils/` recursively. Does the module tree description in AGENTS.md match reality? Are there new modules or deleted modules?

2. **Public API** — read `fermatrica_utils/__init__.py`. Are all public symbols still present, or were any added/removed?

3. **Subpackage modules** — read each module under `fermatrica_utils/`. Are the function descriptions and export tables still accurate?

4. **Tests** — read `tests/` listing. Are the test module descriptions and what-to-test tables up to date?

5. **Conventions** — read sample code or recent commits. Are the import ordering, error handling, type annotation, and performance profiling descriptions still correct?

6. **Command files** — read `.kilo/command/` files. Do they still reference correct paths, function signatures, and workflow steps?

For each discrepancy found, update the relevant file. Do not delete information about deprecated features — instead, add a note marking them as deprecated and noting what replaced them.
