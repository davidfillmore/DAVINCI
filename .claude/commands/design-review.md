---
description: Severity-gated design review loop — review, stop when nothing meets the bar, fix only on approval.
argument-hint: "[high|medium|low] (default: medium)"
---

Run one cycle of the severity-gated design-review loop.

1. Parse the threshold from `$ARGUMENTS` (one of `high`, `medium`, `low`; default `medium` if empty/invalid).

2. Invoke the saved workflow:
   `Workflow({ name: "design-review", args: { threshold: "<threshold>", maxFindings: 5 } })`
   It fans out dimension reviewers → adversarially verifies each finding → returns only findings whose
   *verified* severity meets the bar.

3. Branch on the result:
   - **`converged: true` (no finding at/above the threshold)** → report: "✅ Converged at `<threshold>` — no
     findings meet the bar (max verified severity: `<maxSeverity or none>`)." Then **STOP**. Do not loop, do not
     re-run automatically, do not fix anything.
   - **findings present** → present them tersely: for each, the title, `file:line` locations, severity, and the
     two-sentence summary. Keep it short (the user cannot scroll). Then **ask which, if any, to fix** — list the
     options. Do NOT start fixing.

4. Fixes are always **gated on the user's explicit selection**. When the user approves specific findings:
   - Ground each fix in the actual code first (read it, confirm the finding still holds), implement the minimal
     safe edit, and run the full gate in the `davinci` conda env
     (`HDF5_USE_FILE_LOCKING=FALSE python -m pytest` + `mypy davinci_monet` + `black`/`isort`).
   - Surface any behavior change explicitly. **Never auto-commit or auto-push** — commit only when the user says so
     (per CLAUDE.md).

5. The "loop" is human-paced: after a fix cycle is committed, the user re-runs `/design-review` to re-review. The
   severity bar is the terminator — when a cycle returns `converged`, the loop is done.

Never invent findings to fill a quota; zero findings at the bar is a valid, expected outcome.
