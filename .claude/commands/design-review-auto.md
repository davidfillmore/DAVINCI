---
description: FULL-AUTO design-review loop — review→fix→verify→commit until converged or capped. Branch-only, never pushes/merges.
argument-hint: "[high|medium|low] [maxIterations] (default: medium 4)"
---

Run the design-review loop in **full-auto mode**: keep reviewing and fixing until the severity bar is clear
(`converged`) or a cap is hit. The completion goal is **convergence at the threshold**. This mode acts on its own
judgement, so it is bounded by hard guardrails below and **stops for human review at the end — it never pushes and
never merges**.

## Setup
- Parse `$ARGUMENTS`: first token = threshold (`high`|`medium`|`low`, default `medium`); second = `maxIterations`
  (integer, default `4`).
- If the current branch is `main` or `develop`, create a working branch first (e.g. `auto-design-review`) — never
  commit auto-generated fixes onto a protected branch.
- Announce: threshold, maxIterations, "commits each verified fix to `<branch>`, stops at convergence or cap, will
  not push/merge." Maintain an in-memory `attempted` set of finding keys (`title` + sorted `locations`) for the
  whole run to prevent oscillation/re-opening.

## Loop (repeat up to `maxIterations` times)
1. **Review**: `Workflow({ name: "design-review", args: { threshold: "<threshold>", maxFindings: 5 } })`.
2. **Converged?** If the result is `converged: true`, **STOP** — go to Wrap-up. This is the goal state.
3. **Pick work**: take the returned findings (already at/above the bar) whose key is NOT in `attempted`. If none
   remain (every at-bar finding was already attempted), **STOP (stuck)** — go to Wrap-up and say so.
4. **Fix each picked finding, one at a time** (separate commit each, so every commit is reviewable/revertable):
   a. Ground it: read the cited code and confirm the finding still holds. If it no longer holds (already fixed, or
      misread), add its key to `attempted` and skip.
   b. Implement the **minimal, behavior-preserving** edit. If the only real fix is a large/risky redesign, do NOT
      force it — add to `attempted`, record it as "deferred (needs redesign)", and skip.
   c. **Gate** in the davinci env: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest` then `mypy davinci_monet` then
      `black --check davinci_monet && isort --check-only davinci_monet`.
      - **Green** → `git commit` this fix alone (clear message; explicitly flag any behavior change;
        `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`).
      - **Red** → **revert cleanly** (`git restore --staged --worktree` the touched files; delete any new files) so
        the tree returns to the last good commit, record "fix failed the gate," and move on. Never leave the tree
        broken or commit a red gate.
   d. Add the finding key to `attempted` regardless of outcome.
5. Continue to the next iteration (re-review picks up anything the fixes surfaced or resolved).

## Hard stops (any one ends the loop)
- `converged: true` (goal reached).
- `maxIterations` iterations completed.
- No actionable at-bar finding remains (all attempted).
- Two consecutive iterations produce zero successful commits (thrashing) — stop and report.

## Wrap-up
Report concisely: iterations run, fixes committed (with one-line each + any behavior changes), findings
deferred/failed, and the final state (`converged` or which cap stopped it). Then **stop and hand back** — the user
reviews the branch and decides whether to push/merge. Do not push, do not merge to main, do not run `/design-review`
again automatically.
