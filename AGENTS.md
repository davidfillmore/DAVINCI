# AGENTS.md - DAVINCI

Rules for all AI agents working in this repository. **Read `CLAUDE.md` for full project context** — architecture, conventions, gotchas, config patterns, and styling.

## General Rules

- **Read before editing**: Always read a file before modifying it
- **Preserve existing patterns**: Match the style and conventions of surrounding code
- **Non-destructive edits**: Do not remove user data or code unless explicitly requested
- **Keep the tree lean**: No archived binaries, scratch files, or generated artifacts checked in

## Git Workflow

- **NEVER auto commit or push**: Wait for explicit user confirmation
- **NEVER merge to main**: Only the user decides when to merge
- **After merge, return to develop**: Always switch back to `develop` branch

## Cross-Model Handoff Convention

This repo uses cross-model code reviews and hand-offs. Check for `REVIEW_*.md` or `HANDOFF_*.md` in the repo root at session start.

When writing handoff files, use `REVIEW_<MODEL>.md` or `HANDOFF_<TOPIC>.md` with these sections:

- **Context** — Branch, task, files involved
- **Changes Made** — What was done, with file paths and line references
- **Decisions & Rationale** — Why choices were made (highest-value section)
- **Open Questions / Concerns** — What the next model should investigate
- **Suggested Next Steps** — Specific actionable items

Do NOT track handoff files in git — they are ephemeral working artifacts. Delete once the handoff is complete.

## Planning and Implementation

- **Stop after planning**: After a planning session, always stop and wait for user conversation before proceeding to implementation

## Quick Validation

```bash
conda activate davinci
pytest
mypy davinci_monet
black --check davinci_monet && isort --check davinci_monet
```
