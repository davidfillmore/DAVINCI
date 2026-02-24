# AGENTS.md - DAVINCI-MONET

Rules and conventions for all AI agents working in this repository.

## General Rules

- **Non-destructive edits**: Do not remove user data or code unless explicitly requested
- **Keep the tree lean**: No archived binaries, scratch files, or generated artifacts checked in
- **Preserve existing patterns**: Match the style and conventions of surrounding code
- **Read before editing**: Always read a file before modifying it

## Git Workflow

- **NEVER auto commit or push**: Wait for explicit user confirmation
- **NEVER merge to main**: Only the user decides when to merge
- **After merge, return to develop**: Always switch back to `develop` branch

## Cross-Model Handoff Convention

This repo uses cross-model code reviews and hand-offs (e.g., Claude Opus implements, Codex reviews, then back). Follow these conventions so any model can pick up context cleanly.

### At Session Start

- Check for `REVIEW_*.md` or `HANDOFF_*.md` files in the repo root
- Read any that exist — they contain context from a previous model's work
- Read `git diff` and recent `git log` for additional context

### Writing Handoff Files

Use `REVIEW_<MODEL>.md` or `HANDOFF_<TOPIC>.md` in the repo root. Always include these sections:

```
## Context
Branch, task description, which files are involved

## Changes Made
What was done, with file paths and line references

## Decisions & Rationale
Why choices were made — this prevents the next model from undoing work without understanding why

## Open Questions / Concerns
Things the next model should investigate or address

## Suggested Next Steps
Specific actionable items
```

### Rules

- **One file per task/feature** — scoped context, not a running log
- **Always include Decisions & Rationale** — highest-value section; without it the next model may redo or reverse work
- **Reference file paths and line numbers** — so the next model can verify without searching
- **Git diff supplements the handoff** — the file gives intent, the diff gives the changes
- **Don't delete handoff files** — the user decides when they're no longer needed

## Planning and Implementation

- **Stop after planning**: After a planning session, always stop and wait for user conversation before proceeding to implementation
- **Follow CLAUDE.md**: All project-specific conventions in CLAUDE.md apply

## Quick Validation

```bash
# Activate environment
conda activate davinci-monet

# Run tests
pytest

# Type checking
mypy davinci_monet

# Format check
black --check davinci_monet && isort --check davinci_monet
```
