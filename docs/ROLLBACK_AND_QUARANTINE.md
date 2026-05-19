# Rollback and quarantine plan

## Before editing

Record:

```bash
git status --short
git rev-parse --short HEAD || true
```

## If phase fails

1. Stop.
2. Record the failure in `docs/release-evidence/v0.2.0/validation_results.md`.
3. Add unresolved item to the final audit report.
4. Revert unsafe file or quarantine experimental code behind a feature.
5. Re-run targeted tests.
6. Do not weaken the acceptance gate.

## Rollback commands

If a branch was created:

```bash
git diff > docs/release-evidence/v0.2.0/failed.diff
git restore .
git clean -fd
```

If not using git, copy changed-file list and restore from the preflight package/backup.

## Release rollback

No actual publish should happen. If publish was accidentally run, record the exact command and crate version immediately.
