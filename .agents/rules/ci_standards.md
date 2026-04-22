# CI Standards & Workflow Rules

To maintain high code quality and ensure the GitHub actions/workflows pass, the following rules must be strictly followed before any Git push:

## Verification Checklist
1. **Linting**: Run `ruff check .` to identify any code quality issues.
2. **Formatting**: Run `ruff format .` to ensure consistent code styling.
3. **Type Checking** (Optional but recommended): Run `mypy .` if configured.
4. **Testing**: Run `pytest` or `scripts/test_components.py` to verify no regressions.

## Command Reference
```powershell
# Run from repository root
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format .
```

## Push Condition
> [!IMPORTANT]
> Never push changes to GitHub if `ruff check` returns any errors or warnings. Fix all linting issues immediately.
