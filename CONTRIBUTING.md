# Contributing

## Development Setup

```bash
git clone https://github.com/cole-godfrey/agentarmor.git
cd agentarmor
uv sync --all-extras
uv run pytest
uv run mypy src/ --strict
uv run ruff check src/ tests/
```

## Code Style

- Ruff handles formatting and linting.
- Mypy runs in strict mode.
- Every public function should be fully typed.
- Prefer small, composable modules over feature blobs.

## Testing

- Every resilience feature needs sync and async coverage.
- Target 90%+ coverage for shipped code.
- Use `structlog.testing.capture_logs()` to verify emitted events.
- Preserve backward compatibility when a new workflow policy is optional.

## Pull Request Process

1. Branch from `main`.
2. Add or update tests for the change.
3. Run `uv run pytest`, `uv run mypy src/ --strict`, and `uv run ruff check src/ tests/`.
4. Describe the behavior change clearly in the pull request.

## Adding a New Resilience Feature

Follow the existing pattern:

1. Add config types in `src/agentarmor/_types.py`.
2. Implement feature logic in a focused module.
3. Integrate it through the `@armor` wrapper in `src/agentarmor/_decorator.py`.
4. Emit structured log events for the dashboard and logs.
5. Update `docs/api.md`, examples, and relevant dashboard widgets if needed.

## Reporting Issues

Use the bug report template and include:

- Python version
- AgentArmor version
- Minimal reproduction script
- Expected behavior
- Actual behavior
- Relevant logs or traceback
