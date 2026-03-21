# Changelog

## [0.1.0] - 2026-03-21

### Added

- `@armor` decorator with structured logging for sync and async functions
- Automatic retries with exponential backoff and jitter via `RetryConfig`
- SQLite-backed checkpoint storage for workflow resumption via `Workflow` and `SQLiteCheckpointStore`
- Circuit breakers with automatic fallback model switching via `CircuitBreakerConfig`
- Cost budget enforcement with configurable policies via `BudgetConfig` and `StepCostReport`
- Graceful degradation with partial result collection via `DegradationConfig` and `WorkflowResult`
- Real-time terminal dashboard via `agentarmor[dashboard]`
- CLI entry point via `agentarmor dashboard`
