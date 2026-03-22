# API Reference

## `@armor` decorator

Wrap a sync or async step with Tardigrade's interception layer. The decorator
can apply retries, circuit breakers, checkpoint integration, cost reporting,
and degradation-aware step tracking.

```python
from tardigrade import RetryConfig, armor

@armor(name="fetch", retry=RetryConfig(max_attempts=3))
def fetch(url: str) -> dict:
    return {"ok": True}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | `None` | Explicit step name. Falls back to `__qualname__`. |
| `retry` | `RetryConfig \| bool \| None` | `None` | Retry policy. `True` uses defaults. |
| `circuit_breaker` | `CircuitBreakerConfig \| None` | `None` | Circuit breaker plus optional fallback callable. |

## `RetryConfig`

Controls exponential backoff with jitter for transient failures.

```python
from tardigrade import RetryConfig

RetryConfig(max_attempts=4, base_delay=0.25, jitter=False)
```

| Field | Default | Description |
|---|---|---|
| `max_attempts` | `3` | Total attempts including the first call. |
| `base_delay` | `1.0` | Delay before the first retry in seconds. |
| `max_delay` | `60.0` | Maximum delay cap. |
| `exponential_base` | `2.0` | Multiplier applied per retry. |
| `jitter` | `True` | Adds randomized jitter. |
| `retryable_exceptions` | `(Exception,)` | Exceptions that trigger retries. |

## `CircuitBreakerConfig`

Protects a failing provider by opening the circuit and optionally routing to a
fallback callable with the same signature.

```python
from tardigrade import CircuitBreakerConfig

CircuitBreakerConfig(failure_threshold=3, fallback=my_backup_model)
```

| Field | Default | Description |
|---|---|---|
| `failure_threshold` | `5` | Consecutive failures before opening. |
| `recovery_timeout` | `30.0` | Seconds to wait before probing again. |
| `half_open_max_calls` | `1` | Allowed probe calls while half-open. |
| `success_threshold` | `2` | Consecutive successes needed to close. |
| `fallback` | `None` | Drop-in replacement callable used when open. |
| `monitored_exceptions` | `(Exception,)` | Exceptions that count against the breaker. |

## `Workflow`

Scopes checkpoints, cost tracking, and graceful degradation for a workflow run.

```python
from tardigrade import Workflow

with Workflow("pipeline", run_id="run-001") as wf:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `workflow_id` | `str` | Stable workflow name or pipeline identifier. |
| `run_id` | `str \| None` | Run identifier used for checkpoint resumption. |
| `store` | `CheckpointStore \| None` | Custom checkpoint backend. |
| `budget` | `BudgetConfig \| None` | Per-workflow budget configuration. |
| `pricing` | `dict[str, tuple[float, float]] \| None` | Pricing overrides. |
| `degradation` | `DegradationConfig \| None` | Workflow degradation policy. |

## `BudgetConfig`

Defines budget enforcement and warning behavior.

```python
from tardigrade import BudgetConfig, BudgetPolicy

BudgetConfig(max_budget_usd=1.00, policy=BudgetPolicy.HARD_STOP)
```

| Field | Default | Description |
|---|---|---|
| `max_budget_usd` | required | Hard spend ceiling. |
| `policy` | `HARD_STOP` | `HARD_STOP`, `WARN`, or `DEGRADE`. |
| `warn_at_fraction` | `0.8` | Warning threshold as a fraction of the budget. |
| `on_degrade` | `None` | Callback called when `DEGRADE` triggers. |

## `DegradationConfig`

Controls how workflows behave when individual steps fail.

```python
from tardigrade import DegradationConfig, DegradationPolicy

DegradationConfig(policy=DegradationPolicy.COLLECT, max_failures=2)
```

| Field | Default | Description |
|---|---|---|
| `policy` | `COLLECT` | `RAISE`, `COLLECT`, or `COLLECT_AND_STOP`. |
| `skip_dependent` | `True` | Auto-skip steps that receive a `FailedStep`. |
| `on_step_failure` | `None` | Callback invoked with `(step_name, exception)`. |
| `max_failures` | `None` | Escalates to stop mode after N failures. |

## `WorkflowResult`

Represents the final workflow outcome, including partial results.

```python
result = wf.result
if result and result.is_partial:
    print(result.failed_steps)
```

| Field / Property | Description |
|---|---|
| `status` | `"completed"`, `"partial"`, or `"failed"`. |
| `steps` | Ordered `StepResult` entries. |
| `completed_steps` | Completed or checkpoint-restored steps. |
| `failed_steps` | Failed step results. |
| `skipped_steps` | Steps skipped due to dependency failure or stop mode. |
| `get(step_name)` | Returns a completed value by step name. |

## `StepCostReport` and `report_cost()`

Use `StepCostReport` to report token usage and cost. You can return it as the
last tuple element or push it explicitly with `report_cost()`.

```python
from tardigrade import StepCostReport, armor, report_cost

@armor(name="summarize")
def summarize(text: str) -> str:
    report_cost(StepCostReport(input_tokens=500, output_tokens=200, model="gpt-5.4"))
    return "summary"
```

| Field | Default | Description |
|---|---|---|
| `input_tokens` | `0` | Input token count. |
| `output_tokens` | `0` | Output token count. |
| `model` | `""` | Model identifier used for pricing lookup. |
| `cost_usd` | `None` | Explicit cost override. |

## `Dashboard`

The optional Textual dashboard renders workflow progress, circuit state, cost
tracking, and the event log in real time.

```python
from tardigrade import Dashboard

Dashboard().start_in_thread()
```

| Method | Description |
|---|---|
| `start()` | Runs the dashboard app in the foreground. |
| `start_in_thread()` | Starts the dashboard on a daemon thread. |
| CLI | `tardigrade dashboard` |
