![Tardigrade banner](https://raw.githubusercontent.com/Cole-Godfrey/tardigrade/main/docs/assets/banner.png)

**Resilience middleware that makes AI agents self-healing.**

Retries · Checkpointing · Circuit breakers · Cost budgets · Graceful degradation  
Works with LangGraph, CrewAI, OpenAI SDK, or raw API calls.

[![PyPI version](https://img.shields.io/pypi/v/tardigrade-ai)](https://pypi.org/project/tardigrade-ai/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/Cole-Godfrey/tardigrade/ci.yml?label=tests)](https://github.com/Cole-Godfrey/tardigrade/actions)

![Tardigrade dashboard demo](https://raw.githubusercontent.com/Cole-Godfrey/tardigrade/main/docs/assets/hero.gif)

> An agent with 85% accuracy per step has just **20% end-to-end success** over 10 steps.  
> Tardigrade wraps your agent with production-grade resilience so step 7 failing  
> does not mean starting over from scratch.

## Install

```bash
pip install tardigrade-ai

# With the real-time dashboard:
pip install "tardigrade-ai[dashboard]"
```

## Quickstart

```python
from tardigrade import armor, Workflow, RetryConfig, BudgetConfig, StepCostReport

@armor(name="fetch", retry=RetryConfig(max_attempts=3))
def fetch_data(url: str) -> dict:
    return {"result": "..."}, StepCostReport(input_tokens=500, output_tokens=200, model="gpt-5.4")

@armor(name="analyze")
def analyze(data: dict) -> str:
    return "analysis", StepCostReport(input_tokens=2000, output_tokens=1000, model="gpt-5.4")

with Workflow("my_pipeline", budget=BudgetConfig(max_budget_usd=1.00)) as wf:
    data = fetch_data("https://api.example.com")
    result = analyze(data)
```

If the workflow crashes, rerun it with the same `run_id` and Tardigrade resumes
from the last checkpoint instead of starting from scratch.

## Features

| Feature | What it does | Docs |
|---|---|---|
| **Automatic retries** | Exponential backoff with jitter. Configurable per step. | [API](docs/api.md#armor-decorator) |
| **Checkpointing** | Resume multi-step workflows from the last successful step. | [API](docs/api.md#workflow) |
| **Circuit breakers** | Switch to a fallback model when your primary provider is down. | [API](docs/api.md#circuitbreakerconfig) |
| **Cost budgets** | Hard spending limits per workflow with policy-driven enforcement. | [API](docs/api.md#budgetconfig) |
| **Graceful degradation** | Collect partial results instead of crashing on step failures. | [API](docs/api.md#degradationconfig) |
| **Real-time dashboard** | Terminal UI showing workflow progress, costs, and circuit states. | [API](docs/api.md#dashboard) |
| **Framework-agnostic** | Works with any Python code. No lock-in to LangGraph or CrewAI. | [Examples](docs/examples) |
| **Structured logging** | Every event is a structured log event via `structlog`. | [API](docs/api.md#armor-decorator) |

## Before and After

| | Without Tardigrade | With Tardigrade |
|---|---|---|
| Step 7 of 10 fails | Start over. Waste $3 and 4 minutes. | Resume from step 7. About $0.15 and 30 seconds. |
| Provider goes down | Cascade failure across all agents. | Circuit breaker routes to a backup model. |
| Agent runs away on tokens | Surprise bills and no hard stop. | Hard budget cap with degradation or stop. |
| Debugging a failure | Wall of JSON, grep, and prayer. | Dashboard plus structured event history. |

## Examples

### Retry a flaky step

```python
from tardigrade import RetryConfig, armor

class RateLimitedError(RuntimeError):
    pass

@armor(retry=RetryConfig(max_attempts=4, retryable_exceptions=(RateLimitedError, TimeoutError)))
def call_provider(prompt: str) -> str:
    ...
```

### Switch to a fallback model when the primary is down

```python
from tardigrade import CircuitBreakerConfig, armor

def call_claude_haiku(prompt: str) -> str:
    return "fallback response"

@armor(circuit_breaker=CircuitBreakerConfig(failure_threshold=3, fallback=call_claude_haiku))
def call_gpt54(prompt: str) -> str:
    ...
```

### Resume a checkpointed workflow

```python
from tardigrade import Workflow, armor

@armor(name="fetch")
def fetch() -> str:
    return "data"

with Workflow("pipeline", run_id="run-001"):
    fetch()
```

### Enforce a workflow budget

```python
from tardigrade import BudgetConfig, StepCostReport, Workflow, armor

@armor(name="summarize")
def summarize(text: str) -> str:
    return "summary", StepCostReport(input_tokens=500, output_tokens=200, model="gpt-5.4")

with Workflow("pipeline", budget=BudgetConfig(max_budget_usd=0.50)):
    summarize("hello")
```

### Return partial results instead of crashing

```python
from tardigrade import DegradationConfig, DegradationPolicy, Workflow, armor

@armor(name="enrich")
def enrich(data: dict) -> dict:
    raise ConnectionError("provider down")

with Workflow("pipeline", degradation=DegradationConfig(policy=DegradationPolicy.COLLECT)) as wf:
    enrich({"id": 1})

assert wf.result is not None
assert wf.result.status == "failed"
```

Extended runnable examples live in [docs/examples](docs/examples).

## Dashboard

![Tardigrade dashboard demo](https://raw.githubusercontent.com/Cole-Godfrey/tardigrade/main/docs/assets/hero.gif)

```bash
# CLI
tardigrade dashboard
```

```python
# Programmatic
from tardigrade import Dashboard

Dashboard().start_in_thread()
```

## Demo

Run the self-contained demo script:

```bash
uv run demo/demo_workflow.py
```

Record the hero GIF when `vhs` is installed:

```bash
vhs demo/hero.tape
```

## API and Project Docs

- [API reference](docs/api.md)
- [Contributing guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)
