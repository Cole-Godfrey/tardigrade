from __future__ import annotations

import argparse
import threading
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from agentarmor import (
    BudgetConfig,
    CircuitBreakerConfig,
    Dashboard,
    DegradationConfig,
    DegradationPolicy,
    FailedStep,
    RetryConfig,
    SQLiteCheckpointStore,
    StepCostReport,
    Workflow,
    armor,
)

STORE = SQLiteCheckpointStore(Path(".agentarmor/demo-checkpoints.db"))
RUN_ID = "demo-run-001"
QUIET = False
FETCH_ATTEMPTS = 0
ENRICH_PRIMARY_CALLS = 0


def say(message: str) -> None:
    if not QUIET:
        print(message)


def format_usd(amount: float) -> str:
    precision = 5 if abs(amount) < 0.01 else 4
    quantum = Decimal("1").scaleb(-precision)
    rounded = Decimal(str(amount)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{precision}f}"


def fallback_enrich(data: dict[str, object]) -> tuple[dict[str, object], StepCostReport]:
    time.sleep(0.2)
    say("⚡ enrich: primary open, switching to fallback model")
    return (
        {"items": data["items"], "source": "fallback"},
        StepCostReport(input_tokens=250, output_tokens=120, model="gpt-5-mini"),
    )


@armor(
    name="fetch_data",
    retry=RetryConfig(max_attempts=3, base_delay=0.2, jitter=False),
)
def fetch_data(url: str) -> tuple[dict[str, object], StepCostReport]:
    global FETCH_ATTEMPTS
    FETCH_ATTEMPTS += 1
    time.sleep(0.3)
    if FETCH_ATTEMPTS == 1:
        say("↻ fetch_data: transient timeout, retrying")
        raise TimeoutError(f"timed out calling {url}")
    say("✓ fetch_data: fetched source records")
    return (
        {"items": [1, 2, 3], "url": url},
        StepCostReport(input_tokens=400, output_tokens=150, model="gpt-5-mini"),
    )


@armor(
    name="enrich",
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=1,
        recovery_timeout=2.0,
        success_threshold=1,
        fallback=fallback_enrich,
        monitored_exceptions=(ConnectionError,),
    ),
)
def enrich(data: dict[str, object]) -> tuple[dict[str, object], StepCostReport]:
    global ENRICH_PRIMARY_CALLS
    ENRICH_PRIMARY_CALLS += 1
    time.sleep(0.2)
    say("✗ enrich: primary provider is down")
    raise ConnectionError("enrichment provider unavailable")


@armor(name="analyze")
def analyze(data: dict[str, object]) -> tuple[dict[str, object], StepCostReport]:
    time.sleep(0.2)
    say("✓ analyze: built intermediate analysis")
    return (
        {"count": len(data["items"]), "url": data["url"]},
        StepCostReport(input_tokens=900, output_tokens=250, model="gpt-5.4"),
    )


@armor(name="summarize")
def summarize(analysis: dict[str, object]) -> tuple[str, StepCostReport]:
    time.sleep(0.2)
    say("✓ summarize: generated workflow summary")
    return (
        f"Processed {analysis['count']} items from {analysis['url']}",
        StepCostReport(input_tokens=700, output_tokens=180, model="gpt-5.4"),
    )


def run_pipeline(run_label: str) -> Workflow:
    say(f"\n{run_label}")
    with Workflow(
        "demo-pipeline",
        run_id=RUN_ID,
        store=STORE,
        budget=BudgetConfig(max_budget_usd=1.00),
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ) as workflow:
        raw = fetch_data("https://api.example.com")
        enriched = enrich(raw)
        analysis = analyze(raw)
        summary = summarize(analysis)

    assert workflow.result is not None
    say(f"status={workflow.result.status}")
    say(f"completed={[step.step_name for step in workflow.result.completed_steps]}")
    say(f"failed={[step.step_name for step in workflow.result.failed_steps]}")
    say(f"skipped={[step.step_name for step in workflow.result.skipped_steps]}")
    if not isinstance(enriched, FailedStep):
        say(f"enrich_source={enriched.get('source', 'primary')}")
    if not isinstance(summary, FailedStep):
        say(f"summary={summary}")
    return workflow


def run_demo_sequence() -> Workflow:
    run_pipeline("Run 1: retry, failure, partial result")
    time.sleep(0.5)
    second = run_pipeline("Run 2: checkpoint restore and fallback recovery")

    if not QUIET:
        print(f"\nTotal spend: ${format_usd(second.cost_tracker.total_usd)}")
        print("Per-step costs:")
        for entry in second.cost_tracker.step_costs:
            source = "checkpoint" if entry.get("restored_from_checkpoint") else "live"
            print(
                f"  - {entry['step_name']}: ${format_usd(entry['cost_usd'])} "
                f"({entry['model']}, source={source}, "
                f"cumulative=${format_usd(entry['cumulative_usd'])})"
            )

    return second


def main() -> None:
    global QUIET

    parser = argparse.ArgumentParser(description="Run the AgentArmor demo workflow.")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the AgentArmor dashboard in a background thread.",
    )
    args = parser.parse_args()

    QUIET = args.dashboard
    STORE.clear_run("demo-pipeline", RUN_ID)

    if args.dashboard:
        dashboard = Dashboard()
        errors: list[BaseException] = []

        def producer() -> None:
            try:
                time.sleep(1.0)
                run_demo_sequence()
                time.sleep(2.5)
            except BaseException as exc:  # pragma: no cover - demo runtime guard
                errors.append(exc)
            finally:
                STORE.close()
                dashboard._app.call_from_thread(dashboard._app.exit)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        dashboard.start()
        thread.join(timeout=2.0)
        if errors:
            raise errors[0]
        return

    run_demo_sequence()
    STORE.close()


if __name__ == "__main__":
    main()
