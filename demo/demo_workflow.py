from __future__ import annotations

import argparse
import importlib.util
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from tardigrade import (
    BudgetConfig,
    CircuitBreakerConfig,
    DegradationConfig,
    DegradationPolicy,
    FailedStep,
    RetryConfig,
    SQLiteCheckpointStore,
    StepCostReport,
    Workflow,
    armor,
)

STORE = SQLiteCheckpointStore(Path(".tardigrade/demo-checkpoints.db"))
RUN_ID = "demo-run-001"
QUIET = False


@dataclass(frozen=True)
class DemoTiming:
    fetch_attempt_delay: float
    fetch_retry_delay: float
    enrich_primary_delay: float
    enrich_fallback_delay: float
    circuit_recovery_timeout: float
    analyze_delay: float
    summarize_delay: float
    run_intro_pause: float
    after_step_pause: float
    between_runs_pause: float
    initial_dashboard_delay: float
    final_dashboard_hold: float


SHORT_TIMING = DemoTiming(
    fetch_attempt_delay=0.3,
    fetch_retry_delay=0.2,
    enrich_primary_delay=0.2,
    enrich_fallback_delay=0.2,
    circuit_recovery_timeout=2.0,
    analyze_delay=0.2,
    summarize_delay=0.2,
    run_intro_pause=0.0,
    after_step_pause=0.0,
    between_runs_pause=0.5,
    initial_dashboard_delay=1.0,
    final_dashboard_hold=2.5,
)

VIDEO_TIMING = DemoTiming(
    fetch_attempt_delay=4.0,
    fetch_retry_delay=6.0,
    enrich_primary_delay=4.0,
    enrich_fallback_delay=4.0,
    circuit_recovery_timeout=40.0,
    analyze_delay=5.0,
    summarize_delay=5.0,
    run_intro_pause=2.0,
    after_step_pause=3.0,
    between_runs_pause=6.0,
    initial_dashboard_delay=3.0,
    final_dashboard_hold=26.0,
)


def sleep_for(duration_seconds: float) -> None:
    if duration_seconds > 0:
        time.sleep(duration_seconds)


def timing_for_profile(profile: str) -> DemoTiming:
    if profile == "video":
        return VIDEO_TIMING
    return SHORT_TIMING


class DemoScenario:
    def __init__(self, timing: DemoTiming) -> None:
        self.timing = timing
        self.fetch_attempts = 0
        self.enrich_primary_calls = 0
        self.fetch_data = armor(
            name="fetch_data",
            retry=RetryConfig(
                max_attempts=3,
                base_delay=self.timing.fetch_retry_delay,
                jitter=False,
            ),
        )(self._fetch_data)
        self.enrich = armor(
            name="enrich",
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=self.timing.circuit_recovery_timeout,
                success_threshold=1,
                fallback=self._fallback_enrich,
                monitored_exceptions=(ConnectionError,),
            ),
        )(self._enrich)
        self.analyze = armor(name="analyze")(self._analyze)
        self.summarize = armor(name="summarize")(self._summarize)

    fetch_data: Callable[[str], dict[str, object]]
    enrich: Callable[[dict[str, object]], object]
    analyze: Callable[[dict[str, object]], dict[str, object]]
    summarize: Callable[[dict[str, object]], object]

    def pause_after_step(self) -> None:
        sleep_for(self.timing.after_step_pause)

    def pause_before_run(self) -> None:
        sleep_for(self.timing.run_intro_pause)

    def _fallback_enrich(
        self, data: dict[str, object]
    ) -> tuple[dict[str, object], StepCostReport]:
        sleep_for(self.timing.enrich_fallback_delay)
        say("⚡ enrich: primary open, switching to fallback model")
        return (
            {"items": data["items"], "source": "fallback"},
            StepCostReport(input_tokens=250, output_tokens=120, model="gpt-5-mini"),
        )

    def _fetch_data(self, url: str) -> tuple[dict[str, object], StepCostReport]:
        self.fetch_attempts += 1
        sleep_for(self.timing.fetch_attempt_delay)
        if self.fetch_attempts == 1:
            say("↻ fetch_data: transient timeout, retrying")
            raise TimeoutError(f"timed out calling {url}")
        say("✓ fetch_data: fetched source records")
        return (
            {"items": [1, 2, 3], "url": url},
            StepCostReport(input_tokens=400, output_tokens=150, model="gpt-5-mini"),
        )

    def _enrich(self, data: dict[str, object]) -> tuple[dict[str, object], StepCostReport]:
        self.enrich_primary_calls += 1
        sleep_for(self.timing.enrich_primary_delay)
        say("✗ enrich: primary provider is down")
        raise ConnectionError("enrichment provider unavailable")

    def _analyze(self, data: dict[str, object]) -> tuple[dict[str, object], StepCostReport]:
        sleep_for(self.timing.analyze_delay)
        say("✓ analyze: built intermediate analysis")
        return (
            {"count": len(data["items"]), "url": data["url"]},
            StepCostReport(input_tokens=900, output_tokens=250, model="gpt-5.4"),
        )

    def _summarize(self, analysis: dict[str, object]) -> tuple[str, StepCostReport]:
        sleep_for(self.timing.summarize_delay)
        say("✓ summarize: generated workflow summary")
        return (
            f"Processed {analysis['count']} items from {analysis['url']}",
            StepCostReport(input_tokens=700, output_tokens=180, model="gpt-5.4"),
        )


def say(message: str) -> None:
    if not QUIET:
        print(message)


def format_usd(amount: float) -> str:
    precision = 5 if abs(amount) < 0.01 else 4
    quantum = Decimal("1").scaleb(-precision)
    rounded = Decimal(str(amount)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{precision}f}"


def run_pipeline(run_label: str, scenario: DemoScenario) -> Workflow:
    say(f"\n{run_label}")
    scenario.pause_before_run()
    with Workflow(
        "demo-pipeline",
        run_id=RUN_ID,
        store=STORE,
        budget=BudgetConfig(max_budget_usd=1.00),
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ) as workflow:
        raw = scenario.fetch_data("https://api.example.com")
        scenario.pause_after_step()
        enriched = scenario.enrich(raw)
        scenario.pause_after_step()
        analysis = scenario.analyze(raw)
        scenario.pause_after_step()
        summary = scenario.summarize(analysis)
        scenario.pause_after_step()

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


def run_demo_sequence(timing: DemoTiming) -> Workflow:
    scenario = DemoScenario(timing)
    run_pipeline("Run 1: retry, failure, partial result", scenario)
    sleep_for(timing.between_runs_pause)
    second = run_pipeline("Run 2: checkpoint restore and fallback recovery", scenario)

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


def dashboard_available() -> bool:
    return importlib.util.find_spec("textual") is not None


def should_use_dashboard(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    if os.environ.get("CI"):
        return False
    return dashboard_available()


def main() -> None:
    global QUIET

    parser = argparse.ArgumentParser(description="Run the Tardigrade demo workflow.")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        dest="dashboard",
        help="Force the Tardigrade dashboard on.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_false",
        dest="dashboard",
        help="Force plain terminal output without the dashboard.",
    )
    parser.add_argument(
        "--profile",
        choices=("short", "video"),
        default="short",
        help=(
            "Demo pacing profile. Use 'video' for a narrated recording "
            "that runs about 90 seconds."
        ),
    )
    parser.set_defaults(dashboard=None)
    args = parser.parse_args()

    use_dashboard = should_use_dashboard(args.dashboard)
    timing = timing_for_profile(args.profile)

    if use_dashboard and not dashboard_available():
        raise SystemExit(
            'Dashboard dependencies are not installed. Run: pip install "tardigrade-ai[dashboard]"'
        )
    if not use_dashboard and args.dashboard is None and not dashboard_available():
        print(
            "Dashboard dependencies not installed. "
            "For the full UI, run: uv run --extra dashboard python demo/demo_workflow.py"
        )

    QUIET = use_dashboard
    STORE.clear_run("demo-pipeline", RUN_ID)

    if use_dashboard:
        from tardigrade import Dashboard

        dashboard = Dashboard()
        errors: list[BaseException] = []

        def producer() -> None:
            try:
                sleep_for(timing.initial_dashboard_delay)
                run_demo_sequence(timing)
                sleep_for(timing.final_dashboard_hold)
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

    run_demo_sequence(timing)
    STORE.close()


if __name__ == "__main__":
    main()
