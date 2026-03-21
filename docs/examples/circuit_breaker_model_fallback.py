from __future__ import annotations

import itertools
import time

from agentarmor import (
    CircuitBreakerConfig,
    DegradationConfig,
    DegradationPolicy,
    Workflow,
    armor,
)

failures = itertools.count()


def call_claude_haiku_45(prompt: str) -> str:
    time.sleep(0.1)
    return f"Claude Haiku 4.5 fallback handled: {prompt}"


@armor(
    name="primary_model",
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=1.0,
        success_threshold=1,
        fallback=call_claude_haiku_45,
        monitored_exceptions=(RuntimeError,),
    ),
)
def call_gpt54(prompt: str) -> str:
    time.sleep(0.1)
    if next(failures) < 2:
        raise RuntimeError("OpenAI provider unavailable")
    return f"GPT-5.4 handled: {prompt}"


if __name__ == "__main__":
    with Workflow(
        "model-fallback-demo",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ):
        for _ in range(4):
            print(call_gpt54("Summarize the incident report"))
