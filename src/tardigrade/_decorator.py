from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextvars import Token
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from hashlib import sha256
from inspect import isawaitable
from time import perf_counter
from typing import Any, TypeGuard, cast, overload
from uuid import uuid4

import structlog

from ._checkpoint import CheckpointMetadataStore, CheckpointStore
from ._circuit_breaker import CircuitBreaker
from ._context import ArmorContext, reset_current_armor_context, set_current_armor_context
from ._logging import configure_logging
from ._result import FailedStep, StepResult, StepStatus
from ._serializer import deserialize_result, serialize_result
from ._types import (
    ArmorResult,
    BudgetConfig,
    CheckpointConfig,
    CircuitBreakerConfig,
    P,
    R,
    RetryConfig,
    StepCostReport,
    TardigradeBudgetExceededError,
    TardigradeCircuitOpenError,
)
from ._workflow import Workflow, get_current_workflow


def _safe_repr(value: object) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"


def _build_args_hash(args_repr: list[str], kwargs_repr: dict[str, str]) -> str:
    payload = {"args": args_repr, "kwargs": kwargs_repr}
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _CheckpointState:
    workflow_id: str
    run_id: str
    step_name: str
    store: CheckpointStore
    config: CheckpointConfig


@dataclass(slots=True)
class _InvocationState:
    context: ArmorContext
    token: Token[ArmorContext | None]
    started_at: float
    args_hash: str
    checkpoint_state: _CheckpointState | None


@dataclass(slots=True)
class _RetryState:
    invocation: _InvocationState
    config: RetryConfig
    attempts_made: int = 0
    last_exception: BaseException | None = None


configure_logging()
_logger = structlog.get_logger("tardigrade")
_NO_CHECKPOINT = object()


def _resolve_function_name(target: Callable[..., object], name: str | None) -> str:
    if name is not None:
        return name
    return target.__qualname__


def _resolve_checkpoint_state(
    workflow: Workflow | None,
    step_name: str,
) -> _CheckpointState | None:
    if workflow is None:
        return None

    return _CheckpointState(
        workflow_id=workflow.workflow_id,
        run_id=workflow.run_id,
        step_name=step_name,
        store=workflow.store,
        config=CheckpointConfig(),
    )


def _checkpoint_log_fields(checkpoint_state: _CheckpointState | None) -> dict[str, str]:
    if checkpoint_state is None:
        return {}
    return {
        "workflow_id": checkpoint_state.workflow_id,
        "run_id": checkpoint_state.run_id,
        "step_name": checkpoint_state.step_name,
    }


def _supports_checkpoint_metadata(store: object) -> TypeGuard[CheckpointMetadataStore]:
    return all(
        hasattr(store, attribute)
        for attribute in (
            "save_metadata",
            "load_metadata",
            "asave_metadata",
            "aload_metadata",
        )
    )


def _serialize_checkpoint_cost_report(
    report: StepCostReport | None,
    cost_usd: float,
) -> bytes | None:
    if report is None:
        return None

    persisted_report = StepCostReport(
        input_tokens=report.input_tokens,
        output_tokens=report.output_tokens,
        model=report.model,
        cost_usd=cost_usd,
    )
    return serialize_result(persisted_report)


def _deserialize_checkpoint_cost_report(payload: bytes | None) -> StepCostReport | None:
    if payload is None:
        return None

    report = deserialize_result(payload)
    if isinstance(report, StepCostReport):
        return report

    _logger.warning(
        "checkpoint_cost_metadata_invalid",
        metadata_type=type(report).__name__,
    )
    return None


def _load_checkpoint_cost_report(checkpoint_state: _CheckpointState) -> StepCostReport | None:
    store = checkpoint_state.store
    if not _supports_checkpoint_metadata(store):
        return None

    return _deserialize_checkpoint_cost_report(
        store.load_metadata(
            checkpoint_state.workflow_id,
            checkpoint_state.step_name,
            checkpoint_state.run_id,
        )
    )


async def _aload_checkpoint_cost_report(
    checkpoint_state: _CheckpointState,
) -> StepCostReport | None:
    store = checkpoint_state.store
    if not _supports_checkpoint_metadata(store):
        return None

    return _deserialize_checkpoint_cost_report(
        await store.aload_metadata(
            checkpoint_state.workflow_id,
            checkpoint_state.step_name,
            checkpoint_state.run_id,
        )
    )


def _replay_checkpoint_cost(
    workflow: Workflow | None,
    step_name: str,
    report: StepCostReport | None,
) -> float:
    if workflow is None or report is None:
        return 0.0

    return workflow.cost_tracker.record(
        step_name,
        report,
        restored_from_checkpoint=True,
    )


def _build_context_config(
    retry_config: RetryConfig | None,
    checkpoint_state: _CheckpointState | None,
    circuit_breaker_config: CircuitBreakerConfig | None,
    budget_config: BudgetConfig | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if retry_config is not None:
        config["retry"] = retry_config
    if circuit_breaker_config is not None:
        config["circuit_breaker"] = circuit_breaker_config
    if budget_config is not None:
        config["budget"] = budget_config
    if checkpoint_state is not None:
        config["checkpoint"] = checkpoint_state.config
        config["workflow_id"] = checkpoint_state.workflow_id
        config["run_id"] = checkpoint_state.run_id
        config["step_name"] = checkpoint_state.step_name
    return config


def _begin_call(
    function_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    retry_config: RetryConfig | None,
    checkpoint_state: _CheckpointState | None,
    circuit_breaker_config: CircuitBreakerConfig | None,
    budget_config: BudgetConfig | None,
) -> _InvocationState:
    max_attempts = 1 if retry_config is None else retry_config.max_attempts
    context = ArmorContext(
        function_name=function_name,
        call_id=str(uuid4()),
        attempt=1,
        max_attempts=max_attempts,
        start_time=datetime.now(UTC),
        config=_build_context_config(
            retry_config,
            checkpoint_state,
            circuit_breaker_config,
            budget_config,
        ),
    )
    token = set_current_armor_context(context)

    args_repr = [_safe_repr(arg) for arg in args]
    kwargs_repr = {key: _safe_repr(value) for key, value in sorted(kwargs.items())}
    args_hash = _build_args_hash(args_repr, kwargs_repr)

    _logger.info(
        "step_started",
        function_name=context.function_name,
        call_id=context.call_id,
        attempt=context.attempt,
        max_attempts=context.max_attempts,
        status="started",
        args=args_repr,
        kwargs=kwargs_repr,
        args_hash=args_hash,
        **_checkpoint_log_fields(checkpoint_state),
    )

    return _InvocationState(
        context=context,
        token=token,
        started_at=perf_counter(),
        args_hash=args_hash,
        checkpoint_state=checkpoint_state,
    )


def _duration_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000


def _log_success(state: _InvocationState, result: object) -> ArmorResult:
    envelope = ArmorResult(
        value=result,
        duration_ms=_duration_ms(state.started_at),
        status="success",
        exception=None,
        timestamp=datetime.now(UTC),
    )
    _logger.info(
        "step_completed",
        function_name=state.context.function_name,
        call_id=state.context.call_id,
        duration_ms=envelope.duration_ms,
        attempt=state.context.attempt,
        max_attempts=state.context.max_attempts,
        status=envelope.status,
        args_hash=state.args_hash,
        result=_safe_repr(envelope.value),
        **_checkpoint_log_fields(state.checkpoint_state),
    )
    return envelope


def _log_error(state: _InvocationState, exc: BaseException) -> ArmorResult:
    envelope = ArmorResult(
        value=None,
        duration_ms=_duration_ms(state.started_at),
        status="error",
        exception=exc,
        timestamp=datetime.now(UTC),
    )
    _logger.info(
        "step_failed",
        function_name=state.context.function_name,
        call_id=state.context.call_id,
        duration_ms=envelope.duration_ms,
        attempt=state.context.attempt,
        max_attempts=state.context.max_attempts,
        status=envelope.status,
        args_hash=state.args_hash,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        **_checkpoint_log_fields(state.checkpoint_state),
    )
    return envelope


def _finalize_call(state: _InvocationState) -> None:
    reset_current_armor_context(state.token)


def _resolve_retry_config(retry: RetryConfig | bool | None) -> RetryConfig | None:
    if retry is None or retry is False:
        return None
    if retry is True:
        return RetryConfig()
    return retry


def _resolve_callable_name(target: Callable[..., object]) -> str:
    qualname = getattr(target, "__qualname__", None)
    if isinstance(qualname, str):
        return qualname

    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        return name

    return type(target).__qualname__


def _is_monitored_circuit_exception(
    breaker: CircuitBreaker,
    exc: BaseException,
) -> bool:
    return isinstance(exc, breaker.config.monitored_exceptions)


def _log_circuit_fallback(
    state: _InvocationState,
    breaker: CircuitBreaker,
    fallback: Callable[..., Any],
) -> None:
    _logger.info(
        "circuit_fallback",
        function_name=state.context.function_name,
        call_id=state.context.call_id,
        duration_ms=_duration_ms(state.started_at),
        attempt=state.context.attempt,
        max_attempts=state.context.max_attempts,
        args_hash=state.args_hash,
        fallback_name=_resolve_callable_name(fallback),
        circuit_state=breaker.state.value,
        **_checkpoint_log_fields(state.checkpoint_state),
    )


async def _await_fallback_result(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def _invoke_sync_fallback(
    fallback: Callable[..., Any],
    /,
    *args: object,
    **kwargs: object,
) -> Any:
    result: object = fallback(*args, **kwargs)
    if isawaitable(result):
        return asyncio.run(_await_fallback_result(result))
    return result


async def _invoke_async_fallback(
    fallback: Callable[..., Any],
    /,
    *args: object,
    **kwargs: object,
) -> Any:
    result: object = fallback(*args, **kwargs)
    if isawaitable(result):
        return await result
    return result


def _attach_circuit_breaker(
    wrapper: Callable[..., Any],
    breaker: CircuitBreaker | None,
) -> None:
    if breaker is not None:
        wrapper_obj = cast(Any, wrapper)
        wrapper_obj._circuit_breaker = breaker


def _extract_cost_report(
    result: object,
    context: ArmorContext,
) -> tuple[object, StepCostReport | None]:
    extracted_report = context.cost_report
    if (
        isinstance(result, tuple)
        and len(result) >= 2
        and isinstance(result[-1], StepCostReport)
    ):
        extracted_report = result[-1]
        remaining = result[:-1]
        if len(remaining) == 1:
            return remaining[0], extracted_report
        return remaining, extracted_report
    return result, extracted_report


def _record_cost_report(
    step_name: str,
    workflow: Workflow | None,
    report: StepCostReport | None,
) -> tuple[float, TardigradeBudgetExceededError | None]:
    if report is None:
        return 0.0, None

    if workflow is None:
        _logger.debug(
            "step_cost_reported_no_workflow",
            step_name=step_name,
            model=report.model,
            input_tokens=report.input_tokens,
            output_tokens=report.output_tokens,
            cost_usd=report.cost_usd,
        )
        return 0.0, None

    cost_usd = workflow.cost_tracker.record(step_name, report)
    try:
        workflow.cost_tracker.check_budget(workflow.workflow_id)
    except TardigradeBudgetExceededError as exc:
        return cost_usd, exc
    return cost_usd, None


def _clear_cost_report(invocation: _InvocationState) -> None:
    invocation.context.cost_report = None


def _find_failed_step_argument(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> FailedStep | None:
    for value in args:
        if isinstance(value, FailedStep):
            return value
    for value in kwargs.values():
        if isinstance(value, FailedStep):
            return value
    return None


def _record_step_success(
    workflow: Workflow | None,
    step_name: str,
    value: object,
    duration_ms: float,
    attempt: int,
    cost_usd: float,
    *,
    from_checkpoint: bool,
) -> None:
    if workflow is None:
        return

    workflow.record_step_result(
        StepResult(
            step_name=step_name,
            status=(
                StepStatus.CHECKPOINT_RESTORED
                if from_checkpoint
                else StepStatus.COMPLETED
            ),
            value=value,
            duration_ms=duration_ms,
            attempt=attempt,
            cost_usd=cost_usd,
            from_checkpoint=from_checkpoint,
        )
    )


def _record_step_failure(
    workflow: Workflow | None,
    step_name: str,
    exc: BaseException,
    duration_ms: float,
    attempt: int,
) -> None:
    if workflow is None:
        return

    workflow.record_step_result(
        StepResult(
            step_name=step_name,
            status=StepStatus.FAILED,
            exception=exc,
            duration_ms=duration_ms,
            attempt=attempt,
        )
    )


def _record_step_skipped(
    workflow: Workflow | None,
    step_name: str,
    exc: BaseException,
) -> None:
    if workflow is None:
        return

    workflow.record_step_result(
        StepResult(
            step_name=step_name,
            status=StepStatus.SKIPPED,
            exception=exc,
        )
    )


def _log_step_skipped_dependency(step_name: str, failed_dependency: str) -> None:
    _logger.info(
        "step_skipped_dependency",
        step_name=step_name,
        failed_dependency=failed_dependency,
    )


def _log_step_skipped_max_failures(
    step_name: str,
    workflow: Workflow,
) -> None:
    _logger.info(
        "step_skipped_max_failures",
        step_name=step_name,
        failure_count=workflow.failure_count,
        max_failures=workflow.degradation_config.max_failures,
    )


def _log_step_failed_degraded(step_name: str, exc: BaseException) -> None:
    _logger.info(
        "step_failed_degraded",
        step_name=step_name,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
    )


def _restore_from_checkpoint(checkpoint_state: _CheckpointState) -> object:
    payload = checkpoint_state.store.load(
        checkpoint_state.workflow_id,
        checkpoint_state.step_name,
        checkpoint_state.run_id,
    )
    if payload is None:
        return _NO_CHECKPOINT

    result = deserialize_result(payload)
    _logger.info(
        "step_restored_from_checkpoint",
        function_name=checkpoint_state.step_name,
        duration_ms=0.0,
        status="restored",
        **_checkpoint_log_fields(checkpoint_state),
    )
    return result


async def _arestore_from_checkpoint(checkpoint_state: _CheckpointState) -> object:
    payload = await checkpoint_state.store.aload(
        checkpoint_state.workflow_id,
        checkpoint_state.step_name,
        checkpoint_state.run_id,
    )
    if payload is None:
        return _NO_CHECKPOINT

    result = deserialize_result(payload)
    _logger.info(
        "step_restored_from_checkpoint",
        function_name=checkpoint_state.step_name,
        duration_ms=0.0,
        status="restored",
        **_checkpoint_log_fields(checkpoint_state),
    )
    return result


def _checkpoint_result(
    state: _InvocationState,
    checkpoint_state: _CheckpointState,
    result: object,
    cost_report: StepCostReport | None,
    cost_usd: float,
) -> None:
    payload = serialize_result(result)
    checkpoint_state.store.save(
        checkpoint_state.workflow_id,
        checkpoint_state.step_name,
        checkpoint_state.run_id,
        payload,
    )
    if _supports_checkpoint_metadata(checkpoint_state.store):
        checkpoint_state.store.save_metadata(
            checkpoint_state.workflow_id,
            checkpoint_state.step_name,
            checkpoint_state.run_id,
            _serialize_checkpoint_cost_report(cost_report, cost_usd),
        )
    _logger.info(
        "step_checkpointed",
        function_name=state.context.function_name,
        call_id=state.context.call_id,
        duration_ms=_duration_ms(state.started_at),
        attempt=state.context.attempt,
        max_attempts=state.context.max_attempts,
        status="checkpointed",
        args_hash=state.args_hash,
        **_checkpoint_log_fields(checkpoint_state),
    )


async def _acheckpoint_result(
    state: _InvocationState,
    checkpoint_state: _CheckpointState,
    result: object,
    cost_report: StepCostReport | None,
    cost_usd: float,
) -> None:
    payload = serialize_result(result)
    await checkpoint_state.store.asave(
        checkpoint_state.workflow_id,
        checkpoint_state.step_name,
        checkpoint_state.run_id,
        payload,
    )
    if _supports_checkpoint_metadata(checkpoint_state.store):
        await checkpoint_state.store.asave_metadata(
            checkpoint_state.workflow_id,
            checkpoint_state.step_name,
            checkpoint_state.run_id,
            _serialize_checkpoint_cost_report(cost_report, cost_usd),
        )
    _logger.info(
        "step_checkpointed",
        function_name=state.context.function_name,
        call_id=state.context.call_id,
        duration_ms=_duration_ms(state.started_at),
        attempt=state.context.attempt,
        max_attempts=state.context.max_attempts,
        status="checkpointed",
        args_hash=state.args_hash,
        **_checkpoint_log_fields(checkpoint_state),
    )


def _set_attempt(state: _RetryState, attempt: int) -> None:
    state.attempts_made = attempt
    state.invocation.context.attempt = attempt


def _is_retryable_exception(state: _RetryState, exc: BaseException) -> bool:
    return isinstance(exc, state.config.retryable_exceptions)


def _log_retrying(state: _RetryState, exc: BaseException, delay_seconds: float) -> None:
    _logger.info(
        "step_retrying",
        function_name=state.invocation.context.function_name,
        call_id=state.invocation.context.call_id,
        duration_ms=_duration_ms(state.invocation.started_at),
        attempt=state.attempts_made,
        max_attempts=state.config.max_attempts,
        status="retrying",
        args_hash=state.invocation.args_hash,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        delay_seconds=delay_seconds,
        **_checkpoint_log_fields(state.invocation.checkpoint_state),
    )


def _log_retries_exhausted(state: _RetryState) -> None:
    last_exception = state.last_exception
    if last_exception is None:
        msg = "last_exception must be set before logging exhausted retries"
        raise RuntimeError(msg)

    total_elapsed_ms = _duration_ms(state.invocation.started_at)
    _logger.info(
        "step_failed_all_retries",
        function_name=state.invocation.context.function_name,
        call_id=state.invocation.context.call_id,
        duration_ms=total_elapsed_ms,
        status="error",
        args_hash=state.invocation.args_hash,
        total_attempts=state.attempts_made,
        total_elapsed_ms=total_elapsed_ms,
        last_exception_type=type(last_exception).__name__,
        last_exception_message=str(last_exception),
        **_checkpoint_log_fields(state.invocation.checkpoint_state),
    )


def _handle_retryable_failure(state: _RetryState, exc: BaseException) -> float | None:
    state.last_exception = exc
    if state.attempts_made >= state.config.max_attempts:
        _log_retries_exhausted(state)
        return None

    delay_seconds = state.config.delay_for_attempt(state.attempts_made)
    _log_retrying(state, exc, delay_seconds)
    return delay_seconds


def _run_sync_once(
    target: Callable[P, R],
    invocation: _InvocationState,
    log_success: bool = True,
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    _clear_cost_report(invocation)
    try:
        result = target(*args, **kwargs)
    except BaseException as exc:
        _log_error(invocation, exc)
        raise
    else:
        if log_success:
            _log_success(invocation, result)
        return result


def _run_sync_with_retry(
    target: Callable[P, R],
    invocation: _InvocationState,
    retry_config: RetryConfig,
    log_success: bool = True,
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    retry_state = _RetryState(invocation=invocation, config=retry_config)

    for attempt in range(1, retry_config.max_attempts + 1):
        _set_attempt(retry_state, attempt)
        _clear_cost_report(invocation)
        try:
            result = target(*args, **kwargs)
        except BaseException as exc:
            if not _is_retryable_exception(retry_state, exc):
                _log_error(invocation, exc)
                raise

            delay_seconds = _handle_retryable_failure(retry_state, exc)
            if delay_seconds is None:
                raise

            time.sleep(delay_seconds)
        else:
            if log_success:
                _log_success(invocation, result)
            return result

    msg = "retry loop exited without returning or raising"
    raise RuntimeError(msg)


async def _run_async_once(
    target: Callable[P, Awaitable[Any]],
    invocation: _InvocationState,
    log_success: bool = True,
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Any:
    _clear_cost_report(invocation)
    try:
        result = await target(*args, **kwargs)
    except BaseException as exc:
        _log_error(invocation, exc)
        raise
    else:
        if log_success:
            _log_success(invocation, result)
        return result


async def _run_async_with_retry(
    target: Callable[P, Awaitable[Any]],
    invocation: _InvocationState,
    retry_config: RetryConfig,
    log_success: bool = True,
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Any:
    retry_state = _RetryState(invocation=invocation, config=retry_config)

    for attempt in range(1, retry_config.max_attempts + 1):
        _set_attempt(retry_state, attempt)
        _clear_cost_report(invocation)
        try:
            result = await target(*args, **kwargs)
        except BaseException as exc:
            if not _is_retryable_exception(retry_state, exc):
                _log_error(invocation, exc)
                raise

            delay_seconds = _handle_retryable_failure(retry_state, exc)
            if delay_seconds is None:
                raise

            await asyncio.sleep(delay_seconds)
        else:
            if log_success:
                _log_success(invocation, result)
            return result

    msg = "retry loop exited without returning or raising"
    raise RuntimeError(msg)


@overload
def armor(func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def armor(
    *,
    name: str | None = None,
    retry: RetryConfig | bool | None = None,
    circuit_breaker: CircuitBreakerConfig | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def armor(
    func: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    retry: RetryConfig | bool | None = None,
    circuit_breaker: CircuitBreakerConfig | None = None,
) -> Any:
    def decorator(target: Callable[P, R]) -> Callable[P, R]:
        function_name = _resolve_function_name(target, name)
        retry_config = _resolve_retry_config(retry)
        breaker = CircuitBreaker(circuit_breaker) if circuit_breaker is not None else None
        if breaker is not None:
            breaker.bind(function_name)

        if asyncio.iscoroutinefunction(target):
            async_target = cast(Callable[P, Awaitable[Any]], target)

            @wraps(async_target)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                workflow = get_current_workflow()
                checkpoint_state = _resolve_checkpoint_state(workflow, function_name)
                if checkpoint_state is not None:
                    restored_result = await _arestore_from_checkpoint(checkpoint_state)
                    if restored_result is not _NO_CHECKPOINT:
                        restored_cost_usd = _replay_checkpoint_cost(
                            workflow,
                            function_name,
                            await _aload_checkpoint_cost_report(checkpoint_state),
                        )
                        _record_step_success(
                            workflow,
                            function_name,
                            restored_result,
                            0.0,
                            0,
                            restored_cost_usd,
                            from_checkpoint=True,
                        )
                        return restored_result

                if workflow is not None and workflow.should_degrade_failures():
                    if workflow.should_skip():
                        stop_exc = RuntimeError("workflow stopped after max failures")
                        _log_step_skipped_max_failures(function_name, workflow)
                        _record_step_skipped(workflow, function_name, stop_exc)
                        return FailedStep(function_name, stop_exc)

                    if workflow.degradation_config.skip_dependent:
                        failed_dependency = _find_failed_step_argument(args, dict(kwargs))
                        if failed_dependency is not None:
                            dependency_exc = RuntimeError(
                                f"dependency '{failed_dependency.step_name}' failed"
                            )
                            _log_step_skipped_dependency(
                                function_name,
                                failed_dependency.step_name,
                            )
                            _record_step_skipped(workflow, function_name, dependency_exc)
                            return FailedStep(function_name, dependency_exc)

                if workflow is not None:
                    workflow.cost_tracker.check_budget(workflow.workflow_id)

                state = _begin_call(
                    function_name,
                    args,
                    dict(kwargs),
                    retry_config,
                    checkpoint_state,
                    circuit_breaker,
                    workflow.cost_tracker.budget_config if workflow is not None else None,
                )
                actual_result: object = None
                step_cost_usd = 0.0
                budget_error: TardigradeBudgetExceededError | None = None
                try:
                    try:
                        if breaker is not None and not breaker.can_execute():
                            fallback = breaker.config.fallback
                            if fallback is None:
                                open_exc = TardigradeCircuitOpenError(
                                    function_name,
                                    breaker.state,
                                )
                                _log_error(state, open_exc)
                                raise open_exc

                            _log_circuit_fallback(state, breaker, fallback)
                            try:
                                result = await _invoke_async_fallback(fallback, *args, **kwargs)
                            except BaseException as exc:
                                _log_error(state, exc)
                                raise
                        else:
                            try:
                                if retry_config is None:
                                    result = await _run_async_once(
                                        async_target,
                                        state,
                                        False,
                                        *args,
                                        **kwargs,
                                    )
                                else:
                                    result = await _run_async_with_retry(
                                        async_target,
                                        state,
                                        retry_config,
                                        False,
                                        *args,
                                        **kwargs,
                                    )
                            except BaseException as exc:
                                if breaker is not None:
                                    if _is_monitored_circuit_exception(breaker, exc):
                                        breaker.record_failure()
                                    else:
                                        breaker.release_probe()
                                raise
                            else:
                                if breaker is not None:
                                    breaker.record_success()
                    except Exception as exc:
                        _record_step_failure(
                            workflow,
                            function_name,
                            exc,
                            _duration_ms(state.started_at),
                            state.context.attempt,
                        )
                        if workflow is not None and workflow.should_degrade_failures():
                            _log_step_failed_degraded(function_name, exc)
                            return FailedStep(function_name, exc)
                        raise

                    actual_result, cost_report = _extract_cost_report(result, state.context)
                    step_cost_usd, budget_error = _record_cost_report(
                        function_name,
                        workflow,
                        cost_report,
                    )

                    if checkpoint_state is not None:
                        await _acheckpoint_result(
                            state,
                            checkpoint_state,
                            actual_result,
                            cost_report,
                            step_cost_usd,
                        )

                    _log_success(state, actual_result)
                    _record_step_success(
                        workflow,
                        function_name,
                        actual_result,
                        _duration_ms(state.started_at),
                        state.context.attempt,
                        step_cost_usd,
                        from_checkpoint=False,
                    )
                finally:
                    _finalize_call(state)

                if budget_error is not None:
                    raise budget_error

                return actual_result

            _attach_circuit_breaker(async_wrapper, breaker)
            return cast(Callable[P, R], async_wrapper)

        @wraps(target)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            workflow = get_current_workflow()
            checkpoint_state = _resolve_checkpoint_state(workflow, function_name)
            if checkpoint_state is not None:
                restored_result = _restore_from_checkpoint(checkpoint_state)
                if restored_result is not _NO_CHECKPOINT:
                    restored_cost_usd = _replay_checkpoint_cost(
                        workflow,
                        function_name,
                        _load_checkpoint_cost_report(checkpoint_state),
                    )
                    _record_step_success(
                        workflow,
                        function_name,
                        restored_result,
                        0.0,
                        0,
                        restored_cost_usd,
                        from_checkpoint=True,
                    )
                    return cast(R, restored_result)

            if workflow is not None and workflow.should_degrade_failures():
                if workflow.should_skip():
                    stop_exc = RuntimeError("workflow stopped after max failures")
                    _log_step_skipped_max_failures(function_name, workflow)
                    _record_step_skipped(workflow, function_name, stop_exc)
                    return cast(R, FailedStep(function_name, stop_exc))

                if workflow.degradation_config.skip_dependent:
                    failed_dependency = _find_failed_step_argument(args, dict(kwargs))
                    if failed_dependency is not None:
                        dependency_exc = RuntimeError(
                            f"dependency '{failed_dependency.step_name}' failed"
                        )
                        _log_step_skipped_dependency(
                            function_name,
                            failed_dependency.step_name,
                        )
                        _record_step_skipped(workflow, function_name, dependency_exc)
                        return cast(R, FailedStep(function_name, dependency_exc))

            if workflow is not None:
                workflow.cost_tracker.check_budget(workflow.workflow_id)

            state = _begin_call(
                function_name,
                args,
                dict(kwargs),
                retry_config,
                checkpoint_state,
                circuit_breaker,
                workflow.cost_tracker.budget_config if workflow is not None else None,
            )
            actual_result: object = None
            step_cost_usd = 0.0
            budget_error: TardigradeBudgetExceededError | None = None
            try:
                try:
                    if breaker is not None and not breaker.can_execute():
                        fallback = breaker.config.fallback
                        if fallback is None:
                            open_exc = TardigradeCircuitOpenError(
                                function_name,
                                breaker.state,
                            )
                            _log_error(state, open_exc)
                            raise open_exc

                        _log_circuit_fallback(state, breaker, fallback)
                        try:
                            result = _invoke_sync_fallback(fallback, *args, **kwargs)
                        except BaseException as exc:
                            _log_error(state, exc)
                            raise
                    else:
                        try:
                            if retry_config is None:
                                result = _run_sync_once(
                                    target,
                                    state,
                                    False,
                                    *args,
                                    **kwargs,
                                )
                            else:
                                result = _run_sync_with_retry(
                                    target,
                                    state,
                                    retry_config,
                                    False,
                                    *args,
                                    **kwargs,
                                )
                        except BaseException as exc:
                            if breaker is not None:
                                if _is_monitored_circuit_exception(breaker, exc):
                                    breaker.record_failure()
                                else:
                                    breaker.release_probe()
                            raise
                        else:
                            if breaker is not None:
                                breaker.record_success()
                except Exception as exc:
                    _record_step_failure(
                        workflow,
                        function_name,
                        exc,
                        _duration_ms(state.started_at),
                        state.context.attempt,
                    )
                    if workflow is not None and workflow.should_degrade_failures():
                        _log_step_failed_degraded(function_name, exc)
                        return cast(R, FailedStep(function_name, exc))
                    raise

                actual_result, cost_report = _extract_cost_report(result, state.context)
                step_cost_usd, budget_error = _record_cost_report(
                    function_name,
                    workflow,
                    cost_report,
                )

                if checkpoint_state is not None:
                    _checkpoint_result(
                        state,
                        checkpoint_state,
                        actual_result,
                        cost_report,
                        step_cost_usd,
                    )

                _log_success(state, actual_result)
                _record_step_success(
                    workflow,
                    function_name,
                    actual_result,
                    _duration_ms(state.started_at),
                    state.context.attempt,
                    step_cost_usd,
                    from_checkpoint=False,
                )
            finally:
                _finalize_call(state)

            if budget_error is not None:
                raise budget_error

            return cast(R, actual_result)

        _attach_circuit_breaker(sync_wrapper, breaker)
        return sync_wrapper

    if func is None:
        return decorator

    return decorator(func)
