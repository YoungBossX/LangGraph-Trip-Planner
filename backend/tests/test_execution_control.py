from concurrent.futures import ThreadPoolExecutor

import pytest

from app.workflows.execution_control import (
    ExecutionControl,
    WorkflowCancelledError,
    WorkflowTimeoutError,
)


class MutableClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    "timeout_seconds",
    [True, False, 0, -1, -0.5, float("nan"), float("inf"), float("-inf")],
)
def test_timeout_must_be_positive_finite_and_not_bool(timeout_seconds):
    with pytest.raises(ValueError, match="positive"):
        ExecutionControl(timeout_seconds=timeout_seconds)


def test_remaining_starts_at_timeout_and_clamps_to_zero():
    clock = MutableClock(10.0)
    control = ExecutionControl(timeout_seconds=30, clock=clock)

    assert control.remaining() == 30

    clock.now = 25.5
    assert control.remaining() == 14.5

    clock.now = 45.0
    assert control.remaining() == 0


def test_check_succeeds_while_control_is_active():
    clock = MutableClock(10.0)
    control = ExecutionControl(timeout_seconds=30, clock=clock)

    clock.now = 39.999
    control.check()


@pytest.mark.parametrize("now", [40.0, 41.0], ids=["exact-deadline", "beyond-deadline"])
def test_check_raises_timeout_at_or_beyond_deadline(now):
    clock = MutableClock(10.0)
    control = ExecutionControl(timeout_seconds=30, clock=clock)
    clock.now = now

    with pytest.raises(WorkflowTimeoutError):
        control.check()


def test_cancel_is_idempotent_and_thread_safe():
    control = ExecutionControl(timeout_seconds=30)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: control.cancel(), range(100)))

    control.cancel()

    assert results == [None] * 100
    assert control.is_cancelled is True


def test_cancelled_control_raises_distinct_exception():
    control = ExecutionControl(timeout_seconds=30)
    control.cancel()

    with pytest.raises(WorkflowCancelledError):
        control.check()


def test_explicit_cancellation_takes_precedence_over_timeout():
    clock = MutableClock(10.0)
    control = ExecutionControl(timeout_seconds=30, clock=clock)
    clock.now = 40.0
    control.cancel()

    with pytest.raises(WorkflowCancelledError):
        control.check()
