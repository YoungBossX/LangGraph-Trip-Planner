"""Cooperative workflow deadline and cancellation control."""

import math
import time
from collections.abc import Callable
from numbers import Real
from threading import Event


class WorkflowTimeoutError(TimeoutError):
    """Raised when a workflow reaches its execution deadline."""


class WorkflowCancelledError(Exception):
    """Raised when a workflow is explicitly cancelled."""


class ExecutionControl:
    """Own an absolute monotonic deadline and a thread-safe cancel flag."""

    def __init__(self, timeout_seconds: float, clock: Callable[[], float] = time.monotonic):
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, Real)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")

        self._clock = clock
        self._deadline = clock() + float(timeout_seconds)
        self._cancelled = Event()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def remaining(self) -> float:
        return max(0.0, self._deadline - self._clock())

    def check(self) -> None:
        if self.is_cancelled:
            raise WorkflowCancelledError("Workflow execution was cancelled")
        if self.remaining() <= 0:
            raise WorkflowTimeoutError("Workflow execution timed out")

    def cancel(self) -> None:
        self._cancelled.set()
