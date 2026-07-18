"""Process-local request rate limiting and planning admission controls."""

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from starlette.requests import Request

from app.config import get_settings

_UNKNOWN_CLIENT_IP = "unknown"
_PLANNING_BUSY_RETRY_AFTER = 1
_PLANNING_BUSY_MESSAGE = "The planning service is busy. Please retry shortly."


class PublicAPIError(Exception):
    """An expected API failure with fields safe to serialize publicly."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retry_after = retry_after


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after: int | None = None


@dataclass
class _RateLimitBucket:
    window_seconds: int
    timestamps: deque[float] = field(default_factory=deque)


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class InMemoryRateLimiter:
    """A deterministic sliding-window limiter scoped by operation and client."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, max_keys: int = 10_000) -> None:
        _require_positive_int(max_keys, "max_keys")
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: dict[tuple[str, str], _RateLimitBucket] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _expire_timestamps(bucket: _RateLimitBucket, now: float) -> None:
        expiry_cutoff = now - bucket.window_seconds
        while bucket.timestamps and bucket.timestamps[0] <= expiry_cutoff:
            bucket.timestamps.popleft()

    def _sweep_expired_buckets(self, now: float) -> None:
        expired_keys = []
        for key, bucket in self._buckets.items():
            self._expire_timestamps(bucket, now)
            if not bucket.timestamps:
                expired_keys.append(key)

        for key in expired_keys:
            del self._buckets[key]

    def _capacity_retry_after(self, now: float) -> int:
        return min(
            max(1, math.ceil(bucket.timestamps[-1] + bucket.window_seconds - now))
            for bucket in self._buckets.values()
        )

    async def consume(
        self,
        scope: str,
        client_ip: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision:
        _require_positive_int(limit, "limit")
        _require_positive_int(window_seconds, "window_seconds")

        async with self._lock:
            now = self._clock()
            key = (scope, client_ip)
            bucket = self._buckets.get(key)
            if bucket is not None:
                bucket.window_seconds = window_seconds

            self._sweep_expired_buckets(now)
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    return RateLimitDecision(allowed=False, retry_after=self._capacity_retry_after(now))
                bucket = _RateLimitBucket(window_seconds=window_seconds)
                self._buckets[key] = bucket

            if len(bucket.timestamps) >= limit:
                retry_after = max(1, math.ceil(bucket.timestamps[0] + window_seconds - now))
                return RateLimitDecision(allowed=False, retry_after=retry_after)

            bucket.timestamps.append(now)
            return RateLimitDecision(allowed=True)

    async def reset(self) -> None:
        async with self._lock:
            self._buckets.clear()


class PlanningAdmissionController:
    """Track process-wide and per-client active planning executions."""

    def __init__(self, global_limit: int, per_ip_limit: int) -> None:
        _require_positive_int(global_limit, "global_limit")
        _require_positive_int(per_ip_limit, "per_ip_limit")
        self._global_limit = global_limit
        self._per_ip_limit = per_ip_limit
        self._active_global = 0
        self._active_by_ip: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, client_ip: str) -> "PlanningLease":
        async with self._lock:
            if self._active_global >= self._global_limit or self._active_by_ip.get(client_ip, 0) >= self._per_ip_limit:
                raise PublicAPIError(
                    status_code=429,
                    code="PLANNING_BUSY",
                    message=_PLANNING_BUSY_MESSAGE,
                    retry_after=_PLANNING_BUSY_RETRY_AFTER,
                )

            self._active_global += 1
            self._active_by_ip[client_ip] = self._active_by_ip.get(client_ip, 0) + 1
            return PlanningLease(self, client_ip)

    async def _release(self, lease: "PlanningLease") -> None:
        async with self._lock:
            if lease._released:
                return

            lease._released = True
            self._active_global = max(0, self._active_global - 1)

            active_for_ip = self._active_by_ip.get(lease._client_ip, 0)
            if active_for_ip <= 1:
                self._active_by_ip.pop(lease._client_ip, None)
            else:
                self._active_by_ip[lease._client_ip] = active_for_ip - 1


class PlanningLease:
    """An idempotently releasable planning admission permit."""

    def __init__(self, controller: PlanningAdmissionController, client_ip: str) -> None:
        self._controller = controller
        self._client_ip = client_ip
        self._released = False

    async def release(self) -> None:
        await self._controller._release(self)

    async def __aenter__(self) -> "PlanningLease":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.release()


def get_client_ip(request: Request) -> str:
    client = request.client
    return client.host if client is not None else _UNKNOWN_CLIENT_IP


_rate_limiter: InMemoryRateLimiter | None = None
_planning_admission_controller: PlanningAdmissionController | None = None


def get_rate_limiter() -> InMemoryRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = InMemoryRateLimiter()
    return _rate_limiter


def get_planning_admission_controller() -> PlanningAdmissionController:
    global _planning_admission_controller
    if _planning_admission_controller is None:
        settings = get_settings()
        _planning_admission_controller = PlanningAdmissionController(
            global_limit=settings.planning_global_concurrency,
            per_ip_limit=settings.planning_per_ip_concurrency,
        )
    return _planning_admission_controller


def reset_api_guards() -> None:
    """Reset guard singletons for tests from a quiescent caller."""
    global _planning_admission_controller, _rate_limiter
    if _planning_admission_controller is not None and _planning_admission_controller._active_global:
        raise RuntimeError("Cannot reset API guards while an active planning lease exists")
    _rate_limiter = None
    _planning_admission_controller = None
