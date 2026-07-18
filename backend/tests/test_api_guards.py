import asyncio
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from app.api import guards
from app.api.guards import (
    InMemoryRateLimiter,
    PlanningAdmissionController,
    PublicAPIError,
    get_client_ip,
    get_planning_admission_controller,
    get_rate_limiter,
    reset_api_guards,
)


class MutableClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_rate_limiter_allows_limit_then_rejects_without_consuming_slot():
    clock = MutableClock()
    limiter = InMemoryRateLimiter(clock=clock)

    async def consume_attempts():
        decisions = []
        for _ in range(3):
            decisions.append(await limiter.consume("trip-plan", "203.0.113.5", limit=3, window_seconds=600))
            clock.advance(10)

        rejected = await limiter.consume("trip-plan", "203.0.113.5", limit=3, window_seconds=600)
        clock.advance(570)
        after_expiry = await limiter.consume("trip-plan", "203.0.113.5", limit=3, window_seconds=600)
        return decisions, rejected, after_expiry

    decisions, rejected, after_expiry = asyncio.run(consume_attempts())

    assert [decision.allowed for decision in decisions] == [True, True, True]
    assert rejected.allowed is False
    assert isinstance(rejected.retry_after, int)
    assert rejected.retry_after > 0
    assert after_expiry.allowed is True


def test_rate_limiter_expires_oldest_timestamp_at_exact_window_boundary():
    clock = MutableClock(100.0)
    limiter = InMemoryRateLimiter(clock=clock)

    async def consume_at_boundary():
        first = await limiter.consume("trip-plan", "203.0.113.5", limit=1, window_seconds=60)
        clock.advance(60)
        second = await limiter.consume("trip-plan", "203.0.113.5", limit=1, window_seconds=60)
        return first, second

    first, second = asyncio.run(consume_at_boundary())

    assert first.allowed is True
    assert second.allowed is True


def test_rate_limiter_scopes_and_client_ips_are_independent():
    limiter = InMemoryRateLimiter(clock=MutableClock())

    async def consume_independent_keys():
        first = await limiter.consume("trip-plan", "203.0.113.5", limit=1, window_seconds=60)
        same_key = await limiter.consume("trip-plan", "203.0.113.5", limit=1, window_seconds=60)
        other_scope = await limiter.consume("poi-photo", "203.0.113.5", limit=1, window_seconds=60)
        other_ip = await limiter.consume("trip-plan", "203.0.113.6", limit=1, window_seconds=60)
        return first, same_key, other_scope, other_ip

    first, same_key, other_scope, other_ip = asyncio.run(consume_independent_keys())

    assert first.allowed is True
    assert same_key.allowed is False
    assert other_scope.allowed is True
    assert other_ip.allowed is True


def test_rate_limiter_reset_clears_all_attempts():
    limiter = InMemoryRateLimiter(clock=MutableClock())

    async def consume_reset_and_retry():
        await limiter.consume("trip-plan", "203.0.113.5", limit=1, window_seconds=60)
        rejected = await limiter.consume("trip-plan", "203.0.113.5", limit=1, window_seconds=60)
        await limiter.reset()
        allowed = await limiter.consume("trip-plan", "203.0.113.5", limit=1, window_seconds=60)
        return rejected, allowed

    rejected, allowed = asyncio.run(consume_reset_and_retry())

    assert rejected.allowed is False
    assert allowed.allowed is True


@pytest.mark.parametrize(
    ("limit", "window_seconds"),
    [(0, 60), (-1, 60), (1, 0), (1, -1)],
)
def test_rate_limiter_rejects_invalid_limits(limit, window_seconds):
    limiter = InMemoryRateLimiter(clock=MutableClock())

    with pytest.raises(ValueError):
        asyncio.run(limiter.consume("trip-plan", "203.0.113.5", limit=limit, window_seconds=window_seconds))


def test_rate_limiter_serializes_concurrent_calls_at_threshold():
    limiter = InMemoryRateLimiter(clock=MutableClock())

    async def consume_concurrently():
        ready = asyncio.Event()

        async def consume_once():
            await ready.wait()
            return await limiter.consume("trip-plan", "203.0.113.5", limit=3, window_seconds=600)

        tasks = [asyncio.create_task(consume_once()) for _ in range(20)]
        ready.set()
        return await asyncio.gather(*tasks)

    decisions = asyncio.run(consume_concurrently())

    assert sum(decision.allowed for decision in decisions) == 3
    assert all(decision.retry_after is None for decision in decisions if decision.allowed)
    assert all(decision.retry_after and decision.retry_after > 0 for decision in decisions if not decision.allowed)


def test_rate_limiter_defaults_to_ten_thousand_tracked_keys():
    limiter = InMemoryRateLimiter(clock=MutableClock())

    assert limiter._max_keys == 10_000


def test_rate_limiter_bounds_high_cardinality_and_preserves_existing_keys():
    limiter = InMemoryRateLimiter(clock=MutableClock(), max_keys=3)

    async def fill_and_exceed_capacity():
        initial = [
            await limiter.consume("trip-plan", f"203.0.113.{index}", limit=2, window_seconds=60)
            for index in range(3)
        ]
        existing_allowed = await limiter.consume("trip-plan", "203.0.113.0", limit=2, window_seconds=60)
        existing_rejected = await limiter.consume("trip-plan", "203.0.113.0", limit=2, window_seconds=60)
        overflow = [
            await limiter.consume("trip-plan", f"198.51.100.{index}", limit=1, window_seconds=60)
            for index in range(20)
        ]
        return initial, existing_allowed, existing_rejected, overflow

    initial, existing_allowed, existing_rejected, overflow = asyncio.run(fill_and_exceed_capacity())

    assert all(decision.allowed for decision in initial)
    assert existing_allowed.allowed is True
    assert existing_rejected.allowed is False
    assert all(decision.allowed is False for decision in overflow)
    assert all(isinstance(decision.retry_after, int) and decision.retry_after > 0 for decision in overflow)
    assert len(limiter._buckets) == 3


def test_rate_limiter_sweeps_stale_buckets_before_allocating_new_key():
    clock = MutableClock(100.0)
    limiter = InMemoryRateLimiter(clock=clock, max_keys=2)

    async def replace_stale_keys():
        await limiter.consume("trip-plan", "203.0.113.5", limit=1, window_seconds=10)
        await limiter.consume("poi-photo", "203.0.113.6", limit=1, window_seconds=20)
        clock.advance(20.01)
        return await limiter.consume("trip-plan", "203.0.113.7", limit=1, window_seconds=30)

    decision = asyncio.run(replace_stale_keys())

    assert decision.allowed is True
    assert set(limiter._buckets) == {("trip-plan", "203.0.113.7")}


@pytest.mark.parametrize("max_keys", [0, -1, True])
def test_rate_limiter_rejects_invalid_max_keys(max_keys):
    with pytest.raises(ValueError, match="max_keys"):
        InMemoryRateLimiter(clock=MutableClock(), max_keys=max_keys)


@pytest.mark.parametrize(
    ("elapsed", "expected_retry_after"),
    [(9.99, 1), (8.99, 2)],
    ids=["0.01-seconds-remaining", "1.01-seconds-remaining"],
)
def test_rate_limiter_rounds_fractional_retry_after_up(elapsed, expected_retry_after):
    clock = MutableClock(100.0)
    limiter = InMemoryRateLimiter(clock=clock)

    async def consume_and_reject():
        await limiter.consume("trip-plan", "203.0.113.5", limit=1, window_seconds=10)
        clock.advance(elapsed)
        return await limiter.consume("trip-plan", "203.0.113.5", limit=1, window_seconds=10)

    decision = asyncio.run(consume_and_reject())

    assert decision.allowed is False
    assert decision.retry_after == expected_retry_after


def test_admission_rejects_second_lease_for_same_ip():
    controller = PlanningAdmissionController(global_limit=2, per_ip_limit=1)

    async def acquire_twice():
        lease = await controller.acquire("203.0.113.5")
        try:
            with pytest.raises(PublicAPIError) as exc_info:
                await controller.acquire("203.0.113.5")
            return exc_info.value
        finally:
            await lease.release()

    error = asyncio.run(acquire_twice())

    assert error.status_code == 429
    assert error.code == "PLANNING_BUSY"
    assert isinstance(error.retry_after, int)
    assert error.retry_after > 0


def test_admission_rejects_different_ip_when_global_limit_is_full():
    controller = PlanningAdmissionController(global_limit=1, per_ip_limit=1)

    async def fill_global_limit():
        lease = await controller.acquire("203.0.113.5")
        try:
            with pytest.raises(PublicAPIError) as exc_info:
                await controller.acquire("203.0.113.6")
            return exc_info.value
        finally:
            await lease.release()

    error = asyncio.run(fill_global_limit())

    assert error.status_code == 429
    assert error.code == "PLANNING_BUSY"
    assert error.retry_after > 0


def test_admission_release_permits_reacquisition():
    controller = PlanningAdmissionController(global_limit=1, per_ip_limit=1)

    async def release_and_reacquire():
        first = await controller.acquire("203.0.113.5")
        await first.release()
        second = await controller.acquire("203.0.113.5")
        await second.release()

    asyncio.run(release_and_reacquire())


def test_admission_release_is_idempotent_and_counts_never_go_negative():
    controller = PlanningAdmissionController(global_limit=1, per_ip_limit=2)

    async def release_twice_then_fill():
        lease = await controller.acquire("203.0.113.5")
        await lease.release()
        await lease.release()

        active = await controller.acquire("203.0.113.5")
        try:
            with pytest.raises(PublicAPIError):
                await controller.acquire("203.0.113.6")
        finally:
            await active.release()

    asyncio.run(release_twice_then_fill())


def test_admission_same_ip_acquire_race_allows_exact_per_ip_limit():
    controller = PlanningAdmissionController(global_limit=10, per_ip_limit=3)

    async def race_acquires():
        results = await asyncio.gather(
            *(controller.acquire("203.0.113.5") for _ in range(20)),
            return_exceptions=True,
        )
        leases = [result for result in results if not isinstance(result, BaseException)]
        errors = [result for result in results if isinstance(result, BaseException)]
        await asyncio.gather(*(lease.release() for lease in leases))
        return leases, errors

    leases, errors = asyncio.run(race_acquires())

    assert len(leases) == 3
    assert len(errors) == 17
    assert all(isinstance(error, PublicAPIError) for error in errors)


def test_admission_global_acquire_race_allows_exact_global_limit():
    controller = PlanningAdmissionController(global_limit=4, per_ip_limit=1)

    async def race_acquires():
        results = await asyncio.gather(
            *(controller.acquire(f"203.0.113.{index}") for index in range(20)),
            return_exceptions=True,
        )
        leases = [result for result in results if not isinstance(result, BaseException)]
        errors = [result for result in results if isinstance(result, BaseException)]
        await asyncio.gather(*(lease.release() for lease in leases))
        return leases, errors

    leases, errors = asyncio.run(race_acquires())

    assert len(leases) == 4
    assert len(errors) == 16
    assert all(isinstance(error, PublicAPIError) for error in errors)


def test_admission_concurrent_release_of_same_lease_is_idempotent():
    controller = PlanningAdmissionController(global_limit=1, per_ip_limit=1)

    async def release_concurrently_and_reacquire():
        lease = await controller.acquire("203.0.113.5")
        await asyncio.gather(*(lease.release() for _ in range(20)))
        reacquired = await controller.acquire("203.0.113.6")
        await reacquired.release()

    asyncio.run(release_concurrently_and_reacquire())


def test_admission_context_manager_releases_after_normal_exit():
    controller = PlanningAdmissionController(global_limit=1, per_ip_limit=1)

    async def use_context_manager():
        async with await controller.acquire("203.0.113.5"):
            pass
        reacquired = await controller.acquire("203.0.113.5")
        await reacquired.release()

    asyncio.run(use_context_manager())


def test_admission_context_manager_releases_after_exception():
    controller = PlanningAdmissionController(global_limit=1, per_ip_limit=1)

    async def raise_inside_context_manager():
        with pytest.raises(RuntimeError, match="planning failed"):
            async with await controller.acquire("203.0.113.5"):
                raise RuntimeError("planning failed")
        reacquired = await controller.acquire("203.0.113.5")
        await reacquired.release()

    asyncio.run(raise_inside_context_manager())


def test_admission_finally_releases_after_cancellation_simulation():
    controller = PlanningAdmissionController(global_limit=1, per_ip_limit=1)

    async def cancel_and_reacquire():
        lease = await controller.acquire("203.0.113.5")
        try:
            raise asyncio.CancelledError
        except asyncio.CancelledError:
            pass
        finally:
            await lease.release()

        reacquired = await controller.acquire("203.0.113.5")
        await reacquired.release()

    asyncio.run(cancel_and_reacquire())


def test_admission_actual_task_cancellation_releases_context_lease():
    controller = PlanningAdmissionController(global_limit=1, per_ip_limit=1)

    async def cancel_holder_and_reacquire():
        entered = asyncio.Event()

        async def hold_lease():
            async with await controller.acquire("203.0.113.5"):
                entered.set()
                await asyncio.Future()

        task = asyncio.create_task(hold_lease())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        reacquired = await controller.acquire("203.0.113.5")
        await reacquired.release()

    asyncio.run(cancel_holder_and_reacquire())


@pytest.mark.parametrize(
    ("global_limit", "per_ip_limit"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_admission_rejects_invalid_limits(global_limit, per_ip_limit):
    with pytest.raises(ValueError):
        PlanningAdmissionController(global_limit=global_limit, per_ip_limit=per_ip_limit)


def test_public_api_error_exposes_public_response_fields():
    error = PublicAPIError(
        status_code=429,
        code="RATE_LIMITED",
        message="Too many requests.",
        retry_after=17,
    )

    assert str(error) == "Too many requests."
    assert error.status_code == 429
    assert error.code == "RATE_LIMITED"
    assert error.message == "Too many requests."
    assert error.retry_after == 17


def test_get_client_ip_uses_request_client_and_ignores_forwarded_header():
    request = Request(
        {
            "type": "http",
            "client": ("203.0.113.5", 12345),
            "headers": [(b"x-forwarded-for", b"198.51.100.99")],
        }
    )

    assert get_client_ip(request) == "203.0.113.5"


def test_get_client_ip_has_stable_fallback_without_request_client():
    request_without_client = Request({"type": "http", "headers": []})
    request_with_none_client = Request({"type": "http", "client": None, "headers": []})

    assert get_client_ip(request_without_client) == "unknown"
    assert get_client_ip(request_with_none_client) == "unknown"


def test_guard_singletons_are_cached_configured_and_resettable(monkeypatch):
    configured = SimpleNamespace(planning_global_concurrency=1, planning_per_ip_concurrency=1)
    monkeypatch.setattr(guards, "get_settings", lambda: configured)
    reset_api_guards()

    first_limiter = get_rate_limiter()
    first_controller = get_planning_admission_controller()

    assert get_rate_limiter() is first_limiter
    assert get_planning_admission_controller() is first_controller

    async def assert_configured_limits():
        lease = await first_controller.acquire("203.0.113.5")
        try:
            with pytest.raises(PublicAPIError):
                await first_controller.acquire("203.0.113.6")
        finally:
            await lease.release()

    asyncio.run(assert_configured_limits())

    reset_api_guards()
    second_limiter = get_rate_limiter()
    second_controller = get_planning_admission_controller()

    assert second_limiter is not first_limiter
    assert second_controller is not first_controller
    reset_api_guards()


def test_reset_api_guards_refuses_to_discard_active_singleton_controller():
    reset_api_guards()
    limiter = get_rate_limiter()
    controller = get_planning_admission_controller()

    async def assert_reset_refusal():
        lease = await controller.acquire("203.0.113.5")
        try:
            with pytest.raises(RuntimeError, match="active planning lease"):
                reset_api_guards()
            assert get_rate_limiter() is limiter
            assert get_planning_admission_controller() is controller
        finally:
            await lease.release()

    asyncio.run(assert_reset_refusal())
    reset_api_guards()
