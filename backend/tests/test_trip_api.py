import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.api.guards import PublicAPIError, RateLimitDecision
from app.api.routes import trip
from app.models.schemas import TripPlan, TripRequest
from app.workflows.execution_control import WorkflowTimeoutError


def _request_model() -> TripRequest:
    return TripRequest(
        city="Hangzhou",
        start_date="2026-03-01",
        end_date="2026-03-01",
        travel_days=1,
        transportation="transit",
        accommodation="hotel",
        preferences=["history"],
        free_text_input="",
    )


def _request_json() -> dict:
    return _request_model().model_dump()


def _settings(**overrides):
    values = {
        "planning_rate_limit": 3,
        "planning_rate_window_seconds": 600,
        "planning_global_concurrency": 2,
        "planning_per_ip_concurrency": 1,
        "trip_request_timeout_seconds": 30,
        "sse_heartbeat_seconds": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Limiter:
    def __init__(self, decision=None):
        self.decision = decision or RateLimitDecision(allowed=True)
        self.calls = []

    async def consume(self, scope, client_ip, *, limit, window_seconds):
        self.calls.append((scope, client_ip, limit, window_seconds))
        return self.decision


class _Lease:
    def __init__(self):
        self.release_count = 0

    async def release(self):
        self.release_count += 1


class _Admission:
    def __init__(self, *, error=None):
        self.error = error
        self.client_ips = []
        self.leases = []

    async def acquire(self, client_ip):
        self.client_ips.append(client_ip)
        if self.error is not None:
            raise self.error
        lease = _Lease()
        self.leases.append(lease)
        return lease


class _Workflow:
    def __init__(self, result=None, error=None):
        self.result = result or TripPlan(city="Hangzhou", start_date="2026-03-01", end_date="2026-03-01")
        self.error = error
        self.calls = []

    def plan_trip(self, request, control):
        self.calls.append((request, control))
        if self.error is not None:
            raise self.error
        return self.result

    async def astream_plan(self, request, control=None):
        if False:
            yield request, control


def _install_trip_dependencies(monkeypatch, *, limiter=None, admission=None, workflow=None, settings=None):
    limiter = limiter or _Limiter()
    admission = admission or _Admission()
    workflow = workflow or _Workflow()
    monkeypatch.setattr(trip, "get_rate_limiter", lambda: limiter, raising=False)
    monkeypatch.setattr(trip, "get_planning_admission_controller", lambda: admission, raising=False)
    monkeypatch.setattr(trip, "get_trip_planner_workflow", lambda: workflow)
    monkeypatch.setattr(trip, "get_settings", lambda: settings or _settings(), raising=False)
    return limiter, admission, workflow


def test_main_registers_public_api_error_handler_with_stable_shape_and_retry_header():
    handler = main.app.exception_handlers.get(PublicAPIError)

    assert handler is not None
    response = asyncio.run(
        handler(
            None,
            PublicAPIError(
                status_code=429,
                code="RATE_LIMITED",
                message="Too many requests. Please retry later.",
                retry_after=17,
            ),
        )
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "17"
    assert json.loads(response.body) == {
        "detail": {"code": "RATE_LIMITED", "message": "Too many requests. Please retry later."}
    }


def test_plan_and_stream_share_planning_rate_limit_scope(monkeypatch):
    limiter, admission, _ = _install_trip_dependencies(monkeypatch)
    client = TestClient(main.app)

    plan_response = client.post("/api/trip/plan", json=_request_json())
    stream_response = client.post("/api/trip/plan-stream", json=_request_json())

    assert plan_response.status_code == 200
    assert stream_response.status_code == 200
    assert [call[0] for call in limiter.calls] == ["trip-plan", "trip-plan"]
    assert [call[2:] for call in limiter.calls] == [(3, 600), (3, 600)]
    assert len(admission.leases) == 2
    assert [lease.release_count for lease in admission.leases] == [1, 1]


def test_plan_rate_rejection_has_stable_public_error(monkeypatch):
    limiter = _Limiter(RateLimitDecision(allowed=False, retry_after=23))
    admission = _Admission()
    _install_trip_dependencies(monkeypatch, limiter=limiter, admission=admission)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/api/trip/plan", json=_request_json())

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "23"
    assert response.json() == {
        "detail": {"code": "RATE_LIMITED", "message": "Too many requests. Please retry later."}
    }
    assert admission.client_ips == []


def test_plan_busy_rejection_has_code_and_retry_header(monkeypatch):
    busy = PublicAPIError(429, "PLANNING_BUSY", "The planning service is busy. Please retry shortly.", 1)
    admission = _Admission(error=busy)
    _install_trip_dependencies(monkeypatch, admission=admission)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/api/trip/plan", json=_request_json())

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"
    assert response.json()["detail"]["code"] == "PLANNING_BUSY"


def test_plan_runs_sync_workflow_through_to_thread(monkeypatch):
    _, admission, workflow = _install_trip_dependencies(monkeypatch)
    calls = []

    async def fake_to_thread(function, *args):
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    client = TestClient(main.app)

    response = client.post("/api/trip/plan", json=_request_json())

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert len(calls) == 2
    assert calls[0][1] == ()
    assert calls[1][0] == workflow.plan_trip
    assert calls[1][1][0] == _request_model()
    assert calls[1][1][1] is workflow.calls[0][1]
    assert admission.leases[0].release_count == 1


@pytest.mark.parametrize(
    "timeout_error",
    [WorkflowTimeoutError("workflow deadline"), asyncio.TimeoutError("asyncio deadline"), TimeoutError("deadline")],
    ids=["workflow-timeout", "asyncio-timeout", "builtin-timeout"],
)
def test_plan_workflow_timeout_variants_are_public_504(monkeypatch, timeout_error):
    _, admission, _ = _install_trip_dependencies(monkeypatch, workflow=_Workflow(error=timeout_error))
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/api/trip/plan", json=_request_json())

    assert response.status_code == 504
    assert response.json()["detail"] == {
        "code": "TRIP_TIMEOUT",
        "message": "Trip planning timed out. Please try again.",
    }
    assert admission.leases[0].release_count == 1


@pytest.mark.parametrize("path", ["/api/trip/plan", "/api/trip/plan-stream"])
def test_invalid_trip_request_consumes_rate_slot_before_validation(monkeypatch, path):
    limiter, admission, _ = _install_trip_dependencies(monkeypatch)
    client = TestClient(main.app)

    response = client.post(path, json={**_request_json(), "travel_days": 0})

    assert response.status_code == 422
    assert len(limiter.calls) == 1
    assert limiter.calls[0][0] == "trip-plan"
    assert admission.client_ips == []


def test_malformed_trip_json_consumes_rate_slot_before_decode(monkeypatch):
    limiter, admission, _ = _install_trip_dependencies(monkeypatch)
    client = TestClient(main.app)

    response = client.post(
        "/api/trip/plan",
        content='{"city":',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert len(limiter.calls) == 1
    assert limiter.calls[0][0] == "trip-plan"
    assert admission.client_ips == []


def test_health_route_does_not_consume_public_rate_limit(monkeypatch):
    limiter, _, _ = _install_trip_dependencies(monkeypatch)
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert limiter.calls == []


def test_plan_initialization_is_offloaded_and_uses_same_absolute_deadline(monkeypatch):
    _, admission, workflow = _install_trip_dependencies(monkeypatch)
    remaining_values = iter([9, 4])
    wait_timeouts = []

    class FakeControl:
        def __init__(self, timeout_seconds):
            self.cancel_count = 0

        def remaining(self):
            value = next(remaining_values)
            wait_timeouts.append(value)
            return value

        def cancel(self):
            self.cancel_count += 1

    calls = []

    async def fake_to_thread(function, *args):
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(trip, "ExecutionControl", FakeControl, raising=False)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    client = TestClient(main.app)

    response = client.post("/api/trip/plan", json=_request_json())

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[0][1] == ()
    assert calls[1][0] == workflow.plan_trip
    assert wait_timeouts == [9, 4]
    assert admission.leases[0].release_count == 1


def test_plan_initialization_timeout_is_public_timeout_and_releases(monkeypatch):
    created_controls = []

    class FakeControl:
        def __init__(self, timeout_seconds):
            self.cancel_count = 0
            created_controls.append(self)

        def remaining(self):
            return 0

        def cancel(self):
            self.cancel_count += 1

    limiter, admission, _ = _install_trip_dependencies(monkeypatch)
    monkeypatch.setattr(trip, "ExecutionControl", FakeControl, raising=False)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/api/trip/plan", json=_request_json())

    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "TRIP_TIMEOUT"
    assert len(limiter.calls) == 1
    assert created_controls[0].cancel_count == 1
    assert admission.leases[0].release_count == 1


def test_plan_initialization_error_is_generic_and_releases(monkeypatch, caplog):
    secret = "mcp initialization details"
    limiter, admission, _ = _install_trip_dependencies(monkeypatch)
    monkeypatch.setattr(trip, "get_trip_planner_workflow", lambda: (_ for _ in ()).throw(RuntimeError(secret)))
    client = TestClient(main.app, raise_server_exceptions=False)

    with caplog.at_level("ERROR"):
        response = client.post("/api/trip/plan", json=_request_json())

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "TRIP_FAILED",
        "message": "Trip planning failed. Please try again later.",
    }
    assert secret not in response.text
    assert secret in caplog.text
    assert len(limiter.calls) == 1
    assert admission.leases[0].release_count == 1


def test_stream_initialization_is_offloaded_before_response(monkeypatch):
    _, admission, _ = _install_trip_dependencies(monkeypatch)
    calls = []

    async def fake_to_thread(function, *args):
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    client = TestClient(main.app)

    response = client.post("/api/trip/plan-stream", json=_request_json())

    assert response.status_code == 200
    assert calls and calls[0][1] == ()
    assert len(calls) == 1
    assert admission.leases[0].release_count == 1


def test_stream_initialization_timeout_releases_once(monkeypatch):
    created_controls = []

    class FakeControl:
        def __init__(self, timeout_seconds):
            self.cancel_count = 0
            created_controls.append(self)

        def remaining(self):
            return 0

        def cancel(self):
            self.cancel_count += 1

    _, admission, _ = _install_trip_dependencies(monkeypatch)
    monkeypatch.setattr(trip, "ExecutionControl", FakeControl, raising=False)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/api/trip/plan-stream", json=_request_json())

    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "TRIP_TIMEOUT"
    assert created_controls[0].cancel_count == 1
    assert admission.leases[0].release_count == 1


def test_stream_initialization_error_is_generic_and_releases_once(monkeypatch, caplog):
    secret = "stream mcp startup secret"
    _, admission, _ = _install_trip_dependencies(monkeypatch)
    monkeypatch.setattr(trip, "get_trip_planner_workflow", lambda: (_ for _ in ()).throw(RuntimeError(secret)))
    client = TestClient(main.app, raise_server_exceptions=False)

    with caplog.at_level("ERROR"):
        response = client.post("/api/trip/plan-stream", json=_request_json())

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "TRIP_FAILED",
        "message": "Trip planning failed. Please try again later.",
    }
    assert secret not in response.text
    assert secret in caplog.text
    assert admission.leases[0].release_count == 1


def test_stream_response_construction_failure_releases_once(monkeypatch):
    _, admission, _ = _install_trip_dependencies(monkeypatch)

    def fail_to_construct(*args, **kwargs):
        raise RuntimeError("response construction failed")

    monkeypatch.setattr(trip, "_LifecycleStreamingResponse", fail_to_construct, raising=False)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/api/trip/plan-stream", json=_request_json())

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "TRIP_FAILED"
    assert admission.leases[0].release_count == 1


def test_plan_timeout_cancels_control_and_releases_admission(monkeypatch):
    created_controls = []

    class FakeControl:
        def __init__(self, timeout_seconds):
            self.timeout_seconds = timeout_seconds
            self.cancel_count = 0
            created_controls.append(self)

        def remaining(self):
            return 0

        def cancel(self):
            self.cancel_count += 1

    _install_trip_dependencies(monkeypatch, settings=_settings(trip_request_timeout_seconds=9))
    monkeypatch.setattr(trip, "ExecutionControl", FakeControl, raising=False)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/api/trip/plan", json=_request_json())

    assert response.status_code == 504
    assert response.json() == {
        "detail": {"code": "TRIP_TIMEOUT", "message": "Trip planning timed out. Please try again."}
    }
    assert created_controls[0].timeout_seconds == 9
    assert created_controls[0].cancel_count == 1


def test_plan_task_cancellation_cancels_control_releases_once_and_reraises(monkeypatch):
    entered = asyncio.Event()
    created_controls = []

    class FakeControl:
        def __init__(self, timeout_seconds):
            self.cancel_count = 0
            created_controls.append(self)

        def remaining(self):
            return 30

        def cancel(self):
            self.cancel_count += 1

    limiter, admission, _ = _install_trip_dependencies(monkeypatch)
    monkeypatch.setattr(trip, "ExecutionControl", FakeControl, raising=False)

    async def blocked_to_thread(function, *args):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(asyncio, "to_thread", blocked_to_thread)

    async def cancel_route():
        request = SimpleNamespace(client=SimpleNamespace(host="203.0.113.8"))
        task = asyncio.create_task(trip.plan_trip(_request_model(), request))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_route())

    assert limiter.calls[0][1] == "203.0.113.8"
    assert created_controls[0].cancel_count == 1
    assert admission.leases[0].release_count == 1


def test_plan_unexpected_error_is_logged_and_public_message_is_generic(monkeypatch, caplog):
    secret = "provider secret should stay in logs"
    _, admission, _ = _install_trip_dependencies(monkeypatch, workflow=_Workflow(error=RuntimeError(secret)))
    client = TestClient(main.app, raise_server_exceptions=False)

    with caplog.at_level("ERROR"):
        response = client.post("/api/trip/plan", json=_request_json())

    assert response.status_code == 500
    assert response.json() == {
        "detail": {"code": "TRIP_FAILED", "message": "Trip planning failed. Please try again later."}
    }
    assert secret not in response.text
    assert secret in caplog.text
    assert admission.leases[0].release_count == 1
def test_plan_rate_rejection_preserves_cors_headers(monkeypatch):
    limiter = _Limiter(RateLimitDecision(allowed=False, retry_after=23))
    _install_trip_dependencies(monkeypatch, limiter=limiter)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post(
        "/api/trip/plan",
        json=_request_json(),
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 429
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"


def test_trip_health_initialization_runs_off_event_loop(monkeypatch):
    calls = []
    workflow = _Workflow()

    async def fake_to_thread(function, *args):
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(trip, "get_trip_planner_workflow", lambda: workflow)
    monkeypatch.setattr(trip, "get_settings", lambda: _settings(trip_request_timeout_seconds=5))
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    response = TestClient(main.app).get("/api/trip/health")

    assert response.status_code == 200
    assert calls == [(trip.get_trip_planner_workflow, ())]
    assert response.json()["tools_loaded"] == 0


def test_trip_health_failure_uses_stable_public_message(monkeypatch, caplog):
    secret = "mcp health failure details"
    monkeypatch.setattr(
        trip,
        "get_trip_planner_workflow",
        lambda: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    monkeypatch.setattr(trip, "get_settings", lambda: _settings(trip_request_timeout_seconds=5))

    with caplog.at_level("ERROR"):
        response = TestClient(main.app, raise_server_exceptions=False).get("/api/trip/health")

    assert response.status_code == 503
    assert response.json() == {"detail": trip._HEALTH_UNAVAILABLE_MESSAGE}
    assert secret not in response.text
    assert secret in caplog.text
