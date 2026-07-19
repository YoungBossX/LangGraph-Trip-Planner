import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import main
from app.api.guards import RateLimitDecision
from app.api.routes import poi


class _Limiter:
    def __init__(self, decision=None):
        self.decision = decision or RateLimitDecision(allowed=True)
        self.calls = []

    async def consume(self, scope, client_ip, *, limit, window_seconds):
        self.calls.append((scope, client_ip, limit, window_seconds))
        return self.decision


class _PhotoService:
    def __init__(self, results=None, error=None):
        self.results = iter(results or [])
        self.error = error
        self.queries = []

    def get_photo_url(self, query):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return next(self.results, None)


def _install(monkeypatch, *, limiter=None, service=None, limit=30, window=60):
    limiter = limiter or _Limiter()
    service = service or _PhotoService(["https://example.test/photo.jpg"])
    monkeypatch.setattr(poi, "get_rate_limiter", lambda: limiter, raising=False)
    monkeypatch.setattr(poi, "get_unsplash_service", lambda: service)
    monkeypatch.setattr(
        poi,
        "get_settings",
        lambda: SimpleNamespace(photo_rate_limit=limit, photo_rate_window_seconds=window),
        raising=False,
    )
    return limiter, service


def test_photo_route_uses_photo_rate_limit_scope_and_settings(monkeypatch):
    limiter, service = _install(monkeypatch, limit=7, window=90)
    client = TestClient(main.app)

    response = client.get("/api/poi/photo", params={"name": "West Lake"})

    assert response.status_code == 200
    assert limiter.calls == [("poi-photo", "testclient", 7, 90)]
    assert service.queries == ["West Lake China landmark"]


def test_photo_rate_rejection_has_stable_code_and_retry_header(monkeypatch):
    limiter = _Limiter(RateLimitDecision(allowed=False, retry_after=11))
    _, service = _install(monkeypatch, limiter=limiter)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.get("/api/poi/photo", params={"name": "West Lake"})

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "11"
    assert response.json() == {
        "detail": {"code": "RATE_LIMITED", "message": "Too many requests. Please retry later."}
    }
    assert service.queries == []


def test_invalid_photo_name_consumes_rate_slot_before_validation(monkeypatch):
    limiter, service = _install(monkeypatch)
    client = TestClient(main.app)

    response = client.get("/api/poi/photo", params={"name": "x" * 101})

    assert response.status_code == 422
    assert limiter.calls == [("poi-photo", "testclient", 30, 60)]
    assert service.queries == []


def test_photo_fallback_calls_are_offloaded_to_threads(monkeypatch):
    _, service = _install(monkeypatch, service=_PhotoService([None, "https://example.test/fallback.jpg"]))
    calls = []

    async def fake_to_thread(function, *args):
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    client = TestClient(main.app)

    response = client.get("/api/poi/photo", params={"name": "West Lake"})

    assert response.status_code == 200
    assert response.json()["data"]["photo_url"] == "https://example.test/fallback.jpg"
    assert [args for _, args in calls] == [("West Lake China landmark",), ("West Lake",)]
    assert all(function == service.get_photo_url for function, _ in calls)


def test_photo_null_result_remains_successful_and_compatible(monkeypatch):
    _install(monkeypatch, service=_PhotoService([None, None]))
    client = TestClient(main.app)

    response = client.get("/api/poi/photo", params={"name": "West Lake"})

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["data"] == {"name": "West Lake", "photo_url": None}


def test_photo_provider_error_is_logged_and_public_message_is_generic(monkeypatch, caplog):
    secret = "unsplash credential detail"
    _install(monkeypatch, service=_PhotoService(error=RuntimeError(secret)))
    client = TestClient(main.app, raise_server_exceptions=False)

    with caplog.at_level("ERROR"):
        response = client.get("/api/poi/photo", params={"name": "West Lake"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": {
            "code": "PHOTO_LOOKUP_FAILED",
            "message": "Photo lookup failed. Please try again later.",
        }
    }
    assert secret not in response.text
    assert secret in caplog.text
