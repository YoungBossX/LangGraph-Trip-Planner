import asyncio
import json

import pytest

MAX_BODY_BYTES = 16384
TOO_LARGE_BODY = {
    "detail": {
        "code": "REQUEST_TOO_LARGE",
        "message": "请求体过大",
    }
}


class _RecordingHttpApp:
    def __init__(self):
        self.calls = 0
        self.messages = []

    async def __call__(self, scope, receive, send):
        self.calls += 1
        while True:
            message = await receive()
            self.messages.append(message)
            if not message.get("more_body", False):
                break

        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _http_scope(method="POST", headers=None):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


async def _call_asgi(app, scope, request_messages):
    messages = iter(request_messages)
    sent = []

    async def receive():
        try:
            return next(messages)
        except StopIteration:
            return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent


def _middleware(app):
    from app.api.middleware.body_limit import RequestBodyLimitMiddleware

    return RequestBodyLimitMiddleware(app, max_body_bytes=MAX_BODY_BYTES)


def _assert_too_large_response(sent):
    assert len(sent) == 2
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert (b"content-type", b"application/json") in sent[0]["headers"]
    assert sent[1]["type"] == "http.response.body"
    assert json.loads(sent[1]["body"]) == TOO_LARGE_BODY


def test_declared_oversized_body_returns_413_without_downstream_call():
    downstream = _RecordingHttpApp()
    app = _middleware(downstream)
    scope = _http_scope(headers=[(b"content-length", str(MAX_BODY_BYTES + 1).encode())])

    sent = asyncio.run(_call_asgi(app, scope, []))

    _assert_too_large_response(sent)
    assert downstream.calls == 0


def test_chunked_body_crossing_limit_returns_one_413_response():
    downstream = _RecordingHttpApp()
    app = _middleware(downstream)
    chunks = [
        {"type": "http.request", "body": b"a" * 8192, "more_body": True},
        {"type": "http.request", "body": b"b" * 8193, "more_body": False},
    ]

    sent = asyncio.run(_call_asgi(app, _http_scope(), chunks))

    _assert_too_large_response(sent)
    assert downstream.calls == 1


def test_body_at_exact_limit_passes_with_identical_chunks():
    downstream = _RecordingHttpApp()
    app = _middleware(downstream)
    chunks = [
        {"type": "http.request", "body": b"a" * 8192, "more_body": True},
        {"type": "http.request", "body": b"b" * 8192, "more_body": False},
    ]

    sent = asyncio.run(_call_asgi(app, _http_scope(), chunks))

    assert sent[0]["status"] == 200
    assert downstream.calls == 1
    assert downstream.messages == chunks
    assert b"".join(message["body"] for message in downstream.messages) == b"a" * 8192 + b"b" * 8192


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_get_and_empty_requests_pass(method):
    downstream = _RecordingHttpApp()
    app = _middleware(downstream)
    empty_body = [{"type": "http.request", "body": b"", "more_body": False}]

    sent = asyncio.run(_call_asgi(app, _http_scope(method=method), empty_body))

    assert sent[0]["status"] == 200
    assert downstream.calls == 1
    assert downstream.messages == empty_body


def test_non_http_scope_passes_unchanged():
    received_scopes = []

    async def downstream(scope, receive, send):
        received_scopes.append(scope)

    app = _middleware(downstream)
    scope = {"type": "lifespan", "asgi": {"version": "3.0"}}

    sent = asyncio.run(_call_asgi(app, scope, []))

    assert sent == []
    assert received_scopes == [scope]


def test_main_app_registers_configured_body_limit():
    from app.api.main import app
    from app.api.middleware.body_limit import RequestBodyLimitMiddleware
    from app.config import get_settings

    middleware = next(item for item in app.user_middleware if item.cls is RequestBodyLimitMiddleware)

    assert middleware.kwargs["max_body_bytes"] == get_settings().max_request_body_bytes
