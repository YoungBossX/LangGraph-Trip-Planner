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


def _http_scope(method="POST", headers=None, path="/"):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


async def _call_asgi(app, scope, request_messages, sent=None):
    messages = iter(request_messages)
    sent = [] if sent is None else sent

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


def _header_value(message, name):
    return next((value for header_name, value in message["headers"] if header_name.lower() == name), None)


def _assert_too_large_response(sent, origin=None):
    assert len(sent) == 2
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert _header_value(sent[0], b"content-type") == b"application/json"
    if origin is not None:
        assert _header_value(sent[0], b"access-control-allow-origin") == origin.encode("ascii")
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


def test_overflow_after_response_start_propagates_without_second_start():
    from app.api.middleware.body_limit import _RequestBodyOverflow

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await receive()

    app = _middleware(downstream)
    sent = []
    oversized_body = [{"type": "http.request", "body": b"x" * (MAX_BODY_BYTES + 1), "more_body": False}]

    with pytest.raises(_RequestBodyOverflow):
        asyncio.run(_call_asgi(app, _http_scope(), oversized_body, sent=sent))

    response_starts = [message for message in sent if message["type"] == "http.response.start"]
    assert len(response_starts) == 1
    assert response_starts[0]["status"] == 200


@pytest.mark.parametrize(
    "headers",
    [
        [(b"content-length", b"1"), (b"content-length", str(MAX_BODY_BYTES + 1).encode())],
        [(b"content-length", b"invalid"), (b"content-length", str(MAX_BODY_BYTES + 1).encode())],
    ],
)
def test_any_parseable_oversized_content_length_rejects_before_downstream(headers):
    downstream = _RecordingHttpApp()
    app = _middleware(downstream)

    sent = asyncio.run(_call_asgi(app, _http_scope(headers=headers), []))

    _assert_too_large_response(sent)
    assert downstream.calls == 0


def test_underdeclared_body_crossing_limit_returns_413():
    downstream = _RecordingHttpApp()
    app = _middleware(downstream)
    scope = _http_scope(headers=[(b"content-length", b"100")])
    chunks = [
        {"type": "http.request", "body": b"a" * 8192, "more_body": True},
        {"type": "http.request", "body": b"b" * 8193, "more_body": False},
    ]

    sent = asyncio.run(_call_asgi(app, scope, chunks))

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


def test_http_disconnect_passes_through_unchanged():
    downstream = _RecordingHttpApp()
    app = _middleware(downstream)
    disconnect = {"type": "http.disconnect"}

    sent = asyncio.run(_call_asgi(app, _http_scope(), [disconnect]))

    assert sent[0]["status"] == 200
    assert downstream.calls == 1
    assert downstream.messages == [disconnect]


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


def _main_app_with_allowed_origin():
    from app.api.main import app
    from app.config import get_settings

    return app, get_settings().get_cors_origins_list()[0]


def test_declared_oversized_app_request_includes_cors_headers():
    app, origin = _main_app_with_allowed_origin()
    headers = [
        (b"origin", origin.encode("ascii")),
        (b"content-length", str(MAX_BODY_BYTES + 1).encode()),
    ]

    sent = asyncio.run(_call_asgi(app, _http_scope(headers=headers), []))

    _assert_too_large_response(sent, origin=origin)


def test_chunked_oversized_app_request_includes_cors_headers():
    app, origin = _main_app_with_allowed_origin()
    headers = [
        (b"origin", origin.encode("ascii")),
        (b"content-type", b"application/json"),
    ]
    scope = _http_scope(headers=headers, path="/api/trip/plan")
    chunks = [
        {"type": "http.request", "body": b"a" * 8192, "more_body": True},
        {"type": "http.request", "body": b"b" * 8193, "more_body": False},
    ]

    sent = asyncio.run(_call_asgi(app, scope, chunks))

    _assert_too_large_response(sent, origin=origin)
