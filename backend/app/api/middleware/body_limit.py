"""Streaming ASGI request-body size enforcement."""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_TOO_LARGE_BODY = json.dumps(
    {
        "detail": {
            "code": "REQUEST_TOO_LARGE",
            "message": "请求体过大",
        }
    },
    ensure_ascii=False,
    separators=(",", ":"),
).encode("utf-8")


class _RequestBodyOverflow(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Reject HTTP request bodies that exceed a byte limit while streaming."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if self._declared_body_is_too_large(scope):
            await self._send_too_large(send)
            return

        received_bytes = 0
        response_started = False

        async def receive_with_limit() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise _RequestBodyOverflow
            return message

        async def send_with_state(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive_with_limit, send_with_state)
        except _RequestBodyOverflow:
            if not response_started:
                await self._send_too_large(send)

    def _declared_body_is_too_large(self, scope: Scope) -> bool:
        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                return int(value) > self.max_body_bytes
            except ValueError:
                return False
        return False

    @staticmethod
    async def _send_too_large(send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_TOO_LARGE_BODY)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _TOO_LARGE_BODY})
