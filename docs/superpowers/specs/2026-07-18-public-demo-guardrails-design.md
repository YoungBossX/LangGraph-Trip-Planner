# Public Demo Guardrails And Cancellation Design

## Scope

This specification covers the second remediation batch from the adversarial review for an anonymous public demo:

- per-IP request rate limits;
- per-IP and global planning concurrency limits;
- actual request-body size enforcement;
- bounded request field sizes;
- end-to-end request deadlines;
- bounded MCP tool calls;
- cooperative workflow cancellation;
- SSE heartbeats and disconnect handling;
- frontend stream cancellation and timeout cleanup;
- stable public error responses for these controls.

It does not add user accounts, browser-stored shared secrets, Redis, distributed quotas, billing, or an external identity provider.

## Deployment Model And Trust Boundary

The application remains anonymously accessible. Protection is intentionally sized for a single-process public demonstration, not a multi-tenant production service.

Client identity is `request.client.host`. Application code does not parse `X-Forwarded-For` or similar headers because an internet client can forge them. A deployment behind a reverse proxy must configure Uvicorn's trusted proxy behavior so `request.client.host` is rewritten only for trusted proxy peers.

All limiter state is process-local and resets when the process restarts. Multiple workers or replicas do not share limits. A future multi-instance deployment must move counters and concurrency leases to Redis or another shared atomic store.

## Configurable Defaults

The following settings are added with environment-variable overrides. No `.env` file is modified or committed.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `MAX_REQUEST_BODY_BYTES` | 16384 | Maximum decoded HTTP request-body bytes |
| `PLANNING_RATE_LIMIT` | 3 | Planning attempts allowed per IP per window |
| `PLANNING_RATE_WINDOW_SECONDS` | 600 | Planning rate-limit window |
| `PLANNING_PER_IP_CONCURRENCY` | 1 | Active planning executions per IP |
| `PLANNING_GLOBAL_CONCURRENCY` | 2 | Active planning executions for the process |
| `PHOTO_RATE_LIMIT` | 30 | Photo requests allowed per IP per window |
| `PHOTO_RATE_WINDOW_SECONDS` | 60 | Photo rate-limit window |
| `TRIP_REQUEST_TIMEOUT_SECONDS` | 300 | Absolute planning deadline |
| `MCP_TOOL_TIMEOUT_SECONDS` | 45 | Per MCP tool invocation deadline |
| `SSE_HEARTBEAT_SECONDS` | 15 | Maximum silence between SSE frames |

The frontend stream uses a 310-second absolute timeout and a 45-second inactivity timeout. These values stay slightly above the server deadline and three times the heartbeat interval.

All settings are validated as positive values. Invalid limits fail configuration validation rather than silently disabling protection.

## Admission Control

### Body Limit

An ASGI middleware counts bytes received from the request stream. It rejects a declared oversized `Content-Length` before reading the body and also enforces the limit while consuming chunks, so chunked requests cannot bypass the control. Oversized bodies return `413` with a stable public error code.

The middleware applies to request bodies globally. Normal `GET` and empty-body requests pass without allocation or buffering changes.

### Field Limits

`TripRequest` retains its existing JSON field names and date string types but adds these boundaries:

- `city`: trimmed, 1 to 50 characters;
- `transportation`: trimmed, 1 to 100 characters;
- `accommodation`: trimmed, 1 to 100 characters;
- `preferences`: at most 10 entries, each trimmed and 1 to 30 characters;
- `free_text_input`: trimmed, at most 1000 characters.

The photo endpoint `name` query is trimmed and limited to 1 to 100 characters. Validation errors remain standard FastAPI `422` responses.

### Rate Limits

A focused in-memory limiter stores monotonic timestamps per `(scope, client_ip)` under an `asyncio.Lock`. Expired timestamps are removed on access. Requests over the configured limit return `429`, a stable error code, and `Retry-After` rounded up to the next available time.

Scopes are independent:

- `trip-plan` is shared by `/api/trip/plan` and `/api/trip/plan-stream`;
- `poi-photo` applies to `/api/poi/photo`.

Every endpoint attempt consumes a rate-limit slot, including attempts later rejected by concurrency control or failed by a provider. Health and documentation endpoints are exempt.

The limiter accepts an injected monotonic clock for deterministic unit tests. State has an explicit reset helper for test isolation.

### Concurrency Limits

A planning admission controller tracks one process-wide active count and active counts per client IP under one `asyncio.Lock`. It does not queue excess work. If either limit is reached, it returns `429` immediately with a stable `PLANNING_BUSY` code and a short `Retry-After` value.

Permits are released in `finally` for success, validation errors, provider errors, deadlines, task cancellation, SSE disconnects, and generator closure. The SSE permit is held for the lifetime of the response stream, not merely until `StreamingResponse` is constructed.

## Execution Control

A small workflow execution-control object owns:

- an absolute monotonic deadline;
- a thread-safe cancellation event;
- `cancel()` and `check()` operations;
- distinct timeout and cancellation exception types.

The object is passed into `plan_trip()` and `astream_plan()` and stored in `TripPlannerState`. Each workflow node checks it before invoking an Agent and immediately after the invocation returns. Timeout and cancellation exceptions bypass normal node retry handling and propagate to the route.

This is cooperative cancellation. It prevents new nodes and later retries after cancellation, but it cannot forcibly terminate Python code already blocked inside a synchronous provider call.

## Provider Boundaries

The LLM continues to use the existing per-call `agent_timeout` setting.

The synchronous MCP bridge wraps `_arun()` with `asyncio.wait_for(..., MCP_TOOL_TIMEOUT_SECONDS)`. The running-loop fallback uses a bounded `future.result(timeout=...)` and cancels the future on timeout. The original tool response contract, including `response_format`, must be preserved while touching the wrapper.

Together, provider-level deadlines and execution-control checkpoints ensure a timed-out request stops after the current bounded provider operation rather than continuing through the rest of the graph.

## HTTP Endpoint Behavior

### Non-Streaming Planning

The synchronous workflow runs through `asyncio.to_thread()` so it does not block FastAPI's event loop. The route waits under the absolute request deadline. On timeout it marks the execution control cancelled and returns `504` with `TRIP_TIMEOUT`. A client task cancellation also marks control cancelled before propagating.

Known admission, timeout, and validation failures receive stable public responses. Unexpected internal exceptions are logged with traceback but exposed as a generic message rather than raw provider or infrastructure text.

### Streaming Planning

The SSE route runs workflow production in a dedicated asyncio task and sends events through a queue bounded to 16 items. The response loop waits for either the next workflow event or the heartbeat interval.

- When no workflow event arrives within the interval, it emits an SSE comment heartbeat.
- Before delivering an event and at each heartbeat boundary it checks `request.is_disconnected()`.
- On disconnect, absolute timeout, generator cancellation, or closure, it marks execution control cancelled, cancels the producer task, and releases admission in `finally`.
- A deadline produces an SSE `error` event with stable code `TRIP_TIMEOUT` when the connection is still writable.
- Internal exceptions produce a generic SSE error with a correlation-safe public code; raw exception strings remain server-side.

SSE frames preserve the current `progress`, `result`, and `error` event contract. Error payloads add a `code` field without removing `message` or `step`.

### Photo Lookup

The synchronous Unsplash service call runs through `asyncio.to_thread()` so a slow image request does not block the event loop. Its existing provider timeout remains in force. The route applies the photo rate limit and exposes generic transport failures rather than raw exception text.

## Frontend Lifecycle

`generateTripPlanStream()` accepts an optional external `AbortSignal` and owns an internal controller for deadlines. It implements:

- a 310-second absolute timeout;
- a 45-second inactivity timeout reset whenever response bytes arrive, including heartbeat comments;
- external abort propagation;
- `reader.cancel()` and `reader.releaseLock()` in `finally`;
- cleanup of timers and event listeners on every exit path;
- distinct client errors for rate limit, timeout, user/navigation cancellation, and server failure.

`Home.vue` owns the active controller. Starting a new submission aborts the previous one, and `onBeforeUnmount` aborts the active request. A cancelled request does not display a generic failure toast. The current loading UI and successful result flow remain unchanged in this batch.

## Error Codes

The controls use stable machine-readable codes:

- `REQUEST_TOO_LARGE` (`413`);
- `RATE_LIMITED` (`429`);
- `PLANNING_BUSY` (`429`);
- `TRIP_TIMEOUT` (`504` or SSE error);
- `TRIP_CANCELLED` (internal/SSE only when still connected);
- `TRIP_FAILED` (`500` or SSE error with generic public message).

The implementation may include a `Retry-After` header where applicable. It must not include provider URLs, credentials, raw exception bodies, or stack traces in client responses.

## Testing Strategy

Tests follow red-green-refactor.

1. Limiter unit tests use an injected clock to cover independent scopes, window expiry, exact thresholds, retry timing, and reset behavior.
2. Concurrency tests cover global and per-IP rejection plus release after success, failure, timeout, and cancellation.
3. Middleware tests cover declared and chunked oversized bodies and exact-boundary acceptance.
4. Request-model tests cover trimming and every length/list boundary.
5. API tests replace the workflow with controlled fakes and verify status codes, stable public codes, no event-loop blocking, permit lifetime, generic errors, and timeout mapping.
6. SSE tests cover progress/result compatibility, heartbeat emission, disconnect cancellation, deadline error, producer cancellation, and permit release.
7. MCP wrapper tests cover timeout, future cancellation, and preservation of `response_format`.
8. Frontend service tests use mocked streaming responses to cover external abort, absolute timeout, inactivity timeout, heartbeat resets, reader cleanup, and error classification.
9. Home component or extracted lifecycle tests prove new submissions and unmount abort active work without showing a generic failure.
10. Full backend tests, Ruff, frontend tests, and production build run after focused tests pass.

## Compatibility And Non-Goals

- Existing successful JSON and SSE result payloads remain compatible.
- Existing frontend routes and session storage format do not change.
- No `.env` or credentials are modified.
- No login, JWT, cookie session, CAPTCHA, Redis, or distributed quota is added.
- No claim is made that cooperative cancellation instantly kills an already-running synchronous provider operation.
- Readiness probes, Unsplash credential logging, MCP process reuse, dependency pinning, and Pydantic deprecation cleanup remain separate remediation work.
