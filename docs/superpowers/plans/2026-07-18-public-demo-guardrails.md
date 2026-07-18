# Public Demo Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Protect the anonymous public demo with deterministic input limits, per-IP rate and concurrency controls, bounded provider calls, cooperative workflow cancellation, non-blocking API routes, SSE lifecycle handling, and frontend request cleanup.

**Architecture:** Build the protection as independent layers: request parsing and body middleware, pure in-memory admission primitives, workflow execution control, HTTP/SSE orchestration, and frontend cancellation. Keep successful response contracts compatible and expose stable error codes while retaining raw failures only in server logs.

**Tech Stack:** Python 3.10+, FastAPI/Starlette ASGI, LangGraph, Pydantic v2, pytest/httpx, Vue 3, TypeScript, Vitest

---

## File Structure

- Create `backend/app/api/middleware/__init__.py`: middleware package marker.
- Create `backend/app/api/middleware/body_limit.py`: streaming ASGI request-body byte enforcement.
- Create `backend/app/api/guards.py`: public API error, in-memory limiter, concurrency admission, client identity, and resettable singletons.
- Create `backend/app/workflows/execution_control.py`: deadline/cancellation object and exception types.
- Modify `backend/app/config.py`: validated guardrail settings.
- Modify `backend/app/models/schemas.py`: bounded and trimmed request fields.
- Modify `backend/app/api/main.py`: body middleware and public-error exception handler.
- Modify `backend/app/api/routes/trip.py`: admission, thread offload, timeout, generic errors, SSE producer/heartbeat/disconnect lifecycle.
- Modify `backend/app/api/routes/poi.py`: bounded query, rate limit, thread offload, generic errors.
- Modify `backend/app/workflows/trip_planner_state.py`: execution-control state.
- Modify `backend/app/workflows/trip_planner_graph.py`: cooperative checkpoints and optional control parameters.
- Modify `backend/app/tools/amap_mcp_tools.py`: bounded async bridge and response-contract preservation.
- Create backend tests for request limits, middleware, guards, execution control, API routes, and SSE.
- Modify `backend/tests/test_amap_mcp_tools.py` and existing workflow tests as required for compatibility.
- Modify `frontend/src/services/api.ts`: abortable stream, deadlines, cleanup, and error classification.
- Create `frontend/src/services/tripRequestLifecycle.ts`: active-request replacement and unmount cancellation helper.
- Modify `frontend/src/views/Home.vue`: use the lifecycle helper and suppress cancellation errors.
- Create frontend service and lifecycle tests.

### Task 1: Bound Configuration, Fields, And Request Bodies

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/app/models/schemas.py`
- Create: `backend/app/api/middleware/__init__.py`
- Create: `backend/app/api/middleware/body_limit.py`
- Modify: `backend/app/api/main.py`
- Modify: `backend/app/api/routes/poi.py`
- Create: `backend/tests/test_request_limits.py`
- Create: `backend/tests/test_body_limit.py`
- Modify: `backend/tests/test_config_runtime.py`

- [ ] **Step 1: Write failing settings and request-model tests**

Test exact defaults, reject zero/negative guardrail settings, and cover every accepted/rejected boundary:

```python
def test_trip_request_trims_bounded_text_fields():
    request = TripRequest(
        city="  Hangzhou  ",
        start_date="2026-03-01",
        end_date="2026-03-01",
        travel_days=1,
        transportation="  transit  ",
        accommodation="  hotel  ",
        preferences=["  history  "],
        free_text_input="  quiet pace  ",
    )
    assert request.city == "Hangzhou"
    assert request.preferences == ["history"]
```

Parameterize empty/whitespace city, transport, accommodation, overlong values, 11 preferences, empty preference entries, overlong preference entries, and free text over 1000 characters. Keep existing date tests passing.

- [ ] **Step 2: Write failing ASGI middleware tests**

Use a minimal ASGI app and direct receive/send functions to prove:

- `Content-Length` over 16384 receives `413` without calling downstream;
- chunked data crossing the limit receives `413`;
- exactly 16384 bytes pass unchanged;
- empty and `GET` requests pass;
- the JSON body is `{"detail":{"code":"REQUEST_TOO_LARGE","message":...}}`.

- [ ] **Step 3: Verify RED**

Run:

```powershell
E:\Anaconda\envs\agent\python.exe -m pytest tests/test_request_limits.py tests/test_body_limit.py tests/test_config_runtime.py -q
```

Expected: settings/field boundaries are absent and middleware cannot be imported.

- [ ] **Step 4: Implement validated settings and constrained fields**

Use positive Pydantic settings fields for every default from the design. Preserve date strings and the existing calendar validator. Define reusable constrained text aliases with `Annotated` and `StringConstraints(strip_whitespace=True, ...)`; constrain the preferences list with `Field(max_length=10)`.

For `/api/poi/photo`, use a trimmed Pydantic/FastAPI query type limited to 1..100 characters. Do not add rate limiting yet.

- [ ] **Step 5: Implement streaming body middleware**

Create `RequestBodyLimitMiddleware` as a low-level ASGI wrapper. It must inspect declared length and wrap `receive()` to count actual `http.request` body bytes. Catch its private overflow exception at the middleware boundary and send one stable `413` JSON response without invoking downstream after overflow.

Register it in `main.py` using `settings.max_request_body_bytes`.

- [ ] **Step 6: Verify GREEN**

Run focused tests plus:

```powershell
E:\Anaconda\envs\agent\python.exe -m ruff check app/config.py app/models/schemas.py app/api/main.py app/api/routes/poi.py app/api/middleware tests/test_request_limits.py tests/test_body_limit.py tests/test_config_runtime.py
```

Expected: focused tests and Ruff pass.

### Task 2: Build Deterministic Rate And Concurrency Controls

**Files:**
- Create: `backend/app/api/guards.py`
- Create: `backend/tests/test_api_guards.py`

- [ ] **Step 1: Write failing rate-limiter tests**

Use an injected mutable monotonic clock and test:

- exact threshold accepts 3 planning attempts;
- fourth attempt is rejected with a positive retry delay;
- the oldest timestamp expiry permits the next request;
- scopes and client IPs are independent;
- reset clears state;
- every call at the boundary is serialized safely.

Desired API:

```python
decision = await limiter.consume("trip-plan", "203.0.113.5", limit=3, window_seconds=600)
assert decision.allowed is True
```

- [ ] **Step 2: Write failing admission tests**

Desired API:

```python
lease = await controller.acquire("203.0.113.5")
await lease.release()
```

Cover one-per-IP rejection, global rejection across IPs, idempotent release, release after exception/cancellation simulation, and reacquisition. No request queues are allowed.

- [ ] **Step 3: Verify RED**

Run `E:\Anaconda\envs\agent\python.exe -m pytest tests/test_api_guards.py -q` and expect import failure.

- [ ] **Step 4: Implement focused primitives**

`guards.py` defines:

- `PublicAPIError(status_code, code, message, retry_after=None)`;
- immutable `RateLimitDecision`;
- `InMemoryRateLimiter(clock=time.monotonic)` using deques and one `asyncio.Lock`;
- `PlanningAdmissionController(global_limit, per_ip_limit)` using counts and one lock;
- idempotent `PlanningLease` supporting `async with`;
- `get_client_ip(request)` using only `request.client.host` with a stable fallback;
- resettable singleton getters constructed from `Settings`.

The public exception serializes later as `{"detail":{"code":code,"message":message}}` and may carry `Retry-After`.

- [ ] **Step 5: Verify GREEN**

Run focused tests and Ruff on `guards.py` and its tests.

### Task 3: Add Workflow Deadlines And Bound MCP Calls

**Files:**
- Create: `backend/app/workflows/execution_control.py`
- Modify: `backend/app/workflows/trip_planner_state.py`
- Modify: `backend/app/workflows/trip_planner_graph.py`
- Modify: `backend/app/tools/amap_mcp_tools.py`
- Create: `backend/tests/test_execution_control.py`
- Create: `backend/tests/test_workflow_cancellation.py`
- Modify: `backend/tests/test_amap_mcp_tools.py`

- [ ] **Step 1: Write failing execution-control tests**

Use an injected clock and cover active, expired, explicitly cancelled, thread-safe cancellation, remaining-time clamping, and distinct `WorkflowTimeoutError` / `WorkflowCancelledError` exceptions.

Desired API:

```python
control = ExecutionControl(timeout_seconds=30, clock=clock)
control.check()
control.cancel()
with pytest.raises(WorkflowCancelledError):
    control.check()
```

- [ ] **Step 2: Write failing workflow checkpoint tests**

Mock Agents and prove:

- a pre-cancelled control invokes no Agent;
- cancellation set by an Agent return stops before the next node;
- timeout/cancellation exceptions are not converted into retryable node errors;
- legacy calls without a supplied control remain compatible.

- [ ] **Step 3: Write failing MCP wrapper tests**

Create fake async tools to prove:

- `_arun()` exceeding the configured deadline raises timeout;
- the running-loop future path uses a bounded `result(timeout=...)` and cancels the future;
- `response_format="content_and_artifact"` survives wrapping;
- existing wrapper metadata and arguments remain intact.

- [ ] **Step 4: Verify RED**

Run the three focused test modules and confirm missing control/timeouts fail.

- [ ] **Step 5: Implement execution control and workflow checkpoints**

Add optional control to state and to `create_initial_state()`, `plan_trip()`, and `astream_plan()`. If omitted, construct no deadline for direct legacy workflow calls. Add a helper that checks control before and after every Agent invocation.

In each Agent node, re-raise `WorkflowTimeoutError` and `WorkflowCancelledError` before the generic exception handler so cancellation never enters normal retry recovery.

- [ ] **Step 6: Bound and preserve the MCP wrapper**

Read `mcp_tool_timeout_seconds` from settings. Wrap async calls with `asyncio.wait_for`. Bound `future.result()` with the same timeout and cancel the future when it expires. Copy `response_format` into the new wrapper constructor along with existing metadata.

- [ ] **Step 7: Verify GREEN**

Run focused tests, all workflow tests, and Ruff on touched backend files.

### Task 4: Integrate HTTP, SSE, And Photo Lifecycles

**Files:**
- Modify: `backend/app/api/guards.py`
- Modify: `backend/app/api/main.py`
- Modify: `backend/app/api/routes/trip.py`
- Modify: `backend/app/api/routes/poi.py`
- Create: `backend/tests/test_trip_api.py`
- Create: `backend/tests/test_sse_api.py`
- Create: `backend/tests/test_poi_api.py`

- [ ] **Step 1: Write failing public-error and non-streaming API tests**

Build small FastAPI test apps or use dependency/monkeypatch isolation. Cover:

- planning rate limit shared across both planning endpoints;
- per-IP/global busy rejection with `429`, stable code, and `Retry-After`;
- workflow runs off the event loop in a worker thread;
- total deadline maps to `504 TRIP_TIMEOUT` and marks control cancelled;
- task cancellation marks control cancelled and releases admission;
- unexpected exceptions return generic `TRIP_FAILED`, never raw exception text;
- successful response remains `TripPlanResponse` compatible.

- [ ] **Step 2: Write failing SSE lifecycle tests**

Test the event generator or extracted stream helper directly with controlled async workflows:

- progress/result frame compatibility;
- comment heartbeat after the interval;
- bounded queue behavior;
- disconnect detected both before events and on heartbeat;
- disconnect/deadline/generator close cancels producer and control;
- deadline emits `TRIP_TIMEOUT` when writable;
- admission lease is released exactly once on every exit;
- raw exceptions become generic `TRIP_FAILED` events.

- [ ] **Step 3: Write failing photo-route tests**

Cover per-IP photo rate limits, trimmed query validation, `asyncio.to_thread()` execution, successful null/photo responses, and generic errors without raw provider details.

- [ ] **Step 4: Verify RED**

Run the three API test modules and confirm current blocking/unbounded behavior fails.

- [ ] **Step 5: Register public-error handling and endpoint admission**

Register one FastAPI exception handler for `PublicAPIError`, including `Retry-After` when present. Both planning routes consume the shared `trip-plan` rate scope. Acquire concurrency before execution and release in route/generator `finally`.

- [ ] **Step 6: Implement non-streaming deadline orchestration**

Run `workflow.plan_trip(request, control)` with `asyncio.to_thread()` under the remaining absolute deadline. Map workflow and asyncio timeout exceptions to `TRIP_TIMEOUT`; on `CancelledError`, cancel control and re-raise; on unexpected failures log details and raise generic `TRIP_FAILED`.

- [ ] **Step 7: Implement bounded SSE production**

Use an extracted helper where practical. Producer task sends normalized events to `asyncio.Queue(maxsize=16)`. Consumer waits for `min(heartbeat, control.remaining())`, sends comment heartbeats, checks disconnect before output, maps stable errors, and closes producer/control/lease in `finally`.

- [ ] **Step 8: Offload photo lookup**

Apply the photo limiter, call the synchronous service through `asyncio.to_thread()`, and replace `print`/raw HTTP errors with server logging plus generic public error mapping.

- [ ] **Step 9: Verify GREEN**

Run API tests, the full backend suite, and full Ruff.

### Task 5: Make Frontend Streaming Abortable

**Files:**
- Modify: `frontend/src/services/api.ts`
- Create: `frontend/src/services/api.test.ts`
- Create: `frontend/src/services/tripRequestLifecycle.ts`
- Create: `frontend/src/services/tripRequestLifecycle.test.ts`
- Modify: `frontend/src/views/Home.vue`

- [ ] **Step 1: Write failing stream-service tests**

Mock `fetch`, `ReadableStream`, and fake timers. Cover:

- optional external abort rejects with code `TRIP_CANCELLED`;
- 310-second absolute timeout rejects with `TRIP_TIMEOUT`;
- 45-second inactivity timeout resets on every received chunk, including heartbeat-only chunks;
- `reader.cancel()` and `reader.releaseLock()` execute on result, error, timeout, and abort;
- `429` parses stable JSON code/message and `Retry-After`;
- SSE `error` uses server code and does not lose the message;
- existing progress and result framing remains compatible.

- [ ] **Step 2: Write failing lifecycle-helper tests**

Desired behavior:

```typescript
const lifecycle = createTripRequestLifecycle()
const first = lifecycle.begin()
const second = lifecycle.begin()
expect(first.signal.aborted).toBe(true)
lifecycle.finish(second)
```

Cover replacement, stale finish not clearing the active request, explicit cancel, and idempotence.

- [ ] **Step 3: Verify RED**

Run `npm test -- src/services/api.test.ts src/services/tripRequestLifecycle.test.ts` and expect missing APIs.

- [ ] **Step 4: Implement stream errors, abort composition, and cleanup**

Export `TripStreamError` with stable `code` and optional `retryAfter`. Add an optional third argument to `generateTripPlanStream()` containing `signal`, `absoluteTimeoutMs`, and `inactivityTimeoutMs`, with defaults from the design.

Use one internal controller, propagate external abort, reset inactivity on every chunk, and clean all timers/listeners/readers in `finally`. Parse non-OK JSON before falling back to HTTP status text.

- [ ] **Step 5: Implement request lifecycle in Home**

Create the lifecycle helper, call `begin()` before each submission, pass its signal, and call `finish()` only for the matching controller. Register `onBeforeUnmount(lifecycle.cancel)`. Do not display a generic toast for `TRIP_CANCELLED`; retain current messages for other errors.

- [ ] **Step 6: Verify GREEN**

Run focused frontend tests, all frontend tests, and `npm run build`. Do not recreate or stage public inspiration assets that are currently deleted outside this task.

### Final Verification

- [ ] Run `E:\Anaconda\envs\agent\python.exe -m pytest tests -q` in `backend`.
- [ ] Run `E:\Anaconda\envs\agent\python.exe -m ruff check .` in `backend`.
- [ ] Run `npm test` and `npm run build` in `frontend`.
- [ ] Review the complete batch against `docs/superpowers/specs/2026-07-18-public-demo-guardrails-design.md`.
- [ ] Confirm no `.env`, credentials, dependency lock, unrelated image deletion, or generated directory is staged.
- [ ] Run `git diff --check` on the committed range and record all test counts before claiming completion.
