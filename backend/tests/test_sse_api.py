import asyncio
import json

import pytest

from app.api.routes import trip
from app.models.schemas import TripPlan


class _Lease:
    def __init__(self):
        self.release_count = 0

    async def release(self):
        self.release_count += 1


class _Control:
    def __init__(self, remaining=30):
        self._remaining = remaining
        self.cancel_count = 0

    def remaining(self):
        return self._remaining

    def cancel(self):
        self.cancel_count += 1


class _Request:
    def __init__(self, disconnected=False, sequence=None):
        self._disconnected = disconnected
        self._sequence = iter(sequence) if sequence is not None else None
        self.check_count = 0

    async def is_disconnected(self):
        self.check_count += 1
        if self._sequence is not None:
            try:
                return next(self._sequence)
            except StopIteration:
                return self._disconnected
        return self._disconnected


class _Workflow:
    def __init__(self, events=None, error=None, block=None):
        self.events = events or []
        self.error = error
        self.block = block
        self.closed = False

    async def astream_plan(self, request, control=None):
        try:
            if self.block is not None:
                await self.block.wait()
            if self.error is not None:
                raise self.error
            for event in self.events:
                yield event
        finally:
            self.closed = True


class _InitControl(_Control):
    pass


def _request_model():
    return object()


def _parse_event(frame):
    lines = frame.strip().splitlines()
    return lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))


async def _collect(workflow, *, request=None, control=None, lease=None, heartbeat=0.01):
    return [
        frame
        async for frame in trip._stream_trip_events(
            trip_request=_request_model(),
            request=request or _Request(),
            workflow=workflow,
            control=control or _Control(),
            lease=lease or _Lease(),
            heartbeat_seconds=heartbeat,
        )
    ]


def test_sse_queue_is_bounded_to_sixteen_events():
    queue = trip._create_sse_queue()

    assert queue.maxsize == 16


def test_sse_producer_blocks_when_queue_is_full_without_dropping_events():
    events = [(f"step-{index}", {}) for index in range(18)]
    workflow = _Workflow(events=events)
    queue = trip._create_sse_queue()
    control = _Control()

    async def exercise():
        producer = asyncio.create_task(trip._produce_trip_events(queue, workflow, object(), control))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert queue.qsize() == 16
        assert not producer.done()

        received = []
        while True:
            item = await queue.get()
            if item is trip._SSE_DONE:
                break
            received.append(item)
        await producer
        return received

    assert asyncio.run(exercise()) == events


def test_sse_preserves_progress_and_result_framing():
    plan = TripPlan(city="Hangzhou", start_date="2026-03-01", end_date="2026-03-01")
    workflow = _Workflow(
        events=[
            ("search_attractions", {}),
            ("plan_itinerary", {"trip_plan": plan}),
        ]
    )

    frames = asyncio.run(_collect(workflow))

    progress_event, progress_data = _parse_event(frames[0])
    result_event, result_data = _parse_event(frames[-1])
    assert progress_event == "progress"
    assert progress_data["step"] == "search_attractions"
    assert result_event == "result"
    assert result_data["success"] is True
    assert result_data["data"]["city"] == "Hangzhou"


def test_sse_emits_heartbeat_comment_while_producer_is_idle():
    block = asyncio.Event()
    workflow = _Workflow(block=block)
    request = _Request()
    control = _Control()
    lease = _Lease()

    async def read_one():
        stream = trip._stream_trip_events(
            trip_request=_request_model(),
            request=request,
            workflow=workflow,
            control=control,
            lease=lease,
            heartbeat_seconds=0.001,
        )
        frame = await anext(stream)
        await stream.aclose()
        return frame

    frame = asyncio.run(read_one())

    assert frame == ": heartbeat\n\n"
    assert request.check_count >= 2
    assert control.cancel_count == 1
    assert lease.release_count == 1
    assert workflow.closed is True


def test_sse_disconnect_before_first_event_cleans_up_without_output():
    block = asyncio.Event()
    workflow = _Workflow(block=block)
    request = _Request(disconnected=True)
    control = _Control()
    lease = _Lease()

    frames = asyncio.run(_collect(workflow, request=request, control=control, lease=lease))

    assert frames == []
    assert control.cancel_count == 1
    assert lease.release_count == 1
    assert workflow.closed is True


def test_sse_disconnect_during_heartbeat_does_not_write_heartbeat():
    block = asyncio.Event()
    workflow = _Workflow(block=block)
    request = _Request(sequence=[False, True], disconnected=True)
    control = _Control()
    lease = _Lease()

    frames = asyncio.run(
        _collect(workflow, request=request, control=control, lease=lease, heartbeat=0.001)
    )

    assert frames == []
    assert request.check_count >= 2
    assert control.cancel_count == 1
    assert lease.release_count == 1


def test_sse_generator_close_cancels_producer_control_and_releases_once():
    block = asyncio.Event()
    workflow = _Workflow(block=block)
    control = _Control()
    lease = _Lease()

    async def start_and_close():
        stream = trip._stream_trip_events(
            trip_request=_request_model(),
            request=_Request(),
            workflow=workflow,
            control=control,
            lease=lease,
            heartbeat_seconds=0.001,
        )
        await anext(stream)
        await stream.aclose()

    asyncio.run(start_and_close())

    assert workflow.closed is True
    assert control.cancel_count == 1
    assert lease.release_count == 1


def test_sse_deadline_emits_stable_timeout_error_and_cleans_up():
    workflow = _Workflow(block=asyncio.Event())
    control = _Control(remaining=0)
    lease = _Lease()

    frames = asyncio.run(_collect(workflow, control=control, lease=lease))

    event, data = _parse_event(frames[0])
    assert event == "error"
    assert data == {"code": "TRIP_TIMEOUT", "message": "Trip planning timed out. Please try again."}
    assert control.cancel_count == 1
    assert lease.release_count == 1


def test_sse_workflow_timeout_emits_timeout_error_not_generic_failure():
    from app.workflows.execution_control import WorkflowTimeoutError

    workflow = _Workflow(error=WorkflowTimeoutError("workflow deadline"))

    frames = asyncio.run(_collect(workflow))

    event, data = _parse_event(frames[0])
    assert event == "error"
    assert data == {"code": "TRIP_TIMEOUT", "message": "Trip planning timed out. Please try again."}


def test_sse_consumer_formatting_failure_is_logged_and_generic(caplog):
    class BrokenPlan:
        def model_dump(self):
            raise ValueError("serialization secret")

    workflow = _Workflow(events=[("plan_itinerary", {"trip_plan": BrokenPlan()})])

    with caplog.at_level("ERROR"):
        frames = asyncio.run(_collect(workflow))

    assert [_parse_event(frame)[0] for frame in frames] == ["progress", "error"]
    assert _parse_event(frames[-1])[1] == {
        "code": "TRIP_FAILED",
        "message": "Trip planning failed. Please try again later.",
    }
    assert "serialization secret" in caplog.text


def test_sse_heartbeat_catches_asyncio_timeout_error(monkeypatch):
    original_wait_for = asyncio.wait_for
    calls = 0

    async def fake_wait_for(awaitable, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            awaitable.close()
            raise asyncio.TimeoutError("heartbeat wait")
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    workflow = _Workflow(block=asyncio.Event())
    stream = trip._stream_trip_events(
        trip_request=_request_model(),
        request=_Request(),
        workflow=workflow,
        control=_Control(),
        lease=_Lease(),
        heartbeat_seconds=0.1,
    )

    async def read_one():
        frame = await anext(stream)
        await stream.aclose()
        return frame

    assert asyncio.run(read_one()) == ": heartbeat\n\n"


def test_sse_response_send_failure_before_generator_start_cleans_up_once():
    control = _Control()
    lease = _Lease()
    lifecycle = trip._PlanningLifecycle(control, lease)
    started = False

    async def body():
        nonlocal started
        started = True
        yield b"never sent"

    response = trip._LifecycleStreamingResponse(body(), lifecycle=lifecycle, media_type="text/event-stream")

    async def send(_message):
        raise RuntimeError("send failed")

    async def receive():
        await asyncio.Future()

    async def call_response():
        with pytest.raises(RuntimeError, match="send failed"):
            await response(
                {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""},
                receive,
                send,
            )

    asyncio.run(call_response())

    assert started is False
    assert control.cancel_count == 1
    assert lease.release_count == 1


def test_sse_first_body_send_failure_closes_generator_and_stops_producer_once():
    class StreamingWorkflow:
        def __init__(self):
            self.started = False
            self.closed = False

        async def astream_plan(self, request, control=None):
            self.started = True
            try:
                yield ("search_attractions", {})
                await asyncio.Future()
            finally:
                self.closed = True

    workflow = StreamingWorkflow()
    control = _Control()
    lease = _Lease()
    lifecycle = trip._PlanningLifecycle(control, lease)
    stream = trip._stream_trip_events(
        trip_request=_request_model(),
        request=_Request(),
        workflow=workflow,
        control=control,
        lease=lease,
        heartbeat_seconds=30,
        lifecycle=lifecycle,
    )
    response = trip._LifecycleStreamingResponse(stream, lifecycle=lifecycle, media_type="text/event-stream")
    sent_types = []

    async def send(message):
        sent_types.append(message["type"])
        if message["type"] == "http.response.body":
            raise RuntimeError("body send failed")

    async def receive():
        await asyncio.Future()

    async def exercise():
        with pytest.raises(RuntimeError, match="body send failed"):
            await response(
                {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""},
                receive,
                send,
            )
        await asyncio.sleep(0)
        current = asyncio.current_task()
        return [task for task in asyncio.all_tasks() if task is not current and not task.done()]

    pending = asyncio.run(exercise())

    assert sent_types == ["http.response.start", "http.response.body"]
    assert workflow.started is True
    assert workflow.closed is True
    assert control.cancel_count == 1
    assert lease.release_count == 1
    assert pending == []


def test_sse_response_and_generator_finally_release_shared_lifecycle_once():
    control = _Control()
    lease = _Lease()
    lifecycle = trip._PlanningLifecycle(control, lease)

    async def body():
        try:
            yield b"event"
        finally:
            await lifecycle.cleanup()

    response = trip._LifecycleStreamingResponse(body(), lifecycle=lifecycle, media_type="text/event-stream")
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        await asyncio.Future()

    asyncio.run(
        response(
            {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""},
            receive,
            send,
        )
    )

    assert sent
    assert control.cancel_count == 1
    assert lease.release_count == 1


def test_sse_workflow_exception_is_logged_but_public_error_is_generic(caplog):
    secret = "upstream token leaked"
    workflow = _Workflow(error=RuntimeError(secret))
    control = _Control()
    lease = _Lease()

    with caplog.at_level("ERROR"):
        frames = asyncio.run(_collect(workflow, control=control, lease=lease))

    event, data = _parse_event(frames[0])
    assert event == "error"
    assert data == {"code": "TRIP_FAILED", "message": "Trip planning failed. Please try again later."}
    assert secret not in frames[0]
    assert secret in caplog.text
    assert workflow.closed is True
    assert control.cancel_count == 1
    assert lease.release_count == 1


def test_sse_terminal_workflow_error_uses_generic_code_message_and_step():
    workflow = _Workflow(events=[(trip.NODE_ERROR, {"error": "provider secret"})])

    frames = asyncio.run(_collect(workflow))

    event, data = _parse_event(frames[-1])
    assert event == "error"
    assert data == {
        "code": "TRIP_FAILED",
        "message": "Trip planning failed. Please try again later.",
        "step": trip.NODE_ERROR,
    }
    assert "provider secret" not in frames[-1]
