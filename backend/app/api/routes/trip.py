"""旅行规划API路由 (LangGraph 版本)"""

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...config import get_settings
from ...models.schemas import TripPlanResponse, TripRequest
from ...workflows.execution_control import ExecutionControl, WorkflowTimeoutError
from ...workflows.trip_planner_graph import NODE_ERROR, get_trip_planner_workflow
from ..guards import (
    PlanningLease,
    PublicAPIError,
    get_client_ip,
    get_planning_admission_controller,
    get_rate_limiter,
)

router = APIRouter(prefix="/trip", tags=["旅行规划"])
logger = logging.getLogger(__name__)

_STEP_LABELS = {
    "search_attractions": "正在搜索景点...",
    "check_weather": "正在查询天气...",
    "find_hotels": "正在搜索酒店...",
    "context_ready": "正在整合天气和酒店...",
    "plan_itinerary": "正在生成行程计划...",
    "handle_error": "正在恢复...",
}

_RATE_LIMIT_MESSAGE = "Too many requests. Please retry later."
_TRIP_TIMEOUT_MESSAGE = "Trip planning timed out. Please try again."
_TRIP_FAILED_MESSAGE = "Trip planning failed. Please try again later."
_HEALTH_UNAVAILABLE_MESSAGE = "Trip planner service is unavailable. Please try again later."
_PLANNING_RATE_MARKER = "trip_planning_rate_checked"
_SSE_QUEUE_MAXSIZE = 16
_SSE_DONE = object()
_SSE_FAILED = object()
_SSE_TIMEOUT = object()
_TIMEOUT_EXCEPTIONS = (WorkflowTimeoutError, asyncio.TimeoutError, TimeoutError)
_WAIT_TIMEOUT_EXCEPTIONS = (asyncio.TimeoutError, TimeoutError)


class _PlanningLifecycle:
    """Share idempotent cleanup between the async generator and response wrapper."""

    def __init__(self, control: ExecutionControl, lease: PlanningLease):
        self.control = control
        self.lease = lease
        self._cleanup_task = None

    async def cleanup(self) -> None:
        if self._cleanup_task is None:
            self.control.cancel()
            self._cleanup_task = asyncio.create_task(self.lease.release())
        await asyncio.shield(self._cleanup_task)


class _LifecycleStreamingResponse(StreamingResponse):
    def __init__(self, content, *, lifecycle: _PlanningLifecycle, **kwargs):
        super().__init__(content, **kwargs)
        self.lifecycle = lifecycle

    async def __call__(self, scope, receive, send):
        try:
            return await super().__call__(scope, receive, send)
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                try:
                    await close()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Failed to close SSE body iterator")
            await self.lifecycle.cleanup()


async def _enforce_planning_rate_limit(request: Request) -> bool:
    state = request.scope.setdefault("state", {}) if hasattr(request, "scope") else None
    if state is not None and state.get(_PLANNING_RATE_MARKER):
        return True

    settings = get_settings()
    decision = await get_rate_limiter().consume(
        "trip-plan",
        get_client_ip(request),
        limit=settings.planning_rate_limit,
        window_seconds=settings.planning_rate_window_seconds,
    )
    if not decision.allowed:
        raise PublicAPIError(429, "RATE_LIMITED", _RATE_LIMIT_MESSAGE, decision.retry_after)
    if state is not None:
        state[_PLANNING_RATE_MARKER] = True
    return True


async def _run_sync_with_deadline(function, control: ExecutionControl, *args):
    return await asyncio.wait_for(
        asyncio.to_thread(function, *args),
        timeout=control.remaining(),
    )


async def _ensure_planning_rate_limit(request: Request, rate_checked) -> None:
    if rate_checked is not True:
        await _enforce_planning_rate_limit(request)


@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="生成旅行计划",
    description="根据用户输入的旅行需求,生成详细的旅行计划"
)
async def plan_trip(
    trip_request: TripRequest,
    request: Request,
    rate_checked: bool = Depends(_enforce_planning_rate_limit),
):
    await _ensure_planning_rate_limit(request, rate_checked)
    lease = await get_planning_admission_controller().acquire(get_client_ip(request))
    control = None

    try:
        control = ExecutionControl(get_settings().trip_request_timeout_seconds)
        logger.info("Received trip planning request: %s, %s days", trip_request.city, trip_request.travel_days)
        workflow = await _run_sync_with_deadline(get_trip_planner_workflow, control)
        try:
            trip_plan = await _run_sync_with_deadline(workflow.plan_trip, control, trip_request, control)
        except _TIMEOUT_EXCEPTIONS as exc:
            control.cancel()
            raise PublicAPIError(504, "TRIP_TIMEOUT", _TRIP_TIMEOUT_MESSAGE) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            control.cancel()
            logger.exception("Trip planning workflow failed")
            raise PublicAPIError(500, "TRIP_FAILED", _TRIP_FAILED_MESSAGE) from exc

        logger.info("Trip planning completed successfully")
        return TripPlanResponse(success=True, message="旅行计划生成成功", data=trip_plan)
    except asyncio.CancelledError:
        if control is not None:
            control.cancel()
        raise
    except _TIMEOUT_EXCEPTIONS as exc:
        if control is not None:
            control.cancel()
        raise PublicAPIError(504, "TRIP_TIMEOUT", _TRIP_TIMEOUT_MESSAGE) from exc
    except PublicAPIError:
        raise
    except Exception as exc:
        if control is not None:
            control.cancel()
        logger.exception("Trip planning request failed")
        raise PublicAPIError(500, "TRIP_FAILED", _TRIP_FAILED_MESSAGE) from exc
    finally:
        await lease.release()


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _create_sse_queue() -> asyncio.Queue:
    return asyncio.Queue(maxsize=_SSE_QUEUE_MAXSIZE)


async def _produce_trip_events(queue: asyncio.Queue, workflow, trip_request: TripRequest, control: ExecutionControl):
    try:
        async for item in workflow.astream_plan(trip_request, control=control):
            await queue.put(item)
    except asyncio.CancelledError:
        raise
    except _TIMEOUT_EXCEPTIONS:
        await queue.put(_SSE_TIMEOUT)
    except Exception:
        logger.exception("Streaming trip planning workflow failed")
        await queue.put(_SSE_FAILED)
    else:
        await queue.put(_SSE_DONE)


async def _stream_trip_events(
    *,
    trip_request: TripRequest,
    request: Request,
    workflow,
    control: ExecutionControl,
    lease: PlanningLease,
    heartbeat_seconds: float,
    lifecycle: _PlanningLifecycle | None = None,
):
    lifecycle = lifecycle or _PlanningLifecycle(control, lease)
    producer = None

    async def can_write() -> bool:
        return not await request.is_disconnected()

    try:
        queue = _create_sse_queue()
        producer = asyncio.create_task(_produce_trip_events(queue, workflow, trip_request, control))
        await asyncio.sleep(0)
        while await can_write():
            remaining = control.remaining()
            if remaining <= 0:
                if await can_write():
                    yield _sse_event(
                        "error",
                        {"code": "TRIP_TIMEOUT", "message": _TRIP_TIMEOUT_MESSAGE},
                    )
                return

            try:
                item = await asyncio.wait_for(queue.get(), timeout=min(float(heartbeat_seconds), remaining))
            except _WAIT_TIMEOUT_EXCEPTIONS:
                if control.remaining() <= 0:
                    if await can_write():
                        yield _sse_event(
                            "error",
                            {"code": "TRIP_TIMEOUT", "message": _TRIP_TIMEOUT_MESSAGE},
                        )
                    return

                if await can_write():
                    yield ": heartbeat\n\n"
                    continue
                return

            if item is _SSE_FAILED:
                if await can_write():
                    yield _sse_event(
                        "error",
                        {"code": "TRIP_FAILED", "message": _TRIP_FAILED_MESSAGE},
                    )
                return

            if item is _SSE_TIMEOUT:
                if await can_write():
                    yield _sse_event(
                        "error",
                        {"code": "TRIP_TIMEOUT", "message": _TRIP_TIMEOUT_MESSAGE},
                    )
                return

            if item is _SSE_DONE:
                if await can_write():
                    yield _sse_event(
                        "error",
                        {"code": "TRIP_FAILED", "message": _TRIP_FAILED_MESSAGE},
                    )
                return

            node_name, node_output = item
            label = _STEP_LABELS.get(node_name, node_name)
            if not await can_write():
                return
            yield _sse_event("progress", {"step": node_name, "message": label})

            if node_output.get("error"):
                if node_name == NODE_ERROR:
                    logger.error("Trip planning workflow ended with an error: %s", node_output["error"])
                    if await can_write():
                        yield _sse_event(
                            "error",
                            {
                                "code": "TRIP_FAILED",
                                "message": _TRIP_FAILED_MESSAGE,
                                "step": node_name,
                            },
                        )
                    return

                if not await can_write():
                    return
                yield _sse_event(
                    "progress",
                    {
                        "step": node_name,
                        "message": f"{label}失败，正在尝试恢复...",
                        "recovering": True,
                    },
                )

            if node_output.get("trip_plan"):
                if not await can_write():
                    return
                yield _sse_event(
                    "result",
                    {
                        "success": True,
                        "message": "旅行计划生成成功",
                        "data": node_output["trip_plan"].model_dump(),
                    },
                )
                return
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("SSE consumer failed")
        try:
            if await can_write():
                yield _sse_event(
                    "error",
                    {"code": "TRIP_FAILED", "message": _TRIP_FAILED_MESSAGE},
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unable to check SSE client before error response")
    finally:
        if producer is not None:
            producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer
        await lifecycle.cleanup()


@router.post(
    "/plan-stream",
    summary="流式生成旅行计划",
    description="通过 SSE 流式返回旅行计划生成进度"
)
async def plan_trip_stream(
    trip_request: TripRequest,
    request: Request,
    rate_checked: bool = Depends(_enforce_planning_rate_limit),
):
    await _ensure_planning_rate_limit(request, rate_checked)
    lease = await get_planning_admission_controller().acquire(get_client_ip(request))
    control = None
    lifecycle = None

    try:
        control = ExecutionControl(get_settings().trip_request_timeout_seconds)
        lifecycle = _PlanningLifecycle(control, lease)
        workflow = await _run_sync_with_deadline(get_trip_planner_workflow, control)
        return _LifecycleStreamingResponse(
            _stream_trip_events(
                trip_request=trip_request,
                request=request,
                workflow=workflow,
                control=control,
                lease=lease,
                heartbeat_seconds=get_settings().sse_heartbeat_seconds,
                lifecycle=lifecycle,
            ),
            lifecycle=lifecycle,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except asyncio.CancelledError:
        if lifecycle is not None:
            await lifecycle.cleanup()
        else:
            if control is not None:
                control.cancel()
            await lease.release()
        raise
    except _TIMEOUT_EXCEPTIONS as exc:
        if lifecycle is not None:
            await lifecycle.cleanup()
        else:
            if control is not None:
                control.cancel()
            await lease.release()
        raise PublicAPIError(504, "TRIP_TIMEOUT", _TRIP_TIMEOUT_MESSAGE) from exc
    except Exception as exc:
        if lifecycle is not None:
            await lifecycle.cleanup()
        else:
            if control is not None:
                control.cancel()
            await lease.release()
        logger.exception("Failed to initialize streaming trip planning")
        raise PublicAPIError(500, "TRIP_FAILED", _TRIP_FAILED_MESSAGE) from exc


@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常"
)
async def health_check():
    try:
        workflow = await asyncio.wait_for(
            asyncio.to_thread(get_trip_planner_workflow),
            timeout=get_settings().trip_request_timeout_seconds,
        )
        return {
            "status": "healthy",
            "service": "trip-planner-langgraph",
            "framework": "LangGraph",
            "graph_compiled": True,
            "tools_loaded": len(workflow.tools) if hasattr(workflow, 'tools') else 0,
        }
    except asyncio.CancelledError:
        raise
    except _WAIT_TIMEOUT_EXCEPTIONS as exc:
        logger.error("健康检查初始化超时", exc_info=True)
        raise HTTPException(status_code=503, detail=_HEALTH_UNAVAILABLE_MESSAGE) from exc
    except Exception as exc:
        logger.error("健康检查失败", exc_info=True)
        raise HTTPException(status_code=503, detail=_HEALTH_UNAVAILABLE_MESSAGE) from exc
