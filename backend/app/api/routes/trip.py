"""旅行规划API路由 (LangGraph 版本)"""

import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ...models.schemas import TripRequest, TripPlanResponse
from ...workflows.trip_planner_graph import get_trip_planner_workflow

router = APIRouter(prefix="/trip", tags=["旅行规划"])
logger = logging.getLogger(__name__)

_STEP_LABELS = {
    "search_attractions": "正在搜索景点...",
    "check_weather": "正在查询天气...",
    "find_hotels": "正在搜索酒店...",
    "plan_itinerary": "正在生成行程计划...",
    "handle_error": "正在恢复...",
}


@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="生成旅行计划",
    description="根据用户输入的旅行需求,生成详细的旅行计划"
)
async def plan_trip(request: TripRequest):
    try:
        logger.info(f"📥 收到旅行规划请求: {request.city}, {request.travel_days}天")
        workflow = get_trip_planner_workflow()
        trip_plan = workflow.plan_trip(request)
        logger.info("✅ 旅行计划生成成功")
        return TripPlanResponse(success=True, message="旅行计划生成成功", data=trip_plan)
    except Exception as e:
        logger.error(f"❌ 生成旅行计划失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成旅行计划失败: {str(e)}")


async def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@router.post(
    "/plan-stream",
    summary="流式生成旅行计划",
    description="通过 SSE 流式返回旅行计划生成进度"
)
async def plan_trip_stream(request: TripRequest):
    async def event_generator():
        workflow = get_trip_planner_workflow()
        try:
            async for node_name, node_output in workflow.astream_plan(request):
                label = _STEP_LABELS.get(node_name, node_name)
                yield await _sse_event("progress", json.dumps({
                    "step": node_name,
                    "message": label,
                }, ensure_ascii=False))

                if node_output.get("error"):
                    yield await _sse_event("error", json.dumps({
                        "message": node_output["error"],
                        "step": node_name,
                    }, ensure_ascii=False))

                if node_output.get("trip_plan"):
                    trip_plan = node_output["trip_plan"]
                    yield await _sse_event("result", json.dumps({
                        "success": True,
                        "message": "旅行计划生成成功",
                        "data": trip_plan.model_dump(),
                    }, ensure_ascii=False))
                    return

            yield await _sse_event("error", json.dumps({
                "message": "工作流未能生成结果",
            }, ensure_ascii=False))

        except Exception as e:
            logger.error(f"流式规划失败: {str(e)}", exc_info=True)
            yield await _sse_event("error", json.dumps({
                "message": str(e),
            }, ensure_ascii=False))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常"
)
async def health_check():
    try:
        workflow = get_trip_planner_workflow()
        return {
            "status": "healthy",
            "service": "trip-planner-langgraph",
            "framework": "LangGraph",
            "graph_compiled": True,
            "tools_loaded": len(workflow.tools) if hasattr(workflow, 'tools') else 0,
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")
