"""旅行规划工作流状态定义"""

from typing import Dict, List, Optional

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ..models.schemas import Attraction, Hotel, TripPlan, TripRequest, WeatherInfo
from .execution_control import ExecutionControl


def update_step(prev: str, new: str) -> str:
    """更新步骤，总是使用新值"""
    return new


def replace_value(prev, new):
    """并行分支写同一控制字段时使用最新值，并允许 None 清空。"""
    return new


class TripPlannerState(TypedDict):
    """旅行规划工作流状态"""
    # 输入
    request: TripRequest
    user_input: str
    control: Optional[ExecutionControl]

    # 中间结果
    attractions: List[Attraction]
    weather_info: List[WeatherInfo]
    hotels: List[Hotel]

    # 智能体通信
    messages: Annotated[List[Dict], add_messages]

    # 最终输出
    trip_plan: Optional[TripPlan]
    error: Annotated[Optional[str], replace_value]
    current_step: Annotated[str, update_step]  # 跟踪当前执行步骤

    # 错误恢复
    failed_node: Annotated[Optional[str], replace_value]
    last_failed_node: Optional[str]
    retry_count: int


# 状态辅助函数
def create_initial_state(
    request: TripRequest,
    user_input: str = "",
    control: Optional[ExecutionControl] = None,
) -> TripPlannerState:
    """创建初始状态"""
    return {
        "request": request,
        "user_input": user_input,
        "control": control,
        "attractions": [],
        "weather_info": [],
        "hotels": [],
        "messages": [],
        "trip_plan": None,
        "error": None,
        "current_step": "started",
        "failed_node": None,
        "last_failed_node": None,
        "retry_count": 0,
    }
