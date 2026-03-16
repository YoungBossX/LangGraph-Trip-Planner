"""旅行规划工作流状态定义"""

from typing import Dict, List, Optional

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from ..models.schemas import TripRequest, TripPlan, Attraction, WeatherInfo, Hotel


def update_step(prev: str, new: str) -> str:
    """更新步骤，总是使用新值"""
    return new


class TripPlannerState(TypedDict):
    """旅行规划工作流状态"""
    # 输入
    request: TripRequest
    user_input: str

    # 中间结果
    attractions: List[Attraction]
    weather_info: List[WeatherInfo]
    hotels: List[Hotel]

    # 智能体通信
    messages: Annotated[List[Dict], add_messages]

    # 最终输出
    trip_plan: Optional[TripPlan]
    error: Optional[str]
    current_step: Annotated[str, update_step]  # 跟踪当前执行步骤


# 状态辅助函数
def create_initial_state(request: TripRequest, user_input: str = "") -> TripPlannerState:
    """创建初始状态"""
    return {
        "request": request,
        "user_input": user_input,
        "attractions": [],
        "weather_info": [],
        "hotels": [],
        "messages": [],
        "trip_plan": None,
        "error": None,
        "current_step": "started"
    }