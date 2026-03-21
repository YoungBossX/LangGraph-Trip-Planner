"""Agent 智能体定义 — 每个 Agent 自主调用工具（ReAct 模式）"""

from typing import Dict, Any, List, Tuple
import logging
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from ..services.llm_service import get_llm

logger = logging.getLogger(__name__)

ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。你拥有高德地图工具，请自主完成以下任务：

**你的任务流程：**
1. 根据用户城市和偏好，使用 maps_text_search 搜索景点（可以搜索多次，用不同关键词）
2. 从搜索结果中筛选出最适合旅游的 6 个景点
3. 对每个选中的景点，使用 maps_geo 获取精确经纬度坐标
4. 整理所有信息，输出结构化 JSON

**工具使用要点：**
- maps_text_search：参数为 keywords（搜索词）和 city（城市名），返回 pois 数组
- maps_geo：参数为 address（地址）和 city（城市名），返回 geocodes 数组中的 location 字段（格式 "经度,纬度"）
- 搜索最多 2 次（不同关键词），不要超过 2 次
- 每个景点都必须调用 maps_geo 获取坐标

**最终输出（极其重要，必须严格遵守）：**
- 完成所有工具调用后，直接输出 JSON 数组，第一个字符必须是 [
- 禁止在 JSON 前面写任何文字（如"以下是"、"整理信息"等）
- 禁止使用 Markdown 代码块
- description 不超过 15 个字

格式：
[{"name":"名称","address":"地址","location":{"longitude":120.0,"latitude":30.0},"visit_duration":120,"description":"简短描述","category":"分类","ticket_price":0}]
"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。你拥有高德地图天气工具，请自主完成任务。

**你的任务流程：**
1. 使用 maps_weather 工具查询指定城市的天气预报
2. 从工具返回结果中提取每天的天气数据
3. 整理为结构化 JSON 输出

**工具使用要点：**
- maps_weather：参数为 city（城市名），返回 forecasts 数组，其中 casts 包含每天天气
- 只需调用一次工具

**最终输出格式（严格遵守，只输出 JSON 数组）：**
[
  {
    "date": "2025-06-01",
    "day_weather": "晴",
    "night_weather": "多云",
    "day_temp": 28,
    "night_temp": 18,
    "wind_direction": "南风",
    "wind_power": "1-3级"
  }
]

**注意事项：**
- 所有数据必须来自工具返回的真实值
- 温度必须是纯数字，不带单位
- 日期使用工具返回的真实日期
- 输出必须是纯 JSON 数组，禁止 Markdown
"""

HOTEL_AGENT_PROMPT = """你是酒店搜索专家。你拥有高德地图工具，请自主完成以下任务：

**你的任务流程：**
1. 使用 maps_text_search 搜索酒店（keywords 用"酒店"或用户指定的住宿类型）
2. 从搜索结果中筛选出最合适的 3 个酒店
3. 对每个选中的酒店，使用 maps_geo 获取精确经纬度坐标
4. 整理所有信息，输出结构化 JSON

**工具使用要点：**
- maps_text_search：参数为 keywords 和 city，搜索酒店，只搜索 1 次
- maps_geo：参数为 address 和 city，获取坐标
- 每个酒店都必须调用 maps_geo 获取坐标

**最终输出（极其重要，必须严格遵守）：**
- 完成所有工具调用后，直接输出 JSON 数组，第一个字符必须是 [
- 禁止在 JSON 前面写任何文字
- 禁止使用 Markdown 代码块

格式：
[{"name":"酒店名","address":"地址","location":{"longitude":120.0,"latitude":30.0},"price_range":"200-400元","rating":4.5,"type":"经济型酒店","estimated_cost":300}]
"""

PLANNER_AGENT_PROMPT = """你是行程规划专家。根据提供的景点、天气、酒店信息，生成详细的旅行计划。

硬性要求：
1. 只基于输入中提供的景点、天气、酒店信息进行规划，不要编造新的景点、酒店或天气数据。
2. 行程天数必须与用户请求的 travel_days 一致，日期范围必须与用户请求一致。
3. 每天合理安排景点、餐饮、交通和住宿，避免明显不合理的时间冲突。
4. 景点和酒店的 location 经纬度必须原样复制输入数据中的值。
5. weather_info 应覆盖行程涉及的日期；温度必须是纯数字。
6. meals 中应包含 breakfast、lunch、dinner。
7. budget 必须包含 total_attractions、total_hotels、total_meals、total_transportation、total。
8. 最终回答必须只输出一个 JSON 对象；不要输出任何解释、Markdown、代码块。
9. JSON 顶层格式：
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 0.0, "latitude": 0.0},
        "price_range": "",
        "rating": 4.5,
        "distance": "",
        "type": "经济型酒店",
        "estimated_cost": 300
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "详细地址",
          "location": {"longitude": 0.0, "latitude": 0.0},
          "visit_duration": 120,
          "description": "景点描述",
          "category": "景点",
          "ticket_price": 0
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐建议", "description": "简短说明", "estimated_cost": 20},
        {"type": "lunch", "name": "午餐建议", "description": "简短说明", "estimated_cost": 40},
        {"type": "dinner", "name": "晚餐建议", "description": "简短说明", "estimated_cost": 60}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "东风",
      "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "总体建议",
  "budget": {
    "total_attractions": 0,
    "total_hotels": 0,
    "total_meals": 0,
    "total_transportation": 0,
    "total": 0
  }
}
"""

# ============================================================
# Agent 工厂函数
# ============================================================

def create_attraction_search_agent(tools: List[BaseTool]):
    """创建景点搜索智能体 — 自主调用 search + geo 工具"""
    logger.info("创建景点搜索智能体 (Agent 模式)...")
    llm = get_llm()
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=ATTRACTION_AGENT_PROMPT,
        debug=False,
    )
    logger.info(f"[OK] 景点搜索智能体创建成功，工具: {[t.name for t in tools]}")
    return agent_graph


def create_weather_agent(tools: List[BaseTool]):
    """创建天气查询智能体 — 自主调用 weather 工具"""
    logger.info("创建天气查询智能体 (Agent 模式)...")
    llm = get_llm()
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=WEATHER_AGENT_PROMPT,
        debug=False,
    )
    logger.info(f"[OK] 天气查询智能体创建成功，工具: {[t.name for t in tools]}")
    return agent_graph


def create_hotel_agent(tools: List[BaseTool]):
    """创建酒店推荐智能体 — 自主调用 search + geo 工具"""
    logger.info("创建酒店推荐智能体 (Agent 模式)...")
    llm = get_llm()
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=HOTEL_AGENT_PROMPT,
        debug=False,
    )
    logger.info(f"[OK] 酒店推荐智能体创建成功，工具: {[t.name for t in tools]}")
    return agent_graph


def create_planner_agent(tools: List[BaseTool]):
    """创建行程规划智能体 — 纯文本生成，无工具"""
    logger.info("创建行程规划智能体...")
    llm = get_llm()
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=PLANNER_AGENT_PROMPT,
        debug=False,
    )
    logger.info("[OK] 行程规划智能体创建成功")
    return agent_graph


# ============================================================
# Agent 缓存管理
# ============================================================

_agent_instances: Dict[Tuple[str, Tuple[str, ...]], Any] = {}


def get_agent(agent_type: str, tools: List[BaseTool]) -> Any:
    """获取智能体实例（带缓存）"""
    cache_key = (agent_type, tuple(sorted(tool.name for tool in tools)))

    if cache_key not in _agent_instances:
        logger.info(f"创建 {agent_type} 智能体...")

        creators = {
            "attraction_search": create_attraction_search_agent,
            "weather": create_weather_agent,
            "hotel": create_hotel_agent,
            "planner": create_planner_agent,
        }

        creator = creators.get(agent_type)
        if not creator:
            raise ValueError(f"未知的智能体类型: {agent_type}")

        _agent_instances[cache_key] = creator(tools)

    return _agent_instances[cache_key]


def clear_agent_cache():
    """清空智能体缓存"""
    global _agent_instances
    _agent_instances.clear()
    logger.info("智能体缓存已清空")