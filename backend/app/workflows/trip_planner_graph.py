"""
旅行规划 LangGraph 工作流 — Agent 版本

核心：
- 每个节点不再手动调用工具，而是让 Agent (ReAct) 自主调用
- 节点函数只做三件事：构造 query → 调用 agent → 解析输出
"""

from typing import Dict, Any, List, Optional
import json
import re
import logging
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, END

from .trip_planner_state import TripPlannerState, create_initial_state
from ..agents.agents import get_agent
from ..tools.amap_mcp_tools import get_cached_amap_tools
from ..models.schemas import (
    TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo,
    Location, Hotel, Budget
)

logger = logging.getLogger(__name__)


class TripPlannerWorkflow:
    """多智能体旅行规划工作流 — Agent 版本

    架构说明：
    - 保持 4 节点 LangGraph 编排：search_attractions → check_weather → find_hotels → plan_itinerary
    - 每个节点的 Agent 自主调用工具（ReAct 模式），节点函数只解析结果
    - 景点/酒店 Agent 同时持有 search + geo 工具，自行搜索和补坐标
    """

    def __init__(self):
        logger.info("🔄 初始化 Agent 旅行规划工作流...")

        try:
            # 加载所有 MCP 工具
            self.tools = get_cached_amap_tools()
            if not self.tools:
                raise RuntimeError("未加载到任何工具")
            logger.info(f"✅ 加载了 {len(self.tools)} 个工具: {[t.name for t in self.tools]}")

            # ============================================================
            # 核心变化：按职责分配工具给 Agent
            # 景点/酒店 Agent 需要 search + geo 两个工具（自主搜索 + 自主补坐标）
            # 天气 Agent 只需要 weather 工具
            # 规划 Agent 不需要工具（纯文本推理）
            # ============================================================

            search_geo_tools = [
                t for t in self.tools
                if t.name in ("maps_text_search", "maps_geo")
            ]
            weather_tools = [
                t for t in self.tools
                if t.name == "maps_weather"
            ]

            logger.info(f"搜索+地理编码工具: {[t.name for t in search_geo_tools]}")
            logger.info(f"天气工具: {[t.name for t in weather_tools]}")

            # 创建 Agent — 注意景点和酒店 Agent 都拿 search + geo
            self.attraction_agent = get_agent("attraction_search", search_geo_tools)
            self.weather_agent = get_agent("weather", weather_tools)
            self.hotel_agent = get_agent("hotel", search_geo_tools)
            self.planner_agent = get_agent("planner", [])

            # 构建工作流图
            self.graph = self._build_graph()
            logger.info("✅ Agent 工作流初始化成功")

        except Exception as e:
            logger.error(f"❌ 工作流初始化失败: {str(e)}", exc_info=True)
            raise

    # ========== StateGraph 构建 ==========

    def _build_graph(self) -> StateGraph:
        # 构建状态图，节点函数保持不变，每个节点内部由 Agent 自主调用工具
        workflow = StateGraph(TripPlannerState)
        # 添加节点
        workflow.add_node("search_attractions", self._search_attractions)
        workflow.add_node("check_weather", self._check_weather)
        workflow.add_node("find_hotels", self._find_hotels)
        workflow.add_node("plan_itinerary", self._plan_itinerary)
        workflow.add_node("handle_error", self._handle_error)
        # 设置入口节点
        workflow.set_entry_point("search_attractions")
        # 添加条件边：每个节点根据是否有 error 决定继续下一步还是跳到错误处理
        workflow.add_conditional_edges(
            "search_attractions", self._check_error,
            {"continue": "check_weather", "error": "handle_error"}
        )
        workflow.add_conditional_edges(
            "check_weather", self._check_error,
            {"continue": "find_hotels", "error": "handle_error"}
        )
        workflow.add_conditional_edges(
            "find_hotels", self._check_error,
            {"continue": "plan_itinerary", "error": "handle_error"}
        )
        workflow.add_conditional_edges(
            "plan_itinerary", self._check_error,
            {"continue": END, "error": "handle_error"}
        )

        workflow.add_conditional_edges(
            "handle_error", self._route_after_error,
            {
            "retry_search_attractions": "search_attractions",
            "retry_check_weather": "check_weather",
            "retry_find_hotels": "find_hotels",
            "retry_plan_itinerary": "plan_itinerary",
            "skip_to_plan": "plan_itinerary",
            "end": END
            }
        )

        return workflow.compile()
    
    def _route_after_error(self, state: TripPlannerState) -> str:
        retry_count = state.get("retry_count", 0)
        failed_node = state.get("failed_node", "")

        if retry_count < 2 and failed_node:
            return f"retry_{failed_node}"

        if state.get("attractions") or state.get("weather_info"):
            return "skip_to_plan"

        return "end"

    def _check_error(self, state: TripPlannerState) -> str:
        return "error" if state.get("error") else "continue"

    # ========== 节点: 景点搜索（Agent） ==========

    def _search_attractions(self, state: TripPlannerState) -> Dict[str, Any]:
        """景点搜索节点：Agent 自主调用 search + geo 工具

        Before: Python loop search_tool.invoke() → Agent filter → Python loop geo_tool.invoke()
        After:  Agent 自己搜索 + 筛选 + 补坐标，节点只解析 JSON
        """
        logger.info("📍 [Agent] 搜索景点...")
        try:
            request = state["request"]

            # 构造给 Agent 的查询
            prefs = ', '.join(request.preferences) if request.preferences else '综合'
            query = (
                f"请搜索 {request.city} 的旅游景点。\n"
                f"用户偏好: {prefs}\n"
                f"旅行天数: {request.travel_days} 天\n"
                f"请根据偏好用不同关键词多次搜索（如 '{request.city}历史古迹'、'{request.city}自然风光' 等），"
                f"筛选出最适合的 6 个景点，并用 maps_geo 获取每个景点的坐标。"
            )

            # 调用 Agent — Agent 自主执行 ReAct 循环
            result = self.attraction_agent.invoke(
                self._prepare_agent_input(query, []),
                config={"recursion_limit": 30}  # 给够迭代次数，因为需要多次工具调用
            )

            output = self._extract_agent_output(result)
            logger.info(f"Agent 输出前300字符: {output[:300]}")

            # 解析 Agent 返回的 JSON → Attraction 对象列表
            attractions = self._parse_attractions_from_agent(output, request.city)
            logger.info(f"解析到 {len(attractions)} 个景点")

            return {
                "attractions": attractions,
                "messages": [{"role": "assistant", "content": f"已找到 {len(attractions)} 个景点"}]
            }
        except Exception as e:
            logger.error(f"景点搜索失败: {str(e)}", exc_info=True)
            return {"error": f"景点搜索失败: {str(e)}", "current_step": "error", "failed_node": "search_attractions"}

    # ========== 节点: 天气查询（Agent） ==========

    def _check_weather(self, state: TripPlannerState) -> Dict[str, Any]:
        """天气查询节点：Agent 自主调用 weather 工具"""
        logger.info("🌤️  [Agent] 查询天气...")
        try:
            request = state["request"]
            query = (
                f"请查询 {request.city} 的天气预报。\n"
                f"旅行日期为 {request.start_date} 至 {request.end_date}。"
            )

            result = self.weather_agent.invoke(
                self._prepare_agent_input(query, []),
                config={"recursion_limit": 10}
            )

            output = self._extract_agent_output(result)
            logger.info(f"Agent 输出前300字符: {output[:300]}")

            weather_info = self._parse_weather(output)
            logger.info(f"解析到 {len(weather_info)} 条天气信息")

            return {
                "weather_info": weather_info,
                "messages": [{"role": "assistant", "content": f"已获取 {len(weather_info)} 天天气信息"}]
            }
        except Exception as e:
            logger.error(f"天气查询失败: {str(e)}", exc_info=True)
            return {"error": f"天气查询失败: {str(e)}", "current_step": "error", "failed_node": "check_weather"}

    # ========== 节点: 酒店搜索（Agent） ==========

    def _find_hotels(self, state: TripPlannerState) -> Dict[str, Any]:
        """酒店搜索节点：Agent 自主调用 search + geo 工具

        Before: Python search_tool.invoke() → Agent filter → Python geo_tool.invoke()
        After:  Agent 自己搜索 + 筛选 + 补坐标
        """
        logger.info("🏨 [Agent] 搜索酒店...")
        try:
            request = state["request"]
            query = (
                f"请搜索 {request.city} 的酒店。\n"
                f"用户住宿偏好: {request.accommodation}\n"
                f"请搜索合适的酒店，筛选出 3 个最佳选择，"
                f"并用 maps_geo 获取每个酒店的坐标。"
            )

            result = self.hotel_agent.invoke(
                self._prepare_agent_input(query, []),
                config={"recursion_limit": 20}
            )

            output = self._extract_agent_output(result)
            logger.info(f"Agent 输出前300字符: {output[:300]}")

            hotels = self._parse_hotels_from_agent(output, request)
            logger.info(f"解析到 {len(hotels)} 个酒店")

            return {
                "hotels": hotels,
                "current_step": "hotels_found",
                "messages": [{"role": "assistant", "content": f"已找到 {len(hotels)} 个酒店"}]
            }
        except Exception as e:
            logger.error(f"酒店搜索失败: {str(e)}", exc_info=True)
            return {"error": f"酒店搜索失败: {str(e)}", "current_step": "error", "failed_node": "find_hotels"}

    # ========== 节点: 行程规划（保持不变，本来就是 Agent） ==========

    def _plan_itinerary(self, state: TripPlannerState) -> Dict[str, Any]:
        """行程规划节点：Agent 综合所有数据生成行程计划"""
        logger.info("📋 [Agent] 生成行程计划...")
        try:
            query = self._build_planner_query(
                state["request"], state["attractions"],
                state["weather_info"], state["hotels"]
            )
            logger.info(
                f"传给 planner 的景点数: {len(state['attractions'])}, "
                f"天气数: {len(state['weather_info'])}, 酒店数: {len(state['hotels'])}"
            )

            result = self.planner_agent.invoke(
                self._prepare_agent_input(query, []),
                config={"recursion_limit": 50}
            )

            output = self._extract_agent_output(result)
            logger.info(f"Planner 输出前300字符: {output[:300]}")

            trip_plan = self._parse_trip_plan(output, state["request"])
            logger.info(f"解析到 {len(trip_plan.days)} 天行程")

            return {
                "trip_plan": trip_plan,
                "current_step": "plan_completed",
                "messages": [{"role": "assistant", "content": "行程计划生成完成！"}]
            }
        except Exception as e:
            logger.error(f"行程规划失败: {str(e)}", exc_info=True)
            return {"error": f"行程规划失败: {str(e)}", "current_step": "error", "failed_node": "plan_itinerary"}

    # ========== 节点: 错误处理 ==========

    def _handle_error(self, state: TripPlannerState) -> Dict[str, Any]:
        retry_count = state.get("retry_count", 0)
        failed_node = state.get("failed_node", "未知")
        last_failed = state.get("last_failed_node", "")
        error_msg = state.get("error", "未知错误")

        if failed_node != last_failed:
            retry_count = 0

        logger.warning(f"⚠️ 节点 [{failed_node}] 失败: {error_msg}, 已重试 {retry_count} 次")

        if retry_count < 2:
            return {
                "error": None,
                "retry_count": retry_count + 1,
                "last_failed_node": failed_node,
            }

        if state.get("attractions") or state.get("weather_info"):
            logger.info("有部分数据，跳过失败节点继续规划")
            return {
                "error": None,
                "failed_node": None,
            }

        logger.info("无可用数据，生成备用计划")
        return {
            "trip_plan": self._create_fallback_plan(state["request"]),
            "error": None,
            "failed_node": None,
        }

    # ========== Agent 输出解析：景点 ==========

    def _parse_attractions_from_agent(self, output: str, city: str) -> List[Attraction]:
        """解析 Agent 返回的景点 JSON → Attraction 对象列表

        Agent 已经自主完成了搜索+筛选+补坐标，这里只做 JSON → Pydantic 转换。
        """
        try:
            json_str = self._extract_json(output)
            data = json.loads(json_str)

            # 兼容 Agent 返回 {"attractions": [...]} 或直接 [...]
            if isinstance(data, dict):
                data = data.get("attractions", data.get("results", []))
            if not isinstance(data, list):
                logger.warning("Agent 返回的不是数组格式")
                return []

            attractions = []
            for item in data:
                if not isinstance(item, dict) or not item.get("name", "").strip():
                    continue
                try:
                    attractions.append(Attraction(
                        name=item["name"],
                        address=item.get("address", ""),
                        location=self._parse_location(item.get("location")),
                        visit_duration=item.get("visit_duration", 120),
                        description=item.get("description", ""),
                        category=item.get("category", "景点"),
                        ticket_price=item.get("ticket_price", 0),
                        poi_id=item.get("id") or item.get("poi_id"),
                    ))
                except Exception as e:
                    logger.warning(f"解析景点 '{item.get('name')}' 失败: {e}")

            return attractions

        except Exception as e:
            logger.error(f"解析景点 JSON 失败: {e}")
            return []

    # ========== Agent 输出解析：酒店 ==========

    def _parse_hotels_from_agent(self, output: str, request: TripRequest) -> List[Hotel]:
        """解析 Agent 返回的酒店 JSON → Hotel 对象列表"""
        try:
            json_str = self._extract_json(output)
            data = json.loads(json_str)

            if isinstance(data, dict):
                data = data.get("hotels", data.get("results", []))
            if not isinstance(data, list):
                return []

            hotels = []
            for item in data:
                if not isinstance(item, dict) or not item.get("name", "").strip():
                    continue
                try:
                    hotels.append(Hotel(
                        name=item["name"],
                        address=item.get("address", ""),
                        location=self._parse_location(item.get("location")),
                        price_range=item.get("price_range", ""),
                        rating=item.get("rating"),
                        distance=item.get("distance", ""),
                        type=item.get("type", request.accommodation),
                        estimated_cost=item.get("estimated_cost", 0),
                    ))
                except Exception as e:
                    logger.warning(f"解析酒店 '{item.get('name')}' 失败: {e}")

            return hotels

        except Exception as e:
            logger.error(f"解析酒店 JSON 失败: {e}")
            return []

    # ========== 查询构建 ==========

    def _build_planner_query(self, request: TripRequest, attractions: List[Attraction],
                             weather: List[WeatherInfo], hotels: List[Hotel]) -> str:
        """构建给规划 Agent 的查询"""

        def _attraction_to_dict(a: Attraction) -> dict:
            return {
                "name": a.name, "address": a.address,
                "location": {"longitude": a.location.longitude, "latitude": a.location.latitude} if a.location else None,
                "visit_duration": a.visit_duration, "description": a.description,
                "category": a.category, "ticket_price": a.ticket_price,
            }

        def _weather_to_dict(w: WeatherInfo) -> dict:
            return {
                "date": w.date, "day_weather": w.day_weather, "night_weather": w.night_weather,
                "day_temp": w.day_temp, "night_temp": w.night_temp,
                "wind_direction": w.wind_direction, "wind_power": w.wind_power,
            }

        def _hotel_to_dict(h: Hotel) -> dict:
            return {
                "name": h.name, "address": h.address,
                "location": {"longitude": h.location.longitude, "latitude": h.location.latitude} if h.location else None,
                "price_range": h.price_range, "rating": h.rating,
                "type": h.type, "estimated_cost": h.estimated_cost,
            }

        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

**景点信息（共{len(attractions)}个）:**
{json.dumps([_attraction_to_dict(a) for a in attractions], ensure_ascii=False, indent=2)}

**天气信息:**
{json.dumps([_weather_to_dict(w) for w in weather], ensure_ascii=False, indent=2)}

**酒店信息（共{len(hotels)}个）:**
{json.dumps([_hotel_to_dict(h) for h in hotels], ensure_ascii=False, indent=2)}

**要求:**
1. 每天安排2-3个景点（从上面的景点中选择）
2. 每天必须包含早中晚三餐
3. 每天推荐一个酒店（从上面的酒店中选择）
4. 考虑景点之间的距离和交通方式
5. 景点和酒店的经纬度坐标必须原样复制输入数据中的值
6. 必须包含预算汇总
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"
        return query

    # ========== Agent I/O 工具方法 ==========

    def _prepare_agent_input(self, user_input: str, chat_history: list) -> dict:
        messages = list(chat_history)
        messages.append({"role": "user", "content": user_input})
        return {"messages": messages}

    def _normalize_message_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if "text" in content and isinstance(content["text"], str):
                return content["text"]
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _extract_agent_output(self, result: dict) -> str:
        """从 Agent 结果中提取最终文本输出"""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if isinstance(msg, dict):
                    role = msg.get("role") or msg.get("type")
                    if role in ("assistant", "ai"):
                        content = self._normalize_message_content(msg.get("content", ""))
                        if content:
                            return content
                else:
                    msg_type = getattr(msg, "type", getattr(msg, "role", None))
                    if msg_type in ("assistant", "ai"):
                        content = self._normalize_message_content(getattr(msg, "content", ""))
                        if content:
                            return content

        for key in ("output", "text", "response", "content"):
            if key in result:
                return self._normalize_message_content(result[key])
        return self._normalize_message_content(result)

    # ========== JSON 解析工具 ==========

    def _balanced_json_segments(self, text: str) -> List[str]:
        segments: List[str] = []
        stack: List[str] = []
        start_idx: Optional[int] = None
        in_string = False
        escape = False
        pairs = {"{": "}", "[": "]"}
        closing = {"}": "{", "]": "["}

        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in pairs:
                if not stack:
                    start_idx = idx
                stack.append(ch)
                continue
            if ch in closing and stack:
                if stack[-1] == closing[ch]:
                    stack.pop()
                    if not stack and start_idx is not None:
                        segments.append(text[start_idx:idx + 1].strip())
                        start_idx = None
                else:
                    stack = []
                    start_idx = None
        return segments

    def _safe_load_json(self, text: str) -> Optional[Any]:
        if not text:
            return None
        cleaned = text.strip().replace("\ufeff", "").replace("\u200b", "")
        try:
            return json.loads(cleaned)
        except Exception:
            return None

    def _extract_json(self, response: str, preferred_keys: Optional[List[str]] = None) -> str:
        if not response:
            raise ValueError("响应为空")

        text = response.strip().replace("\ufeff", "").replace("\u200b", "")

        for pattern in [r"```json\s*(.*?)\s*```", r"```python\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            for block in matches:
                block = block.strip()
                if self._safe_load_json(block) is not None:
                    return block

        segments = self._balanced_json_segments(text)
        if preferred_keys:
            for seg in segments:
                parsed = self._safe_load_json(seg)
                if isinstance(parsed, dict):
                    for key in preferred_keys:
                        if key in parsed:
                            return seg
        for seg in segments:
            if self._safe_load_json(seg) is not None:
                return seg
        if self._safe_load_json(text) is not None:
            return text

        # ========== 截断修复 ==========
        # Agent 输出被 max_tokens 截断时，JSON 数组 [ {...}, {... 没闭合
        # 策略：找到最后一个完整的 }, 截断后面的残片，补上 ]
        repaired = self._try_repair_truncated_json(text)
        if repaired is not None:
            return repaired

        raise ValueError("未找到可解析的 JSON 内容")

    # ========== 截断 JSON 修复 ==========

    def _try_repair_truncated_json(self, text: str) -> Optional[str]:
        """尝试修复被 max_tokens 截断的 JSON

        典型场景：Agent 输出 "一些文字...[{完整}, {完整}, {不完整"
        修复策略：找到数组开头 [，找到最后一个完整的 }，截断 + 补 ]
        """
        # 找到第一个 [ 的位置（JSON 数组开始）
        arr_start = text.find('[')
        if arr_start == -1:
            # 也尝试修复截断的 JSON 对象 {...
            obj_start = text.find('{')
            if obj_start == -1:
                return None
            return self._repair_truncated_object(text[obj_start:])

        arr_text = text[arr_start:]

        # 找到最后一个 } 的位置（最后一个完整对象的结尾）
        last_brace = arr_text.rfind('}')
        if last_brace == -1:
            return None

        # 截取到最后一个完整 } ，补上 ]
        candidate = arr_text[:last_brace + 1].rstrip().rstrip(',') + ']'
        parsed = self._safe_load_json(candidate)
        if parsed is not None and isinstance(parsed, list) and len(parsed) > 0:
            logger.warning(f"[截断修复] 成功修复截断的 JSON 数组，保留了 {len(parsed)} 个元素")
            return candidate

        # 如果直接补 ] 不行，可能最后一个对象也不完整，再回退一个
        second_last = arr_text.rfind('},', 0, last_brace)
        if second_last != -1:
            candidate = arr_text[:second_last + 1] + ']'
            parsed = self._safe_load_json(candidate)
            if parsed is not None and isinstance(parsed, list) and len(parsed) > 0:
                logger.warning(f"[截断修复] 回退一个元素后修复成功，保留了 {len(parsed)} 个元素")
                return candidate

        return None

    def _repair_truncated_object(self, text: str) -> Optional[str]:
        """尝试修复截断的 JSON 对象（用于 planner 输出）"""
        # 找 days 数组中最后一个完整天
        # 策略：逐步加 } ] } 直到能解析
        for suffix in ['}', ']}', ']}]}', '"]}]}', '"}]}']:
            for extra in ['', '}', '}}']:
                candidate = text.rstrip().rstrip(',') + extra + suffix
                parsed = self._safe_load_json(candidate)
                if parsed is not None and isinstance(parsed, dict):
                    logger.warning(f"[截断修复] 修复了截断的 JSON 对象")
                    return candidate
        return None

    # ========== 通用数据解析 ==========

    def _parse_location(self, raw_location: Any) -> Optional[Location]:
        if raw_location is None:
            return None
        if isinstance(raw_location, str) and "," in raw_location:
            try:
                parts = raw_location.split(",")
                return Location(longitude=float(parts[0].strip()), latitude=float(parts[1].strip()))
            except (ValueError, IndexError):
                return None
        if not isinstance(raw_location, dict):
            return None
        longitude = raw_location.get("longitude")
        latitude = raw_location.get("latitude")
        if longitude in (None, "") or latitude in (None, ""):
            return None
        try:
            return Location(longitude=float(longitude), latitude=float(latitude))
        except (TypeError, ValueError):
            return None

    def _parse_weather(self, response: str) -> List[WeatherInfo]:
        try:
            json_str = self._extract_json(response, preferred_keys=["weather_info"])
            data = json.loads(json_str)
        except Exception as e:
            logger.error(f"解析天气 JSON 失败: {e}")
            return []

        if isinstance(data, dict):
            data = data.get("weather_info", data.get("forecasts", []))
        if not isinstance(data, list):
            return []

        weather_info: List[WeatherInfo] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                w = WeatherInfo(
                    date=item.get("date", ""),
                    day_weather=item.get("day_weather", item.get("dayweather", "")),
                    night_weather=item.get("night_weather", item.get("nightweather", "")),
                    day_temp=item.get("day_temp", item.get("daytemp", 0)),
                    night_temp=item.get("night_temp", item.get("nighttemp", 0)),
                    wind_direction=item.get("wind_direction", item.get("daywind", "")),
                    wind_power=item.get("wind_power", item.get("daypower", "")),
                )
                if w.date.strip():
                    weather_info.append(w)
            except Exception as e:
                logger.warning(f"跳过天气数据: {e}")
        return weather_info

    def _parse_trip_plan(self, response: str, request: TripRequest) -> TripPlan:
        """解析行程计划 JSON（保持原有逻辑）"""
        try:
            json_str = self._extract_json(response, preferred_keys=["days", "trip_plan"])
            data = json.loads(json_str)
        except Exception as e:
            logger.error(f"解析行程计划 JSON 失败: {e}")
            return self._create_fallback_plan(request)

        if isinstance(data, dict) and isinstance(data.get("trip_plan"), dict):
            data = data["trip_plan"]
        if not isinstance(data, dict):
            return self._create_fallback_plan(request)

        trip_plan = TripPlan(
            city=data.get("city", request.city),
            start_date=data.get("start_date", request.start_date),
            end_date=data.get("end_date", request.end_date),
            days=[], weather_info=[],
            overall_suggestions=data.get("overall_suggestions", ""),
            budget=None,
        )

        # 解析天气
        for wd in data.get("weather_info", []):
            if not isinstance(wd, dict):
                continue
            try:
                wi = WeatherInfo(
                    date=wd.get("date", ""),
                    day_weather=wd.get("day_weather", ""),
                    night_weather=wd.get("night_weather", ""),
                    day_temp=wd.get("day_temp", 0),
                    night_temp=wd.get("night_temp", 0),
                    wind_direction=wd.get("wind_direction", ""),
                    wind_power=wd.get("wind_power", ""),
                )
                if wi.date.strip():
                    trip_plan.weather_info.append(wi)
            except Exception:
                pass

        # 解析每日行程
        for day_idx, day_data in enumerate(data.get("days", [])):
            if not isinstance(day_data, dict):
                continue
            try:
                attractions = []
                for ad in day_data.get("attractions", []):
                    if not isinstance(ad, dict):
                        continue
                    try:
                        a = Attraction(
                            name=ad.get("name", ""),
                            address=ad.get("address", ""),
                            location=self._parse_location(ad.get("location")),
                            visit_duration=ad.get("visit_duration", 0),
                            description=ad.get("description", ""),
                            category=ad.get("category", "景点"),
                            rating=ad.get("rating"),
                            photos=ad.get("photos", []) or [],
                            poi_id=ad.get("poi_id"),
                            image_url=ad.get("image_url"),
                            ticket_price=ad.get("ticket_price", 0),
                            price_text=ad.get("price_text", ""),
                        )
                        if a.name.strip():
                            attractions.append(a)
                    except Exception as e:
                        logger.warning(f"第{day_idx+1}天景点解析失败: {e}")

                meals = []
                for md in day_data.get("meals", []):
                    if not isinstance(md, dict):
                        continue
                    try:
                        m = Meal(**md)
                        if hasattr(m, "name") and getattr(m, "name", "").strip():
                            meals.append(m)
                    except Exception as e:
                        logger.warning(f"第{day_idx+1}天餐饮解析失败: {e}")

                hotel = None
                hd = day_data.get("hotel")
                if isinstance(hd, dict):
                    try:
                        hotel = Hotel(
                            name=hd.get("name", ""),
                            address=hd.get("address", ""),
                            location=self._parse_location(hd.get("location")),
                            price_range=hd.get("price_range", ""),
                            rating=hd.get("rating"),
                            distance=hd.get("distance", ""),
                            type=hd.get("type", ""),
                            estimated_cost=hd.get("estimated_cost", 0),
                        )
                        if not hotel.name.strip():
                            hotel = None
                    except Exception:
                        hotel = None

                day_plan = DayPlan(
                    date=day_data.get("date", ""),
                    day_index=day_data.get("day_index", day_idx),
                    description=day_data.get("description", ""),
                    transportation=day_data.get("transportation", request.transportation),
                    accommodation=day_data.get("accommodation", request.accommodation),
                    hotel=hotel, attractions=attractions, meals=meals,
                )
                trip_plan.days.append(day_plan)
            except Exception as e:
                logger.warning(f"跳过第{day_idx+1}天: {e}")

        # 预算
        bd = data.get("budget")
        if isinstance(bd, dict):
            try:
                trip_plan.budget = Budget(**bd)
            except Exception as e:
                logger.warning(f"预算解析失败: {e}")

        if not trip_plan.days:
            return self._create_fallback_plan(request)

        return trip_plan

    # ========== 备用计划 ==========

    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        try:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        except ValueError:
            start_date = datetime.now()

        days = []
        for i in range(request.travel_days):
            current = start_date + timedelta(days=i)
            days.append(DayPlan(
                date=current.strftime("%Y-%m-%d"), day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}", address=f"{request.city}市",
                        location=Location(
                            longitude=120.1551 + i * 0.01 + j * 0.005,
                            latitude=30.2741 + i * 0.01 + j * 0.005,
                        ),
                        visit_duration=120, description=f"{request.city}的著名景点", category="景点",
                    ) for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐"),
                ],
            ))

        return TripPlan(
            city=request.city, start_date=request.start_date, end_date=request.end_date,
            days=days, weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程，建议提前查看各景点的开放时间。",
        )

    # ========== 入口 ==========

    def plan_trip(self, request: TripRequest) -> TripPlan:
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 开始 Agent 旅行规划工作流...")
        logger.info(f"目的地: {request.city}")
        logger.info(f"{'='*60}\n")

        initial_state: TripPlannerState = create_initial_state(request)
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": 80})

        if final_state.get("error") and not final_state.get("trip_plan"):
            error_msg = final_state.get("error", "未知错误")
            logger.error(f"❌ 旅行规划失败: {error_msg}")
            raise Exception(error_msg)

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ 旅行计划生成完成!")
        logger.info(f"{'='*60}\n")

        return final_state["trip_plan"]

# ========== 全局工作流实例 ==========

_trip_planner_workflow: Optional[TripPlannerWorkflow] = None

def get_trip_planner_workflow() -> TripPlannerWorkflow:
    global _trip_planner_workflow
    if _trip_planner_workflow is None:
        _trip_planner_workflow = TripPlannerWorkflow()
    return _trip_planner_workflow

def reset_workflow():
    global _trip_planner_workflow
    _trip_planner_workflow = None
    logger.info("工作流实例已重置")