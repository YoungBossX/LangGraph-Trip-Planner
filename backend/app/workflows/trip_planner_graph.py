"""旅行规划 LangGraph 工作流"""

from typing import Dict, Any, List, Optional
import json
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

# 设置日志记录
logger = logging.getLogger(__name__)

class TripPlannerWorkflow:
    """多智能体旅行规划工作流 (LangGraph 版本)"""

    def __init__(self):
        """初始化工作流"""
        logger.info("🔄 初始化 LangGraph 旅行规划工作流...")

        try:
            # 初始化工具
            self.tools = get_cached_amap_tools()
            if not self.tools:
                logger.warning("⚠️  未加载到任何工具，工作流可能无法正常工作")
            else:
                logger.info(f"✅ 加载了 {len(self.tools)} 个工具")
                for tool in self.tools:
                    logger.debug(f"  工具: {tool.name} - {tool.description}")

            # 按工具名过滤，每个 Agent 只拿需要的工具
            search_tools = [t for t in self.tools if "text_search" in t.name.lower()]
            weather_tools = [t for t in self.tools if "weather" in t.name.lower()]

            logger.info(f"搜索工具: {[t.name for t in search_tools]}")
            logger.info(f"天气工具: {[t.name for t in weather_tools]}")

            # 创建智能体（统一使用 get_agent）
            logger.info("创建智能体...")
            self.attraction_agent = get_agent("attraction_search", search_tools if search_tools else self.tools)
            self.weather_agent = get_agent("weather", weather_tools if weather_tools else self.tools)
            self.hotel_agent = get_agent("hotel", search_tools if search_tools else self.tools)
            self.planner_agent = get_agent("planner", [])

            # 构建工作流图
            logger.info("构建 StateGraph...")
            self.graph = self._build_graph()

            logger.info("✅ LangGraph 工作流初始化成功")

        except Exception as e:
            logger.error(f"❌ 工作流初始化失败: {str(e)}", exc_info=True)
            raise

    def _prepare_agent_input(self, user_input: str, chat_history: list) -> dict:
        """准备智能体输入格式，将 input 和 chat_history 转换为 messages 格式"""
        messages = []
        # 添加历史消息（如果存在）
        for msg in chat_history:
            # 假设历史消息格式为 {"role": "...", "content": "..."}
            messages.append(msg)
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        return {"messages": messages}

    def _to_json_str(self, obj: Any) -> str:
        if obj is None:
            return ""
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump()
        return json.dumps(obj, ensure_ascii=False)

    def _extract_structured_payload(self, result: dict) -> Any:
        # 1) LangChain 官方 structured output
        payload = result.get("structured_response")
        if payload is not None:
            return payload

        # 2) MCP ToolMessage artifact
        for msg in reversed(result.get("messages", [])):
            # LangChain message object
            artifact = getattr(msg, "artifact", None)
            if isinstance(artifact, dict):
                payload = artifact.get("structured_content") or artifact.get("structuredContent")
                if payload is not None:
                    return payload

            # dict message fallback
            if isinstance(msg, dict):
                artifact = msg.get("artifact")
                if isinstance(artifact, dict):
                    payload = artifact.get("structured_content") or artifact.get("structuredContent")
                    if payload is not None:
                        return payload

        return None

    def _extract_agent_output(self, result: dict) -> str:
        structured = self._extract_structured_payload(result)
        if structured is not None:
            return self._to_json_str(structured)

        if "messages" in result:
            messages = result["messages"]
            for msg in reversed(messages):
                if isinstance(msg, dict):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, dict):
                            return json.dumps(content, ensure_ascii=False)
                        return str(content)
                else:
                    msg_type = getattr(msg, "type", getattr(msg, "role", None))
                    if msg_type in ["assistant", "ai"]:
                        content = getattr(msg, "content", "")
                        if isinstance(content, dict):
                            return json.dumps(content, ensure_ascii=False)
                        if content:
                            return str(content)

        for key in ["output", "text", "response", "content"]:
            if key in result:
                return str(result[key])

        return str(result)

    def _extract_structured_response(self, result: dict):
        """提取 structured_response"""
        if isinstance(result, dict) and "structured_response" in result:
            return result["structured_response"]
        return None

    def _build_graph(self) -> StateGraph:
        """构建 StateGraph"""
        workflow = StateGraph(TripPlannerState)
        # 添加节点
        workflow.add_node("search_attractions", self._search_attractions)
        workflow.add_node("check_weather", self._check_weather)
        workflow.add_node("find_hotels", self._find_hotels)
        workflow.add_node("plan_itinerary", self._plan_itinerary)
        workflow.add_node("handle_error", self._handle_error)
        # 设置入口点
        workflow.set_entry_point("search_attractions")
        # 添加边（正常流程）
        # workflow.add_edge("search_attractions", "check_weather")
        # workflow.add_edge("check_weather", "find_hotels")
        # workflow.add_edge("find_hotels", "plan_itinerary")
        # workflow.add_edge("plan_itinerary", END)
        # 添加错误处理边
        workflow.add_conditional_edges(
            "search_attractions",
            self._check_error,
            {
                "continue": "check_weather",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "check_weather",
            self._check_error,
            {
                "continue": "find_hotels",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "find_hotels",
            self._check_error,
            {
                "continue": "plan_itinerary",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "plan_itinerary",
            self._check_error,
            {
                "continue": END,
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()

    def _search_attractions(self, state: TripPlannerState) -> Dict[str, Any]:
        logger.info("📍 搜索景点...")
        try:
            query = self._build_attraction_query(state["request"])
            logger.info(f"查询内容: {query}")

            result = self.attraction_agent.invoke(
                self._prepare_agent_input(query, []),
                config={"recursion_limit": 25}
            )

            structured = self._extract_structured_response(result)

            if structured is not None:
                attractions = structured.attractions
                logger.info(f"structured_response 返回 {len(attractions)} 个景点")
            else:
                output = self._extract_agent_output(result)
                logger.info(f"回退到文本解析，Agent 输出前300字符: {output[:300]}")
                attractions = self._parse_attractions(output)

            logger.info(f"最终保留 {len(attractions)} 个有效景点")

            if len(attractions) == 0:
                logger.warning("未解析到任何有效景点，后续将使用备用数据或降级结果")

            return {
                "attractions": attractions,
                "messages": [{"role": "assistant", "content": f"已找到 {len(attractions)} 个景点"}]
            }
        except Exception as e:
            logger.error(f"景点搜索失败: {str(e)}", exc_info=True)
            return {
                "error": f"景点搜索失败: {str(e)}",
                "current_step": "error"
        }

    def _check_weather(self, state: TripPlannerState) -> Dict[str, Any]:
        logger.info("🌤️  查询天气...")
        try:
            query = f"查询{state['request'].city}的天气信息"
            logger.info(f"查询内容: {query}")

            result = self.weather_agent.invoke(
                self._prepare_agent_input(query, []),
                config={"recursion_limit": 25}
            )

            structured = self._extract_structured_response(result)

            if structured is not None:
                weather_info = structured.weather_info
                logger.info(f"structured_response 返回 {len(weather_info)} 条天气")
            else:
                output = self._extract_agent_output(result)
                logger.info(f"回退到文本解析，Agent 输出前300字符: {output[:300]}")
                weather_info = self._parse_weather(output)

            logger.info(f"最终保留 {len(weather_info)} 条有效天气信息")

            if len(weather_info) == 0:
                logger.warning("未解析到任何有效天气信息")

            return {
                "weather_info": weather_info,
                "messages": [{"role": "assistant", "content": f"已获取 {len(weather_info)} 天天气信息"}]
            }
        except Exception as e:
            logger.error(f"天气查询失败: {str(e)}", exc_info=True)
            return {
                "error": f"天气查询失败: {str(e)}",
                "current_step": "error"
            }

    def _find_hotels(self, state: TripPlannerState) -> Dict[str, Any]:
        logger.info("🏨 搜索酒店...")
        try:
            query = f"搜索{state['request'].city}的{state['request'].accommodation}酒店"
            logger.info(f"查询内容: {query}")

            result = self.hotel_agent.invoke(
                self._prepare_agent_input(query, []),
                config={"recursion_limit": 25}
            )

            structured = self._extract_structured_response(result)

            if structured is not None:
                hotels = structured.hotels
                logger.info(f"structured_response 返回 {len(hotels)} 个酒店")
            else:
                output = self._extract_agent_output(result)
                logger.info(f"回退到文本解析，Agent 输出前300字符: {output[:300]}")
                hotels = self._parse_hotels(output)

            logger.info(f"最终保留 {len(hotels)} 个有效酒店")

            if len(hotels) == 0:
                logger.warning("未解析到任何有效酒店信息")

            return {
                "hotels": hotels,
                "current_step": "hotels_found",
                "messages": [{"role": "assistant", "content": f"已找到 {len(hotels)} 个酒店"}]
            }
        except Exception as e:
            logger.error(f"酒店搜索失败: {str(e)}", exc_info=True)
            return {
                "error": f"酒店搜索失败: {str(e)}",
                "current_step": "error"
            }

    def _plan_itinerary(self, state: TripPlannerState) -> Dict[str, Any]:
        logger.info("📋 生成行程计划...")
        try:
            query = self._build_planner_query(
                state["request"],
                state["attractions"],
                state["weather_info"],
                state["hotels"]
            )
            logger.info(f"传给 planner 的景点数: {len(state['attractions'])}, 天气数: {len(state['weather_info'])}, 酒店数: {len(state['hotels'])}")

            result = self.planner_agent.invoke(
                self._prepare_agent_input(query, []),
                config={"recursion_limit": 25}
            )

            structured = self._extract_structured_response(result)

            if structured is not None:
                trip_plan = structured.trip_plan
                logger.info(f"structured_response 返回 {len(trip_plan.days)} 天行程")
            else:
                output = self._extract_agent_output(result)
                logger.info(f"回退到文本解析，Planner 输出前300字符: {output[:300]}")
                trip_plan = self._parse_trip_plan(output, state["request"])

            logger.info(f"解析到 {len(trip_plan.days)} 天行程")

            return {
                "trip_plan": trip_plan,
                "current_step": "plan_completed",
                "messages": [{"role": "assistant", "content": "行程计划生成完成！"}]
            }
        except Exception as e:
            logger.error(f"行程规划失败: {str(e)}", exc_info=True)
            return {
                "error": f"行程规划失败: {str(e)}",
                "current_step": "error"
            }

    def _handle_error(self, state: TripPlannerState) -> Dict[str, Any]:
        error_msg = state.get('error', '未知错误')
        logger.warning(f"⚠️  处理错误: {error_msg}")
        fallback_plan = self._create_fallback_plan(state["request"])
        return {
            "trip_plan": fallback_plan,
            "error": None,
            "current_step": "error_handled",
            "messages": [{"role": "assistant", "content": f"遇到错误，已生成备用计划: {error_msg}"}]
        }

    def _check_error(self, state: TripPlannerState) -> str:
        """检查是否有错误"""
        return "error" if state.get("error") else "continue"

    # ============ 辅助方法（从原 trip_planner_agent.py 迁移）============

    def _build_attraction_query(self, request: TripRequest) -> str:
        if request.preferences:
            keywords = "、".join(request.preferences)
        else:
            keywords = "景点"
        return f"搜索{request.city}的{keywords}相关景点，返回6-8个结果"

    def _build_planner_query(self, request: TripRequest, attractions: List[Attraction],
                        weather: List[WeatherInfo], hotels: List[Hotel]) -> str:
        """构建行程规划查询"""
        # 把景点信息序列化为完整 JSON
        attractions_data = []
        for a in attractions:
            attractions_data.append({
            "name": a.name,
            "address": a.address,
            "location": {"longitude": a.location.longitude, "latitude": a.location.latitude} if a.location else None,
            "visit_duration": a.visit_duration,
            "description": a.description,
            "category": a.category,
            "ticket_price": a.ticket_price if hasattr(a, "ticket_price") else 0,
            "price_text": a.price_text if hasattr(a, "price_text") else ""
        })

        # 把天气信息序列化
        weather_data = []
        for w in weather:
            weather_data.append({
                "date": w.date,
                "day_weather": w.day_weather,
                "night_weather": w.night_weather,
                "day_temp": w.day_temp,
                "night_temp": w.night_temp,
                "wind_direction": w.wind_direction,
                "wind_power": w.wind_power
            })

        # 把酒店信息序列化
        hotels_data = []
        for h in hotels:
            hotels_data.append({
                "name": h.name,
                "address": h.address,
                "location": {"longitude": h.location.longitude, "latitude": h.location.latitude} if h.location else None,
                "price_range": h.price_range if hasattr(h, 'price_range') else "",
                "rating": h.rating if hasattr(h, 'rating') else "",
                "type": h.type if hasattr(h, 'type') else "",
                "estimated_cost": h.estimated_cost if hasattr(h, 'estimated_cost') else 0
            })

        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

**景点信息（共{len(attractions)}个）:**
{json.dumps(attractions_data, ensure_ascii=False, indent=2)}

**天气信息:**
{json.dumps(weather_data, ensure_ascii=False, indent=2)}

**酒店信息（共{len(hotels)}个）:**
{json.dumps(hotels_data, ensure_ascii=False, indent=2)}

**要求:**
1. 每天安排2-3个景点（从上面的景点中选择）
2. 每天必须包含早中晚三餐
3. 每天推荐一个酒店（从上面的酒店中选择）
4. 考虑景点之间的距离和交通方式
5. 景点的经纬度坐标使用上面提供的真实数据
6. 必须包含预算汇总
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"

        return query

    def _extract_json(self, response: str) -> str:
        """从响应文本中提取JSON字符串"""
        # 查找JSON代码块
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        elif "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        elif "[" in response and "]" in response:
            # 处理JSON数组
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            json_str = response[json_start:json_end]
        elif "{" in response and "}" in response:
            # 直接查找JSON对象
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]
        else:
            # 如果没有找到JSON，返回原始响应
            json_str = response.strip()
        return json_str
    
    def _parse_location(self, raw_location: Any) -> Optional[Location]:
        """安全解析经纬度，缺失或非法时返回 None"""
        if not raw_location or not isinstance(raw_location, dict):
            return None

        longitude = raw_location.get("longitude")
        latitude = raw_location.get("latitude")

        if longitude in (None, "") or latitude in (None, ""):
            return None

        try:
            return Location(
                longitude=float(longitude),
                latitude=float(latitude)
            )
        except (TypeError, ValueError):
            return None

    def _parse_attractions(self, response: str) -> List[Attraction]:
        """解析景点信息（逐条容错）"""
        json_str = ""

        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
        except Exception as e:
            logger.error(f"解析景点 JSON 失败: {str(e)}")
            logger.error(f"原始响应长度: {len(response)}")
            logger.error(f"原始响应前500字符: {response[:500]}")
            if json_str:
                logger.error(f"提取的JSON字符串前500字符: {json_str[:500]}")
            return []

        # 兼容两种格式：
        # 1. 顶层直接是 list
        # 2. 顶层是 {"attractions": [...]}
        if isinstance(data, dict):
            data = data.get("attractions", [])

        if not isinstance(data, list):
            logger.warning(f"景点数据不是列表，实际类型: {type(data).__name__}")
            return []

        attractions: List[Attraction] = []
        skipped_count = 0

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                skipped_count += 1
                logger.warning(f"跳过第 {idx+1} 条景点：数据不是对象，实际类型={type(item).__name__}")
                continue

            try:
                raw_ticket_price = item.get("ticket_price", 0)
                price_text = item.get("price_text", "")

                # 如果 ticket_price 原始值是字符串，且 price_text 为空，则保留原始价格文本
                if isinstance(raw_ticket_price, str) and not price_text:
                    price_text = raw_ticket_price

                attraction = Attraction(
                    name=item.get("name", ""),
                    address=item.get("address", ""),
                    location=self._parse_location(item.get("location")),
                    visit_duration=item.get("visit_duration", 120),
                    description=item.get("description", ""),
                    category=item.get("category", "景点"),
                    rating=item.get("rating"),
                    photos=item.get("photos", []) or [],
                    poi_id=item.get("poi_id"),
                    image_url=item.get("image_url"),
                    ticket_price=raw_ticket_price,
                    price_text=price_text,
                )

                if not attraction.name.strip():
                    skipped_count += 1
                    logger.warning(f"跳过第 {idx+1} 条景点：name 为空，item={item}")
                    continue

                attractions.append(attraction)

            except Exception as e:
                skipped_count += 1
                logger.warning(
                    f"跳过第 {idx+1} 条景点解析失败: {str(e)}; "
                    f"name={item.get('name', '未知')}; item={item}"
                )
                continue

        logger.info(f"景点解析完成：成功 {len(attractions)} 条，跳过 {skipped_count} 条")
        return attractions

    def _parse_weather(self, response: str) -> List[WeatherInfo]:
        """解析天气信息（逐条容错）"""
        json_str = ""

        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
        except Exception as e:
            logger.error(f"解析天气 JSON 失败: {str(e)}")
            logger.error(f"原始响应长度: {len(response)}")
            logger.error(f"原始响应前500字符: {response[:500]}")
            if json_str:
                logger.error(f"提取的JSON字符串前500字符: {json_str[:500]}")
            return []

        # 兼容两种格式：
        # 1. 顶层直接是 list
        # 2. 顶层是 {"weather_info": [...]}
        if isinstance(data, dict):
            data = data.get("weather_info", [])

        if not isinstance(data, list):
            logger.warning(f"天气数据不是列表，实际类型: {type(data).__name__}")
            return []

        weather_info: List[WeatherInfo] = []
        skipped_count = 0

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                skipped_count += 1
                logger.warning(f"跳过第 {idx+1} 条天气：数据不是对象，实际类型={type(item).__name__}")
                continue

            try:
                weather = WeatherInfo(
                    date=item.get("date", ""),
                    day_weather=item.get("day_weather", ""),
                    night_weather=item.get("night_weather", ""),
                    day_temp=item.get("day_temp", 0),
                    night_temp=item.get("night_temp", 0),
                    wind_direction=item.get("wind_direction", ""),
                    wind_power=item.get("wind_power", "")
                )

                if not weather.date.strip():
                    skipped_count += 1
                    logger.warning(f"跳过第 {idx+1} 条天气：date 为空，item={item}")
                    continue

                weather_info.append(weather)

            except Exception as e:
                skipped_count += 1
                logger.warning(
                    f"跳过第 {idx+1} 条天气解析失败: {str(e)}; "
                    f"date={item.get('date', '未知')}; item={item}"
                )
                continue

        logger.info(f"天气解析完成：成功 {len(weather_info)} 条，跳过 {skipped_count} 条")
        return weather_info

    def _parse_hotels(self, response: str) -> List[Hotel]:
        """解析酒店信息（逐条容错）"""
        json_str = ""

        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
        except Exception as e:
            logger.error(f"解析酒店 JSON 失败: {str(e)}")
            logger.error(f"原始响应长度: {len(response)}")
            logger.error(f"原始响应前500字符: {response[:500]}")
            if json_str:
                logger.error(f"提取的JSON字符串前500字符: {json_str[:500]}")
            return []

        # 兼容两种格式：
        # 1. 顶层直接是 list
        # 2. 顶层是 {"hotels": [...]}
        if isinstance(data, dict):
            data = data.get("hotels", [])

        if not isinstance(data, list):
            logger.warning(f"酒店数据不是列表，实际类型: {type(data).__name__}")
            return []

        hotels: List[Hotel] = []
        skipped_count = 0

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                skipped_count += 1
                logger.warning(f"跳过第 {idx+1} 条酒店：数据不是对象，实际类型={type(item).__name__}")
                continue

            try:
                hotel = Hotel(
                    name=item.get("name", ""),
                    address=item.get("address", ""),
                    location=self._parse_location(item.get("location")),
                    price_range=item.get("price_range", ""),
                    rating=item.get("rating"),
                    distance=item.get("distance", ""),
                    type=item.get("type", ""),
                    estimated_cost=item.get("estimated_cost", 0)
                )

                if not hotel.name.strip():
                    skipped_count += 1
                    logger.warning(f"跳过第 {idx+1} 条酒店：name 为空，item={item}")
                    continue

                hotels.append(hotel)

            except Exception as e:
                skipped_count += 1
                logger.warning(
                    f"跳过第 {idx+1} 条酒店解析失败: {str(e)}; "
                    f"name={item.get('name', '未知')}; item={item}"
                )
                continue

        logger.info(f"酒店解析完成：成功 {len(hotels)} 条，跳过 {skipped_count} 条")
        return hotels

    def _parse_trip_plan(self, response: str, request: TripRequest) -> TripPlan:
        """解析行程计划（逐段/逐条容错）"""
        json_str = ""

        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
        except Exception as e:
            logger.error(f"解析行程计划 JSON 失败: {str(e)}")
            logger.error(f"原始响应长度: {len(response)}")
            logger.error(f"原始响应前500字符: {response[:500]}")
            if json_str:
                logger.error(f"提取的JSON字符串前500字符: {json_str[:500]}")
            return self._create_fallback_plan(request)

        # 兼容两种格式：
        # 1. 顶层直接就是 TripPlan
        # 2. 顶层是 {"trip_plan": {...}}
        if isinstance(data, dict) and isinstance(data.get("trip_plan"), dict):
            data = data["trip_plan"]

        if not isinstance(data, dict):
            logger.error(f"行程计划数据不是对象，实际类型: {type(data).__name__}")
            return self._create_fallback_plan(request)

        trip_plan = TripPlan(
            city=data.get("city", request.city),
            start_date=data.get("start_date", request.start_date),
            end_date=data.get("end_date", request.end_date),
            days=[],
            weather_info=[],
            overall_suggestions=data.get("overall_suggestions", ""),
            budget=None
        )

        # 1) 解析天气信息（逐条容错）
        weather_list = data.get("weather_info", [])
        if isinstance(weather_list, list):
            skipped_weather = 0
            for idx, weather_data in enumerate(weather_list):
                if not isinstance(weather_data, dict):
                    skipped_weather += 1
                    logger.warning(f"跳过第 {idx+1} 条天气：数据不是对象")
                    continue

                try:
                    weather_info = WeatherInfo(
                        date=weather_data.get("date", ""),
                        day_weather=weather_data.get("day_weather", ""),
                        night_weather=weather_data.get("night_weather", ""),
                        day_temp=weather_data.get("day_temp", 0),
                        night_temp=weather_data.get("night_temp", 0),
                        wind_direction=weather_data.get("wind_direction", ""),
                        wind_power=weather_data.get("wind_power", "")
                    )

                    if not weather_info.date.strip():
                        skipped_weather += 1
                        logger.warning(f"跳过第 {idx+1} 条天气：date 为空，item={weather_data}")
                        continue

                    trip_plan.weather_info.append(weather_info)

                except Exception as e:
                    skipped_weather += 1
                    logger.warning(
                        f"跳过第 {idx+1} 条天气解析失败: {str(e)}; item={weather_data}"
                    )

            logger.info(f"行程天气解析完成：成功 {len(trip_plan.weather_info)} 条，跳过 {skipped_weather} 条")

        # 2) 解析每日行程（逐天容错）
        days_list = data.get("days", [])
        if isinstance(days_list, list):
            skipped_days = 0

            for day_idx, day_data in enumerate(days_list):
                if not isinstance(day_data, dict):
                    skipped_days += 1
                    logger.warning(f"跳过第 {day_idx+1} 天：数据不是对象")
                    continue

                try:
                    # 2.1 解析景点（逐条容错）
                    attractions = []
                    skipped_attractions = 0
                    attraction_list = day_data.get("attractions", [])

                    if isinstance(attraction_list, list):
                        for attr_idx, attr_data in enumerate(attraction_list):
                            if not isinstance(attr_data, dict):
                                skipped_attractions += 1
                                logger.warning(
                                    f"第 {day_idx+1} 天跳过第 {attr_idx+1} 个景点：数据不是对象"
                                )
                                continue

                            try:
                                raw_ticket_price = attr_data.get("ticket_price", 0)
                                price_text = attr_data.get("price_text", "")

                                if isinstance(raw_ticket_price, str) and not price_text:
                                    price_text = raw_ticket_price

                                attraction = Attraction(
                                    name=attr_data.get("name", ""),
                                    address=attr_data.get("address", ""),
                                    location=self._parse_location(attr_data.get("location")),
                                    visit_duration=attr_data.get("visit_duration", 0),
                                    description=attr_data.get("description", ""),
                                    category=attr_data.get("category", "景点"),
                                    rating=attr_data.get("rating"),
                                    photos=attr_data.get("photos", []) or [],
                                    poi_id=attr_data.get("poi_id"),
                                    image_url=attr_data.get("image_url"),
                                    ticket_price=raw_ticket_price,
                                    price_text=price_text,
                                )

                                if not attraction.name.strip():
                                    skipped_attractions += 1
                                    logger.warning(
                                        f"第 {day_idx+1} 天跳过第 {attr_idx+1} 个景点：name 为空，item={attr_data}"
                                    )
                                    continue

                                attractions.append(attraction)

                            except Exception as e:
                                skipped_attractions += 1
                                logger.warning(
                                    f"第 {day_idx+1} 天跳过第 {attr_idx+1} 个景点解析失败: {str(e)}; item={attr_data}"
                                )

                    # 2.2 解析餐饮（逐条容错）
                    meals = []
                    skipped_meals = 0
                    meal_list = day_data.get("meals", [])

                    if isinstance(meal_list, list):
                        for meal_idx, meal_data in enumerate(meal_list):
                            if not isinstance(meal_data, dict):
                                skipped_meals += 1
                                logger.warning(
                                    f"第 {day_idx+1} 天跳过第 {meal_idx+1} 个餐饮：数据不是对象"
                                )
                                continue

                            try:
                                meal = Meal(**meal_data)

                                # 如果你的 Meal.name 是必填，这里可以防空
                                if hasattr(meal, "name") and not getattr(meal, "name", "").strip():
                                    skipped_meals += 1
                                    logger.warning(
                                        f"第 {day_idx+1} 天跳过第 {meal_idx+1} 个餐饮：name 为空，item={meal_data}"
                                    )
                                    continue

                                meals.append(meal)

                            except Exception as e:
                                skipped_meals += 1
                                logger.warning(
                                    f"第 {day_idx+1} 天跳过第 {meal_idx+1} 个餐饮解析失败: {str(e)}; item={meal_data}"
                                )

                    # 2.3 解析酒店（单独容错）
                    hotel = None
                    hotel_data = day_data.get("hotel")
                    if isinstance(hotel_data, dict):
                        try:
                            hotel = Hotel(
                                name=hotel_data.get("name", ""),
                                address=hotel_data.get("address", ""),
                                location=self._parse_location(hotel_data.get("location")),
                                price_range=hotel_data.get("price_range", ""),
                                rating=hotel_data.get("rating"),
                                distance=hotel_data.get("distance", ""),
                                type=hotel_data.get("type", ""),
                                estimated_cost=hotel_data.get("estimated_cost", 0)
                            )

                            if not hotel.name.strip():
                                logger.warning(f"第 {day_idx+1} 天酒店 name 为空，已忽略: {hotel_data}")
                                hotel = None

                        except Exception as e:
                            logger.warning(f"第 {day_idx+1} 天酒店解析失败: {str(e)}; item={hotel_data}")
                            hotel = None

                    # 2.4 组装 DayPlan（单天失败不影响整份计划）
                    day_plan = DayPlan(
                        date=day_data.get("date", ""),
                        day_index=day_data.get("day_index", day_idx),
                        description=day_data.get("description", ""),
                        transportation=day_data.get("transportation", request.transportation),
                        accommodation=day_data.get("accommodation", request.accommodation),
                        hotel=hotel,
                        attractions=attractions,
                        meals=meals
                    )

                    trip_plan.days.append(day_plan)
                    logger.info(
                        f"第 {day_idx+1} 天解析完成：景点 {len(attractions)} 个，餐饮 {len(meals)} 个"
                    )

                except Exception as e:
                    skipped_days += 1
                    logger.warning(f"跳过第 {day_idx+1} 天行程解析失败: {str(e)}; item={day_data}")
                    continue

            logger.info(f"行程天数解析完成：成功 {len(trip_plan.days)} 天，跳过 {skipped_days} 天")

        # 3) 解析预算（单独容错）
        budget_data = data.get("budget")
        if isinstance(budget_data, dict):
            try:
                trip_plan.budget = Budget(**budget_data)
            except Exception as e:
                logger.warning(f"预算解析失败，已忽略: {str(e)}; item={budget_data}")

        # 4) 最低保底：如果一天都没解析出来，再 fallback
        if len(trip_plan.days) == 0:
            logger.error("未解析到任何有效行程天数，将返回备用计划")
            return self._create_fallback_plan(request)

        return trip_plan

    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划(当Agent失败时)"""
        # 解析日期
        try:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        except ValueError:
            # 如果日期格式错误，使用当前日期
            start_date = datetime.now()

        # 创建每日行程
        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)

            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}",
                        address=f"{request.city}市",
                        location=Location(longitude=120.1551 + i*0.01 + j*0.005, latitude=30.2741 + i*0.01 + j*0.005),
                        visit_duration=120,
                        description=f"这是{request.city}的著名景点",
                        category="景点"
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐")
                ]
            )
            days.append(day_plan)

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程,建议提前查看各景点的开放时间。"
        )

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """执行旅行规划工作流"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 开始 LangGraph 旅行规划工作流...")
        logger.info(f"目的地: {request.city}")
        logger.info(f"{'='*60}\n")

        # 初始化状态
        initial_state: TripPlannerState = create_initial_state(request)

        # 执行工作流
        final_state = self.graph.invoke(
            initial_state,
            config={"recursion_limit": 50}
        )

        # 检查结果
        if final_state.get("error") and not final_state.get("trip_plan"):
            error_msg = final_state.get("error", "未知错误")
            logger.error(f"❌ 旅行规划失败: {error_msg}")
            raise Exception(error_msg)

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ 旅行计划生成完成!")
        logger.info(f"{'='*60}\n")

        return final_state["trip_plan"]

# 全局工作流实例
_trip_planner_workflow: Optional[TripPlannerWorkflow] = None

def get_trip_planner_workflow() -> TripPlannerWorkflow:
    """获取旅行规划工作流实例（单例模式）"""
    global _trip_planner_workflow

    if _trip_planner_workflow is None:
        _trip_planner_workflow = TripPlannerWorkflow()

    return _trip_planner_workflow

def reset_workflow():
    """重置工作流实例（用于测试或重新配置）"""
    global _trip_planner_workflow
    _trip_planner_workflow = None
    logger.info("工作流实例已重置")