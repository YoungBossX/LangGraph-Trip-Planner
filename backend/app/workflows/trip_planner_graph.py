"""旅行规划 LangGraph 工作流"""

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

# 设置日志记录
logger = logging.getLogger(__name__)

class TripPlannerWorkflow:
    """多智能体旅行规划工作流 (LangGraph 版本)

    架构说明:
    - search_attractions 节点: 代码多关键词搜索 → Agent 智能筛选/描述 → 代码地理编码补坐标
    - check_weather 节点: Agent 调用天气工具 → 解析天气数据
    - find_hotels 节点: 代码搜索 → Agent 智能筛选/描述 → 代码地理编码补坐标
    - plan_itinerary 节点: Agent 综合所有数据生成行程计划
    """

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

            # 按工具名过滤，每个 Agent 只拿需要的工具
            search_tools = [t for t in self.tools if t.name == "maps_text_search"]
            weather_tools = [t for t in self.tools if t.name == "maps_weather"]

            logger.info(f"搜索工具: {[t.name for t in search_tools]}")
            logger.info(f"天气工具: {[t.name for t in weather_tools]}")

            # 创建智能体
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

    # ========== StateGraph 构建 ==========

    def _build_graph(self) -> StateGraph:
        """构建 StateGraph"""
        workflow = StateGraph(TripPlannerState)
        workflow.add_node("search_attractions", self._search_attractions)
        workflow.add_node("check_weather", self._check_weather)
        workflow.add_node("find_hotels", self._find_hotels)
        workflow.add_node("plan_itinerary", self._plan_itinerary)
        workflow.add_node("handle_error", self._handle_error)

        workflow.set_entry_point("search_attractions")

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
        workflow.add_edge("handle_error", END)
        return workflow.compile()

    def _check_error(self, state: TripPlannerState) -> str:
        return "error" if state.get("error") else "continue"

    # ========== 节点: 景点搜索 ==========

    def _search_attractions(self, state: TripPlannerState) -> Dict[str, Any]:
        """景点搜索节点: 代码搜索 → Agent 筛选描述 → 代码补坐标"""
        logger.info("📍 搜索景点...")
        try:
            request = state["request"]
            search_tool = next((t for t in self.tools if t.name == "maps_text_search"), None)

            if not search_tool:
                return {"error": "搜索工具不可用", "current_step": "error"}

            # ---- 第1步: 代码多关键词搜索 ----
            keywords_list = [f"{request.city}旅游景点"]
            pref_map = {
                "历史文化": "历史古迹", "自然风光": "公园风景区",
                "美食": "特色美食", "购物": "商业街",
                "艺术": "博物馆美术馆", "休闲": "休闲公园",
            }
            if request.preferences:
                for pref in request.preferences:
                    keywords_list.append(f"{request.city}{pref_map.get(pref, pref)}")
            keywords_list = list(dict.fromkeys(keywords_list))

            all_pois = []
            seen_ids = set()
            for kw in keywords_list[:7]:
                try:
                    logger.info(f"搜索关键词: {kw}")
                    result = search_tool.invoke({"keywords": kw, "city": request.city, "citylimit": "true"})
                    pois = self._extract_pois_from_tool_result(result)
                    for poi in pois:
                        poi_id = poi.get("id", "")
                        if poi_id and poi_id not in seen_ids:
                            seen_ids.add(poi_id)
                            all_pois.append(poi)
                except Exception as e:
                    logger.warning(f"搜索 '{kw}' 失败: {str(e)}")

            logger.info(f"共搜索到 {len(all_pois)} 个不重复的 POI")

            # ---- 第2步: Agent 智能筛选和描述 ----
            selected_pois = all_pois[:12]  # 给 Agent 最多 12 个候选
            if selected_pois:
                poi_summary = json.dumps(
                    [{"name": p.get("name", ""), "address": p.get("address", ""), "typecode": p.get("typecode", "")}
                     for p in selected_pois],
                    ensure_ascii=False
                )
                enrich_query = (
                    f"以下是在{request.city}搜索到的 POI 列表:\n{poi_summary}\n\n"
                    f"用户偏好: {', '.join(request.preferences) if request.preferences else '综合'}\n"
                    f"请从中选出最适合旅游的 8 个地点，为每个地点补充简短描述和分类。\n"
                    f"直接输出 JSON 数组，格式: "
                    f'[{{"name":"名称","description":"一句话描述","category":"分类","visit_duration":120}}]'
                )
                try:
                    agent_result = self.attraction_agent.invoke(
                        self._prepare_agent_input(enrich_query, []),
                        config={"recursion_limit": 15}
                    )
                    agent_output = self._extract_agent_output(agent_result)
                    logger.info(f"Agent 筛选输出前300字符: {agent_output[:300]}")
                    enriched = self._parse_agent_enrichment(agent_output)
                except Exception as e:
                    logger.warning(f"Agent 筛选失败，使用原始数据: {str(e)}")
                    enriched = {}
            else:
                enriched = {}

            # ---- 第3步: 组装 Attraction 对象 ----
            attractions = []
            # 优先取 Agent 筛选出的名称顺序
            enriched_names = set(enriched.keys())
            ordered_pois = []
            for p in selected_pois:
                if p.get("name", "") in enriched_names:
                    ordered_pois.append(p)
            # 补充 Agent 没选到的
            for p in selected_pois:
                if p.get("name", "") not in enriched_names and len(ordered_pois) < 8:
                    ordered_pois.append(p)

            for poi in ordered_pois[:8]:
                name = poi.get("name", "")
                info = enriched.get(name, {})
                attractions.append(Attraction(
                    name=name,
                    address=poi.get("address", ""),
                    poi_id=poi.get("id"),
                    location=None,
                    visit_duration=info.get("visit_duration", 120),
                    description=info.get("description", ""),
                    category=info.get("category", poi.get("typecode", "景点")),
                    ticket_price=0
                ))

            # ---- 第4步: 代码批量补坐标 ----
            logger.info(f"解析到 {len(attractions)} 个景点，开始补充坐标...")
            attractions = self._fill_locations(attractions, request.city)
            logger.info(f"最终保留 {len(attractions)} 个有效景点")

            if not attractions:
                logger.warning("未解析到任何有效景点")

            return {
                "attractions": attractions,
                "messages": [{"role": "assistant", "content": f"已找到 {len(attractions)} 个景点"}]
            }
        except Exception as e:
            logger.error(f"景点搜索失败: {str(e)}", exc_info=True)
            return {"error": f"景点搜索失败: {str(e)}", "current_step": "error"}

    # ========== 节点: 天气查询 ==========

    def _check_weather(self, state: TripPlannerState) -> Dict[str, Any]:
        """天气查询节点: Agent 调用天气工具"""
        logger.info("🌤️  查询天气...")
        try:
            request = state["request"]
            query = f"查询{request.city}的天气信息，旅行日期为{request.start_date}至{request.end_date}"
            logger.info(f"查询内容: {query}")

            result = self.weather_agent.invoke(
                self._prepare_agent_input(query, []),
                config={"recursion_limit": 25}
            )

            output = self._extract_agent_output(result)
            logger.info(f"Agent 输出前300字符: {output[:300]}")

            weather_info = self._parse_weather(output)
            logger.info(f"最终保留 {len(weather_info)} 条有效天气信息")

            # 如果 Agent 解析失败，尝试代码直接调工具
            if not weather_info:
                logger.warning("Agent 天气解析为空，尝试代码直接获取...")
                weather_info = self._fetch_weather_by_code(request.city)
                logger.info(f"代码直接获取到 {len(weather_info)} 条天气信息")

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

    # ========== 节点: 酒店搜索 ==========

    def _find_hotels(self, state: TripPlannerState) -> Dict[str, Any]:
        """酒店搜索节点: 代码搜索 → Agent 筛选描述 → 代码补坐标"""
        logger.info("🏨 搜索酒店...")
        try:
            request = state["request"]
            search_tool = next((t for t in self.tools if t.name == "maps_text_search"), None)

            if not search_tool:
                return {"error": "搜索工具不可用", "current_step": "error"}

            # ---- 第1步: 代码搜索酒店 ----
            keyword = f"{request.city}{request.accommodation}"
            logger.info(f"搜索关键词: {keyword}")
            result = search_tool.invoke({"keywords": keyword, "city": request.city, "citylimit": "true"})
            pois = self._extract_pois_from_tool_result(result)
            logger.info(f"搜索到 {len(pois)} 个酒店 POI")

            # ---- 第2步: Agent 筛选和描述 ----
            selected_pois = pois[:8]
            if selected_pois:
                poi_summary = json.dumps(
                    [{"name": p.get("name", ""), "address": p.get("address", "")} for p in selected_pois],
                    ensure_ascii=False
                )
                enrich_query = (
                    f"以下是在{request.city}搜索到的酒店列表:\n{poi_summary}\n\n"
                    f"用户住宿偏好: {request.accommodation}\n"
                    f"请从中选出最适合的 5 个酒店，为每个补充预估价格和类型。\n"
                    f"直接输出 JSON 数组，格式: "
                    f'[{{"name":"酒店名","price_range":"200-400元","type":"经济型酒店","estimated_cost":300}}]'
                )
                try:
                    agent_result = self.hotel_agent.invoke(
                        self._prepare_agent_input(enrich_query, []),
                        config={"recursion_limit": 15}
                    )
                    agent_output = self._extract_agent_output(agent_result)
                    logger.info(f"Agent 酒店筛选输出前300字符: {agent_output[:300]}")
                    enriched = self._parse_hotel_enrichment(agent_output)
                except Exception as e:
                    logger.warning(f"Agent 酒店筛选失败，使用原始数据: {str(e)}")
                    enriched = {}
            else:
                enriched = {}

            # ---- 第3步: 组装 Hotel 对象 ----
            hotels = []
            enriched_names = set(enriched.keys())
            ordered_pois = [p for p in selected_pois if p.get("name", "") in enriched_names]
            for p in selected_pois:
                if p.get("name", "") not in enriched_names and len(ordered_pois) < 5:
                    ordered_pois.append(p)

            for poi in ordered_pois[:5]:
                name = poi.get("name", "")
                info = enriched.get(name, {})
                hotels.append(Hotel(
                    name=name,
                    address=poi.get("address", ""),
                    location=None,
                    price_range=info.get("price_range", ""),
                    rating=info.get("rating"),
                    distance="",
                    type=info.get("type", request.accommodation),
                    estimated_cost=info.get("estimated_cost", 0)
                ))

            # ---- 第4步: 代码批量补坐标 ----
            logger.info(f"解析到 {len(hotels)} 个酒店，开始补充坐标...")
            hotels = self._fill_hotel_locations(hotels, request.city)
            logger.info(f"最终保留 {len(hotels)} 个有效酒店")

            if not hotels:
                logger.warning("未解析到任何有效酒店信息")

            return {
                "hotels": hotels,
                "current_step": "hotels_found",
                "messages": [{"role": "assistant", "content": f"已找到 {len(hotels)} 个酒店"}]
            }
        except Exception as e:
            logger.error(f"酒店搜索失败: {str(e)}", exc_info=True)
            return {"error": f"酒店搜索失败: {str(e)}", "current_step": "error"}

    # ========== 节点: 行程规划 ==========

    def _plan_itinerary(self, state: TripPlannerState) -> Dict[str, Any]:
        """行程规划节点: Agent 综合所有数据生成行程计划"""
        logger.info("📋 生成行程计划...")
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
            return {"error": f"行程规划失败: {str(e)}", "current_step": "error"}

    # ========== 节点: 错误处理 ==========

    def _handle_error(self, state: TripPlannerState) -> Dict[str, Any]:
        error_msg = state.get("error", "未知错误")
        logger.warning(f"⚠️  处理错误: {error_msg}")
        fallback_plan = self._create_fallback_plan(state["request"])
        return {
            "trip_plan": fallback_plan,
            "error": None,
            "current_step": "error_handled",
            "messages": [{"role": "assistant", "content": f"遇到错误，已生成备用计划: {error_msg}"}]
        }

    # ========== MCP 工具数据提取 ==========

    def _extract_pois_from_tool_result(self, result: Any) -> List[dict]:
        """从 MCP 工具返回值中提取 pois 数组"""
        try:
            raw = result
            if isinstance(raw, tuple):
                raw = raw[0]
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and "text" in item:
                        raw = item["text"]
                        break
            if isinstance(raw, str):
                data = json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            else:
                return []
            return data.get("pois", [])
        except Exception as e:
            logger.warning(f"提取 pois 失败: {str(e)}")
            return []

    def _extract_location_from_geo(self, geo_result: Any) -> Optional[Location]:
        """从 maps_geo 返回值中提取坐标"""
        try:
            raw = geo_result
            if isinstance(raw, tuple):
                raw = raw[0]
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and "text" in item:
                        raw = item["text"]
                        break
                    elif isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, dict) and "text" in sub:
                                raw = sub["text"]
                                break
            if isinstance(raw, str):
                data = json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            else:
                return None

            geocodes = data.get("geocodes", []) or data.get("return", [])
            if geocodes and isinstance(geocodes, list):
                loc_str = geocodes[0].get("location", "")
                if "," in str(loc_str):
                    parts = str(loc_str).split(",")
                    return Location(
                        longitude=float(parts[0].strip()),
                        latitude=float(parts[1].strip())
                    )
        except Exception as e:
            logger.warning(f"解析 geo 结果失败: {str(e)}")
        return None

    def _extract_weather_from_tool_result(self, result: Any) -> List[WeatherInfo]:
        """从 maps_weather 工具返回值中提取天气信息"""
        try:
            raw = result
            if isinstance(raw, tuple):
                raw = raw[0]
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and "text" in item:
                        raw = item["text"]
                        break
            if isinstance(raw, str):
                data = json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            else:
                return []

            forecasts = data.get("forecasts", [])
            if not forecasts:
                return []

            casts = forecasts[0].get("casts", [])
            weather_list = []
            for cast in casts:
                weather_list.append(WeatherInfo(
                    date=cast.get("date", ""),
                    day_weather=cast.get("dayweather", ""),
                    night_weather=cast.get("nightweather", ""),
                    day_temp=cast.get("daytemp", 0),
                    night_temp=cast.get("nighttemp", 0),
                    wind_direction=cast.get("daywind", ""),
                    wind_power=cast.get("daypower", "")
                ))
            return weather_list
        except Exception as e:
            logger.warning(f"提取天气数据失败: {str(e)}")
            return []

    # ========== 坐标补充 ==========

    def _fill_locations(self, attractions: List[Attraction], city: str) -> List[Attraction]:
        """用 maps_geo 工具批量补充景点坐标"""
        geo_tool = next((t for t in self.tools if t.name == "maps_geo"), None)
        if not geo_tool:
            logger.warning("未找到 maps_geo 工具，无法补充坐标")
            return attractions

        for attr in attractions:
            if attr.location is not None:
                continue
            try:
                address = f"{city}{attr.address}" if attr.address else f"{city}{attr.name}"
                geo_result = geo_tool.invoke({"address": address, "city": city})
                location = self._extract_location_from_geo(geo_result)
                if location:
                    attr.location = location
                    logger.info(f"[GEO] {attr.name} 坐标: {location.longitude}, {location.latitude}")
                else:
                    logger.warning(f"[GEO] {attr.name} 未能获取坐标")
            except Exception as e:
                logger.warning(f"[GEO] {attr.name} 地理编码失败: {str(e)}")
        return attractions

    def _fill_hotel_locations(self, hotels: List[Hotel], city: str) -> List[Hotel]:
        """用 maps_geo 工具批量补充酒店坐标"""
        geo_tool = next((t for t in self.tools if t.name == "maps_geo"), None)
        if not geo_tool:
            logger.warning("未找到 maps_geo 工具，无法补充坐标")
            return hotels

        for hotel in hotels:
            if hotel.location is not None:
                continue
            try:
                address = f"{city}{hotel.address}" if hotel.address else f"{city}{hotel.name}"
                geo_result = geo_tool.invoke({"address": address, "city": city})
                location = self._extract_location_from_geo(geo_result)
                if location:
                    hotel.location = location
                    logger.info(f"[GEO] {hotel.name} 坐标: {location.longitude}, {location.latitude}")
                else:
                    logger.warning(f"[GEO] {hotel.name} 未能获取坐标")
            except Exception as e:
                logger.warning(f"[GEO] {hotel.name} 地理编码失败: {str(e)}")
        return hotels

    def _fetch_weather_by_code(self, city: str) -> List[WeatherInfo]:
        """代码直接调用天气工具（Agent 失败时的兜底）"""
        weather_tool = next((t for t in self.tools if t.name == "maps_weather"), None)
        if not weather_tool:
            return []
        try:
            result = weather_tool.invoke({"city": city})
            return self._extract_weather_from_tool_result(result)
        except Exception as e:
            logger.warning(f"代码天气获取失败: {str(e)}")
            return []

    # ========== Agent 数据丰富化解析 ==========

    def _parse_agent_enrichment(self, output: str) -> Dict[str, dict]:
        """解析 Agent 返回的景点筛选/描述结果，返回 {name: {description, category, visit_duration}}"""
        result = {}
        try:
            json_str = self._extract_json(output)
            data = json.loads(json_str)
            if isinstance(data, dict):
                data = data.get("attractions", data.get("results", []))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("name"):
                        result[item["name"]] = {
                            "description": item.get("description", ""),
                            "category": item.get("category", "景点"),
                            "visit_duration": item.get("visit_duration", 120),
                        }
        except Exception as e:
            logger.warning(f"解析 Agent 景点筛选结果失败: {str(e)}")
        return result

    def _parse_hotel_enrichment(self, output: str) -> Dict[str, dict]:
        """解析 Agent 返回的酒店筛选/描述结果"""
        result = {}
        try:
            json_str = self._extract_json(output)
            data = json.loads(json_str)
            if isinstance(data, dict):
                data = data.get("hotels", data.get("results", []))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("name"):
                        result[item["name"]] = {
                            "price_range": item.get("price_range", ""),
                            "type": item.get("type", ""),
                            "estimated_cost": item.get("estimated_cost", 0),
                            "rating": item.get("rating"),
                        }
        except Exception as e:
            logger.warning(f"解析 Agent 酒店筛选结果失败: {str(e)}")
        return result

    # ========== 查询构建 ==========

    def _build_planner_query(self, request: TripRequest, attractions: List[Attraction],
                             weather: List[WeatherInfo], hotels: List[Hotel]) -> str:
        attractions_data = []
        for a in attractions:
            attractions_data.append({
                "name": a.name, "address": a.address,
                "location": {"longitude": a.location.longitude, "latitude": a.location.latitude} if a.location else None,
                "visit_duration": a.visit_duration, "description": a.description,
                "category": a.category, "ticket_price": a.ticket_price
            })

        weather_data = []
        for w in weather:
            weather_data.append({
                "date": w.date, "day_weather": w.day_weather, "night_weather": w.night_weather,
                "day_temp": w.day_temp, "night_temp": w.night_temp,
                "wind_direction": w.wind_direction, "wind_power": w.wind_power
            })

        hotels_data = []
        for h in hotels:
            hotels_data.append({
                "name": h.name, "address": h.address,
                "location": {"longitude": h.location.longitude, "latitude": h.location.latitude} if h.location else None,
                "price_range": h.price_range, "rating": h.rating,
                "type": h.type, "estimated_cost": h.estimated_cost
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
5. 景点和酒店的经纬度坐标必须原样复制输入数据中的值
6. 必须包含预算汇总
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"
        return query

    # ========== Agent 输入输出工具方法 ==========

    def _prepare_agent_input(self, user_input: str, chat_history: list) -> dict:
        messages = list(chat_history)
        messages.append({"role": "user", "content": user_input})
        return {"messages": messages}

    def _normalize_message_content(self, content: Any) -> str:
        """把 LangChain 返回的各种 content 结构统一转成纯文本"""
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
        """优先提取 assistant 文本内容"""
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

        raise ValueError("未找到可解析的 JSON 内容")

    # ========== 数据解析: 景点 / 天气 / 酒店 / 行程 ==========

    def _parse_location(self, raw_location: Any) -> Optional[Location]:
        """安全解析经纬度"""
        if raw_location is None:
            return None

        # 处理字符串格式 "120.15518,30.27415"
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
        """解析天气信息"""
        json_str = ""
        try:
            json_str = self._extract_json(response, preferred_keys=["weather_info"])
            data = json.loads(json_str)
        except Exception as e:
            logger.error(f"解析天气 JSON 失败: {str(e)}")
            return self._parse_weather_from_markdown(response)

        if isinstance(data, dict):
            data = data.get("weather_info", [])
        if not isinstance(data, list):
            return []

        weather_info: List[WeatherInfo] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            try:
                w = WeatherInfo(
                    date=item.get("date", ""),
                    day_weather=item.get("day_weather", ""),
                    night_weather=item.get("night_weather", ""),
                    day_temp=item.get("day_temp", 0),
                    night_temp=item.get("night_temp", 0),
                    wind_direction=item.get("wind_direction", ""),
                    wind_power=item.get("wind_power", "")
                )
                if w.date.strip():
                    weather_info.append(w)
            except Exception as e:
                logger.warning(f"跳过第 {idx + 1} 条天气: {str(e)}")
        logger.info(f"天气解析完成：成功 {len(weather_info)} 条")
        return weather_info

    def _parse_weather_from_markdown(self, response: str) -> List[WeatherInfo]:
        """兜底解析 Markdown 格式天气"""
        weather_info: List[WeatherInfo] = []
        pattern = re.compile(
            r"\d+\.\s*\*\*(?P<head>[^*]+)\*\*(?P<body>.*?)(?=\n\d+\.\s*\*\*|\Z)", re.DOTALL
        )
        for match in pattern.finditer(response):
            head, body = match.group("head"), match.group("body")
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", head)
            if not date_match:
                continue
            day_line = re.search(r"白天\s*[:：]\s*([^\n]+)", body)
            night_line = re.search(r"夜间\s*[:：]\s*([^\n]+)", body)

            def _parse_line(line: str):
                if not line:
                    return "", 0, "", ""
                w_m = re.match(r"([^，,]+)", line.strip())
                t_m = re.search(r"温度\s*(\d+)", line)
                winds = re.findall(r"[，,]\s*([^，,\n]+)", line)
                return (
                    w_m.group(1).strip() if w_m else "",
                    int(t_m.group(1)) if t_m else 0,
                    winds[0].strip() if len(winds) >= 1 else "",
                    winds[1].strip() if len(winds) >= 2 else "",
                )

            dw, dt, dwd, dwp = _parse_line(day_line.group(1) if day_line else "")
            nw, nt, nwd, nwp = _parse_line(night_line.group(1) if night_line else "")
            try:
                weather_info.append(WeatherInfo(
                    date=date_match.group(1), day_weather=dw, night_weather=nw,
                    day_temp=dt, night_temp=nt,
                    wind_direction=dwd or nwd, wind_power=dwp or nwp,
                ))
            except Exception:
                continue
        return weather_info

    def _parse_trip_plan(self, response: str, request: TripRequest) -> TripPlan:
        """解析行程计划"""
        json_str = ""
        try:
            json_str = self._extract_json(response, preferred_keys=["days", "trip_plan"])
            data = json.loads(json_str)
        except Exception as e:
            logger.error(f"解析行程计划 JSON 失败: {str(e)}")
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
            budget=None
        )

        # 解析天气
        for wd in data.get("weather_info", []):
            if not isinstance(wd, dict):
                continue
            try:
                wi = WeatherInfo(
                    date=wd.get("date", ""), day_weather=wd.get("day_weather", ""),
                    night_weather=wd.get("night_weather", ""),
                    day_temp=wd.get("day_temp", 0), night_temp=wd.get("night_temp", 0),
                    wind_direction=wd.get("wind_direction", ""), wind_power=wd.get("wind_power", "")
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
                # 景点
                attractions = []
                for ad in day_data.get("attractions", []):
                    if not isinstance(ad, dict):
                        continue
                    try:
                        raw_tp = ad.get("ticket_price", 0)
                        pt = ad.get("price_text", "")
                        if isinstance(raw_tp, str) and not pt:
                            pt = raw_tp
                        a = Attraction(
                            name=ad.get("name", ""), address=ad.get("address", ""),
                            location=self._parse_location(ad.get("location")),
                            visit_duration=ad.get("visit_duration", 0),
                            description=ad.get("description", ""),
                            category=ad.get("category", "景点"),
                            rating=ad.get("rating"),
                            photos=ad.get("photos", []) or [],
                            poi_id=ad.get("poi_id"),
                            image_url=ad.get("image_url"),
                            ticket_price=raw_tp, price_text=pt,
                        )
                        if a.name.strip():
                            attractions.append(a)
                    except Exception as e:
                        logger.warning(f"第{day_idx + 1}天景点解析失败: {str(e)}")

                # 餐饮
                meals = []
                for md in day_data.get("meals", []):
                    if not isinstance(md, dict):
                        continue
                    try:
                        m = Meal(**md)
                        if hasattr(m, "name") and getattr(m, "name", "").strip():
                            meals.append(m)
                    except Exception as e:
                        logger.warning(f"第{day_idx + 1}天餐饮解析失败: {str(e)}")

                # 酒店
                hotel = None
                hd = day_data.get("hotel")
                if isinstance(hd, dict):
                    try:
                        hotel = Hotel(
                            name=hd.get("name", ""), address=hd.get("address", ""),
                            location=self._parse_location(hd.get("location")),
                            price_range=hd.get("price_range", ""),
                            rating=hd.get("rating"), distance=hd.get("distance", ""),
                            type=hd.get("type", ""), estimated_cost=hd.get("estimated_cost", 0)
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
                    hotel=hotel, attractions=attractions, meals=meals
                )
                trip_plan.days.append(day_plan)
                logger.info(f"第{day_idx + 1}天: 景点{len(attractions)}个, 餐饮{len(meals)}个")
            except Exception as e:
                logger.warning(f"跳过第{day_idx + 1}天: {str(e)}")

        # 预算
        bd = data.get("budget")
        if isinstance(bd, dict):
            try:
                trip_plan.budget = Budget(**bd)
            except Exception as e:
                logger.warning(f"预算解析失败: {str(e)}")

        if not trip_plan.days:
            logger.error("未解析到任何有效行程天数，返回备用计划")
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
                description=f"第{i + 1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j + 1}", address=f"{request.city}市",
                        location=Location(longitude=120.1551 + i * 0.01 + j * 0.005,
                                          latitude=30.2741 + i * 0.01 + j * 0.005),
                        visit_duration=120, description=f"{request.city}的著名景点", category="景点"
                    ) for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i + 1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i + 1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i + 1}天晚餐", description="晚餐推荐"),
                ]
            ))

        return TripPlan(
            city=request.city, start_date=request.start_date, end_date=request.end_date,
            days=days, weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程，建议提前查看各景点的开放时间。"
        )

    # ========== 入口 ==========

    def plan_trip(self, request: TripRequest) -> TripPlan:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🚀 开始 LangGraph 旅行规划工作流...")
        logger.info(f"目的地: {request.city}")
        logger.info(f"{'=' * 60}\n")

        initial_state: TripPlannerState = create_initial_state(request)

        final_state = self.graph.invoke(initial_state, config={"recursion_limit": 50})

        if final_state.get("error") and not final_state.get("trip_plan"):
            error_msg = final_state.get("error", "未知错误")
            logger.error(f"❌ 旅行规划失败: {error_msg}")
            raise Exception(error_msg)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"✅ 旅行计划生成完成!")
        logger.info(f"{'=' * 60}\n")

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