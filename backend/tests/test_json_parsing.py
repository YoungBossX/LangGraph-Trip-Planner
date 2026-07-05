"""JSON 解析层单元测试 — 覆盖 _extract_json, _parse_attractions, _parse_weather, 截断修复等"""

import json
from unittest.mock import MagicMock, patch

import pytest

# Mock 掉 MCP 工具和 LLM，创建一个可测试的 TripPlannerWorkflow 实例
_mock_tool = MagicMock()
_mock_tool.name = "maps_text_search"

_mock_agent = MagicMock()


@pytest.fixture(scope="module")
def workflow():
    """创建 TripPlannerWorkflow 实例（mock 掉构造函数中的外部依赖）"""
    with (
        patch("app.workflows.trip_planner_graph.get_cached_amap_tools", return_value=[_mock_tool]),
        patch("app.workflows.trip_planner_graph.get_agent", return_value=_mock_agent),
    ):
        from app.workflows.trip_planner_graph import TripPlannerWorkflow
        return TripPlannerWorkflow()


class TestExtractJson:
    """测试 _extract_json — Agent 输出 → JSON 字符串提取"""

    def test_pure_json_array(self, workflow):
        assert json.loads(workflow._extract_json('[{"name":"test"}]')) == [{"name": "test"}]

    def test_json_with_markdown_fence(self, workflow):
        result = workflow._extract_json('```json\n[{"name":"test"}]\n```')
        assert json.loads(result) == [{"name": "test"}]

    def test_json_with_text_prefix(self, workflow):
        result = workflow._extract_json('这是搜索结果：\n[{"name":"西湖"}]\n希望对你有帮助')
        assert json.loads(result) == [{"name": "西湖"}]

    def test_nested_json_object(self, workflow):
        result = workflow._extract_json('{"days":[{"date":"2025-01-01"}],"city":"杭州"}')
        assert json.loads(result)["city"] == "杭州"

    def test_empty_response_raises(self, workflow):
        with pytest.raises(ValueError, match="响应为空"):
            workflow._extract_json("")

    def test_no_json_raises(self, workflow):
        with pytest.raises(ValueError, match="未找到可解析"):
            workflow._extract_json("这是一段没有任何JSON内容的纯文本")


class TestTruncatedJsonRepair:
    """测试截断 JSON 修复 — LLM max_tokens 截断场景"""

    def test_repair_truncated_array(self, workflow):
        # 模拟被截断的 JSON 数组（最后一个对象不完整）
        truncated = (
            '[{"name":"景点1","address":"地址1","location":{"longitude":120.0,"latitude":30.0}},'
            '{"name":"景点2","addr'
        )
        result = workflow._extract_json(truncated)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "景点1"

    def test_repair_truncated_array_mid_object(self, workflow):
        truncated = '[{"name":"景点1"},{"name":"景'
        result = workflow._extract_json(truncated)
        parsed = json.loads(result)
        assert len(parsed) >= 1

    def test_repair_truncated_with_text_prefix(self, workflow):
        truncated = '以下是搜索到的景点：\n[{"name":"西湖","address":"杭州市"}'
        result = workflow._extract_json(truncated)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "西湖"


class TestParseAttractions:
    """测试 _parse_attractions_from_agent"""

    def test_from_agent_output(self, workflow):
        mock_output = _mock_agent_output(
            '[{"name":"西湖","address":"杭州市西湖区","location":{"longitude":120.14,"latitude":30.24},"visit_duration":120,"description":"美丽","category":"自然","ticket_price":0}]'
        )
        result = workflow._parse_attractions_from_agent(mock_output, "杭州")
        assert len(result) == 1
        assert result[0].name == "西湖"
        assert result[0].location.longitude == 120.14
        assert result[0].location.latitude == 30.24

    def test_wrapped_in_dict(self, workflow):
        mock_output = _mock_agent_output(
            '{"attractions":[{"name":"西湖","address":"杭州市","location":{"longitude":120.0,"latitude":30.0}}]}'
        )
        result = workflow._parse_attractions_from_agent(mock_output, "杭州")
        assert len(result) == 1

    def test_empty_input(self, workflow):
        mock_output = _mock_agent_output('[]')
        result = workflow._parse_attractions_from_agent(mock_output, "杭州")
        assert result == []

    def test_skips_invalid_items(self, workflow):
        mock_output = _mock_agent_output(
            '[{"name":"","address":""},{"name":"西湖","address":"杭州","location":{"longitude":120.0,"latitude":30.0}}]'
        )
        result = workflow._parse_attractions_from_agent(mock_output, "杭州")
        assert len(result) == 1
        assert result[0].name == "西湖"

    def test_parse_location_from_string(self, workflow):
        mock_output = _mock_agent_output('[{"name":"测试","address":"某地","location":"120.15,30.25"}]')
        result = workflow._parse_attractions_from_agent(mock_output, "杭州")
        assert result[0].location.longitude == 120.15
        assert result[0].location.latitude == 30.25

    def test_parse_location_none(self, workflow):
        mock_output = _mock_agent_output('[{"name":"测试","address":"某地","location":null}]')
        result = workflow._parse_attractions_from_agent(mock_output, "杭州")
        assert result[0].location is None


class TestParseWeather:
    """测试 _parse_weather"""

    def test_from_agent_output(self, workflow):
        result = workflow._parse_weather(
            '[{"date":"2025-06-01","day_weather":"晴","night_weather":"多云","day_temp":28,"night_temp":18,"wind_direction":"南风","wind_power":"1-3级"}]'
        )
        assert len(result) == 1
        assert result[0].date == "2025-06-01"
        assert result[0].day_temp == 28
        assert result[0].day_weather == "晴"

    def test_wrapped_in_weather_info(self, workflow):
        result = workflow._parse_weather(
            '{"weather_info":[{"date":"2025-06-01","day_weather":"晴","night_weather":"多云","day_temp":25,"night_temp":15,"wind_direction":"东风","wind_power":"2级"}]}'
        )
        assert len(result) == 1

    def test_alternative_field_names(self, workflow):
        result = workflow._parse_weather(
            '[{"date":"2025-06-01","dayweather":"雨","nightweather":"阴","daytemp":"23","nighttemp":"12","daywind":"北","daypower":"3"}]'
        )
        assert len(result) == 1
        assert result[0].day_weather == "雨"

    def test_empty_input(self, workflow):
        result = workflow._parse_weather("")
        assert result == []

    def test_skips_invalid_items(self, workflow):
        result = workflow._parse_weather(
            '[{"date":"","day_weather":"晴"},{"date":"2025-06-01","day_weather":"多云","night_weather":"阴","day_temp":20,"night_temp":10}]'
        )
        assert len(result) == 1
        assert result[0].date == "2025-06-01"


class TestParseLocation:
    """测试 _parse_location"""

    def test_from_dict(self, workflow):
        loc = workflow._parse_location({"longitude": 120.15, "latitude": 30.25})
        assert loc.longitude == 120.15
        assert loc.latitude == 30.25

    def test_from_comma_string(self, workflow):
        loc = workflow._parse_location("120.15,30.25")
        assert loc.longitude == 120.15
        assert loc.latitude == 30.25

    def test_none_input(self, workflow):
        assert workflow._parse_location(None) is None

    def test_invalid_string(self, workflow):
        assert workflow._parse_location("invalid") is None


def _mock_agent_output(json_str: str) -> dict:
    """构造 Agent 返回的模拟 result dict"""
    return {"messages": [{"role": "assistant", "content": json_str}]}
