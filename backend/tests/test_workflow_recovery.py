from unittest.mock import MagicMock, patch

import pytest


def _tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _agent_result(content: str) -> dict:
    return {"messages": [{"role": "assistant", "content": content}]}


def _trip_request():
    from app.models.schemas import TripRequest

    return TripRequest(
        city="Hangzhou",
        start_date="2026-03-01",
        end_date="2026-03-03",
        travel_days=3,
        transportation="public transit",
        accommodation="budget hotel",
        preferences=["history"],
    )


def _state():
    from app.workflows.trip_planner_state import create_initial_state

    return create_initial_state(_trip_request())


@pytest.fixture()
def workflow_and_agents():
    agents = {
        "attraction_search": MagicMock(),
        "weather": MagicMock(),
        "hotel": MagicMock(),
        "planner": MagicMock(),
    }

    def fake_get_agent(agent_type, tools):
        return agents[agent_type]

    tools = [_tool("maps_text_search"), _tool("maps_geo"), _tool("maps_weather")]
    with (
        patch("app.workflows.trip_planner_graph.get_cached_amap_tools", return_value=tools),
        patch("app.workflows.trip_planner_graph.get_agent", side_effect=fake_get_agent),
    ):
        from app.workflows.trip_planner_graph import TripPlannerWorkflow

        yield TripPlannerWorkflow(), agents


def test_search_attractions_empty_agent_output_is_error(workflow_and_agents):
    workflow, agents = workflow_and_agents
    from app.workflows.trip_planner_graph import NODE_ATTRACTIONS

    agents["attraction_search"].invoke.return_value = _agent_result("[]")

    result = workflow._search_attractions(_state())

    assert result["failed_node"] == NODE_ATTRACTIONS
    assert "error" in result
    assert "景点" in result["error"]


def test_check_weather_empty_agent_output_is_error(workflow_and_agents):
    workflow, agents = workflow_and_agents
    from app.workflows.trip_planner_graph import NODE_WEATHER

    agents["weather"].invoke.return_value = _agent_result("[]")

    result = workflow._check_weather(_state())

    assert result["failed_node"] == NODE_WEATHER
    assert "error" in result
    assert "天气" in result["error"]


def test_find_hotels_empty_agent_output_is_error(workflow_and_agents):
    workflow, agents = workflow_and_agents
    from app.workflows.trip_planner_graph import NODE_HOTELS

    agents["hotel"].invoke.return_value = _agent_result("[]")

    result = workflow._find_hotels(_state())

    assert result["failed_node"] == NODE_HOTELS
    assert "error" in result
    assert "酒店" in result["error"]


def test_retry_count_resets_when_different_node_fails(workflow_and_agents):
    workflow, _ = workflow_and_agents
    from app.workflows.trip_planner_graph import NODE_HOTELS, NODE_WEATHER

    state = _state()
    state["failed_node"] = NODE_HOTELS
    state["last_failed_node"] = NODE_WEATHER
    state["retry_count"] = 2
    state["error"] = "previous failure"

    result = workflow._handle_error(state)

    assert result["retry_count"] == 1
    assert result["last_failed_node"] == NODE_HOTELS
    assert result["error"] is None
    assert "failed_node" not in result


def test_skip_to_plan_clears_failed_node_after_retry_budget(workflow_and_agents):
    workflow, _ = workflow_and_agents
    from app.workflows.trip_planner_graph import NODE_WEATHER

    state = _state()
    state["failed_node"] = NODE_WEATHER
    state["last_failed_node"] = NODE_WEATHER
    state["retry_count"] = 2
    state["error"] = "weather unavailable"
    state["attractions"] = [object()]

    result = workflow._handle_error(state)

    assert result["error"] is None
    assert result["failed_node"] is None


def test_plan_trip_raises_after_retry_budget_without_partial_data(workflow_and_agents):
    workflow, agents = workflow_and_agents

    agents["attraction_search"].invoke.side_effect = RuntimeError("LLM unavailable")

    with pytest.raises(Exception, match="景点搜索失败"):
        workflow.plan_trip(_trip_request())

    assert agents["attraction_search"].invoke.call_count == 3


def test_weather_and_hotels_are_parallel_branches_after_attractions(workflow_and_agents):
    workflow, _ = workflow_and_agents
    from app.workflows.trip_planner_graph import NODE_ATTRACTIONS, NODE_CONTEXT, NODE_HOTELS, NODE_PLAN, NODE_WEATHER

    edges = {
        (edge.source, edge.target, edge.data, edge.conditional)
        for edge in workflow.graph.get_graph().edges
    }

    assert any(source == NODE_ATTRACTIONS and target == NODE_WEATHER for source, target, _, _ in edges)
    assert any(source == NODE_ATTRACTIONS and target == NODE_HOTELS for source, target, _, _ in edges)
    assert not any(source == NODE_WEATHER and target == NODE_HOTELS for source, target, _, _ in edges)
    assert any(source == NODE_WEATHER and target == NODE_CONTEXT for source, target, _, _ in edges)
    assert any(source == NODE_HOTELS and target == NODE_CONTEXT for source, target, _, _ in edges)
    assert any(source == NODE_CONTEXT and target == NODE_PLAN for source, target, _, _ in edges)


def test_parallel_context_reaches_planner_with_weather_and_hotels(workflow_and_agents):
    workflow, agents = workflow_and_agents

    agents["attraction_search"].invoke.return_value = _agent_result(
        '[{"name":"West Lake","address":"Hangzhou","location":{"longitude":120.1,"latitude":30.2},'
        '"visit_duration":120,"description":"lake","category":"scenic","ticket_price":0}]'
    )
    agents["weather"].invoke.return_value = _agent_result(
        '[{"date":"2026-03-01","day_weather":"sunny","night_weather":"clear",'
        '"day_temp":20,"night_temp":10,"wind_direction":"east","wind_power":"1"}]'
    )
    agents["hotel"].invoke.return_value = _agent_result(
        '[{"name":"Lake Hotel","address":"Hangzhou","location":{"longitude":120.2,"latitude":30.3},'
        '"price_range":"200-400","rating":4.5,"type":"budget hotel","estimated_cost":300}]'
    )
    agents["planner"].invoke.return_value = _agent_result(
        '{"city":"Hangzhou","start_date":"2026-03-01","end_date":"2026-03-03",'
        '"weather_info":[{"date":"2026-03-01","day_weather":"sunny","night_weather":"clear",'
        '"day_temp":20,"night_temp":10,"wind_direction":"east","wind_power":"1"}],'
        '"days":[{"date":"2026-03-01","day_index":0,"description":"Day 1",'
        '"transportation":"public transit","accommodation":"budget hotel",'
        '"hotel":{"name":"Lake Hotel","address":"Hangzhou",'
        '"location":{"longitude":120.2,"latitude":30.3},"price_range":"200-400",'
        '"rating":4.5,"distance":"","type":"budget hotel","estimated_cost":300},'
        '"attractions":[{"name":"West Lake","address":"Hangzhou",'
        '"location":{"longitude":120.1,"latitude":30.2},"visit_duration":120,'
        '"description":"lake","category":"scenic","ticket_price":0}],'
        '"meals":[{"type":"breakfast","name":"Breakfast"},'
        '{"type":"lunch","name":"Lunch"},{"type":"dinner","name":"Dinner"}]}],'
        '"overall_suggestions":"Enjoy","budget":{"total_attractions":0,"total_hotels":300,'
        '"total_meals":150,"total_transportation":50,"total":500}}'
    )

    trip_plan = workflow.plan_trip(_trip_request())

    assert trip_plan.days[0].hotel.name == "Lake Hotel"
    agents["weather"].invoke.assert_called_once()
    agents["hotel"].invoke.assert_called_once()
    agents["planner"].invoke.assert_called_once()
