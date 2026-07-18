import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import TripRequest
from app.workflows.execution_control import (
    ExecutionControl,
    WorkflowCancelledError,
    WorkflowTimeoutError,
)
from app.workflows.trip_planner_graph import TripPlannerWorkflow
from app.workflows.trip_planner_state import create_initial_state


def _request() -> TripRequest:
    return TripRequest(
        city="Hangzhou",
        start_date="2026-03-01",
        end_date="2026-03-03",
        travel_days=3,
        transportation="public transit",
        accommodation="budget hotel",
        preferences=["history"],
    )


def _tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


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
        yield TripPlannerWorkflow(), agents


def _cancelled_control() -> ExecutionControl:
    control = ExecutionControl(timeout_seconds=30, clock=lambda: 0.0)
    control.cancel()
    return control


def _expired_control() -> ExecutionControl:
    readings = iter([0.0, 30.0])
    return ExecutionControl(timeout_seconds=30, clock=lambda: next(readings))


def test_create_initial_state_stores_optional_control():
    control = ExecutionControl(timeout_seconds=30)

    controlled_state = create_initial_state(_request(), control=control)
    legacy_state = create_initial_state(_request())

    assert controlled_state["control"] is control
    assert legacy_state["control"] is None


def test_plan_trip_and_astream_plan_accept_optional_control_without_legacy_deadline():
    class RecordingGraph:
        def __init__(self):
            self.controls = []

        def invoke(self, state, config):
            self.controls.append(state["control"])
            return {"trip_plan": object(), "error": None}

        async def astream(self, state, config):
            self.controls.append(state["control"])
            yield {"done": {"current_step": "done"}}

    workflow = TripPlannerWorkflow.__new__(TripPlannerWorkflow)
    workflow.graph = RecordingGraph()
    control = ExecutionControl(timeout_seconds=30)

    workflow.plan_trip(_request())
    workflow.plan_trip(_request(), control=control)

    async def collect_streams():
        legacy = [item async for item in workflow.astream_plan(_request())]
        controlled = [item async for item in workflow.astream_plan(_request(), control=control)]
        return legacy, controlled

    legacy_events, controlled_events = asyncio.run(collect_streams())

    assert legacy_events == [("done", {"current_step": "done"})]
    assert controlled_events == [("done", {"current_step": "done"})]
    assert workflow.graph.controls == [None, control, None, control]


def test_pre_cancelled_control_invokes_no_agent(workflow_and_agents):
    workflow, agents = workflow_and_agents

    with pytest.raises(WorkflowCancelledError):
        workflow.plan_trip(_request(), control=_cancelled_control())

    assert all(agent.invoke.call_count == 0 for agent in agents.values())


def test_attraction_agent_cancellation_is_caught_by_post_invoke_checkpoint(workflow_and_agents):
    workflow, agents = workflow_and_agents
    control = ExecutionControl(timeout_seconds=30)

    def cancel_before_return(*args, **kwargs):
        control.cancel()
        return {"messages": [{"role": "assistant", "content": "[]"}]}

    agents["attraction_search"].invoke.side_effect = cancel_before_return

    with pytest.raises(WorkflowCancelledError):
        workflow.plan_trip(_request(), control=control)

    agents["attraction_search"].invoke.assert_called_once()
    agents["weather"].invoke.assert_not_called()
    agents["hotel"].invoke.assert_not_called()
    agents["planner"].invoke.assert_not_called()


@pytest.mark.parametrize(
    ("control_factory", "exception_type"),
    [
        (_cancelled_control, WorkflowCancelledError),
        (_expired_control, WorkflowTimeoutError),
    ],
    ids=["cancelled", "expired"],
)
def test_execution_control_exceptions_escape_direct_node_calls(
    workflow_and_agents, control_factory, exception_type
):
    workflow, agents = workflow_and_agents
    state = create_initial_state(_request(), control=control_factory())

    with pytest.raises(exception_type):
        workflow._search_attractions(state)

    agents["attraction_search"].invoke.assert_not_called()


@pytest.mark.parametrize(
    ("control_factory", "exception_type"),
    [
        (_cancelled_control, WorkflowCancelledError),
        (_expired_control, WorkflowTimeoutError),
    ],
    ids=["cancelled", "expired"],
)
def test_execution_control_exceptions_escape_full_graph_without_retries(
    workflow_and_agents, control_factory, exception_type
):
    workflow, agents = workflow_and_agents

    with pytest.raises(exception_type):
        workflow.plan_trip(_request(), control=control_factory())

    assert all(agent.invoke.call_count == 0 for agent in agents.values())


class CancelOnSecondCheck:
    def __init__(self):
        self.check_count = 0

    def check(self):
        self.check_count += 1
        if self.check_count == 2:
            raise WorkflowCancelledError("cancelled after Agent invocation")


@pytest.mark.parametrize(
    ("node_name", "agent_name"),
    [
        ("_search_attractions", "attraction_search"),
        ("_check_weather", "weather"),
        ("_find_hotels", "hotel"),
        ("_plan_itinerary", "planner"),
    ],
)
def test_each_agent_invocation_has_pre_and_post_checkpoints(workflow_and_agents, node_name, agent_name):
    workflow, agents = workflow_and_agents
    control = CancelOnSecondCheck()
    state = create_initial_state(_request(), control=control)

    with pytest.raises(WorkflowCancelledError):
        getattr(workflow, node_name)(state)

    assert control.check_count == 2
    agents[agent_name].invoke.assert_called_once()
