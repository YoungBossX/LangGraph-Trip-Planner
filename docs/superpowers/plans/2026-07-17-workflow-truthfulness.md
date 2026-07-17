# Workflow Truthfulness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the trip-planning workflow terminate predictably and return only complete plans whose dates, attractions, hotels, and weather are constrained to validated request and acquisition data.

**Architecture:** Keep the existing LangGraph node topology and public response models. Make retry exhaustion terminal, validate request/calendar and weather coverage before planning, then validate and canonicalize the planner's arrangement against acquisition-stage Pydantic objects before accepting it.

**Tech Stack:** Python 3.10+, LangGraph, Pydantic v2, pytest, Ruff

---

## File Structure

- Modify `backend/app/workflows/trip_planner_graph.py`: finite error routing, weather coverage checks, source resolution, and final plan validation.
- Modify `backend/app/models/schemas.py`: calendar validation for `TripRequest` while preserving string date fields for API compatibility.
- Modify `backend/tests/test_workflow_recovery.py`: graph-level retry and valid end-to-end workflow coverage.
- Create `backend/tests/test_trip_request_validation.py`: focused request calendar tests.
- Create `backend/tests/test_plan_validation.py`: focused source-resolution and itinerary-invariant tests.

### Task 1: Make Retry Exhaustion Terminal

**Files:**
- Modify: `backend/app/workflows/trip_planner_graph.py:124-150,338-367`
- Modify: `backend/tests/test_workflow_recovery.py:98-137`

- [ ] **Step 1: Replace the partial-fallback expectation with terminal-error tests**

Add tests equivalent to:

```python
def test_exhausted_weather_failure_is_terminal(workflow_and_agents):
    workflow, _ = workflow_and_agents
    from app.workflows.trip_planner_graph import NODE_WEATHER

    state = _state()
    state.update({
        "failed_node": NODE_WEATHER,
        "last_failed_node": NODE_WEATHER,
        "retry_count": 2,
        "error": "weather unavailable",
        "attractions": [object()],
    })

    result = workflow._handle_error(state)

    assert result["error"] == "weather unavailable"
    assert result["failed_node"] is None
    assert workflow._route_after_error({**state, **result}) == "end"
```

Also add a full-graph planner test in which acquisition Agents return complete data, the planner always raises `RuntimeError("planner unavailable")`, and assertions require exactly three planner calls plus an exception matching `planner unavailable` rather than `GraphRecursionError`.

- [ ] **Step 2: Run the focused recovery tests and verify RED**

Run:

```powershell
E:\Anaconda\envs\agent\python.exe -m pytest tests/test_workflow_recovery.py -q
```

Expected: the new terminal behavior tests fail because `_handle_error()` clears the error and routes to `skip_to_plan`.

- [ ] **Step 3: Remove partial fallback from the graph**

In `trip_planner_graph.py`:

- remove `"skip_to_plan": NODE_PLAN` from `NODE_ERROR` conditional edges;
- make `_route_after_error()` return only retry routes or `"end"`;
- after retry exhaustion, return the original concrete error and clear only `failed_node`;
- do not branch on partial attractions or weather.

The terminal branch should follow this behavior:

```python
logger.error("节点重试耗尽，终止规划: %s", failed_node)
return {
    "error": error_msg,
    "failed_node": None,
    "last_failed_node": failed_node,
}
```

- [ ] **Step 4: Run focused and full recovery tests and verify GREEN**

Run:

```powershell
E:\Anaconda\envs\agent\python.exe -m pytest tests/test_workflow_recovery.py -q
```

Expected: all recovery tests pass and planner failure invokes the planner exactly three times.

- [ ] **Step 5: Run Ruff on the changed files**

Run:

```powershell
E:\Anaconda\envs\agent\python.exe -m ruff check app/workflows/trip_planner_graph.py tests/test_workflow_recovery.py
```

Expected: no violations.

### Task 2: Validate Request Calendar And Weather Coverage

**Files:**
- Modify: `backend/app/models/schemas.py:1-66`
- Modify: `backend/app/workflows/trip_planner_graph.py:210-240`
- Create: `backend/tests/test_trip_request_validation.py`
- Modify: `backend/tests/test_workflow_recovery.py`

- [ ] **Step 1: Write failing request-calendar tests**

Create parameterized tests that construct `TripRequest` and require `ValidationError` for malformed dates, reversed dates, and inclusive-span mismatch:

```python
@pytest.mark.parametrize(
    ("start_date", "end_date", "travel_days"),
    [
        ("not-a-date", "2026-03-03", 3),
        ("2026-03-04", "2026-03-03", 1),
        ("2026-03-01", "2026-03-03", 2),
    ],
)
def test_trip_request_rejects_invalid_calendar(start_date, end_date, travel_days):
    with pytest.raises(ValidationError):
        TripRequest(
            city="Hangzhou",
            start_date=start_date,
            end_date=end_date,
            travel_days=travel_days,
            transportation="public transit",
            accommodation="budget hotel",
        )
```

Add a positive leap-day or normal inclusive-range case.

- [ ] **Step 2: Write failing weather-coverage tests**

Parameterize weather Agent responses for missing, duplicate, and out-of-range dates. Call `_check_weather(_state())` and require `failed_node == NODE_WEATHER` with an error mentioning date coverage. Add a valid response with exactly `2026-03-01`, `2026-03-02`, and `2026-03-03` and require ordered output.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
E:\Anaconda\envs\agent\python.exe -m pytest tests/test_trip_request_validation.py tests/test_workflow_recovery.py -q
```

Expected: invalid requests are accepted and incomplete weather is accepted.

- [ ] **Step 4: Add calendar validation to `TripRequest`**

Keep the public fields as strings. Import `date` and `model_validator`, then add an `after` validator that:

```python
start = date.fromisoformat(self.start_date)
end = date.fromisoformat(self.end_date)
if end < start:
    raise ValueError("end_date must not be earlier than start_date")
if (end - start).days + 1 != self.travel_days:
    raise ValueError("travel_days must match the inclusive date range")
return self
```

Convert `date.fromisoformat()` failures into a stable `ValueError` mentioning ISO `YYYY-MM-DD` format.

- [ ] **Step 5: Add exact weather coverage validation**

Add focused helpers in `TripPlannerWorkflow`:

```python
@staticmethod
def _requested_dates(request: TripRequest) -> List[str]:
    start = date.fromisoformat(request.start_date)
    return [(start + timedelta(days=offset)).isoformat() for offset in range(request.travel_days)]

def _validate_weather_coverage(self, weather: List[WeatherInfo], request: TripRequest) -> List[WeatherInfo]:
    expected = self._requested_dates(request)
    by_date = {}
    for item in weather:
        if item.date in by_date:
            raise ValueError(f"天气日期重复: {item.date}")
        by_date[item.date] = item
    if set(by_date) != set(expected):
        raise ValueError("天气日期未完整覆盖请求范围")
    return [by_date[value] for value in expected]
```

Call the validator in `_check_weather()` immediately after parsing. A failure must be caught by the existing node exception path so the weather node retries.

- [ ] **Step 6: Run focused tests and Ruff and verify GREEN**

Run:

```powershell
E:\Anaconda\envs\agent\python.exe -m pytest tests/test_trip_request_validation.py tests/test_workflow_recovery.py -q
E:\Anaconda\envs\agent\python.exe -m ruff check app/models/schemas.py app/workflows/trip_planner_graph.py tests/test_trip_request_validation.py tests/test_workflow_recovery.py
```

Expected: all focused tests pass with no Ruff violations.

### Task 3: Constrain And Canonicalize Planner Output

**Files:**
- Modify: `backend/app/workflows/trip_planner_graph.py:283-313,762-887`
- Create: `backend/tests/test_plan_validation.py`
- Modify: `backend/tests/test_workflow_recovery.py:158-end`

- [ ] **Step 1: Write failing final-invariant tests**

Build real `TripPlan`, `Attraction`, `Hotel`, and `WeatherInfo` objects and call a wished-for method:

```python
validated = workflow._validate_and_canonicalize_trip_plan(
    plan,
    request,
    source_attractions,
    source_weather,
    source_hotels,
)
```

Parameterize isolated failures for wrong city, wrong plan dates, missing/extra days, wrong date order, noncontiguous `day_index`, missing required meals, duplicate meal types, unknown meal types, unknown attraction, duplicate daily attraction, unknown hotel, and missing hotel. Each case must raise `ValueError` with a stable reason fragment.

- [ ] **Step 2: Write failing canonicalization tests**

Create planner attractions and hotels with known source names or POI IDs but maliciously changed addresses, coordinates, ratings, and descriptions. Assert the validated plan contains deep copies of source objects and that later mutation of the validated plan does not mutate the acquisition lists.

Also test:

- POI ID match takes precedence when supplied;
- unknown supplied POI ID does not fall back to the name;
- ambiguous normalized source names are rejected;
- normalized names collapse whitespace and compare case-insensitively.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```powershell
E:\Anaconda\envs\agent\python.exe -m pytest tests/test_plan_validation.py -q
```

Expected: failure because `_validate_and_canonicalize_trip_plan()` does not exist.

- [ ] **Step 4: Implement source indexes and resolution**

Add private helpers to `TripPlannerWorkflow`:

```python
@staticmethod
def _normalize_entity_name(value: str) -> str:
    return " ".join(value.split()).casefold()
```

Build unique indexes for attraction POI IDs, attraction normalized names, and hotel normalized names. Reject ambiguous source keys. Resolve each planner entity and return `source.model_copy(deep=True)`. If a planner attraction supplies a POI ID, require that ID to resolve and do not fall back to its name.

- [ ] **Step 5: Implement final itinerary invariants**

`_validate_and_canonicalize_trip_plan()` must:

- compare request and plan city/dates;
- compare exact day count, ordered dates, and zero-based indexes;
- require at least one unique resolved attraction per day;
- require one resolved hotel per day;
- require exactly one each of `breakfast`, `lunch`, and `dinner`, allowing only additional `snack` entries;
- replace `plan.weather_info` with validated, ordered deep copies of source weather;
- return a deep-copied validated plan without mutating the parsed plan or acquisition lists.

Call this method in `_plan_itinerary()` immediately after `_parse_trip_plan()`. Its `ValueError` must flow through the existing planner-node retry path.

- [ ] **Step 6: Upgrade the valid graph fixture**

Update `test_parallel_context_reaches_planner_with_weather_and_hotels` so its mocked responses include:

- all three requested weather dates;
- exactly three day plans with dates `2026-03-01` through `2026-03-03` and indexes `0,1,2`;
- breakfast, lunch, and dinner for each day;
- source-resolvable attraction and hotel references.

Alter at least one planner-provided source field and assert the returned value equals the acquisition source, proving graph-level canonicalization.

- [ ] **Step 7: Run focused and full backend verification**

Run:

```powershell
E:\Anaconda\envs\agent\python.exe -m pytest tests/test_plan_validation.py tests/test_workflow_recovery.py -q
E:\Anaconda\envs\agent\python.exe -m pytest tests -q
E:\Anaconda\envs\agent\python.exe -m ruff check .
```

Expected: all tests and Ruff pass. Existing Pydantic deprecation warnings may remain, but no new warnings are introduced.

### Final Review

- [ ] Review the complete diff against `docs/superpowers/specs/2026-07-17-workflow-truthfulness-design.md`.
- [ ] Confirm no `.env`, frontend, API, evaluator, dependency, or generated files changed.
- [ ] Confirm `git diff --check` passes and the working tree contains only reviewed files.
- [ ] Record focused and full-suite verification results before claiming completion.
