import pytest

from app.models.schemas import (
    Attraction,
    DayPlan,
    Hotel,
    Location,
    Meal,
    TripPlan,
    TripRequest,
    WeatherInfo,
)
from app.workflows.trip_planner_graph import TripPlannerWorkflow

REQUESTED_DATES = ["2026-03-01", "2026-03-02", "2026-03-03"]


@pytest.fixture()
def workflow():
    return TripPlannerWorkflow.__new__(TripPlannerWorkflow)


def _request() -> TripRequest:
    return TripRequest(
        city="Hangzhou",
        start_date=REQUESTED_DATES[0],
        end_date=REQUESTED_DATES[-1],
        travel_days=len(REQUESTED_DATES),
        transportation="public transit",
        accommodation="budget hotel",
    )


def _source_attractions() -> list[Attraction]:
    return [
        Attraction(
            name="West Lake",
            address="1 Source Road",
            location=Location(longitude=120.1, latitude=30.2),
            visit_duration=180,
            description="Authoritative lake description",
            category="scenic",
            rating=4.8,
            photos=["west-lake.jpg"],
            poi_id="poi-west",
            image_url="west-lake-cover.jpg",
            ticket_price=15,
            price_text="15 CNY",
        ),
        Attraction(
            name="Lingyin Temple",
            address="2 Source Road",
            location=Location(longitude=120.2, latitude=30.3),
            visit_duration=120,
            description="Authoritative temple description",
            category="culture",
            rating=4.7,
            photos=["lingyin.jpg"],
            poi_id="poi-lingyin",
            image_url="lingyin-cover.jpg",
            ticket_price=30,
            price_text="30 CNY",
        ),
    ]


def _source_weather() -> list[WeatherInfo]:
    return [
        WeatherInfo(
            date="2026-03-03",
            day_weather="sunny",
            night_weather="clear",
            day_temp=21,
            night_temp=11,
            wind_direction="south",
            wind_power="1",
        ),
        WeatherInfo(
            date="2026-03-01",
            day_weather="cloudy",
            night_weather="clear",
            day_temp=18,
            night_temp=8,
            wind_direction="east",
            wind_power="2",
        ),
        WeatherInfo(
            date="2026-03-02",
            day_weather="rain",
            night_weather="rain",
            day_temp=16,
            night_temp=7,
            wind_direction="north",
            wind_power="3",
        ),
    ]


def _source_hotels() -> list[Hotel]:
    return [
        Hotel(
            name="Lake Hotel",
            address="9 Source Hotel Road",
            location=Location(longitude=120.3, latitude=30.4),
            price_range="400-600",
            rating=4.6,
            distance="1 km",
            type="boutique",
            estimated_cost=500,
        )
    ]


def _planner_attraction(name: str, poi_id: str | None = None) -> Attraction:
    return Attraction(
        name=name,
        address="Planner invented attraction address",
        location=Location(longitude=1.0, latitude=2.0),
        visit_duration=5,
        description="Planner invented attraction description",
        category="invented",
        rating=1.0,
        photos=["planner.jpg"],
        poi_id=poi_id,
        image_url="planner-cover.jpg",
        ticket_price=999,
        price_text="planner price",
    )


def _planner_hotel(name: str = "  lake   HOTEL ") -> Hotel:
    return Hotel(
        name=name,
        address="Planner invented hotel address",
        location=Location(longitude=3.0, latitude=4.0),
        price_range="1-2",
        rating=1.2,
        distance="far away",
        type="invented",
        estimated_cost=1,
    )


def _meals() -> list[Meal]:
    return [
        Meal(type=" Breakfast ", name="Breakfast"),
        Meal(type="LUNCH", name="Lunch"),
        Meal(type=" dinner ", name="Dinner"),
        Meal(type="SnAcK", name="Snack"),
    ]


def _valid_plan() -> TripPlan:
    attraction_references = [
        _planner_attraction("  WEST   lake "),
        _planner_attraction("planner supplied name", poi_id="poi-lingyin"),
        _planner_attraction("West Lake"),
    ]
    days = [
        DayPlan(
            date=day_date,
            day_index=day_index,
            description=f"Day {day_index + 1}",
            transportation="public transit",
            accommodation="budget hotel",
            hotel=_planner_hotel(),
            attractions=[attraction_references[day_index]],
            meals=_meals(),
        )
        for day_index, day_date in enumerate(REQUESTED_DATES)
    ]
    return TripPlan(
        city="  Hangzhou  ",
        start_date=REQUESTED_DATES[0],
        end_date=REQUESTED_DATES[-1],
        days=days,
        weather_info=[
            WeatherInfo(
                date="2026-03-01",
                day_weather="planner storm",
                night_weather="planner storm",
                day_temp=99,
                night_temp=99,
            )
        ],
        overall_suggestions="Planner arrangement",
    )


def _validate(workflow: TripPlannerWorkflow, plan: TripPlan, *, attractions=None, weather=None, hotels=None):
    return workflow._validate_and_canonicalize_trip_plan(
        plan,
        _request(),
        attractions if attractions is not None else _source_attractions(),
        weather if weather is not None else _source_weather(),
        hotels if hotels is not None else _source_hotels(),
    )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("city", "Suzhou", "plan city"),
        ("start_date", "2026-03-02", "plan start_date"),
        ("end_date", "2026-03-04", "plan end_date"),
    ],
)
def test_rejects_wrong_plan_identity_fields(workflow, field, value, reason):
    plan = _valid_plan()
    setattr(plan, field, value)

    with pytest.raises(ValueError, match=reason):
        _validate(workflow, plan)


def test_rejects_wrong_day_count(workflow):
    plan = _valid_plan()
    plan.days.pop()

    with pytest.raises(ValueError, match="day count"):
        _validate(workflow, plan)


@pytest.mark.parametrize(
    "dates",
    [
        ["2026-03-02", "2026-03-01", "2026-03-03"],
        ["2026-03-01", "2026-03-02", "2026-03-04"],
    ],
    ids=["out-of-order", "not-requested-sequence"],
)
def test_rejects_day_dates_that_do_not_match_exact_request_sequence(workflow, dates):
    plan = _valid_plan()
    for day, day_date in zip(plan.days, dates, strict=True):
        day.date = day_date

    with pytest.raises(ValueError, match="day dates"):
        _validate(workflow, plan)


@pytest.mark.parametrize("indexes", [[1, 2, 3], [0, 2, 3]], ids=["not-zero-based", "noncontiguous"])
def test_rejects_invalid_day_indexes(workflow, indexes):
    plan = _valid_plan()
    for day, day_index in zip(plan.days, indexes, strict=True):
        day.day_index = day_index

    with pytest.raises(ValueError, match="day_index"):
        _validate(workflow, plan)


def test_rejects_day_without_attractions(workflow):
    plan = _valid_plan()
    plan.days[0].attractions = []

    with pytest.raises(ValueError, match="at least one attraction"):
        _validate(workflow, plan)


def test_rejects_duplicate_source_attraction_within_a_day(workflow):
    plan = _valid_plan()
    plan.days[0].attractions.append(_planner_attraction("different planner name", poi_id="poi-west"))

    with pytest.raises(ValueError, match="duplicate attraction"):
        _validate(workflow, plan)


def test_rejects_unknown_attraction(workflow):
    plan = _valid_plan()
    plan.days[0].attractions = [_planner_attraction("Unknown Place")]

    with pytest.raises(ValueError, match="unknown attraction"):
        _validate(workflow, plan)


def test_rejects_missing_hotel(workflow):
    plan = _valid_plan()
    plan.days[0].hotel = None

    with pytest.raises(ValueError, match="missing hotel"):
        _validate(workflow, plan)


def test_rejects_unknown_hotel(workflow):
    plan = _valid_plan()
    plan.days[0].hotel = _planner_hotel("Unknown Hotel")

    with pytest.raises(ValueError, match="unknown hotel"):
        _validate(workflow, plan)


def test_rejects_missing_required_meal(workflow):
    plan = _valid_plan()
    plan.days[0].meals = [meal for meal in plan.days[0].meals if meal.type.strip().casefold() != "breakfast"]

    with pytest.raises(ValueError, match="missing required meal"):
        _validate(workflow, plan)


def test_rejects_duplicate_required_meal(workflow):
    plan = _valid_plan()
    plan.days[0].meals.append(Meal(type="BREAKFAST", name="Second breakfast"))

    with pytest.raises(ValueError, match="duplicate required meal"):
        _validate(workflow, plan)


def test_rejects_unknown_meal_type(workflow):
    plan = _valid_plan()
    plan.days[0].meals[0].type = "brunch"

    with pytest.raises(ValueError, match="unknown meal type"):
        _validate(workflow, plan)


def test_normalized_entity_names_trim_collapse_whitespace_and_casefold(workflow):
    assert workflow._normalize_entity_name("  West\t LAKE \n") == "west lake"

    attractions = _source_attractions()
    attractions[0].name = "  WEST   Lake "
    attractions[1].name = " LINGYIN\tTEMPLE "
    hotels = _source_hotels()
    hotels[0].name = "  LAKE   hotel "

    validated = _validate(workflow, _valid_plan(), attractions=attractions, hotels=hotels)

    assert validated.days[0].attractions[0].name == "  WEST   Lake "
    assert validated.days[0].hotel.name == "  LAKE   hotel "


def test_attraction_poi_id_match_takes_precedence_over_name(workflow):
    plan = _valid_plan()
    plan.days[0].attractions = [_planner_attraction("Lingyin Temple", poi_id="poi-west")]

    validated = _validate(workflow, plan)

    assert validated.days[0].attractions[0].name == "West Lake"


def test_unknown_supplied_poi_id_does_not_fall_back_to_matching_name(workflow):
    plan = _valid_plan()
    plan.days[0].attractions = [_planner_attraction("West Lake", poi_id="unknown-poi")]

    with pytest.raises(ValueError, match="unknown attraction POI ID"):
        _validate(workflow, plan)


def test_rejects_ambiguous_normalized_source_attraction_names(workflow):
    attractions = _source_attractions()
    attractions.append(
        Attraction(name=" west   LAKE ", address="Duplicate", poi_id="poi-other", location=None)
    )

    with pytest.raises(ValueError, match="ambiguous source attraction name"):
        _validate(workflow, _valid_plan(), attractions=attractions)


def test_rejects_ambiguous_source_attraction_poi_ids(workflow):
    attractions = _source_attractions()
    attractions[1].poi_id = attractions[0].poi_id

    with pytest.raises(ValueError, match="ambiguous source attraction POI ID"):
        _validate(workflow, _valid_plan(), attractions=attractions)


def test_rejects_ambiguous_normalized_source_hotel_names(workflow):
    hotels = _source_hotels()
    hotels.append(Hotel(name=" lake   HOTEL ", address="Duplicate"))

    with pytest.raises(ValueError, match="ambiguous source hotel name"):
        _validate(workflow, _valid_plan(), hotels=hotels)


def test_canonicalizes_source_entities_meals_and_weather(workflow):
    plan = _valid_plan()
    attractions = _source_attractions()
    weather = _source_weather()
    hotels = _source_hotels()

    validated = _validate(workflow, plan, attractions=attractions, weather=weather, hotels=hotels)

    assert validated.city == _request().city
    assert validated.days[0].attractions[0].model_dump() == attractions[0].model_dump()
    assert validated.days[0].attractions[0] is not attractions[0]
    assert validated.days[0].hotel.model_dump() == hotels[0].model_dump()
    assert validated.days[0].hotel is not hotels[0]
    assert [meal.type for meal in validated.days[0].meals] == ["breakfast", "lunch", "dinner", "snack"]

    ordered_weather = sorted(weather, key=lambda item: item.date)
    assert [item.model_dump() for item in validated.weather_info] == [item.model_dump() for item in ordered_weather]
    assert all(actual is not source for actual, source in zip(validated.weather_info, ordered_weather, strict=True))


def test_returned_plan_is_deep_copied_from_parsed_plan_and_sources(workflow):
    plan = _valid_plan()
    attractions = _source_attractions()
    weather = _source_weather()
    hotels = _source_hotels()
    original_plan = plan.model_dump()
    original_attractions = [item.model_dump() for item in attractions]
    original_weather = [item.model_dump() for item in weather]
    original_hotels = [item.model_dump() for item in hotels]

    validated = _validate(workflow, plan, attractions=attractions, weather=weather, hotels=hotels)
    validated.days[0].description = "Changed after validation"
    validated.days[0].meals[0].name = "Changed meal"
    validated.days[0].attractions[0].address = "Changed attraction"
    validated.days[0].attractions[0].location.longitude = 0
    validated.days[0].hotel.address = "Changed hotel"
    validated.weather_info[0].day_weather = "Changed weather"

    assert plan.model_dump() == original_plan
    assert [item.model_dump() for item in attractions] == original_attractions
    assert [item.model_dump() for item in weather] == original_weather
    assert [item.model_dump() for item in hotels] == original_hotels


def test_planner_query_includes_attraction_poi_id(workflow):
    query = workflow._build_planner_query(_request(), _source_attractions(), _source_weather(), _source_hotels())

    assert '"poi_id": "poi-west"' in query
