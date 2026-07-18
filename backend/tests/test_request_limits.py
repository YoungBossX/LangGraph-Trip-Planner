import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from app.api.routes import poi
from app.models import schemas


def _request_data(**overrides):
    data = {
        "city": "Hangzhou",
        "start_date": "2026-03-01",
        "end_date": "2026-03-03",
        "travel_days": 3,
        "transportation": "public transit",
        "accommodation": "budget hotel",
        "preferences": ["history"],
        "free_text_input": "quiet pace",
    }
    data.update(overrides)
    return data


def test_trip_request_trims_bounded_text_fields():
    request = schemas.TripRequest(
        **_request_data(
            city="  Hangzhou  ",
            transportation="  transit  ",
            accommodation="  hotel  ",
            preferences=["  history  ", "  food  "],
            free_text_input="  quiet pace  ",
        )
    )

    assert request.city == "Hangzhou"
    assert request.transportation == "transit"
    assert request.accommodation == "hotel"
    assert request.preferences == ["history", "food"]
    assert request.free_text_input == "quiet pace"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("city", "x"),
        ("city", "x" * 50),
        ("transportation", "x"),
        ("transportation", "x" * 100),
        ("accommodation", "x"),
        ("accommodation", "x" * 100),
        ("free_text_input", ""),
        ("free_text_input", "x" * 1000),
    ],
)
def test_trip_request_accepts_text_field_boundaries(field_name, value):
    request = schemas.TripRequest(**_request_data(**{field_name: value}))

    assert getattr(request, field_name) == value


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("city", ""),
        ("city", "   "),
        ("city", "x" * 51),
        ("transportation", ""),
        ("transportation", "   "),
        ("transportation", "x" * 101),
        ("accommodation", ""),
        ("accommodation", "   "),
        ("accommodation", "x" * 101),
        ("free_text_input", "x" * 1001),
    ],
)
def test_trip_request_rejects_invalid_text_fields(field_name, value):
    with pytest.raises(ValidationError):
        schemas.TripRequest(**_request_data(**{field_name: value}))


def test_trip_request_accepts_preference_boundaries():
    preferences = ["x", "x" * 30, *(f"tag-{index}" for index in range(8))]

    request = schemas.TripRequest(**_request_data(preferences=preferences))

    assert request.preferences == preferences


@pytest.mark.parametrize(
    "preferences",
    [
        [f"tag-{index}" for index in range(11)],
        [""],
        ["   "],
        ["x" * 31],
    ],
)
def test_trip_request_rejects_invalid_preferences(preferences):
    with pytest.raises(ValidationError):
        schemas.TripRequest(**_request_data(preferences=preferences))


def test_trip_request_preserves_json_names_and_date_strings():
    request = schemas.TripRequest(**_request_data())

    assert set(request.model_dump()) == {
        "city",
        "start_date",
        "end_date",
        "travel_days",
        "transportation",
        "accommodation",
        "preferences",
        "free_text_input",
    }
    assert request.start_date == "2026-03-01"
    assert request.end_date == "2026-03-03"
    assert isinstance(request.start_date, str)
    assert isinstance(request.end_date, str)


def test_photo_name_type_trims_and_accepts_boundaries():
    adapter = TypeAdapter(schemas.PhotoName)

    assert adapter.validate_python("  West Lake  ") == "West Lake"
    assert adapter.validate_python("x") == "x"
    assert adapter.validate_python("x" * 100) == "x" * 100


@pytest.mark.parametrize("name", ["", "   ", "x" * 101])
def test_photo_name_type_rejects_invalid_values(name):
    adapter = TypeAdapter(schemas.PhotoName)

    with pytest.raises(ValidationError):
        adapter.validate_python(name)


class _PhotoService:
    def __init__(self):
        self.queries = []

    def get_photo_url(self, query):
        self.queries.append(query)
        return "https://example.test/photo.jpg"


def _photo_client(monkeypatch):
    service = _PhotoService()
    monkeypatch.setattr(poi, "get_unsplash_service", lambda: service)
    app = FastAPI()
    app.include_router(poi.router, prefix="/api")
    return TestClient(app), service


def test_photo_route_trims_name_before_use(monkeypatch):
    client, service = _photo_client(monkeypatch)

    response = client.get("/api/poi/photo", params={"name": "  West Lake  "})

    assert response.status_code == 200
    assert response.json()["data"]["name"] == "West Lake"
    assert service.queries == ["West Lake China landmark"]


@pytest.mark.parametrize("name", ["", "   ", "x" * 101])
def test_photo_route_rejects_invalid_name(monkeypatch, name):
    client, service = _photo_client(monkeypatch)

    response = client.get("/api/poi/photo", params={"name": name})

    assert response.status_code == 422
    assert service.queries == []
