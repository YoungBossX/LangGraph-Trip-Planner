import pytest
from pydantic import ValidationError

from app.models.schemas import TripRequest


def _request_data(**overrides):
    data = {
        "city": "Hangzhou",
        "start_date": "2026-03-01",
        "end_date": "2026-03-03",
        "travel_days": 3,
        "transportation": "public transit",
        "accommodation": "budget hotel",
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    ("overrides", "error_text"),
    [
        ({"start_date": "2026/03/01"}, "dates must use YYYY-MM-DD format"),
        ({"end_date": "2026-02-28"}, "end_date must be on or after start_date"),
        ({"end_date": "2026-03-04"}, "inclusive date range must equal travel_days"),
    ],
)
def test_trip_request_rejects_invalid_calendar(overrides, error_text):
    with pytest.raises(ValidationError, match=error_text):
        TripRequest(**_request_data(**overrides))


def test_trip_request_accepts_inclusive_range_across_leap_day():
    request = TripRequest(
        **_request_data(
            start_date="2024-02-28",
            end_date="2024-03-01",
            travel_days=3,
        )
    )

    assert request.start_date == "2024-02-28"
    assert request.end_date == "2024-03-01"
    assert request.travel_days == 3
