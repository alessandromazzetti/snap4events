import pytest
from pydantic import ValidationError

from models import (
    Event,
    EventCategory,
    Coordinates,
    LocationExtraction,
    EventList,
    RetrievalDecision,
)


def test_event_valid_minimal():
    ev = Event(
        title="Concerto",
        category="concert",
        start_datetime="2026-09-10T20:00:00",
        venue="Teatro Verdi",
        city="Firenze",
        source_url="https://example.com/evento",
    )

    assert ev.title == "Concerto"
    assert ev.category == EventCategory.CONCERT
    assert ev.description is None
    assert ev.coordinates is None
    assert ev.image_urls == []


def test_event_missing_required_field_raises():
    with pytest.raises(ValidationError):
        Event(
            category="concert",
            start_datetime="2026-09-10T20:00:00",
            venue="Teatro Verdi",
            city="Firenze",
            source_url="https://example.com/evento",
            # title missing
        )


def test_event_invalid_category_raises():
    with pytest.raises(ValidationError):
        Event(
            title="Concerto",
            category="not-a-real-category",
            start_datetime="2026-09-10T20:00:00",
            venue="Teatro Verdi",
            city="Firenze",
            source_url="https://example.com/evento",
        )


def test_event_from_km4city_feature_basic_mapping():
    feature = {
        "properties": {
            "name": "Sagra del Tartufo",
            "category": "festival",
            "dateStart": "2026-10-01T18:00:00Z",
            "address": "Piazza Grande",
            "municipality": "Firenze",
            "organizer": "Comune di Firenze",
            "uri": "https://km4city.org/events/1",
        },
        "geometry": {
            "coordinates": [11.2558, 43.7696],  # [lon, lat] as in GeoJSON
        },
    }

    ev = Event.from_km4city_feature(feature)

    assert ev.title == "Sagra del Tartufo"
    assert ev.category == EventCategory.FESTIVAL
    assert ev.venue == "Piazza Grande"
    assert ev.city == "Firenze"
    assert ev.organizer == "Comune di Firenze"
    assert ev.coordinates == Coordinates(latitude=43.7696, longitude=11.2558)


def test_event_from_km4city_feature_missing_fields_use_defaults():
    feature = {"properties": {}, "geometry": {}}

    ev = Event.from_km4city_feature(feature)

    assert ev.title == "Evento senza titolo"
    assert ev.category == EventCategory.OTHER
    assert ev.venue == "N/D"
    assert ev.city == "N/D"
    assert ev.coordinates is None
    assert str(ev.source_url).startswith("https://www.km4city.org")


def test_event_from_km4city_feature_invalid_date_falls_back_without_raising():
    feature = {
        "properties": {"name": "Evento", "dateStart": "not-a-date"},
        "geometry": {},
    }

    # Must not raise even with a garbage date string
    ev = Event.from_km4city_feature(feature)
    assert ev.start_datetime is not None


@pytest.mark.parametrize(
    "raw_category,expected",
    [
        ("live music concert", EventCategory.CONCERT),
        ("sports match", EventCategory.SPORT),
        ("night club party", EventCategory.CLUB),
        ("something totally unrelated", EventCategory.OTHER),
    ],
)
def test_event_from_km4city_feature_category_keyword_matching(raw_category, expected):
    feature = {
        "properties": {"name": "Evento", "category": raw_category},
        "geometry": {},
    }

    ev = Event.from_km4city_feature(feature)
    assert ev.category == expected


def test_location_extraction_model():
    loc = LocationExtraction(target_location="Firenze", latitude=43.7697, longitude=11.2558)
    assert loc.target_location == "Firenze"


def test_event_list_defaults_to_empty():
    events = EventList()
    assert events.events == []


def test_retrieval_decision_defaults_reasoning_to_empty_string():
    decision = RetrievalDecision(use_retrieved_data=True)
    assert decision.reasoning == ""
    assert decision.use_retrieved_data is True
