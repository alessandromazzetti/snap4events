from __future__ import annotations
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List
from datetime import datetime
from enum import Enum


# Event categories
class EventCategory(str, Enum):
    SPORT = "sport"
    CULTURE = "culture"
    CINEMA = "cinema"
    FESTIVAL = "festival"
    CONCERT = "concert"
    CLUB = "club"
    OTHER = "other"


# Geographical coordinates
class Coordinates(BaseModel):
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")


# Event model
class Event(BaseModel):
    title: str = Field(..., description="Event title")
    description: Optional[str] = Field(None, description="Event description")
    category: EventCategory = Field(..., description="Event category")
    start_datetime: datetime = Field(..., description="Event datetime")
    venue: str = Field(..., description="Event venue")
    city: str = Field(..., description="City event")
    coordinates: Optional[Coordinates] = Field(None, description="Event coordinates (optional)")
    organizer: Optional[str] = Field(None, description="Event organizer")
    source_url: HttpUrl = Field(..., description="Event source URL")
    image_urls: List[HttpUrl] = Field(default_factory=list, description="Event image URLs")

    @classmethod
    def from_km4city_feature(cls, feature: dict) -> "Event":
        """
        Converte una Feature GeoJSON restituita dall'endpoint /events/ delle
        Km4City/Snap4City Advanced Smart City API in un'istanza di Event.

        NOTA IMPORTANTE: lo swagger pubblico referenzia lo schema
        'EventsJsonDocument' senza pubblicarne i dettagli campo per campo.
        I nomi qui sotto (name, category, dateStart, address, municipality, ecc.)
        sono quelli tipici usati da Km4City per i servizi/eventi geolocalizzati,
        ma vanno verificati contro una risposta reale prima di andare in
        produzione: stampa `json.dumps(feature, indent=2)` per un evento reale
        e allinea le chiavi usate qui sotto se necessario.
        """
        props = feature.get("properties", {}) or {}
        geometry = feature.get("geometry", {}) or {}

        title = props.get("name") or props.get("title") or "Evento senza titolo"

        raw_category = (
            props.get("category") or props.get("nature") or props.get("subnature") or ""
        ).lower()
        category_keywords = {
            "sport": EventCategory.SPORT,
            "cultur": EventCategory.CULTURE,
            "cinema": EventCategory.CINEMA,
            "festival": EventCategory.FESTIVAL,
            "sagra": EventCategory.FESTIVAL,
            "concert": EventCategory.CONCERT,
            "club": EventCategory.CLUB,
        }
        category = next(
            (v for k, v in category_keywords.items() if k in raw_category),
            EventCategory.OTHER,
        )

        raw_datetime = props.get("dateStart") or props.get("date_time") or props.get("startTime")
        try:
            start_datetime = (
                datetime.fromisoformat(str(raw_datetime).replace("Z", "+00:00"))
                if raw_datetime
                else datetime.now()
            )
        except ValueError:
            start_datetime = datetime.now()

        coordinates = None
        coords = geometry.get("coordinates")
        if coords and len(coords) >= 2:
            # GeoJSON usa l'ordine [longitude, latitude]
            coordinates = Coordinates(latitude=coords[1], longitude=coords[0])

        source_url = props.get("uri") or props.get("serviceUri") or "https://www.km4city.org/"

        return cls(
            title=title,
            description=props.get("description"),
            category=category,
            start_datetime=start_datetime,
            venue=props.get("address") or props.get("venue") or "N/D",
            city=props.get("municipality") or props.get("city") or "N/D",
            coordinates=coordinates,
            organizer=props.get("organizer"),
            source_url=source_url,
            image_urls=props.get("images", []) or [],
        )
