"""
This server exposes all of the agent's funzionalities as independent MCP tools.

The server remains on the same port already used by the project
(http://localhost:8000/mcp).
"""

import json
import os
import re
import sqlite3
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from pydantic import ValidationError
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from models import Event
from token_manager import TokenManager
from km4city_client import KM4CityClient

mcp = FastMCP("Snap4Events Tools")

DB_PATH = "snap4events.db"
PDF_OUTPUT_DIR = "pdf_events"

# Columns that uniquely identify an event for deduplication purposes.
EVENT_IDENTITY_COLUMNS = ("title", "start_datetime", "venue")


# ---------------------------------------------------------------------------
# Database setup (same schema already present in db_mcp_server.py)
# ---------------------------------------------------------------------------

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT,
            start_datetime TEXT,
            venue TEXT,
            city TEXT,
            source_url TEXT,
            raw_json TEXT,
            status TEXT DEFAULT 'new',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(events)").fetchall()}
    if "status" not in existing_cols:
        cursor.execute("ALTER TABLE events ADD COLUMN status TEXT DEFAULT 'new'")
    if "updated_at" not in existing_cols:
        cursor.execute("ALTER TABLE events ADD COLUMN updated_at TIMESTAMP")

    identity_cols = ", ".join(EVENT_IDENTITY_COLUMNS)
    try:
        cursor.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_events_identity
            ON events ({identity_cols})
        """)
    except sqlite3.IntegrityError:
        cursor.execute(f"""
            DELETE FROM events WHERE id NOT IN (
                SELECT MIN(id) FROM events GROUP BY {identity_cols}
            )
        """)
        cursor.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_events_identity
            ON events ({identity_cols})
        """)

    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Km4City client: credential loading
# ---------------------------------------------------------------------------

_km4city_client: Optional[KM4CityClient] = None


def get_km4city_client() -> KM4CityClient:
    global _km4city_client

    if _km4city_client is not None:
        return _km4city_client

    username = os.getenv("SNAP4CITY_USERNAME")
    password = os.getenv("SNAP4CITY_PASSWORD")

    if not username or not password:
        creds_path = "user_credentials.json"
        if os.path.exists(creds_path):
            with open(creds_path) as f:
                creds = json.load(f)
            username = username or creds.get("username")
            password = password or creds.get("password")

    token_manager = TokenManager(username, password) if username and password else None
    _km4city_client = KM4CityClient(token_manager=token_manager)
    return _km4city_client


# ---------------------------------------------------------------------------
# TOOL 1 - Web search
# ---------------------------------------------------------------------------

@mcp.tool()
def search_web(query: str, max_results: int = 4) -> str:
    """
    Searches the web for pages relevant to a text query, using Tavily.

    Use this tool when you need to discover source URLs (event websites,
    venue pages, press releases, etc.) about a topic or
    location, for example before extracting events from fresh sources
    because the data already present in the database (see get_events) is
    missing, insufficient, or outdated.

    Args:
        query: natural-language search query
               (e.g. "concert events Florence this weekend").
        max_results: maximum number of results to return (default 4).

    Returns:
        A JSON string: list of objects {"title", "url", "content"}.
    """
    from tavily import TavilyClient

    try:
        client = TavilyClient()
        response = client.search(query=query, search_depth="basic", max_results=max_results)
        results = [
            {
                "title": res.get("title", ""),
                "url": res.get("url", ""),
                "content": res.get("content", ""),
            }
            for res in response.get("results", [])
        ]
        return json.dumps(results, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Error during web search: {e}"})


# ---------------------------------------------------------------------------
# TOOL 2 - Extract structured data
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_page_text(url: str, max_chars: int = 15000) -> str:
    """
    Downloads a web page and extracts only the visible text (without HTML
    tags, scripts, or CSS).

    Use this tool on a URL obtained from search_web (or provided
    by the user) to read the page content and, by reasoning over the
    returned text, identify the events it contains yourself: title,
    category, date/time, venue, city. There is no tool that performs
    the extraction automatically: the model must identify the events by
    reading the output of this tool, and can then optionally save the
    events found using save_events_to_db.

    Args:
        url: page URL to download.
        max_chars: maximum number of text characters to return
                   (default 15000, to avoid exceeding context limits).

    Returns:
        The extracted page text (plain string), or an
        error message if the download fails.
    """
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            },
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        text_content = soup.get_text(separator=" ", strip=True)[:max_chars]
        return text_content

    except Exception as e:
        return f"Error while scraping {url}: {e}"


# ---------------------------------------------------------------------------
# TOOL 3 - Geocoding a location name
# ---------------------------------------------------------------------------

@mcp.tool()
def geocode_location(location_name: str) -> str:
    """
    Resolves the name of an Italian city, municipality, or point of interest
    into its geographic coordinates (latitude/longitude), using the public
    Nominatim geocoding service (OpenStreetMap).

    Use this tool whenever you need a precise geographic position
    to query the Km4City APIs (search_km4city_events) starting from a
    location name mentioned by the user, instead of estimating or
    guessing the coordinates.

    Args:
        location_name: name of the location to geocode
                        (e.g. "Florence", "Piazza del Duomo, Milan").

    Returns:
        A JSON string: {"location_name", "resolved_name", "latitude",
        "longitude"}, or {"error": ...} if the location is not found.
    """
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location_name, "format": "json", "limit": 1, "countrycodes": "it"},
            headers={"User-Agent": "snap4events-agent/1.0"},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json()

        if not results:
            return json.dumps({"error": f"No match found for '{location_name}'."})

        best = results[0]
        return json.dumps({
            "location_name": location_name,
            "resolved_name": best.get("display_name", location_name),
            "latitude": float(best["lat"]),
            "longitude": float(best["lon"]),
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Error during geocoding: {e}"})


# ---------------------------------------------------------------------------
# TOOL 4 - Geocoding through Km4City API
# ---------------------------------------------------------------------------

@mcp.tool()
def search_km4city_events(
    latitude: float,
    longitude: float,
    range_: str = "week",
    max_dists: float = 10.0,
    max_results: int = 100,
) -> str:
    """
    Searches for official geolocated events through the Km4City/Snap4City
    Advanced Smart City APIs (endpoint /events/), centered on a geographic
    point.

    Use this tool as the PRIMARY and most reliable source of structured
    events for an area (alternative/complementary to web search +
    scraping), when you already have coordinates (e.g. from
    geocode_location).

    Args:
        latitude: latitude of the search center.
        longitude: longitude of the search center.
        range_: time window, one of "day", "week", "month"
                (default "week").
        max_dists: maximum search radius in km (default 10.0).
        max_results: maximum number of results (default 100).

    Returns:
        A JSON string containing the raw GeoJSON returned by the API, or
        {"error": ...} in case of problems (including missing credentials,
        in which case the request is still attempted without a token).
    """
    try:
        client = get_km4city_client()
        data = client.search_events(
            lat=latitude,
            lon=longitude,
            range_=range_,
            max_dists=max_dists,
            max_results=max_results,
        )
        return json.dumps(data, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Error while querying Km4City: {e}"})


# ---------------------------------------------------------------------------
# TOOL 5 - Save events to the DB
# ---------------------------------------------------------------------------

@mcp.tool()
def save_events_to_db(events_json: str) -> str:
    """
    Validates and saves a list of events (JSON objects) to the local SQLite
    database.

    Use this tool to persist events that you have identified — either
    those found by reasoning over the text returned by fetch_page_text,
    or those obtained from search_km4city_events after converting them to
    the expected schema. Each event must comply with the schema:
    {"title": str, "category": one of "sport"/"culture"/"cinema"/
    "festival"/"concert"/"club"/"other", "start_datetime": ISO 8601,
    "venue": str, "city": str, "source_url": URL, "expected_reach": int|null}.

    Deduplication is performed on (title, start_datetime, venue): an event
    that has not been seen before is inserted with status='new'; an event
    that is already present is updated with status='updated'. Nothing is
    ever deleted.

    Args:
        events_json: JSON string containing an event object or a list of
                     event objects.

    Returns:
        A textual summary message (how many new, how many
        updated, and any events discarded because they were invalid).
    """
    try:
        raw_events = json.loads(events_json)
        if not isinstance(raw_events, list):
            raw_events = [raw_events]

        valid_events = []
        skipped = 0

        for raw_event in raw_events:
            try:
                validated = Event(**raw_event)
                valid_events.append(json.loads(validated.model_dump_json()))
            except ValidationError as ve:
                skipped += 1
                print(f"[save_events_to_db] Event discarded (invalid): {ve}")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        inserted_count = 0
        updated_count = 0
        now = datetime.now().isoformat()

        for ev in valid_events:
            cursor.execute("""
                INSERT INTO events (title, category, start_datetime, venue, city, source_url, raw_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)
                ON CONFLICT(title, start_datetime, venue) DO UPDATE SET
                    category=excluded.category,
                    city=excluded.city,
                    source_url=excluded.source_url,
                    raw_json=excluded.raw_json,
                    status='updated',
                    updated_at=excluded.updated_at
            """, (
                ev.get("title", "Untitled"),
                ev.get("category", "other"),
                ev.get("start_datetime"),
                ev.get("venue", "N/A"),
                ev.get("city", "N/A"),
                ev.get("source_url", ""),
                json.dumps(ev, ensure_ascii=False),
                now,
                now,
            ))

            cursor.execute(
                "SELECT created_at, updated_at FROM events WHERE title=? AND start_datetime=? AND venue=?",
                (ev.get("title", "Untitled"), ev.get("start_datetime"), ev.get("venue", "N/A")),
            )
            row = cursor.fetchone()
            if row and row[0] == row[1]:
                inserted_count += 1
            else:
                updated_count += 1

        conn.commit()
        conn.close()

        return (
            f"Saved {len(valid_events)} events to {DB_PATH}: "
            f"{inserted_count} new, {updated_count} updated "
            f"(deduplication on title+start_datetime+venue). "
            f"{skipped} events discarded because they did not conform to the schema."
        )

    except Exception as e:
        return f"Error while saving events: {e}"


# ---------------------------------------------------------------------------
# TOOL 6 - Read events from the DB
# ---------------------------------------------------------------------------

@mcp.tool()
def get_events(
    city: Optional[str] = None,
    category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> str:
    """
    Retrieves events already present in the local SQLite database, with
    optional filters.

    ALWAYS use this tool first, before starting a web search,
    to check whether sufficient and up-to-date data already exists for the
    location/category of interest: if the result is empty, sparse, or
    obviously outdated, proceed with search_web /
    search_km4city_events to retrieve fresh data and then save it with
    save_events_to_db.

    Args:
        city: filter by exact city name (case-insensitive).
        category: filter by category ("sport", "culture", "cinema",
                  "festival", "concert", "club", "other").
        date_from: minimum ISO date/datetime (e.g. "2026-09-07").
        date_to: maximum ISO date/datetime.
        status: filter by row status ("new" or "updated").
        limit: maximum number of rows to return (default 50).

    Returns:
        A JSON string: list of event objects (including id/status/
        row timestamps).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM events WHERE 1=1"
        params = []
        if city:
            query += " AND LOWER(city) = LOWER(?)"
            params.append(city)
        if category:
            query += " AND LOWER(category) = LOWER(?)"
            params.append(category)
        if date_from:
            query += " AND start_datetime >= ?"
            params.append(date_from)
        if date_to:
            query += " AND start_datetime <= ?"
            params.append(date_to)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY start_datetime ASC LIMIT ?"
        params.append(limit)

        rows = cursor.execute(query, params).fetchall()
        conn.close()

        results = [dict(row) for row in rows]
        return json.dumps(results, default=str, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Error while retrieving events: {e}"})


# ---------------------------------------------------------------------------
# TOOL 7 - Event PDF export
# ---------------------------------------------------------------------------

def _safe_filename(text: str) -> str:
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:100]


@mcp.tool()
def export_event_pdf(event_json: str) -> str:
    """
    Generates a PDF document containing the details of a single event
    (title, category, date, time, venue, city, source URL, expected reach) using ReportLab.

    Use this tool when the user explicitly requests a downloadable document/
    event sheet for one or more events, typically after they have already
    been identified (and possibly saved with save_events_to_db).

    Args:
        event_json: JSON string containing the event fields, using the same
                    schema as save_events_to_db.

    Returns:
        The path of the generated PDF file, or an error message.
    """
    try:
        raw_event = json.loads(event_json)
        event = Event(**raw_event)

        os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)
        filename = (
            f"{_safe_filename(event.title)}_"
            f"{event.start_datetime.strftime('%Y%m%d_%H%M')}.pdf"
        )
        filepath = os.path.join(PDF_OUTPUT_DIR, filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "EventTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=15,
        )

        story = [Paragraph(event.title, title_style), Spacer(1, 10)]

        data = [
            ["Category", event.category],
            ["Date", event.start_datetime.strftime("%d/%m/%Y")],
            ["Time", event.start_datetime.strftime("%H:%M")],
            ["Venue", event.venue],
            ["City", event.city],
            ["URL", str(event.source_url)],
            ["Expected reach", str(event.expected_reach)],
        ]

        table = Table(data, colWidths=[40 * mm, 120 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(table)

        doc.build(story)
        return filepath

    except Exception as e:
        return f"Error while generating the PDF: {e}"


if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
