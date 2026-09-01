import sqlite3
import json
from datetime import datetime
from fastmcp import FastMCP

# Init FastMCP server
mcp = FastMCP("Snap4Events DB Server")

DB_PATH = "snap4events.db"

# Identity of an event for deduplication purposes: same title + start_datetime + venue
# is considered the same event seen again (possibly from a re-crawl of the same source).
EVENT_IDENTITY_COLUMNS = ("title", "start_datetime", "venue")


def init_db():
    """Create tables if they don't exist already, and migrate older DBs (created before
    the status/updated_at columns and the dedup unique index existed) in place."""
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

    # Migration: add columns that may be missing on a DB created by an older version
    # of this script (before status/updated_at existed).
    existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(events)").fetchall()}
    if "status" not in existing_cols:
        cursor.execute("ALTER TABLE events ADD COLUMN status TEXT DEFAULT 'new'")
    if "updated_at" not in existing_cols:
        cursor.execute("ALTER TABLE events ADD COLUMN updated_at TIMESTAMP")

    # Unique index enforcing the dedup identity (title, start_datetime, venue).
    # Needed so INSERT ... ON CONFLICT below knows what counts as "the same event".
    # A DB created before this dedup logic existed may already contain duplicate
    # rows (e.g. from repeated test crawls) - the index creation would fail on
    # those, so clean them up first, keeping the oldest row of each group.
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


# Init database
init_db()


@mcp.tool()
def save_events_to_db(events_json: str) -> str:
    """
    Saves a list of JSON-formatted event objects into the local SQLite database.

    Deduplicates on (title, start_datetime, venue): an event not seen before is
    inserted with status='new'; an event already present gets its data refreshed
    and status set to 'updated' (raw_json/category/city/source_url overwritten with
    the latest crawl, updated_at bumped). Nothing is ever deleted here.

    Args:
        events_json: A JSON string representing a list of extracted event objects.
    """
    try:
        events = json.loads(events_json)
        if not isinstance(events, list):
            events = [events]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        inserted_count = 0
        updated_count = 0
        now = datetime.now().isoformat()

        for ev in events:
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
                ev.get("venue", "N/D"),
                ev.get("city", "N/D"),
                ev.get("source_url", ""),
                json.dumps(ev),
                now,
                now
            ))
            if cursor.rowcount == 1:
                # sqlite3 reports rowcount=1 both for a fresh INSERT and for the UPDATE
                # branch of an upsert, so we can't tell them apart from rowcount alone.
                # We check afterwards whether this row was created_at == updated_at (new)
                # or not through SELECT keyed on the identity columns.
                cursor.execute(
                    "SELECT created_at, updated_at FROM events WHERE title=? AND start_datetime=? AND venue=?",
                    (ev.get("title", "Untitled"), ev.get("start_datetime"), ev.get("venue", "N/D"))
                )
                row = cursor.fetchone()
                if row and row[0] == row[1]:
                    inserted_count += 1
                else:
                    updated_count += 1

        conn.commit()
        conn.close()
        return (
            f"Saved {len(events)} events to SQLite database ({DB_PATH}): "
            f"{inserted_count} new, {updated_count} updated (deduplicated by title+start_datetime+venue)."
        )

    except Exception as e:
        return f"Error saving events to DB: {str(e)}"


@mcp.tool()
def get_events(
    city: str = None,
    category: str = None,
    date_from: str = None,
    date_to: str = None,
    status: str = None,
    limit: int = 50,
) -> str:
    """
    Retrieves events from the local SQLite database, with optional filters.

    Args:
        city: filter by exact city name (case-insensitive).
        category: filter by category (e.g. 'sport', 'culture', 'cinema', 'festival', 'concert', 'club').
        date_from: ISO date/datetime (e.g. '2026-08-31'); only events starting on/after this are returned.
        date_to: ISO date/datetime; only events starting on/before this are returned.
        status: filter by row status ('new' or 'updated').
        limit: max number of rows to return (default 50).

    Returns:
        A JSON string: a list of event objects (as stored, including id/status/timestamps).
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
        return json.dumps(results, default=str)

    except Exception as e:
        return json.dumps({"error": f"Error retrieving events from DB: {str(e)}"})


if __name__ == "__main__":
    # Run server on port 8000
    mcp.run(transport="http", port=8000)