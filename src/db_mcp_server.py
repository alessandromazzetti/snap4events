import sqlite3
import json
from fastmcp import FastMCP

# Init FastMCP server
mcp = FastMCP("Snap4Events DB Server")

DB_PATH = "snap4events.db"


def init_db():
    """Create tables if does not exist already."""
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# Init database
init_db()


@mcp.tool()
def save_events_to_db(events_json: str) -> str:
    """
    Saves a list of JSON-formatted event objects into the local SQLite database.

    Args:
        events_json: A JSON string representing a list of extracted event objects.
    """
    try:
        events = json.loads(events_json)
        if not isinstance(events, list):
            events = [events]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        saved_count = 0

        for ev in events:
            cursor.execute("""
                INSERT INTO events (title, category, start_datetime, venue, city, source_url, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ev.get("title", "Untitled"),
                ev.get("category", "other"),
                ev.get("start_datetime"),
                ev.get("venue", "N/D"),
                ev.get("city", "N/D"),
                ev.get("source_url", ""),
                json.dumps(ev)
            ))
            saved_count += 1

        conn.commit()
        conn.close()
        return f"Successfully saved {saved_count} events to SQLite database ({DB_PATH})."

    except Exception as e:
        return f"Error saving events to DB: {str(e)}"


if __name__ == "__main__":
    # Run server on port 8000
    mcp.run(transport="http", port=8000)