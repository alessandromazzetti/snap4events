"""
Tests for db_mcp_server.py, run against fastmcp's in-process Client
(Client(mcp_app) directly, no HTTP server, no port 8000 needed) and a
throwaway SQLite file per test, so the real snap4events.db is never touched.
"""

import json
import tempfile
from pathlib import Path

import pytest
from fastmcp import Client

import db_mcp_server


@pytest.fixture
def mcp_client(tmp_path, monkeypatch):
    """Points db_mcp_server at a fresh temp DB file for the duration of one
    test, then hands back an in-process Client connected to its FastMCP app."""

    db_path = str(tmp_path / "test_events.db")
    monkeypatch.setattr(db_mcp_server, "DB_PATH", db_path)
    db_mcp_server.init_db()

    return Client(db_mcp_server.mcp)


def _event(**overrides):
    base = {
        "title": "Test",
        "category": "concert",
        "start_datetime": "2026-09-10T20:00:00",
        "venue": "Sala Test",
        "city": "Firenze",
        "source_url": "https://example.com/event",
    }
    base.update(overrides)
    return base


async def test_save_and_get_events_roundtrip(mcp_client):
    async with mcp_client as client:
        save_result = await client.call_tool(
            "save_events_to_db",
            {"events_json": json.dumps([_event()])},
        )
        assert "1 new" in save_result.data

        get_result = await client.call_tool("get_events", {"city": "Firenze"})
        events = json.loads(get_result.data)

        assert len(events) == 1
        assert events[0]["title"] == "Test"
        assert events[0]["status"] == "new"


async def test_dedup_upsert_updates_existing_event_instead_of_duplicating(mcp_client):
    async with mcp_client as client:
        await client.call_tool(
            "save_events_to_db",
            {"events_json": json.dumps([_event(category="concert")])},
        )

        # Same identity (title + start_datetime + venue), different category
        second_save = await client.call_tool(
            "save_events_to_db",
            {"events_json": json.dumps([_event(category="festival")])},
        )
        assert "1 updated" in second_save.data

        get_result = await client.call_tool("get_events", {"city": "Firenze"})
        events = json.loads(get_result.data)

        assert len(events) == 1  # still one row, not two
        assert events[0]["category"] == "festival"
        assert events[0]["status"] == "updated"


async def test_get_events_filters_by_city(mcp_client):
    async with mcp_client as client:
        await client.call_tool("save_events_to_db", {"events_json": json.dumps([
            _event(title="Evento Firenze", city="Firenze"),
            _event(title="Evento Milano", city="Milano", venue="Altro Locale"),
        ])})

        result = await client.call_tool("get_events", {"city": "Milano"})
        events = json.loads(result.data)

        assert len(events) == 1
        assert events[0]["title"] == "Evento Milano"


async def test_get_events_filters_by_category(mcp_client):
    async with mcp_client as client:
        await client.call_tool("save_events_to_db", {"events_json": json.dumps([
            _event(title="Concerto", category="concert"),
            _event(title="Partita", category="sport", venue="Stadio"),
        ])})

        result = await client.call_tool("get_events", {"category": "sport"})
        events = json.loads(result.data)

        assert len(events) == 1
        assert events[0]["title"] == "Partita"


async def test_get_events_filters_by_date_range(mcp_client):
    async with mcp_client as client:
        await client.call_tool("save_events_to_db", {"events_json": json.dumps([
            _event(title="Passato", start_datetime="2020-01-01T10:00:00", venue="V1"),
            _event(title="Futuro", start_datetime="2030-01-01T10:00:00", venue="V2"),
        ])})

        result = await client.call_tool("get_events", {"date_from": "2025-01-01"})
        events = json.loads(result.data)

        assert len(events) == 1
        assert events[0]["title"] == "Futuro"


async def test_get_events_respects_limit(mcp_client):
    async with mcp_client as client:
        many_events = [
            _event(title=f"Evento {i}", venue=f"Venue {i}")
            for i in range(5)
        ]
        await client.call_tool("save_events_to_db", {"events_json": json.dumps(many_events)})

        result = await client.call_tool("get_events", {"limit": 2})
        events = json.loads(result.data)

        assert len(events) == 2


async def test_get_events_empty_db_returns_empty_list(mcp_client):
    async with mcp_client as client:
        result = await client.call_tool("get_events", {"city": "Nessuncittà"})
        assert json.loads(result.data) == []
