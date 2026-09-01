from nodes import search_db_node
from fakes import FakeMCPClient


async def test_search_db_no_mcp_client_is_skipped_not_fatal():
    state = {"target_location": "Firenze", "mcp_client": None}
    result = await search_db_node(state)

    assert result["current_step"] == "db_retrieval_skipped"


async def test_search_db_success_stores_parsed_events_list():
    db_events = [{"title": "Evento 1"}, {"title": "Evento 2"}]
    mcp = FakeMCPClient(db_events=db_events)

    state = {"target_location": "Firenze", "mcp_client": mcp}
    result = await search_db_node(state)

    assert result["retrieved_events"] == db_events
    assert result["current_step"] == "events_retrieved"


async def test_search_db_error_sets_retrieved_events_none():
    mcp = FakeMCPClient(raise_on_get_events=True)

    state = {"target_location": "Firenze", "mcp_client": mcp}
    result = await search_db_node(state)

    assert result["retrieved_events"] is None
    assert result["current_step"] == "db_retrieval_error"


async def test_search_db_calls_get_events_with_city_param():
    """Regression test: get_events on the MCP server expects 'city', not
    'location'. A previous bug sent 'location', which the server silently
    ignored (or rejected)."""

    mcp = FakeMCPClient(db_events=[])

    state = {"target_location": "Milano", "mcp_client": mcp}
    await search_db_node(state)

    assert len(mcp.calls) == 1
    tool_name, args = mcp.calls[0]

    assert tool_name == "get_events"
    assert args.get("city") == "Milano"
    assert "location" not in args
