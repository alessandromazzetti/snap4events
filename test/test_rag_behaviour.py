"""
Local test for the RAG decision branch added to the snap4events graph.

  1. "DB is enough"  -> the fake LLM says use_retrieved_data=True
                        -> the graph must stop with the DB events, no web search.

  2. "DB not enough" -> the fake LLM says use_retrieved_data=False
                        -> the graph must fall back to search_sources/extract_events
                        (both patched here so no real HTTP call is made).
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# The graph modules (nodes, builder, state, models) live in src/, a sibling
# of this test/ folder, so make them importable regardless of the cwd this
# script is launched from.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import nodes
from builder import create_agent_app


# ---------------------------------------------------------
# Fake MCP Client
# ---------------------------------------------------------

class FakeMCPClient:
    def __init__(self, db_events):
        self.db_events = db_events
        self.saved_payloads = []

    async def call_tool(self, name, args):
        if name == "get_events":
            return SimpleNamespace(data=json.dumps(self.db_events))

        if name == "save_events_to_db":
            self.saved_payloads.append(args)
            return SimpleNamespace(data=json.dumps({"status": "ok"}))

        raise ValueError(f"Unexpected MCP tool called in test: {name}")


# ---------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------

class FakeLLMClient:
    """Routes canned JSON answers based on which prompt it receives, so the
    same fake client can stand in for every LLM call in the graph."""

    def __init__(self, use_retrieved_data: bool):
        self.use_retrieved_data = use_retrieved_data

    def generate(self, prompt=None, **kwargs):
        if "geographic data extractor" in prompt:
            return json.dumps({
                "target_location": "Firenze",
                "latitude": 43.7697,
                "longitude": 11.2558
            })

        if "deciding whether previously stored event data" in prompt:
            return json.dumps({
                "use_retrieved_data": self.use_retrieved_data,
                "reasoning": "test scenario"
            })

        if "expert data extractor" in prompt:
            return json.dumps({
                "events": [{
                    "title": "Concerto di Test (dal web)",
                    "category": "concert",
                    "start_datetime": "2026-09-10T21:00:00",
                    "venue": "Teatro Test",
                    "city": "Firenze",
                    "source_url": "https://example.com/evento-web"
                }]
            })

        raise AssertionError(f"Unexpected prompt in test: {prompt[:80]}...")


# ---------------------------------------------------------
# Fakes for the web-search branch (Tavily + requests + BeautifulSoup input)
# ---------------------------------------------------------

class FakeTavilyClient:
    def __init__(self, *args, **kwargs):
        pass

    def search(self, query, search_depth="basic", max_results=4):
        return {"results": [{"url": "https://example.com/evento-web"}]}


class FakeResponse:
    status_code = 200
    content = b"<html><body>Concerto di Test il 10 settembre a Firenze</body></html>"

    def raise_for_status(self):
        pass


def fake_requests_get(url, timeout=10):
    return FakeResponse()


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

async def run_scenario(label: str, db_events: list, use_retrieved_data: bool):
    print(f"\n{'=' * 60}\nSCENARIO: {label}\n{'=' * 60}")

    mcp_client = FakeMCPClient(db_events)
    llm_client = FakeLLMClient(use_retrieved_data=use_retrieved_data)

    app = create_agent_app()

    initial_state = {
        "user_query": "Cerco eventi a Firenze questo weekend",
        "target_location": "",
        "latitude": 0.0,
        "longitude": 0.0,
        "discovered_sources": [],
        "raw_page_contents": [],
        "retrieved_events": [],
        "use_retrieved_data": False,
        "events": [],
        "errors": [],
        "current_step": "start",
        "mcp_client": mcp_client,
        "llm_client": llm_client
    }

    with patch("nodes.TavilyClient", FakeTavilyClient), \
         patch("nodes.requests.get", fake_requests_get):
        final_state = await app.ainvoke(initial_state)

    print(f"current_step:      {final_state['current_step']}")
    print(f"use_retrieved_data: {final_state.get('use_retrieved_data')}")
    print(f"events found:      {len(final_state['events'])}")
    for ev in final_state["events"]:
        print(f"  - {ev.title} ({ev.source_url})")
    print(f"save_events_to_db calls made: {len(mcp_client.saved_payloads)}")

    return final_state


async def main():
    # --- Scenario 1: DB has good, relevant events -> LLM should accept them ---
    db_events_good = [
        {
            "title": "Festa in Piazza",
            "category": "festival",
            "start_datetime": "2026-09-06T19:00:00",
            "venue": "Piazza della Signoria",
            "city": "Firenze",
            "source_url": "https://example.com/festa-piazza"
        },
        {
            "title": "Mostra d'arte contemporanea",
            "category": "culture",
            "start_datetime": "2026-09-07T10:00:00",
            "venue": "Palazzo Strozzi",
            "city": "Firenze",
            "source_url": "https://example.com/mostra-arte"
        }
    ]

    final_1 = await run_scenario(
        "DB events accepted by the LLM",
        db_events=db_events_good,
        use_retrieved_data=True
    )

    assert final_1["current_step"] == "retrieval_accepted", "Expected the retrieval-accepted path"
    assert final_1["use_retrieved_data"] is True
    assert len(final_1["events"]) == 2, "Expected the 2 DB events to be reused as-is"
    assert final_1["events"][0].title == "Festa in Piazza"
    print("\n[PASS] Scenario 1: retrieved data used, no web search triggered.")

    # --- Scenario 2: DB events rejected by the LLM -> web search branch runs ---
    final_2 = await run_scenario(
        "DB events rejected by the LLM (fallback to web search)",
        db_events=db_events_good,
        use_retrieved_data=False
    )

    assert final_2["current_step"] == "db_saved", "Expected the web-search path to reach save_events"
    assert final_2["use_retrieved_data"] is False
    assert len(final_2["events"]) == 1, "Expected the 1 event scraped from the (fake) web page"
    assert final_2["events"][0].title == "Concerto di Test (dal web)"
    print("\n[PASS] Scenario 2: web search branch used, event saved via MCP.")

    # --- Scenario 3: DB is empty -> should skip the LLM decision and go to web ---
    final_3 = await run_scenario(
        "DB is empty (no LLM decision call needed)",
        db_events=[],
        use_retrieved_data=True  # irrelevant, retrieved_events is empty
    )

    assert final_3["current_step"] == "db_saved"
    assert final_3["use_retrieved_data"] is False
    print("\n[PASS] Scenario 3: empty DB correctly skipped straight to web search.")

    print("\nAll scenarios passed. ✅")


if __name__ == "__main__":
    asyncio.run(main())
