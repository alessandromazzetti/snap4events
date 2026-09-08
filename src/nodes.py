import asyncio
import os

import requests
import json
from datetime import datetime
from bs4 import BeautifulSoup
from tavily import TavilyClient
from state import AgentState
from models import LocationExtraction, EventList, RetrievalDecision, Event
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors


# ---------------------------------------------------------
# Helper: Extract JSON from LLM response
# ---------------------------------------------------------

def parse_llm_json(response):
    """Extract a JSON object from the LLM response."""

    if isinstance(response, dict):
        return response

    if not isinstance(response, str):
        raise ValueError(
            f"Unsupported LLM response type: {type(response)}"
        )

    text = response.strip()

    # Remove markdown code fences if the model returned ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return json.loads(text)


def salvage_truncated_events_json(response):
    """Best-effort recovery for an LLM response that was cut off mid-JSON
    (typically because the output hit a token limit). Instead of discarding
    the whole batch, walk the 'events' array and keep every event object
    that was fully generated before the truncation point.

    Returns a list of dicts (possibly empty if nothing could be salvaged).
    """

    if not isinstance(response, str):
        return []

    text = response.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    array_start = text.find("[")

    if array_start == -1:
        return []

    decoder = json.JSONDecoder()
    pos = array_start + 1
    salvaged = []

    while True:
        # Skip whitespace/commas between elements
        while pos < len(text) and text[pos] in " \t\n\r,":
            pos += 1

        if pos >= len(text) or text[pos] == "]":
            break

        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            # This is the object that got cut off mid-way: stop here,
            # keep whatever was successfully parsed before it.
            break

        salvaged.append(obj)
        pos = end

    return salvaged


# ---------------------------------------------------------
# Helper: Call ClearML
# ---------------------------------------------------------

async def call_llm(state: AgentState, prompt: str):
    """Call the ClearML client injected into the graph state."""

    llm_client = state.get("llm_client")

    if llm_client is None:
        raise RuntimeError(
            "No ClearML client found in AgentState."
        )

    # Run synchronous requests call without blocking the async graph
    return await asyncio.to_thread(
        llm_client.generate,
        prompt
    )


# ---------------------------------------------------------
# Helper: Extract textual content from API response
# ---------------------------------------------------------

def extract_answer(response):
    """Extract the model's textual answer from the ClearML response."""

    if isinstance(response, str):
        return response

    if not isinstance(response, dict):
        return str(response)

    # OpenAI-compatible response format
    choices = response.get("choices")

    if choices:
        choice = choices[0]

        if isinstance(choice, dict):

            message = choice.get("message")

            if isinstance(message, dict):
                content = message.get("content")

                if isinstance(content, str):
                    return content

            text = choice.get("text")

            if isinstance(text, str):
                return text

    # Legacy / generic response formats
    for key in ("answer", "content", "response", "output", "text"):

        value = response.get(key)

        if isinstance(value, str):
            return value

    return json.dumps(response, ensure_ascii=False)


# ---------------------------------------------------------
# Node 1: Analyze
# ---------------------------------------------------------

async def analyze_query_node(state: AgentState) -> AgentState:
    """Performs query analysis using the ClearML LLM, finding the exact coordinates for the location
    that is being prompted in."""

    query = state.get("user_query", "")
    print(f"\n[NODE 1] Analyzing query: '{query}'")

    prompt = f"""
You are a geographic data extractor specialized in Italian locations.

Analyze the user query and return ONLY valid JSON.

Required JSON format:

{{
    "target_location": "Italian city or geographical entity",
    "latitude": 43.7697,
    "longitude": 11.2556
}}

Follow these strict execution steps:

STEP 1: Scan the text for ANY city, region, monument, or point of interest
(e.g., 'Florence', 'Coliseum', 'Milan').

STEP 2: If a location is found, resolve it to its correct Italian geographical
entity (e.g., 'Florence' -> 'Firenze') and output its coordinates.

STEP 3: ONLY IF the text contains absolutely zero geographic references,
use 'Rome' as the fallback location.

Do not include markdown.
Do not include explanations.

User query:
{query}
"""

    try:
        # Call ClearML directly
        raw_response = await call_llm(state, prompt)

        # Extract textual answer from API response
        answer = extract_answer(raw_response)

        # Parse the JSON returned by the model
        data = parse_llm_json(answer)

        # Validate using the existing Pydantic model
        result = LocationExtraction(**data)

        target_location = result.target_location
        lat = result.latitude
        lon = result.longitude

        print(
            f"[NODE 1] Location: {target_location} "
            f"(lat={lat}, lon={lon})"
        )

    except Exception as e:
        print(f"[NODE 1 ERROR] {e}")

        target_location = "Florence"
        lat, lon = 43.7697, 11.2556

    return {
        **state,
        "target_location": target_location,
        "latitude": lat,
        "longitude": lon,
        "current_step": "query_analyzed"
    }

# ---------------------------------------------------------
# Node 2: Retrieve
# ---------------------------------------------------------

async def search_db_node(state: AgentState) -> AgentState:
    """Look inside the database for already stored events."""

    location = state.get("target_location", "")
    print(f"\n[NODE 2] Looking inside the database for: {location}")

    mcp_client = state.get("mcp_client")

    if not mcp_client:
        print(f"[NODE 2] MCP Client not available.")
        return {
            **state,
            "current_step": "db_retrieval_skipped"
        }

    date_from = datetime.now().strftime("%Y-%m-%d")

    try:
        result = await mcp_client.call_tool(
            "get_events",
            {
                "city": location,
                "date_from": date_from,
                "limit": 50
            }
        )

        # Create a python list out of a JSON string
        if hasattr(result, "data") and result.data:
            events = json.loads(result.data)
        else:
            events = []

        print(f"[NODE 2] Found {len(events)} events in database")

        return {
            **state,
            "retrieved_events": events,
            "current_step": "events_retrieved"
        }

    except Exception as e:
        print(f"[NODE 2 ERROR] Failed to retrieve events: {e}")

        return {
            **state,
            "retrieved_events": None,
            "current_step": "db_retrieval_error"
        }


# ---------------------------------------------------------
# Helper: Convert a raw DB row (as returned by get_events) into an Event
# ---------------------------------------------------------

def event_from_db_row(row: dict) -> Event:
    """Converts a flat event row coming from the SQLite DB (via MCP get_events)
    into an Event pydantic object, ignoring DB-only bookkeeping columns."""

    return Event(
        title=row.get("title", "Untitled"),
        category=row.get("category", "other"),
        start_datetime=row.get("start_datetime"),
        venue=row.get("venue", "N/D"),
        city=row.get("city", "N/D"),
        source_url=row.get("source_url") or "https://www.km4city.org/",
    )


# ---------------------------------------------------------
# Node 2b: RAG decision - let the LLM decide whether to trust retrieval
# ---------------------------------------------------------

async def decide_retrieval_usage_node(state: AgentState) -> AgentState:
    """Asks the LLM whether the events already retrieved from the database are
    sufficient and relevant enough to answer the user query, or whether a fresh
    web search + scrape is needed instead. This is the core RAG routing step:
    the LLM itself decides if the retrieved context should be used."""

    query = state.get("user_query", "")
    location = state.get("target_location", "")
    retrieved_events = state.get("retrieved_events") or []

    print(f"\n[NODE 2b] Deciding whether to use {len(retrieved_events)} retrieved events...")

    # Nothing was retrieved: no point asking the LLM, go straight to web search
    if not retrieved_events:
        print("[NODE 2b] No retrieved events available, will search the web.")
        return {
            **state,
            "use_retrieved_data": False,
            "current_step": "retrieval_rejected_empty"
        }

    current_date = datetime.now().strftime("%Y-%m-%d")

    # Compact summary of the retrieved events (avoid dumping raw_json/status/etc.)
    events_summary = [
        {
            "title": ev.get("title"),
            "category": ev.get("category"),
            "start_datetime": ev.get("start_datetime"),
            "venue": ev.get("venue"),
            "city": ev.get("city"),
        }
        for ev in retrieved_events
    ]

    prompt = f"""
    You are deciding whether previously stored event data is good enough to answer
    a user's request, or whether a fresh web search is needed instead.
    
    Today is {current_date}.
    Target location: {location}
    User query: {query}
    
    Events already stored in the database ({len(events_summary)} total):
    {json.dumps(events_summary, ensure_ascii=False, indent=2)}
    
    Decide whether these stored events are sufficient to answer the user query.
    Consider:
    - Relevance: do they actually match what the user is asking for (location, type of event)?
    - Freshness: are there enough upcoming (not past) events?
    - Coverage: is the number of events reasonable, or clearly too sparse?

    CRITICAL: If you decide the data is sufficient, you MUST identify ONLY the specific 
    events that match the exact timeframe requested by the user.

    Return ONLY valid JSON in this exact format:

    {{
        "use_retrieved_data": true,
        "reasoning": "short explanation",
        "relevant_event_titles": ["Exact Title 1", "Exact Title 2"]
    }}

    Set "use_retrieved_data" to false if the stored events are empty, irrelevant,
    outdated, or clearly insufficient, in which case a web search will be triggered.
    If false, leave "relevant_event_titles" as an empty list [].

    Do not include markdown.
    Do not include explanations outside the JSON.
    """

    try:
        raw_response = await call_llm(state, prompt)
        answer = extract_answer(raw_response)
        data = parse_llm_json(answer)

        # Pop the titles list out of the dictionary BEFORE passing it to Pydantic.
        # This allows us to use the titles for filtering without causing a validation
        # error if 'relevant_event_titles' doesn't exist in models.RetrievalDecision.
        relevant_titles = set(data.pop("relevant_event_titles", []))

        decision = RetrievalDecision(**data)

        print(f"[NODE 2b] Decision: use_retrieved_data={decision.use_retrieved_data} ({decision.reasoning})")

        if decision.use_retrieved_data:
            # Convert the raw DB rows into validated Event objects to reuse downstream
            converted_events = []

            for row in relevant_titles:
                try:
                    converted_events.append(event_from_db_row(row))
                except Exception as conv_err:
                    print(f"[NODE 2b WARNING] Skipping malformed stored event: {conv_err}")

            return {
                **state,
                "use_retrieved_data": True,
                "events": converted_events,
                "current_step": "retrieval_accepted"
            }

        return {
            **state,
            "use_retrieved_data": False,
            "current_step": "retrieval_rejected"
        }

    except Exception as e:
        print(f"[NODE 2b ERROR] {e}")

        # On any failure, fall back to the safer path: search the web
        return {
            **state,
            "use_retrieved_data": False,
            "current_step": "retrieval_decision_error"
        }


def route_after_retrieval_decision(state: AgentState) -> str:
    """Conditional edge: routes to the web-search branch or straight to save,
    depending on the LLM's decision about the retrieved data."""

    return "use_retrieved" if state.get("use_retrieved_data") else "search_web"


# ---------------------------------------------------------
# Node 3: Search
# ---------------------------------------------------------

async def search_sources_node(state: AgentState) -> AgentState:
    """Finds URLs containing info about events in the given location using Tavily."""

    location = state.get("target_location", "")
    query = state.get("user_query", "")  # Extract the original query
    print(f"\n[NODE 3] Searching web for: {location}")

    # Dynamically build the search using the user's actual request
    search_query = f"events festivals clubs {location} {query}"

    try:
        client = TavilyClient()

        # Asynchronous research in order not to block the system
        response = await asyncio.to_thread(
            client.search,
            query=search_query,
            search_depth="basic",
            max_results=4
        )

        discovered_urls = [
            res["url"]
            for res in response.get("results", [])
        ]

        print(f"[NODE 3] Found {len(discovered_urls)} URLs")

    except Exception as e:
        print(f"[NODE 3 ERROR] {e}")
        discovered_urls = []

    return {
        **state,
        "discovered_sources": discovered_urls,
        "current_step": "sources_discovered"
    }


# ---------------------------------------------------------
# Node 4: Extract
# ---------------------------------------------------------

async def extract_events_node(state: AgentState) -> AgentState:
    """Scrapes data from the URLs obtained at node 2, extracts clean text using BeautifulSoup and
    uses the ClearML LLM to parse it into existent Pydantic models."""

    print("\n[NODE 4] Scraping and parsing events...")

    urls = state.get("discovered_sources", [])
    location = state.get("target_location", "the area")
    query = state.get("user_query", "")
    extracted_events_total = []

    current_date = datetime.now().strftime("%Y-%m-%d")

    for url in urls[:2]:

        print(f"  -> Scraping: {url}")

        try:
            # Asynchronous download of the page
            response = await asyncio.to_thread(
                requests.get,
                url,
                timeout=10,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
                }
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.content,
                "html.parser"
            )

            text_content = soup.get_text(
                separator=" ",
                strip=True
            )[:15000]

            prompt = f"""
            You are a strict and precise data extractor. Your primary goal is to find events that EXACTLY match the user's timeframe.

            USER'S ORIGINAL REQUEST: "{query}"
            TODAY'S DATE: {current_date}

            Return ONLY valid JSON.

            Required JSON format:
            {{
                "events": [
                    {{
                        "title": "Event title",
                        "category": "concert",
                        "start_datetime": "2026-10-03T20:00:00",
                        "venue": "Venue name",
                        "city": "Firenze",
                        "source_url": "{url}",
                        "expected_reach": 50000
                    }}
                ]
            }}

            CRITICAL RULES (FOLLOW EXACTLY):

            1. STRICT DATE FILTERING: Read the USER'S ORIGINAL REQUEST carefully. You MUST ONLY extract events that happen exactly during the requested dates (e.g., "first weekend of october"). 

            2. ALLOW EMPTY RESULTS: If the webpage does NOT contain events matching the exact requested timeframe, you MUST return an empty list: {{"events": []}}. Do NOT include events from other days or weeks just to fill the list.

            3. 'category' MUST be exactly one of: "sport", "culture", "cinema", "festival", "concert", "club", "other". (Do not use "market" or anything else).

            4. 'start_datetime' MUST be a VALID ISO 8601 datetime.

            5. 'expected_reach' MUST be an integer number (e.g. 50000). Do not use strings like "50k". If unknown, use null.

            6. Extract AT MOST 8 events, but ONLY those that strictly pass the date filter in Rule 1. Quality is more important than quantity.

            7. If venue is missing, use "N/D". If city is missing, default to '{location}'.

            8. Set source_url strictly to: {url}

            9. Do not invent events, do not include markdown, and do not include explanations.
                        
            10. Do not include markdown.

            11. Do not include explanations.
                
            Webpage Text:

            {text_content}
            """

            # Asynchronous call to ClearML
            raw_response = await call_llm(
                state,
                prompt
            )

            # Extract textual answer from API response
            answer = extract_answer(raw_response)

            try:
                # Parse JSON returned by the model
                data = parse_llm_json(answer)

                # Validate using the existing Pydantic model
                result = EventList(**data)
                page_events = result.events

            except (json.JSONDecodeError, ValueError) as parse_err:
                # The LLM response was likely cut off mid-JSON (token limit).
                # Salvage whatever complete event objects were generated
                # before the truncation point, instead of discarding them all.
                print(
                    f"  -> [WARNING] JSON truncated/invalid for {url} "
                    f"({parse_err}); attempting to salvage partial events."
                )

                salvaged_raw = salvage_truncated_events_json(answer)
                page_events = []

                for raw_event in salvaged_raw:
                    try:
                        page_events.append(Event(**raw_event))
                    except Exception as event_err:
                        print(f"  -> [WARNING] Skipping unsalvageable event: {event_err}")

                print(f"  -> [WARNING] Salvaged {len(page_events)} event(s) out of a truncated response.")

            if page_events:
                extracted_events_total.extend(page_events)

            await asyncio.sleep(2)

        except Exception as e:
            print(
                f"  -> [ERROR] Failed processing {url}: {e}"
            )

    print(
        f"[NODE 4] Total events extracted: "
        f"{len(extracted_events_total)}"
    )

    return {
        **state,
        "events": extracted_events_total,
        "current_step": "events_extracted"
    }


# ---------------------------------------------------------
# Node 5: Save to Database (via MCP)
# ---------------------------------------------------------

async def save_events_node(state: AgentState) -> AgentState:
    """Converts extracted events to JSON and saves them via MCP DB Server."""

    print("\n[NODE 5] Saving events to SQLite database via MCP...")

    events = state.get("events", [])

    if not events:
        print("[NODE 5] No events to save.")

        return {
            **state,
            "current_step": "db_saved_empty"
        }

    # Serialize events in one string
    events_json_list = [
        ev.model_dump(mode="json")
        for ev in events
    ]

    events_json_str = json.dumps(events_json_list)

    try:
        # Calls MCP Server tool through the client
        mcp_client = state.get("mcp_client")

        if mcp_client:
            result = await mcp_client.call_tool(
                "save_events_to_db",
                {"events_json": events_json_str}
            )

            print(f"[NODE 5 SUCCESS] {result}")

        else:
            print(
                "[NODE 5 WARNING] MCP Client not provided "
                "in state. Skipping DB save."
            )

    except Exception as e:
        print(
            f"[NODE 5 ERROR] Failed to save events via MCP: {e}"
        )

    return {
        **state,
        "current_step": "db_saved"
    }

# ---------------------------------------------------------
# Node 6: Export to PDF
# ---------------------------------------------------------
def safe_filename(text: str) -> str:
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:100]


def export_event_to_pdf(event: Event, output_dir: str = "pdf_events"):
    """Define a function that allows to export a pdf from structured data."""

    os.makedirs(output_dir, exist_ok=True)

    filename = (
        f"{safe_filename(event.title)}_"
        f"{event.start_datetime.strftime('%Y%m%d_%H%M')}.pdf"
    )

    filepath = os.path.join(output_dir, filename)

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
        "EventTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=15,
    )

    story = []

    story.append(
        Paragraph(event.title, title_style)
    )

    story.append(Spacer(1, 10))

    data = [
        ["Category", event.category],
        ["Date", event.start_datetime.strftime("%d/%m/%Y")],
        ["Time", event.start_datetime.strftime("%H:%M")],
        ["Venue", event.venue],
        ["City", event.city],
        ["URL", event.source_url],
        ["Expected reach", event.expected_reach]
    ]

    table = Table(
        data,
        colWidths=[40 * mm, 120 * mm]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(table)

    doc.build(story)

    return filepath


async def export_events_pdf_node(state: AgentState) -> AgentState:
    """Define the node which is orchestrated to export every event pdf."""

    print("\n[NODE 6] Exporting events to PDF...")

    events = state.get("events", [])

    pdf_files = []

    for event in events:

        try:
            filepath = export_event_to_pdf(event)

            pdf_files.append(filepath)

            print(f"[NODE 6] PDF created: {filepath}")

        except Exception as e:
            print(
                f"[NODE 6 ERROR] "
                f"Failed to create PDF for '{event.title}': {e}"
            )

    return {
        **state,
        "pdf_files": pdf_files,
        "current_step": "pdf_exported"
    }