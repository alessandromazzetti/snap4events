import asyncio
import requests
import json
from datetime import datetime
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
from models import LocationExtraction, EventList


# ---------------------------------------------------------
# Node 1: Analyze
# ---------------------------------------------------------
async def analyze_query_node(state: AgentState) -> AgentState:
    """Performs query analysis using Google Gemini API, finding the exact coordinates for the location
    that is being prompted in."""

    query = state.get("user_query", "")
    print(f"\n[NODE 1] Analyzing query: '{query}'")

    llm = state.get("model")
    if llm is None:
        print("[NODE 1 WARNING] No model found in state, falling back to default Gemini client.")
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(LocationExtraction)

    prompt = f"""
    You are a geographic data extractor specialized in Italian locations. 
    Analyze the user query and extract the target location.

    Follow these strict execution steps:
    STEP 1: Scan the text for ANY city, region, monument, or point of interest (e.g., 'Florence', 'Coliseum', 'Milan').
    STEP 2: If a location is found, resolve it to its correct Italian geographical entity (e.g., 'Florence' -> 'Firenze') and output its coordinates. Ignore STEP 3.
    STEP 3: ONLY IF the text contains absolutely zero geographic references, use 'Rome' as the fallback location.

    User query: {query}
    """

    try:
        # Asynchronous invoke
        result = await structured_llm.ainvoke(prompt)
        target_location = result.target_location
        lat, lon = result.latitude, result.longitude
        print(f"[NODE 1] Location: {target_location} (lat={lat}, lon={lon})")
    except Exception as e:
        print(f"[NODE 1 ERROR] {e}")
        target_location = "Florence"
        lat, lon = 43.7697, 11.2556

    # **state = dictionary unpacking allows python to take the old state and to copy new values only in it
    return {**state, "target_location": target_location, "latitude": lat, "longitude": lon,
            "current_step": "query_analyzed"}


# ---------------------------------------------------------
# Node 2: Search
# ---------------------------------------------------------
async def search_sources_node(state: AgentState) -> AgentState:
    """Finds URLs containing info about events in the given location using Tavily."""

    location = state.get("target_location", "")
    print(f"\n[NODE 2] Searching web for: {location}")

    search_query = f"best event sites festivals clubs {location} today weekend"

    try:
        client = TavilyClient()
        # Asynchronous research in order not to block the system
        response = await asyncio.to_thread(client.search, query=search_query, search_depth="basic", max_results=4)
        discovered_urls = [res["url"] for res in response.get("results", [])]
        print(f"[NODE 2] Found {len(discovered_urls)} URLs")
    except Exception as e:
        print(f"[NODE 2 ERROR] {e}")
        discovered_urls = []

    return {**state, "discovered_sources": discovered_urls, "current_step": "sources_discovered"}


# ---------------------------------------------------------
# Node 3: Extract
# ---------------------------------------------------------
async def extract_events_node(state: AgentState) -> AgentState:
    """Scrapes data from the URLs obtained at node 2, extracts clean text using BeautifulSoup and
    uses Gemini to parse it into existent Pydantic models."""

    print("\n[NODE 3] Scraping and parsing events...")
    urls = state.get("discovered_sources", [])
    location = state.get("target_location", "the area")
    extracted_events_total = []

    llm = state.get("model")
    if llm is None:
        print("[NODE 3 WARNING] No model found in state, falling back to default Gemini client.")
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(EventList)
    current_date = datetime.now().strftime("%Y-%m-%d")

    for url in urls[:2]:
        print(f"  -> Scraping: {url}")
        try:
            # Asynchronous download of the page
            response = await asyncio.to_thread(requests.get, url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)[:15000]

            prompt = f"""
            You are an expert data extractor. Identify upcoming public events.

            CRITICAL RULES:
            1. 'start_datetime' MUST be a valid ISO 8601 datetime (e.g., '2026-08-14T20:00:00'). Today is {current_date}. Calculate upcoming dates correctly.
            2. If venue/city missing, default to '{location}'.
            3. Set source_url strictly to: {url}

            Webpage Text:
            {text_content}
            """

            # Asynchronous call to LLM
            result = await structured_llm.ainvoke(prompt)
            if result.events:
                extracted_events_total.extend(result.events)

            await asyncio.sleep(2)

        except Exception as e:
            print(f"  -> [ERROR] Failed processing {url}: {e}")

    print(f"[NODE 3] Total events extracted: {len(extracted_events_total)}")
    return {**state, "events": extracted_events_total, "current_step": "events_extracted"}

# ---------------------------------------------------------
# Node 4: Save to Database (via MCP)
# ---------------------------------------------------------
async def save_events_node(state: AgentState) -> AgentState:
    """Converts extracted events to JSON and saves them via MCP DB Server."""
    print("\n[NODE 4] Saving events to SQLite database via MCP...")
    events = state.get("events", [])

    if not events:
        print("[NODE 4] No events to save.")
        return {**state, "current_step": "db_saved_empty"}

    # Serialize events in one string
    events_json_list = [ev.model_dump(mode='json') for ev in events]
    events_json_str = json.dumps(events_json_list)

    try:
        # Calls MCP Server tool through the client
        mcp_client = state.get("mcp_client")
        if mcp_client:
            result = await mcp_client.call_tool("save_events_to_db", {"events_json": events_json_str})
            print(f"[NODE 4 SUCCESS] {result}")
        else:
            print("[NODE 4 WARNING] MCP Client not provided in state. Skipping DB save.")

    except Exception as e:
        print(f"[NODE 4 ERROR] Failed to save events via MCP: {e}")

    return {**state, "current_step": "db_saved"}