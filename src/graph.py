from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from state import AgentState
import requests
from bs4 import BeautifulSoup
from typing import List
from models import Event

#########################
# SETUP
#########################

# Import API keys from a separate document
load_dotenv()

# Defines a properly structured output
class LocationExtraction(BaseModel):
    target_location: str = Field(description="The name of the city, municipality, or location mentioned in the user query.")
    latitude: float = Field(description="The approximate geographical latitude of the location.")
    longitude: float = Field(description="The approximate geographical longitude of the location.")

class EventList(BaseModel):
    events: List[Event] = Field(
        default_factory=[],
        description="A list of structured events extracted from the webpage text."
    )
# Set current date
current_date = datetime.now().strftime("%Y-%m-%d")

#########################
# Methods
#########################

# Analyze user query to identify location
def analyze_query_node(state: AgentState) -> AgentState:
    query = state.get("user_query", "")
    print(f"\n[GRAPH] Analysis of the query: '{query}'")

    # Initialize the model temperature=0 ensures deterministic output, perfect for data extraction
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

    # Bind the model to the Pydantic schema
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
        # The LLM returns a validated Pydantic object directly
        result: LocationExtraction = structured_llm.invoke(prompt)

        target_location = result.target_location
        lat = result.latitude
        lon = result.longitude

        print(f"[LLM SUCCESS] Location identified: {target_location} (lat={lat}, lon={lon})")

    except Exception as e:
        print(f"[LLM ERROR] LLM extraction failed: {e}")
        # Default
        target_location = "Milan"
        lat, lon = 45.4642, 9.1900

    return {
        **state,
        "target_location": target_location,
        "latitude": lat,
        "longitude": lon,
        "current_step": "query_analyzed"
    }

# Searches for events in the given location using Tavily
def search_sources_node(state: AgentState) -> AgentState:
    print("\n[GRAPH] Starting web search via Tavily...")
    location = state.get("target_location", "")

    # Defines the search query to be executed by the LLM
    search_query = f"best event sites festivals clubs {location} today weekend"
    print(f"[SEARCH] Executing query: '{search_query}'")

    try:
        client = TavilyClient()
        # search_depth="basic" is faster and consumes fewer credits
        response = client.search(query=search_query, search_depth="basic", max_results=4)

        # Extract only the URLs from the results
        discovered_urls = [result["url"] for result in response.get("results", [])]

        print(f"[SEARCH] Found {len(discovered_urls)} sources:")
        for url in discovered_urls:
            print(f"  - {url}")

    except Exception as e:
        print(f"[SEARCH ERROR] Tavily failed: {e}")
        discovered_urls = []

    return {
        **state,
        "discovered_sources": discovered_urls,
        "current_step": "sources_discovered"
    }

# Scrapes URls and uses Gemini to extract data
def extract_events_node(state: AgentState) -> AgentState:
    print("\n[GRAPH] Starting web scraping and event extraction...")

    urls = state.get("discovered_sources", [])
    location = state.get("target_location", "the area")
    extracted_events_total = []
    raw_contents = []

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(EventList)

    # Scraping limited to 2 due to limited availability of tokens
    for url in urls[:2]:
        print(f"[SCRAPE] Fetching content from: {url}")
        try:
            # 1. Fetch webpage
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # 2. Extract text using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)

            # Truncate text to avoid exceeding token limits during tests (first 15k chars)
            text_content = text_content[:15000]
            raw_contents.append({"url": url, "snippet": text_content[:200]})

            # 3. Prompt Gemini to extract events
            prompt = f"""
            You are an expert data extractor. Read the following text from a webpage and identify 
            any upcoming public events (concerts, festivals, club nights, cultural events, etc.).

            CRITICAL RULES FOR DATA FORMATTING:
            1. 'start_datetime' MUST be a valid ISO 8601 datetime string (e.g., '2026-08-14T20:00:00'). 
                Today is {current_date}. If the text says 'Friday at 8:00 PM', calculate the actual upcoming date based on today. DO NOT output conversational strings like 'Friday'.
            2. If the exact venue or city is missing, use context clues or default to '{location}'.
            3. Ensure the category matches one of the allowed enum values (sport, culture, cinema, festival, concert, club, other).
            4. Set the source_url strictly to: {url}

            Webpage Text:
            {text_content}
            """

            # 4. LLM Call
            result: EventList = structured_llm.invoke(prompt)

            if result.events:
                extracted_events_total.extend(result.events)
                print(f"[EXTRACT SUCCESS] Found {len(result.events)} events in {url}")
            else:
                print(f"[EXTRACT INFO] No valid events found in {url}")

        except Exception as e:
            print(f"[SCRAPE/EXTRACT ERROR] Failed processing {url}: {e}")

    return {
        **state,
        "events": extracted_events_total,
        "raw_page_contents": raw_contents,
        "current_step": "events_extracted"
    }

#########################
# GRAPH
#########################

# Builds and run Langgraph graph
def create_event_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("search_sources", search_sources_node)
    workflow.add_node("extract_events", extract_events_node)

    # Define flow
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "search_sources")
    workflow.add_edge("search_sources", "extract_events")
    workflow.add_edge("extract_events", END)

    return workflow.compile()

#########################
# MAIN
#########################

if __name__ == "__main__":

    print("=== Langgraph Test: Analyze Query Node ===")

    app = create_event_graph()

    initial_state = {
        "user_query": "I am looking for electronic near Milan this weekend",
        "target_location": "",
        "latitude": 0.0,
        "longitude": 0.0,
        "discovered_sources": [],
        "raw_page_contents": [],
        "events": [],
        "errors": [],
        "current_step": "start"
    }

    final_state = app.invoke(initial_state)

    print("\n[SUCCESS]")