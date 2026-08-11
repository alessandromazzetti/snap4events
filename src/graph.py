import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from state import AgentState

# Import API keys from a separate document
load_dotenv()

# Defines a properly structured output
class LocationExtraction(BaseModel):
    target_location: str = Field(description="The name of the city, municipality, or location mentioned in the user query.")
    latitude: float = Field(description="The approximate geographical latitude of the location.")
    longitude: float = Field(description="The approximate geographical longitude of the location.")


# Analyze user query to identify location
def analyze_query_node(state: AgentState) -> AgentState:
    query = state.get("user_query", "")
    print(f"\n[GRAPH] Analysis of the query: '{query}'")

    # Initialize the model temperature=0 ensures deterministic output, perfect for data extraction
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

    # Bind the model to the Pydantic schema
    structured_llm = llm.with_structured_output(LocationExtraction)

    prompt = f"""
    You are a geographic assistant specialized in Italian locations. 
    Your task is to extract the city, municipality, or location from this user query 
    and provide its approximate geographical coordinates (latitude and longitude).
    
    CRITICAL RULE: Assume all locations, monuments, and points of interest are in Italy 
    unless the user explicitly specifies another country. For example, interpret ambiguous 
    terms like 'Coliseum' as the 'Colosseo' in Rome.
    
    If the request is entirely generic or has no location, use 'Rome' as default.

    User query
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

# Builds and run Langgraph graph
def create_event_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("search_sources", search_sources_node)

    # Define flow
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "search_sources")
    workflow.add_edge("search_sources", END)

    return workflow.compile()

# mock main
if __name__ == "__main__":

    print("=== Langgraph Test: Analyze Query Node ===")

    app = create_event_graph()

    initial_state = {
        "user_query": "I am looking for electronic near Coliseum this weekend",
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

    print("\n[SUCCESS] Final state values:")
    print(f"Target Location: {final_state['target_location']}")
    print(f"Latitude:        {final_state['latitude']}")
    print(f"Longitude:       {final_state['longitude']}")