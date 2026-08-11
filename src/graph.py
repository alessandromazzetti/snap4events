from langgraph.graph import StateGraph, END
from state import AgentState

# Analyze user query to identify location
def analyze_query_node(state: AgentState) -> AgentState:
    query = state.get("user_query", "").lower()
    print(f"\n[GRAPH] Analysing user query...: '{state.get('user_query')}'")

    # Extraction logic TODO implement LLM
    if "milan" in query:
        target_location = "Milan"
        lat, lon = 45.4642, 9.1900
    elif "rome" in query:
        target_location = "Rome"
        lat, lon = 41.8931, 12.4828
    # Default
    else:
        target_location = "Florence"
        lat, lon = 43.7696, 11.2558

    print(f"[GRAPH] Location found: {target_location} (lat={lat}, lon={lon})")

    return {
        **state,
        "target_location": target_location,
        "latitude": lat,
        "longitude": lon,
        "current_step": "query_analyzed"
    }


# Builds and run Langgraph graph
def create_event_graph():
    workflow = StateGraph(AgentState)

    # 1. Analysing
    workflow.add_node("analyze_query", analyze_query_node)

    # 2. Set a graph point
    workflow.set_entry_point("analyze_query")

    workflow.add_edge("analyze_query", END)

    return workflow.compile()


if __name__ == "__main__":
    print("=== Langgraph Test ===")
    app = create_event_graph()

    # Initial state
    initial_state = {
        "user_query": "Finds events in Milan this weekend",
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
    print("\n[SUCCESS] Final state:", final_state)