from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import analyze_query_node, search_sources_node, extract_events_node

def create_agent_app(model=None, mcp_client=None):
    """Define Langgraph graph and models agent flow."""

    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("search_sources", search_sources_node)
    workflow.add_node("extract_events", extract_events_node)

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "search_sources")
    workflow.add_edge("search_sources", "extract_events")
    workflow.add_edge("extract_events", END)

    return workflow.compile()