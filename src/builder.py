from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    analyze_query_node,
    search_db_node,
    decide_retrieval_usage_node,
    route_after_retrieval_decision,
    search_sources_node,
    extract_events_node,
    save_events_node,
    export_events_pdf_node
)

def create_agent_app():
    """Define Langgraph graph and models agent flow."""

    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("search_db", search_db_node)
    workflow.add_node("decide_retrieval", decide_retrieval_usage_node)
    workflow.add_node("search_sources", search_sources_node)
    workflow.add_node("extract_events", extract_events_node)
    workflow.add_node("save_events", save_events_node)
    workflow.add_node("export_pdf", export_events_pdf_node)

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "search_db")
    workflow.add_edge("search_db", "decide_retrieval")

    # RAG decision point: the LLM decides whether the events already retrieved
    # from the database are enough, or whether a fresh web search is needed.
    workflow.add_conditional_edges(
        "decide_retrieval",
        route_after_retrieval_decision,
        {
            # Retrieved data judged sufficient -> events are already in state,
            # skip web search/scrape and go straight to END (no need to re-save
            # events that are already in the DB).
            "use_retrieved": END,
            # Retrieved data judged insufficient/absent -> fall back to the
            # original web search + scrape + save pipeline.
            "search_web": "search_sources",
        },
    )

    workflow.add_edge("search_sources", "extract_events")
    workflow.add_edge("extract_events", "save_events")
    workflow.add_edge("save_events", "export_pdf")
    workflow.add_edge("export_pdf", END)

    return workflow.compile()