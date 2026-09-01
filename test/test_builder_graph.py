from builder import create_agent_app


def test_all_expected_nodes_are_wired_into_the_graph():
    app = create_agent_app()
    graph = app.get_graph()

    expected_nodes = {
        "analyze_query",
        "search_db",
        "decide_retrieval",
        "search_sources",
        "extract_events",
        "save_events",
    }

    assert expected_nodes.issubset(set(graph.nodes.keys()))


def test_entry_point_is_analyze_query():
    app = create_agent_app()
    graph = app.get_graph()

    start_edges = [e for e in graph.edges if e.source == "__start__"]
    assert len(start_edges) == 1
    assert start_edges[0].target == "analyze_query"


def test_linear_edges_before_the_rag_decision():
    app = create_agent_app()
    edges = {(e.source, e.target) for e in app.get_graph().edges}

    assert ("analyze_query", "search_db") in edges
    assert ("search_db", "decide_retrieval") in edges


def test_decide_retrieval_has_a_conditional_branch_to_end_and_to_search():
    """This is the core RAG-routing regression test: if someone edits
    builder.py and accidentally removes one branch of the conditional edge,
    this test catches it even without running the full graph."""

    app = create_agent_app()
    graph = app.get_graph()

    decide_retrieval_edges = [e for e in graph.edges if e.source == "decide_retrieval"]
    targets = {e.target for e in decide_retrieval_edges}

    assert all(e.conditional for e in decide_retrieval_edges), (
        "The edges leaving decide_retrieval must be conditional, "
        "otherwise the RAG decision would have no effect on routing."
    )
    assert "__end__" in targets, "Accepting retrieved data must be able to skip straight to END"
    assert "search_sources" in targets, "Rejecting retrieved data must fall back to the web search branch"


def test_web_search_pipeline_edges_are_linear_and_unconditional():
    app = create_agent_app()
    edges_by_source = {}
    for e in app.get_graph().edges:
        edges_by_source.setdefault(e.source, []).append(e)

    for source, expected_target in [
        ("search_sources", "extract_events"),
        ("extract_events", "save_events"),
        ("save_events", "__end__"),
    ]:
        matching = [e for e in edges_by_source[source] if e.target == expected_target]
        assert matching, f"Expected an edge {source} -> {expected_target}"
        assert not matching[0].conditional, f"{source} -> {expected_target} should be unconditional"
