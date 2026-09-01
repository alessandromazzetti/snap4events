import json

from nodes import analyze_query_node
from fakes import FakeLLMClient


async def test_analyze_query_success():
    llm = FakeLLMClient(response=json.dumps({
        "target_location": "Firenze",
        "latitude": 43.7697,
        "longitude": 11.2558,
    }))

    state = {"user_query": "eventi a Firenze", "llm_client": llm}
    result = await analyze_query_node(state)

    assert result["target_location"] == "Firenze"
    assert result["latitude"] == 43.7697
    assert result["longitude"] == 11.2558
    assert result["current_step"] == "query_analyzed"


async def test_analyze_query_llm_error_falls_back_to_florence():
    llm = FakeLLMClient(raise_error=RuntimeError("simulated API failure"))

    state = {"user_query": "eventi da qualche parte", "llm_client": llm}
    result = await analyze_query_node(state)

    assert result["target_location"] == "Florence"
    assert result["latitude"] == 43.7697
    assert result["longitude"] == 11.2556
    assert result["current_step"] == "query_analyzed"


async def test_analyze_query_malformed_json_falls_back_to_florence():
    llm = FakeLLMClient(response="this is not JSON at all")

    state = {"user_query": "eventi ovunque", "llm_client": llm}
    result = await analyze_query_node(state)

    assert result["target_location"] == "Florence"
    assert result["current_step"] == "query_analyzed"


async def test_analyze_query_missing_required_field_falls_back():
    # Valid JSON, but missing 'latitude'/'longitude' -> Pydantic validation fails
    llm = FakeLLMClient(response=json.dumps({"target_location": "Roma"}))

    state = {"user_query": "eventi a Roma", "llm_client": llm}
    result = await analyze_query_node(state)

    assert result["target_location"] == "Florence"
