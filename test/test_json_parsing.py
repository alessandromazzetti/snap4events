import json

import pytest

from nodes import parse_llm_json, salvage_truncated_events_json, extract_answer


# ---------------------------------------------------------
# parse_llm_json
# ---------------------------------------------------------

def test_parse_llm_json_plain_dict_passthrough():
    data = {"a": 1}
    assert parse_llm_json(data) is data


def test_parse_llm_json_valid_string():
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_llm_json_strips_markdown_fences():
    text = '```json\n{"a": 1}\n```'
    assert parse_llm_json(text) == {"a": 1}


def test_parse_llm_json_invalid_type_raises_value_error():
    with pytest.raises(ValueError):
        parse_llm_json(12345)


def test_parse_llm_json_malformed_json_raises_json_decode_error():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json('{"a": 1')


# ---------------------------------------------------------
# salvage_truncated_events_json
# ---------------------------------------------------------

def test_salvage_full_valid_response_recovers_all_events():
    text = json.dumps({
        "events": [
            {"title": "A"},
            {"title": "B"},
        ]
    })

    result = salvage_truncated_events_json(text)
    assert result == [{"title": "A"}, {"title": "B"}]


def test_salvage_truncated_mid_object_keeps_completed_events():
    # Third event is cut off mid-string, exactly like the real bug report
    truncated = (
        '{"events": ['
        '{"title": "Evento A", "start_datetime": "2026-09-05T20:00:00"}, '
        '{"title": "Evento B", "start_datetime": "2026-09-06T18:00:00"}, '
        '{"title": "Evento C", "start_datetime": "2026-09-07T10:00'
    )

    result = salvage_truncated_events_json(truncated)

    assert len(result) == 2
    assert result[0]["title"] == "Evento A"
    assert result[1]["title"] == "Evento B"


def test_salvage_no_array_returns_empty_list():
    assert salvage_truncated_events_json("not even json") == []


def test_salvage_empty_events_array_returns_empty_list():
    assert salvage_truncated_events_json('{"events": []}') == []


def test_salvage_strips_markdown_fences_before_parsing():
    text = '```json\n{"events": [{"title": "A"}]}\n```'
    assert salvage_truncated_events_json(text) == [{"title": "A"}]


def test_salvage_non_string_input_returns_empty_list():
    assert salvage_truncated_events_json(None) == []
    assert salvage_truncated_events_json({"already": "a dict"}) == []


# ---------------------------------------------------------
# extract_answer
# ---------------------------------------------------------

def test_extract_answer_plain_string_passthrough():
    assert extract_answer("hello") == "hello"


def test_extract_answer_openai_style_choices():
    response = {
        "choices": [
            {"message": {"content": "the answer"}}
        ]
    }
    assert extract_answer(response) == "the answer"


def test_extract_answer_openai_style_legacy_text_field():
    response = {"choices": [{"text": "legacy answer"}]}
    assert extract_answer(response) == "legacy answer"


@pytest.mark.parametrize("key", ["answer", "content", "response", "output", "text"])
def test_extract_answer_generic_fallback_keys(key):
    response = {key: "value here"}
    assert extract_answer(response) == "value here"


def test_extract_answer_unknown_shape_falls_back_to_json_dump():
    response = {"totally": "unexpected", "shape": 1}
    result = extract_answer(response)
    assert json.loads(result) == response


def test_extract_answer_non_str_non_dict_falls_back_to_str():
    assert extract_answer(42) == "42"
