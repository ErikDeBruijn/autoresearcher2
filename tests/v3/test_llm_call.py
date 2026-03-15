"""Tests for LLM call wrapper — JSON parsing only (no real SSH calls)."""
import json
import pytest
from autoresearcher2.v3.llm_call import parse_json_response


def test_parse_direct_json():
    raw = '{"beliefs_added": [{"claim": "test", "confidence": 0.5}]}'
    result = parse_json_response(raw)
    assert result["beliefs_added"][0]["claim"] == "test"


def test_parse_claude_wrapper_format():
    """Claude --output-format json wraps in {"result": "..."}."""
    inner = '{"proposals": [{"intent": "test"}]}'
    raw = json.dumps({"result": inner})
    result = parse_json_response(raw)
    assert result["proposals"][0]["intent"] == "test"


def test_parse_json_in_markdown():
    raw = '''Here is the analysis:

```json
{"delta": {"beliefs_revised": [{"id": "B1", "new_confidence": 0.9}]}}
```

That's my assessment.'''
    result = parse_json_response(raw)
    assert result["delta"]["beliefs_revised"][0]["new_confidence"] == 0.9


def test_parse_invalid_json_raises():
    with pytest.raises(ValueError, match="Could not parse JSON"):
        parse_json_response("This is just text, no JSON here")


def test_parse_json_with_whitespace():
    raw = '  \n  {"key": "value"}  \n  '
    result = parse_json_response(raw)
    assert result["key"] == "value"


def test_parse_wrapper_with_dict_result():
    """When result is already a dict (not string-wrapped)."""
    raw = json.dumps({"result": {"proposals": []}})
    result = parse_json_response(raw)
    assert result["proposals"] == []


def test_parse_claude_full_wrapper():
    """Real claude --output-format json response with many keys."""
    inner_json = '{"beliefs_added": [{"claim": "test", "confidence": 0.5}]}'
    wrapper = {
        "type": "result", "subtype": "success", "is_error": False,
        "result": f"```json\n{inner_json}\n```\n\nExtra text here.",
        "usage": {"input_tokens": 100},
    }
    result = parse_json_response(json.dumps(wrapper))
    assert result["beliefs_added"][0]["claim"] == "test"


def test_parse_json_with_trailing_text():
    """JSON in fences followed by LLM commentary."""
    wrapper = {
        "type": "result",
        "result": '\n\n```json\n{"proposals": [{"intent": "test"}]}\n```\n\nI hope this helps!',
    }
    result = parse_json_response(json.dumps(wrapper))
    assert result["proposals"][0]["intent"] == "test"
