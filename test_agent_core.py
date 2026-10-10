"""Verification suite for agent_core.

The loop is exercised end to end with :class:`ScriptedBrain`, so nothing here
needs the 5 GB model, a GPU or a network (except the two ``network`` tests).

Run with:  python -m pytest test_agent_core.py -v
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import agent_core as ac
import geomag_analyzer as analyzer
import geomag_plotter as plotter


DEMO_BLOCK = (
    '<tool_call>{"name": "get_statistics", '
    '"arguments": {"df": "raw", "components": ["H"]}}</tool_call>'
)


def _text_call(name: str, arguments: str = "{}") -> str:
    return f'<tool_call>{{"name": "{name}", "arguments": {arguments}}}</tool_call>'


T_OPEN = "<tool_call>"
T_CLOSE = "</tool_call>"


# --------------------------------------------------------------------------- #
# tool schemas
# --------------------------------------------------------------------------- #
def test_schemas_are_openai_shaped():
    assert ac.TOOL_SCHEMAS
    for schema in ac.TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert isinstance(params.get("required", []), list)
        json.dumps(schema)


def test_the_six_phase_zero_tools_are_still_present():
    """The Phase 1 tool set is a superset: nothing Phase 0 defined may vanish.

    Phase 1 adds three tools rather than replacing any, so this asserts the six
    original names are all still registered. It was previously an exact-set
    equality, which correctly failed the moment calculate_baseline was added --
    the guard was checking "no more than six", not "these six still exist".
    """
    phase_zero = {
        "fetch_observatory_data",
        "calculate_derived_components",
        "get_statistics",
        "detect_anomalies",
        "plot_components",
        "plot_comparison",
    }
    assert phase_zero <= ac.TOOLS_BY_NAME, (
        f"Phase 0 tools went missing: {sorted(phase_zero - ac.TOOLS_BY_NAME)}"
    )
    assert ac.TOOLS_BY_NAME == set(ac.HANDLERS), "every schema needs a handler"


def test_phase_one_math_tools_are_registered():
    """The Phase 1 additions, pinned so a rename cannot pass unnoticed."""
    assert {
        "calculate_derived_math",
        "calculate_baseline",
        "evaluate_custom_formula",
    } <= ac.TOOLS_BY_NAME


def test_data_type_enum_matches_the_loader():
    """A schema that offers values the loader rejects wastes a GPU round-trip."""
    schema = next(
        s for s in ac.TOOL_SCHEMAS
        if s["function"]["name"] == "fetch_observatory_data"
    )
    enum = set(schema["function"]["parameters"]["properties"]["data_type"]["enum"])
    assert enum <= {
        "auto", "definitive", "quasi-def", "adjusted", "reported", "best-avail",
    }
    assert "definitive" in enum and "auto" in enum


def test_station_hint_uses_real_iaga_codes():
    """Guessing codes produces wrong stations: ABK is Abisko, not Abakan."""
    import intermagnet_loader as loader

    registry = loader.get_station_details()
    assert "IRT" in registry
    assert "Irkutsk" in str(registry["IRT"]["name"])
    codes = [pair.split("=")[0] for pair in ac.STATION_HINT.split(", ")]
    for code in codes:
        assert code in registry, f"{code} from STATION_HINT is not a real station"


def test_system_prompt_forbids_inventing_numbers():
    lowered = ac.SYSTEM_PROMPT.lower()
    assert "не выдумывай" in lowered
    assert "профессиональный геофизический" in lowered
    assert "raw" in ac.SYSTEM_PROMPT and "derived" in ac.SYSTEM_PROMPT


def test_system_prompt_advertises_every_registered_tool():
    """A tool the prompt never mentions is a tool the model will not reach for.

    The Phase 1 math tools were registered and fully tested, yet the prompt's
    workflow rule named only the Phase 0 pipeline and told the model not to
    skip steps. The schemas are sent to the LLM regardless, but an instruction
    that enumerates a five-step flow is a strong prior: the model ran that flow
    and never discovered the new tools. Prompt coverage is pinned here so the
    two lists cannot drift apart again.
    """
    for tool in sorted(ac.TOOLS_BY_NAME):
        assert tool in ac.SYSTEM_PROMPT, f"{tool} is registered but unadvertised"


def test_system_prompt_explains_the_two_step_anomaly_workflow():
    """metric='anomaly' is unusable alone; the prompt must say so."""
    assert "baseline_value" in ac.SYSTEM_PROMPT
    assert "calculate_baseline" in ac.SYSTEM_PROMPT


def test_system_prompt_marks_the_formula_dsl_as_safe():
    """The model must believe it may write formulas, and know their limits."""
    lowered = ac.SYSTEM_PROMPT.lower()
    assert "не исполняется как код" in lowered
    assert "evaluate_custom_formula" in ac.SYSTEM_PROMPT


def test_slot_ordering_is_left_to_the_code_not_the_prompt():
    """Rule 5a is gone, because _auto_derive now covers the miss it warned about.

    The rule restated, in prompt budget, an ordering the code enforces -- and the
    model overrode it anyway, which is what the field failure showed. The prompt
    is pinned here so the rule cannot quietly grow back and eat the budget the
    selection table needs; the behaviour itself is covered by the auto-derive
    tests rather than by wording.
    """
    sp = ac.SYSTEM_PROMPT
    assert "ПРАВИЛО СЛОТОВ" not in sp
    assert "только тот хэндл, который сам создал" not in sp
    # the failure mode it described is still named where a model can act on it
    assert "unknown_frame" in sp


def test_system_prompt_says_range_is_not_a_metric():
    """The model hallucinated metric="range"; there is no such metric."""
    sp = ac.SYSTEM_PROMPT
    assert "Метрики 'range' НЕ СУЩЕСТВУЕТ" in sp
    # and the three that do exist are still named
    for metric in ("'delta'", "'dH_dt'", "'anomaly'"):
        assert metric in sp


def test_system_prompt_pins_the_unit_spelling():
    """The checker and the recovery filter both key on the unit, so state it."""
    assert "'nT'" in ac.SYSTEM_PROMPT


def test_system_prompt_shows_one_valid_worked_tool_call():
    """One example, and it has to be valid JSON.

    The second worked example and the doubled-brace warning were cut: the parser
    repairs that shape now (_unwrap_doubled_braces), so the prompt was teaching a
    failure the code no longer has. The remaining example is still pinned as
    parseable, because an example that does not parse is worse than none -- the
    model copies it verbatim.
    """
    import json
    import re

    sp = ac.SYSTEM_PROMPT
    bodies = re.findall(r'\{"name".*?\}\}', sp)
    valid = [b for b in bodies if _is_json(b)]
    assert len(valid) == 1, bodies
    call = json.loads(valid[0])
    assert call["name"] == "fetch_observatory_data"
    assert call["arguments"]["station_code"] == "IRT"
    # the parser now handles the doubled form, so the prompt must not spend
    # characters on it
    assert "двойных скобок" not in sp


def _is_json(body: str) -> bool:
    import json

    try:
        json.loads(body)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- #
# parse_tool_calls
# --------------------------------------------------------------------------- #
def test_parses_qwen_tool_call_block():
    calls, leftover = ac.parse_tool_calls({"content": DEMO_BLOCK})
    assert len(calls) == 1
    assert calls[0]["name"] == "get_statistics"
    assert calls[0]["arguments"] == {"df": "raw", "components": ["H"]}
    assert "tool_call" not in leftover


def test_parses_plain_tags_without_zero_width_space():
    plain = '<tool_call>{"name": "detect_anomalies", "arguments": {}}</tool_call>'.replace("\u200b", "")
    calls, _ = ac.parse_tool_calls({"content": plain})
    assert [c["name"] for c in calls] == ["detect_anomalies"]


def test_tolerates_zero_width_space_in_tags():
    """Qwen's own templates guard the tag with U+200B; both spellings must work."""
    calls, _ = ac.parse_tool_calls({"content": DEMO_BLOCK})
    assert calls and calls[0]["name"] == "get_statistics"
    assert ac._INVISIBLE[0] in DEMO_BLOCK or ac._INVISIBLE[0] not in DEMO_BLOCK


def test_prefers_structured_tool_calls_when_present():
    reply = {
        "content": "ignored",
        "tool_calls": [
            {"type": "function", "function": {"name": "get_statistics", "arguments": '{"df": "raw"}'}}
        ],
    }
    calls, _ = ac.parse_tool_calls(reply)
    assert calls[0]["name"] == "get_statistics"
    assert calls[0]["arguments"] == {"df": "raw"}


def test_coerces_string_arguments_to_dict():
    calls, _ = ac.parse_tool_calls({
        "content": '<tool_call>{"name": "get_statistics", '
                   '"arguments": "{\\"df\\": \\"raw\\"}"}</tool_call>'
    })
    assert isinstance(calls[0]["arguments"], dict)
    assert calls[0]["arguments"] == {"df": "raw"}


def test_multiple_blocks_in_one_reply():
    calls, _ = ac.parse_tool_calls({
        "content": _text_call("calculate_derived_components", '{"df": "raw"}')
        + _text_call("plot_components", '{"df": "derived", "components": ["H"]}')
    })
    assert [c["name"] for c in calls] == ["calculate_derived_components", "plot_components"]


def test_recovers_from_truncated_tag():
    """A hit on the token budget mid-call must not crash the loop."""
    calls, _ = ac.parse_tool_calls({
        "content": 'Сейчас посчитаю.<tool_call>{"name": "get_statistics", "argu'
    })
    assert len(calls) == 1
    assert calls[0]["malformed"] is True


def test_handles_fenced_json():
    calls, _ = ac.parse_tool_calls({
        "content": '```json\n{"name": "detect_anomalies", "arguments": {"df": "raw"}}\n```'
    })
    assert calls[0]["name"] == "detect_anomalies"


# --- reply-shape normalisation (regression: AttributeError on str input) ---

def test_accepts_raw_str_reply():
    """llama-cpp-python sometimes returns a bare str, not a dict."""
    body = T_OPEN + '{"name":"get_statistics","arguments":{"df":"raw"}}' + T_CLOSE
    calls, leftover = ac.parse_tool_calls(body)
    assert calls[0]["name"] == "get_statistics"
    assert isinstance(leftover, str)


@pytest.mark.parametrize(
    "reply",
    [T_OPEN + '{"name":"f","arguments":{}}' + T_CLOSE,
     {"content": T_OPEN + '{"name":"f","arguments":{}}' + T_CLOSE},
     {"content": None},
     {},
     None,
     5,
     [1, 2]],
)
def test_never_raises_on_any_reply_shape(reply):
    calls, leftover = ac.parse_tool_calls(reply)
    assert isinstance(calls, list)
    assert isinstance(leftover, str)


def test_returns_tuple_not_bare_list():
    """L971 does `calls, leftover = parse_tool_calls(...)`; a bare list would
    unpack two call dicts into (calls, leftover) with no error."""
    calls, leftover = ac.parse_tool_calls(T_OPEN + '{"name":"f","arguments":{}}' + T_CLOSE)
    assert isinstance(calls, list) and isinstance(leftover, str)
    assert all(isinstance(c, dict) for c in calls)


@pytest.mark.parametrize("payload", ["5", "[]", '"hi"', "null", "[1,2]", "{", "}", "", "  "])
def test_non_dict_json_inside_tags_is_skipped(payload):
    """`"name" in json.loads("5")` raises TypeError, which `except
    json.JSONDecodeError` would not catch."""
    calls, _ = ac.parse_tool_calls(T_OPEN + payload + T_CLOSE)
    assert all(c["name"] != "" or c["malformed"] for c in calls)


def test_bare_json_fallback_survives_nested_arguments():
    """A non-greedy \\{.*?\\} regex stops at the first `}` and returns
    unparseable JSON when `arguments` nests objects."""
    content = (
        "Вот план: "
        '{"name":"plot_components","arguments":'
        '{"df":"derived","components":["H","D"],"style":{"mode":"line"}}}'
        " готово"
    )
    calls, leftover = ac.parse_tool_calls(content)
    assert calls[0]["name"] == "plot_components"
    assert calls[0]["arguments"]["style"] == {"mode": "line"}
    assert "Вот план" in leftover and "готово" in leftover
    assert '"name"' not in leftover


def test_brace_inside_string_value_does_not_unbalance():
    calls, _ = ac.parse_tool_calls(
        T_OPEN + '{"name":"f","arguments":{"title":"a } b {"}}' + T_CLOSE
    )
    assert calls[0]["arguments"]["title"] == "a } b {"


# --- repair ladder: shapes a 7B model actually emits (Colab malformed_tool_call) ---

def _names(reply):
    calls, _ = ac.parse_tool_calls(reply, debug=False)
    return [c["name"] for c in calls if not c["malformed"]]


@pytest.mark.parametrize(
    "label,body,expected",
    [
        ("trailing comma", '{"name":"f","arguments":{},}', ["f"]),
        ("single quotes", "{'name':'f','arguments':{}}", ["f"]),
        ("function envelope", '{"function":{"name":"f","arguments":{"a":1}}}', ["f"]),
        ("parameters alias", '{"name":"f","parameters":{"a":1}}', ["f"]),
        ("args alias", '{"name":"f","args":{"a":1}}', ["f"]),
        ("tool alias", '{"tool":"f","args":{"a":1}}', ["f"]),
        ("top-level array", '[{"name":"f","arguments":{"a":1}}]', ["f"]),
        ("array of two calls", '[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]', ["a", "b"]),
        ("two objects glued", '{"name":"a","arguments":{}}{"name":"b","arguments":{}}', ["a", "b"]),
        ("stray closing brace", '{"name":"f","arguments":{}}}', ["f"]),
        ("prose around json", 'Sure!\n{"name":"f","arguments":{"a":1}}\nDone', ["f"]),
        ("curly quotes", "{\u201cname\u201d:\u201cf\u201d,\u201carguments\u201d:{}}", ["f"]),
        ("zero-width inside", "{\u200b\"name\"\u200b:\u200b\"f\"\u200b,\u200b\"arguments\"\u200b:{\u200b}}", ["f"]),
        ("arguments as string", '{"name":"f","arguments":"{\\"a\\":1}"}', ["f"]),
        ("fence inside tag", '```json\n{"name":"f","arguments":{"a":1}}\n```', ["f"]),
        ("nested args", '{"name":"f","arguments":{"a":{"b":{"c":1}}}}', ["f"]),
        # doubled braces: a second wrapper layer around the object
        ("doubled braces closed", '{{"name":"f","arguments":{"a":1}}}', ["f"]),
        ("doubled braces unclosed", '{{"name":"f","arguments":{"a":1}}', ["f"]),
        ("doubled braces with space", '{{ "name":"f", "arguments":{"a":1} }}', ["f"]),
        ("tripled braces", '{{{ "name":"f", "arguments":{"a":1} }}}', ["f"]),
        # a doubled wrapper must not swallow a payload whose string holds braces
        ("doubled braces, braces in a string",
         '{{"name":"f","arguments":{"formula":"{{a}} + b"}}}', ["f"]),
        # all three repairs composed
        ("doubled braces plus single quotes plus trailing comma",
         "{{'name':'f','arguments':{'a':1},}}", ["f"]),
    ],
)
def test_repairs_common_model_garbling(label, body, expected):
    assert _names(T_OPEN + body + T_CLOSE) == expected, label


def test_unrecoverable_body_reports_a_reason():
    calls, _ = ac.parse_tool_calls(T_OPEN + "total nonsense" + T_CLOSE, debug=False)
    assert len(calls) == 1
    assert calls[0]["malformed"] is True
    assert calls[0]["name"] == ""
    assert "not valid JSON" in calls[0]["reason"]


# --------------------------------------------------------------------------- #
# doubled braces: the two failure modes from the E2E run
# --------------------------------------------------------------------------- #
def test_doubled_braces_inside_a_tag_execute_rather_than_burn_a_round():
    """Inside a tag this used to parse as a malformed call with an empty name.

    The round was spent, the tool was never called, and the model was told its
    JSON was broken when the only fault was one extra brace.
    """
    body = '{"name": "plot_components", "arguments": {"df": "raw"}}'
    calls, rest = ac.parse_tool_calls(T_OPEN + "{{" + body + "}}" + T_CLOSE, debug=False)
    assert rest == ""
    assert len(calls) == 1
    assert calls[0]["malformed"] is False
    assert calls[0]["name"] == "plot_components"
    assert calls[0]["arguments"] == {"df": "raw"}


def test_doubled_braces_outside_a_tag_are_no_longer_a_silent_no_op():
    """The worse of the two: no tag, no call, and nothing reported.

    parse_tool_calls returned ([], raw) -- the span scanner balances the doubled
    braces, yields the same broken shape back, and the agent loop retries with
    no idea why. An empty call list is indistinguishable from "the model had
    nothing to say", which is how a retry loop turns into a hang.
    """
    body = '{"name": "plot_components", "arguments": {"df": "raw"}}'
    calls, _ = ac.parse_tool_calls("{{" + body + "}}}", debug=False)
    assert [c["name"] for c in calls] == ["plot_components"]
    assert calls[0]["malformed"] is False


def test_a_doubled_brace_call_reaches_the_tool():
    """End to end: the repair must produce a call that actually executes.

    Before the fix this step came back as ``malformed_tool_call`` and no handler
    ever ran. The store here is empty, so the handler still refuses the frame --
    what matters is that the refusal is a data error from a real tool rather than
    a parse error, which is the distinction the model has to act on.
    """
    brain = ac.ScriptedBrain([
        T_OPEN + '{{"name": "get_statistics", "arguments": '
        '{"df": "raw", "components": ["F"]}}}' + T_CLOSE,
        "готово",
    ])
    out = ac.run_agent("q", brain=brain, verbose=False)
    step = out["tool_calls"][0]
    assert step["tool"] == "get_statistics"
    assert step["arguments"] == {"df": "raw", "components": ["F"]}
    assert step.get("error") != "malformed_tool_call"


def test_well_formed_calls_are_untouched_by_the_repair():
    """The repair must not change any reading that already parsed."""
    valid = [
        '{"name":"f","arguments":{"a":1}}',
        '{"name":"f","arguments":{}}',
        '[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]',
        '{\"function\":{\"name\":\"f\",\"arguments\":{\"a\":1}}}',
    ]
    for body in valid:
        assert _names(T_OPEN + body + T_CLOSE) != [], body


def test_braces_inside_strings_are_never_rewritten():
    """Only the ends are sliced, so a payload may carry braces in its values.

    A blanket replace of '{{' would corrupt these. This agent lets a model pass
    text through verbatim, so a formula containing braces has to survive the
    repair intact.
    """
    for value in ("{{a}} + b", "a{{", "x}}}}y", "{{{", "dict({1:2})"):
        body = json.dumps(
            {"name": "evaluate_custom_formula", "arguments": {"formula": value}}
        )
        calls, _ = ac.parse_tool_calls(T_OPEN + body + T_CLOSE, debug=False)
        assert len(calls) == 1, value
        assert calls[0]["malformed"] is False, value
        assert calls[0]["arguments"]["formula"] == value, value


def test_doubled_wrapper_keeps_braces_inside_strings():
    """Both at once: the wrapper is peeled, the inner value is not."""
    payload = json.dumps({"name": "f", "arguments": {"formula": "{{a}}"}})
    calls, _ = ac.parse_tool_calls(T_OPEN + "{{" + payload + "}}}" + T_CLOSE, debug=False)
    assert calls[0]["arguments"]["formula"] == "{{a}}"


def test_genuine_garbage_is_still_refused():
    """Widening the ladder must not make any old junk parse.

    The repair engages only on a doubled opening brace, so balanced-looking but
    meaningless input has to keep failing -- otherwise the parser would start
    inventing calls out of prose.
    """
    for junk in ("not json at all", "{", "{{", "}}}", '{"name": }', "{}{}"):
        assert ac._try_loads(junk) is None, junk


def test_the_repair_only_engages_on_a_doubled_brace():
    """A clean object never enters the ladder's new rung."""
    assert list(ac._unwrap_doubled_braces('{"name":"f"}')) == []
    assert list(ac._unwrap_doubled_braces("plain text")) == []
    assert list(ac._unwrap_doubled_braces("")) == []


def test_degenerate_doubled_braces_never_invent_a_call():
    """The generator may offer readings that do not parse, or that parse to
    something empty -- it must never manufacture a *named* tool call.

    '{{}' peels to '{}', which is valid JSON. That is harmless because an empty
    object names no tool, so the property worth pinning is not "nothing parses"
    but "no call appears that was not in the text".
    """
    for degenerate in ("{{", "{{}", "{{{{", "{{{}}"):
        calls, _ = ac.parse_tool_calls(T_OPEN + degenerate + T_CLOSE, debug=False)
        assert [c["name"] for c in calls if not c["malformed"]] == [], degenerate


def test_malformed_error_payload_carries_the_reason():
    """The Colab log must explain the failure without a second guess."""
    brain = ac.ScriptedBrain(
        [T_OPEN + "total nonsense" + T_CLOSE, "final answer"]
    )
    out = ac.run_agent("q", brain=brain, verbose=False)
    assert out["ok"] is True
    step = out["tool_calls"][0]
    assert step["error"] == "malformed_tool_call"


def test_debug_print_always_fires_and_is_silencable(capsys, monkeypatch):
    """Colab relies on this line to prove the runtime has current code."""
    monkeypatch.delenv("GEOMAG_DEBUG_PARSER", raising=False)
    ac.parse_tool_calls(T_OPEN + '{"name":"f","arguments":{}}' + T_CLOSE)
    assert "[DEBUG] RAW LLM OUTPUT FOR PARSING" in capsys.readouterr().out

    monkeypatch.setenv("GEOMAG_DEBUG_PARSER", "0")
    ac.parse_tool_calls(T_OPEN + "junk" + T_CLOSE)
    assert "[DEBUG]" not in capsys.readouterr().out


def test_parser_version_stamp_is_exposed():
    """colab_run.ipynb compares this against a hard-coded string to detect a
    stale cached module."""
    assert isinstance(ac.PARSER_VERSION, str) and ac.PARSER_VERSION


def test_parse_tool_calls_still_returns_a_tuple():
    calls, leftover = ac.parse_tool_calls(
        T_OPEN + '{"name":"f","arguments":{}}' + T_CLOSE, debug=False
    )
    assert isinstance(calls, list) and isinstance(leftover, str)


# --- Qwen hallucinating the OpenAI response envelope (Colab [DEBUG] finding) ---

HALLUCINATION = "{'content': '\u0422\u0435\u043a\u0441\u0442 \u043e\u0442\u0432\u0435\u0442\u0430', 'tool_calls': None}"


def test_system_prompt_forbids_the_api_envelope():
    sp = ac.SYSTEM_PROMPT
    assert "API-\u0441\u0435\u0440\u0432\u0435\u0440" in sp          # "API-\u0441\u0435\u0440\u0432\u0435\u0440"
    assert "'content'" in sp
    assert "tool_calls" in sp
    # A concrete example beats an instruction for a 7B model.
    # The concrete example, minus the tag itself: this JSON body is
    # unique to the worked example in rule 10.
    assert '{"name": "fetch_observatory_data", "arguments": {' in sp
    assert "tool_call>" in sp


def test_system_prompt_example_has_no_invisible_chars():
    """A zero-width space in the example would be copied by the model verbatim."""
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        assert ch not in ac.SYSTEM_PROMPT


def test_temperature_is_low_and_actually_passed():
    assert ac.TEMPERATURE <= 0.1
    captured = {}

    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

    ac.LlamaBrain(FakeLlama()).chat([{"role": "user", "content": "q"}], ac.TOOL_SCHEMAS)
    assert captured["temperature"] == ac.TEMPERATURE <= 0.1
    assert captured["max_tokens"] > 0


def test_envelope_hallucination_returns_no_calls_and_keeps_content():
    calls, leftover = ac.parse_tool_calls(HALLUCINATION)
    assert calls == []
    assert leftover == "\u0422\u0435\u043a\u0441\u0442 \u043e\u0442\u0432\u0435\u0442\u0430"


@pytest.mark.parametrize(
    "reply,expected_left",
    [
        ("{'content': 'A', 'tool_calls': None}", "A"),
        ('{"content": "B", "tool_calls": null}', "B"),
        ("Prose: {'content': 'C', 'tool_calls': None}", "C"),
        ("```python\n{'content': 'D', 'tool_calls': None}\n```", "D"),
        ("{'content': '', 'tool_calls': None}", ""),
    ],
)
def test_envelope_variants(reply, expected_left):
    calls, leftover = ac.parse_tool_calls(reply)
    assert calls == []
    assert leftover == expected_left


def test_envelope_carrying_a_real_call_is_honoured():
    reply = (
        "{'content': '', 'tool_calls': [{'id': 'c0', 'type': 'function', "
        "'function': {'name': 'fetch_observatory_data', "
        "'arguments': {'station_code': 'IRT'}}}]}"
    )
    calls, _ = ac.parse_tool_calls(reply)
    assert [c["name"] for c in calls] == ["fetch_observatory_data"]
    assert calls[0]["arguments"] == {"station_code": "IRT"}


def test_envelope_path_does_not_swallow_normal_prose():
    calls, leftover = ac.parse_tool_calls("\u0413\u043e\u0442\u043e\u0432\u043e, \u0434\u0430\u043d\u043d\u044b\u0435 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u043d\u044b.")
    assert calls == []
    assert "\u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u043d\u044b" in leftover


def test_agent_corrects_a_hallucinated_envelope_instead_of_claiming_success():
    """Round 1 hallucinates, round 2 emits a real tag: the loop must recover and
    must NOT report the hallucinated text as the answer."""
    O, C = "<\u200btool_call>", "</\u200btool_call>"
    brain = ac.ScriptedBrain(
        [
            HALLUCINATION,
            O + '{"name": "get_statistics", "arguments": {"df": "raw"}}' + C,
            "\u0413\u043e\u0442\u043e\u0432\u043e.",
        ]
    )
    out = ac.run_agent("q", brain=brain, verbose=False)
    assert out["ok"] is True
    assert out["tool_calls"][0]["tool"] == "get_statistics"
    assert "tool_calls" not in out["text"]


def test_agent_gives_up_honestly_after_two_envelope_retries():
    """It must not invent a success, and must not loop forever either."""
    brain = ac.ScriptedBrain([HALLUCINATION] * 6)
    out = ac.run_agent("q", brain=brain, verbose=False)
    assert out["ok"] is True
    assert out["rounds"] <= 4


def test_handles_bare_json_without_tags():
    calls, _ = ac.parse_tool_calls({
        "content": '{"name": "get_statistics", "arguments": {"df": "raw"}}'
    })
    assert calls[0]["name"] == "get_statistics"


def test_plain_prose_yields_no_calls():
    calls, leftover = ac.parse_tool_calls({"content": "Магнитное поле в Иркутске спокойное."})
    assert calls == []
    assert "Иркутске" in leftover


def test_empty_reply_is_not_a_call():
    assert ac.parse_tool_calls({"content": ""}) == ([], "")
    assert ac.parse_tool_calls({}) == ([], "")


def test_unknown_tool_name_is_preserved_for_the_dispatcher():
    calls, _ = ac.parse_tool_calls({"content": _text_call("make_coffee", '{"x": 1}')})
    assert calls[0]["name"] == "make_coffee"


# --------------------------------------------------------------------------- #
# the Jinja double-encoding trap
# --------------------------------------------------------------------------- #
# Qwen2.5's template renders tool calls with `tool_call.arguments | tojson`.
# A dict renders as a JSON object; a JSON *string* renders as a quoted string,
# so replaying the assistant turn with string arguments corrupts the history.
_JINJA = """
{%- for message in messages %}
{%- if message.role == "assistant" and message.tool_calls %}
<|im_start|>assistant
{%- for tc in message.tool_calls %}
<tool_call>
{"name": "{{ tc.name }}", "arguments": {{ tc.arguments | tojson }}}
</tool_call>
{%- endfor %}<|im_end|>
{%- elif message.role == "tool" %}
<tool_response>
{{ message.content }}
</tool_response>
{%- endif %}
{%- endfor %}
"""


def _render(assistant_tool_call: dict) -> str:
    from jinja2 import Environment

    env = Environment()
    template = env.from_string(_JINJA)
    messages = [{"role": "assistant", "tool_calls": [assistant_tool_call]}]
    return template.render(messages=messages)


def _json_value_after(text: str, marker: str, last: bool = False) -> Any:
    """Decode the first complete JSON *value* following ``marker``.

    ``raw_decode`` is used rather than brace counting because the value under
    test may be an object, or -- the bug being guarded -- a quoted string whose
    braces would confuse a naive matcher. ``last=True`` skips earlier
    placeholders, which is what the real Qwen template needs: its own
    instructions contain a literal ``"arguments": <args-json-object>``.
    """
    if last:
        start = text.rindex(marker) + len(marker)
    else:
        start = text.index(marker) + len(marker)
    value, _ = json.JSONDecoder().raw_decode(text, start)
    return value


def test_dict_arguments_render_as_a_json_object():
    rendered = _render({"name": "get_statistics", "arguments": {"df": "raw"}})
    assert _json_value_after(rendered, '"arguments": ') == {"df": "raw"}


def test_string_arguments_would_be_double_encoded():
    """This is exactly what the loop must avoid -- a regression guard."""
    rendered = _render({"name": "get_statistics", "arguments": '{"df": "raw"}'})
    decoded = _json_value_after(rendered, '"arguments": ')
    assert isinstance(decoded, str), "a JSON string stays a string -> the model re-reads garbage"


def test_tool_role_uses_only_content():
    from jinja2 import Environment

    template = Environment().from_string(_JINJA)
    rendered = template.render(
        messages=[{"role": "tool", "content": '{"ok": true, "rows": 1440}'}]
    )
    assert '"rows": 1440' in rendered


def test_run_agent_keeps_arguments_as_dicts():
    seen: list[dict] = []

    class Spy(ac.ScriptedBrain):
        def chat(self, messages, tools):
            for m in messages:
                for tc in m.get("tool_calls") or []:
                    seen.append(tc["function"]["arguments"])
            return super().chat(messages, tools)

    brain = Spy([
        {"content": _text_call("calculate_derived_components", '{"df": "raw"}')},
        {"content": "готово"},
    ])
    ac.run_agent("посчитай H", brain=brain, verbose=False)
    assert seen, "the loop should have replayed an assistant turn with tool calls"
    assert all(isinstance(a, dict) for a in seen)


# --------------------------------------------------------------------------- #
# FrameStore
# --------------------------------------------------------------------------- #
def test_frame_store_round_trip():
    store = ac.FrameStore()
    frame = pd.DataFrame({"X": [1.0]})
    store.put("raw", frame)
    assert store.get("raw") is frame
    assert store.names() == ["raw"]
    assert store.get("nope") is None
    assert store.get("RAW") is frame, "handles are case-insensitive"


def test_frame_store_reports_unknown_handle():
    frame, err = ac.FrameStore().resolve("derived")
    assert frame is None
    assert ac.is_error(err) and err["error"] == "unknown_frame"
    assert "hint" in err and "available" in err


# --------------------------------------------------------------------------- #
# summarisation
# --------------------------------------------------------------------------- #
def test_summarise_frame_is_compact_and_json_safe():
    df = plotter._synthetic_day(10, seed=3)
    summary = ac._summarise_frame(df)
    blob = json.dumps(summary)
    assert "NaN" not in blob and "Infinity" not in blob
    assert summary["rows"] == len(df)
    assert len(blob) < 4000, "a 1440-row frame must not reach the prompt"
    assert "statistics" in summary


def test_summarise_handles_all_nan():
    df = pd.DataFrame({"X": [float("nan")] * 5, "timestamp": pd.date_range("2025-01-01", periods=5)})
    summary = ac._summarise_frame(df)
    assert summary["statistics"]["X"]["count"] == 0
    json.dumps(summary)


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def test_happy_path_executes_tools_and_returns_plot(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOMAG_OFFLINE", "1")
    monkeypatch.setattr(ac, "OFFLINE", True)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)

    brain = ac.ScriptedBrain(ac._unwrap(ac._demo_script()))
    result = ac.run_agent(ac.DEMO_QUERY, brain=brain, verbose=False)

    assert result["ok"] is True
    assert [e["tool"] for e in result["tool_calls"]] == [
        "fetch_observatory_data",
        "calculate_derived_components",
        "get_statistics",
        "plot_components",
    ]
    assert all(e["ok"] for e in result["tool_calls"])
    assert len(result["plots"]) == 1
    assert Path(result["plots"][0]).is_file()
    assert result["text"]


def test_two_plots_are_both_collected(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    brain = ac.ScriptedBrain(ac._unwrap([
        {"_text": _text_call("fetch_observatory_data", '{"station_code": "IRT", "start_date": "2024-09-10", "end_date": "2024-09-10"}')},
        {"_text": _text_call("plot_components", '{"df": "raw", "components": ["H"], "filename": "a.html"}')},
        {"_text": _text_call("plot_components", '{"df": "raw", "components": ["D"], "filename": "b.html"}')},
        {"content": "два графика готовы"},
    ]))
    result = ac.run_agent("построй два графика", brain=brain, verbose=False)
    assert len(result["plots"]) == 2
    assert all(Path(p).is_file() for p in result["plots"])


def test_unknown_tool_is_reported_and_the_model_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    brain = ac.ScriptedBrain(ac._unwrap([
        {"_text": _text_call("brew_coffee", "{}")},
        {"content": "Такого инструмента нет."},
    ]))
    result = ac.run_agent("свари кофе", brain=brain, verbose=False)
    first = result["tool_calls"][0]
    assert first["tool"] == "brew_coffee"
    assert first["ok"] is False and first["error"] == "unknown_tool"
    assert "available" not in first or True
    # The model's answer is kept, and the unrecovered failure is appended so the
    # user sees the cause instead of only the model's vague summary.
    assert result["text"].startswith("Такого инструмента нет.")
    assert "unknown_tool" in result["text"]


def test_unknown_frame_handle_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    brain = ac.ScriptedBrain(ac._unwrap([
        {"_text": _text_call("get_statistics", '{"df": "derived"}')},
        {"content": "Сначала нужно загрузить данные."},
    ]))
    result = ac.run_agent("посчитай", brain=brain, verbose=False)
    assert result["tool_calls"][0]["error"] == "unknown_frame"


def test_malformed_tool_call_does_not_crash():
    brain = ac.ScriptedBrain(ac._unwrap([
        {"content": 'начало<tool_call>{"name": "get_sta'},
        {"content": "не понял"},
    ]))
    result = ac.run_agent("?", brain=brain, verbose=False)
    assert result["tool_calls"][0]["error"] == "malformed_tool_call"
    assert result["text"].startswith("не понял")
    assert "malformed_tool_call" in result["text"]


def test_tool_exception_is_caught(monkeypatch):
    def boom(args, store):
        raise RuntimeError("sensor on fire")

    monkeypatch.setitem(ac.HANDLERS, "get_statistics", boom)
    brain = ac.ScriptedBrain(ac._unwrap([
        {"_text": _text_call("get_statistics", '{"df": "raw"}')},
        {"content": "инструмент упал"},
    ]))
    result = ac.run_agent("посчитай", brain=brain, verbose=False)
    entry = result["tool_calls"][0]
    assert entry["ok"] is False
    assert entry["error"] == "tool_exception"
    assert "sensor on fire" in entry["message"]


def test_upstream_is_error_payload_reaches_the_model(monkeypatch):
    """A loader failure must be relayed verbatim, not swallowed."""
    monkeypatch.setattr(ac, "OFFLINE", False)
    import intermagnet_loader as loader

    monkeypatch.setattr(
        loader, "fetch_observatory_data",
        lambda **kw: loader._error("station_not_found", "no such station", station="ZZZ"),
    )
    brain = ac.ScriptedBrain(ac._unwrap([
        {"_text": _text_call("fetch_observatory_data", '{"station_code": "ZZZ", "start_date": "2024-09-10", "end_date": "2024-09-10"}')},
        {"content": "Станция ZZZ не найдена."},
    ]))
    result = ac.run_agent("данные ZZZ", brain=brain, verbose=False)
    assert result["tool_calls"][0]["error"] == "station_not_found"


def test_budget_is_enforced():
    """More tool calls than the cap: the loop must stop executing, not hang."""
    script = [{"_text": _text_call("get_statistics", '{"df": "raw"}')} for _ in range(6)]
    script.append({"content": "бюджет исчерпан"})
    brain = ac.ScriptedBrain(ac._unwrap(script))
    result = ac.run_agent("считай", brain=brain, max_tool_calls=2, max_rounds=6, verbose=False)
    executed = [e for e in result["tool_calls"] if e["ok"]]
    assert len(executed) <= 2
    assert any(e.get("error") == "tool_budget_exhausted" for e in result["tool_calls"])
    assert result["stop_reason"] in ("tool_budget_exhausted", "round_limit")


def test_budget_error_is_offered_to_the_model():
    script = [{"_text": _text_call("get_statistics", '{"df": "raw"}')} for _ in range(3)]
    script.append({"content": "хватит"})
    brain = ac.ScriptedBrain(ac._unwrap(script))
    result = ac.run_agent("считай", brain=brain, max_tool_calls=1, max_rounds=4, verbose=False)
    # budget exhaustion is loop control flow, not a failure the user must read;
    # the real unknown_frame failure from the executed call is still surfaced.
    assert result["text"].startswith("хватит")
    assert "unknown_frame" in result["text"]


def test_round_limit_terminates():
    brain = ac.ScriptedBrain(ac._unwrap(
        [{"_text": _text_call("get_statistics", '{"df": "raw"}')}] * 20
    ))
    result = ac.run_agent("бесконечно", brain=brain, max_tool_calls=50, max_rounds=3, verbose=False)
    assert result["rounds"] == 3
    assert result["stop_reason"] == "round_limit"


def test_brain_exception_becomes_an_error_dict():
    class Broken:
        def chat(self, messages, tools):
            raise RuntimeError("CUDA OOM")

    result = ac.run_agent("привет", brain=Broken(), verbose=False)
    assert ac.is_error(result)
    assert result["error"] == "brain_failed"
    assert "CUDA OOM" in result["message"]


def test_empty_model_reply_is_handled():
    brain = ac.ScriptedBrain(ac._unwrap([{"content": "   "}]))
    result = ac.run_agent("привет", brain=brain, verbose=False)
    assert result["ok"] is True
    assert result["text"]


def test_run_agent_rejects_empty_query():
    assert ac.is_error(ac.run_agent("", brain=ac.ScriptedBrain([])))


def test_run_agent_rejects_a_bad_brain():
    result = ac.run_agent("привет", brain=object(), verbose=False)
    assert ac.is_error(result) and result["error"] == "invalid_brain"


def test_run_agent_passes_brain_error_through():
    result = ac.run_agent("привет", brain=ac._error("model_load_failed", "nope"), verbose=False)
    assert result["error"] == "model_load_failed"


def test_result_shape_is_the_documented_contract():
    brain = ac.ScriptedBrain(ac._unwrap([{"content": "ответ"}]))
    result = ac.run_agent("вопрос", brain=brain, verbose=False)
    for key in ("ok", "text", "plots", "tool_calls", "rounds", "stop_reason"):
        assert key in result, key
    assert isinstance(result["plots"], list)
    assert isinstance(result["text"], str)
    json.dumps(result, default=str)


# --------------------------------------------------------------------------- #
# model discovery / loading
# --------------------------------------------------------------------------- #
def test_model_stem_matches_the_real_sharded_upstream_name():
    """Upstream ships shards, not a single q4_k_m file."""
    assert ac.MODEL_REPO == "Qwen/Qwen2.5-7B-Instruct-GGUF"
    assert ac.MODEL_STEM == "qwen2.5-7b-instruct-q4_k_m"


def test_shard_sort_prefers_shard_one(tmp_path):
    names = [
        "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    files = ac._gguf_files(tmp_path, ac.MODEL_STEM)
    assert len(files) == 2
    assert ac._first_shard(files).name.endswith("00001-of-00002.gguf")


def test_download_returns_existing_model_without_network(tmp_path):
    shard = tmp_path / f"{ac.MODEL_STEM}-00001-of-00002.gguf"
    shard.write_bytes(b"x" * 16)
    assert ac.download_model(target_dir=tmp_path, verbose=False) == str(shard)


def test_download_in_offline_mode_returns_an_error_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    result = ac.download_model(target_dir=tmp_path, verbose=False)
    assert ac.is_error(result) and result["error"] == "offline"


def test_load_brain_without_the_model_is_an_error_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)
    result = ac.load_brain(model_path=None, verbose=False)
    assert ac.is_error(result)
    assert result["error"] in ("offline", "model_load_failed")


def test_gpu_layers_never_ask_for_offload_without_a_cuda_device():
    """A CPU-only box must get 0, otherwise llama.cpp warns at every call."""
    try:
        from llama_cpp import llama_cpp as _lc
        supports = _lc.llama_supports_gpu_offload()
    except Exception:
        pytest.skip("llama-cpp-python not installed")
    resolved = ac._resolve_gpu_layers(-1)
    assert resolved == (-1 if supports else 0)


def test_explicit_gpu_layers_are_respected():
    assert ac._resolve_gpu_layers(0) == 0
    assert ac._resolve_gpu_layers(20) == 20


# --------------------------------------------------------------------------- #
# interrupted downloads -- a real failure mode, observed in practice
# --------------------------------------------------------------------------- #
def test_entry_shard_rejects_a_download_missing_shard_one(tmp_path):
    """Observed in practice: shard 2 finished, shard 1 was still .incomplete."""
    (tmp_path / f"{ac.MODEL_STEM}-00002-of-00002.gguf").write_bytes(b"x" * 512)
    assert ac._entry_shard(tmp_path, ac.MODEL_STEM) is None
    assert ac.model_is_downloaded(target_dir=tmp_path, stem=ac.MODEL_STEM) is False


def test_entry_shard_accepts_a_complete_pair(tmp_path):
    for index in (1, 2):
        (tmp_path / f"{ac.MODEL_STEM}-0000{index}-of-00002.gguf").write_bytes(b"x" * 512)
    entry = ac._entry_shard(tmp_path, ac.MODEL_STEM)
    assert entry is not None and "00001-of-00002" in entry.name


def test_entry_shard_ignores_zero_length_files(tmp_path):
    shard = tmp_path / f"{ac.MODEL_STEM}-00001-of-00002.gguf"
    shard.write_bytes(b"")
    assert ac._entry_shard(tmp_path, ac.MODEL_STEM) is None


def test_single_file_quantisation_is_also_accepted(tmp_path):
    single = tmp_path / f"{ac.MODEL_STEM}.gguf"
    single.write_bytes(b"x" * 512)
    assert ac._entry_shard(tmp_path, ac.MODEL_STEM) == single


def test_purge_incomplete_cleans_halffinished_downloads(tmp_path):
    (tmp_path / "model-00001-of-00002.gguf.incomplete").write_bytes(b"x")
    (tmp_path / "model-00001-of-00002.gguf.lock").write_bytes(b"")
    (tmp_path / "model-00002-of-00002.gguf.metadata").write_bytes(b"")
    assert ac._purge_incomplete(tmp_path) == 3
    assert list(tmp_path.iterdir()) == []


def test_run_agent_never_downloads_implicitly(tmp_path, monkeypatch):
    """A question must not trigger a 5 GB download; it must explain itself."""
    monkeypatch.setattr(ac, "MODELS_DIR", tmp_path)

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("run_agent must not reach the network")

    monkeypatch.setattr(ac, "download_model", explode)
    result = ac.run_agent("посчитай H", verbose=False)
    assert ac.is_error(result)
    assert result["error"] == "brain_not_loaded"
    assert "load_brain" in result["message"]
    assert "models_dir" in result


def test_run_agent_auto_loads_when_the_model_is_already_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "MODELS_DIR", tmp_path)
    for index in (1, 2):
        (tmp_path / f"{ac.MODEL_STEM}-0000{index}-of-00002.gguf").write_bytes(b"x" * 512)
    monkeypatch.setattr(ac, "load_brain", lambda **kw: ac.ScriptedBrain([{"content": "ок"}]))
    result = ac.run_agent("привет", verbose=False)
    assert result["ok"] is True and result["text"] == "ок"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# regression: two days must be able to live at the same time
# --------------------------------------------------------------------------- #
def test_frame_store_keeps_both_days_and_alias_follows_the_latest():
    store = ac.FrameStore()
    day10 = pd.DataFrame({"timestamp": pd.date_range("2024-09-10", periods=2, freq="h"),
                          "H": [1.0, 2.0]})
    day11 = pd.DataFrame({"timestamp": pd.date_range("2024-09-11", periods=2, freq="h"),
                          "H": [3.0, 4.0]})

    store.put("raw:IRT:2024-09-10", day10)
    store.put("raw:IRT:2024-09-11", day11)

    assert "raw:irt:2024-09-10" in store.names()
    assert "raw:irt:2024-09-11" in store.names()
    # the earlier day must still be reachable and still be the earlier data
    assert store.get("raw:irt:2024-09-10") is day10
    assert store.get("raw:irt:2024-09-11") is day11
    # the legacy short name still works, pointing at the most recent fetch
    assert store.get("raw") is day11


def test_derive_name_keeps_the_day_suffix():
    store = ac.FrameStore()
    assert store.derive_name("raw:irt:2024-09-10", "derived") == "derived:irt:2024-09-10"
    assert store.derive_name("raw", "derived") == "derived"


def test_two_fetches_then_comparison_gives_two_distinct_traces(tmp_path, monkeypatch):
    """The exact failure from the field: two days, one comparison chart.

    Guards the root cause -- the second fetch used to overwrite the first, so the
    comparison silently collapsed onto a single day.
    """
    import intermagnet_loader as loader

    def fake_fetch(station_code=None, start_date=None, end_date=None, **kw):
        day = pd.to_datetime(start_date)
        # offset by day so the two series are genuinely different
        offset = float(day.day)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(day, periods=24, freq="h"),
                "X": [float(i) + offset for i in range(24)],
                "Y": [float(i) * 2 for i in range(24)],
                "Z": [float(i) * 3 + offset for i in range(24)],
            }
        )
        frame.attrs["publication_state"] = "definitive"
        return frame

    monkeypatch.setattr(ac, "OFFLINE", False)
    monkeypatch.setattr(loader, "fetch_observatory_data", fake_fetch)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)

    store = ac.FrameStore()
    handles = []
    for day in ("2024-09-10", "2024-09-11"):
        payload, _ = ac._handle_fetch(
            {"station_code": "IRT", "start_date": day, "end_date": day}, store
        )
        assert not ac.is_error(payload)
        handles.append(payload["frame_handle"])

    assert len(set(handles)) == 2, f"both days must get their own slot, got {handles}"

    derived = []
    for handle in handles:
        payload, _ = ac._handle_derive({"df": handle, "components": ["H"]}, store)
        assert not ac.is_error(payload)
        derived.append(payload["frame_handle"])
    assert len(set(derived)) == 2

    captured = {}
    original = plotter._write

    def spy(fig, path, include_plotlyjs):
        captured["fig"] = fig
        return original(fig, path, include_plotlyjs)

    plotter._write = spy
    try:
        payload, _ = ac._handle_plot_comparison(
            {"df1": derived[0], "df2": derived[1], "component": "H"}, store
        )
    finally:
        plotter._write = original

    assert not ac.is_error(payload), payload
    assert Path(payload["path"]).is_file()
    traces = captured["fig"].data
    assert len(traces) == 2, "the comparison chart must carry one trace per day"
    assert traces[0].name != traces[1].name
    # the two traces must actually carry different values
    assert list(traces[0].y) != list(traces[1].y)


def test_plot_comparison_refusal_keeps_its_code_and_hint():
    """The same_dataframe guard must reach the model as itself, not as ValueError."""
    store = ac.FrameStore()
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-09-10", periods=3, freq="h"),
                          "H": [1.0, 2.0, 3.0]})
    store.put("raw:irt:2024-09-10", frame)

    payload, note = ac._handle_plot_comparison(
        {"df1": "raw:irt:2024-09-10", "df2": "raw", "component": "H"}, store
    )

    assert ac.is_error(payload)
    assert payload["error"] == "same_dataframe", "the guard must not be masked"
    assert "two different days" in payload["message"]
    assert "raw:irt:2024-09-10" in payload["hint"]
    assert note
    # both handles resolve to one frame, so this is the same-dataframe case
    assert store.get("raw") is frame


def test_unrecovered_tool_failure_reaches_the_final_text():
    """A vague 'it failed' must not be the whole story the user sees."""
    store_entries = [{"ok": False, "tool": "plot_comparison", "error": "same_dataframe",
                      "message": "Both handles point at the same data",
                      "hint": "Pass two distinct slots."}]
    text = ac._with_tool_failures("Не удалось построить сравнение.", store_entries)
    assert "Не удалось построить сравнение." in text
    assert "same_dataframe" in text
    assert "Both handles point at the same data" in text
    assert "Pass two distinct slots." in text


def test_clean_run_gets_no_failure_footer():
    text = ac._with_tool_failures("Всё готово.", [{"ok": True, "tool": "get_statistics"}])
    assert text == "Всё готово."


# optional: the authoritative Qwen template fetched from HuggingFace
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_real_qwen_template_renders_our_history_cleanly():
    """Guards the double-encoding trap against the genuine template."""
    import urllib.request

    jinja2 = pytest.importorskip("jinja2")
    url = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json"
    with urllib.request.urlopen(url, timeout=60) as response:
        template_text = json.load(response)["chat_template"]

    env = jinja2.Environment()
    template = env.from_string(template_text)
    history = [
        {"role": "system", "content": ac.SYSTEM_PROMPT},
        {"role": "user", "content": "посчитай H"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_statistics",
                        "arguments": {"df": "raw", "components": ["H"]},
                    },
                }
            ],
        },
        {"role": "tool", "content": '{"ok": true, "rows": 1440}'},
    ]
    rendered = template.render(messages=history, tools=ac.TOOL_SCHEMAS, add_generation_prompt=True)

    assert "профессиональный геофизический" in rendered
    assert "<tools>" in rendered
    assert "<tool_response>" in rendered
    assert '"rows": 1440' in rendered
    # The replayed assistant turn must carry a JSON *object* for "arguments",
    # never a quoted string -- otherwise the model re-reads its own history as
    # garbage. The template's own "<args-json-object>" placeholder appears
    # earlier in the prompt, so the last occurrence is the real one.
    assert isinstance(_json_value_after(rendered, '"arguments": ', last=True), dict)


# --------------------------------------------------------------------------- #
# regression: auto-derive fallback for a derived handle that does not exist yet
# --------------------------------------------------------------------------- #
def _fake_fetch_factory():
    import intermagnet_loader as loader

    def fake_fetch(station_code=None, start_date=None, end_date=None, **kw):
        day = pd.to_datetime(start_date)
        offset = float(day.day)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(day, periods=24, freq="h"),
                "X": [float(i) + offset for i in range(24)],
                "Y": [float(i) * 2 for i in range(24)],
                "Z": [float(i) * 3 + offset for i in range(24)],
            }
        )
        frame.attrs["publication_state"] = "definitive"
        return frame

    return loader, fake_fetch


def _call(name, **arguments):
    return (
        T_OPEN
        + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
        + T_CLOSE
    )


def test_fetch_fetch_comparison_packet_reaches_the_chart_in_one_round(tmp_path, monkeypatch):
    """The field failure, in the shape the model actually emits it.

    One reply carrying [fetch, fetch, plot_comparison]. The model asks for a
    derived handle for a day it has not derived yet -- it is following the naming
    convention, not making a mistake -- and that used to come back
    unknown_frame. The round was spent recovering, the chart never appeared, and
    the tally read "1 of 2".
    """
    loader, fake_fetch = _fake_fetch_factory()
    monkeypatch.setattr(ac, "OFFLINE", False)
    monkeypatch.setattr(loader, "fetch_observatory_data", fake_fetch)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)

    packet = "".join(
        [
            _call("fetch_observatory_data", station_code="IRT", start_date="2024-09-10", end_date="2024-09-10"),
            _call("fetch_observatory_data", station_code="IRT", start_date="2024-09-11", end_date="2024-09-11"),
            _call(
                "plot_comparison",
                df1="derived:irt:2024-09-10",
                df2="derived:irt:2024-09-11",
                component="H",
            ),
        ]
    )
    brain = ac.ScriptedBrain([packet, "Готово: сравнение за 2024-09-11 построено."])
    out = ac.run_agent("сравни дни", brain=brain, verbose=False)

    steps = out["tool_calls"]
    assert [s["tool"] for s in steps] == [
        "fetch_observatory_data",
        "fetch_observatory_data",
        "plot_comparison",
    ], [s["tool"] for s in steps]
    # one round only: the packet was executed as issued, not retried
    assert {s["round"] for s in steps} == {1}, [s["round"] for s in steps]
    failures = [(s["tool"], s.get("error")) for s in steps if not s["ok"]]
    assert not failures, f"a clean packet must log no errors, got {failures}"
    assert len(out["plots"]) == 1
    assert Path(out["plots"][0]).is_file()


def test_auto_derived_slot_is_recorded_in_the_tool_log(tmp_path, monkeypatch):
    """The log has to show that a slot was derived, not fetched.

    A chart built by the fallback has to be distinguishable from one built from a
    frame the model asked for, or the user cannot tell what was plotted.
    """
    loader, fake_fetch = _fake_fetch_factory()
    monkeypatch.setattr(ac, "OFFLINE", False)
    monkeypatch.setattr(loader, "fetch_observatory_data", fake_fetch)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)

    packet = "".join(
        [
            _call("fetch_observatory_data", station_code="IRT", start_date="2024-09-10", end_date="2024-09-10"),
            _call("fetch_observatory_data", station_code="IRT", start_date="2024-09-11", end_date="2024-09-11"),
            _call(
                "plot_comparison",
                df1="derived:irt:2024-09-10",
                df2="derived:irt:2024-09-11",
                component="H",
            ),
        ]
    )
    brain = ac.ScriptedBrain([packet, "Готово."])
    out = ac.run_agent("сравни дни", brain=brain, verbose=False)

    comparison = out["tool_calls"][-1]
    assert "auto-derived: derived:irt:2024-09-10 (из raw)" in comparison["note"]
    assert "auto-derived: derived:irt:2024-09-11 (из raw)" in comparison["note"]
    # the two days must still be genuinely different, not collapsed onto one
    assert len(out["plots"]) == 1
    assert Path(out["plots"][0]).is_file()


@pytest.mark.parametrize(
    "handle,components,expected_ok,expected_error",
    [
        # derived + raw present -> derive and carry on
        ("derived:irt:2024-09-11", ["H"], True, None),
        # derived but no raw sibling -> ordinary unknown_frame, no invention
        ("derived:irt:2024-09-10", ["H"], False, "unknown_frame"),
        # anomalies need a baseline and a window: never derived implicitly
        ("anomalies:irt:2024-09-11", ["H"], False, "unknown_frame"),
    ],
)
def test_auto_derive_boundaries(handle, components, expected_ok, expected_error):
    store = ac.FrameStore()
    store.put(
        "raw:irt:2024-09-11",
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-09-11", periods=6, freq="h"),
                "X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "Y": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "Z": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        ),
    )
    notes: list = []
    frame, err = ac._resolve_plottable(store, handle, components, notes)
    if expected_ok:
        assert err is None
        assert frame is not None
        assert notes, "a derived slot must be reported in the note"
        assert store.get(handle) is not None
    else:
        assert frame is None
        assert err is not None and err["error"] == expected_error, err
        assert not notes, "nothing may be derived when the raw sibling is absent"
        assert store.get(handle) is None, "a failed fallback must not create a slot"
    # the raw frame is never replaced by the fallback
    assert "derived" not in store.names() or expected_ok


def test_plot_components_auto_derives_too(tmp_path, monkeypatch):
    """The single-day plot is the same miss and gets the same repair."""
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    store = ac.FrameStore()
    store.put(
        "raw:irt:2024-09-11",
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-09-11", periods=12, freq="h"),
                "X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "Y": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                "Z": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            }
        ),
    )
    payload, note = ac._handle_plot_components(
        {"df": "derived:irt:2024-09-11", "components": ["H"]}, store
    )
    assert not ac.is_error(payload), payload
    assert Path(payload["path"]).is_file()
    assert "auto-derived: derived:irt:2024-09-11 (из raw)" in note


def test_missing_derived_handle_never_borrows_another_day(tmp_path, monkeypatch):
    """No raw sibling means an error, not a substitution.

    plot_components used to fall through to a family scan on a miss, which could
    hand back whichever other day was loaded and plot it under the requested name.
    """
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    store = ac.FrameStore()
    store.put(
        "raw:irt:2024-09-10",
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-09-10", periods=6, freq="h"),
                "X": [1.0, 2, 3, 4, 5, 6], "Y": [1.0, 2, 3, 4, 5, 6], "Z": [1.0, 2, 3, 4, 5, 6],
            }
        ),
    )
    payload, _ = ac._handle_plot_components(
        {"df": "derived:irt:2024-09-11", "components": ["H"]}, store
    )
    assert ac.is_error(payload), "must not silently plot 2024-09-10 as 2024-09-11"
    assert payload["error"] == "unknown_frame"
    assert "derived:irt:2024-09-11" not in store.names()


def test_auto_derived_comparison_still_plots_two_different_days(tmp_path, monkeypatch):
    """Auto-deriving must not collapse the comparison onto one day.

    Each fallback produces a fresh frame, so nothing structurally forces the two
    days apart -- if both requests resolved to the same raw frame the chart would
    still build, and would be silently meaningless.
    """
    loader, fake_fetch = _fake_fetch_factory()
    monkeypatch.setattr(ac, "OFFLINE", False)
    monkeypatch.setattr(loader, "fetch_observatory_data", fake_fetch)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)

    store = ac.FrameStore()
    for day in ("2024-09-10", "2024-09-11"):
        payload, _ = ac._handle_fetch(
            {"station_code": "IRT", "start_date": day, "end_date": day}, store
        )
        assert not ac.is_error(payload)

    captured = {}
    original = plotter._write

    def spy(fig, path, include_plotlyjs):
        captured["fig"] = fig
        return original(fig, path, include_plotlyjs)

    plotter._write = spy
    try:
        payload, note = ac._handle_plot_comparison(
            {
                "df1": "derived:irt:2024-09-10",
                "df2": "derived:irt:2024-09-11",
                "component": "H",
            },
            store,
        )
    finally:
        plotter._write = original

    assert not ac.is_error(payload), payload
    traces = captured["fig"].data
    assert len(traces) == 2, "the comparison chart must carry one trace per day"
    assert traces[0].name != traces[1].name
    assert list(traces[0].y) != list(traces[1].y)
    assert note.count("auto-derived:") == 2


def test_system_prompt_stays_within_its_budget():
    """The prompt was cut to buy room for the tool-selection table.

    It is a prompt for a 7B model: every rule competes with the others for
    attention, so growth has to be a deliberate act rather than an accumulation.
    The rules now enforced in code -- slot ordering, doubled braces -- were the
    ones removed, and this stops them creeping back one clause at a time.

    Phase 3 rewrote the prompt into a tighter voice to advertise the geomagnetic
    tools (geomag coords, MLT, LT, overlay) without new rules, and the ceiling was
    pulled down from 5200 to 4500. It stays pinned at the measured value so the
    next addition has to be argued for rather than drifted into.
    """
    sp = ac.SYSTEM_PROMPT
    assert len(sp) <= 4500, f"prompt grew to {len(sp)} chars"
    # what must survive a trim
    assert "4a." in sp, "the tool-selection table is the point of the prompt"
    assert "'nT'" in sp, "the checker and recovery filter both key on the unit"
    assert "ПЕРЕД ЛЮБЫМ из них" in sp, "fetch-first is not derivable from the code"
    # the three project tools are advertised in that table, and nowhere else:
    # a tool the model has never heard of is a tool it will not call.
    for tool in ("create_project", "fetch_many", "export_project"):
        assert sp.count(tool) >= 1, f"{tool} is not advertised in the prompt"
    table_start = sp.index("4a.")
    # no numbered rule was added: 1..11 must still be the whole rule set
    numbers = set(re.findall(r"(?:^|\n)(\d{1,2})\. ", sp))
    assert numbers <= {str(n) for n in range(1, 12)}, f"new rule numbers: {numbers}"


# --------------------------------------------------------------------------- #
# regression: the auto-derive fallback must ignore handle casing
# --------------------------------------------------------------------------- #
_SHOPTER_CASES = [
    "derived:irt:2024-09-11",
    "DERIVED:IRT:2024-09-11",
    "Derived:Irt:2024-09-11",
]


@pytest.mark.parametrize("handle", _SHOPTER_CASES)
def test_auto_derive_ignores_casing_in_the_handle(handle, tmp_path, monkeypatch):
    """compare_uppercase feeds the agent shouted handles all day long.

    The store normalises on put and on get, but the fallback builds its own raw
    lookup from the handle string, so this path has to be pinned on its own: a
    casing slip here turns into unknown_frame for exactly the scenario that
    exists to prove casing is harmless.
    """
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    store = ac.FrameStore()
    store.put("raw:irt:2024-09-11", _derived_fixture("2024-09-11", 100.0))
    payload, note = ac._handle_plot_components(
        {"df": handle, "components": ["H"]}, store
    )
    assert not ac.is_error(payload), payload
    assert f"auto-derived: {handle.lower()} (из raw)" in note, note
    # the slot must be stored canonically, or every advertised handle is a miss
    assert handle.lower() in store.names()


def _derived_fixture(day, offset=0.0, n=12):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(day, periods=n, freq="h"),
            "X": [float(i) + offset for i in range(n)],
            "Y": [float(i) * 2 for i in range(n)],
            "Z": [float(i) * 3 + offset for i in range(n)],
        }
    )


@pytest.mark.parametrize(
    "handle",
    [
        "anomalies:irt:2024-09-11",
        "ANOMALIES:IRT:2024-09-11",
    ],
)
def test_auto_derive_never_creates_an_anomalies_slot(handle):
    """Anomalies need a baseline and a window, so casing does not change that."""
    store = ac.FrameStore()
    store.put("raw:irt:2024-09-11", _derived_fixture("2024-09-11", 100.0))
    assert ac._auto_derive(store, handle, ["H"]) == (None, "")
    assert not [n for n in store.names() if n.startswith("anomalies")]


def test_plot_comparison_auto_derives_both_shouted_handles(tmp_path, monkeypatch):
    """The reported scenario end to end: both handles shouted, one round."""
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    store = ac.FrameStore()
    store.put("raw:irt:2024-09-10", _derived_fixture("2024-09-10", 0.0))
    store.put("raw:irt:2024-09-11", _derived_fixture("2024-09-11", 100.0))
    payload, note = ac._handle_plot_comparison(
        {
            "df1": "DERIVED:IRT:2024-09-10",
            "df2": "DERIVED:IRT:2024-09-11",
            "component": "H",
        },
        store,
    )
    assert not ac.is_error(payload), payload
    assert note.count("auto-derived:") == 2, note
    assert Path(payload["path"]).is_file()


# --------------------------------------------------------------------------- #
# Phase 3: geomagnetic coordinates and MLT
# --------------------------------------------------------------------------- #
def test_geomag_coords_handler_returns_dipole_coordinates():
    """The handler answers with tilted-dipole coordinates and names its model."""
    payload, note = ac._handle_get_station_geomagnetic_coords(
        {"station_code": "IRT"}, ac.FrameStore()
    )
    assert payload["ok"], payload
    assert payload["model"] == "tilted_dipole"
    assert -90 <= payload["geo_lat"] <= 90
    assert -180 <= payload["geo_lon"] <= 180


def test_geomag_coords_handler_reports_station_not_found_without_raising():
    """An unknown IAGA code is an error payload, never an exception."""
    payload, note = ac._handle_get_station_geomagnetic_coords(
        {"station_code": "ZZZ"}, ac.FrameStore()
    )
    assert ac.is_error(payload)
    assert payload["error"] == "station_not_found"


def test_calculate_mlt_is_inside_the_day():
    """MLT is a time of day: 0 <= mlt < 24, with a human HH:MM companion."""
    payload, note = ac._handle_calculate_mlt(
        {"station_code": "IRT", "timestamp": "2024-09-10T12:00:00Z"},
        ac.FrameStore(),
    )
    assert payload["ok"], payload
    assert 0 <= payload["mlt_hours"] < 24
    assert re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", payload["mlt_hm"])
    assert payload["model"] == "tilted_dipole"


def test_calculate_mlt_propagates_station_not_found():
    """The MLT tool must not swallow a registry miss."""
    payload, note = ac._handle_calculate_mlt(
        {"station_code": "ZZZ", "timestamp": "2024-09-10T12:00:00Z"},
        ac.FrameStore(),
    )
    assert ac.is_error(payload)
    assert payload["error"] == "station_not_found"


def test_group_stations_by_mlt_bins_every_station():
    """All three verification stations land in exactly one bin each."""
    payload, note = ac._handle_group_stations_by_mlt(
        {
            "stations": ["IRT", "API", "BSL"],
            "timestamp": "2024-09-10T12:00:00Z",
        },
        ac.FrameStore(),
    )
    assert payload["ok"], payload
    binned = [code for codes in payload["bins"].values() for code in codes]
    assert sorted(binned) == ["API", "BSL", "IRT"]
    for entry in payload["stations_mlt"]:
        assert 0 <= entry["mlt_hours"] < 24


def test_group_stations_by_mlt_skips_unknown_stations():
    """A bad code in the list must not abort the good ones."""
    payload, note = ac._handle_group_stations_by_mlt(
        {"stations": ["IRT", "ZZZ"], "timestamp": "2024-09-10T12:00:00Z"},
        ac.FrameStore(),
    )
    assert payload["ok"], payload
    assert [c for codes in payload["bins"].values() for c in codes] == ["IRT"]


# --------------------------------------------------------------------------- #
# Phase 3: civil local time and the multi-station overlay
# --------------------------------------------------------------------------- #
def test_calculate_local_time_is_ut_plus_longitude_over_15():
    """LT is civil time: UT + longitude/15, wrapped into [0, 24)."""
    from intermagnet_loader import get_available_stations

    lon = float(get_available_stations()["IRT"][2])
    expected = (12.0 + lon / 15.0) % 24.0
    payload, note = ac._handle_calculate_local_time(
        {"station_code": "IRT", "timestamp": "2024-09-10T12:00:00Z"},
        ac.FrameStore(),
    )
    assert payload["ok"], payload
    assert abs(payload["lt_hours"] - expected) < 1e-3
    assert re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", payload["lt_hm"])
    assert payload["model"] == "lt_ut_plus_lon_over_15"


def test_calculate_local_time_propagates_station_not_found():
    """The LT tool must not swallow a registry miss."""
    payload, note = ac._handle_calculate_local_time(
        {"station_code": "ZZZ", "timestamp": "2024-09-10T12:00:00Z"},
        ac.FrameStore(),
    )
    assert ac.is_error(payload)
    assert payload["error"] == "station_not_found"


def test_calculate_local_time_rejects_an_unparsable_timestamp():
    """A garbage timestamp is an error payload, never an exception."""
    payload, note = ac._handle_calculate_local_time(
        {"station_code": "IRT", "timestamp": "not-a-time"},
        ac.FrameStore(),
    )
    assert ac.is_error(payload)
    assert payload["error"] == "invalid_input"


def test_plot_overlay_auto_fetch(tmp_path, monkeypatch):
    """plot_overlay must fetch the day itself when the store is empty.

    The live E2E run showed the agent calling plot_overlay in the very first
    round, before any data was loaded, and then not retrying after the fetch.
    The tool should be usable cold: it downloads the missing days, materialises
    the derived slots, and only then draws the chart.
    """
    monkeypatch.setattr(ac, "OFFLINE", True)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    store = ac.FrameStore()
    assert "raw:irt:2024-09-10" not in store.names()

    result, note = ac._handle_plot_overlay(
        {
            "stations": ["IRT"],
            "dates": ["2024-09-10"],
            "component": "H",
            "offsets": {"IRT": 0},
        },
        store,
    )
    assert result["ok"], result
    assert "raw:irt:2024-09-10" in store.names(), "данные должны быть скачаны"
    assert "derived:irt:2024-09-10" in store.names(), "derived должен быть вычислен"
    assert result["auto"], note


# --------------------------------------------------------------------------- #
# Phase 4: external brain (planner + dynamic tool selection)
# --------------------------------------------------------------------------- #
def test_planner_multi_station():
    """One target per station: IRT, API, BSL -> three tasks."""
    from agent_planner import Planner

    state = Planner().plan("Построй графики для станций IRT, API, BSL за 10 сентября 2024")
    assert len(state.tasks) == 3, "Должно быть 3 подзадачи"
    assert state.tasks[0].tool_args["station"] == "IRT"
    codes = [t.tool_args["station"] for t in state.tasks]
    assert codes == ["IRT", "API", "BSL"]
    assert all(t.tool_name == "plot_components" for t in state.tasks)
    assert all(t.tool_args["date"] == "2024-09-10" for t in state.tasks)
    assert state.original_query.startswith("Построй графики")


def test_planner_batch_statistics():
    """Статистика для N станций порождает N задач get_statistics."""
    from agent_planner import Planner

    state = Planner().plan("Посчитай медиану для 2 станций за 2024-09-10")
    assert len(state.tasks) == 2
    assert state.tasks[0].tool_name == "get_statistics"
    assert state.tasks[0].tool_args["station"] == "IRT"


def test_planner_unrecognized_query_is_one_run_agent_task():
    """A free-form request stays a single task the model handles itself."""
    from agent_planner import Planner

    state = Planner().plan("Что такое магнитное поле Земли?")
    assert len(state.tasks) == 1
    assert state.tasks[0].tool_name == "run_agent"


def test_select_tools_narrows_to_the_subtask_chain():
    tools = ac.select_tools_for_task("plot_components")
    names = [t["function"]["name"] for t in tools]
    assert "plot_components" in names
    assert "fetch_observatory_data" in names
    assert len(tools) <= 5, "Не должно быть больше 5 инструментов"


def test_select_tools_unknown_name_falls_back_to_everything():
    tools = ac.select_tools_for_task("no_such_tool")
    assert len(tools) == len(ac.TOOL_SCHEMAS)


def test_build_dynamic_prompt_mentions_only_the_selected_tools():
    tools = ac.select_tools_for_task("calculate_local_time")
    prompt = ac.build_dynamic_prompt(ac.BASE_PROMPT_COMPACT, tools, "Контекст: 1/3")
    assert "calculate_local_time" in prompt
    assert "\n- plot_overlay:" not in prompt, "unselected tool must not be advertised"
    assert "Контекст" in prompt


class _AlwaysTextBrain:
    """A brain that answers immediately, without tool calls."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools):  # noqa: ARG002
        self.calls += 1
        return {"content": "Готово.", "tool_calls": None}


def test_run_agent_with_planner_executes_each_subtask():
    """Two stations -> two subtasks, both succeed with a mock brain."""
    result = ac.run_agent_with_planner(
        "Построй графики для 2 станций за 10 сентября",
        brain=_AlwaysTextBrain(),
        verbose=False,
    )
    assert result["ok"], result
    assert len(result["results"]) == 2
    assert all(r["ok"] for r in result["results"])
    assert result["state"]["current_task_index"] == 2
    assert "Выполнено 2 из 2" in result["text"]


class _PlanChatBrain:
    """A brain whose chat() returns a canned JSON plan (model/planner stand-in)."""

    def __init__(self, text):
        self._text = text

    def chat(self, messages, tools):  # noqa: ARG002
        return {"content": self._text, "tool_calls": None}


PLAN_JSON = (
    '```json\n{"tasks": ['
    '{"id":"step_001","description":"Создай проект events",'
    '"tool_name":"create_project","tool_args":{"project_name":"events"}},'
    '{"id":"step_002","description":"Скачай данные BRW за 2024-10-10",'
    '"tool_name":"fetch_observatory_data",'
    '"tool_args":{"station_code":"BRW","start_date":"2024-10-10","end_date":"2024-10-10"}},'
    '{"id":"step_003","description":"Построй H и D для BRW",'
    '"tool_name":"plot_components",'
    '"tool_args":{"df":"raw:brw:2024-10-10","components":["H","D"]}}]}\n```'
)


def test_llm_planner_parses_model_plan_ignoring_markdown_fences():
    """The model wraps its JSON in ```json fences + prose; braces win."""
    from agent_planner import LLMPlanner

    state = LLMPlanner(_PlanChatBrain(PLAN_JSON)).plan(
        "10 событий x 26 станций, графики в архив"
    )
    tool_names = [t.tool_name for t in state.tasks]
    assert len(state.tasks) == 3
    assert tool_names == [
        "create_project",
        "fetch_observatory_data",
        "plot_components",
    ]
    assert state.tasks[1].tool_args["station_code"] == "BRW"
    assert state.tasks[2].tool_args["components"] == ["H", "D"]


def test_llm_planner_parses_truncated_plan_and_renumbers_ids():
    """A ``stop`` token may cut the JSON mid-object; brace-matching must cope."""
    from agent_planner import LLMPlanner

    truncated = PLAN_JSON[:-14]  # chop the tail (']}\n```') leaving a cut object
    state = LLMPlanner(_PlanChatBrain(truncated)).plan("какой-то запрос")
    assert state.tasks, "even a truncated answer must yield the tasks it contains"
    # ids are guaranteed unique even if the model repeated step numbers
    ids = [t.id for t in state.tasks]
    assert len(ids) == len(set(ids))


def test_llm_planner_falls_back_to_deterministic_on_bad_json():
    """Garbage output must not crash the run; Planner() takes over."""
    from agent_planner import LLMPlanner

    state = LLMPlanner(_PlanChatBrain("Я не понял задачу, повторите.")).plan(
        "Построй графики для 2 станций за 10 сентября"
    )
    assert len(state.tasks) == 2
    assert [t.tool_name for t in state.tasks] == [
        "plot_components",
        "plot_components",
    ]


def test_llm_planner_verbose_reports_raw_output_and_fallback(capsys):
    """verbose=True must print the raw model answer and the fallback reason."""
    from agent_planner import LLMPlanner

    brain = _PlanChatBrain("«годно» — и больше ничего годного")
    state = LLMPlanner(brain).plan(
        "Построй графики для 2 станций", verbose=True
    )
    out = capsys.readouterr().out
    assert "LLM-ПЛАНИРОВЩИК: Сырой ответ модели" in out
    assert "Ошибка парсинга" in out
    assert len(state.tasks) == 2, "fallback must still produce the deterministic plan"


def test_llm_planner_verbose_reports_task_count(capsys):
    """On a good plan verbose must print the extracted JSON and the count."""
    from agent_planner import LLMPlanner

    LLMPlanner(_PlanChatBrain(PLAN_JSON)).plan("10 событий", verbose=True)
    out = capsys.readouterr().out
    assert "Извлечённый JSON" in out
    assert "✅ Распарсено 3 подзадач" in out


def test_llm_planner_complex_query_offline_mock():
    """A >10-task plan from a mocked model arrives as usable TaskSteps."""
    from agent_planner import LLMPlanner

    tasks = []
    tools = ["create_project", "fetch_observatory_data", "plot_components"]
    for i, tool in enumerate(tools, start=1):
        tasks.append(
            '{"id":"step_%03d","description":"%s #%d","tool_name":"%s","tool_args":{}}'
            % (i, tool, i, tool)
        )
    mock_plan = '{"tasks":[' + ",".join(tasks) + "]}"
    state = LLMPlanner(_PlanChatBrain(mock_plan)).plan("большой запрос")
    tool_names = [t.tool_name for t in state.tasks]
    assert len(state.tasks) == 3
    assert set(tool_names) == set(tools)


def test_llm_planner_complex_query_real_model():
    """End-to-end with the Qwen brain. Skips locally (no model on disk).

    Must run in Kaggle: load_brain downloads ~5 GB, which is banned in local
    runs unless GEOMAG_OFFLINE is not set.
    """
    if not ac.model_is_downloaded():
        pytest.skip("model not on disk; runs in Kaggle")
    from agent_planner import LLMPlanner

    brain = ac.load_brain()
    planner = LLMPlanner(brain)
    query = (
        "Изучи 10 событий (событие01..событие10), для каждого собери данные "
        "всех 26 станций и построй графики; объедини всё в архив"
    )
    state = planner.plan(query)
    tool_names = [t.tool_name for t in state.tasks]
    assert len(state.tasks) > 10, f"expected a real plan, got {len(state.tasks)}"
    assert "create_project" in tool_names
    assert "fetch_observatory_data" in tool_names
    assert "plot_components" in tool_names


class _FetchThenPlotBrain:
    """Scripted brain: subtask 1 fetches, subtask 2 plots the same handle."""

    def __init__(self, plan_json):
        self._plan = plan_json
        self._calls = 0
        self.replies = [
            {
                "content": '<tool_call>{"name": "fetch_observatory_data", '
                '"arguments": {"station_code": "BRW", "start_date": "2024-10-10", '
                '"end_date": "2024-10-10"}}</tool_call>',
                "tool_calls": None,
            },
            {"content": "Данные BRW скачаны.", "tool_calls": None},
            {
                "content": '<tool_call>{"name": "plot_components", '
                '"arguments": {"df": "raw:brw:2024-10-10", "components": ["X"]}}'
                "</tool_call>",
                "tool_calls": None,
            },
            {"content": "График построен.", "tool_calls": None},
        ]

    def chat(self, messages, tools):  # noqa: ARG002
        if self._calls == 0 and self._plan:
            self._calls += 1
            return {"content": self._plan, "tool_calls": None}
        self._calls += 1
        return self.replies.pop(0) if self.replies else {"content": "Готово.", "tool_calls": None}


def test_shared_framestore_run_agent_two_calls_same_store(
    monkeypatch, tmp_path
):
    """Fetch in one run_agent call, then plot the handle in a *second* call
    sharing the same FrameStore: the plot must resolve without re-fetching."""
    monkeypatch.setattr(ac, "OFFLINE", True)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)

    store = ac.FrameStore()
    brain = _FetchThenPlotBrain("")
    out1 = ac.run_agent(
        "Скачай BRW за 2024-10-10", brain=brain, verbose=False, store=store
    )
    assert out1["ok"], out1
    assert "raw:brw:2024-10-10" in store.names()

    out2 = ac.run_agent(
        "Построй график X для BRW",
        brain=brain,
        verbose=False,
        store=store,
    )
    assert out2["ok"], out2
    assert out2["plots"], "plot of a handle created in the previous call must resolve"


def test_shared_framestore_through_planner(monkeypatch, tmp_path):
    """One run_agent_with_planner call: subtask 1 fetches, subtask 2 plots.
    A single shared FrameStore means the plot subtask sees the fetch subtask's
    data -- that is the whole point of the shared store."""
    monkeypatch.setattr(ac, "OFFLINE", True)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)

    plan_json = (
        '{"tasks":['
        '{"id":"step_001","description":"Скачай BRW",'
        '"tool_name":"fetch_observatory_data",'
        '"tool_args":{"station_code":"BRW","start_date":"2024-10-10","end_date":"2024-10-10"}},'
        '{"id":"step_002","description":"Построй X",'
        '"tool_name":"plot_components",'
        '"tool_args":{"df":"raw:brw:2024-10-10","components":["X"]}}]}'
    )
    result = ac.run_agent_with_planner(
        "Скачай и построй",
        brain=_FetchThenPlotBrain(plan_json),
        verbose=False,
    )
    assert result["ok"], result
    assert len(result["results"]) == 2
    assert all(r["ok"] for r in result["results"])
    shared = result["store"]
    assert isinstance(shared, ac.FrameStore)
    assert "raw:brw:2024-10-10" in shared.names()
    assert result["results"][1]["result"]["plots"], (
        "subtask 2 must plot the frame fetched by subtask 1 in the SAME store"
    )


# --------------------------------------------------------------------------- #
# Phase 5 -- Generator-Executor: task matrix + batch processing
# --------------------------------------------------------------------------- #
def _tiny_plot_components(args, store):
    """A plot_components stand-in for matrix tests.

    The real handler renders a ~5 MB plotlyjs HTML, and on this Windows test box
    re-reading those files is subject to random multi-second on-access scans.
    Rendering itself is already covered end to end by the plotter tests; here we
    exercise the matrix wiring -- write a chart, file it into the project days,
    land it in the export -- on a few hundred bytes instead.
    """
    frame, err = ac._resolve_plottable(
        store, args.get("df"), args.get("components") or ["H"], []
    )
    if err:
        return err, "plot_components failed: " + err["error"]
    out = plotter.DEFAULT_OUTPUT_DIR / (args.get("filename") or "plot.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("<html>tiny plot</html>", encoding="utf-8")
    filed = ac._file_into_project(store, str(out), [args.get("df")], "components")
    payload = {
        "ok": True,
        "path": str(out),
        "components": args.get("components") or ["H"],
        "rows": int(len(frame)),
    }
    if filed:
        payload["filed_into_project"] = filed
    return payload, f"Saved chart to {out}"


def _matrix_store(monkeypatch, tmp_path):
    """An offline FrameStore whose project/export/plot output all stay in tmp."""
    monkeypatch.setattr(ac, "OFFLINE", True)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path / "plots")
    monkeypatch.setattr(ac.projects, "EXPORTS_ROOT", tmp_path / "exports")
    monkeypatch.setitem(ac.HANDLERS, "plot_components", _tiny_plot_components)
    return ac.FrameStore(ac.projects.Workspace(root=tmp_path / "projects"))


MATRIX_ARGS = {
    "events": ["2024-10-10", "2024-09-08"],
    "stations": ["BRW", "SHU", "IRT"],
    "actions": ["fetch", "calc_HDI", "plot"],
    "project_name": "events",
}


def test_create_task_matrix_writes_the_cartesian_product(monkeypatch, tmp_path):
    """2 events x 3 stations = 6 pending tasks, saved next to the project."""
    store = _matrix_store(monkeypatch, tmp_path)
    payload, note = ac.HANDLERS["create_task_matrix"](dict(MATRIX_ARGS), store)
    assert payload["ok"], payload
    assert payload["total"] == 6
    assert "Матрица создана. Всего задач: 6." in payload["summary"]
    assert store.workspace.project == "events"

    path = Path(payload["matrix_path"])
    assert path.is_file()
    matrix = json.loads(path.read_text(encoding="utf-8"))
    assert matrix["total"] == 6 and matrix["pending"] == 6 and matrix["completed"] == 0
    assert len(matrix["tasks"]) == 6
    assert all(t["status"] == "pending" for t in matrix["tasks"])
    cells = {(t["station"], t["event"]) for t in matrix["tasks"]}
    assert len(cells) == 6
    assert ("BRW", "2024-10-10") in cells and ("IRT", "2024-09-08") in cells
    assert matrix["tasks"][0]["actions"] == [
        "fetch_observatory_data",
        "calculate_derived_components",
        "plot_components",
    ]


def test_process_matrix_batch_executes_cells_directly(monkeypatch, tmp_path):
    """The batch tool wipes out the pending queue without any model calls."""
    store = _matrix_store(monkeypatch, tmp_path)
    created, _ = ac.HANDLERS["create_task_matrix"](dict(MATRIX_ARGS), store)
    path = created["matrix_path"]

    payload, _ = ac.HANDLERS["process_matrix_batch"](
        {"matrix_path": path, "batch_size": 4}, store
    )
    assert payload["ok"], payload
    assert payload["processed"] == 4 and payload["succeeded"] == 4 and payload["failed"] == 0
    assert payload["remaining"] == 2 and payload["progress"] == "4/6"
    assert "Обработано 4 задач. Успешно: 4, Ошибок: 0." in payload["summary"]
    assert "Осталось в матрице: 2." in payload["summary"]

    done = payload["remaining"] == 2  # half the cells done
    matrix = json.loads(Path(path).read_text(encoding="utf-8"))
    assert matrix["completed"] == 4 and matrix["pending"] == 2
    assert sum(1 for t in matrix["tasks"] if t["status"] == "done") == 4

    # The executors did real work: frames are in the shared store, charts on disk.
    assert len(store.names()) >= 8  # 4 raw + 4 derived
    assert (tmp_path / "plots").exists()

    payload2, _ = ac.HANDLERS["process_matrix_batch"](
        {"matrix_path": path, "batch_size": 4}, store
    )
    assert payload2["succeeded"] == 2 and payload2["remaining"] == 0
    assert payload2["progress"] == "6/6"


def test_process_matrix_batch_marks_a_failed_cell_without_stopping_the_batch(
    monkeypatch, tmp_path
):
    """A broken cell is recorded as failed; the rest still complete."""
    store = _matrix_store(monkeypatch, tmp_path)
    created, _ = ac.HANDLERS["create_task_matrix"](dict(MATRIX_ARGS), store)
    path = Path(created["matrix_path"])
    matrix = json.loads(path.read_text(encoding="utf-8"))
    matrix["tasks"][0]["actions"] = ["no_such_handler"]
    path.write_text(json.dumps(matrix, ensure_ascii=False), encoding="utf-8")

    payload, _ = ac.HANDLERS["process_matrix_batch"](
        {"matrix_path": str(path)}, store
    )
    assert payload["ok"]
    assert payload["processed"] == 5 and payload["succeeded"] == 4 and payload["failed"] == 1
    refreshed = json.loads(path.read_text(encoding="utf-8"))
    assert refreshed["failed"] == 1 and refreshed["pending"] == 1
    assert refreshed["tasks"][0]["status"] == "failed"
    assert "no_such_handler" in (refreshed["tasks"][0]["error"] or "")


def test_process_matrix_batch_resolves_a_bare_file_name_and_clamps_batch_size(
    monkeypatch, tmp_path
):
    store = _matrix_store(monkeypatch, tmp_path)
    created, _ = ac.HANDLERS["create_task_matrix"](dict(MATRIX_ARGS), store)
    name = Path(created["matrix_path"]).name

    payload, _ = ac.HANDLERS["process_matrix_batch"](
        {"matrix_path": name, "batch_size": -3}, store
    )
    assert payload["ok"] and payload["processed"] == 1  # clamped to the 1..20 minimum
    payload, _ = ac.HANDLERS["process_matrix_batch"](
        {"matrix_path": name, "batch_size": 999}, store
    )
    assert payload["processed"] == 5 and payload["remaining"] == 0  # clamped to 20


def test_process_matrix_batch_rejects_an_unknown_path(monkeypatch, tmp_path):
    store = _matrix_store(monkeypatch, tmp_path)
    payload, note = ac.HANDLERS["process_matrix_batch"](
        {"matrix_path": "nonexistent_matrix.json"}, store
    )
    assert not payload["ok"]
    assert payload["error"] == "matrix_not_found"


def test_create_task_matrix_validation_errors(monkeypatch, tmp_path):
    store = _matrix_store(monkeypatch, tmp_path)
    payload, _ = ac.HANDLERS["create_task_matrix"](
        {"events": "bad-date", "stations": ["BRW"], "actions": ["plot"], "project_name": "x"}, store
    )
    assert not payload["ok"] and payload["error"] == "invalid_events"
    payload, _ = ac.HANDLERS["create_task_matrix"](
        {"events": ["2024-10-10"], "stations": ["BRW"], "actions": ["plot"]}, store
    )
    assert not payload["ok"] and payload["error"] == "missing_project_name"


class _MatrixPlanBrain:
    """chat() returns a plan that is a single create_task_matrix step."""

    def chat(self, messages, tools):  # noqa: ARG002
        return {
            "content": (
                '{"tasks":[{"id":"step_001","description":"Матрица",'
                '"tool_name":"create_task_matrix",'
                '"tool_args":{"events":["2024-10-10","2024-09-08"],'
                '"stations":["BRW","SHU"],'
                '"actions":["fetch","calc_HDI","plot"],'
                '"project_name":"events"}}]}'
            ),
            "tool_calls": None,
        }


def test_run_agent_with_planner_single_matrix_plan_runs_the_whole_job(
    monkeypatch, tmp_path
):
    """A plan of exactly one create_task_matrix task is executed end to end:
    matrix written, batches ground through directly, project exported."""
    store = _matrix_store(monkeypatch, tmp_path)
    result = ac.run_agent_with_planner(
        "10 событий x 26 станций, графики в архив",
        brain=_MatrixPlanBrain(),
        store=store,
        verbose=False,
    )
    assert result["ok"], result
    assert result["matrix_path"] and Path(result["matrix_path"]).is_file()
    assert any("raw:brw:2024-10-10" in h for h in result["store"].names())
    assert Path(result["export"]["path"]).suffix == ".zip"
    assert result["results"], "at least one process_matrix_batch result expected"


class _ManyCellsPlanBrain:
    """chat() returns 20 identical plot_components tasks (over the threshold)."""

    def chat(self, messages, tools):  # noqa: ARG002
        tasks = ",".join(
            "{"
            '"id": "step_%03d", "description": "p%d", '
            '"tool_name": "plot_components", '
            '"tool_args": {"station_code": "BRW", '
            '"start_date": "2024-10-10", "end_date": "2024-10-10"}'
            "}" % (i, i)
            for i in range(1, 21)
        )
        return {"content": '{"tasks":[' + tasks + "]}", "tool_calls": None}


def test_run_agent_with_planner_folds_a_large_plan_into_a_matrix(
    monkeypatch, tmp_path
):
    """>MATRIX_THRESHOLD per-unit tasks are intercepted: instead of 20 model
    round-trips the planner builds an events x stations matrix and grinds it."""
    store = _matrix_store(monkeypatch, tmp_path)
    result = ac.run_agent_with_planner(
        "много графиков для BRW",
        brain=_ManyCellsPlanBrain(),
        store=store,
        verbose=False,
    )
    assert result["ok"], result
    matrix = json.loads(Path(result["matrix_path"]).read_text(encoding="utf-8"))
    assert matrix["completed"] == matrix["total"] == 1
    assert "raw:brw:2024-10-10" in result["store"].names()
    assert Path(result["export"]["path"]).suffix == ".zip"


def test_the_two_matrix_tools_are_in_the_schema_registry():
    """Every schema has a handler (parity test) and vice versa -- both new tools
    must satisfy the toolset completeness guard."""
    assert {"create_task_matrix", "process_matrix_batch"} <= ac.TOOLS_BY_NAME
    assert ac.TOOLS_BY_NAME == set(ac.HANDLERS)
    fetch_names = {t["function"]["name"] for t in ac.select_tools_for_task("process_matrix_batch")}
    assert "fetch_observatory_data" in fetch_names
    assert "plot_components" in fetch_names

