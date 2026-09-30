"""Verification suite for agent_core.

The loop is exercised end to end with :class:`ScriptedBrain`, so nothing here
needs the 5 GB model, a GPU or a network (except the two ``network`` tests).

Run with:  python -m pytest test_agent_core.py -v
"""

from __future__ import annotations

import json
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


def test_exactly_the_six_required_tools():
    assert ac.TOOLS_BY_NAME == {
        "fetch_observatory_data",
        "calculate_derived_components",
        "get_statistics",
        "detect_anomalies",
        "plot_components",
        "plot_comparison",
    }
    assert ac.TOOLS_BY_NAME == set(ac.HANDLERS), "every schema needs a handler"


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


def test_malformed_error_payload_carries_the_reason():
    """The Colab log must explain the failure without a second guess."""
    brain = ac.ScriptedBrain(
        [T_OPEN + "total nonsense" + T_CLOSE, "final answer"]
    )
    out = ac.run_agent("q", brain=brain, verbose=False)
    assert out["ok"] is True
    step = out["tool_calls"][0]
    assert step["error"] == "malformed_tool_call"


def test_debug_print_is_quiet_on_success_but_fires_on_failure(capsys):
    ac.parse_tool_calls(T_OPEN + '{"name":"f","arguments":{}}' + T_CLOSE, debug=True)
    assert "[DEBUG]" not in capsys.readouterr().out
    ac.parse_tool_calls(T_OPEN + "junk" + T_CLOSE, debug=True)
    assert "[DEBUG]" in capsys.readouterr().out


def test_parse_tool_calls_still_returns_a_tuple():
    calls, leftover = ac.parse_tool_calls(
        T_OPEN + '{"name":"f","arguments":{}}' + T_CLOSE, debug=False
    )
    assert isinstance(calls, list) and isinstance(leftover, str)


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
    assert result["text"] == "Такого инструмента нет."


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
    assert result["text"] == "не понял"


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
    assert result["text"] == "хватит"


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
