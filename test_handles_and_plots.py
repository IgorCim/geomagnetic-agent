"""Case-insensitive frame handles + multi-plot UI.

Three field bugs are pinned here:

1. ``plot_comparison: unknown_frame: No data under handle 'raw:IRT:2024-09-10'``
   -- a model that echoes back the upper-case station code it typed must still
   resolve the lower-case slot. The store normalises; these tests prove it holds
   for every tool that takes a frame handle.
2. Asking for three days showed only the last chart, because the UI had a single
   ``gr.Plot``. Every figure has to reach the user now.
3. ``ValueError: Cannot process type as image: <class 'dict'>`` -- handing a
   Plotly figure to ``gr.Gallery``. A Gallery only understands raster images, so
   the interactive charts have to be rendered as iframes instead.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import pytest

import agent_core as ac
import geomag_plotter as plotter
import gradio_app as ga


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _day(day: str, offset: float = 0.0) -> pd.DataFrame:
    start = pd.to_datetime(day)
    hours = range(24)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=24, freq="h"),
            "X": [float(i) + offset for i in hours],
            "Y": [float(i) * 2 for i in hours],
            "Z": [float(i) * 3 + offset for i in hours],
        }
    )


@pytest.fixture
def store_with_days(monkeypatch):
    import intermagnet_loader as loader

    def fake_fetch(station_code=None, start_date=None, end_date=None, **kw):
        return _day(start_date, offset=float(pd.to_datetime(start_date).day))

    monkeypatch.setattr(ac, "OFFLINE", False)
    monkeypatch.setattr(loader, "fetch_observatory_data", fake_fetch)
    store = ac.FrameStore()
    for day in ("2024-09-10", "2024-09-11"):
        payload, _ = ac._handle_fetch(
            {"station_code": "IRT", "start_date": day, "end_date": day}, store
        )
        assert not ac.is_error(payload)
    return store


# --------------------------------------------------------------------------- #
# task 1: case-insensitive handles
# --------------------------------------------------------------------------- #
def test_slots_are_stored_lower_case(store_with_days):
    assert store_with_days.names() == [
        "raw:irt:2024-09-10",
        "raw:irt:2024-09-11",
    ]
    # the short family name is an alias, not a stored slot
    assert store_with_days.aliases()["raw"] == "raw:irt:2024-09-11"


def test_the_exact_reported_handle_resolves(store_with_days):
    """The literal string from the field log: 'raw:IRT:2024-09-10'."""
    frame, err = store_with_days.resolve("raw:IRT:2024-09-10")
    assert err is None
    assert frame is store_with_days.get("raw:irt:2024-09-10")


def test_get_statistics_accepts_an_upper_case_handle(store_with_days):
    payload, _ = ac._handle_stats(
        {"df": "raw:IRT:2024-09-10", "components": ["X", "Y", "Z"]}, store_with_days
    )
    assert not ac.is_error(payload), payload
    assert "X" in payload


def test_every_tool_that_takes_a_handle_accepts_any_casing(store_with_days, tmp_path, monkeypatch):
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    cases = [
        ("get_statistics", {"df": "RAW:IRT:2024-09-10", "components": ["X"]}),
        ("calculate_derived_components", {"df": "Raw:IRT:2024-09-11", "components": ["H"]}),
        ("detect_anomalies", {"df": "rAw:IrT:2024-09-10", "component": "X"}),
        ("plot_components", {"df": "RAW:IRT:2024-09-10", "components": ["X"]}),
        (
            "plot_comparison",
            {
                "df1": "RAW:IRT:2024-09-10",
                "df2": "raw:IRT:2024-09-11",
                "component": "X",
            },
        ),
    ]
    for tool, args in cases:
        payload, _ = ac.HANDLERS[tool](args, store_with_days)
        assert not ac.is_error(payload), f"{tool} with {args} -> {payload}"


def test_mixed_case_aliases_still_work(store_with_days):
    for handle in ("RAW", "Raw", "  raw  "):
        frame, err = store_with_days.resolve(handle)
        assert err is None, handle
        # the alias follows the most recent fetch
        assert frame is store_with_days.get("raw:irt:2024-09-11")


def test_unknown_frame_error_shows_lowercase_slots(store_with_days):
    _, err = store_with_days.resolve("raw:irt:2099-01-01")
    assert err["error"] == "unknown_frame"
    assert "raw:irt:2024-09-10" in err["available"]


def test_handle_schema_does_not_pin_an_undersized_enum():
    """The enum is what pushed the model into inventing handles.

    It only ever offered raw/derived/anomalies while fetch advertised
    'raw:irt:2024-09-10', so the model guessed names that were never stored.
    """
    for schema in ac.TOOL_SCHEMAS:
        function = schema["function"]
        for arg_name, spec in function["parameters"]["properties"].items():
            if not arg_name.startswith("df"):
                continue
            assert "enum" not in spec, f"{function['name']}.{arg_name} still has an enum"
            assert "frame_handle" in spec["description"]


def test_system_prompt_states_the_lowercase_rule():
    assert "raw:irt:2024-09-10" in ac.SYSTEM_PROMPT
    assert ac.HANDLE_CASE_RULE in ac.SYSTEM_PROMPT
    assert "нижнем регистре" in ac.SYSTEM_PROMPT


def test_summaries_list_every_slot(store_with_days):
    payload, _ = ac._handle_stats(
        {"df": "RAW:IRT:2024-09-10", "components": ["X"]}, store_with_days
    )
    assert "raw:irt:2024-09-10" in payload["available_handles"]
    assert payload["handle_aliases"]["raw"] == "raw:irt:2024-09-11"


# --------------------------------------------------------------------------- #
# task 2: every plot reaches the UI
# --------------------------------------------------------------------------- #
def _write_plot(path: Path, offset: float = 1.0) -> str:
    """A real saved Plotly document, written to exactly the given path.

    ``plot_components`` is asked for ``output_dir``/``filename`` rather than
    having its return value written by hand, so this is byte-for-byte what a
    plot tool leaves on disk.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = plotter.plot_components(
        _day("2024-09-10", offset=offset),
        components=["H"],
        output_dir=target.parent,
        filename=target.name,
        include_plotlyjs="cdn",
    )
    assert not plotter.is_error(written), written
    return str(target)


def _plot_script(names: list[str]) -> list[dict]:
    """A brain that fetches each day, derives it, then builds one chart.

    The fetches are there because that is the only way a plot call can succeed:
    without data in the store every chart fails with ``unknown_frame`` and there
    is no figure to display.
    """
    script: list[dict] = []
    days = ("2024-09-10", "2024-09-11", "2024-09-12")
    for day in days:
        script.append(
            {
                "_text": '<tool_call>{"name": "fetch_observatory_data", "arguments": '
                f'{{"station_code": "IRT", "start_date": "{day}", '
                f'"end_date": "{day}"}}}}</tool_call>'
            }
        )
    for day in days:
        script.append(
            {
                "_text": '<tool_call>{"name": "calculate_derived_components", '
                f'"arguments": {{"df": "raw:irt:{day}", "components": ["H"]}}}}</tool_call>'
            }
        )
    for name, day in zip(names, days):
        script.append(
            {
                "_text": '<tool_call>{"name": "plot_components", "arguments": '
                f'{{"df": "derived:irt:{day}", "components": ["H"], '
                f'"filename": "{name}"}}}}</tool_call>'
            }
        )
    script.append({"content": "Графики готовы."})
    return script


def test_three_days_fill_three_iframe_slots(tmp_path, monkeypatch):
    names = ["day10.html", "day11.html", "day12.html"]
    for name in names:
        _write_plot(tmp_path / name)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)
    ga.PLOTS_DIR = tmp_path

    brain = ac.ScriptedBrain(ac._unwrap(_plot_script(names)))
    handler = ga.make_chat_handler(brain)
    _messages, slots, _log, saved, files, _box = ga.split_result(
        handler("построй графики за 3 дня", [], [])
    )

    visible = [s for s in slots if s.get("visible")]
    assert len(visible) == 3, "every figure must reach the screen, not just the last"
    assert len(saved) == 3
    assert len(files) == 3
    for slot in visible:
        assert "<iframe" in slot["value"]


def test_an_unreadable_file_does_not_break_the_panel(tmp_path, monkeypatch):
    _write_plot(tmp_path / "good.html")

    real_plot_components = plotter.plot_components

    def flaky(df, components=("X", "Y", "Z"), **kwargs):
        """Second chart is written truncated, the way a killed run leaves it."""
        if kwargs.get("filename") == "bad.html":
            (tmp_path / "bad.html").write_text("<html>truncated", encoding="utf-8")
            return str(tmp_path / "bad.html")
        return real_plot_components(df, components=components, **kwargs)

    monkeypatch.setattr(plotter, "plot_components", flaky)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)
    ga.PLOTS_DIR = tmp_path

    brain = ac.ScriptedBrain(ac._unwrap(_plot_script(["good.html", "bad.html"])))
    handler = ga.make_chat_handler(brain)
    _messages, slots, _log, saved, files, _box = ga.split_result(
        handler("построй два графика", [], [])
    )

    # both files are kept and downloadable even though one cannot be drawn
    assert len(saved) == 2, saved
    assert len([s for s in slots if s.get("visible")]) == 1
    assert len(files) == 2, "the unreadable chart must still be downloadable"
    assert any("bad.html" in p for p in files)


def test_filenames_in_the_answer_are_picked_up(tmp_path, monkeypatch):
    path = _write_plot(tmp_path / "plot_irt_2024_09_10.html")
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    ga.PLOTS_DIR = tmp_path

    result = {
        "ok": True,
        "text": "График сохранён: plot_irt_2024_09_10.html",
        "plots": [],  # the structured list came back empty
        "tool_calls": [],
    }
    assert ga._plot_paths(result) == [path]


def test_plot_paths_does_not_invent_files(tmp_path, monkeypatch):
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    ga.PLOTS_DIR = tmp_path
    result = {
        "ok": True,
        "text": "Файла nope.html у меня нет.",
        "plots": [],
        "tool_calls": [],
    }
    assert ga._plot_paths(result) == []


def test_single_plot_still_renders(tmp_path, monkeypatch):
    path = _write_plot(tmp_path / "one.html")
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    slots = ga._slot_updates([{"question": "q", "path": path}])
    assert len(slots) == ga.MAX_PLOTS_IN_PANEL
    visible = [s for s in slots if s.get("visible")]
    assert len(visible) == 1
    assert "<iframe" in visible[0]["value"]


def test_all_slots_are_hidden_when_nothing_is_saved():
    slots = ga._slot_updates([])
    assert len(slots) == ga.MAX_PLOTS_IN_PANEL
    assert not [s for s in slots if s.get("visible")]


def test_empty_message_does_not_break_the_handler():
    handler = ga.make_chat_handler(object())
    _messages, slots, log, saved, files, box = ga.split_result(handler("", [], []))
    assert not [s for s in slots if s.get("visible")]
    assert log == "" and files == [] and box == ""


# --------------------------------------------------------------------------- #
# task 3: gr.Gallery cannot hold a Plotly chart
# --------------------------------------------------------------------------- #
def test_three_html_paths_survive_gradio_postprocessing(tmp_path, monkeypatch):
    """The reported crash, reproduced through Gradio's own code.

    ``ValueError: Cannot process type as image: <class 'dict'>`` came out of
    ``Gallery._save``. Feeding a Gallery was never going to work: it accepts
    raster images only, and a Plotly figure arrives as a dict. This asserts the
    replacement actually passes through ``gr.HTML.postprocess`` for a real
    three-day answer, so the ValueError cannot come back.
    """
    names = ["day10.html", "day11.html", "day12.html"]
    for name in names:
        _write_plot(tmp_path / name)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)
    ga.PLOTS_DIR = tmp_path

    brain = ac.ScriptedBrain(ac._unwrap(_plot_script(names)))
    handler = ga.make_chat_handler(brain)
    _messages, slots, _log, saved, files, _box = ga.split_result(
        handler("построй графики за 3 дня", [], [])
    )

    component = gr.HTML()
    rendered = []
    for slot in slots:
        # this is the call that used to raise ValueError via Gallery._save
        out = component.postprocess(slot)
        rendered.append(out)

    shown = [r for r in rendered if r.get("visible")]
    assert len(shown) == 3, rendered
    for r in shown:
        assert r["value"].count("<iframe") == 1
    assert len(files) == 3, "gr.Files must still hand over the real files"


def test_no_gallery_component_is_created():
    """The regression, guarded at the source: a Gallery cannot hold a chart."""
    source = Path(ga.__file__).read_text(encoding="utf-8")
    assert "gr.Gallery(" not in source
    assert "Gallery(" not in source.replace("A Gallery", "").replace("a Gallery", "")


def test_the_demo_wires_one_html_slot_per_figure():
    """Slots have to be declared up front: Blocks cannot add them per turn."""
    demo = ga.build_demo(object())
    components = demo.get_config_file()["components"]
    kinds = [c.get("type") for c in components]
    assert "gallery" not in kinds, "no Gallery may be instantiated"
    assert kinds.count("html") >= ga.MAX_PLOTS_IN_PANEL
    assert kinds.count("file") == 1, "gr.Files must survive"
    assert kinds.count("chatbot") == 1


def test_the_declared_slots_are_hidden_at_start():
    demo = ga.build_demo(object())
    labels = [
        c.get("props", {}).get("label")
        for c in demo.get_config_file()["components"]
        if c.get("type") == "html"
    ]
    assert "" in labels
    hidden = [
        c for c in demo.get_config_file()["components"]
        if c.get("type") == "html" and c.get("props", {}).get("visible") is False
    ]
    assert len(hidden) == ga.MAX_PLOTS_IN_PANEL, hidden


def test_handler_and_wiring_agree_on_the_output_count():
    """A silent mismatch here shows up only as a runtime error in the browser."""
    demo = ga.build_demo(object())
    dependencies = demo.get_config_file()["dependencies"]
    assert dependencies, "the demo must wire at least one event"
    expected = 1 + ga.MAX_PLOTS_IN_PANEL + ga.HANDLER_TAIL
    for dependency in dependencies:
        outputs = dependency.get("outputs") or []
        if len(outputs) == expected:
            return
    raise AssertionError(
        f"no event outputs {expected} components "
        f"(messages + {ga.MAX_PLOTS_IN_PANEL} slots + {ga.HANDLER_TAIL} tail)"
    )


def test_iframe_carries_the_document_inline(tmp_path, monkeypatch):
    """srcdoc inlines the chart, so it needs no file route and keeps its modebar."""
    path = _write_plot(tmp_path / "one.html")
    document = Path(path).read_text(encoding="utf-8")
    frame = ga._iframe(path)
    assert frame.startswith("<iframe")
    assert "srcdoc=" in frame

    # the iframe itself must point nowhere: a filesystem path in src is not
    # something a browser can fetch from an http origin. Only the attributes
    # outside the escaped payload are inspected -- the Plotly document itself
    # legitimately contains src= for its CDN script.
    head, _, rest = frame.partition(' srcdoc="')
    assert head == "<iframe", head
    # html.escape(quote=True) leaves no literal quote inside the payload, so the
    # first one closes the attribute
    raw, _, tail = rest.partition('"')
    assert "src" not in head and "src" not in tail, (head, tail)
    assert "style=" in tail and "loading=" in tail, tail
    assert tail.endswith("></iframe>")

    # the payload must be escaped for an attribute context...
    assert "&quot;" in raw or "&lt;" in raw or "&amp;" in raw
    # ...and it must decode back to the original document, Plotly call and all
    import html as html_mod

    restored = html_mod.unescape(raw)
    assert restored == document
    assert "Plotly" in restored
    # the chart stays interactive: our plotter config must survive the round trip
    assert "scrollZoom" in restored, "zoom and pan would be lost"
    assert "responsive" in restored
    assert "Plotly.newPlot(" in restored


def test_iframe_of_a_missing_or_empty_file_is_empty(tmp_path):
    assert ga._iframe(str(tmp_path / "nope.html")) == ""
    empty = tmp_path / "empty.html"
    empty.write_text("   ", encoding="utf-8")
    assert ga._iframe(str(empty)) == ""


def test_seven_charts_use_every_slot(tmp_path, monkeypatch):
    """The panel is a fixed row, so it has to say which charts did not fit."""
    names = [f"day{i}.html" for i in range(10, 17)]
    saved = [{"question": "q", "path": _write_plot(tmp_path / n)} for n in names]
    slots = ga._slot_updates(saved)
    assert len(slots) == ga.MAX_PLOTS_IN_PANEL
    assert len([s for s in slots if s.get("visible")]) == ga.MAX_PLOTS_IN_PANEL


def test_split_result_round_trips():
    out = (["m"], *[{f"i{i}": i} for i in range(ga.MAX_PLOTS_IN_PANEL)],
           "log", [{"path": "p"}], ["p"], "")
    messages, slots, log, saved, files, box = ga.split_result(out)
    assert messages == ["m"]
    assert len(slots) == ga.MAX_PLOTS_IN_PANEL
    assert log == "log" and saved == [{"path": "p"}] and files == ["p"] and box == ""


def test_schemas_are_still_valid_json():
    json.dumps(ac.TOOL_SCHEMAS)


# --------------------------------------------------------------------------- #
# task 3: partial success is stated, not glossed over
# --------------------------------------------------------------------------- #
def _budget_script(names: list[str], budget: int) -> list[dict]:
    """Three days of work, then a filler call that the budget must refuse.

    The filler matters: without it the turn ends cleanly and the tally has
    nothing to say about a missing chart, which is the state the Kaggle report
    described.
    """
    script = _plot_script(names)
    script.insert(
        -1,
        {
            "_text": '<tool_call>{"name": "get_statistics", "arguments": '
            '{"df": "raw:irt:2024-09-10", "components": ["H"]}}</tool_call>'
        },
    )
    return script


def test_budget_is_sized_for_a_three_day_request():
    """A three-day job costs 9 calls; the cap must not cut it short."""
    assert ac.MAX_TOOL_CALLS >= 9 + 7, ac.MAX_TOOL_CALLS
    assert ac.MAX_TOOL_CALLS == 16
    assert ac.MAX_ROUNDS == 18


def test_refused_calls_are_logged_individually(tmp_path, monkeypatch):
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)
    brain = ac.ScriptedBrain(
        ac._unwrap(_budget_script(["d10.html", "d11.html", "d12.html"], budget=9))
    )
    result = ac.run_agent("три дня", brain=brain, max_tool_calls=9, verbose=False)

    refused = [e for e in result["tool_calls"] if not e.get("executed", True)]
    assert len(refused) == 1
    assert refused[0]["tool"] == "get_statistics"
    assert refused[0]["error"] == "tool_budget_exhausted"
    assert refused[0]["executed"] is False
    assert refused[0]["arguments"]["df"] == "raw:irt:2024-09-10"


def test_all_three_charts_survive_a_refused_filler_call(tmp_path, monkeypatch):
    """The Kaggle symptom: three files on disk, one turn cut short.

    Every plot succeeded before the budget ran out, so all three must be in the
    result, on screen and in the file list.
    """
    names = ["day10.html", "day11.html", "day12.html"]
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)
    ga.PLOTS_DIR = tmp_path

    brain = ac.ScriptedBrain(ac._unwrap(_budget_script(names, budget=9)))
    handler = ga.make_chat_handler(brain)
    _messages, slots, _log, saved, files, _box = ga.split_result(
        handler("построй графики за 3 дня", [], [])
    )

    assert len(saved) == 3, saved
    shown = [s for s in slots if s.get("visible")]
    assert len(shown) == 3, "every saved figure must reach the screen"
    assert all("<iframe" in s["value"] for s in shown)
    assert len(files) == 3


def test_plot_result_survives_a_later_refusal(tmp_path, monkeypatch):
    names = ["day10.html", "day11.html", "day12.html"]
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)

    brain = ac.ScriptedBrain(ac._unwrap(_budget_script(names, budget=9)))
    result = ac.run_agent("три дня", brain=brain, max_tool_calls=9, verbose=False)

    assert result["stop_reason"] == "tool_budget_exhausted"
    assert [Path(p).name for p in result["plots"]] == names
    assert all((tmp_path / n).exists() for n in names)


def test_partial_success_states_the_count_and_the_files(tmp_path, monkeypatch):
    """The model's own sentence is optimistic; ours must not be.

    The model claims "three charts ready" after only two were drawn, so the
    tally has to be computed from the log rather than trusted from the text.
    """
    names = ["day10.html", "day11.html", "day12.html"]
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)

    # 8 calls: three fetches, three derives, two plots, then the third is refused
    script = _plot_script(names)[:6] + _plot_script(names)[6:8]
    script.append({
        "_text": '<tool_call>{"name": "plot_components", "arguments": '
        '{"df": "derived:irt:2024-09-12", "components": ["H"], '
        '"filename": "day12.html"}}</tool_call>'
    })
    script.append({"content": "Три графика готовы."})

    brain = ac.ScriptedBrain(ac._unwrap(script))
    result = ac.run_agent("три дня", brain=brain, max_tool_calls=8, verbose=False)

    assert result["stop_reason"] == "tool_budget_exhausted"
    text = result["text"]
    assert "Построено графиков: 2 из 3 запрошенных" in text, text
    assert "day10.html" in text and "day11.html" in text
    assert "day12.html" not in text.split("Файлы:")[1].split("\n")[0]
    assert "Не построено" in text, text


def test_complete_run_still_reports_what_it_saved(tmp_path, monkeypatch):
    names = ["day10.html", "day11.html", "day12.html"]
    for name in names:
        _write_plot(tmp_path / name)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)

    brain = ac.ScriptedBrain(ac._unwrap(_plot_script(names)))
    result = ac.run_agent("три дня", brain=brain, verbose=False)

    assert result["stop_reason"] == "no_tool_calls"
    assert "Построено графиков: 3." in result["text"], result["text"]
    for name in names:
        assert name in result["text"]
    assert "Не построено" not in result["text"]


def test_tally_is_absent_when_nothing_was_plotted(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    brain = ac.ScriptedBrain(ac._unwrap([{
        "_text": '<tool_call>{"name": "get_statistics", "arguments": '
        '{"df": "raw:irt:2024-09-10", "components": ["H"]}}</tool_call>'
    }, {"content": "Нет данных."}]))
    result = ac.run_agent("статистика без данных", brain=brain, verbose=False)
    assert "Построено графиков" not in result["text"]
    assert "Не построено" not in result["text"]


def test_refused_calls_do_not_pollute_the_failure_footer(tmp_path, monkeypatch):
    """A budget cut is reported once, in the tally, not as a list of failures."""
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)
    brain = ac.ScriptedBrain(
        ac._unwrap(_budget_script(["a.html", "b.html", "c.html"], budget=9))
    )
    result = ac.run_agent("три дня", brain=brain, max_tool_calls=9, verbose=False)
    assert "Не удалось выполнить часть операций" not in result["text"], result["text"]


def test_panel_is_large_enough_for_a_three_day_request():
    assert ga.MAX_PLOTS_IN_PANEL >= 4, ga.MAX_PLOTS_IN_PANEL


def test_the_panel_does_not_duplicate_a_repeated_path(tmp_path, monkeypatch):
    """Panel state is shared across turns, so the same file can arrive twice."""
    path = _write_plot(tmp_path / "only.html")
    saved = [{"question": "q", "path": path}, {"question": "q", "path": path}]
    slots = ga._slot_updates(saved)
    shown = [s for s in slots if s.get("visible")]
    assert len(shown) == 1



# --------------------------------------------------------------------------- #
# regression: the tally counts distinct charts, not tool calls
# --------------------------------------------------------------------------- #
def test_tally_counts_three_files_as_three_not_six():
    """Six attempts over three charts is three charts.

    The field case: each chart was attempted twice, the first attempt failing.
    Counting attempts answered "Построено графиков: 3 из 6" for a run where
    all three files existed -- which reads to a user as three failures.
    """
    log = []
    for name in ("day10.html", "day11.html", "day12.html"):
        log.append(
            {
                "round": 1,
                "tool": "plot_components",
                "ok": False,
                "error": "unknown_frame",
                "arguments": {"df": "derived:irt:x", "filename": name},
            }
        )
        log.append(
            {
                "round": 2,
                "tool": "plot_components",
                "ok": True,
                "arguments": {"df": "derived:irt:x", "filename": name},
                "plot": f"C:/out/{name}",
            }
        )
    text = ac._with_plot_tally("Готово.", log)
    assert "Построено графиков: 3." in text, text
    assert " из " not in text, text
    assert "Не построено" not in text, text


def test_tally_does_not_claim_a_chart_twice_for_repeated_successes():
    """The same file written twice is one chart."""
    log = [
        {"round": 1, "tool": "plot_comparison", "ok": True,
         "arguments": {"filename": "cmp.html"}, "plot": "C:/out/cmp.html"},
        {"round": 2, "tool": "plot_comparison", "ok": True,
         "arguments": {"filename": "cmp.html"}, "plot": "C:/out/cmp.html"},
    ]
    text = ac._with_plot_tally("Готово.", log)
    assert "Построено графиков: 1." in text, text


def test_tally_still_reports_a_real_shortfall():
    """De-duplication must not hide a chart that genuinely never arrived."""
    log = [
        {"round": 1, "tool": "plot_components", "ok": True,
         "arguments": {"filename": "a.html"}, "plot": "C:/a.html"},
        {"round": 1, "tool": "plot_components", "ok": True,
         "arguments": {"filename": "b.html"}, "plot": "C:/b.html"},
        {"round": 1, "tool": "plot_components", "ok": False,
         "error": "tool_budget_exhausted", "executed": False,
         "arguments": {"filename": "c.html"}},
    ]
    text = ac._with_plot_tally("Готово.", log)
    assert "Построено графиков: 2 из 3 запрошенных." in text, text
    assert "c.html" in text.split("Не построено")[1]


def test_tally_does_not_blame_the_budget_for_a_handler_refusal():
    """A refused plot is not a budget cut, and the wording used to say it was."""
    log = [
        {"round": 1, "tool": "plot_comparison", "ok": True,
         "arguments": {"filename": "a.html"}, "plot": "C:/a.html"},
        {"round": 1, "tool": "plot_comparison", "ok": False,
         "error": "same_dataframe", "arguments": {"filename": "b.html"}},
    ]
    text = ac._with_plot_tally("Готово.", log)
    assert "лимит вызовов инструментов исчерпан" not in text, text
    assert "вызов не удался" in text, text


def test_tally_names_a_chart_that_was_never_attempted_by_name():
    """A refused call has no plot path, so its intent comes from the arguments."""
    log = [
        {"round": 1, "tool": "plot_components", "ok": True,
         "arguments": {"filename": "a.html"}, "plot": "C:/a.html"},
        {"round": 1, "tool": "plot_components", "ok": False,
         "error": "tool_budget_exhausted", "executed": False,
         "arguments": {"filename": "never_made.html"}},
    ]
    text = ac._with_plot_tally("Готово.", log)
    assert "never_made.html" in text, text


# --------------------------------------------------------------------------- #
# Phase 3: multi-station overlay
# --------------------------------------------------------------------------- #
def test_plot_overlay_two_stations_returns_html_and_offsets(tmp_path, monkeypatch):
    """Two fetched stations overlay onto one HTML file, north first."""
    import intermagnet_loader as loader

    monkeypatch.setattr(ac, "OFFLINE", False)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        loader,
        "fetch_observatory_data",
        lambda station_code=None, start_date=None, end_date=None, **kw: _day(start_date),
    )
    store = ac.FrameStore()
    for code in ("IRT", "API"):
        payload, _ = ac._handle_fetch(
            {"station_code": code, "start_date": "2024-09-10", "end_date": "2024-09-10"},
            store,
        )
        assert not ac.is_error(payload), payload

    payload, note = ac._handle_plot_overlay(
        {
            "stations": ["API", "IRT"],
            "dates": ["2024-09-10"],
            "component": "H",
            "offsets": {"IRT": 150},
        },
        store,
    )
    assert not ac.is_error(payload), payload
    assert Path(payload["plot_path"]).exists()
    assert payload["plot_path"] == payload["path"]
    assert payload["component"] == "H"
    assert payload["date"] == "2024-09-10"
    assert payload["offsets_applied"]["IRT"] == 150.0
    # IRT (geomag-lat ~41) must come before API (southern hemisphere).
    assert payload["stations"].index("IRT") < payload["stations"].index("API")


def test_plot_overlay_auto_fetch_failure_is_reported(tmp_path, monkeypatch):
    """When auto-fetch cannot download, the overlay says so instead of crashing."""
    import intermagnet_loader as loader

    monkeypatch.setattr(ac, "OFFLINE", False)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)

    def broken_fetch(**kw):
        raise TimeoutError("no network in this test")

    monkeypatch.setattr(loader, "fetch_observatory_data", broken_fetch)
    payload, note = ac._handle_plot_overlay(
        {"stations": ["IRT"], "dates": ["2024-09-10"], "component": "H"},
        ac.FrameStore(),
    )
    assert ac.is_error(payload)
    assert payload["error"] == "auto_fetch_failed"


def test_plot_overlay_rejects_an_unknown_time_system(tmp_path, monkeypatch):
    """Only UT/LT/MLT are valid clocks; anything else is invalid_input."""
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    payload, note = ac._handle_plot_overlay(
        {
            "stations": ["IRT"],
            "dates": ["2024-09-10"],
            "component": "H",
            "time_system": "GMT",
        },
        ac.FrameStore(),
    )
    assert ac.is_error(payload)
    assert payload["error"] == "invalid_input"

