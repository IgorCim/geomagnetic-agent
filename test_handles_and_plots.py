"""Case-insensitive frame handles + multi-plot UI.

Two field bugs are pinned here:

1. ``plot_comparison: unknown_frame: No data under handle 'raw:IRT:2024-09-10'``
   -- a model that echoes back the upper-case station code it typed must still
   resolve the lower-case slot. The store normalises; these tests prove it holds
   for every tool that takes a frame handle.
2. Asking for three days showed only the last chart, because the UI had a single
   ``gr.Plot``. Every figure has to reach the user now.
"""

import json
from pathlib import Path

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


def test_three_days_give_three_gallery_items(tmp_path, monkeypatch):
    names = ["day10.html", "day11.html", "day12.html"]
    for name in names:
        _write_plot(tmp_path / name)
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)
    ga.PLOTS_DIR = tmp_path

    brain = ac.ScriptedBrain(ac._unwrap(_plot_script(names)))
    handler = ga.make_chat_handler(brain)
    _messages, gallery, html_block, _log, saved, files, _box = handler(
        "построй графики за 3 дня", [], []
    )

    # offline mode hands out a synthetic frame, so every plot call must succeed
    assert len(saved) == 3, saved
    assert len(gallery) == 3, "every figure must reach the Gallery, not just the last"
    assert len(files) == 3
    labels = [item["label"] for item in gallery]
    assert len(set(labels)) == 3, labels
    assert html_block.count("<iframe") == 3


def test_gallery_survives_an_unreadable_file(tmp_path, monkeypatch):
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
    _messages, gallery, html_block, _log, saved, _files, _box = handler(
        "построй два графика", [], []
    )

    # both files are kept and downloadable even though one cannot be drawn
    assert len(saved) == 2, saved
    assert len(gallery) == 1, "only the readable figure can be drawn"
    assert html_block.count("<iframe") == 1
    assert len(_files) == 2


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
    saved = [{"question": "q", "path": path}]
    assert len(ga._gallery(saved)) == 1
    assert ga._figure_block(saved).count("<iframe") == 1


def test_gallery_is_empty_when_nothing_is_saved():
    assert ga._gallery([]) == []
    assert ga._figure_block([]) == ""


def test_demo_wires_a_gallery_not_a_single_plot():
    """The UI must actually contain a Gallery: a lone gr.Plot hides all but one."""
    source = Path(ga.__file__).read_text(encoding="utf-8")
    assert "gr.Gallery(" in source
    assert "gallery_view" in source
    assert "gallery_view, plot_view" in source


def test_empty_message_does_not_break_the_handler():
    handler = ga.make_chat_handler(object())
    messages, gallery, html_block, log, saved, files, box = handler("", [], [])
    assert gallery == [] and html_block == "" and log == "" and files == [] and box == ""


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
    result, in the Gallery and in the file list.
    """
    names = ["day10.html", "day11.html", "day12.html"]
    monkeypatch.setattr(plotter, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ac, "OFFLINE", True)
    ga.PLOTS_DIR = tmp_path

    brain = ac.ScriptedBrain(ac._unwrap(_budget_script(names, budget=9)))
    handler = ga.make_chat_handler(brain)
    _messages, gallery, html_block, _log, saved, files, _box = handler(
        "построй графики за 3 дня", [], []
    )

    assert len(saved) == 3, saved
    assert len(gallery) == 3, "every saved figure must reach the Gallery"
    assert html_block.count("<iframe") == 3
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


def test_gallery_does_not_duplicate_a_repeated_path(tmp_path, monkeypatch):
    path = _write_plot(tmp_path / "only.html")
    saved = [{"question": "q", "path": path}, {"question": "q", "path": path}]
    assert len(ga._gallery(saved)) == 1

