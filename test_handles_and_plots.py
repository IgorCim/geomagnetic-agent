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
