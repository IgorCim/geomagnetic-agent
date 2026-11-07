"""Regression gates for results a portfolio viewer must actually receive."""
import json
from pathlib import Path

import pytest

import agent_core as ac
import gradio_app as ga
import project_store as ps
from test_handles_and_plots import _write_plot


def test_manual_zip_does_not_overwrite_custom_archive(tmp_path, monkeypatch):
    import zipfile
    monkeypatch.setattr(ps, 'PROJECTS_ROOT', tmp_path / 'projects')
    monkeypatch.setattr(ps, 'EXPORTS_ROOT', tmp_path / 'exports')
    ps.create_project('custom')
    root = ps.project_dir('custom')
    for component in 'XYZ':
        (root / f'{component}.html').write_text(f'<html>{component}</html>')
    custom = ps.export_project('custom', folders={str(i): [f'{c}.html'] for i, c in enumerate('XYZ', 1)})
    assert not ps.is_error(custom), custom
    original = Path(custom['path']).read_bytes()
    first = ga._make_zip('custom')['value']
    second = ga._make_zip('custom')['value']
    assert len({str(first), str(second), custom['path']}) == 3
    assert Path(custom['path']).read_bytes() == original
    with zipfile.ZipFile(custom['path']) as archive:
        for i, c in enumerate('XYZ', 1):
            assert archive.read(f'{i}/{c}.html') == f'<html>{c}</html>'.encode()


def test_plot_tally_merges_recovered_requested_filename_only():
    log = [
        {'tool': 'plot_overlay', 'ok': False, 'arguments': {'filename': 'xy.html'}},
        {'tool': 'plot_components', 'ok': True, 'arguments': {'filename': 'xy.html'}, 'plot': '/plots/xy_INTERMAGNET.html'},
    ]
    text = ac._with_plot_tally('', log)
    assert 'Построено графиков: 1.' in text
    assert 'Не построено' not in text
    log.append({'tool': 'plot_components', 'ok': False, 'arguments': {'filename': 'z.html'}})
    text = ac._with_plot_tally('', log)
    assert '1 из 2' in text
    assert 'z.html' in text


def test_history_is_sent_to_model_but_routing_uses_only_current_question(monkeypatch):
    seen = {}
    def run(query, **kwargs):
        seen.update(kwargs)
        seen["query"] = query
        return {"ok": True, "text": "fixture", "plots": []}
    monkeypatch.setattr(ga, "run_agent", run)
    history = [{"role": "user", "content": "Получи Kyoto и NOAA"},
               {"role": "assistant", "content": "Старый ответ"}]
    ga.make_chat_handler(object())("Сопоставь H IRT с Bz/V", history, [], ac.FrameStore())
    assert seen["routing_query"] == "Сопоставь H IRT с Bz/V"
    assert "Старый ответ" in seen["query"]


def test_partial_failure_keeps_finished_charts_and_original_diagnostics(tmp_path, monkeypatch):
    path = _write_plot(tmp_path / "finished.html")
    monkeypatch.setattr(ga, "run_agent", lambda *a, **k: {
        "ok": False, "text": "Выполнено 1 из 2 задач; отсутствует второй день.",
        "plots": [path], "tool_calls": [{"tool": "plot_components", "ok": True, "plot": path}],
        "stop_reason": "planner",
    })
    messages, slots, log, saved, files, _ = ga.split_result(ga.make_chat_handler(object())("один график", [], []))
    assert path in files and saved
    assert any(slot.get("visible") for slot in slots)
    assert "Выполнено 1 из 2" in messages[-1]["content"]
    assert "plot_components" in log


def test_more_than_twelve_charts_remain_downloadable(tmp_path, monkeypatch):
    paths = [_write_plot(tmp_path / f"chart_{i}.html") for i in range(14)]
    monkeypatch.setattr(ga, "run_agent", lambda *a, **k: {"ok": True, "text": "14 charts", "plots": paths})
    _, slots, _, saved, files, _ = ga.split_result(ga.make_chat_handler(object())("графики", [], []))
    assert set(files) == set(paths)
    assert len(saved) == 14
    assert sum(bool(slot.get("visible")) for slot in slots) == ga.MAX_PLOTS_IN_PANEL


def test_matrix_result_displays_only_successful_cells_in_its_own_project(tmp_path, monkeypatch):
    ws = ps.Workspace(root=tmp_path / "projects")
    ws.create("demo")
    target = ps.project_dir("demo", ws.root)
    good = target / "irt" / "2024-09-10" / "good_INTERMAGNET.html"
    stale = target / "irt" / "2024-09-11" / "stale_INTERMAGNET.html"
    _write_plot(good)
    _write_plot(stale)
    matrix = target / "demo_matrix.json"
    matrix.write_text(json.dumps({"project": "demo", "tasks": [{"station_code": "IRT", "events_ok": ["2024-09-10"], "events_failed": ["2024-09-11"]}]}), encoding="utf-8")
    monkeypatch.setattr(ga, "run_agent_with_planner", lambda *a, **k: {"ok": False, "text": "Неполное покрытие 1/2", "matrix_path": str(matrix), "missing_cells": [{"station": "IRT", "date": "2024-09-11"}]})
    messages, slots, _, _, files, _ = ga.split_result(ga.make_chat_handler(object())("матрица", [], [], ac.FrameStore(ws)))
    assert str(good.resolve()) in files
    assert str(stale.resolve()) not in files
    assert "Неполное покрытие 1/2" in messages[-1]["content"]


def test_session_fallback_does_not_share_the_active_workspace():
    one = ga._apply_session_logon(None, "")
    two = ga._apply_session_logon(None, "")
    one.workspace.project = "first-user-project"
    assert two.workspace.project is None


def test_new_empty_project_can_be_selected_in_the_ui(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "PROJECTS_ROOT", tmp_path / "projects")
    ps.create_project("empty")
    summary = ga._project_summary("empty")
    assert "empty" in summary
    assert ga._refresh_projects("empty")
