"""Check the scenario harness itself: a checker that cannot fail proves nothing.

These run the *verifier* against synthetic run_agent outputs, so the E2E suite
is trusted even though the real model needs a GPU we do not have locally.
"""

import scenarios_e2e as e2e


def _result(tool_calls, text="", plots=()):
    return {"ok": True, "text": text, "plots": list(plots), "tool_calls": tool_calls}


def _ok(tool, args, **extra):
    entry = {"tool": tool, "arguments": args, "ok": True}
    entry.update(extra)
    return entry


def _fetch(start, end="2024-09-10"):
    return _ok("fetch_observatory_data", {"station_code": "IRT",
                                          "start_date": start, "end_date": end})


def _by_name(scenario):
    return next(s for s in e2e.SCENARIOS if s.name == scenario)


def test_good_compare_run_passes(tmp_path):
    plot = tmp_path / "cmp.html"
    plot.write_text("<html></html>", encoding="utf-8")
    scenario = _by_name("compare")
    result = _result(
        [
            _fetch("2024-09-10", "2024-09-10"),
            _fetch("2024-09-11", "2024-09-11"),
            _ok("calculate_derived_components", {"df": "raw:irt:2024-09-10"}),
            _ok("calculate_derived_components", {"df": "raw:irt:2024-09-11"}),
            _ok("plot_comparison",
                {"df1": "derived:irt:2024-09-10", "df2": "derived:irt:2024-09-11",
                 "component": "H"},
                plot=str(plot)),
        ],
        text=(
            "Сравнил 2024-09-10 и 2024-09-11: кривые почти совпадают, "
            "медиана 31300 нТ против 31500 нТ."
        ),
        plots=[str(plot)],
    )
    assert e2e.check(scenario, result) == []


def test_the_original_failure_is_caught(tmp_path):
    """Two fetches, then a comparison that silently reuses one handle."""
    plot = tmp_path / "cmp.html"
    plot.write_text("<html></html>", encoding="utf-8")
    scenario = _by_name("compare")
    result = _result(
        [
            _fetch("2024-09-10", "2024-09-10"),
            _fetch("2024-09-11", "2024-09-11"),
            _ok("plot_comparison",
                {"df1": "raw", "df2": "raw", "component": "H"},
                plot=str(plot)),
        ],
        text="Готово: 10 и 11 сентября 2024 сравнены.",
        plots=[str(plot)],
    )
    problems = e2e.check(scenario, result)
    assert any("same handle" in p for p in problems), problems


def test_missing_tool_is_caught():
    """Both days are fetched, but the comparison never happens."""
    scenario = _by_name("compare")
    result = _result(
        [
            _fetch("2024-09-10", "2024-09-10"),
            _fetch("2024-09-11", "2024-09-11"),
            _ok("get_statistics", {"df": "raw"}),
        ],
        text="Сравнил дни 2024-09-10 и 2024-09-11.",
    )
    problems = e2e.check(scenario, result)
    assert any("plot_comparison" in p for p in problems), problems


def test_second_fetch_missing_is_caught():
    scenario = _by_name("compare")
    result = _result(
        [
            _fetch("2024-09-10", "2024-09-10"),
            _ok("get_statistics", {"df": "raw"}),
        ],
        text="Сравнил дни.",
    )
    problems = e2e.check(scenario, result)
    assert any("fetch_observatory_data" in p for p in problems), problems


def test_failed_tool_is_caught():
    scenario = _by_name("medians")
    failed = _ok("get_statistics", {"df": "raw"})
    failed.update({"ok": False, "error": "unknown_frame", "message": "no data"})
    result = _result(
        [_fetch("2024-09-10", "2024-09-10"), _fetch("2024-09-11", "2024-09-11"), failed],
        text="Не смог посчитать.",
    )
    problems = e2e.check(scenario, result)
    assert any("unknown_frame" in p for p in problems), problems


def test_both_fetches_of_the_same_day_is_caught():
    scenario = _by_name("medians")
    result = _result(
        [
            _fetch("2024-09-10", "2024-09-10"),
            _fetch("2024-09-10", "2024-09-10"),
            _ok("get_statistics", {"df": "raw"}),
        ],
        text="Медиана за 10 и 11 сентября 2024.",
    )
    problems = e2e.check(scenario, result)
    assert any("same window" in p for p in problems), problems


def test_missing_plot_artifact_is_caught():
    scenario = _by_name("one_day")
    result = _result(
        [
            _fetch("2024-09-10", "2024-09-10"),
            _ok("calculate_derived_components", {"df": "raw:irt:2024-09-10"}),
            _ok("plot_components", {"df": "derived:irt:2024-09-10"},
                plot="plots/does_not_exist.html"),
        ],
        text="График готов.",
        plots=["plots/does_not_exist.html"],
    )
    problems = e2e.check(scenario, result)
    assert any("missing" in p for p in problems), problems


def test_answer_omitting_a_day_is_caught():
    scenario = _by_name("medians")
    result = _result(
        [
            _fetch("2024-09-10", "2024-09-10"),
            _fetch("2024-09-11", "2024-09-11"),
            _ok("get_statistics", {"df": "raw"}),
        ],
        text="Медиана примерно 31300 нТ.",
    )
    problems = e2e.check(scenario, result)
    assert any("2024-09-11" in p for p in problems), problems


def test_dry_run_lists_every_scenario():
    assert e2e.main.__module__ == "scenarios_e2e"
    names = {s.name for s in e2e.SCENARIOS}
    assert names == {"one_day", "compare", "medians"}
    for scenario in e2e.SCENARIOS:
        assert scenario.query.strip()
        assert scenario.must_call
