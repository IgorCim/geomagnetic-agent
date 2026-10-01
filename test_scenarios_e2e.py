"""Check the scenario harness itself: a checker that cannot fail proves nothing.

These run the *verifier* against synthetic run_agent outputs, so the E2E suite
is trusted even though the real model needs a GPU we do not have locally.
"""

import scenarios_e2e as e2e
from pathlib import Path


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
    assert any("distinct window" in p for p in problems), problems


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
    assert names == {
        "one_day", "compare", "medians", "compare_uppercase", "three_days",
    }
    for scenario in e2e.SCENARIOS:
        assert scenario.query.strip()
        assert scenario.must_call


# --------------------------------------------------------------------------- #
# compare_uppercase: casing must not decide which frame is used
# --------------------------------------------------------------------------- #
def _uppercase_run(tmp_path, days=("2024-09-10", "2024-09-11"), plot_name="cmp.html"):
    plot = tmp_path / plot_name
    plot.write_text("<html></html>", encoding="utf-8")
    log = [_fetch(day, day) for day in days]
    for day in days:
        # the model shouts the handle it passes in, but a correct tool always
        # advertises the canonical key the store holds
        log.append(_ok("calculate_derived_components",
                       {"df": f"RAW:IRT:{day}"},
                       frame_handle=f"derived:irt:{day}",
                       available=[f"raw:irt:{d}" for d in days]))
    log.append(_ok("plot_comparison",
                   {"df1": f"RAW:IRT:{days[0]}", "df2": f"RAW:IRT:{days[1]}",
                    "component": "H"},
                   plot=str(plot)))
    text = (
        f"Сравнил {days[0]} и {days[1]}.\n\nПостроено графиков: 1.\n"
        f"Файлы: {plot.name}"
    )
    return _result(log, text=text, plots=[str(plot)])


def test_shouted_handles_pass_the_comparison_scenario(tmp_path):
    """A model that shouts every handle must still pass cleanly."""
    assert e2e.check(_by_name("compare_uppercase"), _uppercase_run(tmp_path)) == []


def test_a_handle_the_model_could_not_copy_is_caught(tmp_path):
    """The tool advertised a name that is not in its own available list.

    This is the real defect the field report described: the fetch returns
    'raw:IRT:...' while the store holds 'raw:irt:...', so the model copies a name
    it was never given. The scenario has to notice that mismatch.
    """
    result = _uppercase_run(tmp_path)
    for entry in result["tool_calls"]:
        if entry["tool"] == "calculate_derived_components":
            entry["frame_handle"] = "RAW:IRT:2024-09-10"
            break
    problems = e2e.check(_by_name("compare_uppercase"), result)
    assert any("non-canonical" in p for p in problems), problems


def test_unresolvable_handle_is_caught_even_when_the_error_is_unknown_frame(tmp_path):
    result = _uppercase_run(tmp_path)
    result["tool_calls"].append({
        "tool": "plot_components",
        "arguments": {"df": "RAW:IRT:2099-01-01", "components": ["H"]},
        "ok": False,
        "error": "unknown_frame",
        "message": "No data under handle 'RAW:IRT:2099-01-01'.",
    })
    problems = e2e.check(_by_name("compare_uppercase"), result)
    assert any("could not resolve" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# three_days: a run cut short by the budget must say so
# --------------------------------------------------------------------------- #
DAYS3 = ("2024-09-10", "2024-09-11", "2024-09-12")


def _three_day_run(tmp_path, plotted=DAYS3, claims_all_three=True):
    log = [_fetch(day, day) for day in DAYS3]
    for day in DAYS3:
        log.append(_ok("calculate_derived_components", {"df": f"RAW:IRT:{day}"}))
    plots = []
    for day in plotted:
        path = tmp_path / f"H_{day}.html"
        path.write_text("<html></html>", encoding="utf-8")
        plots.append(str(path))
        log.append(_ok("plot_components", {"df": f"DERIVED:IRT:{day}"},
                       plot=str(path)))
    names = ", ".join(Path(p).name for p in plots)
    if claims_all_three:
        # the model's own optimistic sentence: it does not know the budget cut it
        head = "Построил три графика за 10, 11 и 12 сентября 2024."
        tail = f"Построено графиков: 3.\nФайлы: {names}"
    else:
        head = f"Построил {len(plots)} графика."
        tail = (
            f"Построено графиков: {len(plots)} из 3 запрошенных.\n"
            f"Файлы: {names}\nНе построено: 1 — лимит вызовов инструментов исчерпан."
        )
    return _result(log, text=f"{head}\n\n{tail}", plots=plots)


def test_three_charts_pass_the_three_day_scenario(tmp_path):
    assert e2e.check(_by_name("three_days"), _three_day_run(tmp_path)) == []


def test_a_short_run_fails_even_when_it_is_honest(tmp_path):
    """Honesty does not buy a pass: the scenario demands three charts.

    The tally is checked separately, so a truncated run is reported for the real
    defect (a missing chart) and never for the summary merely disagreeing.
    """
    result = _three_day_run(tmp_path, plotted=DAYS3[:2], claims_all_three=False)
    problems = e2e.check(_by_name("three_days"), result)
    assert any("expected >= 3 plot" in p for p in problems), problems
    assert not any("tally disagrees" in p for p in problems), problems
    assert not any("how many charts" in p for p in problems), problems


def test_a_short_run_claiming_three_is_caught(tmp_path):
    """The field failure: the answer says three, the result holds two."""
    result = _three_day_run(tmp_path, plotted=DAYS3[:2], claims_all_three=True)
    problems = e2e.check(_by_name("three_days"), result)
    assert any("expected >= 3 plot" in p for p in problems), problems
    assert any("tally disagrees" in p for p in problems), problems


def test_a_missing_tally_is_caught(tmp_path):
    """A complete three-day run whose answer never states the count."""
    result = _three_day_run(tmp_path)
    result["text"] = "Построил три графика за 10, 11 и 12 сентября 2024."
    problems = e2e.check(_by_name("three_days"), result)
    assert any("how many charts" in p for p in problems), problems


def test_a_tally_that_forgets_a_file_is_caught(tmp_path):
    result = _three_day_run(tmp_path)
    result["text"] = "Готово.\n\nПостроено графиков: 3.\nФайлы: H_2024-09-10.html"
    problems = e2e.check(_by_name("three_days"), result)
    assert any("does not list the saved file" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# the shouting wrapper
# --------------------------------------------------------------------------- #
class _FakeReply:
    def __init__(self, content):
        self.content = content


class _FakeBrain:
    def __init__(self, replies):
        self.replies = list(replies)
        self.extra = "delegates"

    def __call__(self, messages):
        return _FakeReply(self.replies.pop(0))

    def reset(self):
        self.replies = []


def test_shouting_brain_uppercases_only_handle_values():
    brain = _FakeBrain([
        '<tool_call>{"name": "plot_comparison", "arguments": '
        '{"df1": "raw:irt:2024-09-10", "df2": "derived:irt:2024-09-11", '
        '"component": "H"}}</tool_call>',
    ])
    reply = e2e._ShoutingBrain(brain)([{"role": "user", "content": "x"}])
    assert '"df1": "RAW:IRT:2024-09-10"' in reply.content
    assert '"df2": "DERIVED:IRT:2024-09-11"' in reply.content
    # the component and the tool name must survive untouched
    assert '"component": "H"' in reply.content
    assert '"name": "plot_comparison"' in reply.content


def test_shouting_brain_leaves_plain_answers_alone():
    brain = _FakeBrain(["Просто текст без вызовов инструментов."])
    reply = e2e._ShoutingBrain(brain)([{"role": "user", "content": "x"}])
    assert reply.content == "Просто текст без вызовов инструментов."


def test_shouting_brain_forwards_unknown_attributes():
    brain = _FakeBrain(["hi"])
    wrapper = e2e._ShoutingBrain(brain)
    wrapper.reset()
    assert wrapper.extra == "delegates"
