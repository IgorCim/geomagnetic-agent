"""Adversarial chat tests use the production selector and real handlers."""
from pathlib import Path

import pandas as pd
import pytest

import agent_core as ac
import gradio_app as ga
from chat_session import SessionMemory


@pytest.mark.parametrize('names', [('1', '2', '3'), ('Земля', 'Ветер')])
def test_export_retry_cannot_drop_literal_destination_folders(names):
    from chat_session import Request
    query = 'Экспортируй проект. ' + ' '.join(f'В папку "{n}" положи график.' for n in names)
    contract = Request(query)
    store = ac.FrameStore()
    assert contract.check_call('export_project', {}, store)
    assert contract.check_call('export_project', {'folders': {'Графики': ['a.html']}}, store)
    assert contract.check_call('export_project', {'folders': {names[0]: ['a.html']}}, store)
    assert contract.check_call('export_project', {'folders': {n: ['a.html'] for n in names}}, store) is None
    # Constraints end with the request; ordinary exports remain available.
    assert Request('Экспортируй проект.').check_call('export_project', {}, store) is None


def call(name, **args):
    import json
    return {"content": "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"}


def seed(store, station="irt", day="2024-09-10"):
    handle = f"raw:{station}:{day}"
    store.put(handle, pd.DataFrame({"timestamp": pd.date_range(day, periods=4, freq="h", tz="UTC"),
                                  "X": [3.] * 4, "Y": [4.] * 4, "Z": [12.] * 4, "F": [13.] * 4}))
    return handle


@pytest.mark.parametrize("q", ["Получить Kp/Dst из Kyoto за первый день", "Kyoto Dst за 2024-09-10",
                              "Получи Dst/Kp, статистику и график"])
def test_real_dst_is_not_a_proxy_intent(q):
    selection = ac.explain_tool_selection(q)
    assert "dst_proxy" not in selection["metrics"]
    assert "fetch_geomagnetic_indices" in selection["tools"]
    assert {"get_statistics", "plot_components"} <= set(selection["tools"])


def test_explicit_proxy_still_has_its_physics_tool():
    assert "dst_proxy" in ac.explain_tool_selection("Рассчитай dst_proxy для IRT")["metrics"]


def test_first_day_survives_long_history_and_january_decoy():
    store, memory = ac.FrameStore(), SessionMemory()
    initial = seed(store)
    memory.record([{"ok": True, "tool": "fetch_observatory_data", "frame_handle": initial}], store)
    decoy = seed(store, day="2024-01-01")
    memory.record([{"ok": True, "tool": "fetch_observatory_data", "frame_handle": decoy}], store)
    plan = memory.prepare("Получить OMNI за первый день", store)
    assert plan.date == "2024-09-10" and plan.target == "sw:omni:2024-09-10"
    assert plan.check_call("fetch_solar_wind", {"start_date": "2024-01-01", "end_date": "2024-01-01"}, store)


def test_first_day_without_provenance_asks_instead_of_guessing():
    store = ac.FrameStore()
    seed(store)
    assert SessionMemory().prepare("Получить OMNI за первый день", store).clarification


def test_failed_bou_does_not_replace_successful_irt_anchor():
    store, memory = ac.FrameStore(), SessionMemory()
    h = seed(store)
    memory.record([{"ok": True, "tool": "fetch_observatory_data", "frame_handle": h}], store)
    memory.record([{"ok": False, "tool": "fetch_observatory_data", "frame_handle": "raw:bou:2024-09-10"}], store)
    assert memory.last_good == h and memory.ground == [h]
    assert memory.prepare("Загрузи BOU за 2024-09-10", store).missing([], store)


def test_no_refetch_and_explicit_table_are_enforced():
    store = ac.FrameStore()
    h = seed(store)
    plan = SessionMemory().prepare(f"Построй H из {h}, без новой загрузки", store)
    assert plan.check_call("fetch_observatory_data", {"station_code": "IRT", "start_date": "2024-09-10"}, store)
    assert plan.check_call("plot_components", {"df_descriptor": "raw:bou:2024-09-10"}, store)
    assert plan.check_call("plot_components", {"df_descriptor": h}, store) is None


def test_unknown_handle_and_independent_memory():
    store = ac.FrameStore()
    h = seed(store)
    first, second = SessionMemory(), SessionMemory()
    first.record([{"ok": True, "tool": "fetch_observatory_data", "frame_handle": h}], store)
    assert not second.ground and second.last_good is None
    plan = first.prepare("Получи H из raw:zzz:1900-01-01", store)
    assert "unknown_frame" in plan.clarification and first.last_good == h


def test_summary_does_not_call_model_and_checks_deleted_files(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("Summary must use session facts, not LLM prose")
    monkeypatch.setattr(ga, "run_agent", forbidden)
    store = ac.FrameStore()
    h = seed(store)
    gone = tmp_path / "gone.html"
    result = ga.make_chat_handler(object())("Сводка по станциям, датам и файлам", [], [{"path": str(gone)}], store)
    text = result[0][-1]["content"]
    assert h in text and "отсутствует" in text and "/path/to/logfile" not in text


def test_skipped_bou_fetch_cannot_finish_as_success():
    store = ac.FrameStore()
    seed(store)
    memory = SessionMemory()
    plan = memory.prepare("Загрузи BOU за 2024-09-10", store)
    result = ac.run_agent(plan.query, brain=ac.ScriptedBrain([{"content": "BOU загружен, медиана 999 nT"}] * 3),
                          store=store, contract=plan, verbose=False)
    assert not result["ok"] and result["error"] == "request_incomplete"
    assert "999" not in result["text"]


def test_wrong_date_is_blocked_before_any_source_request(monkeypatch):
    store, memory = ac.FrameStore(), SessionMemory()
    h = seed(store)
    memory.record([{"ok": True, "tool": "fetch_observatory_data", "frame_handle": h}], store)
    plan = memory.prepare("Получи OMNI за первый день", store)
    def forbidden(*a, **k):
        raise AssertionError("Wrong-date source request must never execute")
    monkeypatch.setitem(ac.HANDLERS, "fetch_solar_wind", forbidden)
    brain = ac.ScriptedBrain([call("fetch_solar_wind", start_date="2024-01-01", end_date="2024-01-01"),
                              {"content": "Данные получены."}] + [{"content": "Готово."}] * 3)
    result = ac.run_agent(plan.query, brain=brain, store=store, contract=plan,
                          routing_query=plan.query, verbose=False)
    assert not result["ok"] and result["tool_calls"][0]["error"] == "request_mismatch"
    assert result["tool_calls"][0]["executed"] is False


def test_statistics_answer_uses_tool_numbers_not_model_numbers():
    store = ac.FrameStore()
    h = seed(store)
    plan = SessionMemory().prepare(f"Получи медиану H из {h}", store)
    brain = ac.ScriptedBrain([call("get_statistics", df_descriptor=h, components=["H"]),
                              {"content": "Медиана H 999 nT"}])
    result = ac.run_agent(plan.query, brain=brain, store=store, contract=plan, verbose=False)
    assert result["ok"] and "999" not in result["text"] and "медиана: 5.0" in result["text"]


def test_failed_statistics_cannot_publish_invented_numbers():
    store = ac.FrameStore()
    h = seed(store)
    plan = SessionMemory().prepare(f"Получи медиану H из {h}", store)
    brain = ac.ScriptedBrain([call("get_statistics", df_descriptor=h, components=["missing"]),
                              {"content": "Медиана H 999 nT"}] + [{"content": "999 nT"}] * 3)
    result = ac.run_agent(plan.query, brain=brain, store=store, contract=plan, verbose=False)
    assert not result["ok"] and "999" not in result["text"]


def test_reset_discards_frames_and_context_but_keeps_files(tmp_path):
    artifact = tmp_path / "saved.html"
    artifact.write_text("saved", encoding="utf-8")
    result = ga._reset_chat()
    assert result[0] == [] and not result[-1].names() and not result[-1].conversation.ground
    assert artifact.is_file()


def test_comparison_keeps_both_dates_but_requires_the_new_day():
    store = ac.FrameStore()
    seed(store)
    plan = SessionMemory().prepare("Загрузи IRT за 2024-09-11 и сравни два дня", store)
    assert plan.target == "raw:irt:2024-09-11"
    seed(store, day="2024-09-11")
    assert plan.check_call("plot_comparison", {"df1": "raw:irt:2024-09-10", "df2": "raw:irt:2024-09-11"}, store) is None
    assert plan.check_call("fetch_observatory_data", {"station_code": "IRT", "start_date": "2024-01-01"}, store)


def test_kyoto_statistics_cannot_be_substituted_with_ground_data():
    store = ac.FrameStore()
    seed(store)
    plan = SessionMemory().prepare("Получи Kyoto Dst/Kp за 2024-09-10, статистику", store)
    assert plan.check_call("get_statistics", {"df_descriptor": "raw:irt:2024-09-10"}, store)


def test_missing_derived_components_are_not_counted_as_done():
    store = ac.FrameStore()
    seed(store)
    plan = SessionMemory().prepare("Загрузи IRT за 2024-09-10, H/D/I", store)
    assert any("H/D/I" in m for m in plan.missing([], store))


def test_state_is_bounded_and_preserves_first_anchor():
    store, memory = ac.FrameStore(), SessionMemory()
    for i in range(140):
        day = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date().isoformat()
        h = seed(store, day=day)
        memory.record([{"ok": True, "tool": "fetch_observatory_data", "frame_handle": h}], store)
    assert len(memory.ground) == 128 and len(memory.recent) == 12
    assert memory.prepare("Получи OMNI за первый день", store).date == "2024-01-01"


def test_first_day_can_be_anchored_by_indices_before_ground_data():
    store, memory = ac.FrameStore(), SessionMemory()
    store.put("idx:kyoto:2024-09-09", pd.DataFrame({"Dst": [-10.]}))
    memory.record([{"ok": True, "tool": "fetch_geomagnetic_indices", "frame_handle": "idx:kyoto:2024-09-09"}], store)
    h = seed(store)
    memory.record([{"ok": True, "tool": "fetch_observatory_data", "frame_handle": h}], store)
    assert memory.prepare("Получи OMNI за первый день", store).date == "2024-09-09"
    assert memory.prepare("Построй H за первый день IRT", store).date == "2024-09-10"


def test_ambiguous_followup_to_a_two_frame_comparison_asks_for_handle():
    store, memory = ac.FrameStore(), SessionMemory()
    first, second = seed(store), seed(store, day="2024-09-11")
    memory.record([{"ok": True, "tool": "plot_comparison", "resolved_handles": [first, second]}], store)
    assert memory.prepare("Вернись к корректному handle и построй график", store).clarification
    assert memory.prepare("Теперь D из тех же данных", store).clarification
    assert not memory.prepare(f"Построй H из {first}", store).clarification


def test_lowercase_station_is_still_a_binding_request():
    store = ac.FrameStore()
    plan = SessionMemory().prepare("Загрузи bou за 2024-09-10", store)
    assert plan.station == "bou" and plan.target == "raw:bou:2024-09-10"
    assert plan.check_call("fetch_observatory_data", {"station_code": "IRT"}, store)


def test_model_invented_raw_handle_cannot_reach_plotter_fallback(monkeypatch):
    store = ac.FrameStore()
    seed(store)
    plan = SessionMemory().prepare("Построй H для BOU за 2024-09-10", store)
    def forbidden(*a, **k):
        raise AssertionError("Missing explicit table must not reach a fallback plotter")
    monkeypatch.setitem(ac.HANDLERS, "plot_components", forbidden)
    brain = ac.ScriptedBrain([call("plot_components", df_descriptor="raw:bou:2024-09-10", components=["H"]),
                              {"content": "График готов."}] + [{"content": "Готово."}] * 3)
    result = ac.run_agent(plan.query, brain=brain, store=store, contract=plan, verbose=False)
    assert not result["ok"] and result["plots"] == []
    assert result["tool_calls"][0]["error"] == "request_mismatch"
    assert result["tool_calls"][0]["executed"] is False


@pytest.mark.parametrize("component,wrong", [("X", ["D"]), ("Y", ["H"]),
                                            ("Z", ["H", "D", "I"]), ("F", ["H", "D", "I"])])
def test_wrong_statistics_component_is_blocked_then_recovered(component, wrong):
    store = ac.FrameStore()
    h = seed(store, "bou")
    plan = SessionMemory().prepare(f"Получи инструментом статистику {component} для BOU за 2024-09-10, без повторной загрузки.", store)
    brain = ac.ScriptedBrain([call("get_statistics", df_descriptor=h, components=wrong),
                              call("get_statistics", df_descriptor=h, components=[component]),
                              {"content": "Медиана 999 nT"}])
    result = ac.run_agent(plan.query, brain=brain, store=store, contract=plan, verbose=False)
    assert result["ok"], result
    rejected, accepted = result["tool_calls"]
    assert rejected["ok"] is False and rejected["executed"] is False
    assert rejected["error"] == "request_mismatch"
    assert "components" in rejected["message"] and component in rejected["message"]
    assert set(accepted["statistics"]) == {component}
    assert "999" not in result["text"]


def test_statistics_correction_names_the_actual_requested_component():
    store = ac.FrameStore()
    h = seed(store, "bou")
    plan = SessionMemory().prepare("Получи статистику X для BOU за 2024-09-10", store)
    assert "components" in plan.query and "X" in plan.query
    missing = "; ".join(plan.missing([], store))
    assert "components" in missing and "X" in missing
    assert plan.check_call("get_statistics", {"df_descriptor": h}, store)


def test_partial_index_statistics_can_be_completed_in_separate_calls():
    store = ac.FrameStore()
    store.put("idx:kyoto:2024-09-10", pd.DataFrame({"Dst": [-10., -20.], "Kp": [1., 2.]}))
    h = "idx:kyoto:2024-09-10"
    plan = SessionMemory().prepare(f"Получи статистику Dst и Kp из {h}", store)
    brain = ac.ScriptedBrain([call("get_statistics", df_descriptor=h, components=["dst"]),
                              call("get_statistics", df_descriptor=h, components=["kp"]), {"content": "Готово."}])
    result = ac.run_agent(plan.query, brain=brain, store=store, contract=plan, verbose=False)
    assert result["ok"] and all(e["ok"] for e in result["tool_calls"]), result


@pytest.mark.parametrize("question", [
    "Кратко перечисли станции, даты и результаты, которые можешь подтвердить доступным контекстом и инструментами. Не восстанавливай забытые числа по догадке; явно укажи пробелы.",
    "Перечисли использованные станции, даты и файлы в этом разговоре.",
    "Дай краткую сводку по станциям, датам и файлам.",
])
def test_summary_variants_use_session_facts_without_model(question, tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("Session summary must not reach the model")
    monkeypatch.setattr(ga, "run_agent", forbidden)
    store = ac.FrameStore()
    seed(store, "bou")
    artifact = tmp_path / "comparison.html"
    artifact.write_text("test", encoding="utf-8")
    result = ga.make_chat_handler(object())(question, [], [{"path": str(artifact)}], store)
    text = result[0][-1]["content"]
    assert "raw:bou:2024-09-10" in text and "comparison.html" in text
    assert "существует" in text and "BOX" not in text


@pytest.mark.parametrize("question", ["Перечисли станции каталога INTERMAGNET", "Перечисли даты наблюдений в каталоге OMNI"])
def test_catalog_list_is_not_a_conversation_summary(question):
    assert not SessionMemory.summary_requested(question)


def test_identical_retried_statistics_are_not_repeated_in_answer():
    store = ac.FrameStore()
    h = seed(store, "bou")
    plan = SessionMemory().prepare(f"Получи статистику X из {h}", store)
    brain = ac.ScriptedBrain([call("get_statistics", df_descriptor=h, components=["X"]),
                              call("get_statistics", df_descriptor=h, components=["X"]), {"content": "Готово."}])
    result = ac.run_agent(plan.query, brain=brain, store=store, contract=plan, verbose=False)
    assert result["ok"] and result["text"].count("- X:") == 1


@pytest.mark.parametrize("component", ["X", "Y", "Z", "F", "H", "D", "I"])
def test_ui_fully_bound_statistics_executes_once_without_model(component, monkeypatch):
    store = ac.FrameStore()
    h = seed(store, "bou")
    def forbidden(*a, **k):
        raise AssertionError("Bound statistics must not ask the model to reselect its arguments")
    monkeypatch.setattr(ga, "run_agent", forbidden)
    original, calls = ac.HANDLERS["get_statistics"], []
    def counted(args, session):
        calls.append(dict(args))
        return original(args, session)
    monkeypatch.setitem(ac.HANDLERS, "get_statistics", counted)
    result = ga.make_chat_handler(object())(
        f"Получи инструментом статистику {component} для BOU за 2024-09-10, без повторной загрузки.", [], [], store)
    text = result[0][-1]["content"]
    assert len(calls) == 1 and calls[0] == {"df_descriptor": h, "components": [component]}
    assert f"- {component}:" in text and "Не удалось" not in text
    assert "get_statistics" in ga.split_result(result)[2]


def test_bound_statistics_failure_is_terminal_without_model_retry(monkeypatch):
    store = ac.FrameStore()
    h = seed(store, "bou")
    store.put(h, store.get(h).drop(columns=["F"]))
    monkeypatch.setattr(ga, "run_agent", lambda *a, **k: pytest.fail("No LLM retry after a real tool error"))
    result = ga.make_chat_handler(object())("Получи статистику F для BOU за 2024-09-10", [], [], store)
    assert "unknown_component" in result[0][-1]["content"]


@pytest.mark.parametrize("question", [
    "Получи статистику X для BOU за 2024-09-11",  # table absent
    "Получи статистику X для BOU за 2024-09-10 и построй график",  # multiple operations
    "Получи статистику X для BOU за 2024-09-10 после 12:00",  # time slice
    "Получи статистику X для BOU за 2024-09-10 в 12:00",  # timestamp
    "Получи медиану X для BOU за 2024-09-10, округли до 4 знаков",  # formatting
    "Получи медиану X для BOU за 2024-09-10 с точностью 0.001",  # precision
])
def test_statistics_fast_path_declines_incomplete_or_extended_requests(question):
    store = ac.FrameStore()
    seed(store, "bou")
    plan = SessionMemory().prepare(question, store)
    assert plan.bound_statistics_args(store) is None


def test_bound_statistics_respects_explicit_handle_and_does_not_mutate_frame():
    store = ac.FrameStore()
    h = seed(store, "bou")
    before = store.get(h).copy(deep=True)
    plan = SessionMemory().prepare(f"Получи медиану Y из {h}, без новой загрузки", store)
    result = ac.run_bound_statistics(plan, store)
    assert result["ok"] and result["rounds"] == 0 and result["stop_reason"] == "bound_statistics"
    assert result["tool_calls"][0]["arguments"]["df_descriptor"] == h
    assert result["tool_calls"][0]["statistics"]["Y"]["median"] == 4.0
    pd.testing.assert_frame_equal(before, store.get(h))


def test_bound_statistics_does_not_accept_empty_success_payload(monkeypatch):
    store = ac.FrameStore()
    h = seed(store, "bou")
    plan = SessionMemory().prepare(f"Получи статистику Y из {h}", store)
    monkeypatch.setitem(ac.HANDLERS, "get_statistics", lambda *a: ({"ok": True}, "placeholder"))
    result = ac.run_bound_statistics(plan, store)
    assert not result["ok"] and result["error"] == "request_incomplete"


def test_lost_anchors_do_not_leak_between_browser_sessions(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("An ambiguous first day must not reach the LLM")
    monkeypatch.setattr(ga, "run_agent", forbidden)
    history = [{"role": "user", "content": "Загрузи IRT за 2024-09-10"}]
    result = ga.make_chat_handler(object())("Получи OMNI за первый день", history, [], ac.FrameStore())
    assert "Укажите дату" in result[0][-1]["content"]


def test_twenty_turn_replay_with_production_router_and_handlers(tmp_path, monkeypatch):
    import geomag_plotter as gp
    monkeypatch.setattr(gp, "DEFAULT_OUTPUT_DIR", tmp_path)
    def fetch(station_code, start_date, end_date, **kwargs):
        s = ac.FrameStore()
        return s.get(seed(s, station_code.lower(), start_date))
    monkeypatch.setattr(ac.loader, "fetch_observatory_data", fetch)
    monkeypatch.setattr(ac, "OFFLINE", False)
    # Network loaders are replaced only at their data boundary; all request
    # checks, selection, parsing, handlers, plotting and UI state stay real.
    def indices(start_date, end_date, **kwargs):
        return pd.DataFrame({"timestamp": pd.date_range(start_date, periods=4, freq="h", tz="UTC"),
                             "Dst": [-10., -20., -30., -40.], "Kp": [1., 2., 3., 4.]})
    def wind(start_date, end_date, **kwargs):
        return pd.DataFrame({"timestamp": pd.date_range(start_date, periods=4, freq="h", tz="UTC"),
                             "Bz": [-1.] * 4, "V": [400.] * 4, "Np": [5.] * 4})
    monkeypatch.setattr(ac.space_weather, "fetch_geomagnetic_indices", indices)
    monkeypatch.setattr(ac.space_weather, "fetch_solar_wind", wind)
    brain = ac.ScriptedBrain([])
    handler, store = ga.make_chat_handler(brain), ac.FrameStore()
    history, saved = [], []
    irt, bou = "derived:irt:2024-09-10", "derived:bou:2024-09-10"
    def step(q, packets=(), expected=None):
        nonlocal history, saved
        brain._script = list(packets) + [{"content": "Готово. Медиана 999 nT; всё проверено."}]
        out = handler(q, history, saved, store)
        history, _, log, saved, _, _ = ga.split_result(out)
        text = history[-1]["content"]
        assert "999" not in text, (q, text)
        assert "tool_not_available" not in text, (q, text)
        assert "request_incomplete" not in text and "Не выполнено:" not in text, (q, text, log)
        if expected:
            assert expected in text, (q, text)
        return text
    step("Загрузи IRT за 2024-09-10. Рассчитай H/D/I, получи статистику H и построй график H", [
        call("fetch_observatory_data", station_code="IRT", start_date="2024-09-10", end_date="2024-09-10"),
        call("calculate_derived_components", df_descriptor="raw:irt:2024-09-10", components=["H", "D", "I"]),
        call("get_statistics", df_descriptor=irt, components=["H"]),
        call("plot_components", df_descriptor=irt, components=["H"], filename="irt_h.html")])
    step("Теперь D из тех же данных, не скачивай повторно", [
        call("get_statistics", df_descriptor=irt, components=["D"]),
        call("plot_components", df_descriptor=irt, components=["D"], filename="irt_d.html")])
    step("Посчитай медиану H для того же дня", [call("get_statistics", df_descriptor=irt, components=["H"])], "медиана: 5.0")
    step("Загрузи IRT за 2024-09-11 и сравни H за два дня", [
        call("fetch_observatory_data", station_code="IRT", start_date="2024-09-11", end_date="2024-09-11"),
        call("plot_comparison", df1=irt, df2="raw:irt:2024-09-11", component="H", filename="compare.html")])
    step("Загрузи BOU за 2024-09-10 и рассчитай H/D/I", [
        call("fetch_observatory_data", station_code="BOU", start_date="2024-09-10", end_date="2024-09-10"),
        call("calculate_derived_components", df_descriptor="raw:bou:2024-09-10", components=["H", "D", "I"])])
    for c in ["X", "Y", "Z", "F", "H"]:
        step(f"Получи статистику {c} для BOU за 2024-09-10, без новой загрузки", [
            call("get_statistics", df_descriptor=bou, components=[c])])
    step("Построй X/Y/Z для BOU за 2024-09-10", [call("plot_components", df_descriptor=bou, components=["X", "Y", "Z"], filename="bou_xyz.html")])
    step("Построй H и F для BOU за 2024-09-10", [call("plot_components", df_descriptor=bou, components=["H", "F"], filename="bou_hf.html")])
    # The chronological first request is September, despite a later January decoy.
    decoy = seed(store, day="2024-01-01")
    store.conversation.record([{"tool": "fetch_observatory_data", "ok": True, "frame_handle": decoy}], store)
    step("Вернись к самому первому дню IRT. Построй H без новой загрузки", [call("plot_components", df_descriptor=irt, components=["H"], filename="first.html")])
    step(f"Используй {irt}. Построй H без новой загрузки", [call("plot_components", df_descriptor=irt, components=["H"], filename="explicit.html")])
    step("Получи Kyoto Dst/Kp за первый день, статистику и график", [
        call("fetch_geomagnetic_indices", start_date="2024-09-10", end_date="2024-09-10"),
        call("get_statistics", df_descriptor="idx:kyoto:2024-09-10", components=["dst", "kp"]),
        call("plot_components", df_descriptor="idx:kyoto:2024-09-10", components=["Dst", "Kp"], filename="kyoto.html")])
    step("Получи OMNI за первый день", [call("fetch_solar_wind", start_date="2024-09-10", end_date="2024-09-10")], "sw:omni:2024-09-10")
    step("Сопоставь H первой IRT с Bz/V за 2024-09-10", [call("plot_space_weather_comparison", ground_df_descriptor=irt,
         solar_wind_df_descriptor="sw:omni:2024-09-10", ground_component="H", solar_wind_components=["Bz", "V"], filename="space.html")])
    step("Получи H из raw:zzz:1900-01-01, без новой загрузки", expected="unknown_frame")
    step(f"Получи статистику H и построй H из {irt}, без новой загрузки", [
        call("get_statistics", df_descriptor=irt, components=["H"]),
        call("plot_components", df_descriptor=irt, components=["H"], filename="recovered.html")])
    step("Краткая сводка по станциям, датам и файлам", expected="Сводка по фактическому")
    assert len(history) == 40 and len(saved) >= 8
