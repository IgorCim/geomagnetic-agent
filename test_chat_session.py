"""Adversarial chat tests use the production selector and real handlers."""
from pathlib import Path

import pandas as pd
import pytest

import agent_core as ac
import gradio_app as ga
from chat_session import SessionMemory


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
