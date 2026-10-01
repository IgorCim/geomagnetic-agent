"""Handles must be canonicalised at the boundary, not just tolerated.

H1 is refuted by probe: the store resolves upper-case handles fine. But the probe
found the real defect -- `fetch_observatory_data` *advertises* a mixed-case name
that is not a key in the store:

    reported frame_handle : 'raw:IRT:2024-09-10'
    store really holds    : ['raw:irt:2024-09-10']
    -> is the advertised handle a real key? False

So every handle string the model is told to copy is one it cannot find in
`available_handles`, and correctness depends entirely on the store compensating.
These tests fail on the current code and pin the invariant.
"""

import pandas as pd
import pytest

import agent_core as ac
import intermagnet_loader as loader


def _day(day: str) -> pd.DataFrame:
    start = pd.to_datetime(day)
    hours = range(24)
    offset = float(start.day)
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=24, freq="h"),
        "X": [float(i) + offset for i in hours],
        "Y": [float(i) * 2 for i in hours],
        "Z": [float(i) * 3 + offset for i in hours],
    })


@pytest.fixture
def two_days(monkeypatch):
    def fake_fetch(station_code=None, start_date=None, end_date=None, **kw):
        frame = _day(start_date)
        frame.attrs["publication_state"] = "definitive"
        return frame

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
# the boundary invariant
# --------------------------------------------------------------------------- #
def test_every_advertised_handle_is_a_real_store_key(two_days):
    """The core invariant: what the tool prints, the store can look up."""
    store = two_days
    for day in ("2024-09-10", "2024-09-11"):
        payload, note = ac._handle_fetch(
            {"station_code": "IRT", "start_date": day, "end_date": day}, store
        )
        handle = payload["frame_handle"]
        assert handle in store.names(), (
            f"fetch advertised {handle!r}, which is not a key in {store.names()}"
        )
        assert handle == handle.lower(), f"handle must be lower-case: {handle!r}"
        assert handle in note, f"the note must quote the same handle: {note}"


def test_derive_advertises_a_real_handle(two_days):
    for source in ("raw:irt:2024-09-10", "raw:irt:2024-09-11"):
        payload, _ = ac._handle_derive(
            {"df": source, "components": ["H"]}, two_days
        )
        handle = payload["frame_handle"]
        assert handle in two_days.names(), handle
        assert handle in payload["available_handles"]


def test_anomalies_advertises_a_real_handle(two_days):
    payload, _ = ac._handle_anomalies(
        {"df": "raw:irt:2024-09-10", "component": "X"}, two_days
    )
    handle = payload["frame_handle"]
    assert handle in two_days.names(), handle


def test_fetch_keeps_the_upper_case_code_only_in_metadata(two_days):
    """The station code stays upper-case in the payload; only the handle is lower."""
    payload, _ = ac._handle_fetch(
        {"station_code": "IRT", "start_date": "2024-09-10", "end_date": "2024-09-10"},
        two_days,
    )
    assert payload["station"] == "IRT"
    assert payload["frame_handle"] == "raw:irt:2024-09-10"


# --------------------------------------------------------------------------- #
# incoming handles are normalised before any lookup
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("handle", [
    "raw:IRT:2024-09-10",
    "RAW:irt:2024-09-10",
    "  raw:Irt:2024-09-10  ",
    "RaW:IRT:2024-09-10",
])
def test_incoming_handles_are_normalised_before_lookup(two_days, handle):
    frame, err = two_days.resolve(handle)
    assert err is None, err
    assert frame is not None


def test_unknown_frame_error_lists_available_handles(two_days):
    """A real miss must tell the model what it may use instead."""
    _, err = two_days.resolve("raw:irt:2099-01-01")
    assert err["error"] == "unknown_frame"
    assert err["available"] == two_days.names()
    assert "raw:irt:2024-09-10" in err["available"]
    assert err["aliases"]["raw"] == "raw:irt:2024-09-11"
    assert "available" in err["hint"].lower() or "raw:irt" in err["hint"]


def test_miss_lists_handles_even_when_the_store_is_empty():
    store = ac.FrameStore()
    _, err = store.resolve("raw:irt:2024-09-10")
    assert err["error"] == "unknown_frame"
    assert err["available"] == ["(none yet - call fetch_observatory_data first)"]


def test_no_tool_call_can_emit_an_unknown_handle_for_a_known_day(two_days, tmp_path):
    """Every df argument in the log must resolve, whatever case it arrived in."""
    import geomag_plotter as plotter

    plotter.DEFAULT_OUTPUT_DIR = tmp_path
    cases = [
        ("get_statistics", {"df": "RAW:IRT:2024-09-10", "components": ["X"]}),
        ("calculate_derived_components", {"df": " RAW:IRT:2024-09-11 ", "components": ["H"]}),
        ("detect_anomalies", {"df": "RaW:IRT:2024-09-10", "component": "X"}),
        ("plot_components", {"df": "RAW:IRT:2024-09-11", "components": ["X"]}),
        ("plot_comparison", {
            "df1": "RAW:IRT:2024-09-10",
            "df2": "Raw:Irt:2024-09-11",
            "component": "X",
        }),
    ]
    for tool, args in cases:
        payload, _ = ac.HANDLERS[tool](args, two_days)
        assert not ac.is_error(payload), f"{tool} {args} -> {payload}"


def test_run_agent_uppercase_handles_leave_no_unknown_frame(two_days, tmp_path):
    """End to end: a model that shouts every handle still gets a clean log."""

    def call(name, arguments):
        return {
            "_text": f'<tool_call>{{"name": "{name}", "arguments": {arguments}}}'
                     "</tool_call>"
        }

    import geomag_plotter as plotter

    plotter.DEFAULT_OUTPUT_DIR = tmp_path
    script = [
        call("fetch_observatory_data",
             '{"station_code": "IRT", "start_date": "2024-09-10", "end_date": "2024-09-10"}'),
        call("fetch_observatory_data",
             '{"station_code": "IRT", "start_date": "2024-09-11", "end_date": "2024-09-11"}'),
        call("calculate_derived_components", '{"df": "RAW:IRT:2024-09-10", "components": ["H"]}'),
        call("calculate_derived_components", '{"df": " RAW:IRT:2024-09-11 ", "components": ["H"]}'),
        call("plot_comparison",
             '{"df1": "RAW:IRT:2024-09-10", "df2": "Raw:Irt:2024-09-11", "component": "H"}'),
        {"content": "Сравнил 2024-09-10 и 2024-09-11."},
    ]
    brain = ac.ScriptedBrain(ac._unwrap(script))
    result = ac.run_agent("сравни H за 10 и 11 сентября", brain=brain, verbose=False)

    unknown = [e for e in result["tool_calls"] if e.get("error") == "unknown_frame"]
    assert unknown == [], [e["arguments"] for e in unknown]
    assert all(e["ok"] for e in result["tool_calls"]), result["tool_calls"]
    assert len(result["plots"]) == 1
    # the failure footer must not appear when nothing failed
    assert "Не удалось выполнить" not in result["text"]


def test_uppercase_plot_comparison_has_two_traces(two_days, tmp_path):
    import geomag_plotter as plotter

    plotter.DEFAULT_OUTPUT_DIR = tmp_path
    captured = {}
    real_write = plotter._write

    def spy(fig, path, include_plotlyjs):
        captured["fig"] = fig
        return real_write(fig, path, include_plotlyjs)

    for day in ("2024-09-10", "2024-09-11"):
        ac._handle_derive({"df": f"raw:irt:{day}", "components": ["H"]}, two_days)

    plotter._write = spy
    try:
        payload, _ = ac._handle_plot_comparison({
            "df1": "RAW:IRT:2024-09-10",
            "df2": "RAW:IRT:2024-09-11",
            "component": "H",
        }, two_days)
    finally:
        plotter._write = real_write

    assert not ac.is_error(payload), payload
    assert len(captured["fig"].data) == 2
    assert captured["fig"].data[0].name != captured["fig"].data[1].name
