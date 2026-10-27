"""Verification suite for :mod:`supermag_loader` and its agent_core wiring.

Run with::

    python -m pytest test_supermag_loader.py -v

The offline tests never touch the network: they patch ``supermag_loader._request``
with a small stand-in and feed the exact shapes JHU/APL's own client
(``supermag-api.py``) documents -- a JSON list of per-minute rows for the data
and indices services, and a plain-text station list for the inventory. The
``network``-marked test hits the live service for real and needs a registered
SuperMAG logon in ``SUPERMAG_LOGON``; without one it is skipped.

Endpoint note
-------------
``https://supermag.jhuapl.edu/mag/?sites=...&format=csv`` (the shape the brief
guessed) answers the interactive *Download Data* page, not data: the query is
assembled in JavaScript and posted to a session-backed service. The machine
readable path is the documented Web Service API
(``services/data-api.php``, ``services/indices.php``,
``services/inventory.php``), and JSON is the format it answers -- the CSV forms
are not reachable without a browser session.
"""

from __future__ import annotations

import inspect
import json
import os

import pandas as pd
import pytest

import agent_core as ac
import supermag_loader as sm


# --------------------------------------------------------------------------- #
# offline doubles
# --------------------------------------------------------------------------- #
class _Resp:
    """Minimal stand-in for ``requests.Response`` used by the offline tests."""

    def __init__(self, text: str = "", status: int = 200):
        self.text = text
        self.status_code = status

    def json(self):
        return json.loads(self.text)


def _install(monkeypatch, mapping):
    """Patch ``sm._request`` to answer each URL fragment from *mapping*.

    Returns the list of URLs actually requested, so a test can assert that a
    failure short-circuited before any network call.
    """
    seen: list[str] = []

    def fake(url, *, timeout=None, retries=None, **kwargs):
        seen.append(url)
        for fragment, value in mapping.items():
            if fragment in url:
                if isinstance(value, Exception):
                    raise value
                return value() if callable(value) else value
        return _Resp("", status=404)

    monkeypatch.setattr(sm, "_request", fake)
    return seen


#: Field names and values mirror the client's own example row (SuperMAG
#: returns each component twice: 'nez' in the standard NEZ frame, 'geo' mapped
#: into geographic coordinates -- both baseline removed).
_SITES_JSON = json.dumps([
    {
        "tval": 1726118400.0, "ext": 60.0, "iaga": "IRT",
        "N": {"nez": -3.942651, "geo": -5.964826},
        "E": {"nez": 4.492887, "geo": 0.389075},
        "Z": {"nez": 7.608168, "geo": 7.608168},
    },
    {   # a data gap: the service keeps the minute and sends nulls
        "tval": 1726118460.0, "ext": 60.0, "iaga": "IRT",
        "N": {"nez": None, "geo": None},
        "E": {"nez": None, "geo": None},
        "Z": {"nez": None, "geo": None},
    },
    {
        "tval": 1726118520.0, "ext": 60.0, "iaga": "IRT",
        "N": {"nez": 12.5, "geo": 11.0},
        "E": {"nez": -8.25, "geo": 1.5},
        "Z": {"nez": 0.0, "geo": 0.0},
    },
])

#: The indices service rides along the station that set each extremum
#: (SMLmlat/SMLstid/...); those describe a station, not a time series.
_INDICES_JSON = json.dumps([
    {"tval": 1726118400.0, "SME": 58.887299, "SML": -27.709055, "SMU": 31.178246,
     "SMLmlat": 73.529922, "SMLstid": "HOP", "SMUstid": "VIZ", "SMEnum": 118},
    {"tval": 1726118460.0, "SME": 120.0, "SML": -410.5, "SMU": 15.0,
     "SMLmlat": 67.1, "SMLstid": "NVS", "SMUstid": "ATU", "SMEnum": 140},
])

_INVENTORY_TEXT = "184\nABK\nACK\nIRT\nNVS\nT42\n"

_LOGON = "geomag-test"


@pytest.fixture(autouse=True)
def _clean_logon(monkeypatch):
    """A stale developer logon must not make an offline test hit the network."""
    monkeypatch.delenv(sm.LOGON_ENV, raising=False)


def _with_logon(monkeypatch, name=_LOGON):
    monkeypatch.setenv(sm.LOGON_ENV, name)


# --------------------------------------------------------------------------- #
# offline: request construction
# --------------------------------------------------------------------------- #
def test_core_url_matches_the_official_client():
    url = sm._core_url("data-api.php", _LOGON, "2024-09-12T00:00", 86400)
    assert url.startswith("https://supermag.jhuapl.edu/services/data-api.php?")
    assert "python&nohead" in url
    assert f"logon={_LOGON}" in url
    assert "start=2024-09-12T00:00" in url
    # the client zero-pads the extent to 12 digits; the service accepts both
    assert "extent=000000086400" in url


def test_window_is_inclusive_of_the_end_day():
    start, extent = sm._window(sm._norm_day("2024-09-12", "s"), sm._norm_day("2024-09-12", "e"))
    assert start == "2024-09-12T00:00" and extent == 86400
    _, two_days = sm._window(sm._norm_day("2024-09-12", "s"), sm._norm_day("2024-09-13", "e"))
    assert two_days == 2 * 86400


def test_long_windows_are_refused_before_any_request(monkeypatch):
    _with_logon(monkeypatch)
    seen = _install(monkeypatch, {})
    res = sm.fetch_supermag_data("2024-01-01", "2024-12-31", station_code="IRT")
    assert sm.is_error(res) and res["error"] == "range_too_long"
    assert res["max_days"] == sm.MAX_DAYS
    assert seen == [], "a capped request must not reach the network"


def test_client_cap_matches_the_service_24_day_limit(monkeypatch):
    """The live service rejects 25 days; do not advertise a longer window."""
    _with_logon(monkeypatch)
    seen = _install(monkeypatch, {})
    res = sm.fetch_supermag_data("2024-05-01", "2024-05-25", data_type="indices")
    assert sm.MAX_DAYS == 24
    assert sm.is_error(res) and res["error"] == "range_too_long"
    assert seen == []


def test_service_time_range_error_is_actionable(monkeypatch):
    """A 24-day request reaches the service; its range error must not look like an outage."""
    _with_logon(monkeypatch)
    _install(monkeypatch, {"indices.php": _Resp("ERROR: Time range too long")})
    res = sm.fetch_supermag_data("2024-05-01", "2024-05-24", data_type="indices")
    assert sm.is_error(res) and res["error"] == "range_too_long"
    assert res["max_days"] == sm.MAX_DAYS
    assert "Time range too long" in res["message"]
    assert "24" in res["hint"]


# --------------------------------------------------------------------------- #
# offline: station data (data_type='sites')
# --------------------------------------------------------------------------- #
def test_sites_frame_carries_the_baseline_removed_components(monkeypatch):
    _with_logon(monkeypatch)
    _install(monkeypatch, {"data-api.php": _Resp(_SITES_JSON)})

    frame = sm.fetch_supermag_data("2024-09-12", "2024-09-12", station_code="irt")

    assert isinstance(frame, pd.DataFrame) and not sm.is_error(frame)
    assert list(frame.columns) == [
        "timestamp", "dX", "dY", "dZ", "X_geo", "Y_geo", "Z_geo",
    ]
    # 'nez' -> dX/dY/dZ, the perturbation about zero the analysis tools want
    assert frame.loc[0, "dX"] == pytest.approx(-3.942651)
    assert frame.loc[0, "dY"] == pytest.approx(4.492887)
    assert frame.loc[0, "dZ"] == pytest.approx(7.608168)
    # 'geo' -> the same vector mapped into geographic coordinates
    assert frame.loc[0, "X_geo"] == pytest.approx(-5.964826)
    assert frame.loc[2, "Y_geo"] == pytest.approx(1.5)
    # the station code is normalised into the URL and recorded in attrs
    assert frame.attrs["station"] == "IRT"
    assert frame.attrs["units"] == "nT"


def test_gaps_stay_in_the_frame_as_nan(monkeypatch):
    """A gap must not be interpolated or dropped: the tools count them."""
    _with_logon(monkeypatch)
    _install(monkeypatch, {"data-api.php": _Resp(_SITES_JSON)})

    frame = sm.fetch_supermag_data("2024-09-12", "2024-09-12", station_code="IRT")

    assert len(frame) == 3
    assert frame["dX"].isna().sum() == 1
    assert frame.loc[1, "dX"] is pd.NA or pd.isna(frame.loc[1, "dX"])
    # tz-aware UTC, whatever the underlying pandas unit happens to be
    assert "UTC" in str(frame["timestamp"].dtype)
    assert frame["timestamp"].dt.tz is not None


def test_rows_are_sorted_by_timestamp(monkeypatch):
    _with_logon(monkeypatch)
    shuffled = json.dumps(list(reversed(json.loads(_SITES_JSON))))
    _install(monkeypatch, {"data-api.php": _Resp(shuffled)})

    frame = sm.fetch_supermag_data("2024-09-12", "2024-09-12", station_code="IRT")

    assert frame["timestamp"].is_monotonic_increasing
    assert frame.loc[0, "dX"] == pytest.approx(-3.942651)


def test_the_station_request_names_the_station(monkeypatch):
    _with_logon(monkeypatch)
    seen = _install(monkeypatch, {"data-api.php": _Resp(_SITES_JSON)})
    sm.fetch_supermag_data("2024-09-12", "2024-09-12", data_type="sites", station_code="irt")
    assert any("station=IRT" in url for url in seen)


# --------------------------------------------------------------------------- #
# offline: substorm indices (data_type='indices')
# --------------------------------------------------------------------------- #
def test_indices_frame_is_sme_sml_smu_only(monkeypatch):
    _with_logon(monkeypatch)
    _install(monkeypatch, {"indices.php": _Resp(_INDICES_JSON)})

    frame = sm.fetch_supermag_data("2024-09-12", "2024-09-12", data_type="indices")

    assert isinstance(frame, pd.DataFrame) and not sm.is_error(frame)
    assert list(frame.columns) == ["timestamp", "SME", "SML", "SMU"]
    assert frame.loc[0, "SME"] == pytest.approx(58.887299)
    assert frame.loc[1, "SML"] == pytest.approx(-410.5)
    # the station that set the extremum is not a time series
    assert not any("stid" in str(c).lower() for c in frame.columns)


def test_indices_url_asks_for_the_three_substorm_indices(monkeypatch):
    _with_logon(monkeypatch)
    seen = _install(monkeypatch, {"indices.php": _Resp(_INDICES_JSON)})
    sm.fetch_supermag_data("2024-09-12", "2024-09-12", data_type="indices")
    assert any("indices=sme,sml,smu" in url for url in seen)
    assert not any("station=" in url for url in seen)


def test_a_indices_call_omitting_station_code_does_not_raise(monkeypatch):
    """SME/SML/SMU are global: the documented call passes no station at all."""
    _with_logon(monkeypatch)
    seen = _install(monkeypatch, {"indices.php": _Resp(_INDICES_JSON)})

    frame = sm.fetch_supermag_data(
        start_date="2024-09-12", end_date="2024-09-12", data_type="indices"
    )

    assert isinstance(frame, pd.DataFrame) and not sm.is_error(frame)
    assert list(frame.columns) == ["timestamp", "SME", "SML", "SMU"]
    assert not any("station=" in url for url in seen)


def test_the_signature_leaves_station_code_optional():
    """Dates first; station_code last, defaulted to the 'GLOBAL' placeholder."""
    params = list(inspect.signature(sm.fetch_supermag_data).parameters.values())
    assert [p.name for p in params[:4]] == [
        "start_date", "end_date", "data_type", "station_code",
    ]
    assert params[0].default is inspect.Parameter.empty  # only start_date is a must
    assert params[1].default is None                      # end_date -> start_date
    assert params[2].default == "sites"
    assert params[3].default == "GLOBAL"


def test_the_global_placeholder_is_accepted_positionally(monkeypatch):
    """station_code='GLOBAL' (the default) never reaches the indices URL."""
    _with_logon(monkeypatch)
    seen = _install(monkeypatch, {"indices.php": _Resp(_INDICES_JSON)})

    frame = sm.fetch_supermag_data("2024-09-12", "2024-09-12", "indices", "GLOBAL")

    assert isinstance(frame, pd.DataFrame)
    assert not any("station=" in url for url in seen)


def test_sites_still_need_a_real_station(monkeypatch):
    """The 'GLOBAL' default is a placeholder: sites must name a station."""
    _with_logon(monkeypatch)
    seen = _install(monkeypatch, {})
    res = sm.fetch_supermag_data("2024-09-12", "2024-09-12")
    assert sm.is_error(res) and res["error"] == "station_not_found"
    assert "station_code" in res["message"]
    assert seen == [], "a placeholder station must not reach the network"


# --------------------------------------------------------------------------- #
# offline: the strict error contract
# --------------------------------------------------------------------------- #
def test_missing_logon_is_reported_before_any_request(monkeypatch):
    seen = _install(monkeypatch, {"data-api.php": _Resp(_SITES_JSON)})
    res = sm.fetch_supermag_data("2024-09-12", "2024-09-12", station_code="IRT")
    assert sm.is_error(res) and res["error"] == "missing_logon"
    assert sm.LOGON_ENV in res["hint"]
    assert seen == [], "no logon means no request"


def test_an_explicit_logon_wins_over_the_environment(monkeypatch):
    _install(monkeypatch, {"data-api.php": _Resp(_SITES_JSON)})
    frame = sm.fetch_supermag_data("2024-09-12", "2024-09-12", station_code="IRT", logon=_LOGON)
    assert isinstance(frame, pd.DataFrame)


def test_service_rejecting_the_logon_is_named(monkeypatch):
    _with_logon(monkeypatch)
    _install(monkeypatch, {"data-api.php": _Resp("ERROR: Invalid username")})
    res = sm.fetch_supermag_data("2024-09-12", "2024-09-12", station_code="IRT")
    assert sm.is_error(res) and res["error"] == "invalid_logon"
    assert sm.LOGON_ENV in res["hint"]


def test_empty_window_is_no_data_when_the_station_is_known(monkeypatch):
    _with_logon(monkeypatch)
    _install(monkeypatch, {"data-api.php": _Resp("OK"),
                           "inventory.php": _Resp(_INVENTORY_TEXT)})
    res = sm.fetch_supermag_data("2024-09-12", "2024-09-12", station_code="IRT")
    assert sm.is_error(res) and res["error"] == "no_data"
    assert res["hint"] and "indices" in res["hint"]


def test_empty_window_is_station_not_found_for_an_unknown_code(monkeypatch):
    """The same empty body means two different things; inventory tells them apart."""
    _with_logon(monkeypatch)
    _install(monkeypatch, {"data-api.php": _Resp("OK"),
                           "inventory.php": _Resp(_INVENTORY_TEXT)})
    res = sm.fetch_supermag_data("2024-09-12", "2024-09-12", station_code="ZZZ")
    assert sm.is_error(res) and res["error"] == "station_not_found"
    assert res["station"] == "ZZZ"


def test_a_malformed_station_code_never_reaches_the_network(monkeypatch):
    _with_logon(monkeypatch)
    seen = _install(monkeypatch, {})
    res = sm.fetch_supermag_data("2024-09-12", "2024-09-12", station_code="Irkutsk")
    assert sm.is_error(res) and res["error"] == "station_not_found"
    assert seen == []


def test_indices_without_rows_is_no_data(monkeypatch):
    _with_logon(monkeypatch)
    _install(monkeypatch, {"indices.php": _Resp("OK")})
    res = sm.fetch_supermag_data("2024-09-12", "2024-09-12", data_type="indices")
    assert sm.is_error(res) and res["error"] == "no_data"


def test_bad_dates_and_bad_data_type(monkeypatch):
    _with_logon(monkeypatch)
    seen = _install(monkeypatch, {})

    assert sm.fetch_supermag_data("12.09.2024", station_code="IRT")["error"] == "invalid_date"
    assert sm.fetch_supermag_data("2024-09-13", "2024-09-12", station_code="IRT")["error"] == "invalid_date"
    assert sm.fetch_supermag_data("2024-09-12", "2024-09-12", data_type="melt", station_code="IRT")["error"] == "invalid_data_type"
    assert seen == []


def test_network_failure_is_not_confused_with_missing_data(monkeypatch):
    _with_logon(monkeypatch)
    _install(monkeypatch, {"data-api.php": ConnectionError("boom")})
    res = sm.fetch_supermag_data("2024-09-12", "2024-09-12", station_code="IRT")
    assert sm.is_error(res) and res["error"] == "supermag_unavailable"
    assert res["hint"]


def test_http_error_is_unavailable(monkeypatch):
    _with_logon(monkeypatch)
    _install(monkeypatch, {"indices.php": _Resp("gateway", status=503)})
    res = sm.fetch_supermag_data("2024-09-12", "2024-09-12", data_type="indices")
    assert sm.is_error(res) and res["error"] == "supermag_unavailable"


# --------------------------------------------------------------------------- #
# offline: the inventory parser
# --------------------------------------------------------------------------- #
def test_inventory_parses_the_documented_text_shape():
    assert sm._parse_inventory(_INVENTORY_TEXT) == ["ABK", "ACK", "IRT", "NVS", "T42"]


def test_inventory_parses_a_json_list():
    assert sm._parse_inventory(json.dumps(["ABK", "IRT", "NVS"])) == ["ABK", "IRT", "NVS"]


def test_inventory_handles_an_empty_answer():
    assert sm._parse_inventory("") == []
    assert sm._parse_inventory("0\n") == []


# --------------------------------------------------------------------------- #
# agent_core wiring
# --------------------------------------------------------------------------- #
def test_tool_schema_is_registered():
    schema = next(
        (t for t in ac.TOOL_SCHEMAS
         if t["function"]["name"] == "fetch_supermag_data"),
        None,
    )
    assert schema is not None
    params = schema["function"]["parameters"]
    assert params["required"] == ["start_date"]
    assert params["properties"]["data_type"]["enum"] == ["sites", "indices"]
    desc = schema["function"]["description"].lower()
    assert "supermag" in desc and "sme" in desc and "индекс" in desc
    # the LLM must see that no station stands behind the global product
    assert "station_code" in desc and "необязател" in desc


def test_handler_and_selection_are_wired():
    assert ac.HANDLERS["fetch_supermag_data"] is ac._handle_fetch_supermag
    assert "fetch_supermag_data" in ac.TOOL_SELECTION
    assert "fetch_supermag_data" in ac.TOOL_SELECTION["fetch_supermag_data"]
    assert ac._SCRATCHPAD_STEP_BY_TOOL["fetch_supermag_data"] == "supermag"


def test_offline_sites_call_stores_a_working_handle(monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    store = ac.FrameStore()

    payload, note = ac._handle_fetch_supermag(
        {"station_code": "IRT", "start_date": "2024-09-12", "data_type": "sites"},
        store,
    )

    assert payload["ok"] is True and not sm.is_error(payload)
    assert payload["frame_handle"] == "smag:irt:2024-09-12"
    frame = store.get(payload["frame_handle"])
    assert {"dX", "dY", "dZ"} <= set(frame.columns)
    assert payload["offline"] is True
    assert payload["rows"] == len(frame)
    assert note


def test_offline_indices_call_stores_an_auxiliary_handle(monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    store = ac.FrameStore()

    payload, _ = ac._handle_fetch_supermag(
        {"start_date": "2024-09-12", "data_type": "indices"}, store
    )

    assert payload["ok"] is True
    assert payload["frame_handle"] == "idx:supermag:2024-09-12"
    frame = store.get(payload["frame_handle"])
    assert {"SME", "SML", "SMU"} <= set(frame.columns)
    # the substorm indices are an input, not the working magnetometer frame
    assert payload["frame_handle"].split(":", 1)[0] in ac._AUX_FRAME_FAMILIES


def test_handler_requires_a_station_for_sites(monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    payload, _ = ac._handle_fetch_supermag(
        {"start_date": "2024-09-12", "data_type": "sites"}, ac.FrameStore()
    )
    assert ac.is_error(payload) and payload["error"] == "invalid_input"
    assert "station_code" in payload["message"]


def test_handler_rejects_a_unknown_data_type(monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    payload, _ = ac._handle_fetch_supermag(
        {"station_code": "IRT", "start_date": "2024-09-12", "data_type": "melt"},
        ac.FrameStore(),
    )
    assert ac.is_error(payload) and payload["error"] == "invalid_input"


def test_selector_adds_supermag_for_a_substorm_question():
    picked = [t["function"]["name"]
              for t in ac.select_tools_for_query(
                  "покажи SME, SML и SMU за 2024-09-12 — когда началась суббуря?")]
    assert "fetch_supermag_data" in picked
    assert len(picked) <= ac.TOOLS_PER_TASK


def test_prompt_advertises_the_tool_and_stays_in_budget():
    assert "fetch_supermag_data" in ac.SYSTEM_PROMPT
    assert len(ac.SYSTEM_PROMPT) <= 3200


def test_a_supermag_frame_survives_compaction():
    payload = {
        "ok": True,
        "rows": 1440,
        "frame_handle": "smag:irt:2024-09-12",
        "source": "SuperMAG (baseline removed: daily + yearly)",
        "statistics": {c: {"count": 1440, "min": -50.0, "max": 60.0, "mean": 0.1}
                       for c in ("dX", "dY", "dZ", "X_geo", "Y_geo", "Z_geo")},
    }
    compacted = ac._compact_tool_message(payload)
    body = json.loads(compacted)
    assert body["frame_handle"] == "smag:irt:2024-09-12"
    assert body["rows"] == 1440


# --------------------------------------------------------------------------- #
# live network: the SuperMAG Web Service
# --------------------------------------------------------------------------- #
@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get(sm.LOGON_ENV),
    reason=f"set {sm.LOGON_ENV} to a registered SuperMAG logon to run the live check",
)
def test_live_substorm_indices_are_reachable():
    frame = sm.fetch_supermag_data("2024-09-12", "2024-09-12", data_type="indices")
    assert not sm.is_error(frame), frame
    assert {"SME", "SML", "SMU"} <= set(frame.columns)
    assert len(frame) > 1000  # a full day of 1-minute values
    # a substorm day really does push SML well below zero
    assert float(frame["SML"].min()) < -100.0


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get(sm.LOGON_ENV),
    reason=f"set {sm.LOGON_ENV} to a registered SuperMAG logon to run the live check",
)
def test_live_station_data_is_reachable():
    frame = sm.fetch_supermag_data("2024-09-12", "2024-09-12", data_type="sites", station_code="IRT")
    assert not sm.is_error(frame), frame
    assert {"dX", "dY", "dZ"} <= set(frame.columns)
    assert len(frame) > 1000
