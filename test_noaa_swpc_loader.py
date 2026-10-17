"""Verification suite for :mod:`noaa_swpc_loader` and its agent_core wiring.

Run with::

    python -m pytest test_noaa_swpc_loader.py -v

The offline tests never touch the network: they patch ``noaa_swpc_loader._request``
with a small stand-in and feed the exact JSON shapes the live endpoints return.
The ``network``-marked tests hit ``services.swpc.noaa.gov`` for real and double as
a health check of the NOAA SWPC service.

Endpoint note
-------------
The Phase 6.3 brief named ``json/geoalert.json`` and ``json/geomagnetic_forecast.json``.
Both return HTTP 404 on the live service; the loader deliberately uses the
*working* products instead (``products/alerts.json``,
``products/noaa-planetary-k-index-forecast.json``,
``json/goes/primary/xray-flares-7-day.json``).
"""

from __future__ import annotations

import json
from datetime import date

import pytest

import noaa_swpc_loader as noaa


# --------------------------------------------------------------------------- #
# offline doubles
# --------------------------------------------------------------------------- #
class _Resp:
    """Minimal stand-in for ``requests.Response`` used by the offline tests."""

    def __init__(self, payload=None, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def _install(monkeypatch, mapping):
    """Patch ``noaa._request`` to answer each URL from *mapping*."""

    def fake(url, *, timeout=None, retries=None, **kwargs):
        value = mapping.get(url)
        if value is None:
            return _Resp(status=404)
        return value() if callable(value) else value

    monkeypatch.setattr(noaa, "_request", fake)


#: A geomagnetic G1 alert, a stronger G3 alert, and a non-geomagnetic product
#: that must be filtered out. Field names and text mirror the live service.
_ALERTS_JSON = [
    {
        "product_id": "A20F",
        "issue_datetime": "2026-10-10 20:00:00.000",
        "message": "ALERT: 20 MeV proton flux exceeded 10 pfu\nValid From: 2026 Oct 10 2000 UTC",
    },
    {
        "product_id": "K05A",
        "issue_datetime": "2026-10-10 21:45:58.723",
        "message": (
            "Space Weather Message Code: ALTK05\n"
            "Serial Number: 1695\n"
            "Issue Time: 2026 Oct 10 2145 UTC\n"
            "\n"
            "ALERT: Geomagnetic K-index of 5\n"
            "Threshold Reached: 2026 Oct 10 2145 UTC\n"
            "Synoptic Period: 1800-2100 UTC\n"
            "\n"
            "Active Warning: Yes\n"
            "NOAA Scale: G1 - Minor\n"
            "Valid To: 2026 Oct 11 0300 UTC\n"
        ),
    },
    {
        "product_id": "K07A",
        "issue_datetime": "2026-10-10 19:27:12.427",
        "message": (
            "Space Weather Message Code: ALTK07\n"
            "Issue Time: 2026 Oct 10 1927 UTC\n"
            "\n"
            "ALERT: Geomagnetic K-index of 7\n"
            "Threshold Reached: 2026 Oct 10 1926 UTC\n"
            "NOAA Scale: G3 - Strong\n"
        ),
    },
]

_KP_FORECAST = [
    {"time_tag": "2026-10-16T00:00:00", "kp": 2.33, "observed": "observed", "noaa_scale": None},
    {"time_tag": "2026-10-16T03:00:00", "kp": 4.0, "observed": "estimated", "noaa_scale": None},
    {"time_tag": "2026-10-17T06:00:00", "kp": 5.0, "observed": "predicted", "noaa_scale": "G1"},
    {"time_tag": "2026-10-17T09:00:00", "kp": 3.0, "observed": "predicted", "noaa_scale": None},
]

_FLARES = [
    {"begin_time": "2026-10-17T06:22:00Z", "max_time": "2026-10-17T06:30:00Z",
     "end_time": "2026-10-17T06:40:00Z", "begin_class": "B6.7", "max_class": "C1.2",
     "satellite": 18},
    {"begin_time": "2026-10-10T13:18:00Z", "max_time": "2026-10-10T13:25:00Z",
     "end_time": "2026-10-10T13:30:00Z", "begin_class": "B9.5", "max_class": "M2.7",
     "satellite": 18},
    {"begin_time": "2026-10-05T01:00:00Z", "max_time": "2026-10-05T01:05:00Z",
     "end_time": "2026-10-05T01:10:00Z", "begin_class": "C1.0", "max_class": "C1.0",
     "satellite": 18},
]


def _install_all(monkeypatch):
    _install(monkeypatch, {
        noaa.ALERTS_URL: _Resp(payload=_ALERTS_JSON),
        noaa.KP_FORECAST_URL: _Resp(payload=_KP_FORECAST),
        noaa.XRAY_FLARES_URL: _Resp(payload=_FLARES),
    })


# --------------------------------------------------------------------------- #
# offline: parsing helpers
# --------------------------------------------------------------------------- #
def test_parse_swpc_time_reads_both_spellings():
    a = noaa._parse_swpc_time(" 2026 Oct 10 2145 UTC")
    assert a is not None and a.year == 2026 and a.month == 10 and a.hour == 21
    assert a.tzinfo is not None
    b = noaa._parse_swpc_time("2026-10-17T06:22:00Z")
    assert b is not None and (b.year, b.month, b.day, b.hour) == (2026, 10, 17, 6)
    c = noaa._parse_swpc_time("2026-10-10 21:45:58.723")
    assert c is not None and c.minute == 45 and c.second == 58
    assert noaa._parse_swpc_time("") is None
    assert noaa._parse_swpc_time(None) is None


@pytest.mark.parametrize(
    "kp,scale",
    [(4.9, None), (5.0, "G1"), (6.0, "G2"), (7.0, "G3"), (8.0, "G4"),
     (9.0, "G5"), (None, None)],
)
def test_kp_to_g_scale(kp, scale):
    assert noaa._kp_to_g_scale(kp) == scale


# --------------------------------------------------------------------------- #
# offline: geomagnetic alerts
# --------------------------------------------------------------------------- #
def test_fetch_geomagnetic_alerts_parses_and_filters(monkeypatch):
    _install_all(monkeypatch)
    res = noaa.fetch_geomagnetic_alerts()

    assert res["ok"] is True and not noaa.is_error(res)
    # The A20F radiation alert is dropped; only the two K-index messages remain.
    assert res["count"] == 2
    by_id = {a["product_id"]: a for a in res["data"]}
    assert set(by_id) == {"K05A", "K07A"}

    k5 = by_id["K05A"]
    assert k5["kind"] == "ALERT" and k5["k_index"] == 5.0
    assert k5["g_scale"] == "G1" and k5["severity"] == "Minor"
    assert k5["issue_time"] == "2026-10-10T21:45:00+00:00"
    assert k5["start_time"] == "2026-10-10T21:45:00+00:00"
    assert k5["end_time"] == "2026-10-11T03:00:00+00:00"
    assert k5["active_warning"] is True

    k7 = by_id["K07A"]
    assert k7["k_index"] == 7.0 and k7["g_scale"] == "G3" and k7["severity"] == "Strong"
    # Newest first.
    assert res["data"][0]["product_id"] == "K05A"


def test_fetch_geomagnetic_alerts_empty_is_ok(monkeypatch):
    _install(monkeypatch, {noaa.ALERTS_URL: _Resp(payload=[])})
    res = noaa.fetch_geomagnetic_alerts()
    assert res["ok"] is True and res["data"] == []
    assert res["message"] == "No alerts found"


def test_parse_alert_ignores_non_geomagnetic():
    assert noaa._parse_alert(
        {"product_id": "A20F", "message": "ALERT: 10 MeV protons"}
    ) is None


# --------------------------------------------------------------------------- #
# offline: Kp forecast
# --------------------------------------------------------------------------- #
def test_fetch_solar_wind_forecast_reports_predicted_peak(monkeypatch):
    _install_all(monkeypatch)
    res = noaa.fetch_solar_wind_forecast()

    assert res["ok"] is True
    assert res["count"] == 4
    assert sum(1 for r in res["data"] if r["status"] == "predicted") == 2
    # The peak is taken over predicted rows (kp 5.0 = G1), not the past estimate.
    assert res["max_kp"] == pytest.approx(5.0)
    assert res["max_scale"] == "G1"
    assert res["max_scale_time"] == "2026-10-17T06:00:00+00:00"
    assert all(set(r) == {"time_tag", "kp", "status", "g_scale"} for r in res["data"])


# --------------------------------------------------------------------------- #
# offline: flare catalogue
# --------------------------------------------------------------------------- #
def test_fetch_solar_events_filters_the_window(monkeypatch):
    _install_all(monkeypatch)
    res = noaa.fetch_solar_events("2026-10-10", "2026-10-17")

    assert res["ok"] is True and res["count"] == 2
    classes = [f["max_class"] for f in res["data"]]
    assert "M2.7" in classes and "C1.2" in classes
    # The 2026-10-05 flare is outside the window.
    assert all(f["begin_time"] >= "2026-10-10" for f in res["data"])
    # Newest first, and the class letter is surfaced for ranking.
    assert res["data"][0]["max_class"] == "C1.2"
    assert {f["class_letter"] for f in res["data"]} == {"C", "M"}


def test_fetch_solar_events_empty_window(monkeypatch):
    _install_all(monkeypatch)
    res = noaa.fetch_solar_events("2026-09-01", "2026-09-02")
    assert res["ok"] is True and res["data"] == []
    assert res["message"] == "No events found"


def test_fetch_solar_events_single_day_defaults_end(monkeypatch):
    _install_all(monkeypatch)
    res = noaa.fetch_solar_events("2026-10-17")
    assert res["ok"] is True and res["count"] == 1
    assert res["data"][0]["max_class"] == "C1.2"


def test_fetch_solar_events_rejects_bad_dates():
    res = noaa.fetch_solar_events("not-a-date", "2026-10-17")
    assert noaa.is_error(res) and res["error"] == "invalid_input"
    backwards = noaa.fetch_solar_events("2026-10-17", "2026-10-10")
    assert noaa.is_error(backwards) and backwards["error"] == "invalid_input"


# --------------------------------------------------------------------------- #
# offline: failure contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fetch", [
    noaa.fetch_geomagnetic_alerts,
    noaa.fetch_solar_wind_forecast,
    lambda: noaa.fetch_solar_events("2026-10-10", "2026-10-17"),
])
def test_api_unavailable_returns_the_stable_error(monkeypatch, fetch):
    def boom(url, **kwargs):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(noaa, "_request", boom)
    res = fetch()
    assert noaa.is_error(res)
    assert res["error"] == "noaa_api_unavailable"
    assert "hint" in res


def test_http_404_is_noaa_api_unavailable(monkeypatch):
    _install(monkeypatch, {noaa.ALERTS_URL: _Resp(status=404)})
    res = noaa.fetch_geomagnetic_alerts()
    assert noaa.is_error(res) and res["error"] == "noaa_api_unavailable"


def test_is_error_contract():
    assert noaa.is_error({"ok": False, "error": "noaa_api_unavailable"})
    assert not noaa.is_error({"ok": True, "data": []})
    assert not noaa.is_error(None)


# --------------------------------------------------------------------------- #
# agent_core integration
# --------------------------------------------------------------------------- #
import agent_core as ac  # noqa: E402  (import after the loader-specific tests)


def test_tool_schema_is_registered():
    schema = next(
        (t for t in ac.TOOL_SCHEMAS
         if t["function"]["name"] == "get_noaa_space_weather_context"),
        None,
    )
    assert schema is not None
    params = schema["function"]["parameters"]
    assert params["required"] == ["query_type"]
    assert params["properties"]["query_type"]["enum"] == [
        "alerts", "forecast", "solar_events"
    ]
    desc = schema["function"]["description"].lower()
    assert "fetch_solar_wind" in desc  # points elsewhere for raw solar wind
    assert "noaa" in desc


def test_handler_and_selection_are_wired():
    assert ac.HANDLERS["get_noaa_space_weather_context"] is ac._handle_noaa_context
    assert "get_noaa_space_weather_context" in ac.TOOL_SELECTION
    assert ac._SCRATCHPAD_STEP_BY_TOOL["get_noaa_space_weather_context"] == "noaa_context"


def test_offline_handler_returns_trimmed_digest(monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    store = ac.FrameStore()

    payload, note = ac._handle_noaa_context({"query_type": "forecast"}, store)
    assert payload["ok"] is True and not noaa.is_error(payload)
    assert payload["query_type"] == "forecast"
    assert payload["count"] >= 1 and payload["data"]
    assert payload["max_scale"] == "G1"
    assert "Kp" in payload["summary"]
    assert note

    alerts, _ = ac._handle_noaa_context({"query_type": "alerts"}, store)
    assert alerts["data"][0]["g_scale"] == "G1"

    flares, _ = ac._handle_noaa_context(
        {"query_type": "solar_events", "start_date": "2026-10-10",
         "end_date": "2026-10-17"}, store
    )
    assert flares["count"] == 2 and "M2.7" in flares["summary"]


def test_handler_rejects_unknown_query_type(monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)
    payload, _ = ac._handle_noaa_context({"query_type": "weather"}, ac.FrameStore())
    assert noaa.is_error(payload) and payload["error"] == "invalid_input"


def test_selector_adds_noaa_for_a_warning_question():
    picked = [t["function"]["name"]
              for t in ac.select_tools_for_query(
                  "какие сейчас предупреждения NOAA о магнитной буре и вспышках?")]
    assert "get_noaa_space_weather_context" in picked
    assert len(picked) <= ac.TOOLS_PER_TASK


def test_large_noaa_payload_keeps_its_summary():
    """A 57-row catalogue must not collapse to a bare ``{"ok": true}``."""
    payload = {
        "ok": True,
        "query_type": "solar_events",
        "source": "NOAA SWPC GOES X-ray flare catalogue (7-day)",
        "count": 57,
        "summary": "Вспышек NOAA: 57 (C/M/X: 12); сильнейшая M4.1 в 2026-10-11T02:00:00+00:00.",
        "data": [{"begin_time": f"2026-10-{d:02d}T00:00:00Z", "max_class": "C1.0"}
                 for d in range(1, 18)],
    }
    compacted = ac._compact_tool_message(payload)
    assert len(compacted) <= ac.FRAME_SUMMARY_MAX_CHARS
    body = json.loads(compacted)
    assert body["count"] == 57
    assert "M4.1" in body["summary"]
    assert body["query_type"] == "solar_events"


def test_prompt_advertises_the_tool_and_stays_in_budget():
    assert "get_noaa_space_weather_context" in ac.SYSTEM_PROMPT
    assert len(ac.SYSTEM_PROMPT) <= 3200


# --------------------------------------------------------------------------- #
# live network: the NOAA SWPC service
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_live_kp_forecast_is_reachable():
    """The 3-day forecast always has rows; its peak lands on the G-scale."""
    res = noaa.fetch_solar_wind_forecast()
    assert not noaa.is_error(res), res
    assert res["count"] > 0
    assert isinstance(res["max_kp"], float) and res["max_kp"] >= 0
    if res["max_kp"] >= 5:
        assert res["max_scale"] in {"G1", "G2", "G3", "G4", "G5"}


@pytest.mark.network
def test_live_alerts_and_flares_are_reachable():
    alerts = noaa.fetch_geomagnetic_alerts()
    assert not noaa.is_error(alerts), alerts
    assert isinstance(alerts["data"], list)  # a quiet day legitimately has none
    for a in alerts["data"]:
        assert a["product_id"].upper().startswith("K")
        assert a["g_scale"] in (None, "G1", "G2", "G3", "G4", "G5")

    flares = noaa.fetch_solar_events()
    assert not noaa.is_error(flares), flares
    assert isinstance(flares["data"], list)
    for f in flares["data"]:
        assert f["class_letter"] in {"A", "B", "C", "M", "X", None}
