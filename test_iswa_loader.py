"""Verification suite for :mod:`iswa_loader` and its agent_core wiring.

Run with::

    python -m pytest test_iswa_loader.py -v

The offline tests never touch the network: they patch ``iswa_loader._request``
with a small stand-in and feed the exact shapes the live endpoints return (a
WMS picture or fault report, a HAPI JSON document). The ``network``-marked
tests hit the live ISWA endpoints for real and double as a health check.

Endpoint note (Phase 6.4 / decommissioning)
--------------------------------------------
The brief named ``https://iswa.gsfc.nasa.gov/wmts/wmts.cgi`` (WMS 1.1.1,
``REQUEST=GetMap``, ``LAYERS=ground_mag_perturbation``, ``FORMAT=image/png``).
That layer is now **decommissioned**: NASA CCMC migrated ISWA to
``iswa.ccmc.gsfc.nasa.gov`` and the public ground-magnetic-perturbation WMS
layer did not come with it. Checked 2026-10-20 against the new system: its
catalog (``/api/info``, 1351 datafeeds) has no perturbation/ground-map feed and
no ``wms`` protocol, and every WMS-shaped path on the new domain returns 403 to
``SERVICE=WMS`` (the WAF refuses the keyword). So
:func:`iswa_loader.get_ground_perturbation_map` answers
``service_decommissioned`` immediately, without any network request, carrying a
``fallback_hint`` that points at ``fetch_supermag_data(data_type='indices')``.
``build_wms_url`` still builds the documented request (pinned below) and the
live WMS client stays reachable via ``decommissioned=False`` for a still-live
service. The live forecast test runs against the ISWA HAPI service at
``iswa.ccmc.gsfc.nasa.gov``, which does answer -- verified with the
``time.min``/``time.max`` spelling this server requires (the HAPI 2.0
``start``/``stop`` pair is rejected with ``status.code 1400``).
"""

from __future__ import annotations

import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest

import iswa_loader as iswa


# --------------------------------------------------------------------------- #
# offline doubles
# --------------------------------------------------------------------------- #
class _Resp:
    """Minimal stand-in for ``requests.Response`` used by the offline tests."""

    def __init__(self, *, content: bytes = b"", payload=None, status: int = 200,
                 ctype: str = ""):
        self.content = content
        self._payload = payload
        self.status_code = status
        self.headers = {"Content-Type": ctype} if ctype else {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def _hapi(rows, *, names, code: int = 1200, message: str = "OK"):
    """A HAPI ``data`` document exactly as iswa.ccmc.gsfc.nasa.gov returns it."""
    return _Resp(
        payload={
            "status": {"code": code, "message": message},
            "parameters": [{"name": name} for name in names],
            "data": rows,
        },
        ctype="application/json",
    )


def _install(monkeypatch, *, kp=None, dst=None, raise_exc=None):
    """Patch ``iswa._request`` to answer each HAPI dataset from *kp* / *dst*.

    ``None`` (the default) answers HTTP 404, which is what an unknown dataset
    gets from the live service; *raise_exc* fails every call like a dead
    network. Both datasets are keyed by the ``id=`` the loader sends, so a
    test can make Kp work while Dst is down (and vice versa).
    """

    def fake(url, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        if f"id={iswa.KP_DATASET}&" in url:
            value = kp
        elif f"id={iswa.DST_DATASET}&" in url:
            value = dst
        else:
            value = None
        if value is None:
            return _Resp(status=404)
        return value() if callable(value) else value

    monkeypatch.setattr(iswa, "_request", fake)


#: The pinned clock: the forecast window becomes 2026-10-19T12:00:00Z ..
#: 2026-10-22T12:00:00Z on a three-hour grid, i.e. exactly 25 rows.
_NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def fixed_now(monkeypatch):
    """Pin the loader's ``_now()`` seam so the window is an exact constant."""
    monkeypatch.setattr(iswa, "_now", lambda: _NOW)
    return _NOW


#: Live HAPI shapes, verified 2026-10-19: Kp rows are arrays in the dataset's
#: own parameter order, Dst rows are arrays of (Time, Dst).
_KP_NAMES = ["Time", "KP_3H", "Kp_observed", "Kp_estimated", "Kp_predicted",
             "StartTime", "MidTime", "EndTime"]
_DST_NAMES = ["Time", "Dst"]

_KP_ROWS = [
    # observed wins over KP_3H and over the model columns beside it
    ["2026-10-19T12:00:00Z", 3.0, 3.0, None, None, "2026-10-19T12:00:00Z", "", ""],
    # no observation: the estimated column carries the value (4.33, not KP_3H 4.0)
    ["2026-10-19T15:00:00Z", 4.0, None, 4.33, None, "", "", ""],
    # observed 6.0 must beat Kp_predicted 6.67 in the same row
    ["2026-10-19T18:00:00Z", 6.0, 6.0, None, 6.67, "", "", ""],
    # a future bucket: observed/estimated null, only the prediction is filled
    ["2026-10-21T12:00:00Z", 5.0, None, None, 5.0, "", "", ""],
    # KP_3H's -1 is a fill value, not a Kp of -1
    ["2026-10-21T15:00:00Z", -1, None, None, None, "", "", ""],
]

_DST_ROWS = [
    ["2026-10-19T12:00:00Z", -45.0],
    ["2026-10-19T15:00:00Z", -60.0],
    ["2026-10-19T18:00:00Z", -72.0],
    ["2026-10-20T00:00:00Z", -66.0],
    ["2026-10-20T12:00:00Z", -30.0],  # the last quicklook observation: now
]


# --------------------------------------------------------------------------- #
# offline: the documented WMS request
# --------------------------------------------------------------------------- #
def test_build_wms_url_is_the_documented_getmap_request():
    """The exact endpoint and parameter spellings the Phase 6.4 brief names.

    ``SRS=EPSG:4326`` is a deliberate addition: WMS 1.1.1 makes SRS mandatory,
    so a compliant server rejects the request without it. Commas and colons
    stay literal (``urlencode(..., safe=',:')``), which is why TIME, BBOX and
    SRS read the way the OGC spec writes them.
    """
    url = iswa.build_wms_url("2024-09-12T12:00:00Z")
    endpoint, _, query = url.partition("?")
    assert endpoint == "https://iswa.gsfc.nasa.gov/wmts/wmts.cgi"
    params = dict(pair.split("=", 1) for pair in query.split("&"))
    assert params == {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": "ground_mag_perturbation",
        "TIME": "2024-09-12T12:00:00Z",
        "BBOX": "-180,-90,180,90",
        "WIDTH": "1024",
        "HEIGHT": "512",
        "FORMAT": "image/png",
        "SRS": "EPSG:4326",
    }
    assert "%3A" not in url and "%2C" not in url  # TIME/BBOX/SRS stay literal


def test_build_wms_url_normalises_time_bbox_and_size():
    url = iswa.build_wms_url(
        "2024-09-12 12:00", bbox=(0, -60, 90, 60), width=640, height=320
    )
    assert "TIME=2024-09-12T12:00:00Z" in url
    assert "BBOX=0,-60,90,60" in url
    assert "WIDTH=640" in url and "HEIGHT=320" in url
    assert url.startswith(iswa.WMS_ENDPOINT + "?")
    offset = iswa.build_wms_url("2024-09-12T14:00:00+02:00")
    assert "TIME=2024-09-12T12:00:00Z" in offset  # converted to UTC


@pytest.mark.parametrize("stamp", ["", " ", "yesterday", "2024-13-40T00:00:00Z",
                                   None, 5])
def test_build_wms_url_rejects_a_bad_timestamp(stamp):
    with pytest.raises(ValueError):
        iswa.build_wms_url(stamp)


@pytest.mark.parametrize("bbox", [
    (0, 0, 999, 90),
    (-180, -90, 180),
    (10, 0, -10, 90),
    "north,south,east,west",
])
def test_build_wms_url_rejects_a_bad_bbox(bbox):
    with pytest.raises(ValueError):
        iswa.build_wms_url("2024-09-12T12:00:00Z", bbox=bbox)


@pytest.mark.parametrize("size", [0, -1, 99999])
def test_build_wms_url_rejects_impossible_pixel_sizes(size):
    with pytest.raises(ValueError):
        iswa.build_wms_url("2024-09-12T12:00:00Z", width=size)


# --------------------------------------------------------------------------- #
# offline: the map layer is decommissioned (the live default), and the WMS
# client behind it still classifies what a still-live service would answer
# --------------------------------------------------------------------------- #
def test_map_default_returns_service_decommissioned_without_network(monkeypatch):
    """The documented, honest answer for the retired layer -- and no request."""

    def forbidden(url, **kwargs):
        raise AssertionError("the decommissioned layer must not be probed")

    monkeypatch.setattr(iswa, "_request", forbidden)
    res = iswa.get_ground_perturbation_map("2024-09-12T12:00:00Z")
    assert iswa.is_error(res)
    assert res["error"] == "service_decommissioned"
    assert "decommissioned" in res["message"]
    assert "ground magnetic perturbation" in res["message"]
    # scenario 1: the global picture goes through SuperMAG indices + plot_components
    assert "fetch_supermag_data(data_type='indices')" in res["fallback_hint"]
    assert "SME/SML/SMU" in res["fallback_hint"]
    assert "plot_components" in res["fallback_hint"]
    # scenario 2: a station comparison MUST start from fetch_observatory_data
    assert "fetch_observatory_data" in res["fallback_hint"]
    assert "plot_space_weather_comparison" in res["fallback_hint"]
    assert "CANNOT be used as ground data" in res["fallback_hint"]


def test_map_decommissioned_still_validates_its_arguments(monkeypatch):
    """A malformed call is invalid_input, not a decommission notice (no request)."""

    def forbidden(url, **kwargs):
        raise AssertionError("no request may be sent")

    monkeypatch.setattr(iswa, "_request", forbidden)
    cases = [
        {"timestamp": "вчера"},
        {"timestamp": "2024-09-12T12:00:00Z", "bbox": (0, 0, 999, 90)},
        {"timestamp": "2024-09-12T12:00:00Z", "width": 0},
        {"timestamp": None},
    ]
    for args in cases:
        res = iswa.get_ground_perturbation_map(**args)
        assert iswa.is_error(res) and res["error"] == "invalid_input", args
        assert res["hint"]


def test_live_png_is_saved_with_the_documented_payload(monkeypatch, tmp_path):
    """A still-live service (``decommissioned=False``) still saves a real PNG."""
    urls = []

    def fake(url, **kwargs):
        urls.append(url)
        return _Resp(content=iswa.synthetic_map_png(32, 16), ctype="image/png")

    monkeypatch.setattr(iswa, "_request", fake)
    res = iswa.get_ground_perturbation_map(
        "2024-09-12T12:00:00Z", output_dir=tmp_path, decommissioned=False
    )

    assert res["ok"] is True
    assert set(res) == {"ok", "map_path", "metadata"}
    meta = res["metadata"]
    assert meta["timestamp"] == "2024-09-12T12:00:00Z"
    assert meta["bbox"] == [-180.0, -90.0, 180.0, 90.0]
    assert meta["width"] == 1024 and meta["height"] == 512
    assert meta["layer"] == "ground_mag_perturbation"
    path = Path(res["map_path"])
    assert path.exists()
    assert path.read_bytes().startswith(b"\x89PNG")
    assert ":" not in path.name  # Windows-safe: the timestamp's colons are dashes
    assert len(urls) == 1
    assert "TIME=2024-09-12T12:00:00Z" in urls[0]
    assert "LAYERS=ground_mag_perturbation" in urls[0]


@pytest.mark.parametrize("status", [401, 500])
def test_map_live_http_error_is_iswa_unavailable(monkeypatch, status):
    monkeypatch.setattr(iswa, "_request", lambda url, **kw: _Resp(status=status))
    res = iswa.get_ground_perturbation_map(
        "2024-09-12T12:00:00Z", decommissioned=False
    )
    assert iswa.is_error(res) and res["error"] == "iswa_unavailable"
    assert f"HTTP {status}" in res["detail"]
    assert res["hint"]


@pytest.mark.parametrize("status", [403, 404])
def test_map_live_403_404_resolve_to_service_decommissioned(monkeypatch, status):
    """The WAF/dead-CGI spelling of the same truth: the layer is gone."""
    monkeypatch.setattr(iswa, "_request", lambda url, **kw: _Resp(status=status))
    res = iswa.get_ground_perturbation_map(
        "2024-09-12T12:00:00Z", decommissioned=False
    )
    assert iswa.is_error(res) and res["error"] == "service_decommissioned"
    assert f"HTTP {status}" in res["message"]
    assert "SME/SML/SMU" in res["fallback_hint"]


def test_map_dead_network_is_iswa_unavailable(monkeypatch):
    def boom(url, **kwargs):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(iswa, "_request", boom)
    res = iswa.get_ground_perturbation_map(
        "2024-09-12T12:00:00Z", decommissioned=False
    )
    assert iswa.is_error(res) and res["error"] == "iswa_unavailable"
    assert "no route to host" in res["detail"]


def test_map_html_error_page_is_iswa_unavailable(monkeypatch):
    """A 200 carrying NASA's HTML error page is not a map."""
    html = b"<!DOCTYPE html><html><body><h1>403 Forbidden</h1></body></html>"
    monkeypatch.setattr(iswa, "_request",
                        lambda url, **kw: _Resp(content=html, ctype="text/html"))
    res = iswa.get_ground_perturbation_map(
        "2024-09-12T12:00:00Z", decommissioned=False
    )
    assert iswa.is_error(res) and res["error"] == "iswa_unavailable"
    assert "HTML" in res["detail"]


def test_map_service_exception_is_no_data(monkeypatch):
    """A working WMS answers a TIME it cannot cover with a fault report."""
    fault = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<ServiceExceptionReport version="1.1.1">\n'
        b'  <ServiceException code="NoData">'
        b"No image available for time 2024-09-12T12:00:00Z"
        b"</ServiceException>\n"
        b"</ServiceExceptionReport>"
    )
    monkeypatch.setattr(iswa, "_request",
                        lambda url, **kw: _Resp(content=fault, ctype="text/xml"))
    res = iswa.get_ground_perturbation_map(
        "2024-09-12T12:00:00Z", decommissioned=False
    )
    assert iswa.is_error(res) and res["error"] == "no_data"
    assert "No image available" in res["detail"]
    assert res["timestamp"] == "2024-09-12T12:00:00Z"


def test_map_empty_body_is_no_data(monkeypatch):
    monkeypatch.setattr(iswa, "_request", lambda url, **kw: _Resp(content=b""))
    res = iswa.get_ground_perturbation_map(
        "2024-09-12T12:00:00Z", decommissioned=False
    )
    assert iswa.is_error(res) and res["error"] == "no_data"


# --------------------------------------------------------------------------- #
# offline: helpers
# --------------------------------------------------------------------------- #
def test_parse_timestamp_accepts_the_spellings_a_model_types():
    for stamp, want in [
        ("2024-09-12T12:00:00Z", "2024-09-12T12:00:00Z"),
        ("2024-09-12 12:00", "2024-09-12T12:00:00Z"),
        ("2024-09-12T12", "2024-09-12T12:00:00Z"),
        ("2024-09-12", "2024-09-12T00:00:00Z"),
        ("2024-09-12T14:00:00+02:00", "2024-09-12T12:00:00Z"),
    ]:
        assert iswa.iso_z(iswa.parse_timestamp(stamp)) == want, stamp
    for bad in ("", "   ", "yesterday", "2024-99-99", None, 42):
        with pytest.raises(ValueError):
            iswa.parse_timestamp(bad)


def test_kp_column_preference_observed_beats_every_model_value():
    rows = [
        {"Time": "2026-10-20T03:00:00Z", "KP_3H": 6.0, "Kp_observed": 2.0,
         "Kp_estimated": 5.0, "Kp_predicted": 4.0},
        {"Time": "2026-10-20T06:00:00Z", "KP_3H": 4.0, "Kp_observed": None,
         "Kp_estimated": None, "Kp_predicted": 3.33},
        {"Time": "2026-10-20T09:00:00Z", "KP_3H": 5.0, "Kp_observed": None,
         "Kp_estimated": 4.66, "Kp_predicted": None},
        {"Time": "2026-10-20T12:00:00Z", "KP_3H": 7.0, "Kp_observed": None,
         "Kp_estimated": None, "Kp_predicted": None},
        {"Time": "2026-10-20T15:00:00Z", "KP_3H": -1, "Kp_observed": None,
         "Kp_estimated": None, "Kp_predicted": None},
    ]
    values, kinds = iswa._kp_index(rows)
    assert values["2026-10-20T03:00:00Z"] == 2.0
    assert kinds["2026-10-20T03:00:00Z"] == "observed"
    assert values["2026-10-20T06:00:00Z"] == 3.33
    assert kinds["2026-10-20T06:00:00Z"] == "predicted"
    assert values["2026-10-20T09:00:00Z"] == 4.66
    assert kinds["2026-10-20T09:00:00Z"] == "estimated"
    assert values["2026-10-20T12:00:00Z"] == 7.0
    assert kinds["2026-10-20T12:00:00Z"] == "reported"
    assert "2026-10-20T15:00:00Z" not in values  # -1 is a fill, not a Kp


def test_dst_index_keeps_numbers_and_drops_missing_values():
    rows = [
        {"Time": "2026-10-20T00:00:00Z", "Dst": -42.0},
        {"Time": "2026-10-20T01:00:00Z", "Dst": None},
        {"Time": "2026-10-20T02:00:00Z", "Dst": "nan"},
        {"Time": "2026-10-20T03:00:00Z"},
        {"nonsense": True},
    ]
    assert iswa._dst_index(rows) == {"2026-10-20T00:00:00Z": -42.0}


def test_grid_is_three_hourly_and_inclusive_at_both_ends():
    start = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    stop = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)
    stamps = iswa._grid(start, stop, 3)
    assert stamps[0] == start and stamps[-1] == stop
    assert len(stamps) == 25
    gaps = {(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])}
    assert gaps == {3 * 3600}


def test_synthetic_map_png_is_a_well_formed_png():
    png = iswa.synthetic_map_png(40, 20)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])  # IHDR follows the signature
    assert (width, height) == (40, 20)
    assert png[-8:-4] == b"IEND"


def test_save_map_png_writes_the_file_and_sanitises_the_name(tmp_path):
    png = iswa.synthetic_map_png(8, 8)
    payload = iswa.save_map_png(png, "2024-09-12T12:00:00Z", output_dir=tmp_path)
    name = "ground_perturbation_2024-09-12T12-00-00Z.png"
    assert payload["map_path"] == (tmp_path / name).as_posix()
    assert Path(payload["map_path"]).read_bytes() == png
    with pytest.raises(ValueError):
        iswa.save_map_png(b"<html>not a png</html>", "2024-09-12T12:00:00Z",
                          output_dir=tmp_path)


def test_is_error_contract():
    assert iswa.is_error({"ok": False, "error": "iswa_unavailable"})
    assert not iswa.is_error({"ok": True, "map_path": "plots/x.png"})
    assert not iswa.is_error(None)


# --------------------------------------------------------------------------- #
# offline: Dst/Kp forecast
# --------------------------------------------------------------------------- #
def test_forecast_merges_kp_predictions_with_the_dst_quicklook(monkeypatch,
                                                               fixed_now):
    _install(monkeypatch, kp=_hapi(_KP_ROWS, names=_KP_NAMES),
             dst=_hapi(_DST_ROWS, names=_DST_NAMES))
    res = iswa.get_dst_kp_forecast()

    assert res["ok"] is True
    assert res["count"] == len(res["forecast"]) == 25
    by = {row["timestamp"]: row for row in res["forecast"]}

    assert by["2026-10-19T12:00:00Z"] == {
        "timestamp": "2026-10-19T12:00:00Z", "dst": -45.0,
        "kp": 3.0, "kp_kind": "observed",
    }
    assert by["2026-10-19T15:00:00Z"]["kp"] == 4.33
    assert by["2026-10-19T15:00:00Z"]["kp_kind"] == "estimated"
    assert by["2026-10-19T18:00:00Z"]["kp"] == 6.0  # observed beats predicted 6.67
    assert by["2026-10-19T18:00:00Z"]["kp_kind"] == "observed"

    future = by["2026-10-21T12:00:00Z"]
    assert future["kp"] == 5.0 and future["kp_kind"] == "predicted"
    assert future["dst"] is None  # no Dst forecast exists anywhere in ISWA
    assert by["2026-10-21T15:00:00Z"]["kp"] is None  # KP_3H fill, nothing else
    assert by["2026-10-21T15:00:00Z"]["kp_kind"] is None

    assert by["2026-10-20T12:00:00Z"]["dst"] == -30.0
    assert by["2026-10-20T15:00:00Z"]["dst"] is None  # after the last observation

    assert res["window"] == {
        "start": "2026-10-19T12:00:00Z",
        "stop": "2026-10-22T12:00:00Z",
        "step_hours": 3,
        "now": "2026-10-20T12:00:00Z",
    }
    assert res["source"].startswith("ISWA HAPI")
    assert "NOAA_KP_P3H" in res["sources"]["kp"]
    assert "dst_quicklook" in res["sources"]["dst"]
    assert "no Dst forecast" in res["sources"]["dst"]
    assert "warnings" not in res


def test_hapi_requests_use_time_min_time_max_not_start_stop(monkeypatch,
                                                            fixed_now):
    """This server answers code 1400 to the HAPI 2.0 start/stop pair."""
    seen = []

    def fake(url, **kwargs):
        seen.append(url)
        return _hapi([], names=_KP_NAMES)

    monkeypatch.setattr(iswa, "_request", fake)
    iswa.get_dst_kp_forecast()

    assert len(seen) == 2
    assert f"id={iswa.KP_DATASET}" in seen[0]
    assert f"id={iswa.DST_DATASET}" in seen[1]
    for url in seen:
        assert "time.min=" in url and "time.max=" in url
        assert "format=json" in url
        assert "&start=" not in url and "&stop=" not in url
    assert "time.min=2026-10-19T12:00:00Z" in seen[0]
    assert "time.max=2026-10-22T12:00:00Z" in seen[0]


def test_forecast_dead_network_is_iswa_forecast_unavailable(monkeypatch,
                                                            fixed_now):
    _install(monkeypatch, raise_exc=ConnectionError("no route to host"))
    res = iswa.get_dst_kp_forecast()
    assert iswa.is_error(res) and res["error"] == "iswa_forecast_unavailable"
    assert "no route to host" in res["detail"]
    assert res["hint"]


def test_forecast_http_error_is_iswa_forecast_unavailable(monkeypatch, fixed_now):
    _install(monkeypatch, kp=_Resp(status=403))
    res = iswa.get_dst_kp_forecast()
    assert iswa.is_error(res) and res["error"] == "iswa_forecast_unavailable"
    assert "403" in res["detail"]


def test_hapi_application_error_is_iswa_forecast_unavailable(monkeypatch,
                                                             fixed_now):
    """HAPI reports bad requests inside a 200 body -- the code must be 1200."""
    _install(monkeypatch, kp=_hapi([], names=[], code=1400,
                                   message="'time.min' parameter is required"))
    res = iswa.get_dst_kp_forecast()
    assert iswa.is_error(res) and res["error"] == "iswa_forecast_unavailable"
    assert "1400" in res["detail"]


def test_two_empty_datasets_are_no_data(monkeypatch, fixed_now):
    _install(monkeypatch, kp=_hapi([], names=_KP_NAMES),
             dst=_hapi([], names=_DST_NAMES))
    res = iswa.get_dst_kp_forecast()
    assert iswa.is_error(res) and res["error"] == "no_data"
    assert res["hint"]


def test_dst_outage_keeps_the_kp_outlook_and_says_so(monkeypatch, fixed_now):
    """Dst is context, not the forecast: its loss must not fail the tool."""
    _install(monkeypatch, kp=_hapi(_KP_ROWS, names=_KP_NAMES),
             dst=_Resp(status=404))
    res = iswa.get_dst_kp_forecast()

    assert res["ok"] is True
    assert res["count"] == 25
    assert all(row["dst"] is None for row in res["forecast"])
    assert any(row["kp"] is not None for row in res["forecast"])
    assert res["warnings"] and "dst unavailable" in res["warnings"][0]


def test_kp_outage_alone_is_fatal(monkeypatch, fixed_now):
    """Without Kp there is no outlook to return -- unlike the Dst half."""
    _install(monkeypatch, kp=_Resp(status=500),
             dst=_hapi(_DST_ROWS, names=_DST_NAMES))
    res = iswa.get_dst_kp_forecast()
    assert iswa.is_error(res) and res["error"] == "iswa_forecast_unavailable"


def test_kp_empty_but_dst_present_is_still_a_result(monkeypatch, fixed_now):
    _install(monkeypatch, kp=_hapi([], names=_KP_NAMES),
             dst=_hapi(_DST_ROWS, names=_DST_NAMES))
    res = iswa.get_dst_kp_forecast()
    assert res["ok"] is True and res["count"] == 25
    assert all(row["kp"] is None for row in res["forecast"])
    assert any(row["dst"] is not None for row in res["forecast"])


# --------------------------------------------------------------------------- #
# agent_core integration
# --------------------------------------------------------------------------- #
import agent_core as ac  # noqa: E402  (import after the loader-specific tests)


def test_tool_schemas_are_registered():
    by_name = {t["function"]["name"]: t for t in ac.TOOL_SCHEMAS}
    assert {"get_iswa_perturbation_map", "get_iswa_forecast"} <= set(by_name)

    params = by_name["get_iswa_perturbation_map"]["function"]["parameters"]
    assert params["required"] == ["timestamp"]
    assert set(params["properties"]) == {"timestamp", "bbox", "width", "height"}
    assert params["properties"]["timestamp"]["type"] == "string"
    desc = by_name["get_iswa_perturbation_map"]["function"]["description"].lower()
    assert "iswa" in desc and "ground_mag_perturbation" in desc

    forecast = by_name["get_iswa_forecast"]["function"]
    assert forecast["parameters"]["properties"] == {}
    assert not forecast["parameters"].get("required")
    fdesc = forecast["description"]
    assert "dst" in fdesc.lower() and "kp" in fdesc.lower()
    # the contract that keeps a prediction from being quoted as an observation
    assert "kp_kind" in fdesc


def test_handler_and_selection_are_wired():
    assert ac.HANDLERS["get_iswa_perturbation_map"] is ac._handle_iswa_perturbation_map
    assert ac.HANDLERS["get_iswa_forecast"] is ac._handle_iswa_forecast
    assert ac.TOOL_SELECTION["get_iswa_perturbation_map"] == ["get_iswa_perturbation_map"]
    assert ac.TOOL_SELECTION["get_iswa_forecast"] == ["get_iswa_forecast"]
    assert ac._SCRATCHPAD_STEP_BY_TOOL["get_iswa_perturbation_map"] == "iswa_map"
    assert ac._SCRATCHPAD_STEP_BY_TOOL["get_iswa_forecast"] == "iswa_forecast"


def test_selector_offers_the_map_for_a_perturbation_question():
    picked = [t["function"]["name"] for t in ac.select_tools_for_query(
        "покажи карту геомагнитных возмущений на 2024-09-12T12:00:00Z")]
    assert "get_iswa_perturbation_map" in picked
    assert len(picked) <= ac.TOOLS_PER_TASK


def test_selector_offers_the_forecast_for_an_outlook_question():
    picked = [t["function"]["name"] for t in ac.select_tools_for_query(
        "прогноз Dst и Kp из ISWA на 48 часов")]
    assert "get_iswa_forecast" in picked
    assert len(picked) <= ac.TOOLS_PER_TASK


def test_offline_map_handler_reports_the_decommission(monkeypatch, tmp_path):
    """Offline mirrors the live truth: the layer is gone, no PNG is written."""
    monkeypatch.setattr(ac, "OFFLINE", True)
    monkeypatch.setattr(iswa, "DEFAULT_OUTPUT_DIR", tmp_path)

    payload, note = ac._handle_iswa_perturbation_map(
        {"timestamp": "2024-09-12T12:00:00Z"}, ac.FrameStore()
    )
    assert iswa.is_error(payload)
    assert payload["error"] == "service_decommissioned"
    assert "fetch_supermag_data(data_type='indices')" in payload["fallback_hint"]
    assert list(tmp_path.iterdir()) == []  # nothing was written
    assert "service_decommissioned" in note
    assert "2024-09-12T12:00:00Z" in note


def test_offline_map_handler_passes_arguments_through_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "OFFLINE", True)
    monkeypatch.setattr(iswa, "DEFAULT_OUTPUT_DIR", tmp_path)

    payload, _ = ac._handle_iswa_perturbation_map(
        {"timestamp": "2024-09-12T12:00:00Z", "bbox": [0, -60, 90, 60],
         "width": 64, "height": 32},
        ac.FrameStore(),
    )
    # valid arguments still reach the decommission notice (no map is produced)
    assert payload["error"] == "service_decommissioned"
    assert list(tmp_path.iterdir()) == []


def test_offline_map_handler_rejects_a_missing_timestamp(monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "OFFLINE", True)
    monkeypatch.setattr(iswa, "DEFAULT_OUTPUT_DIR", tmp_path)

    payload, note = ac._handle_iswa_perturbation_map({}, ac.FrameStore())
    # the strict-argument gate names the missing argument before the loader runs
    assert iswa.is_error(payload) and payload["error"] == "missing_required_argument"
    assert payload["missing"] == ["timestamp"]
    assert list(tmp_path.iterdir()) == []  # nothing was written
    assert note


def test_offline_map_handler_rejects_a_bad_timestamp(monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "OFFLINE", True)
    monkeypatch.setattr(iswa, "DEFAULT_OUTPUT_DIR", tmp_path)

    payload, _ = ac._handle_iswa_perturbation_map({"timestamp": "вчера"},
                                                  ac.FrameStore())
    assert iswa.is_error(payload) and payload["error"] == "invalid_input"
    assert list(tmp_path.iterdir()) == []


def test_offline_forecast_handler_returns_the_outlook(monkeypatch):
    monkeypatch.setattr(ac, "OFFLINE", True)

    payload, note = ac._handle_iswa_forecast({}, ac.FrameStore())
    assert payload["ok"] is True
    assert payload["count"] == 25
    rows = payload["forecast"]
    assert len(rows) == ac._ISWA_MAX_ROWS
    # the kept slice is the outlook: nothing behind window.now survives
    assert rows[0]["timestamp"] >= payload["window"]["now"]
    assert all(r["timestamp"] >= payload["window"]["now"] for r in rows)
    # rows strictly after now are the forecast half
    ahead = [r for r in rows if r["timestamp"] > payload["window"]["now"]]
    assert ahead and all(r["kp_kind"] == "predicted" for r in ahead)
    # and the forecast half carries no Dst: ISWA publishes no Dst forecast
    assert all(r["dst"] is None for r in ahead)
    for row in rows:
        assert set(row) == {"timestamp", "dst", "kp", "kp_kind"}
        assert row["kp_kind"] in ("observed", "predicted")
        assert row["kp"] is not None and 0 <= row["kp"] <= 9
    assert "Kp пик 5.7 (наблюдение)" in payload["summary"]
    assert "-66" in payload["summary"]
    assert "offline" in payload["source"]
    assert note


# --------------------------------------------------------------------------- #
# the 400-character result cap must keep what the model quotes back
# --------------------------------------------------------------------------- #
def test_map_payload_survives_compaction_with_its_path():
    """``map_path`` is the deliverable: the cap must not eat it."""
    payload = {
        "ok": True,
        "map_path": "plots/ground_perturbation_2024-09-12T12-00-00Z.png",
        "metadata": {
            "timestamp": "2024-09-12T12:00:00Z",
            "bbox": [-180.0, -90.0, 180.0, 90.0],
            "width": 1024,
            "height": 512,
            "layer": "ground_mag_perturbation",
        },
        "source": "ISWA WMS (ground_mag_perturbation)",
    }
    compacted = ac._compact_tool_message(payload)
    assert len(compacted) <= ac.FRAME_SUMMARY_MAX_CHARS
    body = json.loads(compacted)
    assert body["map_path"] == payload["map_path"]
    assert body["ok"] is True
    # a production-sized payload fits whole, so metadata rides along untouched
    assert body["metadata"]["timestamp"] == "2024-09-12T12:00:00Z"


def test_large_forecast_payload_keeps_count_and_summary():
    """25 rows are far over the cap: the digest must survive intact."""
    rows = [
        {"timestamp": f"2026-10-{19 + i // 8:02d}T{(i * 3) % 24:02d}:00:00Z",
         "dst": None, "kp": 4.33, "kp_kind": "predicted"}
        for i in range(25)
    ]
    payload = {
        "ok": True,
        "count": 25,
        "summary": (
            "Прогноз ISWA: строк 25, до 2026-10-22T12:00:00Z; "
            "Kp пик 5.7 (наблюдение); Dst минимум -66.0 nT "
            "(наблюдения; вперёд не прогнозируется)."
        ),
        "source": "ISWA HAPI (NOAA_KP_P3H Kp + dst_quicklook Dst)",
        "forecast": rows,
        "window": {"start": "2026-10-19T12:00:00Z", "stop": "2026-10-22T12:00:00Z",
                   "step_hours": 3, "now": "2026-10-20T12:00:00Z"},
    }
    assert len(json.dumps(payload, ensure_ascii=False)) > ac.FRAME_SUMMARY_MAX_CHARS
    compacted = ac._compact_tool_message(payload)
    assert len(compacted) <= ac.FRAME_SUMMARY_MAX_CHARS
    body = json.loads(compacted)
    assert body["count"] == 25
    assert body["ok"] is True
    assert "Kp пик 5.7" in body["summary"]
    assert body["source"].startswith("ISWA HAPI")


def test_prompt_advertises_both_iswa_tools_and_stays_in_budget():
    assert "get_iswa_perturbation_map" in ac.SYSTEM_PROMPT
    assert "get_iswa_forecast" in ac.SYSTEM_PROMPT
    assert "4e." in ac.SYSTEM_PROMPT
    assert len(ac.SYSTEM_PROMPT) <= 3200


def test_prompt_marks_the_map_decommissioned_and_points_at_supermag():
    """The prompt must tell the model the map is gone and what to use instead."""
    sp = ac.SYSTEM_PROMPT
    assert "get_iswa_perturbation_map" in sp
    assert "ДЕКОМИССИРОВАНА" in sp
    # the fallback the agent should reach for
    assert "fetch_supermag_data" in sp
    assert "'indices'" in sp


def test_prompt_forbids_indices_as_ground_data_for_space_weather_plot():
    """Rule 4f: the loop the field showed must be closed in the prompt.

    The model had been passing ``idx:supermag:...`` as ``ground_df_descriptor``
    to ``plot_space_weather_comparison`` after the map error, and looping on
    ``unknown_component`` (no 'H' column). Rule 4f states the station-handle
    requirement and names ``plot_components`` for indices.
    """
    sp = ac.SYSTEM_PROMPT
    assert "4f." in sp
    assert "plot_space_weather_comparison" in sp
    assert "ground_df_descriptor" in sp
    assert "idx:supermag" in sp and "idx:kyoto" in sp
    assert "raw:irt:2024-09-12" in sp
    assert "plot_components" in sp


def test_total_prompt_budget_including_the_iswa_schemas():
    budget = ac.prompt_budget(ac.SYSTEM_PROMPT, ac.TOOL_SCHEMAS, ac.N_CTX)
    assert budget["est_tokens"] <= ac.N_CTX // 2, budget


def test_decommissioned_payload_survives_compaction_with_its_fallback_hint():
    """``fallback_hint`` is the whole point: the cap must not eat it.

    The hint names two different fixes and is longer than the ordinary 400-char
    result cap, so it rides the roomier ``FALLBACK_HINT_MAX_CHARS`` budget and
    must arrive whole -- a half-sentence here is exactly what made the model
    loop on the wrong tool.
    """
    payload = iswa._decommissioned()
    assert len(ac._dumps(payload)) <= ac.FALLBACK_HINT_MAX_CHARS
    compacted = ac._compact_tool_message(payload)
    body = json.loads(compacted)  # valid JSON, not a mid-string cut
    assert body["error"] == "service_decommissioned"
    assert body["fallback_hint"] == payload["fallback_hint"]  # untruncated
    assert "fetch_supermag_data(data_type='indices')" in body["fallback_hint"]
    assert "CANNOT be used as ground data" in body["fallback_hint"]


# --------------------------------------------------------------------------- #
# live network: the ISWA endpoints
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_live_ground_perturbation_map_reports_the_decommission():
    """The map layer is retired for real, so the live tool says so, no request."""
    res = iswa.get_ground_perturbation_map("2024-09-12T12:00:00Z")
    assert iswa.is_error(res)
    assert res["error"] == "service_decommissioned"
    assert "fetch_supermag_data(data_type='indices')" in res["fallback_hint"]


@pytest.mark.network
def test_live_wms_client_is_still_wired(monkeypatch):
    """``decommissioned=False`` still sends the documented request to the network.

    The endpoint is dead, so a real call cannot return a picture; the point here
    is that the live client path is intact -- it reaches the network and comes
    back with an honest failure (unavailable or decommissioned), never a crash.
    """
    res = iswa.get_ground_perturbation_map(
        "2024-09-12T12:00:00Z", decommissioned=False
    )
    assert iswa.is_error(res)
    assert res["error"] in ("iswa_unavailable", "service_decommissioned", "no_data")


@pytest.mark.network
def test_live_dst_kp_forecast_is_reachable():
    res = iswa.get_dst_kp_forecast()
    assert not iswa.is_error(res), res
    rows = res["forecast"]
    assert res["count"] == len(rows) > 0

    stamps = [row["timestamp"] for row in rows]
    assert stamps == sorted(stamps)
    gaps = {
        (iswa.parse_timestamp(b) - iswa.parse_timestamp(a)).total_seconds()
        for a, b in zip(stamps, stamps[1:])
    }
    assert gaps == {3 * 3600}  # one three-hour grid, no holes

    kps = [row["kp"] for row in rows if row["kp"] is not None]
    assert kps and all(0 <= kp <= 9 for kp in kps)
    kinds = {row["kp_kind"] for row in rows if row["kp"] is not None}
    assert kinds <= {"observed", "predicted", "estimated", "reported"}

    dsts = [row["dst"] for row in rows if row["dst"] is not None]
    assert dsts and all(-500 <= dst <= 100 for dst in dsts)

    # the point of the tool: Kp values *after* now, i.e. an actual forecast
    now = res["window"]["now"]
    assert any(row["timestamp"] >= now and row["kp"] is not None for row in rows)
    assert "NOAA_KP_P3H" in res["sources"]["kp"]
    assert "dst_quicklook" in res["sources"]["dst"]
