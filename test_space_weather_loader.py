"""Verification suite for :mod:`space_weather_loader`.

Run with::

    python -m pytest test_space_weather_loader.py -v

Network-dependent tests are marked ``network``; skip them with
``-m "not network"``. Those tests hit the live NASA CDAWeb and Kyoto WDC HAPI
endpoints, so they also serve as a periodic health check of the two services.

Physical-validation note
------------------------
The brief for Phase 6.1 assumed **2024-09-10** was a storm day. The live indices
say otherwise: on 2024-09-10 the Dst index ran 0..+20 nT and Kp peaked at 2.3 --
a *quiet* day. The geomagnetic storm in that window actually peaked on
**2024-09-12** (Dst min -121 nT, Kp 7, Bz min -26 nT). The network tests below
therefore prove the southward-Bz / negative-Dst physics on 2024-09-12, and
assert only what is genuinely true of 2024-09-10 (the IMF still turns southward
there, just weakly).
"""

from __future__ import annotations

import pandas as pd
import pytest

import space_weather_loader as sw


# --------------------------------------------------------------------------- #
# offline: HAPI CSV decoding
# --------------------------------------------------------------------------- #
class _Resp:
    """Minimal stand-in for ``requests.Response`` used by the offline tests."""

    def __init__(self, text: str = "", status: int = 200, payload=None):
        self.text = text
        self.status_code = status
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def test_parse_hapi_csv_maps_names_and_fills():
    names = ["Time", "flow_speed", "proton_density"]
    fills = {"flow_speed": "99999.9", "proton_density": "999.99"}
    text = (
        "2024-09-12T00:00:00.000Z,99999.9,999.99\n"
        "2024-09-12T00:01:00.000Z,450.5,5.25\n"
    )
    df = sw._parse_hapi_csv(text, names, fills)
    assert list(df.columns) == ["flow_speed", "proton_density", "timestamp"]
    assert pd.isna(df.loc[0, "flow_speed"]) and pd.isna(df.loc[0, "proton_density"])
    assert df.loc[1, "flow_speed"] == pytest.approx(450.5)
    assert df.loc[1, "proton_density"] == pytest.approx(5.25)
    assert df["timestamp"].dt.tz is not None  # UTC-aware


def test_parse_hapi_csv_skips_comments_and_empty_bodies():
    assert sw._parse_hapi_csv("", ["Time"], {}) is None
    assert sw._parse_hapi_csv("# comment only\n", ["Time"], {}) is None
    # A row of the wrong width is discarded rather than misaligned.
    assert sw._parse_hapi_csv("2024-01-01T00:00:00Z,1\n", ["Time", "a", "b"], {}) is None


# --------------------------------------------------------------------------- #
# offline: full fetch paths against a fake HAPI server
# --------------------------------------------------------------------------- #
_OMNI_INFO = {
    "parameters": [
        {"name": "Time", "fill": None},
        {"name": "flow_speed", "fill": "99999.9"},
        {"name": "proton_density", "fill": "999.99"},
        {"name": "T", "fill": "9999999.0"},
        {"name": "BX_GSE", "fill": "9999.99"},
        {"name": "BY_GSM", "fill": "9999.99"},
        {"name": "BZ_GSM", "fill": "9999.99"},
    ]
}
_OMNI_CSV = (
    "2024-09-12T00:00:00.000Z,99999.9,999.99,9999999.0,9999.99,9999.99,9999.99\n"
    "2024-09-12T00:01:00.000Z,450.5,5.25,151856,3.49,-4.41,-6.19\n"
)
_DST_INFO = {
    "parameters": [
        {"name": "Time", "fill": None},
        {"name": "dstValue", "fill": "99999"},
        {"name": "versionCode", "fill": None},
    ]
}
_KP_INFO = {
    "parameters": [
        {"name": "Time", "fill": None},
        {"name": "kpValue", "fill": "99.9"},
        {"name": "apValue", "fill": "999"},
        {"name": "versionCode", "fill": None},
    ]
}
_DST_CSV = "2024-09-12T00:29:30Z,-58,10\n2024-09-12T01:29:30Z,-121,10\n"
_KP_CSV = "2024-09-12T01:30:00Z,7.00,48,20\n"


def _install_cdaweb(monkeypatch, data_text: str = _OMNI_CSV) -> None:
    def fake(method, url, *, params=None, timeout=None, retries=None):
        if url.endswith("/info"):
            return _Resp(payload=_OMNI_INFO)
        return _Resp(text=data_text)

    monkeypatch.setattr(sw, "_request", fake)


def _install_kyoto(monkeypatch) -> None:
    def fake(method, url, *, params=None, timeout=None, retries=None):
        dataset = (params or {}).get("id")
        if url.endswith("/info"):
            return _Resp(payload=_DST_INFO if dataset == sw.DST_DATASET else _KP_INFO)
        return _Resp(text=_DST_CSV if dataset == sw.DST_DATASET else _KP_CSV)

    monkeypatch.setattr(sw, "_request", fake)


def test_fetch_solar_wind_offline_end_to_end(monkeypatch):
    _install_cdaweb(monkeypatch)
    df = sw.fetch_solar_wind("2024-09-12", "2024-09-12")

    assert isinstance(df, pd.DataFrame) and not sw.is_error(df)
    assert list(df.columns) == ["timestamp", "V", "Np", "T", "Bx", "By", "Bz"]
    assert len(df) == 2  # the fill-only row is kept as NaN, not dropped
    assert pd.isna(df.loc[0, "V"]) and pd.isna(df.loc[0, "Bz"])
    assert df.loc[1, "V"] == pytest.approx(450.5)
    assert df.loc[1, "Bz"] == pytest.approx(-6.19)
    assert df["timestamp"].is_monotonic_increasing
    assert df.attrs["dataset"] == sw.SOLAR_WIND_DATASETS[0]


def test_fetch_solar_wind_empty_range_is_no_data(monkeypatch):
    _install_cdaweb(monkeypatch, data_text="")
    res = sw.fetch_solar_wind("2024-09-12", "2024-09-12")
    assert sw.is_error(res) and res["error"] == "no_data"
    assert "hint" in res


def test_fetch_solar_wind_api_unavailable(monkeypatch):
    def boom(method, url, **kwargs):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(sw, "_request", boom)
    res = sw.fetch_solar_wind("2024-09-12", "2024-09-12")
    assert sw.is_error(res) and res["error"] == "api_unavailable"
    assert "hint" in res


def test_fetch_solar_wind_rejects_bad_dates():
    res = sw.fetch_solar_wind("not-a-date", "2024-09-12")
    assert sw.is_error(res) and res["error"] == "invalid_input"
    backwards = sw.fetch_solar_wind("2024-09-12", "2024-09-11")
    assert sw.is_error(backwards) and backwards["error"] == "invalid_input"


def test_fetch_indices_offline_broadcasts_kp_over_its_three_hours(monkeypatch):
    _install_kyoto(monkeypatch)
    df = sw.fetch_geomagnetic_indices("2024-09-12", "2024-09-12")

    assert isinstance(df, pd.DataFrame) and not sw.is_error(df)
    assert list(df.columns) == ["timestamp", "Dst", "Kp"]
    # Dst is floored from its :29:30 centre to the hour.
    assert set(df["Dst"].dropna().astype(float)) == {-58.0, -121.0}
    # Kp = 7.0 is a 3-hour value -> present for 00, 01 and 02 UT.
    kp_rows = df[df["Kp"] == 7.0]
    assert len(kp_rows) == 3
    assert list(kp_rows["timestamp"].dt.hour) == [0, 1, 2]


def test_fetch_indices_api_unavailable(monkeypatch):
    def boom(method, url, **kwargs):
        raise ConnectionError("kyoto down")

    monkeypatch.setattr(sw, "_request", boom)
    res = sw.fetch_geomagnetic_indices("2024-09-12", "2024-09-12")
    assert sw.is_error(res) and res["error"] == "api_unavailable"


def test_is_error_contract():
    assert sw.is_error({"ok": False, "error": "no_data"})
    assert not sw.is_error(pd.DataFrame())
    assert not sw.is_error({"ok": True})


# --------------------------------------------------------------------------- #
# live network: the OMNI + Kyoto services
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_live_solar_wind_2024_09_10_has_southward_bz():
    """2024-09-10 was quiet, but the IMF still turned southward (Bz < 0)."""
    df = sw.fetch_solar_wind("2024-09-10", "2024-09-10")
    assert not sw.is_error(df), df
    assert list(df.columns) == ["timestamp", "V", "Np", "T", "Bx", "By", "Bz"]
    assert len(df) == 1440
    assert df["V"].notna().sum() > 500
    assert float(df["Bz"].min()) < 0
    assert float(df["V"].mean()) > 300


@pytest.mark.network
def test_live_indices_2024_09_10_load_dst_and_kp():
    """The loader returns both indices; 2024-09-10 itself was quiet."""
    df = sw.fetch_geomagnetic_indices("2024-09-10", "2024-09-10")
    assert not sw.is_error(df), df
    assert list(df.columns) == ["timestamp", "Dst", "Kp"]
    assert df["Dst"].notna().any() and df["Kp"].notna().any()
    # Documented reality: this day is NOT the storm (Dst stays >= 0, Kp <= 3).
    assert float(df["Dst"].min()) >= 0
    assert float(df["Kp"].max()) <= 3


@pytest.mark.network
def test_live_dst_is_negative_during_the_real_storm_2024_09_12():
    """Physical validation: the 2024-09-12 storm drove Dst strongly negative."""
    df = sw.fetch_geomagnetic_indices("2024-09-12", "2024-09-12")
    assert not sw.is_error(df), df
    assert float(df["Dst"].min()) < -50
    assert float(df["Kp"].max()) >= 5


@pytest.mark.network
def test_live_strong_southward_bz_during_the_storm_2024_09_12():
    """A storm needs sustained southward IMF; 2024-09-12 delivers it."""
    df = sw.fetch_solar_wind("2024-09-12", "2024-09-12")
    assert not sw.is_error(df), df
    assert float(df["Bz"].min()) < -10


@pytest.mark.network
def test_live_multi_day_window_is_concatenated():
    df = sw.fetch_geomagnetic_indices("2024-09-11", "2024-09-12")
    assert not sw.is_error(df), df
    hours = set(df["timestamp"].dt.day)
    assert {11, 12} <= hours
