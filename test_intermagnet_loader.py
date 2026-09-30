"""Verification suite for intermagnet_loader.

Run with:  python -m pytest test_intermagnet_loader.py -v
Network-dependent tests are marked ``network``; skip them with ``-m "not network"``.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

import intermagnet_loader as im


# --------------------------------------------------------------------------- #
# offline: parsers
# --------------------------------------------------------------------------- #
def _iaga_line(stamp: str, doy: int, x: str, y: str, z: str, f: str) -> str:
    """Build a correctly aligned 70-char IAGA-2002 data record.

    Layout: ``YYYY-MM-DD``(10) space ``HH:MM:SS.mmm``(12) space ``DOY``(3)
    pad(3) then X/Y/Z/F right-aligned in 10-char cells. Building the fixture
    programmatically keeps the column arithmetic out of the assertions.
    """
    date_part, time_part = stamp.split(" ")
    line = f"{date_part:10s} {time_part:12s} {doy:>3d}   "
    assert len(line) == 30
    for field in (x, y, z, f):
        line += f"{field:>10s}"
    assert len(line) == 70
    return line


IAGA_SAMPLE = "\n".join(
    [
        " Format                 IAGA-2002                                    |",
        " Source of Data         Test institute                               |",
        " Station Name           Test                                         |",
        " IAGA Code              TST                                         |",
        "DATE       TIME         DOY     TSTX      TSTY      TSTZ      TSTF   |",
        _iaga_line("2021-03-10 00:00:00.000", 69, "18311.70", "-1363.20", "57674.00", "60526.50"),
        _iaga_line("2021-03-10 00:01:00.000", 69, "18311.80", "-1363.20", "57674.00", "60526.60"),
        _iaga_line("2021-03-10 00:02:00.000", 69, "99999.00", "99999.00", "99999.00", "99999.00"),
        _iaga_line("2021-03-10 00:03:00.000", 69, "-12.34", "999.99", "-45678.00", "45678.12"),
        _iaga_line("2021-03-10 00:04:00.000", 69, "", "", "100.00", "200.00"),
    ]
)


def test_parse_iaga2002_handles_negatives_and_sentinels():
    df = im.parse_iaga2002(IAGA_SAMPLE)
    assert len(df) == 5
    assert list(df.columns) == ["timestamp", "X", "Y", "Z", "F"]
    assert df["timestamp"].iloc[0] == pd.Timestamp("2021-03-10 00:00:00")
    assert df["X"].iloc[0] == pytest.approx(18311.70)
    assert df["Y"].iloc[0] == pytest.approx(-1363.20)
    assert df["Z"].iloc[0] == pytest.approx(57674.00)
    assert df["F"].iloc[0] == pytest.approx(60526.50)


def test_parse_iaga2002_maps_99999_to_nan():
    df = im.parse_iaga2002(IAGA_SAMPLE)
    row = df.iloc[2]
    assert all(pd.isna(row[c]) for c in ("X", "Y", "Z", "F"))


def test_parse_iaga2002_maps_blank_fields_to_nan():
    row = im.parse_iaga2002(IAGA_SAMPLE).iloc[4]
    assert pd.isna(row["X"]) and pd.isna(row["Y"])
    assert row["Z"] == pytest.approx(100.00)
    assert row["F"] == pytest.approx(200.00)


def test_parse_iaga2002_ignores_headers_and_empty_input():
    assert im.parse_iaga2002("").empty
    assert im.parse_iaga2002("no data here\n# comment\n").empty


def test_parse_iaga2002_keeps_legit_negative_values():
    row = im.parse_iaga2002(IAGA_SAMPLE).iloc[3]
    assert row["X"] == pytest.approx(-12.34)
    assert row["Z"] == pytest.approx(-45678.00)


# --------------------------------------------------------------------------- #
# offline: helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("definitive", "definitive"),
        ("Quasi-definitive", "quasi-def"),
        ("provisional", "adjusted"),
        ("variation", "reported"),
        ("best", "best-avail"),
        ("AUTO", "auto"),
        (None, "auto"),
    ],
)
def test_norm_type_aliases(raw, expected):
    assert im._norm_type(raw) == expected


def test_norm_type_rejects_garbage():
    with pytest.raises(ValueError):
        im._norm_type("nonsense")


def test_norm_station_is_case_insensitive():
    assert im._norm_station(" irt ") == "IRT"


@pytest.mark.parametrize(
    "raw,expected",
    [("2025-03-10", date(2025, 3, 10)), ("2025/03/10", date(2025, 3, 10)),
     ("20250310", date(2025, 3, 10)), (date(2025, 3, 10), date(2025, 3, 10))],
)
def test_norm_date_formats(raw, expected):
    assert im._norm_date(raw, "d") == expected


def test_chunk_dates_splits_long_ranges():
    chunks = im._chunk_dates(date(2025, 1, 1), date(2025, 3, 31), 31)
    assert chunks[0][0] == date(2025, 1, 1)
    assert chunks[-1][1] == date(2025, 3, 31)
    assert all(
        (b - a).days + 1 <= 31 for a, b in chunks
    )
    assert chunks[0][1] + pd.Timedelta(days=1) == chunks[1][0]


def test_chunk_dates_keeps_short_range_intact():
    assert im._chunk_dates(date(2025, 3, 10), date(2025, 3, 11), 31) == [
        (date(2025, 3, 10), date(2025, 3, 11))
    ]


def test_clear_cache_returns_count(tmp_path, monkeypatch):
    monkeypatch.setattr(im, "CACHE_DIR", tmp_path)
    (tmp_path / "a.csv").write_text("x")
    (tmp_path / "b.csv").write_text("x")
    assert im.clear_cache() == 2
    assert im.clear_cache() == 0


# --------------------------------------------------------------------------- #
# offline: error contract
# --------------------------------------------------------------------------- #
def test_invalid_input_returns_llm_error():
    out = im.fetch_observatory_data("IRT", "2025-03-11", "2025-03-10", use_cache=False)
    assert im.is_error(out)
    assert out["ok"] is False
    assert out["error"] == "invalid_input"
    for key in ("ok", "error", "message", "station", "start_date", "end_date", "attempts"):
        assert key in out


def test_unknown_station_error_is_explicit():
    out = im.fetch_observatory_data("ZZZ", "2025-03-10", "2025-03-11", use_cache=False)
    assert im.is_error(out)
    assert out["error"] == "station_not_found"
    assert out["station"] == "ZZZ"
    assert "hint" in out


def test_bad_data_type_error():
    out = im.fetch_observatory_data("IRT", "2025-03-10", "2025-03-11", data_type="bogus")
    assert im.is_error(out) and out["error"] == "invalid_input"


def test_pre_network_date_is_no_data():
    out = im.fetch_observatory_data("IRT", "1990-01-01", "1990-01-02", use_cache=False)
    assert im.is_error(out) and out["error"] == "no_data"
    assert out["attempts"]


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_station_catalogue_shape():
    stations = im.get_available_stations()
    assert len(stations) > 150
    assert "IRT" in stations
    name, lat, lon = stations["IRT"]
    assert isinstance(name, str) and name
    assert lat is None or -90 <= lat <= 90


@pytest.mark.network
def test_fetch_irt_shape_and_columns():
    df = im.fetch_observatory_data("IRT", "2025-03-10", "2025-03-11", use_cache=False)
    assert not im.is_error(df), df
    assert list(df.columns) == ["timestamp", "X", "Y", "Z", "F"]
    assert len(df) == 2 * 1440
    assert df["timestamp"].is_monotonic_increasing
    assert not df["timestamp"].duplicated().any()
    for col in ("X", "Y", "Z", "F"):
        assert pd.api.types.is_numeric_dtype(df[col])


@pytest.mark.network
def test_f_is_physically_consistent_with_xyz():
    df = im.fetch_observatory_data("IRT", "2025-03-10", "2025-03-10", use_cache=False)
    assert not im.is_error(df), df
    ok = df.dropna()
    f_calc = (ok["X"] ** 2 + ok["Y"] ** 2 + ok["Z"] ** 2) ** 0.5
    assert (f_calc - ok["F"]).abs().max() < 0.02


@pytest.mark.network
def test_iaga2002_parser_matches_covjson_for_same_request():
    """The two wire formats must decode to identical numbers."""
    params = {
        "observatoryIagaCode": "ABK",
        "samplesPerDay": "Minute",
        "dataStartDate": "2025-03-10",
        "dataDuration": 1,
        "publicationState": "quasi-def",
        "orientation": "XYZF",
        "recordTermination": "UNIX",
    }
    text = im._request("GET", im.GIN_URL, params={**params, "Request": "GetData", "Format": "iaga2002"}).text
    payload = im._request("GET", im.GIN_URL, params={**params, "Request": "GetData", "Format": "json"}).json()
    a = im.parse_iaga2002(text).set_index("timestamp")
    b = im._parse_covjson(payload).set_index("timestamp")
    assert len(a) == len(b) > 0
    assert a.index.equals(b.index)
    for col in im.COMPONENTS:
        assert a[col].equals(b[col]), f"{col} differs between iaga2002 and covJson"


@pytest.mark.network
def test_definitive_falls_back_when_unavailable():
    """IRT has no definitive data for this period; the loader must still answer."""
    df = im.fetch_observatory_data("IRT", "2025-03-10", "2025-03-11",
                                   data_type="definitive", use_cache=False)
    assert not im.is_error(df), df
    assert len(df) == 2880
    states = df.attrs["publication_states"]
    assert states and states[0] in ("definitive", "quasi-def", "adjusted", "reported", "best-avail")
    assert df.attrs["backends"]


@pytest.mark.network
def test_kyoto_provides_historical_definitive():
    df = im.fetch_observatory_data("IRT", "2021-03-10", "2021-03-10",
                                   data_type="definitive", use_cache=False)
    assert not im.is_error(df), df
    assert "kyoto_hapi" in df.attrs["backends"]
    assert len(df) == 1440
    assert df["F"].notna().all()


@pytest.mark.network
def test_long_range_is_chunked():
    df = im.fetch_observatory_data("ABK", "2025-03-01", "2025-03-10",
                                   data_type="auto", use_cache=False)
    assert not im.is_error(df), df
    assert len(df) == 10 * 1440
    assert len(df.attrs["source"]) == 1


@pytest.mark.network
def test_cache_round_trip_is_lossless(tmp_path, monkeypatch):
    monkeypatch.setattr(im, "CACHE_DIR", tmp_path)
    df = im.fetch_observatory_data("IRT", "2025-03-10", "2025-03-10", use_cache=False)
    assert not im.is_error(df), df
    path = im.save_to_cache(df, "IRT", (date(2025, 3, 10), date(2025, 3, 10)), "definitive")
    assert path.exists()
    back = im.load_from_cache("IRT", "2025-03-10", "2025-03-10", "definitive")
    assert back is not None
    assert len(back) == len(df)
    assert (back["F"].values == df["F"].values).all()
    assert back["timestamp"].iloc[0] == df["timestamp"].iloc[0]
    assert back.attrs["cache_meta"]["rows"] == len(df)


@pytest.mark.network
def test_second_identical_call_hits_cache():
    first = im.fetch_observatory_data("IRT", "2025-03-10", "2025-03-10", use_cache=True)
    assert not im.is_error(first), first
    second = im.fetch_observatory_data("IRT", "2025-03-10", "2025-03-10", use_cache=True)
    assert second.attrs.get("from_cache") is True
    assert len(second) == len(first)
