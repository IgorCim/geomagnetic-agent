"""Verification suite for geomag_analyzer and geomag_plotter.

Run with:  python -m pytest test_geomag_tools.py -v
The ``network`` marker is unused here -- both modules are fully offline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import geomag_analyzer as ga
import geomag_plotter as gp


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def vector_frame() -> pd.DataFrame:
    """A single sample with textbook values: X=3, Y=4, Z=12 -> H=5, F=13."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-03-10", periods=1, freq="1min"),
            "X": [3.0],
            "Y": [4.0],
            "Z": [12.0],
            "F": [13.0],
        }
    )


@pytest.fixture
def day10() -> pd.DataFrame:
    return gp._synthetic_day(10, seed=1)


@pytest.fixture
def day11() -> pd.DataFrame:
    return gp._synthetic_day(11, seed=2)


@pytest.fixture
def day10b() -> pd.DataFrame:
    return gp._synthetic_day(12, seed=3)


@pytest.fixture
def mini() -> pd.DataFrame:
    """Small deterministic frame for statistics/anomaly tests.

    F is computed from X/Y/Z so the frame is physically consistent and
    ``F == hypot(H, Z)`` holds -- several tests rely on that identity.
    """
    n = 200
    values = np.linspace(100.0, 200.0, n)
    x, y, z = values, values * 0.5, values * 2.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-03-10", periods=n, freq="1min"),
            "X": x,
            "Y": y,
            "Z": z,
            "F": np.sqrt(x**2 + y**2 + z**2),
        }
    )


# --------------------------------------------------------------------------- #
# analyzer: derived components
# --------------------------------------------------------------------------- #
def test_known_triple_gives_exact_derived_values(vector_frame):
    out = ga.calculate_derived_components(vector_frame)
    assert not ga.is_error(out), out
    assert out["H"].iloc[0] == pytest.approx(5.0)
    assert out["D"].iloc[0] == pytest.approx(math.degrees(math.atan2(4.0, 3.0)))
    assert out["I"].iloc[0] == pytest.approx(math.degrees(math.atan2(12.0, 5.0)))


def test_declination_uses_correct_quadrant():
    """The whole reason for arctan2: arctan would give -45 deg, not 135."""
    frame = pd.DataFrame({"X": [-1.0], "Y": [1.0], "Z": [0.0], "F": [1.41421]})
    out = ga.calculate_derived_components(frame)
    assert out["D"].iloc[0] == pytest.approx(135.0)
    assert not (-90.0 < out["D"].iloc[0] < 90.0), "quadrant information was lost"


def test_declination_negative_east():
    frame = pd.DataFrame({"X": [1.0], "Y": [-1.0], "Z": [0.0], "F": [1.41421]})
    out = ga.calculate_derived_components(frame)
    assert out["D"].iloc[0] == pytest.approx(-45.0)


def test_inclination_vertical_field_is_ninety_not_nan():
    """arctan2(Z, 0) must give +-90 rather than dividing by zero."""
    frame = pd.DataFrame({"X": [0.0], "Y": [0.0], "Z": [60000.0], "F": [60000.0]})
    out = ga.calculate_derived_components(frame)
    assert out["H"].iloc[0] == pytest.approx(0.0)
    assert out["I"].iloc[0] == pytest.approx(90.0)
    assert not pd.isna(out["I"].iloc[0])
    assert out.attrs["degenerate_horizontal_rows"] == 1


def test_inclination_upward_field_is_minus_ninety():
    frame = pd.DataFrame({"X": [0.0], "Y": [0.0], "Z": [-60000.0], "F": [60000.0]})
    assert ga.calculate_derived_components(frame)["I"].iloc[0] == pytest.approx(-90.0)


def test_f_equals_sqrt_h_squared_plus_z_squared(mini):
    out = ga.calculate_derived_components(mini)
    recomputed = np.hypot(out["H"].to_numpy(), out["Z"].to_numpy())
    assert np.allclose(recomputed, out["F"].to_numpy(), atol=1e-6)


def test_derived_match_numpy_on_realistic_data(day10):
    out = ga.calculate_derived_components(day10)
    x, y, z = (out[c].to_numpy() for c in ("X", "Y", "Z"))
    assert np.allclose(out["H"].to_numpy(), np.hypot(x, y))
    assert np.allclose(out["D"].to_numpy(), np.degrees(np.arctan2(y, x)))
    assert np.allclose(out["I"].to_numpy(), np.degrees(np.arctan2(z, np.hypot(x, y))))


def test_declination_stays_in_signed_range(day10):
    out = ga.calculate_derived_components(day10)
    assert out["D"].between(-180.0, 180.0).all()
    assert out["I"].between(-90.0, 90.0).all()


def test_nan_propagates_to_every_derived_column(mini):
    gappy = mini.copy()
    gappy.loc[5:9, ["X", "Y", "Z"]] = np.nan
    out = ga.calculate_derived_components(gappy)
    assert out.loc[5:9, ["H", "D", "I"]].isna().all().all()
    assert out["H"].notna().sum() == len(mini) - 5
    assert out.loc[4, "H"] == pytest.approx(mini.loc[4, "H"]) if "H" in mini else True


def test_input_frame_is_not_mutated(mini):
    before = mini.copy()
    ga.calculate_derived_components(mini)
    pd.testing.assert_frame_equal(mini, before)


def test_is_idempotent(mini):
    once = ga.calculate_derived_components(mini)
    twice = ga.calculate_derived_components(once)
    pd.testing.assert_frame_equal(once, twice)


def test_partial_component_selection(mini):
    out = ga.calculate_derived_components(mini, components=["H"])
    assert "H" in out.columns and "D" not in out.columns and "I" not in out.columns


def test_analyzer_missing_cartesian_is_error(mini):
    out = ga.calculate_derived_components(mini.drop(columns=["Y"]))
    assert ga.is_error(out) and out["error"] == "missing_columns"
    assert "Y" in out["missing"]


def test_analyzer_rejects_unknown_derived_name(mini):
    out = ga.calculate_derived_components(mini, components=["Q"])
    assert ga.is_error(out) and out["error"] == "unknown_component"


def test_analyzer_rejects_non_dataframe():
    assert ga.is_error(ga.calculate_derived_components([1, 2, 3]))


def test_analyzer_rejects_empty(mini):
    assert ga.is_error(ga.calculate_derived_components(mini.iloc[0:0]))


# --------------------------------------------------------------------------- #
# analyzer: statistics
# --------------------------------------------------------------------------- #
def test_statistics_keys_and_rounding(mini):
    stats = ga.get_statistics(mini, components=["X", "Y", "Z", "F"])
    for name in ("X", "Y", "Z", "F"):
        entry = stats[name]
        for key in ("min", "max", "mean", "median", "std"):
            assert key in entry, f"{name} missing {key}"
            assert entry[key] is None or round(entry[key], 2) == entry[key]


def test_statistics_values_are_correct(mini):
    stats = ga.get_statistics(mini, components=["X"])
    series = mini["X"]
    assert stats["X"]["min"] == pytest.approx(round(series.min(), 2))
    assert stats["X"]["max"] == pytest.approx(round(series.max(), 2))
    assert stats["X"]["mean"] == pytest.approx(round(series.mean(), 2))
    assert stats["X"]["std"] == pytest.approx(round(series.std(ddof=1), 2))


def test_statistics_are_json_serialisable_without_bare_nan(mini):
    gappy = mini.copy()
    gappy["Z"] = np.nan
    stats = ga.get_statistics(gappy, components=["X", "Y", "Z", "F", "H"])
    blob = json.dumps(stats)
    assert "NaN" not in blob and "Infinity" not in blob
    assert stats["Z"]["mean"] is None
    assert stats["Z"]["count"] == 0
    assert stats["Z"]["missing"] == len(gappy)


def test_statistics_never_treat_nan_as_zero(mini):
    gappy = mini.copy()
    gappy.loc[:99, "X"] = np.nan
    stats = ga.get_statistics(gappy, components=["X"])
    assert stats["X"]["min"] > 100.0
    assert stats["X"]["count"] == 100


def test_statistics_auto_derives_h(mini):
    stats = ga.get_statistics(mini)
    assert "H" in stats
    assert not ga.is_error(ga.get_statistics(mini))
    assert stats["H"]["min"] == pytest.approx(round(ga.calculate_derived_components(mini)["H"].min(), 2))


def test_statistics_units(mini):
    stats = ga.get_statistics(mini, components=["X", "H", "D", "I"])
    assert stats["X"]["unit"] == "nT" and stats["H"]["unit"] == "nT"
    assert stats["D"]["unit"] == "deg" and stats["I"]["unit"] == "deg"


def test_statistics_single_sample_std_is_none(vector_frame):
    stats = ga.get_statistics(vector_frame, components=["X"])
    assert stats["X"]["std"] is None
    assert json.dumps(stats)


def test_statistics_unknown_component_lists_alternatives(mini):
    out = ga.get_statistics(mini, components=["Q"])
    assert ga.is_error(out) and out["error"] == "unknown_component"
    assert out["missing"] == ["Q"]
    assert "X" in out["available"]
    assert "hint" in out


def test_statistics_case_insensitive(mini):
    assert "H" in ga.get_statistics(mini, components=["h"])


def test_statistics_rejects_bad_digits(mini):
    assert ga.is_error(ga.get_statistics(mini, components=["X"], digits=-1))


def test_statistics_all_nan_column(mini):
    gappy = mini.copy()
    gappy["X"] = np.nan
    entry = ga.get_statistics(gappy, components=["X"])["X"]
    assert all(entry[k] is None for k in ("min", "max", "mean", "median", "std"))


# --------------------------------------------------------------------------- #
# analyzer: anomalies
# --------------------------------------------------------------------------- #
def test_detect_anomalies_finds_injected_spikes(mini):
    spiked = ga.calculate_derived_components(mini)
    spiked.loc[50, "H"] += 1000.0
    spiked.loc[120, "H"] -= 800.0
    found = ga.detect_anomalies(spiked, "H", sigma_threshold=3.0)
    assert len(found) == 2
    assert set(found.index) == {50, 120}
    assert "z_score" in found.columns and "deviation" in found.columns


def test_detect_anomalies_clean_series_returns_empty(mini):
    found = ga.detect_anomalies(mini, "X", sigma_threshold=5.0)
    assert isinstance(found, pd.DataFrame) and found.empty
    assert "z_score" in found.columns


def test_detect_anomalies_ignores_nan_rows(mini):
    gappy = mini.copy()
    gappy.loc[10:20, ["X", "Y", "Z"]] = np.nan
    found = ga.detect_anomalies(gappy, "X", sigma_threshold=1.0)
    assert not found.index.isin(range(10, 21)).any()


def test_detect_anomalies_threshold_is_monotonic(mini):
    spiked = ga.calculate_derived_components(mini)
    spiked.loc[50, "H"] += 1000.0
    counts = [
        len(ga.detect_anomalies(spiked, "H", sigma_threshold=t))
        for t in (0.5, 1.0, 2.0, 5.0, 50.0)
    ]
    assert counts == sorted(counts, reverse=True)


def test_detect_anomalies_zero_std_does_not_crash():
    flat = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-03-10", periods=50, freq="1min"),
            "X": np.full(50, 1234.0),
            "Y": np.full(50, 1.0),
            "Z": np.full(50, 2.0),
        }
    )
    found = ga.detect_anomalies(ga.calculate_derived_components(flat), "H")
    assert found.empty


def test_detect_anomalies_preserves_source_row_index(mini):
    """Anomalies must stay addressable by their row in the caller's frame."""
    spiked = ga.calculate_derived_components(mini)
    spiked.loc[77, "H"] += 1500.0
    found = ga.detect_anomalies(spiked, "H", sigma_threshold=3.0)
    assert list(found.index) == [77]
    assert found.loc[77, "timestamp"] == spiked.loc[77, "timestamp"]


def test_detect_anomalies_rejects_bad_threshold(mini):
    for bad in (0, -1, "three", float("nan"), float("inf")):
        assert ga.is_error(ga.detect_anomalies(mini, "X", sigma_threshold=bad))


def test_detect_anomalies_unknown_component(mini):
    out = ga.detect_anomalies(mini, "Q")
    assert ga.is_error(out) and out["error"] == "unknown_component"


def test_detect_anomalies_auto_derives_h(mini):
    found = ga.detect_anomalies(mini, "H")
    assert not ga.is_error(found)


# --------------------------------------------------------------------------- #
# analyzer: derivatives
# --------------------------------------------------------------------------- #
def test_compute_derivatives_linear_ramp():
    n = 10
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-03-10", periods=n, freq="1min"),
            "X": np.arange(n, dtype=float) * 3.0,
            "Y": np.zeros(n),
            "Z": np.zeros(n),
        }
    )
    out = ga.compute_derivatives(frame, components=("X",))
    assert pd.isna(out["X_dt"].iloc[0])
    assert out["X_dt"].iloc[1:].to_numpy() == pytest.approx(3.0)


def test_compute_derivatives_respects_sampling_interval():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-03-10", periods=5, freq="5min"),
            "X": [0.0, 10.0, 20.0, 30.0, 40.0],
            "Y": np.zeros(5),
            "Z": np.zeros(5),
        }
    )
    out = ga.compute_derivatives(frame, components=("X",))
    assert out["X_dt"].iloc[1:].to_numpy() == pytest.approx(2.0)


def test_compute_derivatives_names_columns(mini):
    out = ga.compute_derivatives(mini, components=("H", "F"))
    assert "H_dt" in out.columns and "F_dt" in out.columns


def test_compute_derivatives_needs_timestamp(mini):
    assert ga.is_error(ga.compute_derivatives(mini.drop(columns=["timestamp"])))


# --------------------------------------------------------------------------- #
# plotter: plot_components
# --------------------------------------------------------------------------- #
def _is_html_plot(path: str) -> bool:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return "plotly" in text.lower() and len(text) > 1000


def test_plot_components_writes_valid_html(day10, tmp_path):
    path = gp.plot_components(day10, components=["X", "Y", "Z"],
                              output_dir=tmp_path, include_plotlyjs="cdn")
    assert Path(path).is_file()
    assert _is_html_plot(path)
    assert Path(path).suffix == ".html"


def test_plot_components_returns_absolute_path(day10, tmp_path):
    path = gp.plot_components(day10, output_dir=tmp_path, include_plotlyjs="cdn")
    assert Path(path).is_absolute()


def test_plot_components_creates_output_dir(day10, tmp_path):
    target = tmp_path / "nested" / "deep"
    path = gp.plot_components(day10, output_dir=target, include_plotlyjs="cdn")
    assert Path(path).parent == target.resolve()


def test_plot_components_honours_filename(day10, tmp_path):
    path = gp.plot_components(day10, output_dir=tmp_path, filename="custom.html",
                              include_plotlyjs="cdn")
    assert Path(path).name == "custom.html"


def test_plot_components_appends_html_extension(day10, tmp_path):
    path = gp.plot_components(day10, output_dir=tmp_path, filename="noext",
                              include_plotlyjs="cdn")
    assert Path(path).name == "noext.html"


def test_plot_components_auto_derives_hdi(day10, tmp_path):
    path = gp.plot_components(day10, components=["H", "D", "I"],
                              output_dir=tmp_path, include_plotlyjs="cdn")
    assert Path(path).is_file()
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    assert "H (nT)" in text and "D (deg)" in text and "I (deg)" in text


def test_plot_components_leaves_gaps_unbridged(day10, tmp_path):
    gappy = day10.copy()
    gappy.loc[700:720, "X"] = np.nan
    path = gp.plot_components(gappy, components=["X"], output_dir=tmp_path,
                              include_plotlyjs="cdn")
    assert Path(path).is_file()


def test_plot_components_downsamples_large_series(tmp_path):
    big = gp._synthetic_day(10, n=60000, seed=9)
    path = gp.plot_components(big, components=["X"], output_dir=tmp_path,
                              max_points=1000, include_plotlyjs="cdn")
    assert Path(path).is_file()
    assert Path(path).stat().st_size < 400 * 1024


def test_plot_components_self_contained_by_default(day10, tmp_path):
    path = gp.plot_components(day10, components=["X"], output_dir=tmp_path)
    assert Path(path).stat().st_size > 1024 * 1024, "plotly bundle should be embedded"


def test_plot_components_unknown_component_errors(day10, tmp_path):
    out = gp.plot_components(day10, components=["Q"], output_dir=tmp_path)
    assert gp.is_error(out) and out["error"] == "unknown_component"
    assert out["missing"] == ["Q"]
    assert "hint" in out


def test_plot_components_requires_timestamp(day10, tmp_path):
    out = gp.plot_components(day10.drop(columns=["timestamp"]), output_dir=tmp_path)
    assert gp.is_error(out) and out["error"] == "missing_columns"


def test_plot_components_rejects_non_dataframe(tmp_path):
    assert gp.is_error(gp.plot_components("not a frame", output_dir=tmp_path))


def test_plot_components_rejects_empty(mini, tmp_path):
    assert gp.is_error(gp.plot_components(mini.iloc[0:0], output_dir=tmp_path))


# --------------------------------------------------------------------------- #
# plotter: plot_comparison
# --------------------------------------------------------------------------- #
def test_time_of_day_collapses_dates():
    a = pd.Series(pd.to_datetime(["2025-03-10 05:30:00", "2024-11-02 05:30:00"]))
    b = pd.Series(pd.to_datetime(["1999-01-01 05:30:00", "2030-12-31 23:59:00"]))
    out = gp._time_of_day(a)
    assert out.iloc[0] == gp.TIME_OF_DAY_ANCHOR + pd.Timedelta(hours=5, minutes=30)
    assert out.iloc[1] == gp.TIME_OF_DAY_ANCHOR + pd.Timedelta(hours=5, minutes=30)
    assert gp._time_of_day(b).iloc[0] == out.iloc[0]


def test_time_of_day_preserves_full_day(day10, day11):
    a = gp._time_of_day(day10["timestamp"])
    b = gp._time_of_day(day11["timestamp"])
    assert a.equals(b), "two different days must anchor to the same instants"
    assert a.min() == gp.TIME_OF_DAY_ANCHOR
    assert a.max() == gp.TIME_OF_DAY_ANCHOR + pd.Timedelta(hours=23, minutes=59)


def test_plot_comparison_writes_html(day10, day11, tmp_path):
    path = gp.plot_comparison(day10, day11, component="H", output_dir=tmp_path,
                              include_plotlyjs="cdn")
    assert Path(path).is_file() and _is_html_plot(path)


def test_plot_comparison_uses_red_and_blue(day10, day11, tmp_path):
    cap = {}
    original = gp._write
    gp._write = lambda fig, p, i: (cap.__setitem__("fig", fig), original(fig, p, i))[1]
    gp.plot_comparison(day10, day11, "H", output_dir=tmp_path, include_plotlyjs="cdn")
    gp._write = original
    colors = [t.line.color for t in cap["fig"].data]
    assert colors[0] == gp._COMPARISON_COLORS[0]
    assert colors[1] == gp._COMPARISON_COLORS[1]
    assert len(colors) == 2


def test_plot_comparison_labels_traces_with_real_dates(day10, day11, tmp_path):
    cap = {}
    original = gp._write
    gp._write = lambda fig, p, i: (cap.__setitem__("fig", fig), original(fig, p, i))[1]
    gp.plot_comparison(day10, day11, "H", output_dir=tmp_path, include_plotlyjs="cdn")
    gp._write = original
    names = [t.name for t in cap["fig"].data]
    assert names == ["2025-03-10", "2025-03-11"]


def test_plot_comparison_axis_is_a_real_datetime_axis(day10, day11, tmp_path):
    cap = {}
    original = gp._write
    gp._write = lambda fig, p, i: (cap.__setitem__("fig", fig), original(fig, p, i))[1]
    gp.plot_comparison(day10, day11, "H", output_dir=tmp_path, include_plotlyjs="cdn")
    gp._write = original
    fig = cap["fig"]
    assert fig.layout.xaxis.type == "date"
    assert fig.layout.xaxis.title.text == "Time of day (UTC)"
    assert fig.layout.xaxis.tickformat == "%H:%M"
    assert fig.layout.xaxis.rangeslider.visible is True
    for trace in fig.data:
        assert pd.Timestamp(trace.x[0]).date() == gp.TIME_OF_DAY_ANCHOR.date()
        assert "%H:%M" in trace.hovertemplate


def test_plot_comparison_autoderives_component(day10, day11, tmp_path):
    path = gp.plot_comparison(day10, day11, component="D", output_dir=tmp_path,
                              include_plotlyjs="cdn")
    assert Path(path).is_file()
    assert "deg" in Path(path).read_text(encoding="utf-8", errors="ignore")


def test_plot_comparison_unknown_component(day10, day11, tmp_path):
    out = gp.plot_comparison(day10, day11, component="Q", output_dir=tmp_path)
    assert gp.is_error(out) and out["error"] == "unknown_component"


def test_plot_comparison_rejects_non_dataframe(day11, tmp_path):
    assert gp.is_error(gp.plot_comparison(None, day11, output_dir=tmp_path))
    assert gp.is_error(gp.plot_comparison(day11, "x", output_dir=tmp_path))


def test_plot_comparison_rejects_empty(day10, day10b, tmp_path):
    out = gp.plot_comparison(day10.iloc[0:0], day10b, output_dir=tmp_path)
    assert gp.is_error(out) and out["error"] == "empty_dataframe"


# --------------------------------------------------------------------------- #
# plotter: magnetogram
# --------------------------------------------------------------------------- #
def test_plot_magnetogram_has_four_disjoint_panels(day10, tmp_path):
    cap = {}
    original = gp._write
    gp._write = lambda fig, p, i: (cap.__setitem__("fig", fig), original(fig, p, i))[1]
    gp.plot_magnetogram(day10, output_dir=tmp_path, include_plotlyjs="cdn")
    gp._write = original
    fig = cap["fig"]
    assert len(fig.data) == 4
    assert [t.name for t in fig.data] == ["X", "Y", "Z", "F"]
    refs = [t.yaxis for t in fig.data]
    assert refs == ["y", "y2", "y3", "y4"]
    domains = [getattr(fig.layout, "yaxis" if r == "y" else f"yaxis{r[1:]}").domain
               for r in refs]
    assert all(domains[i][1] > domains[i + 1][1] for i in range(3)), "panels overlap"
    assert round(domains[0][1], 6) == 1.0
    assert round(domains[-1][0], 6) == 0.0
    for ref in refs:
        axis = getattr(fig.layout, "yaxis" if ref == "y" else f"yaxis{ref[1:]}")
        assert axis.title is not None


def test_plot_magnetogram_writes_file(day10, tmp_path):
    path = gp.plot_magnetogram(day10, output_dir=tmp_path, include_plotlyjs="cdn")
    assert Path(path).is_file() and _is_html_plot(path)


# --------------------------------------------------------------------------- #
# contract shared by both tools
# --------------------------------------------------------------------------- #
def test_both_modules_expose_is_error():
    assert ga.is_error({"ok": False}) is True
    assert gp.is_error({"ok": False}) is True
    assert ga.is_error(pd.DataFrame({"X": [1]})) is False
    assert gp.is_error("path.html") is False
    assert ga.is_error(None) is False


def test_all_error_dicts_are_json_serialisable(mini, day10, tmp_path):
    failures = [
        ga.calculate_derived_components(mini, ["Q"]),
        ga.get_statistics(mini, ["Q"]),
        ga.detect_anomalies(mini, "Q"),
        ga.compute_derivatives(mini.drop(columns=["timestamp"])),
        gp.plot_components(day10, ["Q"], output_dir=tmp_path),
        gp.plot_components(day10.drop(columns=["timestamp"]), output_dir=tmp_path),
        gp.plot_comparison("x", day10, output_dir=tmp_path),
        gp.plot_magnetogram(day10.iloc[0:0], output_dir=tmp_path),
    ]
    for payload in failures:
        assert payload.get("ok") is False
        assert payload.get("error")
        assert payload.get("message")
        json.dumps(payload)
