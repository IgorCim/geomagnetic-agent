"""Mathematical analysis of geomagnetic time series produced by
:mod:`intermagnet_loader`.

This is one of the agent's "hands": pure, deterministic, side-effect free
transformations over a DataFrame of INTERMAGNET minute means.

Component conventions
---------------------
Cartesian, INTERMAGNET/IAGA sign convention, all in nT:

* ``X`` -- North component
* ``Y`` -- East component
* ``Z`` -- Down component (positive **downwards**)
* ``F`` -- total field, ``sqrt(X**2 + Y**2 + Z**2)``
* ``H`` -- horizontal field magnitude, ``sqrt(X**2 + Y**2)``
* ``D`` -- declination, degrees, signed from true North, positive East
* ``I`` -- inclination, degrees, positive **downwards**

Why ``arctan2`` and not ``arctan``
---------------------------------
``D = arctan2(Y, X)`` and ``I = arctan2(Z, H)`` are used rather than the
``arctan(Y/X)`` form for three reasons:

1. **Branch correctness.** ``arctan`` returns only ``(-90, 90)`` degrees, so
   ``arctan`` cannot express a declination of e.g. 200 degrees without manual
   quadrant correction. ``arctan2`` returns the full ``(-180, 180]`` directly.
2. **No division by zero.** When ``H == 0`` (a purely vertical field) ``I`` is
   exactly +-90 degrees; ``arctan(Z/H)`` would be ``0/0 -> NaN`` or
   ``inf``. ``arctan2`` handles the degenerate case natively.
3. **Correct sign for negative H/Z.** ``arctan`` loses the quadrant
   information, which matters because both ``Y`` and ``Z`` are signed in the
   INTERMAGNET convention.

Missing-data policy
-------------------
Every function propagates ``NaN`` rather than imputing, dropping or
interpolating. A sample with a missing input yields ``NaN`` in each dependent
output -- for example a gap in ``X`` produces ``NaN`` in ``H``, ``D`` and ``I``.
Silently interpolating would invent magnetic field values, which is never
acceptable for a scientific record; recovering real gaps is the job of the
data-quality tooling, not this module.

Return-value contract (see ``is_error``)
----------------------------------------
Every public function either returns the requested value or, on failure, a
``dict`` of the shape ``{"ok": False, "error": ..., "message": ..., ...}``.
All values are JSON-serialisable: ``NaN`` and infinities are converted to
``None`` so that an LLM can pass the result straight into a tool call. Use
:func:`is_error` to branch.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "calculate_derived_components",
    "get_statistics",
    "detect_anomalies",
    "compute_derivatives",
    "component_units",
    "is_error",
    "CARTESIAN",
    "DERIVED",
    "COMPONENT_UNITS",
    "REQUIRED_FOR_DERIVED",
]

log = logging.getLogger("geomag_analyzer")

CARTESIAN: tuple[str, ...] = ("X", "Y", "Z")
DERIVED: tuple[str, ...] = ("H", "D", "I")
REQUIRED_FOR_DERIVED: tuple[str, ...] = ("X", "Y", "Z")

COMPONENT_UNITS: dict[str, str] = {
    "X": "nT",
    "Y": "nT",
    "Z": "nT",
    "F": "nT",
    "H": "nT",
    "S": "nT",
    "D": "deg",
    "I": "deg",
}

_ANGULAR: frozenset[str] = frozenset({"D", "I"})


# --------------------------------------------------------------------------- #
# contract helpers
# --------------------------------------------------------------------------- #
def is_error(result: Any) -> bool:
    """True when a call failed. Lets an LLM branch without isinstance checks."""
    return isinstance(result, dict) and result.get("ok") is False


def _error(
    code: str,
    message: str,
    *,
    requested: Sequence[str] | str | None = None,
    available: Sequence[str] | None = None,
    missing: Sequence[str] | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    """Build the stable, JSON-serialisable error payload for LLM callers."""
    payload: dict[str, Any] = {
        "ok": False,
        "error": code,
        "message": message,
        "requested": list(requested) if isinstance(requested, (list, tuple)) else requested,
        "available": list(available) if available is not None else None,
        "missing": list(missing) if missing is not None else None,
    }
    if hint:
        payload["hint"] = hint
    return payload


def _jsonable(value: Any, digits: int | None = None) -> Any:
    """Convert numpy scalars to plain Python, NaN/inf to None, optionally round.

    Keeping every statistic JSON-clean matters because these dictionaries are
    handed straight to an LLM tool interface, and bare ``NaN`` is not valid
    strict JSON -- several providers reject the whole message.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def component_units(component: str) -> str | None:
    """Return the unit of ``component`` (``'nT'``, ``'deg'``) or ``None``."""
    return COMPONENT_UNITS.get(str(component).strip().upper())


def _validate_frame(df: Any) -> pd.DataFrame | None:
    if not isinstance(df, pd.DataFrame):
        return _error(
            "invalid_input",
            f"Expected a pandas DataFrame, got {type(df).__name__}.",
        )
    if df.empty:
        return _error("empty_dataframe", "The DataFrame has no rows.")
    return None


def _normalise_component(name: Any) -> str:
    text = str(name).strip()
    return text.upper() if text else text


def _resolve_components(
    df: pd.DataFrame, components: Sequence[str]
) -> tuple[dict[str, str], list[str]]:
    """Work out how each requested component can be obtained.

    Returns ``(mapping, unavailable)`` where mapping is
    ``{component: "present" | "derived"}``. Derived components are computed on
    demand from X/Y/Z rather than treated as an error, so a caller can ask for
    ``"H"`` straight after loading a raw loader DataFrame.
    """
    mapping: dict[str, str] = {}
    unavailable: list[str] = []
    for raw in components:
        name = _normalise_component(raw)
        if name in df.columns:
            mapping[name] = "present"
        elif name in DERIVED and all(c in df.columns for c in REQUIRED_FOR_DERIVED):
            mapping[name] = "derived"
        else:
            unavailable.append(name)
    return mapping, unavailable


def _with_derived(df: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    """Return ``df`` guaranteed to contain every name in ``names``."""
    needed = [n for n in names if n not in df.columns]
    if not needed:
        return df
    if all(c in df.columns for c in REQUIRED_FOR_DERIVED):
        return calculate_derived_components(df)
    raise KeyError(f"cannot derive {needed}: missing {list(REQUIRED_FOR_DERIVED)}")


# --------------------------------------------------------------------------- #
# 1. derived components
# --------------------------------------------------------------------------- #
def calculate_derived_components(
    df: pd.DataFrame, components: Sequence[str] = DERIVED
) -> pd.DataFrame | dict[str, Any]:
    """Add horizontal field ``H``, declination ``D`` and inclination ``I``.

    Parameters
    ----------
    df:
        DataFrame with ``timestamp`` and the Cartesian components ``X``, ``Y``,
        ``Z``. ``F`` is optional and is not used.
    components:
        Subset of ``('H', 'D', 'I')`` to compute. Anything already present is
        left untouched, so the function is safely re-runnable and idempotent.

    Returns
    -------
    pandas.DataFrame
        A **new** DataFrame (the input is never mutated, so a DataFrame still
        referenced by the loader cache stays pristine) with the requested
        columns appended. Row order and ``timestamp`` are preserved exactly.
        ``H`` in nT; ``D`` and ``I`` in degrees. ``NaN`` in any input propagates
        to every dependent output.
    dict
        On failure -- non-DataFrame input, empty frame, or missing X/Y/Z.

    Notes
    -----
    ``D = degrees(arctan2(Y, X))`` lands in ``(-180, 180]`` and ``I =
    degrees(arctan2(Z, H))`` in ``(-90, 90]``, both with no post-hoc
    normalisation needed. Where ``X == Y == 0`` the horizontal direction is
    undefined; ``arctan2(0, 0)`` yields ``0`` rather than ``NaN``, which keeps
    the column numeric. That case is flagged through
    ``result.attrs['degenerate_horizontal_rows']``.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem

    wanted = [_normalise_component(c) for c in components]
    unknown = [c for c in wanted if c not in DERIVED]
    if unknown:
        return _error(
            "unknown_component",
            f"{unknown} are not derivable components.",
            requested=wanted,
            available=list(DERIVED),
            hint=f"Derivable components are {list(DERIVED)}.",
        )
    if not wanted:
        return df.copy()

    missing = [c for c in REQUIRED_FOR_DERIVED if c not in df.columns]
    if missing:
        return _error(
            "missing_columns",
            f"Cannot compute {wanted} without the Cartesian components; "
            f"missing {missing}.",
            requested=wanted,
            available=[c for c in df.columns],
            missing=missing,
            hint="Load data with intermagnet_loader.fetch_observatory_data(), "
            "which always returns X, Y, Z, F.",
        )

    out = df.copy()
    x = pd.to_numeric(out["X"], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(out["Y"], errors="coerce").to_numpy(dtype="float64")
    z = pd.to_numeric(out["Z"], errors="coerce").to_numpy(dtype="float64")

    degenerate = int(np.count_nonzero((x == 0.0) & (y == 0.0) & np.isfinite(x)))

    if "H" in wanted:
        out["H"] = np.hypot(x, y)
    if "D" in wanted:
        out["D"] = np.degrees(np.arctan2(y, x))
    if "I" in wanted:
        h = np.hypot(x, y)
        out["I"] = np.degrees(np.arctan2(z, h))

    out.attrs.update(getattr(df, "attrs", {}))
    out.attrs["derived_components"] = [c for c in wanted if c in out.columns]
    if degenerate:
        out.attrs["degenerate_horizontal_rows"] = degenerate
    return out


# --------------------------------------------------------------------------- #
# 2. statistics
# --------------------------------------------------------------------------- #
def get_statistics(
    df: pd.DataFrame,
    components: Sequence[str] = ("X", "Y", "Z", "F", "H"),
    digits: int = 2,
) -> dict[str, Any] | dict[str, Any]:
    """Summarise one or more components.

    Parameters
    ----------
    df:
        DataFrame of magnetometer data. ``H``/``D``/``I`` are derived
        automatically when ``X``/``Y``/``Z`` are present, so the default
        component list works on a raw loader result.
    components:
        Component names to summarise; case-insensitive.
    digits:
        Decimal places every numeric statistic is rounded to (default 2).

    Returns
    -------
    dict
        ``{component: {"count", "missing", "coverage_pct", "min", "max",
        "range", "mean", "median", "std", "unit"}}`` or an error dict.

    Notes
    -----
    ``NaN`` samples are excluded from every statistic rather than counted as
    zero, and ``std`` is the sample standard deviation (``ddof=1``), so a
    single-sample series reports ``std = None`` instead of a misleading 0.

    ``mean``/``median`` of the angular components ``D`` and ``I`` are ordinary
    linear statistics. They are meaningful for a single day, but a series that
    wraps through +-180 degrees (a long interval near the poles, or a full year)
    has an undefined linear mean. Use a circular mean for those cases.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem

    requested = [_normalise_component(c) for c in components]
    if not requested:
        return _error("invalid_input", "components must not be empty.")
    if not isinstance(digits, int) or digits < 0:
        return _error("invalid_input", f"digits must be a non-negative int, got {digits!r}.")

    mapping, unavailable = _resolve_components(df, requested)
    if unavailable:
        return _error(
            "unknown_component",
            f"Component(s) {unavailable} are neither present in the DataFrame "
            f"nor derivable from X/Y/Z.",
            requested=requested,
            available=[c for c in df.columns],
            missing=unavailable,
            hint="Available columns: "
            + ", ".join(map(str, df.columns))
            + ". Known components: "
            + ", ".join(COMPONENT_UNITS)
            + ". Derive H/D/I with calculate_derived_components().",
        )

    working = _with_derived(df, mapping) if "derived" in mapping.values() else df

    stats: dict[str, Any] = {}
    for name in requested:
        series = pd.to_numeric(working[name], errors="coerce").dropna()
        total = int(series.shape[0])
        valid = int(series.shape[0])
        missing = int(working[name].isna().sum())
        entry: dict[str, Any] = {
            "count": valid,
            "missing": missing,
            "coverage_pct": _jsonable(100.0 * valid / len(working) if len(working) else 0.0, 2),
            "unit": COMPONENT_UNITS.get(name),
        }
        if valid == 0:
            entry.update({"min": None, "max": None, "range": None,
                          "mean": None, "median": None, "std": None})
            stats[name] = entry
            continue

        lo = float(series.min())
        hi = float(series.max())
        entry.update(
            {
                "min": _jsonable(lo, digits),
                "max": _jsonable(hi, digits),
                "range": _jsonable(hi - lo, digits),
                "mean": _jsonable(series.mean(), digits),
                "median": _jsonable(series.median(), digits),
                "std": _jsonable(series.std(ddof=1), digits),
            }
        )
        stats[name] = entry
        del total

    return stats


# --------------------------------------------------------------------------- #
# 3. anomaly detection
# --------------------------------------------------------------------------- #
def detect_anomalies(
    df: pd.DataFrame,
    component: str = "H",
    sigma_threshold: float = 3.0,
) -> pd.DataFrame | dict[str, Any]:
    """Flag samples whose value departs from the series mean by more than
    ``sigma_threshold`` standard deviations.

    Parameters
    ----------
    df:
        DataFrame of magnetometer data. ``H``/``D``/``I`` are derived on demand
        from ``X``/``Y``/``Z``.
    component:
        Component to test, default ``'H'`` (horizontal field). Case-insensitive.
    sigma_threshold:
        Number of standard deviations. Must be a positive number.

    Returns
    -------
    pandas.DataFrame
        Only the anomalous samples, in the input's row order, carrying the extra
        columns ``deviation`` (value minus mean, nT or deg) and ``z_score``.
        The **index is preserved** from the input frame, so
        ``result.loc[50]`` still refers to input row 50 and the caller can
        rejoin against the source data. Empty -- but with the same columns --
        when there are no anomalies.
    dict
        On failure.

    Notes
    -----
    ``NaN`` samples can never satisfy the threshold comparison and so are
    excluded automatically; they are *not* reported as anomalies. ``std`` is the
    sample standard deviation (``ddof=1``).

    **Interpretation caveat.** This is a plain outlier test over the whole
    series, which is the right tool for spotting spikes, dropouts and
    single-sample glitches. It is *not* a geomagnetic storm detector: ``H``
    carries a strong diurnal variation of tens of nT, so a 3-sigma threshold
    over a multi-day window will flag ordinary solar-cycle behaviour rather than
    a storm. For storm detection subtract a quiet-day reference (pass the
    deviation) or threshold the rate of change via :func:`compute_derivatives`.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem

    name = _normalise_component(component)
    if not name:
        return _error("invalid_input", "component must be a non-empty string.")
    try:
        threshold = float(sigma_threshold)
    except (TypeError, ValueError):
        return _error(
            "invalid_input",
            f"sigma_threshold must be a number, got {sigma_threshold!r}.",
        )
    if not np.isfinite(threshold) or threshold <= 0:
        return _error(
            "invalid_input",
            f"sigma_threshold must be a positive finite number, got {sigma_threshold!r}.",
        )

    mapping, unavailable = _resolve_components(df, [name])
    if unavailable:
        return _error(
            "unknown_component",
            f"Component {name!r} is neither present in the DataFrame nor "
            f"derivable from X/Y/Z.",
            requested=name,
            available=[c for c in df.columns],
            missing=[name],
            hint="Available columns: "
            + ", ".join(map(str, df.columns))
            + ". Known components: "
            + ", ".join(COMPONENT_UNITS),
        )

    working = _with_derived(df, mapping)
    values = pd.to_numeric(working[name], errors="coerce")

    valid = values.dropna()
    if valid.shape[0] < 2:
        return pd.DataFrame(
            columns=[*working.columns, "deviation", "z_score"]
        ).astype({"timestamp": values.dtype} if "timestamp" in working.columns else {})

    mean = float(valid.mean())
    std = float(valid.std(ddof=1))

    if std == 0.0 or not np.isfinite(std):
        flagged = pd.Series(False, index=working.index)
    else:
        z = (values - mean).abs() / std
        flagged = (z > threshold).fillna(False)

    anomalous = working.loc[flagged].copy()
    anomalous["deviation"] = (values.loc[flagged] - mean).astype("float64")
    anomalous["z_score"] = ((values.loc[flagged] - mean) / std).astype("float64")
    anomalous.attrs.update(getattr(df, "attrs", {}))
    anomalous.attrs["anomaly_detection"] = {
        "component": name,
        "sigma_threshold": threshold,
        "mean": _jsonable(mean, 4),
        "std": _jsonable(std, 4),
        "n_tested": int(valid.shape[0]),
        "n_flagged": int(len(anomalous)),
    }
    return anomalous


# --------------------------------------------------------------------------- #
# 4. derivatives (magnetic disturbance rate)
# --------------------------------------------------------------------------- #
def compute_derivatives(
    df: pd.DataFrame,
    components: Sequence[str] = ("H", "F"),
    time_column: str = "timestamp",
) -> pd.DataFrame | dict[str, Any]:
    """Add first time-derivatives, the standard measure of magnetic disturbance.

    Adds ``<C>_dt`` for each component in ``components``, giving the change per
    minute -- the quantity used for storm detection, and the input to Dst-style
    disturbance indices. ``H``/``D``/``I`` are derived on demand.

    With the default one-minute sampling the units are nT/min. For other
    cadences the result is the change per sampling interval, which the caller
    must convert (multiply by 1440 for nT/day). The first row is ``NaN`` because
    it has no predecessor.

    Physical reference: |dH/dt| above roughly 20-30 nT/min is the conventional
    threshold for a geomagnetic storm.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if time_column not in df.columns:
        return _error(
            "missing_columns",
            f"Time column {time_column!r} is not in the DataFrame.",
            available=[c for c in df.columns],
            missing=[time_column],
        )

    names = [_normalise_component(c) for c in components]
    mapping, unavailable = _resolve_components(df, names)
    if unavailable:
        return _error(
            "unknown_component",
            f"Component(s) {unavailable} are neither present nor derivable.",
            requested=names,
            available=[c for c in df.columns],
            missing=unavailable,
        )

    out = _with_derived(df, mapping).copy()
    ordered = pd.to_datetime(out[time_column], errors="coerce")
    step = ordered.diff().dt.total_seconds().div(60.0)

    for name in names:
        values = pd.to_numeric(out[name], errors="coerce")
        out[f"{name}_dt"] = values.diff().div(step)
        if name in _ANGULAR:
            pass

    out.attrs.update(getattr(df, "attrs", {}))
    out.attrs["derivative_components"] = [f"{n}_dt" for n in names]
    out.attrs["derivative_unit"] = "nT per sampling interval"
    return out


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _load_demo() -> tuple[pd.DataFrame, str]:
    """Prefer real cached magnetometer data, else synthesise a 2-day series."""
    cache = pd.DataFrame()
    source = "synthetic"
    parquet = sorted((__import__("pathlib").Path(__file__).resolve().parent / "cache")
                     .glob("*.parquet"))
    for path in parquet:
        try:
            candidate = pd.read_parquet(path)
        except Exception:
            continue
        if {"X", "Y", "Z"} <= set(candidate.columns) and len(candidate) > 100:
            cache = candidate
            source = f"cache:{path.name}"
            break

    if not cache.empty:
        return cache, source

    rng = np.random.default_rng(20250910)
    stamps = pd.date_range("2025-03-10", periods=2 * 1440, freq="1min")
    t = np.arange(stamps.size) / 1440.0
    storm = 120.0 * np.exp(-(((t - 0.55) / 0.03) ** 2))
    h = 18000.0 + 45.0 * np.sin(2 * np.pi * t) - storm + rng.normal(0, 2.0, stamps.size)
    x = h + 900.0 * np.sin(2 * np.pi * t + 0.4)
    y = -1500.0 + 700.0 * np.cos(2 * np.pi * t) - 0.55 * storm
    z = 57500.0 + 400.0 * np.sin(2 * np.pi * t + 1.1) - 1.8 * storm
    frame = pd.DataFrame(
        {"timestamp": stamps, "X": x, "Y": y, "Z": z,
         "F": np.sqrt(x**2 + y**2 + z**2)}
    )
    return frame, source


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 24)

    df, source = _load_demo()
    print("=" * 78)
    print(f"geomag_analyzer self-test -- data source: {source}")
    print("=" * 78)
    print(f"rows={len(df)}  columns={list(df.columns)}")
    print(df.head(3).to_string())

    print()
    print("=" * 78)
    print("STEP 1 - calculate_derived_components: H, D, I")
    print("=" * 78)
    derived = calculate_derived_components(df)
    if is_error(derived):
        print("FAILED:", derived)
        return 1
    print(derived[["timestamp", "X", "Y", "Z", "H", "D", "I", "F"]].head(5).to_string())

    h_calc = np.hypot(derived["X"].to_numpy(), derived["Y"].to_numpy())
    print(f"\nH matches hypot(X, Y)      : {np.allclose(derived['H'].to_numpy(), h_calc)}")
    d_calc = np.degrees(np.arctan2(derived["Y"].to_numpy(), derived["X"].to_numpy()))
    i_calc = np.degrees(np.arctan2(derived["Z"].to_numpy(), h_calc))
    print(f"D matches arctan2(Y, X)    : {np.allclose(derived['D'].to_numpy(), d_calc)}")
    print(f"I matches arctan2(Z, H)    : {np.allclose(derived['I'].to_numpy(), i_calc)}")
    f_from_hz = np.sqrt(h_calc**2 + derived["Z"].to_numpy() ** 2)
    print(f"sqrt(H^2+Z^2) matches F    : "
          f"{np.allclose(f_from_hz, derived['F'].to_numpy(), atol=1e-6)}")
    print(f"D range (deg)              : "
          f"[{derived['D'].min():.4f}, {derived['D'].max():.4f}]")
    print(f"I range (deg)              : "
          f"[{derived['I'].min():.4f}, {derived['I'].max():.4f}]")

    print()
    print("=" * 78)
    print("STEP 2 - NaN handling")
    print("=" * 78)
    gappy = df.copy()
    gappy.loc[10:15, ["X", "Y", "Z"]] = np.nan
    gappy_out = calculate_derived_components(gappy)
    print(gappy_out[["timestamp", "X", "Y", "Z", "H", "D", "I"]].iloc[9:17].to_string())
    nan_rows = gappy_out.loc[10:15, ["H", "D", "I"]].isna().all(axis=1)
    print(f"\nall 6 gapped rows fully NaN in H/D/I: {bool(nan_rows.all())} "
          f"({int(nan_rows.sum())}/6)")
    print(f"neighbour rows still numeric         : "
          f"{bool(gappy_out['H'].iloc[9] == calculate_derived_components(df)['H'].iloc[9])}")

    print()
    print("=" * 78)
    print("STEP 3 - get_statistics")
    print("=" * 78)
    stats = get_statistics(derived)
    import json
    print(json.dumps(stats, indent=2))
    print(f"\nJSON-serialisable (no bare NaN): "
          f"{'NaN' not in json.dumps(stats) and 'Infinity' not in json.dumps(stats)}")
    print(f"H auto-derived on a raw loader frame: "
          f"{'H' in get_statistics(df) and not is_error(get_statistics(df))}")

    print()
    print("=" * 78)
    print("STEP 4 - detect_anomalies")
    print("=" * 78)
    spikes = derived.copy()
    spikes.loc[500, "H"] += 900.0
    spikes.loc[1200, "H"] -= 750.0
    anomalies = detect_anomalies(spikes, component="H", sigma_threshold=3.0)
    print(f"injected 2 spikes -> detected {len(anomalies)}: "
          f"{list(anomalies['timestamp'].astype(str))}")
    print(anomalies[["timestamp", "H", "deviation", "z_score"]].to_string(index=False))
    print(f"\nclean series -> {len(detect_anomalies(derived))} anomalies (expected 0)")
    for thr in (1.0, 2.0, 3.0, 5.0):
        print(f"  sigma={thr:>3} -> {len(detect_anomalies(spikes, 'H', thr)):>5d} flags")

    print()
    print("=" * 78)
    print("STEP 5 - compute_derivatives (storm detection input)")
    print("=" * 78)
    deriv = compute_derivatives(derived, components=("H", "F"))
    print(f"columns added: {deriv.attrs['derivative_components']}")
    peak = deriv["H_dt"].abs().idxmax()
    print(f"max |dH/dt| = {deriv['H_dt'].abs().max():.2f} nT/min at "
          f"{deriv.loc[peak, 'timestamp']}")
    print(f"storm (>20 nT/min) samples: {int((deriv['H_dt'].abs() > 20).sum())}")

    print()
    print("=" * 78)
    print("STEP 6 - error contract")
    print("=" * 78)
    for label, out in (
        ("unknown component", get_statistics(derived, components=["Q"])),
        ("missing column", calculate_derived_components(df.drop(columns=["Y"]))),
        ("bad threshold", detect_anomalies(derived, sigma_threshold=-1)),
        ("not a DataFrame", get_statistics("nope")),
    ):
        print(f"{label:20s} -> {out['error']:18s} {out['message'][:72]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
