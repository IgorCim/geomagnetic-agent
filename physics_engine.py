"""Physical engine: the scientific layer on top of :mod:`geomag_math`.

Stages 1-2 answer "what do the numbers look like" (raw components, H/D/I,
statistics). This module answers "what do the numbers *mean* physically":

* the quiet Sq baseline a storm departs from,
* the Sq-removed deltas,
* the GIC proxy (the rate of change of the field),
* a simplified Dst proxy (the equatorial ring-current depression),
* the solar-wind coupling function ``E_y = V * B_z``,
* sudden commencement (SSC) detection.

It is deliberately thin and pure: every function takes a DataFrame and returns
either an enriched DataFrame (new physical columns, ready to plot and to feed
the next tool) or a JSON-safe error dict. No network, no globals, no model.
The single entry point the LLM sees is the ``apply_physics_formula`` meta-tool
in :mod:`agent_core`, which resolves a frame handle, calls one of these, and
stores the result back into the :class:`~agent_core.FrameStore`.

Conventions (identical to :mod:`geomag_math`)
---------------------------------------------
* ``X`` north, ``Y`` east, ``Z`` down, all in nT; ``H = hypot(X, Y)``.
* timestamps are UTC and INTERMAGNET minute means, so a first difference of
  ``H`` already *is* nT/min.
* ``NaN`` propagates and is never imputed.

Why the night **median** and not the mean
-----------------------------------------
The quiet baseline is the level a storm is measured against, so it must not be
dragged around by the very disturbance it is meant to reference. The night
window (00:00-04:00 UT, the quietest part of the geomagnetic day: minimal ring
current, electrojet far from the subsolar meridian) is mostly quiet but still
contains the occasional spike -- a substorm onset, a data glitch, an outlier
sample. A mean moves toward every spike; a median does not. On a window with a
handful of clean points and one 200 nT spike the mean shifts by tens of nT
while the median barely moves, so the median is the more stable reference.

Return-value contract (see :func:`is_error`)
--------------------------------------------
Every public function either returns the requested value or, on failure, a
``dict`` shaped like ``{"ok": False, "error": ..., "message": ...}``. All values
are JSON-serialisable (``NaN``/``inf`` become ``None``). Use :func:`is_error` to
branch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    # physics
    "extract_sq_baseline",
    "calculate_deltas",
    "calculate_gic_proxy",
    "calculate_dst_proxy",
    "calculate_coupling_function",
    "align_solar_wind",
    "detect_ssc",
    "calculate_burton_coupling",
    "calculate_akasofu_epsilon",
    "calculate_peak_dhdt",
    "detect_substorm_onset",
    "calculate_eej_index",
    # contract + constants
    "is_error",
    "SSC_THRESHOLD_NT_PER_MIN",
    "SSC_MIN_SEPARATION_MIN",
    "SSC_STEP_WINDOW_MIN",
    "DEFAULT_SQ_METHOD",
    "BURTON_A",
    "BURTON_TAU_HOURS",
    "AKASOFU_L0_RE",
    "MU_0_H_PER_M",
    "PEAK_DHDT_WINDOW_SECONDS",
    "SUBSTORM_DROP_NT_PER_MIN",
    "SUBSTORM_MIN_DURATION_MIN",
]

#: Default hour window (UTC) used as the geomagnetic night for the Sq baseline.
#: Same window as :data:`geomag_math.DEFAULT_NIGHT_HOURS`, repeated here so this
#: module stands alone.
_NIGHT_HOURS: tuple[int, int] = (0, 4)

#: Quiet-time estimator used by :func:`extract_sq_baseline` when none is given.
DEFAULT_SQ_METHOD = "night_median"

#: Default SSC threshold: a sudden commencement is a sharp, near-step increase
#: of the horizontal field, conventionally a jump of a few nT/min upwards.
SSC_THRESHOLD_NT_PER_MIN = 5.0

#: Two upward onsets closer than this belong to the same sudden commencement.
#: An SSC is a step, so a one-minute step spans two or three samples above the
#: threshold; without a separation window a single step was reported as two or
#: three "events" and the count was meaningless.
SSC_MIN_SEPARATION_MIN = 30.0

#: Window used to measure the sustained step of an impulse, in minutes. A
#: one-sample spike can have a huge dH/dt but no 5-minute step; this amplitude
#: separates a real sudden impulse from a plotting artifact and lets the payload
#: report "8 candidates, max step 62 nT" instead of calling every onset an SSC.
SSC_STEP_WINDOW_MIN = 5.0

#: Burton coupling constants (Burton, McPherron & Russell 1975): the ring-current
#: injection efficiency ``a`` (nT per unit of the coupling function, per hour) and
#: the ring-current decay time ``tau`` in hours.
BURTON_A = 0.005
BURTON_TAU_HOURS = 8.0

#: Akasofu's ``l0`` length scale in Earth radii, used by the epsilon parameter.
AKASOFU_L0_RE = 7.0

#: Vacuum permeability, H/m. The SI epsilon is a Poynting flux times an area:
#: ``V * B^2 * l0^2`` is not a power until it is divided by mu0. Omitting it
#: made the result low by exactly ``1/mu0`` = 7.958e5 (89.5 kW instead of the
#: 71.2 GW a 450 km/s, 10 nT wind actually delivers).
MU_0_H_PER_M = 4e-7 * np.pi

#: Default rolling window (seconds) for the peak dH/dt GIC metric. INTERMAGNET
#: minute means make 10 samples a ten-minute window.
PEAK_DHDT_WINDOW_SECONDS = 10

#: Substorm onset detection thresholds: an SML drop steeper than this for at least
#: this many minutes marks the expansion phase.
SUBSTORM_DROP_NT_PER_MIN = 50.0
SUBSTORM_MIN_DURATION_MIN = 3

#: One Earth radius in metres, for the SI epsilon value.
_RE_METRES = 6.371e6

#: One erg/s in watts (1 erg = 1e-7 J).
_ERG_PER_S_IN_W = 1e-7


def is_error(result: Any) -> bool:
    """True when a call failed. Lets an LLM branch without isinstance checks."""
    return isinstance(result, dict) and result.get("ok") is False


def _error(
    code: str,
    message: str,
    *,
    hint: str | None = None,
    missing: list[str] | None = None,
    available: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build the stable, JSON-serialisable error payload for LLM callers."""
    payload: dict[str, Any] = {
        "ok": False,
        "error": code,
        "message": message,
        "missing": missing,
        "available": available,
    }
    if hint:
        payload["hint"] = hint
    payload.update(extra)
    return payload


def _jsonable(value: Any, digits: int | None = None) -> Any:
    """Convert numpy scalars to plain Python, NaN/inf to None, optionally round."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def _validate_frame(df: Any) -> dict[str, Any] | None:
    if not isinstance(df, pd.DataFrame):
        return _error("invalid_input", f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        return _error("empty_dataframe", "The DataFrame has no rows.")
    return None


def _require_columns(df: pd.DataFrame, needed: tuple[str, ...]) -> dict[str, Any] | None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return _error(
            "missing_columns",
            f"Cannot compute without {list(needed)}; missing {missing}.",
            missing=missing,
            available=[str(c) for c in df.columns],
        )
    return None


def _hour_of_day(df: pd.DataFrame) -> pd.Series | None:
    if "timestamp" not in df.columns:
        return None
    stamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if stamps.isna().all():
        return None
    return stamps.dt.hour


def _horizontal(df: pd.DataFrame) -> np.ndarray:
    """``H`` series as float64: the stored ``H`` column, else ``hypot(X, Y)``."""
    if "H" in df.columns:
        return pd.to_numeric(df["H"], errors="coerce").to_numpy(dtype="float64")
    x = pd.to_numeric(df["X"], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(df["Y"], errors="coerce").to_numpy(dtype="float64")
    return np.hypot(x, y)


def _first_difference(h: np.ndarray) -> np.ndarray:
    """``h[i] - h[i-1]`` with the first row left NaN, not back-filled with zero."""
    out = np.full(h.shape, np.nan, dtype="float64")
    out[1:] = np.diff(h, n=1)
    return out


# --------------------------------------------------------------------------- #
# 1. quiet Sq baseline
# --------------------------------------------------------------------------- #
def extract_sq_baseline(
    df: pd.DataFrame,
    method: str = DEFAULT_SQ_METHOD,
    night_hours: tuple[int, int] = _NIGHT_HOURS,
) -> pd.DataFrame | dict[str, Any]:
    """Extract the quiet Sq baseline from the night window.

    Adds the baseline columns ``X_sq``, ``Y_sq``, ``Z_sq``, ``H_sq`` and
    ``F_sq`` to a copy of ``df`` (each constant across the frame), so
    :func:`calculate_deltas` can subtract them row by row. The baseline is
    derived from the Cartesian components only; ``H_sq`` and ``F_sq`` are built
    with ``hypot`` for the same numerical reason as :mod:`geomag_math`.

    ``method`` is ``"night_median"`` (default) or ``"night_mean"``. The median is
    the default because it is robust to the outlier samples a mostly-quiet night
    still contains (see the module docstring). ``night_hours`` defaults to the
    geomagnetic night 00:00-04:00 UT and wraps past midnight like
    :func:`geomag_math.get_nighttime_baseline`.

    Returns the enriched DataFrame, or an error dict when there are no night
    samples or the frame has no usable timestamp.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    problem = _require_columns(df, ("X", "Y", "Z"))
    if problem is not None:
        return problem

    method_norm = str(method).strip().lower()
    if method_norm not in ("night_median", "night_mean"):
        return _error(
            "unknown_method",
            f"method={method!r} is not supported.",
            available=["night_median", "night_mean"],
            hint="'night_median' is the robust default; 'night_mean' matches a classic mean.",
        )

    hours = _hour_of_day(df)
    if hours is None:
        return _error(
            "missing_timestamp",
            "A nighttime Sq baseline needs a 'timestamp' column to know the hour.",
            available=[str(c) for c in df.columns],
            hint="Loader frames always carry a UTC 'timestamp' column.",
        )

    try:
        start_hour, end_hour = int(night_hours[0]), int(night_hours[1])
    except (TypeError, ValueError, IndexError):
        return _error("invalid_input", f"night_hours={night_hours!r} must be a pair of hours.")

    if start_hour <= end_hour:
        mask = (hours >= start_hour) & (hours < end_hour)
    else:  # wraps past midnight, e.g. (22, 6)
        mask = (hours >= start_hour) | (hours < end_hour)
    if not mask.any():
        return _error(
            "no_night_data",
            f"No samples fall inside the night window "
            f"({start_hour:02d}:00-{end_hour:02d}:00 UT).",
            hint="Check the data coverage, or pass a window that matches it.",
        )

    def _quiet(column: str) -> float:
        values = pd.to_numeric(df[column], errors="coerce")[mask].dropna()
        if values.empty:
            return float("nan")
        return float(values.median() if method_norm == "night_median" else values.mean())

    x_sq = _quiet("X")
    y_sq = _quiet("Y")
    z_sq = _quiet("Z")
    h_sq = float(np.hypot(x_sq, y_sq)) if np.isfinite(x_sq) and np.isfinite(y_sq) else float("nan")
    f_sq = float(np.hypot(h_sq, z_sq)) if np.isfinite(h_sq) and np.isfinite(z_sq) else float("nan")

    out = df.copy()
    out["X_sq"] = x_sq
    out["Y_sq"] = y_sq
    out["Z_sq"] = z_sq
    out["H_sq"] = h_sq
    out["F_sq"] = f_sq

    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["sq_method"] = method_norm
    out.attrs["sq_night_hours_ut"] = [start_hour, end_hour]
    out.attrs["sq_baseline"] = {
        "X": _jsonable(x_sq, 2),
        "Y": _jsonable(y_sq, 2),
        "Z": _jsonable(z_sq, 2),
        "H": _jsonable(h_sq, 2),
        "F": _jsonable(f_sq, 2),
    }
    return out


# --------------------------------------------------------------------------- #
# 2. Sq-removed deltas
# --------------------------------------------------------------------------- #
def calculate_deltas(df: pd.DataFrame) -> pd.DataFrame | dict[str, Any]:
    """Subtract the Sq baseline and report the vector deltas.

    Adds ``dX``, ``dY``, ``dZ`` = component minus its ``*_sq`` level, the
    horizontal delta ``dH = hypot(dX, dY)`` and the total delta
    ``F_delta = hypot(dH, dZ)``. When the frame carries no ``*_sq`` columns
    (it is a raw fetch), the quiet baseline is materialised first with
    :func:`extract_sq_baseline`, so the deltas are physical rather than the raw
    values dressed up as differences.

    Returns the enriched DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    problem = _require_columns(df, ("X", "Y", "Z"))
    if problem is not None:
        return problem

    out = df.copy()
    if not {"X_sq", "Y_sq", "Z_sq"} <= set(out.columns):
        baselined = extract_sq_baseline(out)
        if not is_error(baselined):
            out = baselined
            out.attrs = dict(getattr(df, "attrs", {}))

    x = pd.to_numeric(out["X"], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(out["Y"], errors="coerce").to_numpy(dtype="float64")
    z = pd.to_numeric(out["Z"], errors="coerce").to_numpy(dtype="float64")

    def _level(column: str) -> np.ndarray:
        if column not in out.columns:
            return np.zeros_like(x)
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype="float64")
        return np.where(np.isfinite(values), values, 0.0)

    dx = x - _level("X_sq")
    dy = y - _level("Y_sq")
    dz = z - _level("Z_sq")

    out["dX"] = dx
    out["dY"] = dy
    out["dZ"] = dz
    out["dH"] = np.hypot(dx, dy)
    out["F_delta"] = np.hypot(np.hypot(dx, dy), dz)
    return out


# --------------------------------------------------------------------------- #
# 3. GIC proxy (dH/dt)
# --------------------------------------------------------------------------- #
def calculate_gic_proxy(df: pd.DataFrame) -> pd.DataFrame | dict[str, Any]:
    """Rate of change of the horizontal field, the standard GIC proxy.

    Geomagnetically induced currents in a power grid are driven by the
    horizontal-field *rate of change*, so the proxy is ``dH/dt`` in nT/min. For
    INTERMAGNET minute means the sampling interval is one minute and a first
    difference is already nT/min. The first sample is NaN by definition.

    Adds the ``dH_dt`` column and records the peak absolute value in
    ``attrs['dH_dt_peak_abs_nT_per_min']``. Returns the enriched DataFrame, or an
    error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if "H" not in df.columns:
        problem = _require_columns(df, ("X", "Y"))
        if problem is not None:
            return problem

    out = df.copy()
    dh_dt = _first_difference(_horizontal(out))
    out["dH_dt"] = dh_dt
    peak = np.nanmax(np.abs(dh_dt)) if np.isfinite(dh_dt).any() else float("nan")

    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["dH_dt_peak_abs_nT_per_min"] = _jsonable(peak, 2)
    return out


# --------------------------------------------------------------------------- #
# 4. simplified Dst proxy
# --------------------------------------------------------------------------- #
def calculate_dst_proxy(
    df: pd.DataFrame, latitude: float | None = None
) -> pd.DataFrame | dict[str, Any]:
    """Simplified Dst proxy from the Sq-removed horizontal field.

    The real Dst index is the equatorward depression of the horizontal field
    caused by the ring current; the standard first-order construction is to
    remove Sq and then reduce the mid-latitude depression to the equator. This
    proxy does exactly that:

    * ``dH = H - H_sq`` (Sq removal),
    * ``dst_proxy = dH * cos(latitude)`` (equatorial reduction; the factor is 1
      at the magnetic equator, where the ring-current effect is largest, and
      shrinks with latitude).

    A storm's rapid ``H`` decrease therefore yields a negative value, matching
    the sign of the real index. This is an *approximation* -- the usable 0-100
    nT index additionally needs the quiet-time reference and the magnetometer's
    magnetic latitude -- and is labelled a proxy throughout.

    Missing baselines are materialised with :func:`extract_sq_baseline`, so a raw
    frame is accepted directly. Returns the enriched DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if "H" not in df.columns:
        problem = _require_columns(df, ("X", "Y"))
        if problem is not None:
            return problem

    out = df.copy()
    if "H_sq" not in out.columns and not {"X_sq", "Y_sq"} <= set(out.columns):
        baselined = extract_sq_baseline(out)
        if not is_error(baselined):
            out = baselined
            out.attrs = dict(getattr(df, "attrs", {}))

    h = _horizontal(out)

    def _level(column: str) -> np.ndarray | None:
        if column not in out.columns:
            return None
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype="float64")
        return values if np.isfinite(values).any() else None

    h_sq = _level("H_sq")
    if h_sq is None:
        x_sq = _level("X_sq")
        y_sq = _level("Y_sq")
        if x_sq is not None or y_sq is not None:
            x_values = x_sq if x_sq is not None else np.zeros_like(h)
            y_values = y_sq if y_sq is not None else np.zeros_like(h)
            h_sq = np.hypot(x_values, y_values)
    if h_sq is None:
        h_sq = np.zeros_like(h)
    h_sq = np.where(np.isfinite(h_sq), h_sq, 0.0)

    lat = float(latitude) if latitude is not None else 0.0
    factor = float(np.cos(np.radians(lat))) if np.isfinite(lat) else 1.0

    out["dst_proxy"] = (h - h_sq) * factor
    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["dst_proxy_latitude_used"] = _jsonable(lat, 2)
    return out


# --------------------------------------------------------------------------- #
# 5. solar-wind coupling function
# --------------------------------------------------------------------------- #

def _solar_wind_series(
    solar_wind: Any, df: pd.DataFrame, *names: str
) -> "pd.Series | None":
    """A solar-wind column as a float64 Series aligned to ``df``'s length.

    ``solar_wind`` is the aligned frame the caller produced with
    :func:`align_solar_wind` (or any frame carrying the column). A scalar column
    is broadcast to ``len(df)``; a vector must already match ``df`` row for row,
    which is exactly what alignment guarantees. Returns ``None`` when the column
    is absent, so the caller can name what is missing.
    """
    if solar_wind is None:
        return None
    column = _column_ci(solar_wind, *names) if isinstance(solar_wind, pd.DataFrame) else None
    if column is None:
        return None
    values = pd.to_numeric(solar_wind[column], errors="coerce").to_numpy(dtype="float64")
    if values.size == 1:
        values = np.full(len(df), values[0], dtype="float64")
    return pd.Series(values)


def _column_ci(df: pd.DataFrame, *names: str) -> str | None:
    """Return the first real column matching *names* case-insensitively."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        hit = lookup.get(name.lower())
        if hit is not None:
            return hit
    return None


def _coverage(stamps: pd.Series) -> dict[str, Any] | None:
    """UTC first/last valid timestamp, or ``None`` when there are none.

    An alignment failure must say *which* windows were compared; "no coverage"
    alone sent the model back to the same date it had already fetched.
    """
    valid = pd.to_datetime(stamps, errors="coerce", utc=True).dropna()
    if valid.empty:
        return None
    return {
        "start": valid.min().isoformat(),
        "end": valid.max().isoformat(),
        "n_samples": int(valid.size),
    }


def _finite_count(series: pd.Series) -> int:
    """How many real (non-NaN, non-inf) numbers a column carries."""
    return int(np.isfinite(series.to_numpy(dtype="float64")).sum())


def align_solar_wind(
    observatory_df: pd.DataFrame,
    solar_wind_df: pd.DataFrame,
    *,
    lag_minutes: float = 0.0,
    limit: int = 30,
) -> pd.DataFrame | dict[str, Any]:
    """Resample the solar wind onto the observatory's own minute grid.

    ``E_y = V * B_z`` only means something when the L1 sample and the ground
    sample describe the same instant, and the two feeds rarely share a clock:
    OMNI is bow-shock shifted, the magnetometer is not, and either can have
    instrument gaps. This function:

    * takes the observatory ``timestamp`` axis as the output axis (the frame you
      already want to enrich), so the result has exactly one V/Bz pair per
      observatory row;
    * shifts the solar-wind stamps by ``lag_minutes`` before aligning, which is
      where a bow-shock -> station propagation delay is applied as a first-order
      correction (add the delay to move L1 data later in time);
    * ``interpolate``s linearly *in time* across the union of both clocks with
      ``limit`` consecutive samples (30 by default = half an hour of 1-minute
      OMNI), so short dropouts are bridged and long outages stay ``NaN`` instead
      of being invented;
    * never extrapolates past the edges of the solar-wind coverage
      (``limit_area='inside'``).

    Inputs are matched by column name case-insensitively (``V``/``speed``,
    ``Bz``/``bz_gsm``), duplicate solar-wind stamps keep the last sample.

    Returns the observatory frame plus ``V``, ``Bz`` and ``Bz_gsm`` columns, with
    the fill counts in ``attrs['solar_wind_alignment']``, or an error dict.
    """
    problem = _validate_frame(observatory_df)
    if problem is not None:
        return problem
    problem = _validate_frame(solar_wind_df)
    if problem is not None:
        return problem

    obs_ts_col = _column_ci(observatory_df, "timestamp", "time", "datetime")
    if obs_ts_col is None:
        return _error(
            "missing_timestamp",
            "The observatory frame needs a 'timestamp' column to align the solar wind.",
            available=[str(c) for c in observatory_df.columns],
        )
    sw_ts_col = _column_ci(solar_wind_df, "timestamp", "time", "datetime")
    if sw_ts_col is None:
        return _error(
            "missing_timestamp",
            "The solar-wind frame needs a 'timestamp' column.",
            available=[str(c) for c in solar_wind_df.columns],
        )

    v_col = _column_ci(solar_wind_df, "V", "V_sw", "Vx", "speed", "solar_wind_speed")
    bz_col = _column_ci(solar_wind_df, "Bz", "Bz_gsm", "BzGSM", "B_z", "bz_gsm", "B_z_gsm")
    if v_col is None or bz_col is None:
        return _error(
            "missing_columns",
            "Solar-wind alignment needs a speed column 'V' and an IMF column 'Bz'.",
            missing=[name for name, col in (("V", v_col), ("Bz", bz_col)) if col is None],
            available=[str(c) for c in solar_wind_df.columns],
        )

    obs_ts = pd.to_datetime(observatory_df[obs_ts_col], errors="coerce", utc=True)
    sw_ts = pd.to_datetime(solar_wind_df[sw_ts_col], errors="coerce", utc=True)
    obs_valid = obs_ts.dropna()
    if obs_valid.empty:
        return _error(
            "missing_timestamp",
            "No valid timestamps in the observatory frame.",
            hint="Loader frames carry UTC timestamps in 'timestamp'.",
        )

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return _error("invalid_input", f"limit={limit!r} must be an integer number of samples.")
    if limit < 0:
        return _error("invalid_input", f"limit={limit!r} must be non-negative.")
    try:
        lag = pd.Timedelta(minutes=float(lag_minutes))
    except (TypeError, ValueError):
        return _error("invalid_input", f"lag_minutes={lag_minutes!r} must be a number.")
    sw_ts = sw_ts + lag

    obs_coverage = _coverage(obs_ts)
    sw_coverage = _coverage(sw_ts)
    overlap: dict[str, str] | None = None
    if obs_coverage is not None and sw_coverage is not None:
        overlap_start = max(
            pd.Timestamp(obs_coverage["start"]),
            pd.Timestamp(sw_coverage["start"]),
        )
        overlap_end = min(
            pd.Timestamp(obs_coverage["end"]),
            pd.Timestamp(sw_coverage["end"]),
        )
        if overlap_start <= overlap_end:
            overlap = {
                "start": overlap_start.isoformat(),
                "end": overlap_end.isoformat(),
            }

    def _one(col: str) -> pd.Series:
        values = pd.to_numeric(solar_wind_df[col], errors="coerce")
        series = pd.Series(values.to_numpy(dtype="float64"), index=pd.DatetimeIndex(sw_ts))
        series = series[series.index.notna()]
        # Collapse duplicate stamps so the reindex target stays unique.
        return series.groupby(level=0).last() if not series.empty else series

    v_series = _one(v_col)
    bz_series = _one(bz_col)
    if v_series.empty or bz_series.empty:
        return _error(
            "solar_wind_no_finite_values",
            "The solar-wind frame has no valid timestamps after applying the lag; "
            "V/Bz cannot be aligned.",
            reason="no_valid_timestamps",
            observatory_coverage=obs_coverage,
            solar_wind_coverage=sw_coverage,
            lag_minutes=_jsonable(float(lag_minutes), 1),
        )

    v_source_finite = _finite_count(v_series)
    bz_source_finite = _finite_count(bz_series)
    if v_source_finite == 0 or bz_source_finite == 0:
        columns = {
            "V": {
                "source_column": str(v_col),
                "source_finite": v_source_finite,
                "source_rows": int(len(v_series)),
            },
            "Bz": {
                "source_column": str(bz_col),
                "source_finite": bz_source_finite,
                "source_rows": int(len(bz_series)),
            },
        }
        empty = [name for name, info in columns.items() if info["source_finite"] == 0]
        return _error(
            "solar_wind_no_finite_values",
            f"The solar-wind frame has columns V and Bz, but the required "
            f"column(s) {', '.join(empty)} have no finite values "
            f"({v_source_finite} finite V, {bz_source_finite} finite Bz). "
            "Nothing can be aligned.",
            empty=empty,
            columns=columns,
            observatory_coverage=obs_coverage,
            solar_wind_coverage=sw_coverage,
            overlap=overlap,
            lag_minutes=_jsonable(float(lag_minutes), 1),
            hint=(
                "The wind frame was fetched but its V/Bz samples are all NaN. "
                "Re-fetch the solar wind for this date and check the loader output."
            ),
        )

    obs_index_unique = pd.DatetimeIndex(sorted(set(obs_valid)))

    def _resample(series: pd.Series) -> np.ndarray:
        union = obs_index_unique.union(series.index).sort_values()
        filled = (
            series.reindex(union)
            .interpolate(method="time", limit=limit, limit_area="inside")
        )
        # Reindex back onto the observatory's own axis (order and duplicates kept).
        return filled.reindex(pd.DatetimeIndex(obs_ts)).to_numpy(dtype="float64")

    v_aligned = _resample(v_series)
    bz_aligned = _resample(bz_series)
    v_aligned_finite = int(np.isfinite(v_aligned).sum())
    bz_aligned_finite = int(np.isfinite(bz_aligned).sum())
    if v_aligned_finite == 0 or bz_aligned_finite == 0:
        columns = {
            "V": {
                "source_column": str(v_col),
                "source_finite": v_source_finite,
                "aligned_finite": v_aligned_finite,
                "n_observatory": int(len(observatory_df)),
            },
            "Bz": {
                "source_column": str(bz_col),
                "source_finite": bz_source_finite,
                "aligned_finite": bz_aligned_finite,
                "n_observatory": int(len(observatory_df)),
            },
        }
        empty = [
            name for name, info in columns.items() if info["aligned_finite"] == 0
        ]
        if overlap is None:
            return _error(
                "solar_wind_no_time_overlap",
                "The observatory frame and the solar-wind frame do not overlap in "
                f"time: station {obs_coverage['start'] if obs_coverage else '?'} to "
                f"{obs_coverage['end'] if obs_coverage else '?'}, wind "
                f"{sw_coverage['start'] if sw_coverage else '?'} to "
                f"{sw_coverage['end'] if sw_coverage else '?'} "
                f"(lag {float(lag_minutes):g} min). V/Bz cannot be aligned.",
                observatory_coverage=obs_coverage,
                solar_wind_coverage=sw_coverage,
                overlap=None,
                lag_minutes=_jsonable(float(lag_minutes), 1),
                hint=(
                    "Fetch solar wind for the station's UTC date (for example "
                    "fetch_solar_wind(start_date='2024-09-12', "
                    "end_date='2024-09-12')), or use a station frame inside the "
                    "wind coverage."
                ),
            )
        return _error(
            "solar_wind_no_aligned_values",
            "The windows overlap "
            f"({overlap['start']} to {overlap['end']}) and the source has finite "
            "V/Bz, but no observatory row received a finite value "
            f"({v_aligned_finite} finite V, {bz_aligned_finite} finite Bz of "
            f"{len(observatory_df)} rows).",
            empty=empty,
            columns=columns,
            observatory_coverage=obs_coverage,
            solar_wind_coverage=sw_coverage,
            overlap=overlap,
            lag_minutes=_jsonable(float(lag_minutes), 1),
            hint=(
                "The wind timestamps fall between the station samples; use a "
                "station frame whose minutes lie inside the wind coverage."
            ),
        )

    out = observatory_df.copy()
    out["V"] = v_aligned
    out["Bz"] = bz_aligned
    out["Bz_gsm"] = bz_aligned

    # Np/Bx/By are optional, but Burton (Np) and Akasofu (Bx, By) need them: if
    # the solar-wind frame carries them, put them on the same aligned clock so no
    # formula ever has to touch the unaligned source frame.
    aligned_extra: dict[str, np.ndarray] = {}
    for target, aliases in (
        ("Np", ("Np", "n_p", "density", "n")),
        ("Bx", ("Bx", "B_x", "Bx_gsm", "BxGSM", "bx_gsm")),
        ("By", ("By", "B_y", "By_gsm", "ByGSM", "by_gsm")),
    ):
        source = _column_ci(solar_wind_df, *aliases)
        if source is None:
            continue
        extra_series = _one(source)
        if extra_series.empty:
            continue
        aligned_extra[target] = _resample(extra_series)
        out[target] = aligned_extra[target]

    out.attrs = dict(getattr(observatory_df, "attrs", {}))
    out.attrs["solar_wind_alignment"] = {
        "lag_minutes": _jsonable(float(lag_minutes), 1),
        "interp_limit_samples": limit,
        "n_observatory": int(len(observatory_df)),
        "n_solar_wind": int(len(solar_wind_df)),
        "n_finite_V": int(np.isfinite(v_aligned).sum()),
        "n_finite_Bz": int(np.isfinite(bz_aligned).sum()),
        "observatory_coverage": obs_coverage,
        "solar_wind_coverage": sw_coverage,
        "overlap": overlap,
        "aligned_extra_columns": sorted(aligned_extra),
        "source": "OMNI (L1)",
    }
    return out


def calculate_coupling_function(
    df: pd.DataFrame, solar_wind: Any = None
) -> pd.DataFrame | dict[str, Any]:
    """Solar-wind coupling function ``E_y = V * B_z``.

    The interplanetary electric field that drives geomagnetic activity is
    ``E_y = -V * B_z``; with the common sign convention (``B_z`` southward is
    negative) a southward field gives a positive coupling, so this module
    reports ``E_y = V * B_z`` and lets the caller read the sign. Units are the
    conventional proxied mV/m when ``V`` is in km/s and ``B_z`` in nT.

    This is the one function that cannot be answered from the magnetometer frame
    alone: it needs solar-wind data (L1 / OMNIWeb). The caller passes the aligned
    solar-wind frame through ``solar_wind`` -- typically the output of
    :func:`align_solar_wind`, whose ``V``/``Bz`` columns already sit on the
    observatory's own minute grid. Without it the function returns
    ``missing_solar_wind`` rather than inventing a number.

    ``V`` and ``Bz`` are read from the frame by name (case-insensitive, with the
    usual aliases: ``V``/``V_sw``/``speed``, ``Bz``/``Bz_gsm``/``B_z``). A
    single-row (scalar) frame is broadcast to every observatory row, so the
    classic "one number for the whole frame" behaviour is preserved.

    Returns the enriched DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem

    if solar_wind is None:
        return _error(
            "missing_solar_wind",
            "Ey requires solar-wind data: E_y = V * B_z.",
            hint=(
                "Pass 'solar_wind_handle' with a solar-wind frame handle "
                "(e.g. 'sw:omni:2024-09-12'); the system aligns it automatically."
            ),
        )
    if not isinstance(solar_wind, pd.DataFrame):
        return _error(
            "invalid_solar_wind",
            f"solar_wind must be an aligned solar-wind DataFrame, got {type(solar_wind).__name__}.",
        )

    v_series = _solar_wind_series(solar_wind, df, "V", "V_sw", "Vx", "speed", "solar_wind_speed")
    bz_series = _solar_wind_series(solar_wind, df, "Bz", "Bz_gsm", "BzGSM", "B_z", "bz_gsm", "B_z_gsm")
    v_finite = None if v_series is None else _finite_count(v_series)
    bz_finite = None if bz_series is None else _finite_count(bz_series)
    missing = [
        name for name, count in (("V", v_finite), ("Bz", bz_finite)) if count is None
    ]
    empty = [
        name for name, count in (("V", v_finite), ("Bz", bz_finite)) if count == 0
    ]
    if missing or empty:
        details: list[str] = []
        if missing:
            details.append(f"missing columns {missing}")
        if empty:
            details.append(f"columns with no finite values {empty}")
        return _error(
            "invalid_solar_wind" if missing else "solar_wind_no_finite_values",
            "Solar-wind coupling needs finite V and Bz: " + "; ".join(details) + ".",
            missing=missing or None,
            empty=empty or None,
            available=[str(c) for c in solar_wind.columns],
            hint=(
                "Fetch the solar wind with fetch_solar_wind; it carries V and Bz."
                if missing
                else "The wind frame has V/Bz columns but no finite values; "
                "re-fetch it for this date and check the loader output."
            ),
        )

    v_arr = v_series.to_numpy(dtype="float64")
    bz_arr = bz_series.to_numpy(dtype="float64")
    if v_arr.size != len(df) or bz_arr.size != len(df):
        return _error(
            "length_mismatch",
            f"Solar wind has {v_arr.size} rows but the frame has {len(df)}.",
            hint=(
                "The system aligns solar wind with align_solar_wind; pass the aligned "
                "frame, or a handle the handler can align."
            ),
        )

    out = df.copy()
    out.attrs = dict(getattr(df, "attrs", {}))

    # -- scalar path: a constant frame yields one E_y for the whole frame -----
    if np.unique(v_arr).size == 1 and np.unique(bz_arr).size == 1 and np.isfinite(v_arr[0]) and np.isfinite(bz_arr[0]):
        ey = float(v_arr[0]) * float(bz_arr[0])
        out["E_y"] = ey
        out.attrs["ey_coupling"] = {
            "mode": "scalar",
            "V": _jsonable(v_arr[0], 1),
            "Bz": _jsonable(bz_arr[0], 2),
            "E_y_mV_m": _jsonable(ey, 2),
        }
        return out

    # -- vector path: one E_y per observatory row -----------------------------
    ey = v_arr * bz_arr
    finite = np.isfinite(ey)
    if not finite.any():
        return _error(
            "solar_wind_no_aligned_values",
            "V and Bz carry source values, but no finite V/Bz pair survived "
            "alignment for this frame.",
            n_samples=int(len(df)),
            n_finite=0,
            hint=(
                "Check the date overlap and the interpolation limit; the solar-wind "
                "record may have no coverage over these observatory minutes."
            ),
        )

    out["E_y"] = ey
    peak = float(np.nanmax(np.abs(ey)))

    def _reduce(arr: Any, fn: Any) -> Any:
        arr = np.asarray(arr, dtype="float64")
        return _jsonable(fn(arr)) if np.isfinite(arr).any() else None

    out.attrs["ey_coupling"] = {
        "mode": "vector",
        "n_samples": int(len(df)),
        "n_finite": int(finite.sum()),
        "V_mean": _reduce(v_arr, np.nanmean),
        "Bz_mean": _reduce(bz_arr, np.nanmean),
        "Bz_min": _reduce(bz_arr, np.nanmin),
        "E_y_mV_m": _jsonable(peak, 2),
        "E_y_mV_m_peak_abs": _jsonable(peak, 2),
        "E_y_mV_m_mean": _reduce(ey, np.nanmean),
        "E_y_mV_m_min": _reduce(ey, np.nanmin),
        "E_y_mV_m_max": _reduce(ey, np.nanmax),
    }
    return out


# --------------------------------------------------------------------------- #
# 6. sudden commencement
# --------------------------------------------------------------------------- #
def detect_ssc(
    df: pd.DataFrame, threshold_nT_per_min: float = SSC_THRESHOLD_NT_PER_MIN
) -> pd.DataFrame | dict[str, Any]:
    """Detect a sudden commencement (SSC) from a sharp rise in ``H``.

    A sudden commencement is the near-step increase of the horizontal field when
    an interplanetary shock hits the magnetopause: a rapid rise over a minute or
    two, distinct from the slow diurnal variation. It is detected here as
    ``dH/dt`` exceeding ``threshold_nT_per_min`` (default 5 nT/min), on the
    upward side.

    ``dH/dt`` is normalised to nT per MINUTE from the real timestamp cadence, so
    a 5-minute station is not read as a five-times steeper jump. Onsets closer
    than :data:`SSC_MIN_SEPARATION_MIN` are clustered into one event and the
    steepest sample is kept: one SSC is a step, not every sample above the
    threshold.

    Adds ``ssc_flag`` (1 where a sample is part of an onset) and ``dH_dt``, and
    records ALL onset events in ``attrs['ssc_count']``. The payload list
    ``attrs['ssc_events']`` keeps the first ten and sets
    ``attrs['ssc_events_truncated']`` when there are more. Each event carries
    ``dH_dt_nT_per_min`` (the steepest sample) and ``step_nT`` (the difference of
    the 5-minute means after and before the onset).

    ``step_nT`` is the amplitude, and it is what makes the output honest: a
    one-sample spike has a huge rate but almost no step. The events are sudden
    impulses -- SI/SSC *candidates*. A true sudden commencement additionally
    needs a solar-wind shock or a multi-station coincidence, neither of which
    this single-frame detector can supply. Returns the enriched DataFrame, or an
    error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if "H" not in df.columns:
        problem = _require_columns(df, ("X", "Y"))
        if problem is not None:
            return problem

    try:
        threshold = float(threshold_nT_per_min)
    except (TypeError, ValueError):
        return _error(
            "invalid_input",
            f"threshold_nT_per_min={threshold_nT_per_min!r} is not a number.",
        )

    out = df.copy()
    h = _horizontal(out)
    dh = _first_difference(h)

    stamps = None
    if "timestamp" in df.columns:
        stamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # dH/dt must be per MINUTE, not per sample. Minute means make the two
    # identical; a 5-minute station read a 3 nT/min rise as 15 and tripped the
    # threshold. When there is no usable clock, assume one sample per minute.
    rate = dh
    if stamps is not None and len(stamps) > 1 and stamps.notna().all():
        dt_min = stamps.diff().dt.total_seconds().to_numpy() / 60.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(dt_min > 0, dh / dt_min, np.nan)

    onset = np.zeros(rate.shape, dtype="int64")
    onset[(rate > threshold) & np.isfinite(rate)] = 1

    out["dH_dt"] = rate
    out["ssc_flag"] = onset

    # One SSC is a step, not every sample above the threshold. Cluster onsets
    # that fall within SSC_MIN_SEPARATION_MIN of the previous one and keep the
    # steepest sample as the event: a two-sample step used to be two events.
    events: list[dict[str, Any]] = []
    last_time: pd.Timestamp | None = None
    last_index: int | None = None
    for i in np.where(onset == 1)[0]:
        stamp = None
        if stamps is not None and not pd.isna(stamps.iloc[i]):
            stamp = stamps.iloc[i]
        close = False
        if events:
            if last_time is not None and stamp is not None:
                close = (stamp - last_time) < pd.Timedelta(
                    minutes=SSC_MIN_SEPARATION_MIN
                )
            elif last_time is None and stamp is None and last_index is not None:
                close = (int(i) - last_index) <= 2
        if close:
            if float(rate[i]) > float(events[-1]["dH_dt_nT_per_min"]):
                events[-1].update({
                    "index": int(i),
                    "dH_dt_nT_per_min": _jsonable(rate[i], 2),
                })
                if stamp is not None:
                    events[-1]["timestamp"] = _jsonable(stamp)
            last_time = stamp if stamp is not None else last_time
            last_index = int(i)
            continue
        event: dict[str, Any] = {
            "index": int(i),
            "dH_dt_nT_per_min": _jsonable(rate[i], 2),
        }
        if stamp is not None:
            event["timestamp"] = _jsonable(stamp)
        events.append(event)
        last_time = stamp
        last_index = int(i)

    # A steep dH/dt can be a one-sample spike. The 5-minute pre/post means give
    # the sustained step -- the amplitude a human reads off the trace -- so the
    # payload can say "8 candidates, max step 62 nT" instead of calling every
    # onset a confirmed SSC.
    def _step_nT(event: dict[str, Any]) -> float | None:
        i = int(event["index"])
        stamp = None
        raw = event.get("timestamp")
        if raw is not None:
            try:
                stamp = pd.Timestamp(raw)
            except (TypeError, ValueError):
                stamp = None
        if stamp is not None and stamps is not None:
            try:
                window = pd.Timedelta(minutes=SSC_STEP_WINDOW_MIN)
                pre = h[
                    ((stamps >= stamp - window) & (stamps < stamp)).to_numpy()
                ]
                post = h[
                    ((stamps >= stamp) & (stamps <= stamp + window)).to_numpy()
                ]
            except (TypeError, ValueError):
                pre = h[max(0, i - 5):i]
                post = h[i:i + 5]
        else:
            pre = h[max(0, i - 5):i]
            post = h[i:i + 5]
        if len(pre) == 0 or len(post) == 0:
            return None
        return float(np.nanmean(post) - np.nanmean(pre))

    for event in events:
        event["step_nT"] = _jsonable(_step_nT(event), 1)

    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["ssc_events"] = events[:10]
    out.attrs["ssc_count"] = len(events)
    out.attrs["ssc_events_truncated"] = len(events) > 10
    out.attrs["ssc_threshold_nT_per_min"] = _jsonable(threshold, 2)
    out.attrs["ssc_min_separation_min"] = SSC_MIN_SEPARATION_MIN
    out.attrs["ssc_step_window_min"] = SSC_STEP_WINDOW_MIN
    return out


# --------------------------------------------------------------------------- #
# 7. Burton coupling function (Dst prediction)
# --------------------------------------------------------------------------- #

def calculate_burton_coupling(
    df: pd.DataFrame,
    solar_wind: Any = None,
    indices: Any = None,
) -> pd.DataFrame | dict[str, Any]:
    """Burton et al. (1975) ring-current coupling and a one-step Dst forecast.

    The classic first-order ring-current model integrates the injection of
    energy from a southward IMF against the ring current's own decay. ``Dst``
    here is the SIGNED index (negative during a storm), so a southward IMF
    drives it further negative while decay drives it back toward zero::

        dDst/dt = -a * sqrt(Np) * V * Bs - Dst / tau
        Bs      = -Bz   (Bz < 0, i.e. southward; 0 otherwise)
        a       = 0.005, tau = 8 h

    The injected term is the "coupling function" (positive, in nT/h); the decay
    term closes the balance. A one-hour Euler step from the current ``Dst``
    gives the estimate -- a local step, not a storm forecast, and it starts from
    ``Dst = 0`` when no indices frame was supplied.

    Inputs are read from frames, not from a polymorphic mapping:

    * ``solar_wind`` -- the aligned solar-wind frame (see
      :func:`align_solar_wind`) carrying ``Np`` (cm^-3), ``V`` (km/s) and ``Bz``
      (nT). A constant frame is reduced to its single value.
    * ``indices`` -- optional frame carrying the current ``Dst`` (nT; e.g. a
      ``fetch_geomagnetic_indices`` frame). When absent, ``Dst`` is taken as 0,
      which still yields the injection term (a valid "still quiet" start).

    Writes the constant ``burton_coupling`` column (the injected dDst/dt term in
    nT/h) across the frame and records the forecast in ``attrs['burton']``.
    Returns the enriched DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if solar_wind is None or not isinstance(solar_wind, pd.DataFrame):
        return _error(
            "missing_solar_wind",
            "Burton coupling needs solar-wind data (Np, V, Bz).",
            hint=(
                "Pass 'solar_wind_handle' with a solar-wind frame handle "
                "(e.g. 'sw:omni:2024-09-12'); the system aligns it automatically."
            ),
        )

    def _column(names: tuple[str, ...]) -> tuple[str | None, np.ndarray]:
        column = _column_ci(solar_wind, *names)
        if column is None:
            return None, np.array([], dtype="float64")
        values = pd.to_numeric(solar_wind[column], errors="coerce").to_numpy(
            dtype="float64"
        )
        return column, values

    def _mean(values: np.ndarray) -> float | None:
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else None

    np_col, np_values = _column(("Np", "n_p", "density", "n"))
    v_col, v_values = _column(("V", "V_sw", "Vx", "speed", "solar_wind_speed"))
    bz_col, bz_values = _column(
        ("Bz", "B_z", "Bz_gsm", "BzGSM", "bz_gsm", "B_z_gsm")
    )
    np_val = _mean(np_values)
    v_val = _mean(v_values)
    bz_val = _mean(bz_values)

    required = (
        ("Np", np_col, np_values),
        ("V", v_col, v_values),
        ("Bz", bz_col, bz_values),
    )
    missing = [name for name, column, _ in required if column is None]
    empty = [
        name
        for name, column, values in required
        if column is not None and not np.isfinite(values).any()
    ]
    if missing or empty:
        details: list[str] = []
        if missing:
            details.append(f"missing columns {missing}")
        if empty:
            details.append(f"columns with no finite values {empty}")
        return _error(
            "invalid_solar_wind" if missing else "solar_wind_no_finite_values",
            "Burton coupling needs finite Np, V and Bz: "
            + "; ".join(details)
            + ".",
            missing=missing or None,
            empty=empty or None,
            available=[str(c) for c in solar_wind.columns],
            hint="fetch_solar_wind frames carry Np, V and Bz.",
        )

    dst_val: float | None = None
    if isinstance(indices, pd.DataFrame):
        column = _column_ci(indices, "Dst", "dst")
        if column is not None:
            values = pd.to_numeric(indices[column], errors="coerce")
            values = values[np.isfinite(values)]
            if not values.empty:
                dst_val = float(values.mean())
    if dst_val is None:
        dst_val = 0.0

    bs = -float(bz_val) if float(bz_val) < 0 else 0.0
    coupling = BURTON_A * np.sqrt(max(float(np_val), 0.0)) * float(v_val) * bs
    # Dst is the SIGNED index (negative during a storm). The injection drives it
    # more negative; the decay drives it back toward zero:
    #     dDst/dt = -coupling - Dst/tau
    # The previous form (+coupling - Dst/tau) flipped both terms, so a southward
    # IMF "recovered" the field to +24 nT instead of deepening it to -77 nT.
    decay = -float(dst_val) / BURTON_TAU_HOURS
    ddst_dt = -coupling + decay
    predicted = float(dst_val) + ddst_dt  # one-hour Euler step

    out = df.copy()
    out["burton_coupling"] = coupling
    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["burton"] = {
        "Np": _jsonable(np_val, 2),
        "V": _jsonable(v_val, 1),
        "Bz": _jsonable(bz_val, 2),
        "Bs": _jsonable(bs, 2),
        "Dst_input": _jsonable(dst_val, 2),
        "a": BURTON_A,
        "tau_hours": BURTON_TAU_HOURS,
        "coupling_function": _jsonable(coupling, 4),
        "decay_term": _jsonable(decay, 4),
        "dDst_dt_nT_per_h": _jsonable(ddst_dt, 4),
        "predicted_dst": _jsonable(predicted, 2),
        "convention": "Dst is the signed index (negative during a storm)",
        "mode": "one_hour_step_from_Dst_input",
    }
    return out


# --------------------------------------------------------------------------- #
# 8. Akasofu epsilon parameter (energy input)
# --------------------------------------------------------------------------- #
def calculate_akasofu_epsilon(
    df: pd.DataFrame, solar_wind: Any = None
) -> pd.DataFrame | dict[str, Any]:
    """Akasofu's epsilon parameter: the solar-wind energy input to the magnetosphere.

    ::

        epsilon = V * B_total^2 * l0^2 * sin^4(theta/2) / mu_0
        B_total = sqrt(Bx^2 + By^2 + Bz^2)
        theta   = arctan(By / Bz)
        l0      = 7 RE

    ``V`` in m/s, ``B`` in tesla and ``l0`` in metres give epsilon in watts
    after division by the vacuum permeability ``mu_0``: ``V * B^2 * l0^2`` has
    the units of T^2 m^3/s, not watts. The ``sin^4(theta/2)`` factor couples the
    field orientation to the dayside reconnection geometry: it vanishes for a
    purely northward IMF and peaks for a southward one.

    Inputs are read from the aligned solar-wind frame passed through
    ``solar_wind`` (see :func:`align_solar_wind`): ``V`` in km/s and
    ``Bx``/``By``/``Bz`` in nT. A constant frame is reduced to its single value.

    Writes the constant ``akasofu_epsilon`` column (in watts) across the frame and
    records both unit systems in ``attrs['akasofu']``. Returns the enriched
    DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if solar_wind is None or not isinstance(solar_wind, pd.DataFrame):
        return _error(
            "missing_solar_wind",
            "Akasofu epsilon needs the solar-wind vector and speed (V, Bx, By, Bz).",
            hint=(
                "Pass 'solar_wind_handle' with a solar-wind frame handle "
                "(e.g. 'sw:omni:2024-09-12'); the system aligns it automatically."
            ),
        )

    def _column(names: tuple[str, ...]) -> tuple[str | None, np.ndarray]:
        column = _column_ci(solar_wind, *names)
        if column is None:
            return None, np.array([], dtype="float64")
        values = pd.to_numeric(solar_wind[column], errors="coerce").to_numpy(
            dtype="float64"
        )
        return column, values

    def _mean(values: np.ndarray) -> float | None:
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else None

    v_col, v_values = _column(("V", "V_sw", "Vx", "speed", "solar_wind_speed"))
    bx_col, bx_values = _column(("Bx", "B_x", "Bx_gsm", "BxGSM", "bx_gsm"))
    by_col, by_values = _column(("By", "B_y", "By_gsm", "ByGSM", "by_gsm"))
    bz_col, bz_values = _column(
        ("Bz", "B_z", "Bz_gsm", "BzGSM", "bz_gsm", "B_z_gsm")
    )
    v_val = _mean(v_values)
    bx_val = _mean(bx_values)
    by_val = _mean(by_values)
    bz_val = _mean(bz_values)

    required = (
        ("V", v_col, v_values),
        ("Bx", bx_col, bx_values),
        ("By", by_col, by_values),
        ("Bz", bz_col, bz_values),
    )
    missing = [name for name, column, _ in required if column is None]
    empty = [
        name
        for name, column, values in required
        if column is not None and not np.isfinite(values).any()
    ]
    if missing or empty:
        details: list[str] = []
        if missing:
            details.append(f"missing columns {missing}")
        if empty:
            details.append(f"columns with no finite values {empty}")
        return _error(
            "invalid_solar_wind" if missing else "solar_wind_no_finite_values",
            "Akasofu epsilon needs finite V, Bx, By and Bz: "
            + "; ".join(details)
            + ".",
            missing=missing or None,
            empty=empty or None,
            available=[str(c) for c in solar_wind.columns],
            hint="fetch_solar_wind frames carry V, Bx, By and Bz.",
        )

    b_total_nt = float(np.sqrt(bx_val**2 + by_val**2 + bz_val**2))
    theta = float(np.arctan2(by_val, bz_val)) if bz_val != 0 else float(np.pi / 2) * float(np.sign(by_val) or 1)
    sin_half = float(np.sin(theta / 2.0))
    orientation = sin_half**4

    v_si = float(v_val) * 1e3            # km/s -> m/s
    b_si = b_total_nt * 1e-9             # nT   -> T
    l0_m = AKASOFU_L0_RE * _RE_METRES    # RE   -> m

    epsilon_w = (
        v_si * (b_si**2) * (l0_m**2) * orientation / MU_0_H_PER_M
    )
    epsilon_erg_s = epsilon_w / _ERG_PER_S_IN_W  # W -> erg/s

    out = df.copy()
    out["akasofu_epsilon"] = epsilon_w
    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["akasofu"] = {
        "V_kms": _jsonable(v_val, 1),
        "B_total_nT": _jsonable(b_total_nt, 2),
        "theta_rad": _jsonable(theta, 4),
        "l0_RE": AKASOFU_L0_RE,
        "epsilon_W": _jsonable(epsilon_w, 4),
        "epsilon_erg_s": _jsonable(epsilon_erg_s, 4),
    }
    return out


# --------------------------------------------------------------------------- #
# 9. peak dH/dt over a rolling window (GIC severity)
# --------------------------------------------------------------------------- #
def calculate_peak_dhdt(
    df: pd.DataFrame,
    window_seconds: int = PEAK_DHDT_WINDOW_SECONDS,
) -> pd.DataFrame | dict[str, Any]:
    """Peak ``|dH/dt|`` over a sliding window, graded for GIC risk.

    The instantaneous ``dH/dt`` (:func:`calculate_gic_proxy`) is spiky; the
    physical driver of a GIC event is the largest change over a short interval.
    This rolls a ``window_seconds``-long window (default 10 samples on a
    1-minute grid) over ``|dH/dt|`` and takes the maximum inside it.

    Adds the ``peak_dhdt`` column (nT/min) and records the peak value, its
    timestamp and a ``low``/``medium``/``high`` warning (thresholds 50 and 200
    nT/min) in ``attrs['peak_dhdt']``. Returns the enriched DataFrame, or an
    error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if "H" not in df.columns:
        problem = _require_columns(df, ("X", "Y"))
        if problem is not None:
            return problem

    try:
        window = int(window_seconds)
    except (TypeError, ValueError):
        return _error("invalid_input", f"window_seconds={window_seconds!r} must be an integer.")
    if window < 1:
        return _error("invalid_input", f"window_seconds={window_seconds!r} must be >= 1.")

    out = df.copy()
    dh_dt = _first_difference(_horizontal(out))
    abs_rate = pd.Series(np.abs(dh_dt))
    rolled = abs_rate.rolling(window=window, min_periods=1).max().to_numpy(dtype="float64")
    out["dH_dt"] = dh_dt
    out["peak_dhdt"] = rolled

    peak = float(np.nanmax(rolled)) if np.isfinite(rolled).any() else float("nan")
    peak_index: int | None = None
    if np.isfinite(rolled).any():
        peak_index = int(np.nanargmax(rolled))

    if not np.isfinite(peak):
        level = None
    elif peak < 50.0:
        level = "low"
    elif peak <= 200.0:
        level = "medium"
    else:
        level = "high"

    peak_time: Any = None
    if peak_index is not None and "timestamp" in df.columns:
        stamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if peak_index < len(stamps) and not pd.isna(stamps.iloc[peak_index]):
            peak_time = str(stamps.iloc[peak_index])

    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["peak_dhdt"] = {
        "window_seconds": window,
        "peak_dhdt_nT_min": _jsonable(peak, 2),
        "peak_time": peak_time,
        "warning_level": level,
    }
    return out


# --------------------------------------------------------------------------- #
# 10. substorm onset from SML
# --------------------------------------------------------------------------- #
def detect_substorm_onset(
    df: pd.DataFrame,
    drop_threshold_nT_per_min: float = SUBSTORM_DROP_NT_PER_MIN,
    min_duration_min: int = SUBSTORM_MIN_DURATION_MIN,
) -> pd.DataFrame | dict[str, Any]:
    """Detect a substorm onset as a sustained steep drop of the SML index.

    The auroral electrojet intensifies at substorm onset, which drives the SML
    (SuperMAG lower envelope) sharply negative. Onset is called at the first
    sample of the earliest run where ``dSML/dt`` stays below
    ``-drop_threshold_nT_per_min`` (default -50 nT/min) for at least
    ``min_duration_min`` minutes (default 3).

    Needs an ``SML`` column (from :func:`supermag_loader` ``data_type='indices'``).
    Adds the ``sml_dh_dt`` column and records the onset in ``attrs['substorm']``
    with a confidence graded by how far the drop exceeds the threshold. Returns
    the enriched DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    sml_col = _column_ci(df, "SML", "SML_index", "sml")
    if sml_col is None:
        return _error(
            "missing_columns",
            "Substorm onset needs an 'SML' column (SuperMAG indices).",
            missing=["SML"],
            available=[str(c) for c in df.columns],
            hint="Fetch it with fetch_supermag_data(data_type='indices').",
        )

    try:
        threshold = float(drop_threshold_nT_per_min)
        duration = int(min_duration_min)
    except (TypeError, ValueError):
        return _error(
            "invalid_input",
            "drop_threshold_nT_per_min and min_duration_min must be numeric.",
        )
    if duration < 1:
        return _error("invalid_input", f"min_duration_min={min_duration_min!r} must be >= 1.")

    sml = pd.to_numeric(df[sml_col], errors="coerce").to_numpy(dtype="float64")
    dsml_dt = _first_difference(sml)

    out = df.copy()
    out["sml_dh_dt"] = dsml_dt

    steep = (dsml_dt < -abs(threshold)) & np.isfinite(dsml_dt)
    onset_index: int | None = None
    sml_drop = float("nan")
    run = 0
    for i, is_steep in enumerate(steep):
        run = run + 1 if is_steep else 0
        if run >= duration:
            onset_index = i - duration + 1
            # Drop across the detected window (first SML minus last SML).
            window = sml[onset_index : i + 1]
            if np.isfinite(window).sum() >= 2:
                sml_drop = float(window[0] - window[-1])
            break

    if onset_index is None:
        confidence = None
        onset_time = None
    else:
        stamps = None
        if "timestamp" in df.columns:
            stamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if stamps is not None and onset_index < len(stamps) and not pd.isna(stamps.iloc[onset_index]):
            onset_time = str(stamps.iloc[onset_index])
        else:
            onset_time = None
        depth = abs(sml_drop) if np.isfinite(sml_drop) else 0.0
        confidence = "high" if depth >= 2 * duration * abs(threshold) else (
            "medium" if depth >= duration * abs(threshold) else "low"
        )

    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["substorm"] = {
        "onset_time": onset_time,
        "sml_drop": _jsonable(sml_drop, 2),
        "confidence": confidence,
        "threshold_nT_per_min": _jsonable(threshold, 2),
        "min_duration_min": duration,
    }
    return out


# --------------------------------------------------------------------------- #
# 11. equatorial electrojet (EEJ) index
# --------------------------------------------------------------------------- #
def calculate_eej_index(
    df: pd.DataFrame, reference: Any = None
) -> pd.DataFrame | dict[str, Any]:
    """Equatorial electrojet strength: equatorial minus off-equator H.

    ``EEJ = H(equatorial) - H(reference)``, where the reference station sits
    6-8 degrees away in latitude so the Sq variation largely cancels and the
    difference isolates the electrojet's enhancement along the dip equator. It is
    usually computed from two INTERMAGNET stations.

    The equatorial frame is ``df``; the reference frame is passed through
    ``reference`` -- either a resolved DataFrame carrying ``H`` (or ``X``/``Y``)
    or a ready ``H_reference`` series/array. Writes the ``eej_index`` column and
    records the mean, peak and peak time in ``attrs['eej']``. Returns the
    enriched DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if "H" not in df.columns and not {"X", "Y"} <= set(df.columns):
        problem = _require_columns(df, ("H",))
        if problem is not None:
            return problem

    n_rows = len(df)
    ref_values: np.ndarray | None = None
    ref_label: str | None = None

    if isinstance(reference, pd.DataFrame):
        if "H" in reference.columns:
            ref_values = pd.to_numeric(reference["H"], errors="coerce").to_numpy(dtype="float64")
            ref_label = "frame:H"
        elif {"X", "Y"} <= set(reference.columns):
            ref_values = _horizontal(reference)
            ref_label = "frame:hypot(X,Y)"
    elif isinstance(reference, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        try:
            ref_values = pd.to_numeric(pd.Series(list(reference)), errors="coerce").to_numpy(
                dtype="float64"
            )
            ref_label = "array"
        except (TypeError, ValueError):
            ref_values = None

    if ref_values is None:
        return _error(
            "missing_reference",
            "EEJ needs the reference station's H series (6-8 degrees off the equator).",
            hint=(
                "Pass 'reference_handle' with the reference station's frame handle "
                "(e.g. 'raw:abg:2023-09-12'); the system resolves it automatically."
            ),
        )
    if ref_values.size != n_rows:
        return _error(
            "length_mismatch",
            f"The reference H has {ref_values.size} values but the equatorial frame has {n_rows} rows.",
            hint="Align the two station frames to the same minute grid first.",
        )

    h_eq = _horizontal(df)
    eej = h_eq - ref_values

    out = df.copy()
    out["eej_index"] = eej

    peak_index: int | None = None
    if np.isfinite(eej).any():
        peak_index = int(np.nanargmax(eej))
    peak_time: Any = None
    if peak_index is not None and "timestamp" in df.columns:
        stamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if peak_index < len(stamps) and not pd.isna(stamps.iloc[peak_index]):
            peak_time = str(stamps.iloc[peak_index])

    mean_eej = float(np.nanmean(eej)) if np.isfinite(eej).any() else float("nan")
    peak_eej = float(np.nanmax(eej)) if np.isfinite(eej).any() else float("nan")

    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["eej"] = {
        "eej_nT": _jsonable(peak_eej, 2),
        "eej_mean_nT": _jsonable(mean_eej, 2),
        "eej_peak_nT": _jsonable(peak_eej, 2),
        "peak_time": peak_time,
        "reference_source": ref_label,
    }
    return out
