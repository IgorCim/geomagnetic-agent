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
    "DEFAULT_SQ_METHOD",
    "BURTON_A",
    "BURTON_TAU_HOURS",
    "AKASOFU_L0_RE",
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

#: Burton coupling constants (Burton, McPherron & Russell 1975): the ring-current
#: injection efficiency ``a`` (nT per unit of the coupling function, per hour) and
#: the ring-current decay time ``tau`` in hours.
BURTON_A = 0.005
BURTON_TAU_HOURS = 8.0

#: Akasofu's ``l0`` length scale in Earth radii, used by the epsilon parameter.
AKASOFU_L0_RE = 7.0

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
def _first_param(params: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    lowered = {str(k).lower(): k for k in params}
    for alias in aliases:
        if alias in lowered:
            return params[lowered[alias]]
    return None


def _column_ci(df: pd.DataFrame, *names: str) -> str | None:
    """Return the first real column matching *names* case-insensitively."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        hit = lookup.get(name.lower())
        if hit is not None:
            return hit
    return None


def _as_numeric(value: Any) -> tuple[Any, bool]:
    """Classify a coupling input as ``(float_or_array, is_vector)``.

    A scalar becomes ``(float, False)``; a Series/Index/ndarray/list/tuple
    becomes ``(float64 ndarray, True)``. Returns ``(None, ...)`` when the value
    cannot be coerced to numbers, so the caller can report a clean error.
    """
    if isinstance(value, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        try:
            arr = pd.to_numeric(pd.Series(list(value)), errors="coerce")
        except (TypeError, ValueError):
            return None, True
        return arr.to_numpy(dtype="float64"), True
    try:
        return float(value), False
    except (TypeError, ValueError):
        return None, False


def _describe_input(value: Any) -> str:
    """Short, safe rendering of a coupling input for an error message."""
    if isinstance(value, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        try:
            return f"{type(value).__name__}(len={len(value)})"
        except TypeError:
            return type(value).__name__
    return repr(value)


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
            "no_data",
            "The solar-wind frame has no usable V/Bz samples to align.",
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

    out = observatory_df.copy()
    out["V"] = v_aligned
    out["Bz"] = bz_aligned
    out["Bz_gsm"] = bz_aligned
    out.attrs = dict(getattr(observatory_df, "attrs", {}))
    out.attrs["solar_wind_alignment"] = {
        "lag_minutes": _jsonable(float(lag_minutes), 1),
        "interp_limit_samples": limit,
        "n_observatory": int(len(observatory_df)),
        "n_solar_wind": int(len(solar_wind_df)),
        "n_finite_V": int(np.isfinite(v_aligned).sum()),
        "n_finite_Bz": int(np.isfinite(bz_aligned).sum()),
        "source": "OMNI (L1)",
    }
    return out


def calculate_coupling_function(
    df: pd.DataFrame, external_params: dict[str, Any] | None = None
) -> pd.DataFrame | dict[str, Any]:
    """Solar-wind coupling function ``E_y = V * B_z``.

    The interplanetary electric field that drives geomagnetic activity is
    ``E_y = -V * B_z``; with the common sign convention (``B_z`` southward is
    negative) a southward field gives a positive coupling, so this module
    reports ``E_y = V * B_z`` and lets the caller read the sign. Units are the
    conventional proxied mV/m when ``V`` is in km/s and ``B_z`` in nT.

    This is the one function that cannot be answered from the magnetometer frame
    alone: it needs solar-wind data (L1 / OMNIWeb). Without ``external_params``
    it returns ``missing_external_params`` with a hint rather than inventing a
    number. Accepted keys are case-insensitive: ``V``/``v``/``speed``,
    ``Bz``/``bz``/``B_z``/``bz_gsm``.

    Two input shapes are accepted:

    * **scalars** -- ``{'V': 450, 'Bz': -5}``: a single ``E_y`` is written to
      every row and recorded in ``attrs['ey_coupling']`` (classic behaviour);
    * **vectors** -- ``{'V': series_or_array, 'Bz': series_or_array}``, typically
      the ``V``/``Bz`` columns produced by :func:`align_solar_wind`: a per-row
      ``E_y`` is written to ``out['E_y']`` and the summary reports the peak,
      mean, min and max. A scalar paired with a vector is broadcast.

    Returns the enriched DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem

    if external_params is None:
        return _error(
            "missing_external_params",
            "Ey requires solar-wind data: E_y = V * B_z. Provide V (km/s) and Bz (nT).",
            hint="Load them from OMNIWeb (L1) or align_solar_wind; keys 'V' and 'Bz'.",
        )
    if not isinstance(external_params, dict):
        return _error(
            "invalid_external_params",
            f"external_params must be a mapping, got {type(external_params).__name__}.",
            hint="For example {'V': 450, 'Bz': -5} or {'V': series, 'Bz': series}.",
        )

    v_raw = _first_param(external_params, ("v", "v_sw", "vsw", "solar_wind_speed", "speed"))
    bz_raw = _first_param(external_params, ("bz", "b_z", "bz_gsm", "bzgsm", "b_z_gsm"))
    if v_raw is None or bz_raw is None:
        return _error(
            "invalid_external_params",
            "Both V and Bz are required for E_y = V * B_z.",
            available=sorted(str(k) for k in external_params),
            hint="Keys accepted: 'V' and 'Bz' (case-insensitive).",
        )

    v_val, v_vec = _as_numeric(v_raw)
    bz_val, bz_vec = _as_numeric(bz_raw)
    if v_val is None or bz_val is None:
        return _error(
            "invalid_external_params",
            f"V={_describe_input(v_raw)} and Bz={_describe_input(bz_raw)} must both be numeric.",
        )

    out = df.copy()
    out.attrs = dict(getattr(df, "attrs", {}))

    # -- scalar path: one number for the whole frame (classic behaviour) ------
    if not (v_vec or bz_vec):
        if not (np.isfinite(v_val) and np.isfinite(bz_val)):
            return _error(
                "invalid_external_params",
                "V and Bz must both be finite numbers.",
                hint="Check that the solar-wind record has no gaps at this time.",
            )
        ey = float(v_val) * float(bz_val)
        out["E_y"] = ey
        out.attrs["ey_coupling"] = {
            "mode": "scalar",
            "V": _jsonable(v_val, 1),
            "Bz": _jsonable(bz_val, 2),
            "E_y_mV_m": _jsonable(ey, 2),
        }
        return out

    # -- vector path: one E_y per observatory row -----------------------------
    n_rows = len(df)
    v_arr = np.full(n_rows, float(v_val)) if not v_vec else np.asarray(v_val, dtype="float64")
    bz_arr = np.full(n_rows, float(bz_val)) if not bz_vec else np.asarray(bz_val, dtype="float64")
    if (v_vec and v_arr.size != n_rows) or (bz_vec and bz_arr.size != n_rows):
        return _error(
            "length_mismatch",
            f"V has {v_arr.size} values and Bz has {bz_arr.size}, but the frame has {n_rows} rows.",
            hint=(
                "Align the solar wind to the observatory first: "
                "align_solar_wind(observatory_df, solar_wind_df) returns matching V and Bz."
            ),
        )

    ey = v_arr * bz_arr
    finite = np.isfinite(ey)
    if not finite.any():
        return _error(
            "invalid_external_params",
            "No finite V/Bz pair survived alignment for this frame.",
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
        "n_samples": int(n_rows),
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

    Adds ``ssc_flag`` (1 where a sample is part of an onset) and ``dH_dt``, and
    records up to ten onset events in ``attrs['ssc_events']`` with their index,
    timestamp (when available) and rate. Returns the enriched DataFrame, or an
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
    dh_dt = _first_difference(_horizontal(out))
    onset = np.zeros(dh_dt.shape, dtype="int64")
    onset[(dh_dt > threshold) & np.isfinite(dh_dt)] = 1

    out["dH_dt"] = dh_dt
    out["ssc_flag"] = onset

    stamps = None
    if "timestamp" in df.columns:
        stamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    events: list[dict[str, Any]] = []
    for i in np.where(onset == 1)[0][:10]:
        event: dict[str, Any] = {
            "index": int(i),
            "dH_dt_nT_per_min": _jsonable(dh_dt[i], 2),
        }
        if stamps is not None and not pd.isna(stamps.iloc[i]):
            event["timestamp"] = _jsonable(stamps.iloc[i])
        events.append(event)

    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["ssc_events"] = events
    out.attrs["ssc_count"] = len(events)
    out.attrs["ssc_threshold_nT_per_min"] = _jsonable(threshold, 2)
    return out


# --------------------------------------------------------------------------- #
# 7. Burton coupling function (Dst prediction)
# --------------------------------------------------------------------------- #
def _scalar_param(params: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    """First finite numeric parameter matching *aliases*, or ``None``."""
    raw = _first_param(params, aliases)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def calculate_burton_coupling(
    df: pd.DataFrame, external_params: dict[str, Any] | None = None
) -> pd.DataFrame | dict[str, Any]:
    """Burton et al. (1975) ring-current coupling and a one-step Dst forecast.

    The classic first-order ring-current model integrates the injection of
    energy from a southward IMF against the ring current's own decay::

        dDst/dt = a * sqrt(Np) * V * Bs - Dst / tau
        Bs      = -Bz   (Bz < 0, i.e. southward; 0 otherwise)
        a       = 0.005, tau = 8 h

    The injected term is the "coupling function"; the decay term closes the
    balance. A one-hour Euler step from the current ``Dst`` gives the predicted
    value. Inputs come from ``external_params`` (case-insensitive keys): ``Np``
    (or ``density``) in cm^-3, ``V`` in km/s, ``Bz`` in nT, and the current
    ``Dst`` (or ``dst``) in nT; a stored solar-wind handle is accepted by the
    caller and flattened to the same keys.

    Writes the constant ``burton_coupling`` column (the injected dDst/dt term in
    nT/h) across the frame and records the forecast in ``attrs['burton']``.
    Returns the enriched DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if external_params is None or not isinstance(external_params, dict):
        return _error(
            "missing_external_params",
            "Burton coupling needs solar-wind data and a current Dst.",
            hint=(
                "Provide Np (cm^-3), V (km/s), Bz (nT) and Dst (nT), e.g. "
                "{'Np': 5, 'V': 450, 'Bz': -10, 'Dst': -30}."
            ),
        )

    np_val = _scalar_param(external_params, ("np", "n_p", "density", "n"))
    v_val = _scalar_param(external_params, ("v", "v_sw", "vsw", "speed", "solar_wind_speed"))
    bz_val = _scalar_param(external_params, ("bz", "b_z", "bz_gsm", "bzgsm", "b_z_gsm"))
    dst_val = _scalar_param(external_params, ("dst", "dst_current", "current_dst", "dst_now"))

    missing = [
        name
        for name, value in (("Np", np_val), ("V", v_val), ("Bz", bz_val), ("Dst", dst_val))
        if value is None
    ]
    if missing:
        return _error(
            "invalid_external_params",
            f"Burton coupling needs {missing} but they are missing or non-numeric.",
            missing=missing,
            available=sorted(str(k) for k in external_params),
            hint="Keys accepted: 'Np', 'V', 'Bz', 'Dst' (case-insensitive).",
        )

    bs = -float(bz_val) if float(bz_val) < 0 else 0.0
    coupling = BURTON_A * np.sqrt(max(float(np_val), 0.0)) * float(v_val) * bs
    decay = float(dst_val) / BURTON_TAU_HOURS
    ddst_dt = coupling - decay
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
        "mode": "one_hour_step",
    }
    return out


# --------------------------------------------------------------------------- #
# 8. Akasofu epsilon parameter (energy input)
# --------------------------------------------------------------------------- #
def calculate_akasofu_epsilon(
    df: pd.DataFrame, external_params: dict[str, Any] | None = None
) -> pd.DataFrame | dict[str, Any]:
    """Akasofu's epsilon parameter: the solar-wind energy input to the magnetosphere.

    ::

        epsilon = V * B_total^2 * l0^2 * sin^4(theta/2)
        B_total = sqrt(Bx^2 + By^2 + Bz^2)
        theta   = arctan(By / Bz)
        l0      = 7 RE

    ``V`` in m/s, ``B`` in tesla and ``l0`` in metres give epsilon in watts, the
    standard form. The ``sin^4(theta/2)`` factor couples the field orientation to
    the dayside reconnection geometry: it vanishes for a purely northward IMF and
    peaks for a southward one. Solar-wind inputs come from ``external_params``
    (case-insensitive): ``V`` in km/s, ``Bx``/``By``/``Bz`` in nT.

    Writes the constant ``akasofu_epsilon`` column (in watts) across the frame and
    records both unit systems in ``attrs['akasofu']``. Returns the enriched
    DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if external_params is None or not isinstance(external_params, dict):
        return _error(
            "missing_external_params",
            "Akasofu epsilon needs the solar-wind vector and speed.",
            hint="Provide V (km/s), Bx, By, Bz (nT), e.g. {'V': 450, 'Bx': 2, 'By': -3, 'Bz': -5}.",
        )

    v_val = _scalar_param(external_params, ("v", "v_sw", "vsw", "speed", "solar_wind_speed"))
    bx_val = _scalar_param(external_params, ("bx", "b_x", "bx_gsm", "bxgsm"))
    by_val = _scalar_param(external_params, ("by", "b_y", "by_gsm", "bygsm"))
    bz_val = _scalar_param(external_params, ("bz", "b_z", "bz_gsm", "bzgsm", "b_z_gsm"))

    missing = [
        name
        for name, value in (("V", v_val), ("Bx", bx_val), ("By", by_val), ("Bz", bz_val))
        if value is None
    ]
    if missing:
        return _error(
            "invalid_external_params",
            f"Akasofu epsilon needs {missing} but they are missing or non-numeric.",
            missing=missing,
            available=sorted(str(k) for k in external_params),
            hint="Keys accepted: 'V', 'Bx', 'By', 'Bz' (case-insensitive).",
        )

    b_total_nt = float(np.sqrt(bx_val**2 + by_val**2 + bz_val**2))
    theta = float(np.arctan2(by_val, bz_val)) if bz_val != 0 else float(np.pi / 2) * float(np.sign(by_val) or 1)
    sin_half = float(np.sin(theta / 2.0))
    orientation = sin_half**4

    v_si = float(v_val) * 1e3            # km/s -> m/s
    b_si = b_total_nt * 1e-9             # nT   -> T
    l0_m = AKASOFU_L0_RE * _RE_METRES    # RE   -> m

    epsilon_w = v_si * (b_si**2) * (l0_m**2) * orientation
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
    df: pd.DataFrame, external_params: dict[str, Any] | None = None
) -> pd.DataFrame | dict[str, Any]:
    """Equatorial electrojet strength: equatorial minus off-equator H.

    ``EEJ = H(equatorial) - H(reference)``, where the reference station sits
    6-8 degrees away in latitude so the Sq variation largely cancels and the
    difference isolates the electrojet's enhancement along the dip equator. It is
    usually computed from two INTERMAGNET stations.

    The equatorial frame is ``df``; the reference frame is passed through
    ``external_params`` under ``reference`` (a stored frame handle or a mapping
    with ``H``), or as a ready ``H_reference`` series/array. Writes the
    ``eej_index`` column and records the mean, peak and peak time in
    ``attrs['eej']``. Returns the enriched DataFrame, or an error dict.
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

    if external_params is not None and isinstance(external_params, dict):
        raw_ref = _first_param(external_params, ("h_reference", "reference_h", "reference_frame"))
        if raw_ref is None:
            raw_ref = _first_param(external_params, ("reference", "reference_df", "df2"))
        if isinstance(raw_ref, pd.DataFrame):
            if "H" in raw_ref.columns:
                ref_values = pd.to_numeric(raw_ref["H"], errors="coerce").to_numpy(dtype="float64")
                ref_label = "mapping:H"
            elif {"X", "Y"} <= set(raw_ref.columns):
                ref_values = _horizontal(raw_ref)
                ref_label = "mapping:hypot(X,Y)"
        elif isinstance(raw_ref, (pd.Series, pd.Index, np.ndarray, list, tuple)):
            try:
                ref_values = pd.to_numeric(pd.Series(list(raw_ref)), errors="coerce").to_numpy(
                    dtype="float64"
                )
                ref_label = "array"
            except (TypeError, ValueError):
                ref_values = None
        elif isinstance(raw_ref, str) and raw_ref.strip():
            return _error(
                "unresolved_reference",
                "calculate_eej_index needs the resolved reference frame, not its handle.",
                hint="Resolve the handle via the store, then pass its H column.",
            )

    if ref_values is None:
        return _error(
            "missing_external_params",
            "EEJ needs the reference station's H series (6-8 degrees off the equator).",
            hint=(
                "Pass external_params={'reference': <ref frame with H>} or "
                "{'H_reference': <array>}."
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
