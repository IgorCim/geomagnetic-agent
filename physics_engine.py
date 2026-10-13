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
    "detect_ssc",
    # contract + constants
    "is_error",
    "SSC_THRESHOLD_NT_PER_MIN",
    "DEFAULT_SQ_METHOD",
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

    Returns the enriched DataFrame, or an error dict.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem

    if external_params is None:
        return _error(
            "missing_external_params",
            "Ey requires solar-wind data: E_y = V * B_z. Provide V (km/s) and Bz (nT).",
            hint="Load them from OMNIWeb (L1)? Keys accepted: 'V' and 'Bz'.",
        )
    if not isinstance(external_params, dict):
        return _error(
            "invalid_external_params",
            f"external_params must be a mapping, got {type(external_params).__name__}.",
            hint="For example {'V': 450, 'Bz': -5}.",
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
    try:
        v_val = float(v_raw)
        bz_val = float(bz_raw)
    except (TypeError, ValueError):
        return _error(
            "invalid_external_params",
            f"V={v_raw!r} and Bz={bz_raw!r} must both be numbers.",
        )
    if not (np.isfinite(v_val) and np.isfinite(bz_val)):
        return _error(
            "invalid_external_params",
            "V and Bz must both be finite numbers.",
            hint="Check that the solar-wind record has no gaps at this time.",
        )

    ey = v_val * bz_val
    out = df.copy()
    out["E_y"] = ey
    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["ey_coupling"] = {
        "V": _jsonable(v_val, 1),
        "Bz": _jsonable(bz_val, 2),
        "E_y_mV_m": _jsonable(ey, 2),
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
