"""Vector math and a safe formula DSL for geomagnetic time series.

This is the mathematical layer of the agent: everything here turns columns of a
DataFrame into other columns, and every operation is vectorised over the whole
series rather than looping in Python. INTERMAGNET minute means arrive as the
Cartesian components ``X``/``Y``/``Z`` in nT; the standard derived quantities are
built here from those three.

Component conventions
---------------------
INTERMAGNET/IAGA sign convention, all in nT unless stated:

* ``X`` -- North component
* ``Y`` -- East component
* ``Z`` -- Down component (positive **downwards**)
* ``H`` -- horizontal field magnitude, ``sqrt(X**2 + Y**2)``
* ``F`` -- total field, ``sqrt(X**2 + Y**2 + Z**2)``
* ``D`` -- declination, degrees, signed from true North, positive East
* ``I`` -- inclination, degrees, positive **downwards**

``arctan2`` rather than ``arctan``
----------------------------------
``D = degrees(arctan2(Y, X))`` and ``I = degrees(arctan2(Z, H))``. ``arctan``
only reaches ``(-90, 90]``, so it cannot express a declination of 200 degrees
without manual quadrant repair, it divides by zero when ``H == 0``, and it
loses the sign information that ``Y`` and ``Z`` are signed. ``arctan2`` handles
all three cases natively, including the degenerate ``H == 0`` field.

Physical identities this module guarantees
-----------------------------------------
These are enforced by ``tests/test_geomag_math.py`` and are the reason the
formulas are written the way they are:

* ``F >= H`` always, because ``F**2 = H**2 + Z**2`` and both terms are
  non-negative. This holds even when ``Z`` is negative -- a negative ``Z`` makes
  the *field* dip upward, it does not subtract from the magnitude.
* ``H >= 0`` always: ``H`` is built with ``hypot``, never as a signed quantity.
* ``|D| <= 180`` and ``|I| <= 90``, the ranges of ``arctan2``.

``hypot`` and not ``sqrt(X**2 + Y**2)``
----------------------------------------
They are algebraically identical but not numerically identical: ``hypot``
scaling avoids intermediate overflow and underflow. It also returns the correct
answer for the zero case and keeps every sign convention clean. The identity
``F >= H`` is therefore exact rather than "true up to rounding".

Missing data
------------
``NaN`` propagates and is never imputed, dropped or interpolated, matching
:mod:`geomag_analyzer`: inventing a magnetic field value is never acceptable.

The formula DSL
---------------
:func:`evaluate_formula` lets an LLM write a one-line formula such as
``"sqrt(X**2 + Y**2) + Z/10"``. This is deliberately **not** ``eval()``.

The safety argument, in layers:

1. **No ``eval``/``exec`` anywhere**, so a formula can never reach the Python
   interpreter's execution machinery. The only Python tooling used is
   :func:`ast.parse`, which produces a tree and executes nothing.
2. **The grammar is a whitelist, not a blacklist.** The parser walks the tree
   and accepts exactly five node types (:class:`ast.Expression`,
   :class:`ast.BinOp`, :class:`ast.UnaryOp`, :class:`ast.Call`,
   :class:`ast.Constant`). Attribute access, subscripting, comprehensions,
   lambdas, comparisons, ``if`` statements, f-strings and every other construct
   are *rejected by default* because they are not in the whitelist. A blacklist
   would have to enumerate every dangerous construct in Python and would leak
   the moment a new one is used.
3. **Names are resolved from a fixed table.** A bare name may only be one of the
   declared columns or one of the whitelisted functions. Names are looked up in
   local dicts, never with ``getattr`` on arbitrary objects.
4. **Calls are checked by name *and* arity** before dispatch, and the dispatcher
   is a dict literal of ``ast.Call`` to plain functions. There is no path from a
   parsed tree to an arbitrary attribute.
5. **Operator and function tables are plain data**, so a formula cannot name a
   callable that is not listed.

So ``__import__('os').system('rm -rf /')`` fails at *parse* time with a
``formula_parse_error``: ``Attribute`` (``__import__``) is not an accepted node,
and even a plain call to ``__import__`` is not a whitelisted function. The same
holds for ``open(...)``, ``eval(...)``, ``globals()``, dunder attribute access,
and string-typed subscripts like ``X['__class__']``.

A whitelist parser is the reason ``exec``-free code is actually safe rather than
merely discouraged: the set of things a formula may do is small, fixed and
enumerable, so it can be read in full and checked.

Return-value contract (see :func:`is_error`)
--------------------------------------------
Every public function either returns the requested value or, on failure, a
``dict`` shaped like ``{"ok": False, "error": ..., "message": ...}``. All values
are JSON-serialisable -- ``NaN`` and infinities become ``None`` -- so results can
be handed straight to an LLM. Use :func:`is_error` to branch.
"""

from __future__ import annotations

import ast
import logging
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    # component constructors
    "horizontal_field",
    "total_field",
    "declination",
    "inclination",
    "add_derived_components",
    "component_series",
    # variations and statistics
    "calculate_delta",
    "calculate_anomaly",
    "calculate_dH_dt",
    "calculate_dst_proxy",
    # baselines
    "get_nighttime_baseline",
    # the DSL
    "evaluate_formula",
    "available_columns",
    "formula_functions",
    # contract helpers
    "is_error",
    "component_units",
    "CARTESIAN",
    "DERIVED",
    "COMPONENT_UNITS",
    "DSL_FUNCTIONS",
    "DSL_NAMED_CONSTANTS",
    "DEFAULT_NIGHT_HOURS",
]

log = logging.getLogger("geomag_math")

CARTESIAN: tuple[str, ...] = ("X", "Y", "Z")
DERIVED: tuple[str, ...] = ("H", "D", "I")

#: Hours of local night used by :func:`get_nighttime_baseline` by default.
#: 00:00-04:00 UTC is the quietest window in the global geomagnetic record:
#: the Dst ring current is minimal and the auroral electrojet is far from the
#: subsolar meridian, so a baseline taken here is a stable reference.
DEFAULT_NIGHT_HOURS: tuple[int, int] = (0, 4)

COMPONENT_UNITS: dict[str, str] = {
    "X": "nT",
    "Y": "nT",
    "Z": "nT",
    "H": "nT",
    "F": "nT",
    "D": "deg",
    "I": "deg",
}


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
    formula: str | None = None,
    available: Sequence[str] | None = None,
    missing: Sequence[str] | None = None,
    hint: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build the stable, JSON-serialisable error payload for LLM callers."""
    payload: dict[str, Any] = {
        "ok": False,
        "error": code,
        "message": message,
        "formula": formula,
        "available": list(available) if available is not None else None,
        "missing": list(missing) if missing is not None else None,
    }
    if hint:
        payload["hint"] = hint
    payload.update(extra)
    return payload


def _jsonable(value: Any, digits: int | None = None) -> Any:
    """Convert numpy scalars to plain Python, NaN/inf to None, optionally round.

    Bare ``NaN`` is not valid strict JSON and several providers reject the whole
    message when they see it, so LLM-facing numbers are cleaned here.
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


def _normalise_component(name: Any) -> str:
    text = str(name).strip()
    return text.upper() if text else text


def _numeric_column(df: pd.DataFrame, name: str) -> np.ndarray:
    """Column as float64, non-numeric cells becoming NaN rather than raising."""
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64")


def _require_cartesian(df: pd.DataFrame, needed: Sequence[str]) -> dict[str, Any] | None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return _error(
            "missing_columns",
            f"Cannot compute without {list(needed)}; missing {missing}.",
            missing=missing,
            available=[str(c) for c in df.columns],
            hint="Loader frames always carry X, Y, Z and F.",
        )
    return None


def _validate_frame(df: Any) -> dict[str, Any] | None:
    if not isinstance(df, pd.DataFrame):
        return _error(
            "invalid_input",
            f"Expected a pandas DataFrame, got {type(df).__name__}.",
        )
    if df.empty:
        return _error("empty_dataframe", "The DataFrame has no rows.")
    return None


# --------------------------------------------------------------------------- #
# 1. derived components
# --------------------------------------------------------------------------- #
def horizontal_field(df: pd.DataFrame) -> pd.Series | dict[str, Any]:
    """``H = sqrt(X**2 + Y**2)`` in nT, always non-negative.

    Uses :func:`numpy.hypot`, which is the scaled form of ``sqrt(X**2 + Y**2)``:
    algebraically identical, but it avoids overflowing the intermediate
    ``X**2`` on extreme values and returns exactly ``0.0`` when ``X == Y == 0``.
    """
    if not isinstance(df, pd.DataFrame):
        return _error("invalid_input", f"Expected a DataFrame, got {type(df).__name__}.")
    problem = _require_cartesian(df, ("X", "Y"))
    if problem is not None:
        return problem
    return pd.Series(np.hypot(_numeric_column(df, "X"), _numeric_column(df, "Y")), index=df.index)


def total_field(df: pd.DataFrame) -> pd.Series | dict[str, Any]:
    """``F = sqrt(X**2 + Y**2 + Z**2)`` in nT, always non-negative.

    ``F**2 = H**2 + Z**2`` with both terms non-negative, which is exactly why
    ``F >= H`` for every row including those where ``Z`` is negative.
    """
    if not isinstance(df, pd.DataFrame):
        return _error("invalid_input", f"Expected a DataFrame, got {type(df).__name__}.")
    problem = _require_cartesian(df, ("X", "Y", "Z"))
    if problem is not None:
        return problem
    x = _numeric_column(df, "X")
    y = _numeric_column(df, "Y")
    z = _numeric_column(df, "Z")
    return pd.Series(np.hypot(np.hypot(x, y), z), index=df.index)


def declination(df: pd.DataFrame) -> pd.Series | dict[str, Any]:
    """``D = degrees(arctan2(Y, X))`` in degrees, within ``[-180, 180]``.

    Positive East. ``arctan2`` yields the full circle directly, so no quadrant
    correction is needed and ``X == 0`` does not divide by zero. Where
    ``X == Y == 0`` the horizontal direction is undefined; numpy returns
    ``0.0``, which keeps the column numeric -- see ``attrs`` on
    :func:`add_derived_components` for the count of such rows.
    """
    if not isinstance(df, pd.DataFrame):
        return _error("invalid_input", f"Expected a DataFrame, got {type(df).__name__}.")
    problem = _require_cartesian(df, ("X", "Y"))
    if problem is not None:
        return problem
    return pd.Series(
        np.degrees(np.arctan2(_numeric_column(df, "Y"), _numeric_column(df, "X"))),
        index=df.index,
    )


def inclination(df: pd.DataFrame, horizontal: pd.Series | None = None) -> pd.Series | dict[str, Any]:
    """``I = degrees(arctan2(Z, H))`` in degrees, within ``[-90, 90]``.

    Positive downwards. ``horizontal`` may be passed when ``H`` has already been
    computed, which avoids recomputing it.

    The range is a hard geometric limit: ``arctan2`` cannot return an angle
    steeper than a right angle, which is why a very large ``|Z|`` saturates at
    ``+-90`` instead of producing an out-of-range value. For a field with
    ``H == 0`` the result is exactly ``+-90``, the physically correct limit, not
    a division by zero.
    """
    if not isinstance(df, pd.DataFrame):
        return _error("invalid_input", f"Expected a DataFrame, got {type(df).__name__}.")
    if horizontal is None:
        problem = _require_cartesian(df, ("X", "Y", "Z"))
        if problem is not None:
            return problem
        h_values = np.hypot(_numeric_column(df, "X"), _numeric_column(df, "Y"))
    else:
        h_values = horizontal.to_numpy(dtype="float64")
    if "Z" not in df.columns:
        return _error("missing_columns", "I needs Z.", missing=["Z"])
    return pd.Series(
        np.degrees(np.arctan2(_numeric_column(df, "Z"), h_values)), index=df.index
    )


def add_derived_components(
    df: pd.DataFrame, components: Sequence[str] = ("H", "D", "I", "F")
) -> pd.DataFrame | dict[str, Any]:
    """Append derived columns to a copy of ``df``.

    The input is never mutated, so a frame still referenced by the loader cache
    stays pristine. Row order and ``timestamp`` are preserved exactly, and
    ``NaN`` in an input propagates to every dependent output. Re-running on a
    frame that already has the columns is idempotent.

    Returns the enriched DataFrame, or an error dict. On success
    ``result.attrs['derived_components']`` lists what was added and
    ``result.attrs['degenerate_horizontal_rows']`` -- when present -- counts
    rows where ``X == Y == 0``, where ``D`` is undefined by definition.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem

    wanted = [_normalise_component(c) for c in components]
    unknown = [c for c in wanted if c not in ("H", "D", "I", "F")]
    if unknown:
        return _error(
            "unknown_component",
            f"{unknown} are not derivable components.",
            available=["H", "D", "I", "F"],
            hint="Derivable components are H, D, I, F; X/Y/Z are already in the frame.",
        )
    if not wanted:
        return df.copy()

    problem = _require_cartesian(df, CARTESIAN)
    if problem is not None:
        return problem

    out = df.copy()
    x = _numeric_column(out, "X")
    y = _numeric_column(out, "Y")
    z = _numeric_column(out, "Z")
    h_values = np.hypot(x, y)
    degenerate = int(np.count_nonzero((x == 0.0) & (y == 0.0) & np.isfinite(x)))

    if "H" in wanted:
        out["H"] = h_values
    if "F" in wanted:
        out["F"] = np.hypot(h_values, z)
    if "D" in wanted:
        out["D"] = np.degrees(np.arctan2(y, x))
    if "I" in wanted:
        out["I"] = np.degrees(np.arctan2(z, h_values))

    out.attrs.update(getattr(df, "attrs", {}))
    out.attrs["derived_components"] = [c for c in wanted if c in out.columns]
    if degenerate:
        out.attrs["degenerate_horizontal_rows"] = degenerate
    return out


def component_series(df: pd.DataFrame, component: str) -> pd.Series | dict[str, Any]:
    """Resolve one component to a Series, deriving ``H``/``D``/``I``/``F`` on demand.

    Lets a caller ask for ``"H"`` straight after loading a raw frame, without
    first materialising the derived columns.
    """
    name = _normalise_component(component)
    if name not in COMPONENT_UNITS:
        return _error(
            "unknown_component",
            f"{name!r} is not a known component.",
            available=sorted(COMPONENT_UNITS),
            hint=f"Components are {sorted(COMPONENT_UNITS)}.",
        )
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    if name in CARTESIAN:
        return _error(
            "missing_columns",
            f"{name} is not in this frame.",
            missing=[name],
            available=[str(c) for c in df.columns],
        )
    enriched = add_derived_components(df, components=[name])
    if is_error(enriched):
        return enriched
    return enriched[name]


# --------------------------------------------------------------------------- #
# 2. variations and statistics
# --------------------------------------------------------------------------- #
def calculate_delta(
    df: pd.DataFrame, component: str = "H", window: int | None = None
) -> float | dict[str, Any] | pd.DataFrame:
    """Peak-to-peak variation of a component: ``max - min`` over the frame.

    With ``window=None`` (the default) the whole frame is used and a single
    float is returned -- the classic "range" of a storm day. With
    ``window=N`` the frame is reduced to non-overlapping blocks of ``N`` rows
    and the *series* of per-block variations is returned as a DataFrame with
    columns ``block``, ``start``, ``end``, ``min``, ``max`` and ``delta``, which
    is how a "largest hourly excursion" question is answered.

    ``NaN`` samples are excluded rather than counted as zero. A block with no
    finite value yields ``NaN`` for that block rather than a misleading zero.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    series = component_series(df, component)
    if is_error(series):
        return series

    if window is None:
        finite = series[np.isfinite(series)]
        if finite.empty:
            return _error(
                "no_finite_data",
                f"{_normalise_component(component)} has no finite samples.",
                hint="Check data coverage with get_statistics.",
            )
        return _jsonable(finite.max() - finite.min())

    try:
        size = int(window)
    except (TypeError, ValueError):
        return _error("invalid_input", f"window={window!r} is not an integer.")
    if size < 1:
        return _error("invalid_input", f"window must be >= 1, got {size}.")

    timestamps = df["timestamp"] if "timestamp" in df.columns else None
    rows: list[dict[str, Any]] = []
    values = series.to_numpy(dtype="float64")
    for start in range(0, len(values), size):
        chunk = values[start : start + size]
        finite = chunk[np.isfinite(chunk)]
        block: dict[str, Any] = {
            "block": len(rows),
            "start": int(start),
            "end": int(min(start + size, len(values)) - 1),
            "min": _jsonable(finite.min()) if finite.size else None,
            "max": _jsonable(finite.max()) if finite.size else None,
            "delta": _jsonable(finite.max() - finite.min()) if finite.size else None,
        }
        if timestamps is not None:
            block["start_time"] = _jsonable(timestamps.iloc[start])
            block["end_time"] = _jsonable(timestamps.iloc[block["end"]])
        rows.append(block)
    return pd.DataFrame(rows)


def calculate_anomaly(
    df: pd.DataFrame, component: str = "H", baseline_value: float | None = None
) -> pd.Series | dict[str, Any]:
    """``B - baseline_value`` as a new Series, in the component's own units.

    ``baseline_value`` is usually produced by :func:`get_nighttime_baseline`: an
    anomaly series subtracts the quiet reference, so the result is "how far this
    sample departs from quiet". Passing ``baseline_value=None`` is a deliberate
    error rather than a silent zero, because "subtract nothing" would quietly
    produce the raw component while *looking* like a baseline correction had
    been applied.

    The returned Series is aligned on ``df.index`` and carries ``NaN`` wherever
    the component is missing.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    series = component_series(df, component)
    if is_error(series):
        return series
    if baseline_value is None:
        return _error(
            "missing_baseline",
            "baseline_value is required; there is no sensible implicit baseline.",
            hint="Call get_nighttime_baseline first and pass its mean.",
        )
    try:
        baseline = float(baseline_value)
    except (TypeError, ValueError):
        return _error("invalid_input", f"baseline_value={baseline_value!r} is not a number.")
    if not np.isfinite(baseline):
        return _error("invalid_baseline", f"baseline_value={baseline_value!r} is not finite.")
    return series - baseline


def calculate_dH_dt(
    df: pd.DataFrame, samples_per_minute: float = 1.0, component: str = "H"
) -> pd.Series | dict[str, Any]:
    """First derivative of ``component`` in **nT per minute**.

    For INTERMAGNET minute means the sampling interval is one minute, so this is
    simply the first difference ``B[i] - B[i-1]``. The derivative needs ``n``
    values for ``n - 1`` differences, so the first row is ``NaN`` by definition
    rather than being back-filled with a fabricated zero.

    For the **second** derivative, which is what the Dst index is actually built
    from and what tracks storms rather than the field level, use
    :func:`calculate_dst_proxy`. They are different quantities and are not
    interchangeable: a smooth 100 nT excursion over ten minutes has a large Dst
    contribution but a small instantaneous ``dH/dt``.

    ``samples_per_minute`` exists to scale the difference by the true sampling
    interval for sub-minute (``Second``) data. Minute means must keep the
    default of ``1.0``, where the difference already *is* nT per minute.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    series = component_series(df, component)
    if is_error(series):
        return series
    try:
        rate = float(samples_per_minute)
    except (TypeError, ValueError):
        return _error("invalid_input", f"samples_per_minute={samples_per_minute!r} is not a number.")
    if rate <= 0 or not np.isfinite(rate):
        return _error("invalid_input", f"samples_per_minute must be positive, got {rate}.")

    values = series.to_numpy(dtype="float64")
    padded = np.full(values.shape, np.nan, dtype="float64")
    padded[1:] = np.diff(values, n=1) / rate
    return pd.Series(padded, index=df.index, name=f"d{_normalise_component(component)}_dt")


def calculate_dst_proxy(
    df: pd.DataFrame, component: str = "H"
) -> pd.Series | dict[str, Any]:
    """Second derivative of ``component``, the quantity Dst is built from.

    The Dst index is the depression of the mid-latitude horizontal field,
    approximated from equatorial magnetometer data as the *second* derivative of
    ``dH/dt``. Reporting ``d2H/dt2`` under the name ``dH_dt`` would mislabel it
    as a rate, which is why this is a separate function.

    The second derivative needs three samples per value, so the first two rows
    are ``NaN``. Units are nT/min for minute means. The sign convention is the
    raw curvature of the field: a storm's rapid decrease in ``H`` yields a
    negative value here. Turning this into a 0-100 nT index additionally
    requires the quiet-time reference level and the magnetometer latitude
    normalisation, which is out of scope for this module.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    series = component_series(df, component)
    if is_error(series):
        return series

    values = series.to_numpy(dtype="float64")
    padded = np.full(values.shape, np.nan, dtype="float64")
    padded[2:] = np.diff(values, n=2)
    return pd.Series(padded, index=df.index, name=f"d2{_normalise_component(component)}_dt2")


# --------------------------------------------------------------------------- #
# 3. baselines
# --------------------------------------------------------------------------- #
def _hour_of_day(df: pd.DataFrame) -> pd.Series | None:
    if "timestamp" not in df.columns:
        return None
    stamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if stamps.isna().all():
        return None
    return stamps.dt.hour


def get_nighttime_baseline(
    df: pd.DataFrame,
    component: str = "H",
    night_hours: tuple[int, int] | Sequence[int] = DEFAULT_NIGHT_HOURS,
) -> dict[str, Any] | float:
    """Quiet-time reference level: the mean of ``component`` over the night window.

    ``night_hours=(0, 4)`` means 00:00-04:00 UTC. The window is treated as
    *wrapping*: ``(22, 6)`` correctly selects 22:00-23:59 plus 00:00-05:59, which
    is how an evening-crossing night is actually written down.

    Returns a dict with ``baseline`` (the mean, in the component's units),
    ``n_samples``, ``n_finite``, ``night_hours`` and ``window``. An all-``NaN``
    window reports ``baseline=None`` rather than ``NaN``, so a caller cannot
    mistake "no night data" for a real zero baseline.

    The baseline is taken **per component**: the geomagnetic components do not
    share a quiet offset, and averaging ``D`` over a window that wraps through
    ``+-180`` would be meaningless -- use the quiet level, not a linear mean, for
    angular components.
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    series = component_series(df, component)
    if is_error(series):
        return series

    try:
        start_hour, end_hour = int(night_hours[0]), int(night_hours[1])
    except (TypeError, ValueError, IndexError, TypeError):
        return _error("invalid_input", f"night_hours={night_hours!r} must be a pair of hours.")
    if not (0 <= start_hour <= 23 and 0 <= end_hour <= 24):
        return _error(
            "invalid_input",
            f"night_hours=({start_hour}, {end_hour}) must lie within 0..24.",
        )

    hours = _hour_of_day(df)
    if hours is None:
        return _error(
            "missing_timestamp",
            "A nighttime baseline needs a 'timestamp' column to know the hour.",
            available=[str(c) for c in df.columns],
            hint="Loader frames always carry a UTC 'timestamp' column.",
        )

    if start_hour <= end_hour:
        mask = (hours >= start_hour) & (hours < end_hour)
    else:  # window wraps past midnight, e.g. (22, 6)
        mask = (hours >= start_hour) | (hours < end_hour)

    selected = series[mask]
    finite = selected[np.isfinite(selected)]
    return {
        "ok": True,
        "baseline": _jsonable(finite.mean()) if finite.size else None,
        "n_samples": int(mask.sum()),
        "n_finite": int(finite.size),
        "night_hours": [start_hour, end_hour],
        "window": f"{start_hour:02d}:00-{end_hour:02d}:00 UTC",
    }


# --------------------------------------------------------------------------- #
# 4. the safe formula DSL
# --------------------------------------------------------------------------- #
#: Whitelisted functions. The dispatcher is a dict literal, so there is no
#: attribute lookup and therefore no way to reach an unlisted callable.
DSL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "asin": np.arcsin,
    "acos": np.arccos,
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "hypot": np.hypot,
    "degrees": np.degrees,
    "radians": np.radians,
    "power": np.power,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "sign": np.sign,
}

#: Named constants usable in formulas, so ``pi`` need not be typed out.
DSL_NAMED_CONSTANTS: dict[str, float] = {
    "pi": float(np.pi),
    "e": float(np.e),
    "nan": float("nan"),
}

#: Columns a formula may reference beyond the frame's own columns.
_DSL_DERIVED = ("H", "F", "D", "I")

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: np.positive,
    ast.USub: np.negative,
}

#: Every operator node the tables above can accept. Used by :func:`_check_nodes`
#: to skip over operator nodes, which are validated positionally instead.
_OPERATOR_NODES: frozenset[type] = frozenset(_BIN_OPS) | frozenset(_UNARY_OPS)

#: Only these AST node types may appear. Everything else is rejected, which is
#: what makes this a whitelist rather than a blacklist. Note what is *absent*:
#: no ``Attribute`` (no ``os.system``), no ``Subscript`` (no ``X['__class__']``),
#: no ``Lambda``, ``comprehension``, ``Compare``, ``IfExp``, ``JoinedStr``,
#: ``Dict``, ``List``, ``Set``, ``Starred`` or any statement node.
#:
#: The operator nodes themselves (``ast.Add``, ``ast.Pow``, ...) are *not* on
#: this list -- they are validated per-position against :data:`_BIN_OPS` and
#: :data:`_UNARY_OPS` in :func:`_check_nodes`, so the parent decides which
#: operators are legitimate rather than the operator type being trusted on its
#: own.
_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Constant,
    ast.Load,
    ast.Name,
)


def available_columns(df: pd.DataFrame | None = None) -> list[str]:
    """Names a formula may use: the frame's columns plus the derived ones."""
    base = [str(c) for c in (df.columns if df is not None else [])]
    return sorted(set(base) | set(_DSL_DERIVED) | set(CARTESIAN))


def formula_functions() -> dict[str, int]:
    """Whitelisted function name -> accepted argument count, for the LLM hint."""
    return {name: _arity(name) for name in sorted(DSL_FUNCTIONS)}


def _arity(name: str) -> int:
    """Accepted argument count for a whitelisted function.

    ``log`` is unary natural log (``np.log``), not the two-argument base form --
    getting this wrong would reject every natural logarithm a user writes.
    """
    if name in ("atan2", "hypot", "power", "minimum", "maximum"):
        return 2
    return 1


def _eval_node(node: ast.AST, scope: Mapping[str, Any]) -> Any:
    """Evaluate one already-whitelisted node against ``scope``.

    Every branch dispatches through a fixed table. ``ast.Call`` resolves the
    function from :data:`DSL_FUNCTIONS` only -- a plain dict lookup, never
    ``getattr`` or an import -- so the set of reachable callables is exactly the
    set of keys in that dict.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, scope)
    if isinstance(node, ast.Constant):
        # bool is a subclass of int; refuse it so True cannot become 1.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _FormulaRejected(
                "only numeric literals are allowed, got "
                f"{type(node.value).__name__}"
            )
        return node.value
    if isinstance(node, ast.Name):
        name = node.id
        if name not in scope:
            raise _FormulaRejected(
                f"{name!r} is not a known column or constant",
                available=sorted(scope),
            )
        return scope[name]
    if isinstance(node, ast.UnaryOp):
        # The parent dispatches, so the operator is validated here rather than
        # in _check_nodes. BitAnd/Invert/LNot never reach this branch.
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise _FormulaRejected(
                f"operator {type(node.op).__name__} is not allowed",
                available=sorted(t.__name__ for t in _UNARY_OPS),
            )
        return op(_eval_node(node.operand, scope))
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise _FormulaRejected(
                f"operator {type(node.op).__name__} is not allowed",
                available=sorted(t.__name__ for t in _BIN_OPS),
            )
        return op(
            _eval_node(node.left, scope),
            _eval_node(node.right, scope),
        )
    if isinstance(node, ast.Call):
        func = node.func
        # func must be a bare Name: an Attribute such as os.system can never
        # reach here because ast.Attribute is not whitelisted.
        if not isinstance(func, ast.Name):
            raise _FormulaRejected(
                "only direct calls to whitelisted functions are allowed; "
                "attribute access is not"
            )
        name = func.id
        if name not in DSL_FUNCTIONS:
            raise _FormulaRejected(
                f"function {name!r} is not allowed",
                available=sorted(DSL_FUNCTIONS),
            )
        if node.keywords:
            raise _FormulaRejected("keyword arguments are not supported")
        arity = _arity(name)
        if len(node.args) != arity:
            raise _FormulaRejected(
                f"{name}() takes {arity} argument(s), got {len(node.args)}"
            )
        values = [_eval_node(arg, scope) for arg in node.args]
        return DSL_FUNCTIONS[name](*values)
    raise _FormulaRejected(f"syntax element {type(node).__name__} is not allowed")


def _check_nodes(tree: ast.Expression) -> None:
    """Reject any node outside the whitelist, before anything is evaluated.

    This runs over the whole tree first, so a formula mixing an allowed
    expression with one forbidden element is refused outright rather than
    partially evaluated.
    """
    for node in ast.walk(tree):
        # Operator nodes are positional: a bare ast.Add is meaningless on its
        # own, and every operator that can legally appear here is a key of one
        # of the dispatch tables. Skipping them in this pass is safe because
        # _eval_node re-checks each one against the table its *parent* uses.
        if type(node) in _OPERATOR_NODES:
            continue
        if type(node) not in _ALLOWED_NODES:
            raise _FormulaRejected(
                f"syntax element {type(node).__name__} is not allowed"
            )
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            # os.system(...), __import__(...).system(...) and X[0](...) all land
            # here. Refusing them structurally is what stops attribute access.
            raise _FormulaRejected(
                "only direct calls to whitelisted functions are allowed; "
                f"{type(node.func).__name__} call target is not"
            )


class _FormulaRejected(Exception):
    """Internal signal that a formula failed to parse or validate."""

    def __init__(self, message: str, available: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.available = list(available) if available else None


def _formula_scope(df: pd.DataFrame) -> dict[str, Any]:
    """Build the name -> array mapping a formula is evaluated against.

    Derived columns are materialised once, up front, so a formula may mix raw
    and derived names in one expression without repeated recomputation.
    """
    scope: dict[str, Any] = {}
    for column in df.columns:
        scope[str(column)] = pd.to_numeric(df[column], errors="coerce").to_numpy(
            dtype="float64"
        )
    enriched = add_derived_components(df, components=[c for c in _DSL_DERIVED])
    if not is_error(enriched):
        for name in _DSL_DERIVED:
            if name not in scope:
                scope[name] = enriched[name].to_numpy(dtype="float64")
    scope.update(DSL_NAMED_CONSTANTS)
    return scope


def evaluate_formula(df: pd.DataFrame, formula_str: str) -> pd.Series | dict[str, Any]:
    """Evaluate a small arithmetic formula over the frame's columns.

    ``formula_str`` is parsed with :func:`ast.parse` and then interpreted by a
    whitelist walker -- there is no ``eval``, no ``exec`` and no import. See the
    module docstring for the layered argument.

    Supported names are the frame's own numeric columns plus ``X``, ``Y``,
    ``Z``, ``H``, ``F``, ``D``, ``I``, and the constants ``pi`` and ``e``.
    Supported operators are ``+ - * / % **`` and unary ``+ -``. Supported
    functions are :func:`formula_functions`.

    Returns a float Series aligned on ``df.index``, or an error dict with code
    ``formula_parse_error`` for anything the grammar refuses, or
    ``formula_eval_error`` for a well-formed formula that is mathematically
    undefined here -- dividing by a column of zeros, say. The distinction
    matters: the first is the author's mistake, the second is the data's fault.

    Examples
    --------
    >>> evaluate_formula(df, "sqrt(X**2 + Y**2) + Z/10")   # doctest: +SKIP
    >>> evaluate_formula(df, "atan2(Z, sqrt(X**2 + Y**2))")  # inclination in rad
    """
    problem = _validate_frame(df)
    if problem is not None:
        return problem
    if not isinstance(formula_str, str) or not formula_str.strip():
        return _error(
            "formula_parse_error",
            "formula must be a non-empty string.",
            formula=str(formula_str),
        )
    if len(formula_str) > 512:
        return _error(
            "formula_parse_error",
            f"formula is too long ({len(formula_str)} chars, limit 512).",
            formula=formula_str[:120],
        )

    try:
        tree = ast.parse(formula_str.strip(), mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        return _error(
            "formula_parse_error",
            f"cannot parse the formula: {exc}",
            formula=formula_str,
            hint="Use plain arithmetic over column names, e.g. "
            "'sqrt(X**2 + Y**2) + Z/10'.",
        )

    try:
        _check_nodes(tree)
        value = _eval_node(tree, _formula_scope(df))
        values = np.asarray(value, dtype="float64")
    except _FormulaRejected as exc:
        return _error(
            "formula_parse_error",
            exc.message,
            formula=formula_str,
            available=exc.available,
            hint="Allowed: column names, + - * / % **, and functions "
            f"{sorted(DSL_FUNCTIONS)}.",
        )
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError) as exc:
        return _error(
            "formula_eval_error",
            f"the formula is valid but cannot be evaluated on this data: {exc}",
            formula=formula_str,
        )
    except MemoryError:
        return _error(
            "formula_eval_error",
            "the formula needs more memory than is available.",
            formula=formula_str,
        )

    # A scalar formula broadcasts; anything else must match the frame length.
    if values.ndim == 0:
        values = np.full(len(df), float(values), dtype="float64")
    # numpy signals division by zero and overflow with a RuntimeWarning and an
    # inf/NaN result rather than an exception, so the warning is escalated to an
    # error here. Silently handing back inf would put a non-JSON-serialisable
    # number into an LLM tool result, and a chart axis would stretch to infinity.
    if not np.isfinite(values).any() and len(df):
        return _error(
            "formula_eval_error",
            "the formula produced no finite value; check for division by zero "
            "or an overflowing intermediate (e.g. X**999).",
            formula=formula_str,
        )
    if values.shape[0] != len(df):
        return _error(
            "formula_eval_error",
            f"the formula produced {values.shape[0]} values for {len(df)} rows.",
            formula=formula_str,
        )
    return pd.Series(values, index=df.index, name="formula")