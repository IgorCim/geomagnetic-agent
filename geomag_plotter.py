"""Interactive Plotly visualisation of geomagnetic time series.

Second of the agent's "hands": turns a DataFrame from
:mod:`intermagnet_loader` (optionally enriched by :mod:`geomag_analyzer`) into a
standalone interactive HTML file that a scientist can open in any browser, zoom,
hover and export without a Python runtime.

Headless-safe by design
-----------------------
Everything here is file-based. ``fig.show()`` is never called and
``write_html(..., auto_open=False)`` is used throughout, so the module is safe in
a container, on a server, or over SSH where no browser or display exists.
Plotly's ``kaleido`` static-image engine is *not* required: PNG/SVG export would
need it, but interactive HTML does not.

Output
------
HTML files are self-contained by default (``include_plotlyjs=True`` bundles the
~4 MB Plotly bundle) so a plot still opens on a laptop with no network -- the
normal case for field and campaign work. Pass ``include_plotlyjs='cdn'`` to
trade offline capability for a ~10 KB file.

Return contract
---------------
Plot functions return the **absolute path as a string** on success, or an error
dict (``{"ok": False, "error": ..., ...}``) on failure. Use
:func:`is_error` to branch. Paths are strings rather than ``Path`` objects so
they drop straight into an LLM tool response.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from geomag_analyzer import (
    COMPONENT_UNITS,
    DERIVED,
    calculate_derived_components,
    is_error,
)

__all__ = [
    "plot_components",
    "plot_comparison",
    "plot_overlay",
    "plot_space_weather_comparison",
    "plot_magnetogram",
    "is_error",
    "DEFAULT_OUTPUT_DIR",
    "TIME_OF_DAY_ANCHOR",
]

log = logging.getLogger("geomag_plotter")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "plots"

TIME_OF_DAY_ANCHOR = pd.Timestamp("2000-01-01")

GL_THRESHOLD = 5000

_SERIES_COLOR = {
    "X": "#1f77b4",
    "Y": "#d62728",
    "Z": "#2ca02c",
    "F": "#9467bd",
    "H": "#ff7f0e",
    "D": "#8c564b",
    "I": "#e377c2",
    "S": "#7f7f7f",
}
_COMPARISON_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _error(
    code: str,
    message: str,
    *,
    requested: Any = None,
    available: Sequence[str] | None = None,
    missing: Sequence[str] | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": code,
        "message": message,
        "requested": requested,
        "available": list(available) if available is not None else None,
        "missing": list(missing) if missing is not None else None,
    }
    if hint:
        payload["hint"] = hint
    return payload


def _check_frame(df: Any, label: str) -> dict[str, Any] | None:
    if not isinstance(df, pd.DataFrame):
        return _error("invalid_input", f"{label} must be a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        return _error("empty_dataframe", f"{label} is empty.")
    if "timestamp" not in df.columns:
        return _error(
            "missing_columns",
            f"{label} has no 'timestamp' column.",
            available=[str(c) for c in df.columns],
            missing=["timestamp"],
            hint="Data must come from intermagnet_loader.fetch_observatory_data().",
        )
    return None


#: Units of the physics columns :mod:`physics_engine` appends to a frame. Kept
#: beside the standard components so a delta is still labelled ("dH (nT)") when
#: it is plotted straight from an enriched frame instead of becoming a nameless
#: trace.
_PHYSICS_COMPONENTS: dict[str, str] = {
    "X_sq": "nT",
    "Y_sq": "nT",
    "Z_sq": "nT",
    "H_sq": "nT",
    "F_sq": "nT",
    "dX": "nT",
    "dY": "nT",
    "dZ": "nT",
    "dH": "nT",
    "F_delta": "nT",
    "dH_dt": "nT/min",
    "dst_proxy": "nT",
    "E_y": "mV/m",
    "ssc_flag": "",
}

#: Units of the L1 solar-wind columns :mod:`space_weather_loader` publishes and
#: :func:`physics_engine.align_solar_wind` carries onto the observatory clock.
#: Kept here so a ground-vs-space chart can label the right-hand axis ("Bz (nT)",
#: "V (km/s)") instead of printing a bare number.
_SOLAR_WIND_UNITS: dict[str, str] = {
    "V": "km/s",
    "Np": "cm^-3",
    "T": "K",
    "Bx": "nT",
    "By": "nT",
    "Bz": "nT",
    "Bz_gsm": "nT",
}

#: Every named column and its unit. Standard components win on conflict; the
#: physics and solar-wind names never collide with them.
_PLOT_UNITS: dict[str, str] = {
    **COMPONENT_UNITS,
    **_PHYSICS_COMPONENTS,
    **_SOLAR_WIND_UNITS,
}



def _column_unit(name: Any) -> str:
    """Unit for *name*, case-insensitively, covering physics-derived columns."""
    text = str(name)
    if text in _PLOT_UNITS:
        return _PLOT_UNITS[text]
    target = text.lower()
    for column, unit in _PLOT_UNITS.items():
        if column.lower() == target:
            return unit
    return ""


def _resolve_column(df: pd.DataFrame, name: Any) -> str | None:
    """The exact column in *df* matching *name* ignoring case, else ``None``.

    ``plot_components`` used to upper-case every request, so the physics columns
    (``dX``, ``dst_proxy``, ``H_sq``, ...) never matched their own frame and came
    back as ``unknown_component``. Matching the real column lets a request for
    "dx" draw the "dX" series.
    """
    if name in df.columns:
        return str(name)
    target = str(name).lower()
    for column in df.columns:
        if str(column).lower() == target:
            return str(column)
    return None


def _resolve_columns(
    df: pd.DataFrame, components: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Split requests into ``(resolved_exact_names, missing_requested_names)``."""
    resolved: list[str] = []
    missing: list[str] = []
    for raw in components:
        name = str(raw).strip()
        if not name:
            continue
        column = _resolve_column(df, name)
        if column is not None:
            resolved.append(column)
        else:
            missing.append(name)
    return resolved, missing


def _has_cartesian(df: pd.DataFrame) -> bool:
    """True when X/Y/Z are present under any casing (they are derivable then)."""
    return all(_resolve_column(df, c) is not None for c in ("X", "Y", "Z"))


def _prepare(
    df: pd.DataFrame, components: Sequence[str], label: str
) -> tuple[pd.DataFrame, list[str]] | dict[str, Any]:
    """Ensure every requested component is present, deriving H/D/I if needed.

    Returns ``(frame, resolved)`` where *resolved* is the exact list of column
    names to draw, or an error payload. Matching is case-insensitive against the
    frame's real columns, so a request for ``'dx'`` draws the ``'dX'`` series
    that :func:`physics_engine.calculate_deltas` stored.
    """
    requested = [str(c).strip() for c in components if str(c).strip()]
    if not requested:
        return _error("invalid_input", f"No components requested for {label}.")

    resolved, missing = _resolve_columns(df, requested)
    if missing:
        derivable = [
            name for name in missing if name.upper() in DERIVED and _has_cartesian(df)
        ]
        if derivable:
            out = calculate_derived_components(df)
            if is_error(out):
                return _error(
                    "unknown_component",
                    f"Cannot derive {derivable} for {label}: {out['message']}",
                    requested=requested,
                    available=[str(c) for c in df.columns],
                    missing=missing,
                )
            df = out
            resolved, missing = _resolve_columns(df, requested)
    if missing:
        available = [str(c) for c in df.columns]
        return _error(
            "unknown_component",
            f"{label} is missing component(s) {missing} and they cannot be derived.",
            requested=requested,
            available=available,
            missing=missing,
            hint="Available columns: "
            + ", ".join(available)
            + ". Known components: "
            + ", ".join([*COMPONENT_UNITS, *_PHYSICS_COMPONENTS])
            + ". H, D and I are derived automatically when X, Y and Z exist.",
        )
    return df, resolved


def _slug(text: str, fallback: str = "plot") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_").lower()
    return cleaned[:60] or fallback


def _downsample(
    x: pd.Series, y: pd.Series, max_points: int
) -> tuple[pd.Series, pd.Series, bool]:
    """Stride-sample a trace so huge series do not produce 100 MB HTML files.

    A one-year minute series is ~525k points per component; browsers choke on
    that many SVG nodes, so a plot is capped by default. The agent is told
    through the returned flag and a log warning so it can offer a narrower
    window rather than silently showing a coarse curve.
    """
    n = len(x)
    if max_points is None or n <= max_points:
        return x, y, False
    step = int(np.ceil(n / max_points))
    return x.iloc[::step], y.iloc[::step], True


def _time_of_day(stamps: pd.Series) -> pd.Series:
    """Re-anchor timestamps to a fixed date, keeping only time-of-day.

    ``2025-03-10 05:30`` and ``2024-11-02 05:30`` both map to
    ``2000-01-01 05:30``, so two different days land on top of each other while
    the axis stays a genuine datetime axis -- which means Plotly still renders
    ``HH:MM`` ticks, keeps a working rangeslider and gives correct hover text,
    none of which a string- or numeric-axis workaround would preserve.
    """
    ts = pd.to_datetime(stamps, errors="coerce")
    midnight = ts.dt.normalize()
    return TIME_OF_DAY_ANCHOR + (ts - midnight)


def _write(fig: go.Figure, path: Path, include_plotlyjs: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        auto_open=False,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )
    return str(path.resolve())


def _unit_label(components: Sequence[str]) -> str:
    units = {_column_unit(c) for c in components}
    if len(units) == 1:
        return units.pop() or "value"
    return "value"


# --------------------------------------------------------------------------- #
# 1. multi-component time series
# --------------------------------------------------------------------------- #
def plot_components(
    df: pd.DataFrame,
    components: Sequence[str] = ("X", "Y", "Z"),
    title: str = "Geomagnetic components",
    output_dir: str | Path | None = None,
    filename: str | None = None,
    max_points: int = GL_THRESHOLD,
    include_plotlyjs: Any = True,
) -> str | dict[str, Any]:
    """Plot one or more components of a single time series.

    Components with the same unit share one axis.  When the requested
    components use different units (for example ``H`` in nT and ``D``/``I`` in
    degrees), each component gets its own stacked panel so a large H trace
    cannot flatten the angular components.

    Parameters
    ----------
    df:
        DataFrame with ``timestamp`` plus the requested components. ``H``, ``D``
        and ``I`` are derived automatically from ``X``/``Y``/``Z`` when absent.
    components:
        Component names, default ``('X', 'Y', 'Z')``. Case-insensitive.
    title:
        Figure title; also used to build the default filename.
    output_dir:
        Destination directory, default ``./plots`` beside this module.
    filename:
        Explicit ``.html`` filename. Default is derived from ``title``.
    max_points:
        Maximum points drawn per trace. Longer series are stride-sampled and a
        warning is logged. Pass ``None`` to disable.
    include_plotlyjs:
        ``True`` bundles Plotly for a self-contained offline file; ``'cdn'``
        keeps the file small but needs internet to open.

    Returns
    -------
    str
        Absolute path to the written HTML file.
    dict
        Error payload on failure.

    Notes
    -----
    The figure switches to WebGL (``go.Scattergl``) automatically when the data
    is large, because DOM-based SVG traces degrade badly past a few thousand
    points. Missing samples are left as gaps rather than bridged -- connecting
    across a data gap would imply measurements that were never made.
    """
    problem = _check_frame(df, "df")
    if problem is not None:
        return problem

    prepared = _prepare(df, components, "df")
    if isinstance(prepared, dict):
        return prepared
    df, names = prepared
    stamps = pd.to_datetime(df["timestamp"], errors="coerce")
    total = int(stamps.notna().sum())
    heavy = total > GL_THRESHOLD or len(names) * total > 4 * GL_THRESHOLD
    trace_cls = go.Scattergl if heavy else go.Scatter

    units = [_column_unit(name) or "value" for name in names]
    distinct_units = list(dict.fromkeys(units))
    mixed_units = len(distinct_units) > 1

    # H (nT), D (deg) and I (deg) on one axis makes D/I invisible next to a
    # 10 000 nT H trace, and it labels the axis "value".  When units differ,
    # give each component its own panel and a real unit label.  Same-unit
    # series (X/Y/Z/F or a single component) keep the compact shared axis.
    if mixed_units:
        fig = make_subplots(
            rows=len(names),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
        )
    else:
        fig = go.Figure()

    sampled_any = False
    for index, name in enumerate(names, start=1):
        values = pd.to_numeric(df[name], errors="coerce")
        xs, ys, sampled = _downsample(stamps, values, max_points)
        sampled_any = sampled_any or sampled
        unit = _column_unit(name)
        trace = trace_cls(
            x=xs,
            y=ys,
            name=f"{name} ({unit})" if unit else name,
            mode="lines",
            line={"width": 1, "color": _SERIES_COLOR.get(name)},
            connectgaps=False,
            hovertemplate=(
                f"<b>{name}</b><br>%{{x}}<br>{unit}: %{{y:.2f}}"
                "<extra></extra>"
            ),
        )
        if mixed_units:
            fig.add_trace(trace, row=index, col=1)
        else:
            fig.add_trace(trace)

    if sampled_any:
        log.warning(
            "Series longer than max_points=%s; plots were stride-sampled. "
            "Narrow the date range to resolve individual minutes.",
            max_points,
        )

    if mixed_units:
        for index, name in enumerate(names, start=1):
            fig.update_yaxes(
                title_text=f"{name} ({_column_unit(name) or 'value'})",
                gridcolor="rgba(128,128,128,0.2)",
                row=index,
                col=1,
            )
        fig.update_xaxes(
            title_text="Time (UTC)",
            gridcolor="rgba(128,128,128,0.2)",
            row=len(names),
            col=1,
        )
        fig.update_layout(height=max(520, 215 * len(names) + 100))
    else:
        unit = _unit_label(names)
        fig.update_layout(
            xaxis={"title": "Time (UTC)", "rangeslider": {"visible": False},
                   "gridcolor": "rgba(128,128,128,0.2)"},
            yaxis={"title": unit, "gridcolor": "rgba(128,128,128,0.2)"},
        )

    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        template="plotly_white",
        margin={"t": 90},
    )
    fig.update_xaxes(rangeslider_visible=False)

    target = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    name = filename or f"plot_{_slug(title)}.html"
    if not name.endswith(".html"):
        name += ".html"
    return _write(fig, target / name, include_plotlyjs)


# --------------------------------------------------------------------------- #
# 2. two-day comparison
# --------------------------------------------------------------------------- #
def plot_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    component: str = "H",
    title: str = "Day comparison",
    output_dir: str | Path | None = None,
    filename: str | None = None,
    max_points: int = GL_THRESHOLD,
    include_plotlyjs: Any = True,
) -> str | dict[str, Any]:
    """Overlay two days of one component on a shared time-of-day axis.

    The central problem is that the two DataFrames carry different dates, so a
    plain datetime axis would leave a month-long gap between the traces and they
    could never be compared. Each timestamp is therefore reduced to its
    time-of-day and re-anchored to :data:`TIME_OF_DAY_ANCHOR` -- see
    :func:`_time_of_day`. Because the result is still a real datetime axis,
    Plotly formats the ticks as ``HH:MM``, the rangeslider and hover keep
    working, and the two curves sit directly on top of each other.

    Parameters
    ----------
    df1, df2:
        The two DataFrames to compare. ``H``/``D``/``I`` are derived on demand.
    component:
        Single component to compare, default ``'H'``.
    title:
        Figure title, also used for the default filename.
    output_dir, filename, max_points, include_plotlyjs:
        As for :func:`plot_components`.

    Returns
    -------
    str
        Absolute path to the written HTML file.
    dict
        Error payload on failure.

    Notes
    -----
    Trace labels carry each frame's real date, so the legend tells the scientist
    which curve is which. If the two frames are shifted by whole minutes the
    curves still overlay exactly; if a frame has duplicate times-of-day the
    points are kept, since that indicates irregular sampling rather than an
    error. Rows outside 00:00-24:00 cannot occur, but times are masked on the
    anchor date regardless so the axis can never be corrupted.
    """
    for index, frame in (("df1", df1), ("df2", df2)):
        problem = _check_frame(frame, index)
        if problem is not None:
            return problem

    prepared1 = _prepare(df1, [component], "df1")
    if isinstance(prepared1, dict):
        return prepared1
    prepared2 = _prepare(df2, [component], "df2")
    if isinstance(prepared2, dict):
        return prepared2
    df1, names1 = prepared1
    df2, names2 = prepared2
    # Each frame reads its own resolved name ("dX" vs "DX"); label and unit use
    # the first frame's spelling.
    name = names1[0]
    columns = (names1[0], names2[0])

    unit = _column_unit(name) or "value"
    total = 0
    fig = go.Figure()
    labels: list[str] = []

    for idx, (frame, column, color) in enumerate(
        zip((df1, df2), columns, _COMPARISON_COLORS)
    ):
        stamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        total += int(stamps.notna().sum())
        values = pd.to_numeric(frame[column], errors="coerce")
        xs, ys, sampled = _downsample(_time_of_day(stamps), values, max_points)
        if sampled:
            log.warning("df%s longer than max_points=%s; stride-sampled.", idx + 1, max_points)

        distinct = pd.to_datetime(frame["timestamp"], errors="coerce").dt.date.dropna().unique()
        if len(distinct) == 1:
            label = f"{distinct[0]}"
        elif len(distinct) == 0:
            label = f"df{idx + 1}"
        else:
            label = f"{min(distinct)} .. {max(distinct)}"
        labels.append(label)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=label,
                mode="lines",
                line={"width": 1.2, "color": color},
                opacity=0.85,
                connectgaps=False,
                hovertemplate=(
                    f"<b>{name}</b> ({label})<br>%{{x|%H:%M}}<br>"
                    f"{unit}: %{{y:.2f}}<extra></extra>"
                ),
            )
        )

    if len(set(labels)) == 1:
        log.warning(
            "df1 and df2 both cover %s -- the comparison shows a single curve. "
            "Check that the two DataFrames are really different days.",
            labels[0],
        )

    fig.update_layout(
        title={"text": f"{title}  [{name}, {', '.join(labels)}]", "x": 0.01, "xanchor": "left"},
        xaxis={
            "title": "Time of day (UTC)",
            "type": "date",
            # explicit format, so ticks stay HH:MM at every zoom level --
            # the anchored year (2000-01-01) is an implementation detail
            # and must never leak into the axis labels.
            "tickformat": "%H:%M",
            "rangeslider": {"visible": True, "thickness": 0.06},
            "gridcolor": "rgba(128,128,128,0.2)",
        },
        yaxis={"title": unit, "gridcolor": "rgba(128,128,128,0.2)"},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        template="plotly_white",
        margin={"t": 90, "b": 70},
    )

    target = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_name = filename or f"compare_{name.lower()}_{_slug(title)}.html"
    if not out_name.endswith(".html"):
        out_name += ".html"
    path = _write(fig, target / out_name, include_plotlyjs)
    log.info("Comparison plotted: %s over %s points", name, total)
    return path


# --------------------------------------------------------------------------- #
# 2b. multi-station overlay
# --------------------------------------------------------------------------- #
_OVERLAY_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#17becf",
)

_TIME_SYSTEM_LABEL = {
    "UT": "Time of day (UT)",
    "LT": "Local time (LT)",
    "MLT": "Magnetic local time (MLT)",
}


def plot_overlay(
    frames: dict[str, pd.DataFrame],
    component: str = "H",
    offsets: dict[str, float] | None = None,
    time_system: str = "UT",
    time_shifts: dict[str, float] | None = None,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    max_points: int = GL_THRESHOLD,
    include_plotlyjs: Any = True,
) -> str | dict[str, Any]:
    """Overlay one component from several stations on a single shared axis.

    Each station's curve is drawn on the same chart with a per-station vertical
    ``offset`` (nT) so quiet-day and storm-day stations stay legible at the same
    time, and a per-station ``time_shifts`` (hours) that slides the trace onto a
    UT, LT or MLT clock. The x axis is a real datetime axis anchored to
    :data:`TIME_OF_DAY_ANCHOR`, so ticks stay ``HH:MM`` and hover/rangeslider
    keep working, exactly as in :func:`plot_comparison`.

    Parameters
    ----------
    frames:
        ``{station_code: DataFrame}`` in the order to draw them (top first). The
        caller is responsible for ordering -- the agent sorts by geomagnetic
        latitude so northern stations sit on top.
    component:
        Single component to plot, default ``'H'``. ``H``/``D``/``I`` are derived
        on demand.
    offsets:
        Display-only vertical shift per station, in nT. Never written back to a
        frame; a missing station gets ``0``.
    time_system:
        ``'UT'``, ``'LT'`` or ``'MLT'`` -- used only to label the x axis; the
        actual shift is supplied by ``time_shifts``.
    time_shifts:
        Hours to add to each station's time-of-day, keyed by station code.
    title, output_dir, filename, max_points, include_plotlyjs:
        As for :func:`plot_components`.

    Returns
    -------
    str
        Absolute path to the written HTML file.
    dict
        Error payload on failure.
    """
    if not isinstance(frames, dict) or not frames:
        return _error("invalid_input", "frames must be a non-empty {station: DataFrame} mapping.")

    system = str(time_system or "UT").strip().upper()
    if system not in _TIME_SYSTEM_LABEL:
        return _error(
            "invalid_input",
            f"time_system must be one of UT/LT/MLT, got {time_system!r}.",
        )

    name = str(component).strip()
    offsets = offsets or {}
    time_shifts = time_shifts or {}

    fig = go.Figure()
    total = 0
    ordered: list[str] = []
    for index, (station, frame) in enumerate(frames.items()):
        code = str(station).strip().upper() or f"station{index + 1}"
        problem = _check_frame(frame, code)
        if problem is not None:
            return problem
        prepared = _prepare(frame, [name], code)
        if isinstance(prepared, dict):
            return prepared
        frame, resolved = prepared
        column = resolved[0]

        try:
            offset = float(offsets.get(station, offsets.get(code, 0.0)))
        except (TypeError, ValueError):
            return _error("invalid_input", f"offset for {code} is not a number.")
        try:
            shift = float(time_shifts.get(station, time_shifts.get(code, 0.0)))
        except (TypeError, ValueError):
            return _error("invalid_input", f"time shift for {code} is not a number.")

        stamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        total += int(stamps.notna().sum())
        values = pd.to_numeric(frame[column], errors="coerce") + offset
        xs = _time_of_day(stamps) + pd.to_timedelta(shift, unit="h")
        xs, ys, sampled = _downsample(xs, values, max_points)
        if sampled:
            log.warning("%s longer than max_points=%s; stride-sampled.", code, max_points)

        unit = _column_unit(name)
        suffix = f" ({offset:+g} nT)" if offset else ""
        ordered.append(code)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=f"{code}{suffix}",
                mode="lines",
                line={"width": 1.2, "color": _OVERLAY_COLORS[index % len(_OVERLAY_COLORS)]},
                opacity=0.9,
                connectgaps=False,
                hovertemplate=(
                    f"<b>{code}</b><br>%{{x|%H:%M}}<br>{unit}: %{{y:.2f}}"
                    "<extra></extra>"
                ),
            )
        )

    unit = _unit_label([name])
    axis_title = _TIME_SYSTEM_LABEL[system]
    fig.update_layout(
        title={"text": title or f"Overlay {', '.join(ordered)}", "x": 0.01, "xanchor": "left"},
        xaxis={
            "title": axis_title,
            "type": "date",
            "tickformat": "%H:%M",
            "rangeslider": {"visible": True, "thickness": 0.06},
            "gridcolor": "rgba(128,128,128,0.2)",
        },
        yaxis={"title": unit, "gridcolor": "rgba(128,128,128,0.2)"},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        template="plotly_white",
        margin={"t": 90, "b": 70},
    )

    target = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_name = filename or f"overlay_{name.lower()}_{_slug(title or 'stations')}.html"
    if not out_name.endswith(".html"):
        out_name += ".html"
    path = _write(fig, target / out_name, include_plotlyjs)
    log.info("Overlay plotted: %s stations, %s points", len(ordered), total)
    return path


# --------------------------------------------------------------------------- #
# 2c. ground magnetometer vs solar wind (dual axis)
# --------------------------------------------------------------------------- #
#: The ground trace is red and the solar-wind traces cool colours, so the eye
#: can keep "what the field did" apart from "what drove it" at a glance.
_GROUND_TRACE_COLOR = "#d62728"
_SOLAR_WIND_COLORS = ("#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf")


def plot_space_weather_comparison(
    df: pd.DataFrame,
    ground_component: str = "H",
    solar_wind_components: Sequence[str] = ("Bz", "V"),
    title: str = "Earth vs space weather",
    output_dir: str | Path | None = None,
    filename: str | None = None,
    max_points: int = GL_THRESHOLD,
    include_plotlyjs: Any = True,
) -> str | dict[str, Any]:
    """Draw a ground component and the solar wind on aligned panels.

    The two feeds have nothing in common to overlay: the magnetometer carries
    ``X/Y/Z/F`` (nT) while the solar wind carries speed and the IMF in km/s and
    nT, and OMNI is bow-shock shifted onto a different clock. ``plot_comparison``
    can only compare the *same* component of two days, so a "did the southward
    field drive the storm" chart had no tool that could draw it. This one is that
    tool: the caller passes a frame already aligned onto the observatory clock by
    :func:`physics_engine.align_solar_wind`, the ground component goes on its own
    panel, and solar-wind components with one unit share a right-hand axis.  If
    the requested wind components use different units (the normal ``Bz`` in nT
    plus ``V`` in km/s case), each unit gets its own panel instead of one axis
    where V's hundreds of km/s flatten Bz's few nT.

    Parameters
    ----------
    df:
        The aligned frame: ``timestamp`` plus the ground components and the
        solar-wind columns (``V``, ``Bz``, ...). ``H``/``D``/``I`` are derived on
        demand from ``X``/``Y``/``Z``.
    ground_component:
        Single ground component for the left axis, default ``'H'``. Physics
        columns such as ``'dst_proxy'`` are accepted when they are in the frame.
    solar_wind_components:
        Solar-wind components for the right axis, default ``('Bz', 'V')``.
        ``Bz``/``V`` are what :func:`physics_engine.align_solar_wind` produces;
        other columns already present in *df* (``By``, ``Np``, ...) are drawn too.
    title, output_dir, filename, max_points, include_plotlyjs:
        As for :func:`plot_components`.

    Returns
    -------
    str
        Absolute path to the written HTML file.
    dict
        Error payload on failure.
    """
    problem = _check_frame(df, "aligned frame")
    if problem is not None:
        return problem

    requested = [str(c).strip() for c in solar_wind_components if str(c).strip()]
    if not requested:
        return _error("invalid_input", "At least one solar-wind component is required.")

    prepared = _prepare(df, [ground_component], "ground")
    if isinstance(prepared, dict):
        return prepared
    df, ground_names = prepared
    ground_name = ground_names[0]

    sw_names, sw_missing = _resolve_columns(df, requested)
    if not sw_names:
        available = [str(c) for c in df.columns]
        return _error(
            "unknown_component",
            f"No solar-wind component(s) {sw_missing} found in the aligned frame.",
            requested=requested,
            available=available,
            missing=sw_missing,
            hint=(
                "Pass a frame aligned to the observatory clock (Bz and V are added "
                "by physics_engine.align_solar_wind). Known solar-wind columns: "
                + ", ".join(_SOLAR_WIND_UNITS)
                + "."
            ),
        )
    if sw_missing:
        log.warning("solar-wind component(s) not present and skipped: %s", sw_missing)

    stamps = pd.to_datetime(df["timestamp"], errors="coerce")
    total = int(stamps.notna().sum())

    units: list[str] = []
    for name in sw_names:
        unit = _column_unit(name) or "value"
        if unit not in units:
            units.append(unit)
    # Bz (nT) and V (km/s) on one right axis is unreadable: V spans hundreds of
    # km/s while Bz spans a few nT.  When the requested wind components have
    # different units, stack one panel per unit instead of forcing a shared
    # axis.  A single unit keeps the compact Earth-vs-wind dual-axis view.
    mixed_wind_units = len(units) > 1

    if mixed_wind_units:
        fig = make_subplots(
            rows=1 + len(units),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
        )
    else:
        fig = go.Figure()

    ground_values = pd.to_numeric(df[ground_name], errors="coerce")
    ground_unit = _column_unit(ground_name) or "value"
    gx, gy, sampled = _downsample(stamps, ground_values, max_points)
    if sampled:
        log.warning("ground series longer than max_points=%s; stride-sampled.", max_points)
    ground_trace = go.Scatter(
        x=gx,
        y=gy,
        name=f"{ground_name} ({ground_unit})",
        mode="lines",
        line={"width": 1.4, "color": _GROUND_TRACE_COLOR},
        connectgaps=False,
        hovertemplate=(
            f"<b>{ground_name}</b><br>%{{x|%H:%M}}<br>"
            f"{ground_unit}: %{{y:.2f}}<extra></extra>"
        ),
    )
    if mixed_wind_units:
        fig.add_trace(ground_trace, row=1, col=1)
    else:
        ground_trace.yaxis = "y"
        fig.add_trace(ground_trace)

    for index, name in enumerate(sw_names):
        values = pd.to_numeric(df[name], errors="coerce")
        xs, ys, sampled = _downsample(stamps, values, max_points)
        if sampled:
            log.warning("%s longer than max_points=%s; stride-sampled.", name, max_points)
        unit = _column_unit(name) or "value"
        wind_trace = go.Scatter(
            x=xs,
            y=ys,
            name=f"{name} ({unit})",
            mode="lines",
            line={
                "width": 1.2,
                "color": _SOLAR_WIND_COLORS[index % len(_SOLAR_WIND_COLORS)],
                "dash": "dot",
            },
            connectgaps=False,
            hovertemplate=(
                f"<b>{name}</b><br>%{{x|%H:%M}}<br>"
                f"{unit}: %{{y:.2f}}<extra></extra>"
            ),
        )
        if mixed_wind_units:
            fig.add_trace(wind_trace, row=2 + units.index(unit), col=1)
        else:
            wind_trace.yaxis = "y2"
            fig.add_trace(wind_trace)

    if mixed_wind_units:
        fig.update_yaxes(
            title_text=f"{ground_name} ({ground_unit})",
            tickfont={"color": _GROUND_TRACE_COLOR},
            title_font={"color": _GROUND_TRACE_COLOR},
            gridcolor="rgba(128,128,128,0.2)",
            row=1,
            col=1,
        )
        for index, unit in enumerate(units, start=2):
            names_for_unit = [
                name for name in sw_names if (_column_unit(name) or "value") == unit
            ]
            fig.update_yaxes(
                title_text=", ".join(names_for_unit) + f" ({unit})",
                gridcolor="rgba(128,128,128,0.2)",
                row=index,
                col=1,
            )
        fig.update_xaxes(
            title_text="Time (UTC)",
            type="date",
            tickformat="%H:%M",
            rangeslider={"visible": True, "thickness": 0.06},
            gridcolor="rgba(128,128,128,0.2)",
            row=1 + len(units),
            col=1,
        )
        fig.update_layout(height=max(600, 210 * (1 + len(units)) + 100))
    else:
        right_title = "Solar wind (" + ", ".join(units) + ")" if units else "Solar wind"
        fig.update_layout(
            xaxis={
                "title": "Time (UTC)",
                "type": "date",
                "tickformat": "%H:%M",
                "rangeslider": {"visible": True, "thickness": 0.06},
                "gridcolor": "rgba(128,128,128,0.2)",
            },
            yaxis={
                "title": {
                    "text": f"{ground_name} ({ground_unit})",
                    "font": {"color": _GROUND_TRACE_COLOR},
                },
                "tickfont": {"color": _GROUND_TRACE_COLOR},
                "gridcolor": "rgba(128,128,128,0.2)",
            },
            yaxis2={
                "title": {"text": right_title},
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            },
        )

    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        template="plotly_white",
        margin={"t": 90, "b": 70},
    )

    target = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    stem = _slug(f"{ground_name}_vs_{'_'.join(sw_names)}", "space_weather")
    out_name = filename or f"space_{stem}.html"
    if not out_name.endswith(".html"):
        out_name += ".html"
    path = _write(fig, target / out_name, include_plotlyjs)
    log.info(
        "Space-weather comparison plotted: %s vs %s, %s points",
        ground_name,
        ", ".join(sw_names),
        total,
    )
    return path


# --------------------------------------------------------------------------- #
# 3. magnetogram (classic 4-panel)
# --------------------------------------------------------------------------- #
def plot_magnetogram(
    df: pd.DataFrame,
    title: str = "Magnetogram",
    output_dir: str | Path | None = None,
    filename: str | None = None,
    max_points: int = GL_THRESHOLD,
    include_plotlyjs: Any = True,
) -> str | dict[str, Any]:
    """Draw a four-panel X/Y/Z/F magnetogram with a shared time axis.

    This is the presentation geophysicists expect from an observatory
    magnetogram: stacked traces, not overlaid ones, so each component keeps its
    own amplitude scale. Traces are aligned on a secondary y-axis so small
    components such as Z are not visually flattened by the much larger F.
    """
    problem = _check_frame(df, "df")
    if problem is not None:
        return problem
    prepared = _prepare(df, ("X", "Y", "Z", "F"), "df")
    if isinstance(prepared, dict):
        return prepared
    prepared, resolved = prepared
    # Panels are always X/Y/Z/F, but the frame may spell them differently;
    # index the exact resolved column so a casing slip cannot KeyError.
    column_of = {str(c).upper(): c for c in resolved}

    stamps = pd.to_datetime(prepared["timestamp"], errors="coerce")
    total = int(stamps.notna().sum())
    heavy = total > GL_THRESHOLD
    trace_cls = go.Scattergl if heavy else go.Scatter
    panels = ("X", "Y", "Z", "F")

    gap = 0.035
    panel_height = (1.0 - gap * (len(panels) - 1)) / len(panels)
    axis_refs = ["y"] + [f"y{i + 1}" for i in range(1, len(panels))]

    axis_kwargs: dict[str, Any] = {}
    for idx, (name, ref) in enumerate(zip(panels, axis_refs)):
        top = 1.0 - idx * (panel_height + gap)
        bottom = max(0.0, min(1.0, top - panel_height))
        axis_kwargs["yaxis" if idx == 0 else f"yaxis{idx + 1}"] = {
            "domain": [bottom, max(bottom, min(1.0, top))],
            "anchor": "x" if idx == 0 else "free",
            "position": 0.0,
            "title": {"text": f"{name} (nT)", "font": {"size": 12}},
            "gridcolor": "rgba(128,128,128,0.15)",
            "zeroline": False,
        }

    fig = go.Figure()
    for name, ref in zip(panels, axis_refs):
        values = pd.to_numeric(prepared[column_of[name]], errors="coerce")
        xs, ys, _ = _downsample(stamps, values, max_points)
        fig.add_trace(
            trace_cls(
                x=xs,
                y=ys,
                name=name,
                yaxis=ref,
                line={"width": 1, "color": _SERIES_COLOR.get(name)},
                connectgaps=False,
                hovertemplate=f"<b>{name}</b><br>%{{x}}<br>nT: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        xaxis={"title": "Time (UTC)", "gridcolor": "rgba(128,128,128,0.15)",
               "zeroline": False},
        hovermode="x",
        template="plotly_white",
        height=900,
        showlegend=False,
        margin={"t": 90, "l": 70},
        **axis_kwargs,
    )
    fig.update_xaxes(range=[stamps.min(), stamps.max()], rangeslider_visible=False)

    target = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    name_out = filename or f"magnetogram_{_slug(title)}.html"
    if not name_out.endswith(".html"):
        name_out += ".html"
    return _write(fig, target / name_out, include_plotlyjs)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _synthetic_day(day: int, n: int = 1440, seed: int = 0) -> pd.DataFrame:
    """Build a physically plausible day of minute data with a storm signature."""
    rng = np.random.default_rng(seed + day)
    stamps = pd.date_range(f"2025-03-{day:02d}", periods=n, freq="1min")
    t = np.arange(n) / 1440.0
    storm = 150.0 * np.exp(-(((t - 0.52) / 0.035) ** 2))
    h = 18000.0 + 40.0 * np.sin(2 * np.pi * t) - storm + rng.normal(0, 2.0, n)
    x = h + 800.0 * np.sin(2 * np.pi * t + 0.3)
    y = -1400.0 + 650.0 * np.cos(2 * np.pi * t) - 0.6 * storm
    z = 57400.0 + 350.0 * np.sin(2 * np.pi * t + 1.0) - 1.7 * storm
    return pd.DataFrame(
        {"timestamp": stamps, "X": x, "Y": y, "Z": z, "F": np.sqrt(x**2 + y**2 + z**2)}
    )


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 150)

    print("=" * 78)
    print("geomag_plotter self-test")
    print("=" * 78)

    day10 = _synthetic_day(10, seed=1)
    day11 = _synthetic_day(11, seed=2)

    print("\nSTEP 1 - plot_components (X, Y, Z, F overlaid)")
    p1 = plot_components(day10, components=["X", "Y", "Z", "F"],
                         title="IRT synthetic 2025-03-10")
    print(f"  -> {p1}")
    p1b = plot_components(day10, components=["H", "D", "I"],
                          title="IRT derived H D I", filename="derived_hdi.html")
    print(f"  -> {p1b}   (H/D/I auto-derived)")

    print("\nSTEP 2 - plot_comparison (H, two different days, time-of-day axis)")
    p2 = plot_comparison(day10, day11, component="H",
                         title="Comparison of 10 and 11 March 2025")
    print(f"  -> {p2}")

    print("\n  verifying the date-overlay actually collapses the axis:")
    a = _time_of_day(day10["timestamp"])
    b = _time_of_day(day11["timestamp"])
    print(f"    raw  day10[0] = {day10['timestamp'].iloc[0]}")
    print(f"    raw  day11[0] = {day11['timestamp'].iloc[0]}")
    print(f"    anchored day10[0] = {a.iloc[0]}")
    print(f"    anchored day11[0] = {b.iloc[0]}")
    print(f"    anchored series identical : {a.equals(b)}")
    print(f"    spans exactly 24h         : "
          f"{(a.max() - a.min()) == pd.Timedelta(hours=23, minutes=59)}")

    print("\nSTEP 3 - plot_magnetogram (4-panel)")
    p3 = plot_magnetogram(day10, title="IRT magnetogram 2025-03-10")
    print(f"  -> {p3}")

    print("\nSTEP 4 - NaN gaps are left as gaps, not bridged")
    gappy = day10.copy()
    gappy.loc[700:720, "X"] = np.nan
    p4 = plot_components(gappy, components=["X"], title="gap handling",
                         filename="gaps.html")
    print(f"  -> {p4}  (connectgaps=False)")

    print("\nSTEP 5 - large series auto-switch to WebGL + downsample")
    big = _synthetic_day(10, n=120000, seed=3)
    p5 = plot_components(big, components=["X"], title="large series",
                         filename="large.html", max_points=8000)
    print(f"  -> {p5}")
    size = Path(p5).stat().st_size / 1024
    print(f"  file size with 120k points capped at 8000: {size:.0f} KiB")
    p5b = plot_components(big, components=["X"], title="large cdn",
                          filename="large_cdn.html", max_points=8000,
                          include_plotlyjs="cdn")
    print(f"  -> {p5b}  ({Path(p5b).stat().st_size / 1024:.0f} KiB with CDN)")

    print("\nSTEP 6 - error contract")
    for label, out in (
        ("missing component", plot_components(day10, components=["Q"])),
        ("no timestamp", plot_components(day10.drop(columns=["timestamp"]))),
        ("not a DataFrame", plot_comparison("a", day11)),
        ("empty frame", plot_components(day10.iloc[0:0])),
        ("D on raw frame", plot_components(day10, components=["D"])),
    ):
        if isinstance(out, str):
            print(f"  {label:20s} -> OK {Path(out).name}")
        else:
            print(f"  {label:20s} -> {out['error']:18s} {out['message'][:60]}")

    print("\n" + "=" * 78)
    print(f"all artefacts written to: {DEFAULT_OUTPUT_DIR}")
    for f in sorted(DEFAULT_OUTPUT_DIR.glob("*.html")):
        print(f"  {f.name:34s} {f.stat().st_size / 1024:8.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
