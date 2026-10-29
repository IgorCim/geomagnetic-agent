"""Canonical chart names and the metadata that explains them.

The model chooses *what* to plot: a station, a date, components, a source
handle.  It must not choose the filename.  Filenames are a pure function of
those choices, so two runs of the same chart cannot disagree about what the
file is called, and a chart filed into a project keeps the name it was drawn
under.

The source is deliberately the last token before ``.html``::

    IRT_2024-12-10_F_INTERMAGNET.html
    IRT_2024-05-14_H_vs_OMNI_2024-05-01_Bz-V_INTERMAGNET-OMNI.html

That makes the station/date/component prefix sort naturally while keeping the
source visible.  Sorting itself is *not* filename sorting: the project index
sorts on the metadata fields, so a user can ask for date-first, source-first or
a custom source order without renaming anything.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "KNOWN_SOURCES",
    "MAX_STEM",
    "HandleInfo",
    "parse_handle",
    "sources_for_handles",
    "source_from_filename",
    "components_filename",
    "comparison_filename",
    "overlay_filename",
    "space_weather_filename",
    "ensure_source_suffix",
    "safe_filename",
    "components_title",
    "comparison_title",
    "overlay_title",
    "space_weather_title",
    "human_date",
    "chart_metadata",
    "describe_handles",
]

KNOWN_SOURCES = ("INTERMAGNET", "OMNI", "KYOTO", "SUPERMAG", "NOAA", "ISWA")
_ALLOWED_SOURCE_TOKENS = set(KNOWN_SOURCES) | {"UNKNOWN"}
MAX_STEM = 140

_SOURCE_BY_FAMILY = {
    "raw": "INTERMAGNET",
    "derived": "INTERMAGNET",
    "smag": "SUPERMAG",
}
_SOURCE_BY_NAMESPACE = {
    "omni": "OMNI",
    "kyoto": "KYOTO",
    "supermag": "SUPERMAG",
    "noaa": "NOAA",
    "iswa": "ISWA",
}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SAFE_RE = re.compile(r"[^0-9A-Za-z._-]+")
_MULTI_US_RE = re.compile(r"_+")


@dataclass(frozen=True)
class HandleInfo:
    """What a frame handle says about provenance and cell coordinates."""

    handle: str
    family: str
    station: str | None
    start: str | None
    source: str
    dataset: str | None = None

    @property
    def located(self) -> bool:
        """True when the handle names a station or a date we can put in a name."""
        return bool(self.station or self.start)


def _safe_token(value: Any, fallback: str = "") -> str:
    """A filename-safe token that keeps case (``Bz`` must stay ``Bz``)."""
    text = str(value or "").strip()
    text = _SAFE_RE.sub("_", text)
    text = _MULTI_US_RE.sub("_", text).strip("._-")
    return text or fallback


def _date_token(value: Any) -> str:
    text = str(value or "").strip()
    if not _DATE_RE.fullmatch(text):
        return ""
    try:
        date.fromisoformat(text)
    except ValueError:
        return ""
    return text


def _date_span(start: Any, end: Any = None) -> str:
    first = _date_token(start)
    last = _date_token(end)
    if first and last and last != first:
        return f"{first}_{last}"
    return first or last


def _components_token(components: Iterable[Any]) -> str:
    if isinstance(components, str):
        components = [components]
    tokens: list[str] = []
    seen: set[str] = set()
    for component in components:
        token = _safe_token(component)
        key = token.casefold()
        if token and key not in seen:
            seen.add(key)
            tokens.append(token)
    return "-".join(tokens)


def _sources_token(sources: Iterable[Any]) -> str:
    """A validated, de-duplicated source suffix (``INTERMAGNET-OMNI``).

    A source is an enum, not free text.  An arbitrary long token from a legacy
    filename must become ``UNKNOWN`` rather than a 200-character suffix that
    makes the chart impossible to open on Windows.
    """
    unique: list[str] = []
    for source in sources:
        text = str(source or "").strip().upper()
        for piece in text.split("-"):
            piece = piece.strip()
            if not piece:
                continue
            if piece not in _ALLOWED_SOURCE_TOKENS:
                piece = "UNKNOWN"
            if piece not in unique:
                unique.append(piece)
    return "-".join(unique) or "UNKNOWN"


def _limit_stem(stem: str, suffix: str) -> str:
    """Bound the name while keeping the source token last."""
    if len(stem) <= MAX_STEM:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    max_suffix = max(1, MAX_STEM - len(digest) - 4)
    if len(suffix) > max_suffix:
        suffix = suffix[:max_suffix].rstrip("._-") or "UNKNOWN"
    room = MAX_STEM - len(suffix) - len(digest) - 3
    head = stem[: max(1, room)].rstrip("._-")
    return f"{head}__{digest}_{suffix}"


def _finalize(parts: Iterable[Any], sources: Iterable[Any]) -> str:
    suffix = _sources_token(sources)
    body = "_".join(
        token for token in (_safe_token(part) for part in parts) if token
    )
    stem = f"{body}_{suffix}" if body else suffix
    return _limit_stem(stem, suffix) + ".html"


def parse_handle(handle: Any) -> HandleInfo:
    """Parse ``raw:irt:2024-12-10`` and friends into provenance.

    Unknown families are kept as ``UNKNOWN`` rather than guessed: a wrong
    source tag is worse than an honest one, because it survives into the
    project manifest and the archive index.
    """
    text = str(handle or "").strip()
    parts = text.split(":")
    family = parts[0].strip().lower() if parts else ""
    dataset = parts[1].strip().lower() if len(parts) > 1 else None
    station: str | None = None
    start: str | None = None

    if family in ("raw", "derived", "smag"):
        if len(parts) > 1:
            station = _safe_token(parts[1]).upper() or None
        if len(parts) > 2:
            # An invalid calendar date must not become a date token.  A handle
            # like ``raw:irt:2024-13-01`` is not a usable cell coordinate.
            start = _date_token(parts[2]) or None
        source = _SOURCE_BY_FAMILY.get(family, "UNKNOWN")
    elif family in ("sw", "idx"):
        if len(parts) > 2:
            start = _date_token(parts[2]) or None
        source = _SOURCE_BY_NAMESPACE.get(dataset or "", "UNKNOWN")
    else:
        if len(parts) > 1:
            station = _safe_token(parts[1]).upper() or None
        if len(parts) > 2:
            start = _date_token(parts[2]) or None
        source = _SOURCE_BY_FAMILY.get(family) or _SOURCE_BY_NAMESPACE.get(
            family, "UNKNOWN"
        )

    return HandleInfo(
        handle=text,
        family=family,
        station=station,
        start=start,
        source=source,
        dataset=dataset,
    )


def describe_handles(handles: Iterable[Any]) -> dict[str, list[str]]:
    if isinstance(handles, str):
        handles = [handles]
    infos = [parse_handle(handle) for handle in handles if str(handle or "").strip()]
    sources: list[str] = []
    stations: list[str] = []
    dates: list[str] = []
    for info in infos:
        if info.source not in sources:
            sources.append(info.source)
        if info.station and info.station not in stations:
            stations.append(info.station)
        if info.start and info.start not in dates:
            dates.append(info.start)
    return {
        "sources": sources or ["UNKNOWN"],
        "stations": stations,
        "dates": dates,
        "handles": [info.handle for info in infos],
    }


def sources_for_handles(handles: Iterable[Any]) -> list[str]:
    return describe_handles(handles)["sources"]


def components_filename(
    handle: Any,
    components: Sequence[Any],
    *,
    variant: str | None = None,
) -> str:
    """``IRT_2024-12-10_F_INTERMAGNET.html`` for a component chart."""
    info = parse_handle(handle)
    parts: list[Any] = []
    if info.station:
        parts.append(info.station)
    if info.start:
        parts.append(_date_span(info.start))
    component_token = _components_token(components)
    if component_token:
        parts.append(component_token)
    if variant is None:
        # H/D/I are the derived products and X/Y/Z/F are the raw ones.  A
        # derived frame also carries the raw columns, so plotting F from
        # ``derived:...`` would otherwise collide with raw F.  Tag only that
        # overlap; the normal component charts keep the short canonical name.
        requested = {str(c).strip().upper() for c in components}
        if info.family == "derived" and requested & {"X", "Y", "Z", "F"}:
            variant = "derived"
    if variant:
        parts.append(variant)
    return _finalize(parts, [info.source])


def comparison_filename(handle1: Any, handle2: Any, component: Any) -> str:
    """Name a two-frame comparison, e.g. ``..._vs_..._H_INTERMAGNET.html``."""
    first = parse_handle(handle1)
    second = parse_handle(handle2)
    parts: list[Any] = []
    if first.station:
        parts.append(first.station)
    if first.start:
        parts.append(first.start)
    parts.append("vs")
    if second.station and second.station != first.station:
        parts.append(second.station)
    if second.start:
        parts.append(second.start)
    component_token = _components_token([component])
    if component_token:
        parts.append(component_token)
    return _finalize(parts, [first.source, second.source])


def overlay_filename(
    stations: Sequence[Any],
    start: Any,
    component: Any,
    sources: Iterable[Any] | None = None,
) -> str:
    """Name a multi-station overlay, with the source still last."""
    if isinstance(stations, str):
        stations = [stations]
    station_tokens: list[str] = []
    for station in stations:
        token = _safe_token(station).upper()
        if token and token not in station_tokens:
            station_tokens.append(token)
    station_token = "-".join(station_tokens)
    parts: list[Any] = ["OVERLAY"]
    if station_token:
        parts.append(station_token)
    date_token = _date_span(start)
    if date_token:
        parts.append(date_token)
    component_token = _components_token([component])
    if component_token:
        parts.append(component_token)
    return _finalize(parts, sources or ["INTERMAGNET"])


def space_weather_filename(
    ground_handle: Any,
    solar_wind_handle: Any,
    ground_component: Any,
    solar_wind_components: Sequence[Any],
) -> str:
    """Name a ground-vs-solar-wind chart.

    The solar-wind side gets its source name in the body as well as the suffix:
    ``..._vs_OMNI_...`` reads correctly even after the suffix is stripped by a
    human renaming a file.
    """
    ground = parse_handle(ground_handle)
    solar = parse_handle(solar_wind_handle)
    parts: list[Any] = []
    if ground.station:
        parts.append(ground.station)
    if ground.start:
        parts.append(ground.start)
    ground_token = _components_token([ground_component])
    if ground_token:
        parts.append(ground_token)
    parts.append("vs")
    if solar.station:
        parts.append(solar.station)
    else:
        parts.append(solar.source or (solar.dataset or "").upper() or "SOLAR")
    if solar.start:
        parts.append(solar.start)
    solar_token = _components_token(solar_wind_components)
    if solar_token:
        parts.append(solar_token)
    return _finalize(parts, [ground.source, solar.source])


def _source_suffix_parts(stem: str) -> tuple[str, list[str]]:
    """Split the trailing source run from *stem*.

    The canonical spelling is one hyphenated token (``INTERMAGNET-OMNI``), but
    older charts used two underscore-separated tokens
    (``INTERMAGNET_OMNI``).  Both spellings describe the same provenance, so
    the whole trailing run is consumed and returned as canonical source pieces.
    """
    parts = stem.split("_")
    end = len(parts)
    tokens: list[str] = []
    while end > 1:
        pieces = parts[end - 1].upper().split("-")
        if not pieces or not all(piece in _ALLOWED_SOURCE_TOKENS for piece in pieces):
            break
        tokens.insert(0, parts[end - 1].upper())
        end -= 1
    canonical: list[str] = []
    for token in tokens:
        for piece in token.split("-"):
            if piece and piece not in canonical:
                canonical.append(piece)
    return "_".join(parts[:end]), canonical


def _strip_source_suffix(stem: str) -> str:
    """Drop one or more trailing source tokens.

    Handles both ``..._INTERMAGNET-OMNI`` and the older two-token spelling
    ``..._INTERMAGNET_OMNI``.  Only the trailing run is removed, so a station
    or component named like a source cannot be mistaken for provenance.
    """
    return _source_suffix_parts(stem)[0]


def ensure_source_suffix(filename: Any, sources: Iterable[Any]) -> str:
    """Append the source suffix to an explicit filename if it is missing.

    An explicit name is a user preference, not an escape hatch from provenance:
    ``mine.html`` becomes ``mine_INTERMAGNET.html``.  A stale or partial source
    suffix is replaced, so adding OMNI to an INTERMAGNET name does not produce
    ``..._INTERMAGNET_INTERMAGNET-OMNI``.
    """
    name = Path(str(filename or "").strip()).name
    if not name:
        name = "chart.html"
    stem = name[:-5] if name.lower().endswith(".html") else name
    suffix = _sources_token(sources)
    stem = _strip_source_suffix(_safe_token(stem, "chart"))
    return _limit_stem(f"{stem}_{suffix}", suffix) + ".html"


def safe_filename(filename: Any, sources: Iterable[Any] | None = None) -> str:
    """A safe, canonical ``.html`` name with a source token last.

    If the caller does not know the source, the existing suffix is preserved;
    a name without one gets ``_UNKNOWN`` rather than silently losing provenance.
    """
    name = Path(str(filename or "").strip()).name
    if not name:
        name = "chart.html"
    if sources is None:
        sources = [source_from_filename(name)]
    return ensure_source_suffix(name, sources)


def source_from_filename(filename: Any) -> str:
    """Read the source tag from a canonical filename, or return ``UNKNOWN``.

    Legacy two-token suffixes are normalised instead of losing their first
    source: ``..._INTERMAGNET_OMNI.html`` reads as ``INTERMAGNET-OMNI``, and
    :func:`safe_filename` rewrites it to the canonical one-token spelling.
    """
    name = Path(str(filename or "").strip()).name
    stem = name[:-5] if name.lower().endswith(".html") else name
    _, sources = _source_suffix_parts(stem)
    return "-".join(sources) if sources else "UNKNOWN"


def human_date(value: Any) -> str:
    """``2024-12-10`` -> ``10.12.2024`` for a chart title."""
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text).strftime("%d.%m.%Y")
    except ValueError:
        return text


def _human_components(components: Sequence[Any]) -> str:
    if isinstance(components, str):
        components = [components]
    return "/".join(str(c).strip() for c in components if str(c or "").strip())


def components_title(handle: Any, components: Sequence[Any]) -> str:
    info = parse_handle(handle)
    bits = []
    if info.station:
        bits.append(info.station)
    if info.start:
        bits.append(human_date(info.start))
    component_text = _human_components(components)
    if component_text:
        bits.append(component_text)
    if info.source and info.source != "UNKNOWN":
        bits.append(f"— {info.source}")
    return " ".join(bits) or "Geomagnetic components"


def comparison_title(handle1: Any, handle2: Any, component: Any) -> str:
    first = parse_handle(handle1)
    second = parse_handle(handle2)
    left = " ".join(
        part for part in (first.station, human_date(first.start)) if part
    )
    right = " ".join(
        part for part in (second.station, human_date(second.start)) if part
    )
    source = _sources_token([first.source, second.source])
    return f"{left or 'first'} vs {right or 'second'} — {component} ({source})"


def overlay_title(stations: Sequence[Any], start: Any, component: Any) -> str:
    codes = ", ".join(str(s).strip().upper() for s in stations if str(s or "").strip())
    return f"Overlay {codes} {human_date(start)} — {component}"


def space_weather_title(
    ground_handle: Any,
    solar_wind_handle: Any,
    ground_component: Any,
    solar_wind_components: Sequence[Any],
) -> str:
    ground = parse_handle(ground_handle)
    solar = parse_handle(solar_wind_handle)
    left = " ".join(
        part for part in (ground.station, human_date(ground.start), ground_component)
        if part
    )
    source = solar.source if solar.source != "UNKNOWN" else solar.dataset
    # A descriptor such as ``sw:omni:2024-05-01`` names the dataset start, not
    # the slice drawn on this chart.  Printing it as a data date ("OMNI
    # 01.05.2024" on an 18 May chart) is misleading.  Keep the descriptor
    # visible, but only use a human date when the two handles share it.
    if solar.start and ground.start and solar.start == ground.start:
        wind_label = " ".join(part for part in (source, human_date(solar.start)) if part)
    elif solar.start:
        handle = str(solar_wind_handle or "").strip()
        wind_label = (
            f"{source} ({handle})"
            if ":" in handle
            else " ".join(part for part in (source, human_date(solar.start)) if part)
        )
    else:
        wind_label = source
    right = " ".join(
        part
        for part in (wind_label, _human_components(solar_wind_components))
        if part
    )
    return f"{left or 'ground'} vs {right or 'solar wind'}"


def chart_metadata(
    filename: Any,
    handles: Iterable[Any],
    kind: str,
    *,
    components: Sequence[Any] | None = None,
    product: str | None = None,
    relative_path: str | None = None,
    sha256: str | None = None,
    size: int | None = None,
) -> dict[str, Any]:
    """The machine-readable description stored next to every filed chart."""
    described = describe_handles(handles)
    if isinstance(components, str):
        components = [components]
    component_list = [str(c).strip() for c in (components or []) if str(c or "").strip()]
    return {
        "file": Path(str(filename or "")).name,
        "relative_path": relative_path,
        "source": "-".join(described["sources"]),
        "sources": described["sources"],
        "station": described["stations"][0] if described["stations"] else "",
        "stations": described["stations"],
        "date": described["dates"][0] if described["dates"] else "",
        "dates": described["dates"],
        "components": component_list,
        "kind": str(kind or "chart"),
        "product": product,
        "handles": described["handles"],
        "sha256": sha256,
        "bytes": size,
    }
