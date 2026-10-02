"""Geographic and geomagnetic coordinate conversions for INTERMAGNET stations.

This module answers the questions that show up as soon as a storm is analysed
across more than one observatory: *how far apart are these two stations*, *why
do they see the disturbance differently*, and *how do I stack their curves on
one chart without them lying about each other*.

The dipole approximation, and its limits
----------------------------------------
:func:`geographic_to_geomagnetic` converts a geographic position into
**geomagnetic** coordinates using an idealised tilted dipole, and is the
standard first-order correction for why, for example, Tromso at 67N and
Nurmijarvi at 61N experience very different storm-time variations despite
similar geographic latitude. The geomagnetic pole sits near 80.7N, 72.7W, so
the field lines converge there and stations at high geomagnetic latitude see
much larger disturbances.

Being explicit about what this is not: the real field is offset from a dipole
by up to several hundred nT in declination (the longitude of the
geomagnetic pole moves over decades, and the eccentric dipole absorbs part of
that). This function therefore returns a **first-order approximation**, and
never a field value. It is the right tool for *ordering* stations by how close
they are to the pole, and the wrong tool for computing a declination. For
accuracy, use a full IGRF coefficient set or the published AACGM/CGM
transformations; the ``model`` field in the return payload records which
approximation produced the numbers so a caller cannot mistake one for the
other.

Why geomagnetic latitude beats geographic latitude for storm analysis
----------------------------------------------------------------------
The disturbance amplitude scales with geomagnetic latitude, not geographic.
Sorting by geomagnetic latitude therefore groups stations that *behave*
alike, which is what an overlay comparison needs: two stations at 65N and 67N
geographic can have geomagnetic latitudes 75 and 58, and their curves should
not be expected to match.

Overlay offsets
---------------
:func:`calculate_offsets_for_overlay` stacks curves vertically by adding a
constant to each one. The offset is a **display** device only and is never
written back into the stored data: mixing offsets into the values would make
the trace impossible to compare against the original series and would corrupt
any later numeric analysis. :func:`apply_offsets` returns a *new* frame.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "haversine_distance",
    "geographic_to_geomagnetic",
    "geomagnetic_to_geographic",
    "sort_stations_by_lat_m",
    "calculate_offsets_for_overlay",
    "apply_offsets",
    "station_coordinates",
    "is_error",
    "EARTH_RADIUS_KM",
    "GEOMAGNETIC_POLE",
    "DEFAULT_BASE_OFFSET",
]

log = logging.getLogger("geomag_coords")

#: Mean Earth radius in km (IUGG). The haversine formula is spherical, so this
#: is the radius of the sphere being assumed; real distances over regional
#: networks differ by at most ~0.5% from the ellipsoidal value.
EARTH_RADIUS_KM: float = 6371.0088

#: North geomagnetic pole of the centred dipole, epoch ~2020.
#: (80.7N, 72.7W) is the standard IAGA reference for a centred dipole; the true
#: dipole pole drifts and the eccentric dipole differs, which is exactly why
#: callers must treat the output as first-order.
GEOMAGNETIC_POLE: tuple[float, float] = (80.7, -72.7)

#: Default vertical separation between stacked curves, in nT.
DEFAULT_BASE_OFFSET: float = 100.0

#: Column names understood by the station helpers.
_LAT_KEYS = ("lat", "latitude", "geomag_lat", "geomagnetic_latitude", "mlat")
_LON_KEYS = ("lon", "lng", "longitude", "long", "geomag_lon", "geomagnetic_longitude")


def is_error(result: Any) -> bool:
    """True when a call failed. Lets an LLM branch without isinstance checks."""
    return isinstance(result, dict) and result.get("ok") is False


def _error(
    code: str,
    message: str,
    *,
    available: Sequence[str] | None = None,
    missing: Sequence[str] | None = None,
    hint: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": code,
        "message": message,
        "available": list(available) if available is not None else None,
        "missing": list(missing) if missing is not None else None,
    }
    if hint:
        payload["hint"] = hint
    payload.update(extra)
    return payload


def _as_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}={value!r} is not a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{label}={value!r} is not finite")
    return number


def _check_latitude(lat: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude {lat} is outside -90..90")


def _check_longitude(lon: float) -> None:
    # Longitudes are allowed to be any real number: 370 and -190 are legitimate
    # spellings of the same meridian, and rejecting them would be pedantry.
    # They are normalised internally with modulo arithmetic.
    if not math.isfinite(lon):
        raise ValueError("longitude must be finite")


# --------------------------------------------------------------------------- #
# 1. distance
# --------------------------------------------------------------------------- #
def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float | dict[str, Any]:
    """Great-circle distance between two points in kilometres.

    Uses the **haversine** form rather than the spherical law of cosines
    because the latter loses precision for short distances: ``acos`` of a
    number very close to 1 amplifies floating-point error, so two stations 100m
    apart can disagree noticeably. The haversine form subtracts before taking
    the square root and stays accurate at small separations.

    Angles are in **degrees**. Longitudes outside ``-180..180`` are normalised,
    so ``(lon=370)`` and ``(lon=10)`` are treated as the same meridian.
    Antipodal points are handled without a domain error.

    Identical points return exactly ``0.0``.
    """
    try:
        phi1 = math.radians(_as_float(lat1, "lat1"))
        phi2 = math.radians(_as_float(lat2, "lat2"))
        lam1 = math.radians(_as_float(lon1, "lon1"))
        lam2 = math.radians(_as_float(lon2, "lon2"))
        _check_latitude(math.degrees(phi1))
        _check_latitude(math.degrees(phi2))
        _check_longitude(lam1)
        _check_longitude(lam2)
    except ValueError as exc:
        return _error("invalid_input", str(exc))

    delta_phi = phi2 - phi1
    delta_lam = lam2 - lam1
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lam / 2.0) ** 2
    )
    # clamp guards the acos domain: rounding can push a just-under-1 value a
    # few ulps past 1 for nearly-antipodal points.
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# 2. geographic -> geomagnetic
# --------------------------------------------------------------------------- #
def _pole_frame(pole_lat: float, pole_lon: float) -> tuple[tuple[float, float, float], ...]:
    """Right-handed basis whose z axis is the geomagnetic pole.

    Built from the pole direction and the direction of increasing longitude
    there, so that a rotation expressed in this frame is a genuine rigid
    rotation of the sphere. Keeping it explicit is what makes
    :func:`geomagnetic_to_geographic` the exact transpose of
    :func:`geographic_to_geomagnetic` rather than a separately-derived formula
    that can drift from it.
    """
    phi_p = math.radians(pole_lat)
    lam_p = math.radians(pole_lon)
    z = (math.cos(phi_p) * math.cos(lam_p), math.cos(phi_p) * math.sin(lam_p), math.sin(phi_p))
    # Direction of increasing longitude at the pole.
    x = (-math.sin(lam_p), math.cos(lam_p), 0.0)
    # y = z x x, completing the right-handed set.
    y = (
        z[1] * x[2] - z[2] * x[1],
        z[2] * x[0] - z[0] * x[2],
        z[0] * x[1] - z[1] * x[0],
    )
    return (x, y, z)


def _validate_pole(pole_lat: Any, pole_lon: Any) -> tuple[float, float] | dict[str, Any]:
    try:
        phi_p = _as_float(pole_lat, "pole_lat")
        lam_p = _as_float(pole_lon, "pole_lon")
        _check_latitude(phi_p)
    except ValueError as exc:
        return _error(
            "invalid_input",
            str(exc),
            hint="The dipole pole latitude must be -90..90 degrees.",
        )
    return phi_p, lam_p


def geographic_to_geomagnetic(
    lat: float,
    lon: float,
    pole_lat: float = GEOMAGNETIC_POLE[0],
    pole_lon: float = GEOMAGNETIC_POLE[1],
) -> dict[str, Any] | float:
    """Convert geographic coordinates into geomagnetic ones (tilted dipole).

    Implemented as a rotation of the unit sphere into the frame whose z axis is
    the geomagnetic pole. Geomagnetic latitude is then the angle from that pole,
    so the pole itself maps to exactly 90 degrees and the equator stays the
    equator.

    Returns a dict with ``geomag_lat``, ``geomag_lon``, ``model='tilted_dipole'``
    and the pole actually used. Recording the model matters: this is a
    first-order approximation, and a caller must be able to see that it is not a
    full IGRF transform.

    Validation: a latitude outside ``-90..90`` is an error rather than being
    clamped, because a bad latitude is a data bug and clamping it would hide the
    bug behind a plausible-looking answer.
    """
    try:
        phi = _as_float(lat, "lat")
        lam = ( _as_float(lon, "lon") + 180.0) % 360.0 - 180.0
        _check_latitude(phi)
    except ValueError as exc:
        return _error(
            "invalid_input",
            str(exc),
            hint="Latitude must be -90..90 degrees; longitude is taken modulo 360.",
        )
    pole = _validate_pole(pole_lat, pole_lon)
    if isinstance(pole, dict):
        return pole
    phi_p, lam_p = pole

    x_ax, y_ax, z_ax = _pole_frame(phi_p, lam_p)
    phi_r = math.radians(phi)
    lam_r = math.radians(lam)
    # Position vector on the unit sphere in geographic coordinates.
    r = (math.cos(phi_r) * math.cos(lam_r), math.cos(phi_r) * math.sin(lam_r), math.sin(phi_r))
    # Project into the pole frame; the resulting spherical angles are geomagnetic.
    sin_lat = sum(r[i] * z_ax[i] for i in range(3))
    cos_lat_east = sum(r[i] * x_ax[i] for i in range(3))
    cos_lat_north = sum(r[i] * y_ax[i] for i in range(3))
    # Clamp before asin/atan2: rounding at the pole can push the dot product a
    # few ulps outside [-1, 1] and would otherwise raise ValueError.
    sin_lat = max(-1.0, min(1.0, sin_lat))
    geomag_lat = math.degrees(math.asin(sin_lat))
    # Longitude is undefined at either pole. Two separate singularities reach
    # this function and both must be caught:
    #
    # * the *geomagnetic* pole (lat == pole_lat): the perpendicular components
    #   are both ~0, so atan2 of two denormal floats returns an arbitrary angle;
    # * the *geographic* pole (lat == +-90): every meridian meets there, so the
    #   geographic longitude carries no information even though the projected
    #   components are healthy. Keying only on the component magnitudes -- the
    #   first version of this check -- reported a confident-looking 90 degrees
    #   for every longitude at the pole, and the value could not round trip.
    degenerate = (
        math.hypot(cos_lat_east, cos_lat_north) < 1e-9
        or abs(abs(phi) - 90.0) < 1e-9
    )
    geomag_lon = (
        None
        if degenerate
        else round(math.degrees(math.atan2(cos_lat_north, cos_lat_east)), 4)
    )

    return {
        "ok": True,
        "geomag_lat": round(geomag_lat, 4),
        "geomag_lon": geomag_lon,
        "degenerate": degenerate,
        "model": "tilted_dipole",
        "pole_lat": phi_p,
        "pole_lon": lam_p,
        "note": (
            "First-order dipole approximation; not a full IGRF or AACGM "
            "transform. Use for ordering stations by geomagnetic latitude."
        ),
    }


def geomagnetic_to_geographic(
    geomag_lat: float,
    geomag_lon: float,
    pole_lat: float = GEOMAGNETIC_POLE[0],
    pole_lon: float = GEOMAGNETIC_POLE[1],
) -> dict[str, Any] | float:
    """Inverse of :func:`geographic_to_geomagnetic`.

    Applies the transpose rotation, so it is exact up to floating point for any
    pair of angles -- which is what lets the transform be verified by round trip
    rather than against a table of expected values.
    """
    try:
        phi_m = _as_float(geomag_lat, "geomag_lat")
        lam_m = _as_float(geomag_lon, "geomag_lon")
        _check_latitude(phi_m)
    except ValueError as exc:
        return _error("invalid_input", str(exc))
    pole = _validate_pole(pole_lat, pole_lon)
    if isinstance(pole, dict):
        return pole
    phi_p, lam_p = pole

    x_ax, y_ax, z_ax = _pole_frame(phi_p, lam_p)
    phi_r = math.radians(phi_m)
    lam_r = math.radians(lam_m)
    cos_lat = math.cos(phi_r)
    # Point on the unit sphere in geomagnetic coordinates...
    m = (
        cos_lat * math.cos(lam_r),
        cos_lat * math.sin(lam_r),
        math.sin(phi_r),
    )
    # ...mapped back through the basis. The basis is orthonormal, so composing
    # along it is the inverse rotation:
    #     r_j = sum_i m_i * axis_i[j]
    # Note the index order: summing m against each axis *vector* would compute a
    # dot product instead, which is the transpose of the rotation.
    basis = (x_ax, y_ax, z_ax)
    r = tuple(
        sum(m[i] * basis[i][j] for i in range(3)) for j in range(3)
    )
    sin_phi = max(-1.0, min(1.0, r[2]))
    lat = math.degrees(math.asin(sin_phi))
    # Degeneracy is keyed on the INPUT geomagnetic latitude, not the recovered
    # one: at geomag_lat = +-90 the point is the pole, every meridian meets
    # there, and its geomagnetic longitude is meaningless. Testing the output
    # latitude would miss it -- the pole's *geographic* latitude is 80.7, nowhere
    # near 90 -- which is why the first version of this check never fired.
    degenerate = abs(abs(phi_m) - 90.0) < 1e-9
    lon = (
        None
        if degenerate
        else (math.degrees(math.atan2(r[1], r[0])) + 180.0) % 360.0 - 180.0
    )

    return {
        "ok": True,
        "lat": round(lat, 4),
        "lon": None if degenerate else round(lon, 4),
        "degenerate": degenerate,
        "pole_lat": phi_p,
        "pole_lon": lam_p,
        "model": "tilted_dipole_inverse",
    }


# --------------------------------------------------------------------------- #
# 3. station helpers
# --------------------------------------------------------------------------- #
def _station_value(station: Mapping[str, Any] | Any, keys: Sequence[str]) -> Any:
    """Fetch the first present key from a dict-like station, else try attrs."""
    if isinstance(station, Mapping):
        for key in keys:
            if key in station and station[key] is not None:
                return station[key]
        return None
    for key in keys:
        value = getattr(station, key, None)
        if value is not None:
            return value
    return None


def station_coordinates(
    stations: Iterable[Mapping[str, Any] | Any],
) -> tuple[list[dict[str, Any]] | list[dict[str, Any]], list[dict[str, Any]]]:
    """Split station-like objects into ``(usable, unusable)``.

    Accepts mappings or objects with attributes. A station is usable when a
    latitude (geographic or geomagnetic) and a longitude can be read from it;
    unusable entries are returned with a reason so a caller can report *which*
    station is missing coordinates instead of silently dropping it.
    """
    usable: list[dict[str, Any]] = []
    unusable: list[dict[str, Any]] = []
    for station in stations:
        lat = _station_value(station, _LAT_KEYS)
        lon = _station_value(station, _LON_KEYS)
        name = _station_value(station, ("code", "name", "station", "station_code")) or "?"
        has_geomag = _station_value(station, ("geomag_lat", "geomagnetic_latitude", "mlat")) is not None
        if lat is None or lon is None:
            unusable.append({"station": name, "reason": "missing latitude/longitude"})
            continue
        try:
            lat_f = _as_float(lat, "lat")
            lon_f = _as_float(lon, "lon")
            _check_latitude(lat_f)
        except ValueError as exc:
            unusable.append({"station": name, "reason": str(exc)})
            continue
        usable.append(
            {
                "station": name,
                "lat": lat_f,
                "lon": lon_f,
                "has_geomag": has_geomag,
                # The original object, carried through so callers can read
                # columns station_coordinates() does not itself normalise
                # (such as a supplied geomagnetic latitude). Keyed on the entry
                # rather than zipped afterwards, because an unusable station
                # between two usable ones would silently misalign a zip().
                "_source": station,
            }
        )
    return usable, unusable


def sort_stations_by_lat_m(
    stations: Iterable[Mapping[str, Any] | Any],
    descending: bool = False,
    use_geographic: bool = True,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Sort stations by geomagnetic latitude for an overlay comparison.

    A station that already carries a geomagnetic latitude uses it directly. One
    that carries only geographic coordinates is converted with
    :func:`geographic_to_geomagnetic` and tagged ``'source': 'derived'``, so
    the caller can tell a supplied geomagnetic latitude from a computed
    approximation. Sorting is ascending by geomagnetic latitude unless
    ``descending`` (equatorward-first ordering).

    Stations missing usable coordinates are dropped and reported under
    ``'skipped'`` on the returned list-as-dict, rather than being assigned a
    made-up latitude.
    """
    usable, unusable = station_coordinates(stations)
    resolved: list[dict[str, Any]] = []
    for entry in usable:
        # Read a supplied geomagnetic latitude from the ORIGINAL object: the
        # normalised entry keeps only lat/lon, so looking for it there discarded
        # it and every station silently fell through to being recomputed.
        provided = _station_value(
            entry.get("_source"), ("geomag_lat", "geomagnetic_latitude", "mlat")
        )
        if provided is not None:
            try:
                provided_value = _as_float(provided, "geomag_lat")
            except ValueError:
                pass  # unusable value: fall through to the geographic conversion
            else:
                resolved.append(
                    {
                        key: value for key, value in entry.items() if key != "_source"
                    }
                    | {"geomag_lat": provided_value, "source": "provided"}
                )
                continue
        converted = geographic_to_geomagnetic(entry["lat"], entry["lon"])
        if is_error(converted):
            unusable.append({"station": entry["station"], "reason": converted["message"]})
            continue
        resolved.append(
            {
                # _source is an implementation detail and must not reach the
                # caller: it holds a reference to the caller's own object.
                **{k: v for k, v in entry.items() if k != "_source"},
                "geomag_lat": converted["geomag_lat"],
                "geomag_lon": converted["geomag_lon"],
                # Always "derived": reaching here means no usable geomag_lat was
                # supplied, whatever has_geomag said about the key's presence.
                "source": "derived",
            }
        )

    resolved.sort(key=lambda item: item["geomag_lat"], reverse=descending)
    return {
        "ok": True,
        "stations": resolved,
        "skipped": unusable,
        "order": "descending" if descending else "ascending",
    }





def calculate_offsets_for_overlay(
    stations: Iterable[Mapping[str, Any] | Any],
    base_offset: float = DEFAULT_BASE_OFFSET,
) -> dict[str, Any] | list[float]:
    """Vertical offsets that keep overlaid curves from overlapping.

    Returns ``base_offset * rank``: the first station sits at 0, the second at
    ``base_offset``, and so on. Because the offset is proportional to rank, two
    adjacent traces are always exactly ``base_offset`` nT apart regardless of
    their amplitude -- which keeps a quiet-day station and a storm-day station
    equally legible. Curves whose vertical range exceeds ``base_offset`` will
    still overlap, so the value should be chosen with the known field
    variation in mind (100 nT suits the ~10-100 nT variations of a storm).

    Offsets are keyed by station name and are a **display-only** quantity; see
    :func:`apply_offsets`.
    """
    try:
        offset = _as_float(base_offset, "base_offset")
    except ValueError as exc:
        return _error("invalid_input", str(exc))
    if offset == 0.0:
        return _error(
            "invalid_input",
            "base_offset=0 would stack every curve on top of the others.",
        )

    usable, unusable = station_coordinates(stations)
    if not usable:
        return _error(
            "no_stations",
            "No station with usable coordinates was supplied.",
            hint="Each station needs a latitude and a longitude.",
        )
    offsets = {
        entry["station"]: offset * rank for rank, entry in enumerate(usable)
    }
    return {
        "ok": True,
        "offsets": offsets,
        "base_offset": offset,
        "n_stations": len(usable),
        "skipped": unusable,
        "note": "Display-only vertical offsets; never written back to the data.",
    }


def apply_offsets(
    df: pd.DataFrame,
    offset: float,
    column: str = "H",
    new_column: str | None = None,
) -> pd.DataFrame | dict[str, Any]:
    """Return a **copy** of ``df`` with ``offset`` added to ``column``.

    The input is never mutated. Writing the offset into a *new* column (rather
    than overwriting the component) is what keeps the offset from corrupting the
    real data: the original ``H`` remains available for any numeric analysis,
    and the shifted copy exists purely for plotting.
    """
    if not isinstance(df, pd.DataFrame):
        return _error("invalid_input", f"Expected a DataFrame, got {type(df).__name__}.")
    if column not in df.columns:
        return _error(
            "missing_columns",
            f"{column!r} is not in this frame.",
            available=[str(c) for c in df.columns],
        )
    try:
        value = _as_float(offset, "offset")
    except ValueError as exc:
        return _error("invalid_input", str(exc))

    out = df.copy()
    target = new_column or f"{column}_offset"
    out[target] = out[column].astype("float64") + value
    return out