"""ISWA (Integrated Space Weather Analysis) loader -- Phase 6.4.

Two ISWA products answer the two questions this module exists for:

* **"How does the storm look right now, globally?"** -- a Ground Magnetic
  Perturbation Map, an OGC WMS ``GetMap`` picture of the whole globe.
  **(Decommissioned.)**
* **"What comes next?"** -- a merged Dst/Kp outlook.

The WMS layer is retired
------------------------
NASA CCMC migrated ISWA to ``https://iswa.ccmc.gsfc.nasa.gov/`` and the public
``ground_mag_perturbation`` WMS layer did not come with it. Checked 2026-10-20
against the new system itself:

* its full catalog (``https://iswa.ccmc.gsfc.nasa.gov/api/info``, 1351
  datafeeds) has no ``perturbation``/``ground magnet`` feed, no ``/ground``
  datatree, and no ``wms`` protocol at all (the protocols are ftp/https/mv/sftp);
* every WMS-shaped path on the new domain -- ``/wms``, ``/iswa/wms``,
  ``/geoserver/wms``, ``/maps/wms``, ``/app/wms`` -- answers HTTP 404 bare and
  HTTP 403 to ``SERVICE=WMS&REQUEST=GetCapabilities`` (NASA's WAF refuses the
  keyword before Apache sees it);
* the new app serves imagery through a servlet/REST file API
  (``FilesInRangeServlet`` / ``api/search`` -> files under ``iswa_data_tree``),
  not map tiles, and that API exposes no ground-perturbation product.

So there is no endpoint to retry against: the layer is *gone*, not flaky.
:func:`get_ground_perturbation_map` therefore answers ``service_decommissioned``
immediately -- no request is sent -- and its ``fallback_hint`` points the model
at the SuperMAG indices it already has (``fetch_supermag_data(data_type=
'indices')`` for SME/SML/SMU, the same science question) and at
:func:`get_iswa_forecast` for the Dst/Kp outlook. The live WMS client is kept
intact and reachable via ``decommissioned=False``: a deployment pointed at a
still-live map service (or a restored layer) fetches a real PNG with exactly
the documented request, and a 403/404 on that path also resolves to
``service_decommissioned``. ``build_wms_url`` still builds the documented
request, pinned by offline tests.

The live ISWA data system is the NASA CCMC HAPI service at
``https://iswa.ccmc.gsfc.nasa.gov/hapi`` -- ``data?id=<dataset>&time.min=
<ISO>&time.max=<ISO>&format=json``. Note the ``time.min``/``time.max``
spelling: this server does not accept the HAPI 2.0 ``start``/``stop`` pair,
it answers ``code 1400: 'time.min' parameter is required``. The two datasets
behind :func:`get_dst_kp_forecast` are:

* ``NOAA_KP_P3H`` -- NOAA SWPC 3-hourly Kp with ``Kp_observed``,
  ``Kp_estimated`` and ``Kp_predicted`` columns. Its catalogue stops three
  days past "now", which *is* the forecast: the future rows carry only the
  predicted/estimated columns.
* ``dst_quicklook`` -- the WDC Kyoto quicklook Dst, hourly, current to
  within an hour of now (verified: stopDate tracks the current day).

No ISWA dataset forecasts Dst. The SWMF real-time Dst series
(``SWMF2011_RT_DST_P1M`` stops 2025-08-21, ``SWMF2023_RT_GMlog_P1M`` stops
2025-12-16, ``SWMF2008_RT_DST_P1M`` stopped in 2017) are all stale, so
``dst`` in the payload is the Kyoto *nowcast* and is ``None`` for
timestamps ahead of the last observation. ``sources`` says exactly that;
the loader never invents a Dst prediction.

Column selection for Kp
-----------------------
Each row may carry several Kp spellings, so the preference is
observed -> predicted -> estimated -> reported (``KP_3H``): an observation is
never replaced by a model value, and a future bucket (where ``Kp_observed`` is
null) falls through to ``Kp_predicted``. ``kp_kind`` in every row records which
one was used, so the caller can never present a prediction as an observation.
"""

from __future__ import annotations

import logging
import struct
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)

__all__ = [
    "WMS_ENDPOINT",
    "WMS_LAYER",
    "HAPI_BASE_URL",
    "KP_DATASET",
    "DST_DATASET",
    "FORECAST_HOURS_BACK",
    "FORECAST_HOURS_AHEAD",
    "FORECAST_STEP_HOURS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "build_wms_url",
    "get_ground_perturbation_map",
    "get_dst_kp_forecast",
    "parse_timestamp",
    "iso_z",
    "save_map_png",
    "synthetic_map_png",
    "is_error",
]


# --------------------------------------------------------------------------- #
# endpoints and defaults
# --------------------------------------------------------------------------- #
#: The WMS the brief names for ground magnetic perturbation maps.
WMS_ENDPOINT = "https://iswa.gsfc.nasa.gov/wmts/wmts.cgi"
WMS_VERSION = "1.1.1"
WMS_LAYER = "ground_mag_perturbation"
WMS_SRS = "EPSG:4326"
WMS_FORMAT = "image/png"

#: The live ISWA data system (NASA CCMC), HAPI protocol.
HAPI_BASE_URL = "https://iswa.ccmc.gsfc.nasa.gov/hapi"
KP_DATASET = "NOAA_KP_P3H"
DST_DATASET = "dst_quicklook"

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3

#: One :func:`get_dst_kp_forecast` window: a day of context behind "now" (the
#: Dst quicklook stops at the last observation, so the past half is what gives
#: the rows real Dst values) and two days of Kp outlook ahead. The Kp dataset
#: itself runs ~3 days past now, so the 48 h ask is always inside its range.
FORECAST_HOURS_BACK = 24
FORECAST_HOURS_AHEAD = 48
FORECAST_STEP_HOURS = 3

#: Where a map lands: ``plots/`` beside the other artefacts.
DEFAULT_OUTPUT_DIR = Path("plots")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_DEFAULT_BBOX = (-180.0, -90.0, 180.0, 90.0)

_UNAVAILABLE_HINT = (
    "The ISWA WMS endpoint is unreachable or refused the request. It is probed "
    "as-is (no credentials, no session): check connectivity and retry. For the "
    "global storm picture meanwhile use fetch_supermag_data(data_type='indices') "
    "for SME/SML/SMU, and get_iswa_forecast() for the Dst/Kp outlook."
)

#: What the public WMS map layer became after NASA CCMC migrated ISWA to
#: ``iswa.ccmc.gsfc.nasa.gov``: the old ``ground_mag_perturbation`` layer is
#: gone. Its full catalog (``/api/info``, 1351 datafeeds) carries no
#: ground-perturbation map feed at all, no WMS protocol, and every WMS-shaped
#: path on the new domain is answered with 403 by the WAF -- so there is
#: nothing to retry against. The map tool therefore degrades gracefully and
#: points the model at the SuperMAG indices it already has.
_DECOMMISSIONED_MESSAGE = (
    "NASA ISWA has decommissioned the public WMS ground magnetic perturbation "
    "map layer."
)
_DECOMMISSIONED_FALLBACK = (
    "For the global substorm picture, use "
    "fetch_supermag_data(data_type='indices') to get SME/SML/SMU indices, which "
    "serve the exact same scientific purpose. For the Dst/Kp outlook use "
    "get_iswa_forecast()."
)

_FORECAST_HINT = (
    "The ISWA HAPI service (https://iswa.ccmc.gsfc.nasa.gov/hapi) did not answer. "
    "It is a public, unauthenticated endpoint: check connectivity and retry. "
    "The same Kp outlook is also available as get_noaa_space_weather_context "
    "with query_type='forecast'."
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def is_error(value: Any) -> bool:
    """True when a payload is a failure dict. Same contract as the other loaders."""
    return isinstance(value, dict) and value.get("ok") is False


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error": code, "message": message}
    payload.update(extra)
    return payload


def parse_timestamp(value: Any) -> datetime:
    """Any ISO-8601 spelling a model may type -> tz-aware UTC ``datetime``.

    Accepts ``2024-09-12T12:00:00Z``, ``2024-09-12 12:00``, ``2024-09-12T12``
    and a bare ``2024-09-12``; a missing zone means UTC. Raises ``ValueError``
    for anything else, which the caller turns into ``invalid_input``.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1]
        try:
            dt = datetime.fromisoformat(text.replace(" ", "T"))
        except ValueError:
            raise ValueError(
                f"timestamp must be ISO 8601 (e.g. 2024-09-12T12:00:00Z), got {value!r}"
            ) from None
    else:
        raise ValueError(
            "timestamp is required (ISO 8601, e.g. 2024-09-12T12:00:00Z)"
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    """UTC ``datetime`` -> ``YYYY-MM-DDTHH:MM:SSZ``, the one spelling we quote."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_bbox(bbox: Any) -> tuple[float, float, float, float]:
    """``(minx, miny, maxx, maxy)`` from a tuple, a list or a comma string."""
    if bbox is None:
        return _DEFAULT_BBOX
    if isinstance(bbox, str):
        bbox = bbox.replace(";", ",").split(",")
    try:
        parts = [float(part) for part in bbox]
    except (TypeError, ValueError):
        raise ValueError(
            f"bbox must be 4 numbers (minx, miny, maxx, maxy), got {bbox!r}"
        ) from None
    if len(parts) != 4:
        raise ValueError(
            f"bbox must have 4 numbers (minx, miny, maxx, maxy), got {len(parts)}"
        )
    minx, miny, maxx, maxy = parts
    if not (-180 <= minx < maxx <= 180) or not (-90 <= miny < maxy <= 90):
        raise ValueError(
            "bbox must satisfy -180<=minx<maxx<=180 and -90<=miny<maxy<=90, "
            f"got {parts}"
        )
    return (minx, miny, maxx, maxy)


def _norm_size(value: Any, name: str) -> int:
    """A pixel count, coerced from the int-or-string forms a model may send."""
    try:
        size = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer pixel count, got {value!r}") from None
    if not 1 <= size <= 4096:
        raise ValueError(f"{name} must be between 1 and 4096 pixels, got {size}")
    return size


# --------------------------------------------------------------------------- #
# infrastructure
# --------------------------------------------------------------------------- #
_session: requests.Session | None = None


def _http() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": "geomagnetic-agent/1.0 (space-weather research tooling)",
                "Accept": "image/png, application/json, */*",
            }
        )
        _session = s
    return _session


def _request(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> requests.Response:
    """GET with exponential backoff on transient failures (429/5xx/timeouts)."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _http().get(url, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                backoff = 2 ** (attempt - 1)
                log.warning(
                    "HTTP %s from %s, retry %s/%s in %ss",
                    resp.status_code, url, attempt, retries, backoff,
                )
                time.sleep(backoff)
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                backoff = 2 ** (attempt - 1)
                log.warning(
                    "Request to %s failed (%s), retry %s/%s in %ss",
                    url, exc, attempt, retries, backoff,
                )
                time.sleep(backoff)
    raise ConnectionError(f"All {retries} attempts to {url} failed: {last_exc}")


def _unavailable(detail: str = "") -> dict[str, Any]:
    return _error(
        "iswa_unavailable",
        "ISWA WMS is unavailable.",
        hint=_UNAVAILABLE_HINT,
        detail=detail,
    )


def _forecast_unavailable(detail: str = "") -> dict[str, Any]:
    return _error(
        "iswa_forecast_unavailable",
        "ISWA HAPI API is unavailable.",
        hint=_FORECAST_HINT,
        detail=detail,
    )


def _no_data(timestamp: str, detail: str = "") -> dict[str, Any]:
    return _error(
        "no_data",
        f"ISWA has no ground perturbation map for {timestamp}.",
        hint=(
            "The WMS has no layer coverage for that TIME. Pick a timestamp the "
            "service publishes coverage for (near-real-time products only cover "
            "the last hours), or fall back to get_iswa_forecast()."
        ),
        timestamp=timestamp,
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# task 1 -- ground magnetic perturbation map (WMS)
# --------------------------------------------------------------------------- #
def build_wms_url(
    timestamp: str,
    bbox: tuple = _DEFAULT_BBOX,
    width: int = 1024,
    height: int = 512,
    *,
    layer: str = WMS_LAYER,
    endpoint: str | None = None,
) -> str:
    """The exact OGC WMS 1.1.1 ``GetMap`` request for one perturbation map.

    Parameters
    ----------
    timestamp : str
        ISO 8601 instant; sent as ``TIME``.
    bbox : tuple
        ``(minx, miny, maxx, maxy)`` in degrees, WMS order.
    width, height : int
        Output size in pixels.

    Returns
    -------
    str
        ``<endpoint>?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=...&
        TIME=...&BBOX=...&WIDTH=...&HEIGHT=...&FORMAT=image/png&SRS=EPSG:4326``.
        ``SRS`` is added to the brief's parameter list because WMS 1.1.1 makes
        it mandatory: without it a compliant server refuses the request.
        Commas, colons and slashes are left unescaped (all three are legal in a
        query string), so the URL stays readable and matches the documented
        form -- ``FORMAT=image/png``, ``TIME=...Z``, ``SRS=EPSG:4326``.
    """
    iso = iso_z(parse_timestamp(timestamp))
    minx, miny, maxx, maxy = _norm_bbox(bbox)
    params = {
        "SERVICE": "WMS",
        "VERSION": WMS_VERSION,
        "REQUEST": "GetMap",
        "LAYERS": str(layer or WMS_LAYER),
        "TIME": iso,
        "BBOX": f"{minx:g},{miny:g},{maxx:g},{maxy:g}",
        "WIDTH": str(_norm_size(width, "width")),
        "HEIGHT": str(_norm_size(height, "height")),
        "FORMAT": WMS_FORMAT,
        "SRS": WMS_SRS,
    }
    return f"{endpoint or WMS_ENDPOINT}?{urlencode(params, safe=',:/')}"


def save_map_png(
    data: bytes,
    timestamp: str,
    bbox: tuple | None = None,
    width: int = 1024,
    height: int = 512,
    *,
    layer: str = WMS_LAYER,
    endpoint: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write PNG *data* as ``plots/ground_perturbation_<timestamp>.png``.

    The one place that turns bytes into the documented payload, shared by the
    live path and by the offline path (which feeds it a synthetic image). The
    filename keeps every character Windows forbids out (``:`` -> ``-``), so the
    same timestamp produces the same path on any platform.

    ``metadata`` stays deliberately small: the whole payload must fit under
    ``FRAME_SUMMARY_MAX_CHARS`` (400) so ``_compact_tool_message`` never has to
    drop ``map_path`` from what the model is about to quote back to the user.

    Raises ``ValueError`` for a non-PNG body or malformed arguments; the live
    caller turns that into ``invalid_input``/``no_data`` instead.
    """
    if not isinstance(data, (bytes, bytearray)) or not bytes(data).startswith(_PNG_MAGIC):
        raise ValueError("map body is not a PNG")
    iso = iso_z(parse_timestamp(timestamp))
    bbox_t = _norm_bbox(bbox)
    w = _norm_size(width, "width")
    h = _norm_size(height, "height")
    name = f"ground_perturbation_{iso.replace(':', '-')}.png"
    target = Path(output_dir) if output_dir is not None else Path(DEFAULT_OUTPUT_DIR)
    target.mkdir(parents=True, exist_ok=True)
    path = target / name
    path.write_bytes(bytes(data))
    return {
        "ok": True,
        "map_path": path.as_posix(),
        "metadata": {
            "timestamp": iso,
            "bbox": list(bbox_t),
            "width": w,
            "height": h,
            "layer": layer or WMS_LAYER,
        },
    }


def _decommissioned(timestamp: str = "", detail: str = "") -> dict[str, Any]:
    """The honest answer for the retired WMS map layer (no request is sent).

    After the ISWA migration the layer does not exist on any domain, so the
    tool reports the decommission rather than dressing a dead endpoint up as a
    transient outage. ``fallback_hint`` is what the model is meant to act on.
    The payload is kept to exactly the four documented keys -- ``ok``,
    ``error``, ``message``, ``fallback_hint`` -- so its JSON fits under the
    400-character result cap and the hint is never truncated away from the
    model. *timestamp* and *detail* are folded into the message instead of
    becoming extra keys that would push it over the cap.
    """
    message = _DECOMMISSIONED_MESSAGE
    if timestamp:
        message = f"{message} Requested instant: {timestamp}."
    if detail:
        message = f"{message} ({detail})"
    return _error(
        "service_decommissioned",
        message,
        fallback_hint=_DECOMMISSIONED_FALLBACK,
    )


def get_ground_perturbation_map(
    timestamp: str,
    bbox: tuple = (-180, -90, 180, 90),
    width: int = 1024,
    height: int = 512,
    *,
    layer: str = WMS_LAYER,
    endpoint: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
    output_dir: str | Path | None = None,
    decommissioned: bool = True,
) -> dict[str, Any]:
    """Fetch one ISWA ground magnetic perturbation map and save it as a PNG.

    Returns
    -------
    dict
        ``{"ok": True, "map_path": "plots/ground_perturbation_<ts>.png",
        "metadata": {"timestamp", "bbox", "width", "height", "layer"}}`` on
        success, or one of the failure shapes:

        * ``{"ok": False, "error": "invalid_input", ...}`` -- the arguments
          could not be interpreted (no request was sent);
        * ``{"ok": False, "error": "service_decommissioned", ...}`` -- the
          public WMS map layer was retired by NASA CCMC during the ISWA
          migration. This is the answer the live tool gives today, together
          with ``fallback_hint`` naming SuperMAG indices and the Dst/Kp
          forecast. No network request is made for it;
        * ``{"ok": False, "error": "iswa_unavailable", ...}`` -- a request was
          sent but the endpoint was unreachable, refused, or answered with
          something that is not a picture;
        * ``{"ok": False, "error": "no_data", ...}`` -- the service answered
          properly but has no image for that TIME.

    The WMS client is kept intact (``decommissioned=False``): a deployment that
    runs against a still-live map service can fetch a real PNG, and the request
    builder stays the documented one. The default is the honest degraded path.
    """
    try:
        iso = iso_z(parse_timestamp(timestamp))
        bbox_t = _norm_bbox(bbox)
        w = _norm_size(width, "width")
        h = _norm_size(height, "height")
    except ValueError as exc:
        return _error(
            "invalid_input",
            str(exc),
            hint=(
                "Pass 'timestamp' as ISO 8601, e.g. '2024-09-12T12:00:00Z', "
                "bbox as [minx, miny, maxx, maxy], width/height as pixel counts."
            ),
        )

    if decommissioned:
        # No timestamp and no detail: the payload must stay under the 400-char
        # result cap so fallback_hint survives whole, and the handler's note
        # already names the requested instant.
        return _decommissioned()

    url = build_wms_url(iso, bbox_t, w, h, layer=layer, endpoint=endpoint)
    try:
        resp = _request(url, timeout=timeout, retries=retries)
    except (ConnectionError, requests.RequestException) as exc:
        log.warning("ISWA WMS unreachable at %s: %s", url, exc)
        return _unavailable(str(exc))

    if resp.status_code in (403, 404):
        # The WAF/dead CGI spelling of the same truth: the layer is gone.
        return _decommissioned(detail=f"{url} -> HTTP {resp.status_code}")
    if resp.status_code != 200:
        return _unavailable(f"{url} -> HTTP {resp.status_code}")

    body = resp.content or b""
    ctype = str(resp.headers.get("Content-Type") or "")
    if not body:
        return _no_data(iso, "empty response body")
    if body.startswith(_PNG_MAGIC):
        return save_map_png(
            body,
            iso,
            bbox_t,
            w,
            h,
            layer=layer,
            endpoint=endpoint,
            output_dir=output_dir,
        )

    head = body[:4096]
    if b"ServiceException" in head:
        detail = _wms_fault(head)
        return _no_data(iso, detail)
    if b"<html" in head.lower():
        return _unavailable(f"WMS returned HTML instead of a picture (content-type={ctype})")
    return _unavailable(f"WMS returned a non-picture body (content-type={ctype})")


def _wms_fault(body: bytes) -> str:
    """Pull the human line out of a WMS ``ServiceExceptionReport``."""
    text = body.decode("utf-8", "replace")
    for tag in ("ServiceExceptionText", "ServiceException"):
        opening, marker, closing = f"<{tag}", "</{tag}>", f"</{tag}>"
        start = text.find(opening)
        if start < 0:
            continue
        start = text.find(">", start)
        end = text.find(closing)
        if 0 <= start < end:
            line = text[start + 1 : end].strip().splitlines()
            if line and line[0].strip():
                return line[0].strip()[:300]
    return "WMS ServiceExceptionReport"


def synthetic_map_png(width: int = 16, height: int = 16) -> bytes:
    """A real (if schematic) PNG for offline runs -- no network, no PIL.

    A two-tone 16x16 grid: it is not the map, and ``metadata['offline']`` says
    so, but it is a well-formed PNG, so the file the model points the user at
    opens in the same viewer that opens a live one. ``width``/``height`` are
    accepted for signature symmetry with the live path and used verbatim.
    """
    w = _norm_size(width, "width")
    h = _norm_size(height, "height")
    rows = []
    for y in range(h):
        row = bytearray(b"\x00")  # per-scanline filter: none
        for x in range(w):
            shade = 40 if (x * 2 // w + y * 2 // h) % 2 else 150
            row += bytes((shade, shade + 30, 90))
        rows.append(bytes(row))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        _PNG_MAGIC
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


# --------------------------------------------------------------------------- #
# task 2 -- Dst/Kp outlook (HAPI)
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    """UTC now. A seam for the tests; callers get the wall clock."""
    return datetime.now(timezone.utc)


def _hapi_data(
    dataset: str,
    start: datetime,
    stop: datetime,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """One HAPI ``data`` call. Returns ``(rows, None)`` or ``(None, detail)``.

    Rows come back keyed by the dataset's own parameter names, whatever order
    the server lists them in -- the name is the contract, not the position.
    """
    url = f"{HAPI_BASE_URL}/data?" + urlencode(
        {
            "id": dataset,
            "time.min": iso_z(start),
            "time.max": iso_z(stop),
            "format": "json",
        },
        safe=":,",
    )
    try:
        resp = _request(url, timeout=timeout, retries=retries)
    except (ConnectionError, requests.RequestException) as exc:
        return None, str(exc)
    if resp.status_code != 200:
        return None, f"{url} -> HTTP {resp.status_code}"
    try:
        doc = resp.json()
    except ValueError:
        return None, f"{url} returned non-JSON"
    if not isinstance(doc, dict):
        return None, f"{url} returned {type(doc).__name__}, not a HAPI document"

    # HAPI reports application-level failures in a 200 body (code 1400 etc).
    status = doc.get("status")
    code = status.get("code") if isinstance(status, dict) else None
    if code is not None and code != 1200:
        message = status.get("message") if isinstance(status, dict) else ""
        return None, f"HAPI {code}: {message}"

    names = [
        p.get("name")
        for p in (doc.get("parameters") or [])
        if isinstance(p, dict)
    ]
    rows: list[dict[str, Any]] = []
    for raw in doc.get("data") or []:
        if isinstance(raw, dict):
            rows.append(raw)
        elif isinstance(raw, (list, tuple)) and names:
            rows.append(dict(zip(names, raw)))
    return rows, None


def _row_time(row: dict[str, Any]) -> str | None:
    """The row's own timestamp, normalised to ``...Z`` (None if unusable)."""
    raw = row.get("Time") or row.get("time") or row.get("timestamp")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return iso_z(parse_timestamp(raw))
    except ValueError:
        return None


def _num(value: Any) -> float | None:
    """JSON number -> float; ``null``/``""``/NaN -> None. Fill values are kept
    out by the caller, which knows each column's fill (-1 for ``KP_3H``)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("null", "nan", "none"):
            return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _kp_index(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, float], dict[str, str]]:
    """``{timestamp: k}`` plus ``{timestamp: kind}`` from the Kp dataset.

    Preference: observed -> predicted -> estimated -> reported. An observation
    is never replaced by a model value, and a future bucket (``Kp_observed`` is
    null there) falls through to ``Kp_predicted``.
    """
    values: dict[str, float] = {}
    kinds: dict[str, str] = {}
    for row in rows:
        stamp = _row_time(row)
        if stamp is None or stamp in values:
            continue
        candidates = (
            ("observed", row.get("Kp_observed")),
            ("predicted", row.get("Kp_predicted")),
            ("estimated", row.get("Kp_estimated")),
            ("reported", row.get("KP_3H")),
        )
        for kind, raw in candidates:
            value = _num(raw)
            # KP_3H uses -1 as its fill; Kp itself is never negative.
            if value is None or (kind == "reported" and value < 0):
                continue
            values[stamp] = value
            kinds[stamp] = kind
            break
    return values, kinds


def _dst_index(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    """``{timestamp: Dst}`` from the quicklook dataset (hourly observations)."""
    values: dict[str, float] = {}
    for row in rows:
        stamp = _row_time(row)
        if stamp is None or stamp in values:
            continue
        raw = row.get("Dst", row.get("dst"))
        value = _num(raw)
        if value is not None:
            values[stamp] = value
    return values


def _grid(start: datetime, stop: datetime, step_hours: int) -> list[datetime]:
    """Timestamps from the step-aligned floor of *start* to *stop*, inclusive."""
    step = timedelta(hours=step_hours)
    stamp = start.replace(minute=0, second=0, microsecond=0)
    stamp = stamp.replace(hour=(stamp.hour // step_hours) * step_hours)
    out: list[datetime] = []
    while stamp <= stop:
        out.append(stamp)
        stamp += step
    return out


def get_dst_kp_forecast() -> dict[str, Any]:
    """A merged Dst/Kp outlook from ISWA: predicted Kp, quicklook Dst.

    The window is ``FORECAST_HOURS_BACK`` behind now through
    ``FORECAST_HOURS_AHEAD`` ahead, on a ``FORECAST_STEP_HOURS`` grid.

    Returns
    -------
    dict
        ``{"ok": True, "forecast": [{"timestamp": "...", "dst": -50.0,
        "kp": 5.0, "kp_kind": "predicted"}, ...], "count": N, "source": ...,
        "window": {"start", "stop", "step_hours", "now"}, "sources": {...}}``.
        ``window.now`` is the instant the grid was built around, which is what
        separates the nowcast half of the series from the forecast half.
        ``dst`` is ``None`` for timestamps ahead of the
        last Kyoto quicklook observation -- ISWA publishes no Dst forecast
        (see the module docstring), so the loader reports what exists rather
        than projecting a value. ``kp_kind`` is one of 'observed',
        'predicted', 'estimated', 'reported', or ``None`` where the dataset has
        no value for that bucket.

        Failures: ``{"ok": False, "error": "iswa_forecast_unavailable", ...}``
        when HAPI cannot be reached or rejects the request, and
        ``{"ok": False, "error": "no_data", ...}`` when it answers with no
        rows at all.
    """
    now = _now()
    start = (now - timedelta(hours=FORECAST_HOURS_BACK)).replace(
        minute=0, second=0, microsecond=0
    )
    stop = (now + timedelta(hours=FORECAST_HOURS_AHEAD)).replace(
        minute=0, second=0, microsecond=0
    )

    kp_rows, kp_err = _hapi_data(KP_DATASET, start, stop)
    if kp_err is not None:
        return _forecast_unavailable(kp_err)

    dst_rows, dst_err = _hapi_data(DST_DATASET, start, stop)
    warnings: list[str] = []
    if dst_err is not None:
        # Dst is context, not the forecast: a Dst outage must not lose the Kp
        # outlook the tool exists to return. The loss is stated, not hidden.
        log.warning("ISWA Dst dataset unavailable: %s", dst_err)
        warnings.append(f"dst unavailable: {dst_err}")

    kp_by_time, kp_kind = _kp_index(kp_rows or [])
    dst_by_time = _dst_index(dst_rows or [])

    if not kp_by_time and not dst_by_time:
        return _error(
            "no_data",
            f"ISWA returned no rows for {iso_z(start)}..{iso_z(stop)}.",
            hint=(
                "The datasets cover the requested window on paper but returned "
                "nothing. Retry, or narrow the question to NOAA SWPC via "
                "get_noaa_space_weather_context(query_type='forecast')."
            ),
            window={"start": iso_z(start), "stop": iso_z(stop)},
        )

    forecast: list[dict[str, Any]] = []
    for stamp in _grid(start, stop, FORECAST_STEP_HOURS):
        iso = iso_z(stamp)
        kp = kp_by_time.get(iso)
        forecast.append(
            {
                "timestamp": iso,
                "dst": dst_by_time.get(iso),
                "kp": kp,
                "kp_kind": kp_kind.get(iso) if kp is not None else None,
            }
        )

    payload: dict[str, Any] = {
        "ok": True,
        "forecast": forecast,
        "count": len(forecast),
        "source": f"ISWA HAPI ({KP_DATASET} Kp + {DST_DATASET} Dst)",
        "window": {
            "start": iso_z(start),
            "stop": iso_z(stop),
            "step_hours": FORECAST_STEP_HOURS,
            "now": iso_z(now),
        },
        "sources": {
            "kp": (
                f"ISWA HAPI {KP_DATASET}: NOAA SWPC 3-hourly Kp "
                "(observed / estimated / predicted columns)"
            ),
            "dst": (
                f"ISWA HAPI {DST_DATASET}: WDC Kyoto quicklook Dst, hourly "
                "observations (no Dst forecast exists in ISWA)"
            ),
        },
    }
    if warnings:
        payload["warnings"] = warnings
    return payload
