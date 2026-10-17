"""NOAA SWPC space-weather context loader (Phase 6.3).

The three earlier loaders answer *what happened* (INTERMAGNET magnetograms),
*why* (OMNI solar wind) and *how strong* (Kyoto Dst/Kp). This module adds the
**official forecast and warning layer** from NOAA's Space Weather Prediction
Center -- the operational voice a forecaster reads before a storm arrives.

What "space weather context" means here
---------------------------------------
* :func:`fetch_geomagnetic_alerts` -- current ALERT/WARNING/WATCH messages for
  geomagnetic activity (K-index thresholds, NOAA G1-G5 storm scale).
* :func:`fetch_solar_wind_forecast` -- the 3-day planetary Kp forecast, i.e. the
  predicted geomagnetic disturbance level (with the G-scale it maps to).
* :func:`fetch_solar_events` -- the GOES X-ray flare catalogue (C/M/X class),
  the solar *driver* half of the storm story.

Data sources (JSON only, no HTML scraping)
------------------------------------------
NOAA SWPC publishes machine-readable JSON at ``services.swpc.noaa.gov``. The
endpoints used are the live ones; several plausible-looking paths in the Phase
6.3 brief (``json/geoalert.json``, ``json/geomagnetic_forecast.json``) return
HTTP 404 and are intentionally *not* used:

* alerts   -- ``/products/alerts.json``
  A flat list of ``{product_id, issue_datetime, message}``. The geomagnetic
  ones carry ``K04A``/``K05W``/``K07A`` style ids and a free-text message that
  names the K-index and the ``NOAA Scale: G# - word``. The K-index number and
  the G-scale are parsed out of that text.
* forecast -- ``/products/noaa-planetary-k-index-forecast.json``
  One row per 3 hours: ``{time_tag, kp, observed, noaa_scale}`` where
  ``observed`` is ``observed``/``estimated``/``predicted`` and ``noaa_scale``
  is already ``G1``/``G3``/``null``.
* flares   -- ``/json/goes/primary/xray-flares-7-day.json``
  The GOES primary-satellite flare catalogue for the last 7 days, with
  ``begin_time``/``max_time``/``end_time`` and ``begin_class``/``max_class``.

Contract (for tool/LLM callers)
-------------------------------
* Success  -> ``{"ok": True, "data": [...]}`` (plus small summary metadata).
* No data  -> ``{"ok": True, "data": [], "message": "No events found"}``.
* Failure -> ``{"ok": False, "error": "noaa_api_unavailable", "hint": ...}``.

Branch on failure with :func:`is_error`; the ``hint`` explains how to recover.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timezone
from typing import Any

import requests

__all__ = [
    "fetch_geomagnetic_alerts",
    "fetch_solar_wind_forecast",
    "fetch_solar_events",
    "is_error",
    "ALERTS_URL",
    "KP_FORECAST_URL",
    "XRAY_FLARES_URL",
    "XRAY_FLARES_LATEST_URL",
    "SOLAR_PROBABILITIES_URL",
]

log = logging.getLogger("noaa_swpc_loader")

#: Live NOAA SWPC JSON products. Verified reachable 2026-10; the two paths named
#: in the Phase 6.3 brief (geoalert/geomagnetic_forecast) 404 and are dropped.
ALERTS_URL = "https://services.swpc.noaa.gov/products/alerts.json"
KP_FORECAST_URL = (
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
)
XRAY_FLARES_URL = (
    "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"
)
XRAY_FLARES_LATEST_URL = (
    "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"
)
SOLAR_PROBABILITIES_URL = "https://services.swpc.noaa.gov/json/solar_probabilities.json"

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3

#: NOAA planetary Kp -> G-scale. G1 starts at Kp=5; each unit adds a level.
_G_SCALE_WORDS: dict[str, str] = {
    "G1": "Minor",
    "G2": "Moderate",
    "G3": "Strong",
    "G4": "Severe",
    "G5": "Extreme",
}

_session: requests.Session | None = None


# --------------------------------------------------------------------------- #
# contract helpers
# --------------------------------------------------------------------------- #
def is_error(result: Any) -> bool:
    """True when a fetch failed. Lets a caller branch without isinstance checks."""
    return isinstance(result, dict) and result.get("ok") is False


def _error(
    code: str, message: str, *, hint: str | None = None, **extra: Any
) -> dict[str, Any]:
    """Build the stable, self-describing error payload returned to callers."""
    payload: dict[str, Any] = {"ok": False, "error": code, "message": message}
    payload.update(extra)
    if hint:
        payload["hint"] = hint
    return payload


def _ok(data: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """Success payload. An empty ``data`` list also carries the "no data" note."""
    payload: dict[str, Any] = {"ok": True, "data": data, "count": len(data)}
    payload.update(extra)
    if not data and "message" not in payload:
        payload["message"] = "No events found"
    return payload


# --------------------------------------------------------------------------- #
# infrastructure
# --------------------------------------------------------------------------- #
def _http() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": "geomagnetic-agent/1.0 (space-weather research tooling)",
                "Accept": "application/json, */*",
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


_UNAVAILABLE_HINT = (
    "NOAA SWPC (https://services.swpc.noaa.gov) is unreachable or rejected the "
    "request. Check connectivity and retry; the operational products are usually "
    "up even during storms."
)


def _unavailable(detail: str = "") -> dict[str, Any]:
    return _error(
        "noaa_api_unavailable",
        "NOAA SWPC JSON API is unavailable.",
        hint=_UNAVAILABLE_HINT,
        detail=detail,
    )


def _fetch_json(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[Any, dict[str, Any] | None]:
    """GET a JSON document, or return an ``noaa_api_unavailable`` marker."""
    try:
        resp = _request(url, timeout=timeout, retries=retries)
    except (ConnectionError, requests.RequestException) as exc:
        log.warning("NOAA SWPC unreachable at %s: %s", url, exc)
        return None, {"error": "noaa_api_unavailable", "detail": str(exc)}
    if resp.status_code != 200:
        return None, {
            "error": "noaa_api_unavailable",
            "detail": f"{url} -> HTTP {resp.status_code}",
        }
    try:
        return resp.json(), None
    except ValueError:
        return None, {
            "error": "noaa_api_unavailable",
            "detail": f"{url} returned non-JSON",
        }


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def _parse_swpc_time(text: Any) -> datetime | None:
    """Parse the two SWPC time spellings into a UTC-aware ``datetime``.

    Handles the message-line form ``2026 Oct 16 1939 UTC`` and the ISO-ish
    ``2026-10-16 19:39:45.267`` / ``2026-10-16T19:39:45Z`` forms.
    """
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    for fmt in (
        "%Y %b %d %H%M UTC",
        "%Y %b %d %H%M",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            parsed = datetime.strptime(raw[: len(fmt) + 6], fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _norm_day(value: Any, label: str) -> date:
    """Accept ``YYYY-MM-DD`` (or date/datetime) and return a ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text[: len(fmt) + 2], fmt).date()
            except ValueError:
                continue
    raise ValueError(f"{label} must be YYYY-MM-DD, got {value!r}")


def _kp_to_g_scale(kp: float | None) -> str | None:
    """Planetary Kp -> NOAA G-scale (G1 at Kp=5 ... G5 at Kp=9)."""
    if kp is None:
        return None
    if kp >= 9:
        return "G5"
    if kp >= 8:
        return "G4"
    if kp >= 7:
        return "G3"
    if kp >= 6:
        return "G2"
    if kp >= 5:
        return "G1"
    return None


def _g_scale_word(scale: str | None) -> str | None:
    return _G_SCALE_WORDS.get(scale) if scale else None


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: ``ALERT: Geomagnetic K-index of 5`` / ``WARNING: ...`` / ``WATCH: ...``.
_KIND_RE = re.compile(r"\b(ALERT|WARNING|WATCH)\b", re.IGNORECASE)
_K_INDEX_RE = re.compile(r"K-index of (\d+)", re.IGNORECASE)
_SCALE_RE = re.compile(r"NOAA Scale:\s*(G[1-5])\s*(?:-\s*([A-Za-z]+))?", re.IGNORECASE)
_ACTIVE_RE = re.compile(r"Active Warning:\s*(YES|NO)", re.IGNORECASE)

#: Field -> the message label that introduces its timestamp.
_TIME_FIELDS: tuple[tuple[str, str], ...] = (
    ("issue_time", "Issue Time:"),
    ("start_time", "Threshold Reached:"),
    ("start_time", "Valid From:"),
    ("end_time", "Valid To:"),
)


def _parse_alert(item: dict[str, Any]) -> dict[str, Any] | None:
    """Turn one ``alerts.json`` row into a structured geomagnetic alert.

    Returns ``None`` for products that are not geomagnetic K-index messages, so
    the caller only ever sees storm-relevant warnings.
    """
    product_id = str(item.get("product_id") or "").strip()
    message = str(item.get("message") or "")
    # Keep only geomagnetic K-index products; alerts.json also carries solar
    # radiation, aurora and other bulletins that are not storm warnings.
    if not (product_id.upper().startswith("K") or _K_INDEX_RE.search(message)):
        return None

    kind_match = _KIND_RE.search(message)
    kind = kind_match.group(1).upper() if kind_match else "MESSAGE"

    k_match = _K_INDEX_RE.search(message)
    k_index = float(k_match.group(1)) if k_match else None
    if k_index is None:
        # Product ids are K##A/K##W; fall back to the id when the text is terse.
        id_match = re.match(r"K(\d{2})", product_id, re.IGNORECASE)
        if id_match:
            k_index = float(id_match.group(1))

    scale_match = _SCALE_RE.search(message)
    if scale_match:
        g_scale = scale_match.group(1).upper()
        severity = scale_match.group(2)
    else:
        g_scale = _kp_to_g_scale(k_index)
        severity = _g_scale_word(g_scale)

    stamps: dict[str, str | None] = {}
    for field, label in _TIME_FIELDS:
        if field in stamps:
            continue
        idx = message.find(label)
        if idx < 0:
            continue
        parsed = _parse_swpc_time(message[idx + len(label) : idx + len(label) + 22])
        stamps[field] = _iso(parsed)

    issued = _parse_swpc_time(item.get("issue_datetime"))
    active_match = _ACTIVE_RE.search(message)

    return {
        "product_id": product_id,
        "kind": kind,
        "k_index": k_index,
        "g_scale": g_scale,
        "severity": severity or _g_scale_word(g_scale),
        "issue_time": stamps.get("issue_time") or _iso(issued),
        "start_time": stamps.get("start_time"),
        "end_time": stamps.get("end_time"),
        "active_warning": (
            active_match.group(1).upper() == "YES" if active_match else None
        ),
        "summary": f"{kind}: Geomagnetic K-index"
        + (f" of {int(k_index)}" if k_index is not None else "")
        + (f" ({g_scale} - {_g_scale_word(g_scale)})" if g_scale else ""),
    }


def fetch_geomagnetic_alerts(
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Fetch the current NOAA geomagnetic ALERT/WARNING/WATCH messages.

    Returns
    -------
    dict
        ``{"ok": True, "data": [alert, ...], "count": N, "source": ...}`` where
        each alert has ``product_id``, ``kind``, ``k_index``, ``g_scale``,
        ``severity``, ``issue_time``, ``start_time``, ``end_time``,
        ``active_warning`` and a one-line ``summary``. Newest first. When no
        geomagnetic product is present the payload is
        ``{"ok": True, "data": [], "message": "No alerts found"}``.
        On failure, ``{"ok": False, "error": "noaa_api_unavailable", ...}``.
    """
    raw, err = _fetch_json(ALERTS_URL, timeout=timeout, retries=retries)
    if err is not None:
        return _unavailable(err.get("detail", ""))

    parsed: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            alert = _parse_alert(item)
            if alert is not None:
                parsed.append(alert)

    # Newest first: an operator reads the latest warning at the top.
    parsed.sort(key=lambda a: a.get("issue_time") or "", reverse=True)
    return _ok(
        parsed,
        source="NOAA SWPC alerts.json",
        **({"message": "No alerts found"} if not parsed else {}),
    )


def fetch_solar_wind_forecast(
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Fetch NOAA's 3-day planetary Kp forecast (the geomagnetic forecast).

    The endpoint mixes the recent past (``observed``/``estimated``) with the
    future (``predicted``); both are returned, tagged by ``status``, so the
    caller can separate "what Kp has done" from "what it is forecast to do".

    Returns
    -------
    dict
        ``{"ok": True, "data": [row, ...], "count": N, "max_kp": ...,
        "max_scale": "G#"/None, "max_scale_time": ISO, "source": ...}`` where a
        row is ``{time_tag, kp, status, g_scale}``. ``max_kp``/``max_scale`` are
        taken over the *predicted* rows (falling back to all rows).
        On failure, ``{"ok": False, "error": "noaa_api_unavailable", ...}``.
    """
    raw, err = _fetch_json(KP_FORECAST_URL, timeout=timeout, retries=retries)
    if err is not None:
        return _unavailable(err.get("detail", ""))

    rows: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        kp = _num(item.get("kp"))
        if kp is None:
            continue
        scale = item.get("noaa_scale")
        g_scale = str(scale).strip().upper() if scale else _kp_to_g_scale(kp)
        stamps = _parse_swpc_time(item.get("time_tag"))
        rows.append(
            {
                "time_tag": _iso(stamps),
                "kp": kp,
                "status": str(item.get("observed") or "unknown").strip().lower(),
                "g_scale": g_scale,
            }
        )

    forecast_rows = [r for r in rows if r["status"] == "predicted"] or rows
    peak = max(forecast_rows, key=lambda r: r["kp"]) if forecast_rows else None

    return _ok(
        rows,
        source="NOAA SWPC noaa-planetary-k-index-forecast.json",
        max_kp=peak["kp"] if peak else None,
        max_scale=peak["g_scale"] if peak else None,
        max_scale_time=peak["time_tag"] if peak else None,
        **({"message": "No forecast found"} if not rows else {}),
    )


def fetch_solar_events(
    start_date: Any = None,
    end_date: Any = None,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Fetch the GOES X-ray flare catalogue, optionally filtered by date.

    Parameters
    ----------
    start_date, end_date:
        ``YYYY-MM-DD`` (or date/datetime), inclusive. Either may be ``None``:
        with neither, the whole 7-day catalogue is returned. The upstream
        product only covers the last 7 days, so a range outside that window
        yields no rows (not an error).

    Returns
    -------
    dict
        ``{"ok": True, "data": [flare, ...]}`` where each flare has
        ``begin_time``, ``max_time``, ``end_time``, ``begin_class``,
        ``max_class``, ``class_letter`` and ``satellite`` -- sorted newest
        first. Empty -> ``{"ok": True, "data": [], "message": "No events found"}``.
        Bad dates -> ``{"ok": False, "error": "invalid_input", ...}``.
        Unreachable API -> ``{"ok": False, "error": "noaa_api_unavailable", ...}``.
    """
    try:
        start = _norm_day(start_date, "start_date") if start_date else None
        end = _norm_day(end_date, "end_date") if end_date else (start if start else None)
    except (TypeError, ValueError) as exc:
        return _error(
            "invalid_input",
            str(exc),
            hint="Pass dates as 'YYYY-MM-DD', e.g. "
            "fetch_solar_events('2026-10-10', '2026-10-17').",
        )
    if start and end and end < start:
        return _error(
            "invalid_input",
            f"end_date {end} precedes start_date {start}.",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

    raw, err = _fetch_json(XRAY_FLARES_URL, timeout=timeout, retries=retries)
    if err is not None:
        return _unavailable(err.get("detail", ""))

    flares: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        begin = _parse_swpc_time(item.get("begin_time"))
        if start is not None and (
            begin is None or not (start <= begin.date() <= end)
        ):
            continue
        max_class = str(item.get("max_class") or "").strip()
        flares.append(
            {
                "begin_time": _iso(_parse_swpc_time(item.get("begin_time"))),
                "max_time": _iso(_parse_swpc_time(item.get("max_time"))),
                "end_time": _iso(_parse_swpc_time(item.get("end_time"))),
                "begin_class": str(item.get("begin_class") or "").strip() or None,
                "max_class": max_class or None,
                "class_letter": max_class[:1].upper() if max_class else None,
                "satellite": item.get("satellite"),
            }
        )

    # Newest first; the flare an analyst wants is usually the most recent one.
    flares.sort(key=lambda f: f.get("max_time") or "", reverse=True)
    return _ok(
        flares,
        source="NOAA SWPC GOES X-ray flare catalogue (7-day)",
        **({"message": "No events found"} if not flares else {}),
    )


# --------------------------------------------------------------------------- #
# manual smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    for name, result in (
        ("alerts", fetch_geomagnetic_alerts()),
        ("forecast", fetch_solar_wind_forecast()),
        ("flares", fetch_solar_events()),
    ):
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        if is_error(result):
            print("ERROR:", result)
            continue
        print(f"count={result['count']} source={result.get('source')}")
        for row in result["data"][:5]:
            print(" ", row)
