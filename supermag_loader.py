"""SuperMAG Web Service loader: baseline-removed data and substorm indices.

What SuperMAG adds
------------------
The earlier loaders answer *what the field was* (INTERMAGNET), *what the solar
wind did* (OMNI) and *how strong the storm was* (Kyoto Dst/Kp). SuperMAG answers
a third question: what the **disturbance** was. Every component it serves has the
main field -- the daily and the yearly baseline -- already removed, so ``dX/dY/dZ``
are perturbations about zero, and its 1-minute SME/SML/SMU indices are the global
auroral electrojet strength used to time substorms.

Endpoints (verified against JHU/APL's own Python client, ``supermag-api.py``)
----------------------------------------------------------------------------
Three REST services live under ``https://supermag.jhuapl.edu/services/``:

* ``data-api.php``   -- station data, a JSON list of one object per minute;
* ``indices.php``    -- SME/SML/SMU, a JSON list of one object per minute;
* ``inventory.php``  -- which stations hold data in the window (plain text).

The request shape is the one the official client builds::

    ?python&nohead&start=YYYY-MM-DDTHH:MM&logon=NAME&extent=SECONDS[&station=IRT][&indices=sme,sml,smu]

JSON, not CSV: the CSV/HTML forms sit behind the interactive download page and
need a browser session, while this endpoint answers with one JSON object per
minute -- which is exactly what the official client parses and what pandas eats
without a second guess about the column names.

Credentials
-----------
The services are gated on a registered SuperMAG logon (registration is free).
Put yours in the ``SUPERMAG_LOGON`` environment variable; without it every call
fails with ``missing_logon`` instead of silently fetching nothing.

Contract
--------
* success -> ``pandas.DataFrame`` with a UTC ``timestamp`` column plus the
  requested components; data gaps stay in the frame as NaN rows;
* failure -> ``{"ok": False, "error": ..., "message": ..., "hint": ...}``.

Error codes
-----------
Data-shaped (the two the agent is expected to act on):
``station_not_found``, ``no_data``. Infrastructure: ``missing_logon``,
``invalid_logon``, ``supermag_unavailable``. Argument-shaped: ``invalid_date``,
``invalid_data_type``, ``range_too_long``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

__all__ = [
    "fetch_supermag_data",
    "fetch_supermag_inventory",
    "is_error",
    "get_logon",
    "DATA_URL",
    "INDICES_URL",
    "INVENTORY_URL",
    "LOGON_ENV",
    "GLOBAL_STATION",
]

log = logging.getLogger("supermag_loader")

BASE_URL = "https://supermag.jhuapl.edu/services/"
DATA_URL = BASE_URL + "data-api.php"
INDICES_URL = BASE_URL + "indices.php"
INVENTORY_URL = BASE_URL + "inventory.php"

#: The services authenticate a registered user by name; registration is free.
LOGON_ENV = "SUPERMAG_LOGON"

DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3

#: A 1-minute request is 1440 rows a day. "Please be considerate when using the
#: API" is the service's own first sentence, so the window is bounded and the
#: caller is told to split a longer study instead of pulling a year at once.
MAX_DAYS = 32

#: ``data_type`` values the loader understands.
DATA_TYPES = ("sites", "indices")

#: Default for ``station_code``. The substorm indices are a global product, so
#: no station stands behind them: the placeholder keeps the station argument
#: optional. It is deliberately not a real code -- for ``data_type='sites'`` it
#: is refused up front instead of being sent to the service.
GLOBAL_STATION = "GLOBAL"

_STATION_RE = re.compile(r"^[A-Za-z0-9]{2,6}$")

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


#: The hint the two data-shaped failures share: both mean "this window has
#: nothing for this station", and the caller's fix is the same.
_PERIOD_HINT = (
    "Проверьте код станции или попробуйте другой период. "
    "Для индексов используйте data_type='indices'."
)

_LOGON_HINT = (
    "Сервис SuperMAG требует зарегистрированный логин (регистрация бесплатная: "
    "https://supermag.jhuapl.edu). Задайте переменную окружения SUPERMAG_LOGON "
    "вашим именем пользователя и повторите запрос."
)


def get_logon(explicit: Any = None) -> str | None:
    """Resolve the SuperMAG logon from *explicit* or the environment."""
    for candidate in (explicit, os.environ.get(LOGON_ENV)):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


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
                "Accept": "application/json, text/plain, */*",
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


# --------------------------------------------------------------------------- #
# request building
# --------------------------------------------------------------------------- #
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


def _window(start: date, end: date) -> tuple[str, int]:
    """``(start, extent)`` for the URL: midnight of *start*, seconds through *end*.

    ``end`` is inclusive on the user-facing side (``end_date='2024-09-12'`` means
    the whole day), so the extent covers ``end + 1 day - start``.
    """
    if end < start:
        raise ValueError("end_date is before start_date")
    extent = int((end + timedelta(days=1) - start).total_seconds())
    return f"{start.isoformat()}T00:00", extent


def _core_url(page: str, logon: str, start: str, extent: int) -> str:
    """The URL shape the official ``supermag-api.py`` client builds."""
    return (
        f"{BASE_URL}{page}?python&nohead"
        f"&start={start}&logon={logon}&extent={extent:012d}"
    )


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #
def _classify_service_error(body: str) -> dict[str, Any]:
    """Turn the service's ``ERROR: ...`` line into a contract payload."""
    text = body.strip()
    lowered = text.lower()
    if "username" in lowered or "logon" in lowered:
        code = "missing_logon" if "no username" in lowered else "invalid_logon"
        return _error(code, f"SuperMAG rejected the request: {text}", hint=_LOGON_HINT)
    return _error(
        "supermag_unavailable",
        "SuperMAG Web Service returned an error.",
        hint=(
            "Повторите запрос позже; если ошибка повторяется, проверьте логин "
            "SUPERMAG_LOGON и формат дат (YYYY-MM-DD)."
        ),
        detail=text,
    )


def _read_body(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[str | None, dict[str, Any] | None]:
    """GET one service URL and return its body, or the failure payload.

    The service answers ``OK`` (a well-formed request with nothing in it), its
    payload, or a bare ``ERROR: ...`` line -- all with HTTP 200, so the body is
    classified here rather than by the status code. The payload is JSON for the
    data and indices services and plain text for the inventory, which is why the
    body travels unparsed.
    """
    try:
        resp = _request(url, timeout=timeout, retries=retries)
    except (ConnectionError, requests.RequestException) as exc:
        log.warning("SuperMAG unreachable at %s: %s", url, exc)
        return None, _error(
            "supermag_unavailable",
            "SuperMAG Web Service is unreachable.",
            hint=(
                "Проверьте доступ к supermag.jhuapl.edu и повторите; сервис "
                "обычно работает даже во время бурь."
            ),
            detail=str(exc),
        )
    if resp.status_code != 200:
        return None, _error(
            "supermag_unavailable",
            f"SuperMAG answered HTTP {resp.status_code}.",
            hint="Повторите запрос позже; если ошибка повторяется — проверьте SUPERMAG_LOGON.",
            detail=f"{url} -> HTTP {resp.status_code}",
        )

    body = (resp.text or "").strip()
    if not body or body == "OK":
        return "", None
    if body.startswith("ERROR"):
        return None, _classify_service_error(body)
    return body, None


def _read_rows(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """The JSON half of :func:`_read_body`: one list of row objects."""
    body, err = _read_body(url, timeout=timeout, retries=retries)
    if err is not None:
        return None, err
    if not body:
        return [], None
    try:
        rows = json.loads(body)
    except ValueError:
        return None, _error(
            "supermag_unavailable",
            "SuperMAG returned a body that is not JSON.",
            hint="Повторите запрос позже; формат ответа сервиса изменился.",
            detail=body[:200],
        )
    if rows is None:
        return [], None
    if not isinstance(rows, list):
        return None, _error(
            "supermag_unavailable",
            "SuperMAG returned an unexpected JSON shape (expected a list of rows).",
            detail=str(type(rows)),
        )
    return [r for r in rows if isinstance(r, dict)], None


def _timestamp(value: Any) -> Any:
    """SuperMAG's ``tval`` (Unix seconds) -> a UTC timestamp, else NaT."""
    try:
        stamp = pd.to_datetime(float(value), unit="s", utc=True)
    except (TypeError, ValueError):
        return pd.NaT
    return stamp


def _scalar(value: Any) -> float | None:
    """A JSON number (or NaN) -> float; anything else -> None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _component(row: dict[str, Any], key: str, subkey: str) -> float | None:
    """``N -> {'nez': -3.9, 'geo': -5.9}`` -> one number.

    ``nez`` is the component in the standard SuperMAG NEZ frame, ``geo`` the
    same vector mapped into geographic coordinates; both are baseline removed.
    A flat scalar (some service versions) is accepted as the ``nez`` value.
    """
    value = row.get(key)
    if isinstance(value, dict):
        return _scalar(value.get(subkey))
    return _scalar(value) if subkey == "nez" else None


# --------------------------------------------------------------------------- #
# frame builders
# --------------------------------------------------------------------------- #
def _sites_frame(rows: list[dict[str, Any]], station: str) -> pd.DataFrame:
    """Minute rows -> the dX/dY/dZ frame the analysis tools expect.

    ``dX/dY/dZ`` are the baseline-removed perturbations in the standard NEZ
    frame (SuperMAG's own product, equal to ``X/Y/Z`` in nT once the main field
    is gone); the ``*_geo`` columns carry the same vector mapped geographically.
    """
    frame = pd.DataFrame(
        {
            "timestamp": [_timestamp(r.get("tval")) for r in rows],
            "dX": [_component(r, "N", "nez") for r in rows],
            "dY": [_component(r, "E", "nez") for r in rows],
            "dZ": [_component(r, "Z", "nez") for r in rows],
            "X_geo": [_component(r, "N", "geo") for r in rows],
            "Y_geo": [_component(r, "E", "geo") for r in rows],
            "Z_geo": [_component(r, "Z", "geo") for r in rows],
        }
    )
    frame.attrs = {
        "source": "SuperMAG (baseline removed: daily + yearly)",
        "station": station,
        "data_type": "sites",
        "units": "nT",
    }
    return frame


#: The three global substorm indices the ``indices`` service is asked for.
INDEX_COLUMNS = ("SME", "SML", "SMU")


def _indices_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Minute rows -> the SME/SML/SMU frame.

    SME is the global electrojet strength, SML/SMU its lower/upper envelope;
    a substorm shows up as SML dropping sharply while SMU rises. The station
    fields the service rides along (``SMLmlat``, ``SMLstid``, ...) describe
    whichever magnetometer set the extremum at that minute, so they are kept
    out of the frame -- they are not time series.
    """
    data: dict[str, list[Any]] = {"timestamp": [_timestamp(r.get("tval")) for r in rows]}
    for name in INDEX_COLUMNS:
        data[name] = [_scalar(r.get(name)) for r in rows]
    frame = pd.DataFrame(data)
    frame.attrs = {
        "source": "SuperMAG indices (SME/SML/SMU, 1-minute, global)",
        "data_type": "indices",
        "units": "nT",
    }
    return frame


# --------------------------------------------------------------------------- #
# public fetchers
# --------------------------------------------------------------------------- #
def fetch_supermag_inventory(
    start_date: Any,
    end_date: Any,
    *,
    logon: Any = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> list[str] | dict[str, Any]:
    """Station codes holding data in ``[start_date, end_date]``, or an error.

    Only used on the failure path: an empty data response means either "wrong
    station code" or "quiet window", and the inventory is what tells them apart.
    """
    name = get_logon(logon)
    if not name:
        return _error(
            "missing_logon",
            "SuperMAG logon is not configured.",
            hint=_LOGON_HINT,
            env=LOGON_ENV,
        )
    try:
        start, extent = _window(_norm_day(start_date, "start_date"), _norm_day(end_date, "end_date"))
    except ValueError as exc:
        return _error("invalid_date", str(exc), hint="Формат дат: YYYY-MM-DD, end_date >= start_date.")

    url = _core_url("inventory.php", name, start, extent)
    body, err = _read_body(url, timeout=timeout, retries=retries)
    if err is not None:
        return err
    return _parse_inventory(body)


def _parse_inventory(body: str) -> list[str]:
    """Station codes out of the inventory answer.

    The documented answer is plain text: a count on the first line, then one
    station code per line. A JSON list is accepted as well, because the
    service's own sample URLs use ``fmt=json`` and the two shapes have been
    observed on the same path.
    """
    text = body.strip()
    if not text:
        return []
    items: list[Any]
    if text[0] in "[{\"":
        try:
            decoded = json.loads(text)
        except ValueError:
            decoded = None
        items = decoded if isinstance(decoded, list) else []
    else:
        items = text.splitlines()

    codes: list[str] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            token = str(item.get("station") or item.get("stid") or "").strip()
        else:
            token = str(item).strip()
        if not token:
            continue
        # First line is the station count, which is not a station.
        if index == 0 and token.isdigit():
            continue
        code = re.split(r"[\s,;]+", token)[0]
        if _STATION_RE.match(code) and code not in codes:
            codes.append(code.upper())
    return codes


def fetch_supermag_data(
    start_date: str,
    end_date: str | None = None,
    data_type: str = "sites",
    station_code: str = GLOBAL_STATION,
    *,
    logon: Any = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> pd.DataFrame | dict[str, Any]:
    """SuperMAG minute data, or a strict error payload.

    The dates come first and ``station_code`` is optional: it defaults to
    :data:`GLOBAL_STATION`, a placeholder rather than a code, so asking for
    the global indices needs no station at all.

    ``data_type='sites'``  -> station data with the baseline already removed
    (``dX/dY/dZ`` in nT, plus the geographically mapped ``*_geo`` columns);
    ``station_code`` must then be a real code (e.g. ``'IRT'``).
    ``data_type='indices'`` -> the global substorm indices ``SME/SML/SMU``;
    ``station_code`` is then ignored (the indices are a global product).

    The window is inclusive on both ends and capped at :data:`MAX_DAYS`.
    """
    kind = str(data_type or "sites").strip().lower()
    if kind not in DATA_TYPES:
        return _error(
            "invalid_data_type",
            f"data_type must be one of {list(DATA_TYPES)}, got {data_type!r}.",
            allowed=list(DATA_TYPES),
            hint="Используйте data_type='sites' для станции или 'indices' для SME/SML/SMU.",
        )

    end = end_date if end_date is not None else start_date
    try:
        start_day = _norm_day(start_date, "start_date")
        end_day = _norm_day(end, "end_date")
        start, extent = _window(start_day, end_day)
    except ValueError as exc:
        return _error(
            "invalid_date", str(exc), hint="Формат дат: YYYY-MM-DD, end_date >= start_date."
        )

    if (end_day - start_day).days + 1 > MAX_DAYS:
        return _error(
            "range_too_long",
            f"A SuperMAG request is capped at {MAX_DAYS} days ({(end_day - start_day).days + 1} requested).",
            max_days=MAX_DAYS,
            hint=f"Разбейте период на окна не длиннее {MAX_DAYS} дней и вызовите инструмент повторно.",
        )

    # The global product has no station behind it: drop the argument entirely
    # so neither the URL nor an error payload ever echoes the placeholder.
    station = "" if kind == "indices" else str(station_code or "").strip().upper()
    if kind == "sites":
        if not station or station == GLOBAL_STATION:
            return _error(
                "station_not_found",
                f"station_code is required for data_type='sites'; {GLOBAL_STATION!r} "
                "is only a placeholder, not a station.",
                station=station or None,
                hint="Укажите код станции, напр. 'IRT', либо data_type='indices' "
                     "для глобальных индексов SME/SML/SMU.",
            )
        if not _STATION_RE.match(station):
            return _error(
                "station_not_found",
                f"{station_code!r} is not a SuperMAG station code.",
                station=station_code,
                hint=_PERIOD_HINT,
            )

    name = get_logon(logon)
    if not name:
        return _error(
            "missing_logon",
            "SuperMAG logon is not configured.",
            hint=_LOGON_HINT,
            env=LOGON_ENV,
        )

    if kind == "indices":
        url = _core_url("indices.php", name, start, extent) + "&indices=sme,sml,smu"
    else:
        url = _core_url("data-api.php", name, start, extent) + f"&station={station}"

    rows, err = _read_rows(url, timeout=timeout, retries=retries)
    if err is not None:
        return err
    if not rows:
        return _no_data_or_unknown_station(kind, station, start_day, end_day, name,
                                           timeout=timeout, retries=retries)

    frame = _sites_frame(rows, station) if kind == "sites" else _indices_frame(rows)
    frame = _sort_by_time(frame)
    if frame.empty or frame["timestamp"].isna().all():
        return _no_data_or_unknown_station(kind, station, start_day, end_day, name,
                                           timeout=timeout, retries=retries)
    return frame


def _sort_by_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Order by timestamp; a NaT-safe no-op when the column is already sorted."""
    if "timestamp" in frame.columns and not frame.empty:
        try:
            frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        except TypeError:  # mixed types would only come from a broken payload
            pass
    return frame


def _no_data_or_unknown_station(
    kind: str,
    station: str,
    start_day: date,
    end_day: date,
    logon: str,
    *,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    """An empty response is ambiguous; the inventory is what resolves it.

    ``station_not_found`` -- the code is not among the stations that hold data
    for this window (a typo'd code, or a station outside the network's reach).
    ``no_data``           -- the station is there but the window is empty.
    """
    if kind != "sites":
        return _error(
            "no_data",
            f"SuperMAG returned no {kind} rows for {start_day}..{end_day}.",
            station=station or None,
            start_date=start_day.isoformat(),
            end_date=end_day.isoformat(),
            hint=_PERIOD_HINT,
        )

    inventory = fetch_supermag_inventory(
        start_day, end_day, logon=logon, timeout=timeout, retries=retries
    )
    if not is_error(inventory) and station not in inventory:
        return _error(
            "station_not_found",
            f"Station {station} holds no SuperMAG data for {start_day}..{end_day}.",
            station=station,
            start_date=start_day.isoformat(),
            end_date=end_day.isoformat(),
            hint=_PERIOD_HINT,
        )
    return _error(
        "no_data",
        f"SuperMAG returned no data for {station} on {start_day}..{end_day}.",
        station=station,
        start_date=start_day.isoformat(),
        end_date=end_day.isoformat(),
        hint=_PERIOD_HINT,
    )
