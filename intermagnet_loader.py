"""Loader for INTERMAGNET ground-based geomagnetic observatory data.

Data sources (both speak the same IAGA-2002 fixed-width layout, which is what
makes the multi-source design cheap):

* Edinburgh GIN web service -- https://imag-data.bgs.ac.uk/GIN_V1/GINServices
  Primary source. Hosts 154 observatories, publishes up to yesterday, and is the
  only one of the two that exposes the ``GetDataDirectory`` availability service
  and the ``GetCapabilities`` station metadata (lat/lon/elevation/embargo).
* WDC for Geomagnetism, Kyoto (HAPI 3.3.1) -- https://wdc.kugi.kyoto-u.ac.jp/hapi
  Secondary source. Hosts 326 observatories and carries the historical archive
  (roughly 1998-2021/2022) that the Edinburgh GIN no longer serves, which is how
  definitive data is recovered for older intervals.

Public API
----------
``get_available_stations()``     -> {IAGA code: (name, lat, lon)}
``fetch_observatory_data()``     -> pandas.DataFrame | error dict
``save_to_cache()``              -> pathlib.Path
``load_from_cache()``            -> pandas.DataFrame | None
``is_error()``                   -> bool

Conventions
-----------
* All timestamps are naive UTC. The unit follows the pandas version: ``datetime64[ns]``
  on pandas 1.x/2.x, ``datetime64[us]`` on pandas 3.x -- read ``df.dtypes["timestamp"]``
  rather than assuming. INTERMAGNET files are UTC by
  convention and this keeps the CSV round-trip lossless.
* Component order is requested as ``orientation=XYZF``, giving
  ``X``=North, ``Y``=East, ``Z``=Down (INTERMAGNET sign convention: Z positive
  downwards), ``F``=total field, all in nT.
* ``end_date`` is INCLUSIVE -- asking for 2025-03-10..2025-03-11 returns two
  full days.
* Missing samples are normalised to ``NaN``. IAGA-2002 encodes them as the
  ``99999.00`` sentinel in fixed-width fields; covJson encodes them as ``null``;
  blank fields occur in IAGA files that carry the missing-data indicator instead.
* ``fetch_observatory_data()`` returns a ``DataFrame`` on success and a plain
  ``dict`` on failure so that an LLM agent can branch on the payload. Every error
  dict has a stable shape -- see ``_error()``.

Terms of use for INTERMAGNET data: https://intermagnet.org/data_conditions.html
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

__all__ = [
    "get_available_stations",
    "get_station_details",
    "fetch_observatory_data",
    "save_to_cache",
    "load_from_cache",
    "clear_cache",
    "is_error",
    "parse_iaga2002",
    "COMPONENTS",
]

log = logging.getLogger("intermagnet_loader")

GIN_URL = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
KYOTO_HAPI_URL = "https://wdc.kugi.kyoto-u.ac.jp/hapi"

CACHE_DIR = Path(__file__).resolve().parent / "cache"
STATION_CACHE_TTL_S = 24 * 60 * 60

DEFAULT_TIMEOUT = 90
DEFAULT_MAX_RETRIES = 3
CHUNK_DAYS = 31

COMPONENTS: tuple[str, ...] = ("X", "Y", "Z", "F")

GIN_FALLBACK_CHAIN: tuple[str, ...] = (
    "definitive",
    "quasi-def",
    "adjusted",
    "reported",
    "best-avail",
)

KYOTO_FALLBACK_CHAIN: tuple[str, ...] = ("definitive", "best-avail")

_DATA_TYPE_ALIASES = {
    "definitive": "definitive",
    "final": "definitive",
    "def": "definitive",
    "quasi-definitive": "quasi-def",
    "quasi-def": "quasi-def",
    "quasidef": "quasi-def",
    "quasi_definitive": "quasi-def",
    "provisional": "adjusted",
    "adjusted": "adjusted",
    "adj": "adjusted",
    "variation": "reported",
    "variometer": "reported",
    "reported": "reported",
    "rep": "reported",
    "best-avail": "best-avail",
    "best_avail": "best-avail",
    "best": "best-avail",
    "best-available": "best-avail",
    "auto": "auto",
    "any": "auto",
}

_TS_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)
_DATA_LINE_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")

_session: requests.Session | None = None


# --------------------------------------------------------------------------- #
# infrastructure
# --------------------------------------------------------------------------- #
def _http() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": "geomagnetic-agent/1.0 (INTERMAGNET research tooling)",
                "Accept": "*/*",
            }
        )
        _session = s
    return _session


def _request(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _http().request(method, url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                backoff = 2 ** (attempt - 1)
                log.warning(
                    "HTTP %s from %s, retry %s/%s in %ss",
                    resp.status_code,
                    url,
                    attempt,
                    retries,
                    backoff,
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
                    url,
                    exc,
                    attempt,
                    retries,
                    backoff,
                )
                time.sleep(backoff)
    raise ConnectionError(f"All {retries} attempts to {url} failed: {last_exc}")


def _error(
    code: str,
    message: str,
    *,
    station: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    requested_type: str | None = None,
    tried: Sequence[dict[str, Any]] | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    """Build the stable, self-describing error payload returned to LLM callers."""
    payload: dict[str, Any] = {
        "ok": False,
        "error": code,
        "message": message,
        "station": station,
        "start_date": start_date,
        "end_date": end_date,
        "requested_data_type": requested_type,
        "attempts": list(tried or []),
    }
    if hint:
        payload["hint"] = hint
    return payload


# --------------------------------------------------------------------------- #
# input normalisation
# --------------------------------------------------------------------------- #
def _norm_date(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(text[: len(fmt) + 4], fmt).date()
            except ValueError:
                continue
    raise ValueError(
        f"{label} must be a YYYY-MM-DD string or a date/datetime, got {value!r}"
    )


def _norm_station(code: Any) -> str:
    if not isinstance(code, str):
        raise ValueError(f"station_code must be a string, got {code!r}")
    norm = code.strip().upper()
    if not norm:
        raise ValueError("station_code must not be empty")
    return norm


def _norm_type(value: Any) -> str:
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise ValueError(f"data_type must be a string, got {value!r}")
    key = value.strip().lower().replace(" ", "-")
    if key not in _DATA_TYPE_ALIASES:
        valid = sorted({"auto", "definitive", "quasi-def", "adjusted", "reported", "best-avail"})
        raise ValueError(f"Unknown data_type {value!r}. Valid values: {valid}")
    return _DATA_TYPE_ALIASES[key]


def _chunk_dates(start: date, end: date, days: int) -> list[tuple[date, date]]:
    span = (end - start).days + 1
    if span <= days:
        return [(start, end)]
    out: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        out.append((cur, stop))
        cur = stop + timedelta(days=1)
    return out


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _to_float(field: str) -> float:
    """Parse one IAGA-2002 fixed-width field, mapping every missing-data
    encoding to NaN (blank field, ``99999`` sentinel, ``-99999``)."""
    text = field.strip()
    if not text:
        return float("nan")
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    if value != value or abs(value) >= 99999.0:
        return float("nan")
    return value


def parse_iaga2002(text: str) -> pd.DataFrame:
    """Parse an INTERMAGNET/WDC IAGA-2002 minute data file into a DataFrame.

    The data block served by both the Edinburgh GIN and Kyoto WDC is a
    70-character fixed-width record::

        2025-03-10 00:00:00.000 069     11135.50   2109.00  52358.70  53571.27
        |----- timestamp ----| |doy| |---- X ----| |--- Y ---| |--- Z ---| |--- F ---|

    The four component fields are right-aligned in 10-character cells, so they
    are read by tail offset (the last 40 characters of the line) rather than by
    absolute column. That keeps the parser correct for negative components and
    for the timestamp-width variations found between providers. The DOY column is
    redundant with the date and is ignored.
    """
    stamps: list[pd.Timestamp] = []
    rows: list[list[float]] = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if not _DATA_LINE_RE.match(line):
            continue
        m = _TS_RE.match(line)
        if not m:
            continue
        time_part = m.group("time")
        if "." in time_part:
            iso = f"{m.group('date')}T{time_part}"
        else:
            iso = f"{m.group('date')}T{time_part}+00:00"
        try:
            stamps.append(pd.Timestamp(iso).tz_localize(None))
        except ValueError:
            continue
        tail = line[-40:]
        rows.append(
            [
                _to_float(tail[0:10]),
                _to_float(tail[10:20]),
                _to_float(tail[20:30]),
                _to_float(tail[30:40]),
            ]
        )

    if not stamps:
        return pd.DataFrame(columns=["timestamp", *COMPONENTS])

    df = pd.DataFrame(rows, columns=list(COMPONENTS))
    df.insert(0, "timestamp", pd.DatetimeIndex(stamps, name="timestamp"))
    return df.sort_values("timestamp").reset_index(drop=True)


def _parse_covjson(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse the INTERMAGNET common-data-format (covJson) payload.

    Structure is column-oriented: ``{"datetime": [...], "X": [...], ...,
    "@info": {...}}`` with ``null`` for missing samples.
    """
    stamps = payload.get("datetime") or []
    data = {"timestamp": pd.to_datetime(list(stamps), utc=True, errors="coerce")}
    for comp in COMPONENTS:
        data[comp] = payload.get(comp)
    df = pd.DataFrame(data)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    for comp in COMPONENTS:
        df[comp] = pd.to_numeric(df[comp], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _has_data(df: pd.DataFrame) -> bool:
    if df.empty or "timestamp" not in df.columns:
        return False
    values = [df[c] for c in COMPONENTS if c in df.columns]
    if not values:
        return False
    return bool(values[0].notna().any())


# --------------------------------------------------------------------------- #
# station catalogue
# --------------------------------------------------------------------------- #
def _fetch_gin_capabilities() -> dict[str, dict[str, Any]]:
    resp = _request("GET", GIN_URL, params={"Request": "GetCapabilities", "Format": "json"})
    if resp.status_code != 200:
        raise RuntimeError(f"GetCapabilities failed: HTTP {resp.status_code}")
    payload = resp.json()
    out: dict[str, dict[str, Any]] = {}
    for obs in payload.get("ObservatoryList", []):
        code = (obs.get("IagaCode") or "").strip().upper()
        if not code:
            continue
        out[code] = {
            "name": (obs.get("Name") or "").strip(),
            "latitude": obs.get("Latitude"),
            "longitude": obs.get("Longitude"),
            "elevation": obs.get("Elevation"),
            "data_embargo_hours": obs.get("DataEmbargoHours"),
            "sources": ["edinburgh_gin"],
        }
    return out


def _fetch_kyoto_catalog() -> dict[str, dict[str, Any]]:
    resp = _request("GET", f"{KYOTO_HAPI_URL}/catalog", timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Kyoto catalog failed: HTTP {resp.status_code}")
    payload = resp.json()
    out: dict[str, dict[str, Any]] = {}
    for entry in payload.get("catalog", []):
        if entry.get("x_category") != "stn":
            continue
        code = (entry.get("x_station") or "").strip().upper()
        if not code:
            continue
        title = (entry.get("title") or "").strip()
        name = re.sub(r"\s*\((?:Best Available|Definitive).*$", "", title).strip()
        if code in out:
            continue
        out[code] = {
            "name": name,
            "latitude": None,
            "longitude": None,
            "elevation": None,
            "data_embargo_hours": None,
            "sources": ["kyoto_hapi"],
        }
    return out


def _station_cache_path() -> Path:
    return CACHE_DIR / "_stations.json"


def _load_station_cache() -> dict[str, Any] | None:
    path = _station_cache_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if time.time() - payload.get("fetched_at", 0) > STATION_CACHE_TTL_S:
        return None
    return payload.get("stations")


def _load_station_registry(refresh: bool) -> dict[str, dict[str, Any]]:
    if not refresh:
        cached = _load_station_cache()
        if cached:
            return cached

    registry: dict[str, dict[str, Any]] = {}
    problems: list[str] = []

    try:
        registry.update(_fetch_gin_capabilities())
    except Exception as exc:
        problems.append(f"edinburgh_gin: {exc}")

    try:
        for code, meta in _fetch_kyoto_catalog().items():
            if code in registry:
                if "kyoto_hapi" not in registry[code]["sources"]:
                    registry[code]["sources"].append("kyoto_hapi")
            else:
                registry[code] = meta
    except Exception as exc:
        problems.append(f"kyoto_hapi: {exc}")

    if not registry:
        raise RuntimeError(
            "Could not load station catalogue from any source: " + "; ".join(problems)
        )
    for problem in problems:
        log.warning("Station catalogue partially degraded -- %s", problem)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _station_cache_path().write_text(
            json.dumps({"fetched_at": time.time(), "stations": registry}, indent=1),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("Could not persist station cache: %s", exc)

    return registry


def get_station_details(refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Return the merged station registry with full metadata per IAGA code."""
    return _load_station_registry(refresh=refresh)


def get_available_stations(refresh: bool = False) -> dict[str, tuple[str, float | None, float | None]]:
    """Return ``{IAGA_CODE: (name, latitude, longitude)}`` for the whole network.

    Merges the Edinburgh GIN (154 observatories, with coordinates) with the Kyoto
    WDC HAPI catalogue (326 observatories, no coordinates) so the returned
    dictionary covers the broadest network available from public endpoints.
    Stations known only to Kyoto have ``None`` coordinates.

    Use :func:`get_station_details` when you need elevation, embargo hours or to
    know which backends serve a station.
    """
    registry = _load_station_registry(refresh=refresh)
    return {
        code: (meta.get("name", ""), meta.get("latitude"), meta.get("longitude"))
        for code, meta in sorted(registry.items())
    }


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
_gin_station_memo: dict[str, bool] = {}


def _gin_known_station(code: str) -> bool:
    """Whether the Edinburgh GIN mirrors this observatory. Memoised per process."""
    if code not in _gin_station_memo:
        try:
            _gin_station_memo[code] = code in _fetch_gin_capabilities()
        except Exception:
            _gin_station_memo[code] = False
    return _gin_station_memo[code]


def _fetch_gin(
    code: str, start: date, end: date, state: str, samples_per_day: str
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Fetch one chunk from the Edinburgh GIN. Returns (df, info).

    covJson is tried first because it is the structured INTERMAGNET common data
    format and needs no fixed-width parsing; IAGA-2002 is the fallback. Both are
    validated for real samples, because the GIN answers HTTP 200 with a
    full-length but entirely null payload when a publication state is unavailable.
    """
    duration = (end - start).days + 1
    params = {
        "Request": "GetData",
        "observatoryIagaCode": code,
        "samplesPerDay": samples_per_day,
        "dataStartDate": start.isoformat(),
        "dataDuration": duration,
        "publicationState": state,
        "orientation": "XYZF",
        "recordTermination": "UNIX",
    }
    info: dict[str, Any] = {"backend": "edinburgh_gin", "state": state}

    resp = _request("GET", GIN_URL, params={**params, "Format": "json"})
    if resp.status_code != 200:
        info.update(ok=False, format="covJson", note=f"HTTP {resp.status_code}")
        return None, info
    try:
        payload = resp.json()
    except ValueError:
        info.update(ok=False, format="covJson", note="response was not valid JSON")
        return None, info
    df = _parse_covjson(payload)
    if _has_data(df):
        info.update(ok=True, format="covJson", note="ok")
        return df, info
    json_note = "HTTP 200 but every sample null/missing"

    resp = _request("GET", GIN_URL, params={**params, "Format": "iaga2002"})
    if resp.status_code != 200:
        info.update(ok=False, format="iaga2002", note=f"{json_note}; iaga2002 HTTP {resp.status_code}")
        return None, info
    df_iaga = parse_iaga2002(resp.text)
    if _has_data(df_iaga):
        info.update(ok=True, format="iaga2002", note=f"covJson {json_note}, iaga2002 ok")
        return df_iaga, info
    info.update(ok=False, format="iaga2002", note=f"{json_note}; iaga2002 also empty")
    return None, info


def _fetch_kyoto(
    code: str, start: date, end: date, state: str, samples_per_day: str
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Fetch one chunk from the Kyoto WDC HAPI service. Returns (df, info).

    Kyoto publishes IAGA-2002 only, with two variants: ``min_<code>`` for best
    available and ``min_<code>_definitive``. Its INTERMAGNET mirror ends around
    2022, so it is used for historical/definitive retrieval rather than recency.
    """
    prefix = {"minute": "min", "second": "sec", "hour": "hour"}.get(samples_per_day, "min")
    dataset = (
        f"{prefix}_{code.lower()}"
        if state == "best-avail"
        else f"{prefix}_{code.lower()}_definitive"
    )
    info: dict[str, Any] = {"backend": "kyoto_hapi", "state": state, "format": "iaga2002",
                            "dataset": dataset}

    resp = _request(
        "GET",
        f"{KYOTO_HAPI_URL}/data",
        params={
            "id": dataset,
            "start": f"{start.isoformat()}T00:00:00Z",
            "stop": f"{end.isoformat()}T23:59:59Z",
            "format": "iaga2002",
        },
    )
    if resp.status_code != 200:
        detail = ""
        try:
            detail = resp.json().get("status", {}).get("message", "")
        except ValueError:
            detail = resp.text[:200]
        info.update(ok=False, note=f"HTTP {resp.status_code}: {detail}")
        return None, info
    df = parse_iaga2002(resp.text)
    if _has_data(df):
        info.update(ok=True, note="ok")
        return df, info
    info.update(ok=False, note="no samples in range")
    return None, info


def _fetch_chunk(
    code: str,
    start: date,
    end: date,
    states: Sequence[str],
    samples_per_day: str,
) -> tuple[pd.DataFrame | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Try every state x backend combination for one chunk, best first.

    Returns ``(df, resolved_infos, attempts)``. Backends are attempted per state:
    if Edinburgh has nothing for a publication state -- because the state is
    unavailable for the period, or the station is not mirrored there at all --
    Kyoto is tried before falling through to the next, less mature, state.
    """
    attempts: list[dict[str, Any]] = []
    for state in states:
        for label, fetcher in (("edinburgh_gin", _fetch_gin), ("kyoto_hapi", _fetch_kyoto)):
            if label == "kyoto_hapi" and state not in KYOTO_FALLBACK_CHAIN:
                continue
            if label == "edinburgh_gin" and not _gin_known_station(code):
                attempts.append(
                    {"backend": label, "state": state, "ok": False,
                     "note": "station not mirrored by this provider"}
                )
                continue
            df, info = fetcher(code, start, end, state, samples_per_day)
            attempts.append(info)
            if df is not None:
                return df, [info], attempts
    return None, [], attempts


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def _cache_key(station: str, start: date, end: date, data_type: str, samples_per_day: str) -> Path:
    stem = (
        f"{station}_{start.isoformat()}_{end.isoformat()}"
        f"_{data_type}_{samples_per_day}"
    )
    return CACHE_DIR / f"{stem}.parquet"


def _use_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def save_to_cache(
    df: pd.DataFrame,
    station: str,
    dates: str | date | tuple[date, date] | Sequence[date],
    data_type: str = "auto",
    samples_per_day: str = "Minute",
    meta: dict[str, Any] | None = None,
) -> Path:
    """Persist a fetched DataFrame so repeat agent requests never re-hit the API.

    ``dates`` accepts ``"2025-03-10_2025-03-11"``, a ``(start, end)`` pair, or any
    two-element sequence of dates. Parquet is used when ``pyarrow`` is installed
    and CSV otherwise; the written path is returned either way. Sidecar metadata
    records the publication state and backend that actually served the data.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("save_to_cache() needs a non-empty DataFrame")

    start, end = _coerce_dates(dates)
    path = _cache_key(_norm_station(station), start, end, data_type, samples_per_day)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if _use_parquet():
        df.to_parquet(path, index=False)
    else:
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)

    try:
        payload = {
            "station": station,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "data_type": data_type,
            "samples_per_day": samples_per_day,
            "rows": int(len(df)),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if meta:
            payload.update(meta)
        path.with_suffix(".meta.json").write_text(
            json.dumps(payload, indent=1, default=str), encoding="utf-8"
        )
    except OSError as exc:
        log.warning("Could not write cache metadata: %s", exc)
    return path


def _coerce_dates(dates: Any) -> tuple[date, date]:
    if isinstance(dates, str):
        for sep in ("_", "..", " to ", "/"):
            if sep in dates:
                left, right = dates.split(sep, 1)
                return _norm_date(left, "start_date"), _norm_date(right, "end_date")
        raise ValueError(f"Could not split dates string {dates!r}")
    if isinstance(dates, (tuple, list)) and len(dates) == 2:
        return _norm_date(dates[0], "start_date"), _norm_date(dates[1], "end_date")
    if isinstance(dates, date) and not isinstance(dates, datetime):
        return dates, dates
    raise ValueError(f"Unsupported dates value {dates!r}")


def load_from_cache(
    station: str,
    start_date: Any,
    end_date: Any,
    data_type: str = "auto",
    samples_per_day: str = "Minute",
) -> pd.DataFrame | None:
    """Return a cached DataFrame for the exact request, or ``None`` on a miss."""
    start, end = _norm_date(start_date, "start_date"), _norm_date(end_date, "end_date")
    path = _cache_key(_norm_station(station), start, end, data_type, samples_per_day)
    if not path.is_file():
        return None
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=["timestamp"])
    except Exception as exc:
        log.warning("Unreadable cache entry %s (%s), ignoring", path, exc)
        return None
    meta_path = path.with_suffix(".meta.json")
    if meta_path.is_file():
        try:
            df.attrs["cache_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return df


def clear_cache() -> int:
    """Delete every cached artefact. Returns the number of files removed."""
    if not CACHE_DIR.is_dir():
        return 0
    removed = 0
    for item in CACHE_DIR.iterdir():
        if item.is_file():
            item.unlink()
            removed += 1
    return removed


# --------------------------------------------------------------------------- #
# public fetch
# --------------------------------------------------------------------------- #
def is_error(result: Any) -> bool:
    """True when a fetch failed. Lets an agent branch without isinstance checks."""
    return isinstance(result, dict) and result.get("ok") is False


def fetch_observatory_data(
    station_code: str,
    start_date: Any,
    end_date: Any,
    data_type: str = "definitive",
    samples_per_day: str = "Minute",
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame | dict[str, Any]:
    """Download and parse geomagnetic data for one observatory.

    Parameters
    ----------
    station_code:
        Three/four letter IAGA code, e.g. ``"IRT"``. Case-insensitive.
    start_date, end_date:
        ``YYYY-MM-DD`` (or ``date``/``datetime``). ``end_date`` is inclusive.
    data_type:
        ``"definitive"``, ``"quasi-definitive"``, ``"provisional"``/``"adjusted"``,
        ``"variation"``/``"reported"``, ``"best-avail"``, or ``"auto"``. Friendly
        aliases are accepted. With ``"auto"`` (or when a specific type yields
        nothing) the loader walks the publication-state chain
        definitive -> quasi-definitive -> adjusted -> reported -> best-available.
    samples_per_day:
        ``"Minute"`` (default), ``"Second"`` or ``"Hour"``.
    use_cache:
        Serve from and populate the on-disk cache.
    refresh:
        Bypass the station-catalogue cache and re-fetch it.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp`` (naive UTC, ``datetime64[ns]`` or ``datetime64[us]``
        depending on the installed pandas), ``X`` (North), ``Y``
        (East), ``Z`` (Down), ``F`` (total), all nT, sorted ascending and
        de-duplicated. Missing samples are ``NaN``. The resolved publication
        state and backend are attached in ``df.attrs``.
    dict
        On failure, an LLM-readable error with keys ``ok=False``, ``error``,
        ``message``, ``station``, ``start_date``, ``end_date``,
        ``requested_data_type``, ``attempts`` and optionally ``hint``.
    """
    try:
        station = _norm_station(station_code)
        start = _norm_date(start_date, "start_date")
        end = _norm_date(end_date, "end_date")
        wanted = _norm_type(data_type)
    except ValueError as exc:
        return _error("invalid_input", str(exc), requested_type=str(data_type))

    if end < start:
        return _error(
            "invalid_input",
            f"end_date {end.isoformat()} is before start_date {start.isoformat()}",
            station=station,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

    if use_cache and not refresh:
        cached = load_from_cache(station, start, end, wanted, samples_per_day)
        if cached is not None:
            cached.attrs["from_cache"] = True
            return cached

    try:
        known = _load_station_registry(refresh=refresh)
    except RuntimeError as exc:
        return _error(
            "catalogue_unavailable",
            f"Could not reach any station catalogue: {exc}",
            station=station,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            requested_type=wanted,
        )

    if station not in known:
        near = [c for c in known if station[:1] == c[:1]][:8]
        return _error(
            "station_not_found",
            f"Station {station!r} is not in the INTERMAGNET catalogue "
            f"({len(known)} observatories known from Edinburgh GIN + Kyoto WDC).",
            station=station,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            requested_type=wanted,
            hint=(
                "Check the code against get_available_stations(). Codes starting "
                f"{station[:1]!r} that do exist: {near or 'none'}. Note BGI and NGD "
                "are absent from both public providers."
            ),
        )

    if wanted == "auto":
        states = GIN_FALLBACK_CHAIN
    else:
        states = (wanted,) + tuple(s for s in GIN_FALLBACK_CHAIN if s != wanted)

    frames: list[pd.DataFrame] = []
    resolved: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _chunk_dates(start, end, CHUNK_DAYS):
        df, infos, attempts = _fetch_chunk(
            station, chunk_start, chunk_end, states, samples_per_day
        )
        if df is None:
            return _error(
                "no_data",
                f"No {station} data for {chunk_start.isoformat()}.."
                f"{chunk_end.isoformat()} at any publication state "
                f"({sum(1 for a in attempts if not a.get('ok'))} attempts failed).",
                station=station,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                requested_type=wanted,
                tried=attempts,
                hint=(
                    "Definitive data can lag by up to a year; try data_type='auto' "
                    "or a historical year. Stations served only by Kyoto have no "
                    "data after ~2022."
                ),
            )
        frames.append(df)
        for info in infos:
            resolved.append({"chunk": f"{chunk_start}..{chunk_end}", **info})

    if not frames:
        return _error(
            "no_data",
            f"No data assembled for {station}.",
            station=station,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            requested_type=wanted,
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="first")
        .reset_index(drop=True)
    )
    combined = combined[["timestamp", *COMPONENTS]]
    combined.attrs["station"] = station
    combined.attrs["samples_per_day"] = samples_per_day
    combined.attrs["source"] = resolved
    combined.attrs["publication_states"] = sorted({r["state"] for r in resolved})
    combined.attrs["backends"] = sorted({r["backend"] for r in resolved})

    if use_cache:
        try:
            save_to_cache(
                combined,
                station,
                (start, end),
                wanted,
                samples_per_day,
                meta={"source": resolved, "publication_states": combined.attrs["publication_states"]},
            )
            combined.attrs["cached"] = True
        except Exception as exc:
            log.warning("Could not cache %s: %s", station, exc)
    return combined


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("=" * 78)
    print("STEP 1 - station catalogue (5 of them)")
    print("=" * 78)
    stations = get_available_stations()
    print(f"total stations available: {len(stations)}")
    registry = get_station_details()
    edinburgh = sum(1 for m in registry.values() if "edinburgh_gin" in m["sources"])
    print(f"  served by Edinburgh GIN: {edinburgh}")
    print(f"  served only by Kyoto:   {len(stations) - edinburgh}\n")
    for i, (code, (name, lat, lon)) in enumerate(stations.items()):
        if i >= 5:
            break
        print(f"  {code:5s} {name[:44]:46s} lat={lat} lon={lon}")

    print()
    print("=" * 78)
    print("STEP 2 - fetch IRT (Irkutsk), 2 days of the previous year")
    print("=" * 78)
    today = date.today()
    start = date(today.year - 1, 3, 10)
    end = start + timedelta(days=1)
    print(f"requesting {start.isoformat()} .. {end.isoformat()} "
          f"data_type='definitive' (expect fallback to a provisional state)")

    result = fetch_observatory_data("IRT", start, end, data_type="definitive")
    if is_error(result):
        print("FAILED:")
        print(json.dumps(result, indent=2)[:2000])
        return 1

    df = result
    print(f"\nrequested state : definitive")
    print(f"resolved states : {df.attrs.get('publication_states')}")
    print(f"backend         : {df.attrs.get('backends')}")
    for info in df.attrs.get("source", []):
        print(f"  {info['chunk']}  state={info['state']:10s} fmt={info['format']:9s} "
              f"backend={info['backend']}")
    print(f"rows={len(df)}  from_cache={df.attrs.get('from_cache', False)}  "
          f"expected={2 * 1440}")
    print(f"coverage: {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(f"NaN per column: {df.isna().sum().to_dict()}")

    print()
    print("=" * 78)
    print("STEP 3 - df.head() / df.describe()")
    print("=" * 78)
    print(df.head())
    print()
    print(df.describe())

    print()
    print("=" * 78)
    print("STEP 4 - error handling for LLM agents")
    print("=" * 78)
    bad = fetch_observatory_data("BGI", start, end)
    print("station absent from both providers ->", json.dumps(
        {k: bad.get(k) for k in ("ok", "error", "message", "hint")}, ensure_ascii=False))
    unknown = fetch_observatory_data("ZZZ", start, end)
    print("unknown station                   ->", json.dumps(
        {k: unknown.get(k) for k in ("ok", "error", "message")}, ensure_ascii=False))
    empty = fetch_observatory_data("IRT", date(1990, 1, 1), date(1990, 1, 2))
    print("pre-network date                  ->", json.dumps(
        {k: empty.get(k) for k in ("ok", "error", "message")}, ensure_ascii=False))
    inverted = fetch_observatory_data("IRT", end, start)
    print("end before start                  ->", json.dumps(
        {k: inverted.get(k) for k in ("ok", "error", "message")}, ensure_ascii=False))
    badtype = fetch_observatory_data("IRT", start, end, data_type="nonsense")
    print("bad data_type                     ->", json.dumps(
        {k: badtype.get(k) for k in ("ok", "error", "message")}, ensure_ascii=False))

    print()
    print("=" * 78)
    print("STEP 5 - save to cache/test.csv and verify a cache hit")
    print("=" * 78)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE_DIR / "test.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({csv_path.stat().st_size / 1024:.1f} KiB)")
    again = fetch_observatory_data("IRT", start, end, data_type="definitive")
    print(f"second call served from cache: {again.attrs.get('from_cache', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
