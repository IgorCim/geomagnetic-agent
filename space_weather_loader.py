"""Solar-wind and geomagnetic-index loader (Phase 6: space weather).

The magnetometer tells us *what* the field did; this module fetches the
*reason* -- the solar wind streaming past L1 -- and the classic indices that
summarise how the magnetosphere answered.

Data sources
------------
* **NASA CDAWeb HAPI** -- https://cdaweb.gsfc.nasa.gov/hapi
  The OMNI 1-minute near-Earth solar-wind set (``OMNI_HRO2_1MIN``). OMNI merges
  measurements from the L1 monitors (ACE/DSCOVR/Wind) and time-shifts the
  magnetic field to the bow shock, so a row is a clean "input" the geospace
  response can be regressed against. Parameters used here: ``flow_speed`` (V,
  km/s), ``proton_density`` (Np, cm^-3), ``T`` (K) and the field components
  ``BX_GSE``/``BY_GSM``/``BZ_GSM`` (nT).
* **WDC for Geomagnetism, Kyoto** -- https://wdc.kugi.kyoto-u.ac.jp/hapi
  The Dst ring-current index (hourly, ``hour_dst``) and the planetary Kp index
  (3-hourly, ``hour3h_kp``). Kyoto is the *producing* institution for Dst, so
  this is the canonical source rather than a mirror.

Both servers speak HAPI over plain HTTP and return CSV, which keeps this module
headless-safe: no browser, no display, no static-image engine.

Conventions
-----------
``Bz`` is reported in **GSM**, the frame in which the north-south IMF component
drives reconnection; it is what :func:`physics_engine.calculate_coupling_function`
accepts (``bz_gsm``). ``Bx`` is identical in GSE and GSM (the frames share the X
axis), and ``By`` is taken in GSM as is conventional for solar-wind work.

Contract (for tool/LLM callers)
-------------------------------
* Success  -> :class:`pandas.DataFrame`.
* Failure  -> a self-describing ``dict`` with ``{"ok": False, "error": ...}``.
  ``error`` is one of ``invalid_input``, ``api_unavailable`` or ``no_data``;
  branch on it with :func:`is_error`. ``hint`` explains how to recover.

Gaps in the OMNI record (instrument outages, or minutes with no plasma data)
are returned as ``NaN`` rows rather than dropped, so the time axis stays
faithful and downstream ``ffill``/interpolation is the caller's choice.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

__all__ = [
    "fetch_solar_wind",
    "fetch_geomagnetic_indices",
    "is_error",
    "CDAWEB_HAPI_URL",
    "KYOTO_HAPI_URL",
    "SOLAR_WIND_DATASETS",
    "DST_DATASET",
    "KP_DATASET",
]

log = logging.getLogger("space_weather_loader")

CDAWEB_HAPI_URL = "https://cdaweb.gsfc.nasa.gov/hapi"
KYOTO_HAPI_URL = "https://wdc.kugi.kyoto-u.ac.jp/hapi"

#: OMNI 1-minute products, best first. ``OMNI_HRO2_1MIN`` is the current
#: bow-shock time-shifted 1-minute set; ``OMNI_HRO_1MIN`` is the earlier
#: revision, kept as a fallback in case the primary is retired.
SOLAR_WIND_DATASETS: tuple[str, ...] = ("OMNI_HRO2_1MIN", "OMNI_HRO_1MIN")

#: Kyoto WDC HAPI datasets for the ring-current and planetary indices.
DST_DATASET = "hour_dst"
KP_DATASET = "hour3h_kp"

DEFAULT_TIMEOUT = 90
DEFAULT_MAX_RETRIES = 3

#: OMNI HAPI parameter -> public column. Bx is frame-invariant (GSE == GSM on
#: the X axis); By/Bz are the GSM components used for coupling studies.
_SOLAR_WIND_COLUMNS: dict[str, str] = {
    "flow_speed": "V",
    "proton_density": "Np",
    "T": "T",
    "BX_GSE": "Bx",
    "BY_GSM": "By",
    "BZ_GSM": "Bz",
}

_SOLAR_WIND_ORDER: tuple[str, ...] = ("timestamp", "V", "Np", "T", "Bx", "By", "Bz")
_INDEX_ORDER: tuple[str, ...] = ("timestamp", "Dst", "Kp")

_session: requests.Session | None = None


# --------------------------------------------------------------------------- #
# contract helpers
# --------------------------------------------------------------------------- #
def is_error(result: Any) -> bool:
    """True when a fetch failed. Lets a caller branch without isinstance checks."""
    return isinstance(result, dict) and result.get("ok") is False


def _error(code: str, message: str, *, hint: str | None = None, **extra: Any) -> dict[str, Any]:
    """Build the stable, self-describing error payload returned to callers."""
    payload: dict[str, Any] = {"ok": False, "error": code, "message": message}
    payload.update(extra)
    if hint:
        payload["hint"] = hint
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
    """GET/POST with exponential backoff on transient failures.

    A HAPI server that 429s us under a burst (both CDAWeb and Kyoto do) is
    retried rather than surfaced as an outage.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _http().request(method, url, params=params, timeout=timeout)
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


def _safe_get(
    url: str, params: dict[str, Any], timeout: int, retries: int
) -> tuple[requests.Response | None, dict[str, Any] | None]:
    """GET that turns transport failures into an ``api_unavailable`` marker."""
    try:
        return _request("GET", url, params=params, timeout=timeout, retries=retries), None
    except (ConnectionError, requests.RequestException) as exc:
        return None, {"error": "api_unavailable", "detail": str(exc)}


# --------------------------------------------------------------------------- #
# input normalisation
# --------------------------------------------------------------------------- #
def _norm_day(value: Any, label: str) -> date:
    """Accept ``YYYY-MM-DD`` (or date/datetime) and return a ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(text[: len(fmt) + 4], fmt).date()
            except ValueError:
                continue
    raise ValueError(
        f"{label} must be a YYYY-MM-DD string or a date/datetime, got {value!r}"
    )


# --------------------------------------------------------------------------- #
# HAPI plumbing
# --------------------------------------------------------------------------- #
def _parse_hapi_csv(
    text: str, names: list[str], fills: dict[str, Any]
) -> pd.DataFrame | None:
    """Decode a header-less HAPI CSV body against the ``/info`` schema.

    HAPI's default CSV has no header, and fill values (``99999.9``, ``999.99``,
    ...) stand in for missing samples. ``names`` comes from ``/info`` so the
    columns land in the documented order; each fill is then rewritten to
    ``NaN``. Returns ``None`` when the body carries no data rows.
    """
    if not names:
        return None
    width = len(names)
    rows = [
        line.split(",")
        for line in text.splitlines()
        if line and not line.startswith("#")
    ]
    rows = [row for row in rows if len(row) == width]
    if not rows:
        return None

    frame = pd.DataFrame(rows, columns=names)
    if "Time" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["Time"], errors="coerce", utc=True)
        frame = frame.drop(columns=["Time"])

    for column in frame.columns:
        if column == "timestamp":
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        fill = fills.get(column)
        if fill is not None:
            try:
                values = values.mask(values == float(fill))
            except (TypeError, ValueError):
                pass
        frame[column] = values
    return frame


def _fetch_table(
    base: str,
    dataset: str,
    start_iso: str,
    stop_iso: str,
    time_keys: tuple[str, str],
    timeout: int,
    retries: int,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Fetch one HAPI dataset as a DataFrame, or an error marker.

    Returns ``(df, None)`` on success (``df`` may be ``None`` when the range is
    empty), or ``(None, error)`` when the service is unreachable / rejects us.
    """
    resp, err = _safe_get(f"{base}/info", {"id": dataset}, timeout, retries)
    if err is not None:
        return None, err
    if resp.status_code != 200:
        return None, {
            "error": "api_unavailable",
            "detail": f"/info?id={dataset} -> HTTP {resp.status_code}",
        }
    try:
        info = resp.json()
    except ValueError:
        return None, {
            "error": "api_unavailable",
            "detail": f"/info?id={dataset} returned non-JSON",
        }
    params = info.get("parameters") or []
    names = [str(p.get("name")) for p in params if p.get("name")]
    if not names:
        return None, {
            "error": "api_unavailable",
            "detail": f"dataset {dataset} advertises no parameters",
        }
    fills = {
        str(p.get("name")): p.get("fill")
        for p in params
        if p.get("name") and p.get("fill") not in (None, "")
    }

    query = {"id": dataset, time_keys[0]: start_iso, time_keys[1]: stop_iso, "format": "csv"}
    data_resp, err = _safe_get(f"{base}/data", query, timeout, retries)
    if err is not None:
        return None, err
    if data_resp.status_code != 200:
        detail = ""
        try:
            detail = data_resp.json().get("status", {}).get("message", "")
        except ValueError:
            detail = data_resp.text[:200]
        return None, {
            "error": "api_unavailable",
            "detail": f"/data?id={dataset} -> HTTP {data_resp.status_code}: {detail}",
        }
    return _parse_hapi_csv(data_resp.text, names, fills), None


def _window(start: date, end: date) -> tuple[str, str]:
    """Inclusive day range -> half-open HAPI ISO window ``[start, end+1d)``."""
    stop = end + timedelta(days=1)
    return f"{start.isoformat()}T00:00:00Z", f"{stop.isoformat()}T00:00:00Z"


# --------------------------------------------------------------------------- #
# public fetchers
# --------------------------------------------------------------------------- #
def fetch_solar_wind(
    start_date: Any,
    end_date: Any,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> pd.DataFrame | dict[str, Any]:
    """Download near-Earth solar-wind parameters from NASA CDAWeb (OMNI).

    Parameters
    ----------
    start_date, end_date:
        ``YYYY-MM-DD`` (or ``date``/``datetime``). Inclusive.
    timeout, retries:
        HAPI transport tuning; the defaults (90 s, 3 tries) suit the public
        servers.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp`` (UTC), ``V`` (km/s), ``Np`` (cm^-3), ``T`` (K),
        ``Bx``, ``By``, ``Bz`` (nT). One row per minute; instrument gaps are
        ``NaN``. ``df.attrs`` records the dataset and the GSM note.
    dict
        ``{"ok": False, "error": "invalid_input" | "api_unavailable" | "no_data",
        "hint": ...}`` on failure.

    Examples
    --------
    >>> df = fetch_solar_wind("2024-09-12", "2024-09-12")   # doctest: +SKIP
    >>> df[["V", "Bz"]].describe()                          # doctest: +SKIP
    """
    try:
        start = _norm_day(start_date, "start_date")
        end = _norm_day(end_date, "end_date")
    except (TypeError, ValueError) as exc:
        return _error(
            "invalid_input",
            str(exc),
            hint="Pass dates as 'YYYY-MM-DD', e.g. fetch_solar_wind('2024-09-12', '2024-09-12').",
        )
    if end < start:
        return _error(
            "invalid_input",
            f"end_date {end} precedes start_date {start}.",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

    start_iso, stop_iso = _window(start, end)

    table: pd.DataFrame | None = None
    dataset_used: str | None = None
    detail = ""
    for dataset in SOLAR_WIND_DATASETS:
        table, err = _fetch_table(
            CDAWEB_HAPI_URL, dataset, start_iso, stop_iso,
            ("time.min", "time.max"), timeout, retries,
        )
        if err is not None:
            detail = err.get("detail") or err.get("error") or ""
            log.warning("CDAWeb dataset %s unavailable: %s", dataset, detail)
            continue
        dataset_used = dataset
        break

    if dataset_used is None:
        return _error(
            "api_unavailable",
            "NASA CDAWeb HAPI is unreachable or rejected the request for "
            + ", ".join(SOLAR_WIND_DATASETS)
            + ".",
            hint="Check connectivity to https://cdaweb.gsfc.nasa.gov/hapi and retry.",
            detail=detail,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

    if table is None or table.empty:
        return _error(
            "no_data",
            f"OMNI returned no solar-wind samples for {start}..{end}.",
            hint="OMNI starts in 1995 and lags real time by a few days; "
            "pick a date inside that window.",
            dataset=dataset_used,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

    frame = pd.DataFrame({"timestamp": table["timestamp"]})
    for source, target in _SOLAR_WIND_COLUMNS.items():
        frame[target] = table[source] if source in table.columns else np.nan
    frame = frame.loc[:, _SOLAR_WIND_ORDER]

    if frame.dropna(subset=["V", "Np", "T", "Bx", "By", "Bz"], how="all").empty:
        return _error(
            "no_data",
            f"OMNI returned only fill values for {start}..{end}; no usable samples.",
            hint="Pick a day with real solar-wind coverage.",
            dataset=dataset_used,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    frame.attrs["source"] = "NASA CDAWeb HAPI (OMNI)"
    frame.attrs["dataset"] = dataset_used
    frame.attrs["start_date"] = start.isoformat()
    frame.attrs["end_date"] = end.isoformat()
    frame.attrs["note"] = "By/Bz are GSM; Bx is identical in GSE and GSM; gaps are NaN."
    return frame


def _merge_indices(
    dst: pd.DataFrame | None, kp: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Combine hourly Dst and 3-hourly Kp onto one hourly time axis.

    Dst is published at the centre of each UT hour and is floored to the hour.
    Kp is a *3-hour* index, published at the centre of its window (01:30 covers
    00:00-03:00 UT), so its value is broadcast across the three hours it
    describes. The result is a dense hourly frame with both columns.
    """
    pieces: list[pd.DataFrame] = []

    if dst is not None and not dst.empty and "dstValue" in dst.columns:
        d = dst.loc[:, ["timestamp", "dstValue"]].rename(columns={"dstValue": "Dst"})
        d = d.dropna(subset=["timestamp"])
        d["timestamp"] = d["timestamp"].dt.floor("h")
        d = d.drop_duplicates(subset=["timestamp"], keep="last")
        pieces.append(d)

    if kp is not None and not kp.empty and "kpValue" in kp.columns:
        k = kp.loc[:, ["timestamp", "kpValue"]].rename(columns={"kpValue": "Kp"})
        k = k.dropna(subset=["timestamp"])
        expanded: list[dict[str, Any]] = []
        for block_start, value in zip(k["timestamp"].dt.floor("3h"), k["Kp"]):
            for hour in range(3):
                expanded.append(
                    {"timestamp": block_start + pd.Timedelta(hours=hour), "Kp": value}
                )
        if expanded:
            pieces.append(pd.DataFrame(expanded))

    if not pieces:
        return None

    frame = pieces[0]
    for piece in pieces[1:]:
        frame = frame.merge(piece, on="timestamp", how="outer")
    for column in ("Dst", "Kp"):
        if column not in frame.columns:
            frame[column] = np.nan
    return (
        frame.loc[:, _INDEX_ORDER]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_geomagnetic_indices(
    start_date: Any,
    end_date: Any,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> pd.DataFrame | dict[str, Any]:
    """Download the Dst and Kp indices from WDC for Geomagnetism, Kyoto.

    Parameters
    ----------
    start_date, end_date:
        ``YYYY-MM-DD`` (or ``date``/``datetime``). Inclusive.
    timeout, retries:
        HAPI transport tuning.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp`` (UTC, hourly), ``Dst`` (nT) and ``Kp``
        (0-9 scale). Dst is the hourly ring-current index; Kp is the 3-hourly
        planetary index broadcast over the three hours it covers.
    dict
        ``{"ok": False, "error": "invalid_input" | "api_unavailable" | "no_data",
        "hint": ...}`` on failure.

    Examples
    --------
    >>> df = fetch_geomagnetic_indices("2024-09-12", "2024-09-12")  # doctest: +SKIP
    >>> float(df["Dst"].min()) < -50                                 # doctest: +SKIP
    True
    """
    try:
        start = _norm_day(start_date, "start_date")
        end = _norm_day(end_date, "end_date")
    except (TypeError, ValueError) as exc:
        return _error(
            "invalid_input",
            str(exc),
            hint="Pass dates as 'YYYY-MM-DD', e.g. "
            "fetch_geomagnetic_indices('2024-09-12', '2024-09-12').",
        )
    if end < start:
        return _error(
            "invalid_input",
            f"end_date {end} precedes start_date {start}.",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

    start_iso, stop_iso = _window(start, end)

    dst, dst_err = _fetch_table(
        KYOTO_HAPI_URL, DST_DATASET, start_iso, stop_iso,
        ("start", "stop"), timeout, retries,
    )
    kp, kp_err = _fetch_table(
        KYOTO_HAPI_URL, KP_DATASET, start_iso, stop_iso,
        ("start", "stop"), timeout, retries,
    )

    if dst_err is not None or kp_err is not None:
        failed = [name for name, err in (("Dst", dst_err), ("Kp", kp_err)) if err is not None]
        details = "; ".join(
            str(err.get("detail") or err.get("error"))
            for err in (dst_err, kp_err)
            if err is not None
        )
        return _error(
            "api_unavailable",
            "Kyoto WDC HAPI could not serve " + " and ".join(failed) + ".",
            hint="Retry shortly; the Kyoto HAPI service is occasionally rate-limited.",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            detail=details,
        )

    combined = _merge_indices(dst, kp)
    if combined is None or combined.empty:
        return _error(
            "no_data",
            f"No Dst/Kp samples for {start}..{end}.",
            hint="Kyoto covers 1957+; pick a date inside that range.",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

    combined.attrs["source"] = "WDC for Geomagnetism, Kyoto (HAPI)"
    combined.attrs["datasets"] = {"Dst": DST_DATASET, "Kp": KP_DATASET}
    combined.attrs["start_date"] = start.isoformat()
    combined.attrs["end_date"] = end.isoformat()
    combined.attrs["note"] = "Hourly axis; Kp (3-hourly) broadcast over its window."
    return combined


# --------------------------------------------------------------------------- #
# manual smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    for day in ("2024-09-10", "2024-09-12"):
        print(f"\n{'=' * 72}\n{day}\n{'=' * 72}")
        wind = fetch_solar_wind(day, day)
        print("\n-- solar wind (OMNI / CDAWeb) --")
        print(wind if is_error(wind) else wind.head())

        indices = fetch_geomagnetic_indices(day, day)
        print("\n-- Dst / Kp (Kyoto WDC) --")
        print(indices if is_error(indices) else indices.head())

        if not is_error(indices):
            print(
                f"Dst min={indices['Dst'].min():.0f} nT, "
                f"Kp max={indices['Kp'].max():.1f}"
            )
        if not is_error(wind):
            print(f"Bz min={wind['Bz'].min():.2f} nT (negative = southward)")
