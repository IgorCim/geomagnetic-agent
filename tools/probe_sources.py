"""Bounded public-endpoint probes; no credentials or model are used.

An HTTP 200 catalog proves reachability, not correct data retrieval. This
script deliberately makes no claim about authenticated SuperMAG requests.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import requests

ENDPOINTS = {
    "INTERMAGNET catalog": "https://imag-data.bgs.ac.uk/GIN_V1/GINServices?Request=GetCapabilities&Format=json",
    "OMNI metadata": "https://cdaweb.gsfc.nasa.gov/hapi/info?id=OMNI_HRO2_1MIN",
    "Kyoto catalog": "https://wdc.kugi.kyoto-u.ac.jp/hapi/catalog",
    "NOAA alerts": "https://services.swpc.noaa.gov/products/alerts.json",
    "NOAA forecast": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json",
    "ISWA catalog": "https://iswa.ccmc.gsfc.nasa.gov/hapi/catalog",
    "SuperMAG public service": "https://supermag.jhuapl.edu/services/indices.php?python&nohead",
}


def probe(item):
    name, url = item
    try:
        response = requests.get(url, timeout=(5, 20), headers={"User-Agent": "geomagnetic-agent-verification/1.0"})
        record = {"name": name, "url": url, "http_status": response.status_code, "content_type": response.headers.get("Content-Type"), "bytes": len(response.content)}
        try:
            value = response.json()
            record["shape"] = type(value).__name__
            record["keys_or_rows"] = list(value)[:12] if isinstance(value, dict) else len(value) if isinstance(value, list) else None
        except ValueError:
            record["text_prefix"] = response.text[:180]
        return record
    except requests.RequestException as exc:
        return {"name": name, "url": url, "request_error": type(exc).__name__, "message": str(exc)[:400]}


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(__file__).resolve().parents[1] / "verification" / (stamp + "_sources")
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(probe, ENDPOINTS.items()))
    report = {"time_utc": stamp, "scope": "Public endpoint reachability only; no authenticated SuperMAG, no full source coverage, no scientific validation", "results": results}
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Evidence:", output)


if __name__ == "__main__":
    main()
