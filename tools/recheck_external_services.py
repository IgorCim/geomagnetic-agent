"""Bounded official-service probes; login is used only in explicit authenticated mode."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import requests
from urllib.parse import urlencode


URLS = {
    "supermag_home": "https://supermag.jhuapl.edu/",
    "supermag_software": "https://supermag.jhuapl.edu/info/?page=software",
    "supermag_legacy_indices": "https://supermag.jhuapl.edu/services/indices.php?python&nohead",
    "supermag_hapi_indices": "https://supermag.jhuapl.edu/hapi/indices_all/catalog",
    "iswa_cygnets": "https://iswa.ccmc.gsfc.nasa.gov/catalog/cygnets",
    "iswa_feeds": "https://iswa.ccmc.gsfc.nasa.gov/catalog/data-feeds",
    "iswa_legacy_wms": "https://iswa.gsfc.nasa.gov/wmts/wmts.cgi?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("discovery", "details", "catalogs", "authenticated", "runtime"), default="discovery")
    options = parser.parse_args()
    urls = URLS if options.phase == "discovery" else {
        "supermag_mag_api": "https://supermag.jhuapl.edu/mag/?tab=api",
        "supermag_indices_api": "https://supermag.jhuapl.edu/indices/?tab=api",
        "iswa_catalog_js": "https://iswa.ccmc.gsfc.nasa.gov/catalog/assets/index-C7Mm8j3P.js",
    }
    if options.phase == "catalogs":
        urls = {
            "supermag_python_client": "https://supermag.jhuapl.edu/mag/lib/content/api/supermag-api.py",
            "iswa_all_feeds": "https://iswa.ccmc.gsfc.nasa.gov/api/info?mode=active%2Cevent%2Cretired",
            "iswa_cygnet_search": "https://iswa.ccmc.gsfc.nasa.gov/meta/library/cygnet-search.json",
        }
    if options.phase == "runtime":
        urls = {
            "llama_cuda_index": "https://abetlen.github.io/llama-cpp-python/whl/cu121/llama-cpp-python/",
            "llama_pinned_wheel": "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35-cu121/llama_cpp_python-0.3.35-py3-none-manylinux_2_35_x86_64.whl",
        }
    secret = os.environ.get("SUPERMAG_LOGON", "") if options.phase == "authenticated" else ""
    if options.phase == "authenticated":
        if not secret:
            raise RuntimeError("SUPERMAG_LOGON required; value is never printed")
        query = urlencode({"start": "2024-05-10T00:00:00", "extent": 60, "logon": secret})
        base = "https://supermag.jhuapl.edu/services/"
        urls = {
            "supermag_authenticated_indices": base + "indices.php?" + query + "&indices=sme,sml,smu&python&nohead",
            "supermag_authenticated_inventory": base + "inventory.php?" + query + "&python&nohead",
            "supermag_authenticated_sites": base + "data-api.php?" + query + "&station=IRT&python&nohead",
            "supermag_documented_data": base + "data.php?" + query + "&station=IRT&python&nohead",
        }

    def safe(value):
        return str(value).replace(secret, "[redacted]") if secret else str(value)
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "verification" / (stamp + "_external_recheck")
    output.mkdir(parents=True)

    def probe(item):
        name, url = item
        row = {"name": name, "url": safe(url)}
        try:
            method = requests.head if name == "llama_pinned_wheel" else requests.get
            response = method(url, timeout=(5, 20), allow_redirects=True, headers={"User-Agent": "geomagnetic-agent/verification"})
            body = response.content.replace(secret.encode(), b"[redacted]") if secret else response.content
            (output / (name + ".txt")).write_bytes(body)
            row.update(status=response.status_code, final_url=safe(response.url),
                       content_type=response.headers.get("Content-Type"), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest())
        except requests.RequestException as exc:
            row.update(error=type(exc).__name__, detail=safe(exc))
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(probe, urls.items()))
    (output / "result.json").write_text(json.dumps({"time_utc": stamp, "scope": f"{options.phase} probes; HTTP success alone is not data validation; login redacted", "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("EVIDENCE:", output)


if __name__ == "__main__":
    main()
