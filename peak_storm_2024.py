"""Build a small scientific supplement for the 10-12 May 2024 storm peak.

The main 300-chart gate deliberately covers 1-10 May (B), 14 May-2 June (A)
and 14-18 May (C).  The Dst minimum (-406 nT) and Kp 9 occurred on 11 May, so
the gate contains the onset but not the peak.  This script adds 20 charts
outside that gate:

* 5 high-latitude stations x 3 days (10/11/12 May) H-D-I panels = 15 charts;
* 5 stations on 11 May, H vs OMNI Bz/V = 5 charts.

It does not touch ``stress_g3solehz``, ``stress_3ffd1r21`` or the 300-chart
matrix code.  Run it from the repository root:

    python peak_storm_2024.py

Output defaults to ``stress_runs/peak_storm_2024_supplement``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

import chart_naming as cn
import geomag_analyzer as ga
import geomag_plotter as gp
import intermagnet_loader as il
import physics_engine as pe
import space_weather_loader as sw

STATIONS = ("BRW", "CMO", "SHU", "FRD", "BOU")
DATES = ("2024-05-10", "2024-05-11", "2024-05-12")
PEAK_DATE = "2024-05-11"
OMNI_HANDLE = "sw:omni:2024-05-01"
KYOTO_HANDLE = "idx:kyoto:2024-05-01"
DEFAULT_OUTPUT = Path("stress_runs") / "peak_storm_2024_supplement"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _day(frame: pd.DataFrame, day: str) -> pd.DataFrame:
    stamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    mask = stamps.dt.strftime("%Y-%m-%d") == day
    return frame.loc[mask].copy()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _row(
    *,
    path: Path,
    station: str,
    day: str,
    components: Iterable[str],
    product: str,
    source: str,
) -> dict[str, object]:
    return {
        "file": path.name,
        "station": station.upper(),
        "date": day,
        "components": "-".join(components),
        "product": product,
        "source": source,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _html_index(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    trs = "\n".join(
        "<tr>"
        f"<td><a href=\"charts/{row['file']}\">{row['file']}</a></td>"
        f"<td>{row['station']}</td><td>{row['date']}</td>"
        f"<td>{row['components']}</td><td>{row['product']}</td>"
        f"<td>{row['source']}</td>"
        "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<title>Peak storm 2024 supplement</title>
<style>
body {{ font: 14px/1.45 system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: .45rem .5rem; text-align: left; }}
th {{ background: #f4f4f4; }}
code {{ background: #f6f6f6; padding: .1rem .25rem; }}
</style>
<h1>10–12 мая 2024: supplement к 300-графиковому gate</h1>
<p>Пик бури: Dst <b>{summary['dst_min']} nT</b> в {summary['dst_min_time']};
Kp <b>{summary['kp_max']}</b>. Основной 300-набор содержит onset 10 мая, но не
11 мая; этот supplement добавлен отдельно и не меняет
<code>stress_g3solehz</code>.</p>
<p>20 графиков: 5 станций × 3 дня H/D/I и 5 станций за 11 мая H vs OMNI Bz/V.
Источники: INTERMAGNET best-avail, NASA CDAWeb/OMNI, WDC Kyoto.</p>
<table>
<thead><tr><th>Файл</th><th>Станция</th><th>Дата</th><th>Компоненты</th>
<th>Продукт</th><th>Источник</th></tr></thead>
<tbody>{trs}</tbody>
</table>
<p>Полные метаданные: <a href="index.csv">index.csv</a>,
<a href="manifest.json">manifest.json</a>. Числа Dst/Kp — Kyoto WDC на момент
генерации; Dst может быть provisional.</p>
</html>
"""


def build(
    output: Path,
    *,
    stations: tuple[str, ...] = STATIONS,
    dates: tuple[str, ...] = DATES,
    make_zip: bool = True,
) -> dict[str, object]:
    output = output.resolve()
    charts_dir = output / "charts"
    data_dir = output / "data"
    charts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # One request per station for the whole window; the same frame feeds three
    # daily charts and is saved for reproducibility.
    station_frames: dict[str, pd.DataFrame] = {}
    for station in stations:
        frame = il.fetch_observatory_data(
            station, dates[0], dates[-1], data_type="best-avail"
        )
        if isinstance(frame, dict):
            raise RuntimeError(
                f"{station}: {frame.get('error')}: {frame.get('message')}"
            )
        frame = ga.calculate_derived_components(frame)
        station_frames[station] = frame
        _write_csv(frame, data_dir / f"{station}_{dates[0]}_{dates[-1]}_derived.csv")

    omni = sw.fetch_solar_wind(dates[0], dates[-1])
    if isinstance(omni, dict):
        raise RuntimeError(f"OMNI: {omni.get('error')}: {omni.get('message')}")
    _write_csv(omni, data_dir / "omni_2024-05-10_2024-05-12.csv")

    kyoto = sw.fetch_geomagnetic_indices(dates[0], dates[-1])
    if isinstance(kyoto, dict):
        raise RuntimeError(f"Kyoto: {kyoto.get('error')}: {kyoto.get('message')}")
    _write_csv(kyoto, data_dir / "kyoto_2024-05-10_2024-05-12.csv")

    rows: list[dict[str, object]] = []
    for station in stations:
        frame = station_frames[station]
        for day in dates:
            daily = _day(frame, day)
            filename = cn.components_filename(
                f"derived:{station.lower()}:{day}", ["H", "D", "I"]
            )
            title = (
                f"{station} {day[8:10]}.{day[5:7]}.{day[0:4]} H/D/I — INTERMAGNET"
            )
            path = gp.plot_components(
                daily,
                ["H", "D", "I"],
                title=title,
                output_dir=charts_dir,
                filename=filename,
                include_plotlyjs=True,
            )
            if gp.is_error(path):
                raise RuntimeError(f"{filename}: {path}")
            rows.append(
                _row(
                    path=Path(path),
                    station=station,
                    day=day,
                    components=("H", "D", "I"),
                    product="components",
                    source="INTERMAGNET",
                )
            )

    # The 11 May H-vs-Bz/V views are the actual peak comparison.  The same OMNI
    # frame is aligned separately to each station's minute clock.
    omni_peak = _day(omni, PEAK_DATE)
    for station in stations:
        daily = _day(station_frames[station], PEAK_DATE)
        aligned = pe.align_solar_wind(daily, omni_peak)
        if pe.is_error(aligned):
            raise RuntimeError(f"{station} align: {aligned}")
        filename = cn.space_weather_filename(
            f"derived:{station.lower()}:{PEAK_DATE}",
            OMNI_HANDLE,
            "H",
            ["Bz", "V"],
        )
        title = cn.space_weather_title(
            f"derived:{station.lower()}:{PEAK_DATE}",
            OMNI_HANDLE,
            "H",
            ["Bz", "V"],
        )
        path = gp.plot_space_weather_comparison(
            aligned,
            ground_component="H",
            solar_wind_components=("Bz", "V"),
            title=title,
            output_dir=charts_dir,
            filename=filename,
            include_plotlyjs=True,
        )
        if gp.is_error(path):
            raise RuntimeError(f"{filename}: {path}")
        rows.append(
            _row(
                path=Path(path),
                station=station,
                day=PEAK_DATE,
                components=("H", "Bz", "V"),
                product="space_weather",
                source="INTERMAGNET-OMNI",
            )
        )

    rows.sort(key=lambda row: (str(row["date"]), str(row["station"]), str(row["file"])))
    with (output / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dst = kyoto.dropna(subset=["Dst"])
    kp = kyoto.dropna(subset=["Kp"])
    dst_row = dst.loc[dst["Dst"].idxmin()]
    kp_row = kp.loc[kp["Kp"].idxmax()]
    summary = {
        "dst_min": round(float(dst_row["Dst"]), 2),
        "dst_min_time": str(dst_row["timestamp"]),
        "kp_max": round(float(kp_row["Kp"]), 2),
        "kp_max_time": str(kp_row["timestamp"]),
        "charts": len(rows),
        "stations": list(stations),
        "dates": list(dates),
        "peak_date": PEAK_DATE,
    }
    (output / "index.html").write_text(
        _html_index(rows, summary), encoding="utf-8"
    )
    manifest = {
        "supplement": "peak_storm_2024",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outside_300": True,
        "does_not_modify": ["stress_g3solehz", "stress_3ffd1r21"],
        "charts": len(rows),
        "counts": {
            "components": 3 * len(stations),
            "space_weather": len(stations),
            "total": len(rows),
        },
        "summary": summary,
        "sources": {
            "ground": "INTERMAGNET best-avail via Edinburgh GIN",
            "solar_wind": "NASA CDAWeb OMNI (sw:omni:2024-05-01)",
            "indices": "WDC Kyoto (Dst, Kp)",
            "supermag": "not used locally: SUPERMAG_LOGON absent",
        },
        "files": {
            "charts": "charts/",
            "index": "index.csv",
            "readme": "README.md",
            "data": "data/",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        f"""# Peak storm 2024 supplement

Это **не часть 300-графикового gate** и не замена `stress_g3solehz`.
Supplement добавлен потому, что 300-набор заканчивается 10 мая (B) или
начинается 14 мая (A/C), а пик бури пришёлся на 11 мая.

## Что внутри

- {len(rows)} HTML: {len(stations)} станций × 3 дня H/D/I и {len(stations)}
  станций за 11 мая H vs OMNI Bz/V.
- `index.csv`, `index.html`, `manifest.json`.
- `data/` — derived-ряды станций, OMNI и Kyoto за 10–12 мая.
- `PEAK_STORM_2024_SUPPLEMENT.zip` — те же файлы одним архивом.

## Пик

| Параметр | Значение |
|---|---:|
| Dst min | {summary['dst_min']} nT |
| Dst min time | {summary['dst_min_time']} |
| Kp max | {summary['kp_max']} |
| Kp max time | {summary['kp_max_time']} |

Dst/Kp — WDC Kyoto, запрос {datetime.now(timezone.utc).date().isoformat()};
Dst может быть provisional. SuperMAG локально не использован:
`SUPERMAG_LOGON` не задан. NOAA/ISWA в supplement не добавлялись, потому что
их текущее окно не является историческим маем 2024.

## Как читать

- H-панель — nT; D/I — градусы, отдельные панели.
- Bz — nT, V — км/с, отдельные панели; это специально, чтобы V не
  «сплющивал» Bz.
- В C-заголовке `OMNI (sw:omni:2024-05-01)` — идентификатор descriptor, а не
  дата сравниваемых данных.
""",
        encoding="utf-8",
    )

    if make_zip:
        archive = output / "PEAK_STORM_2024_SUPPLEMENT.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(output.rglob("*")):
                if path.is_file() and path != archive:
                    zf.write(path, path.relative_to(output).as_posix())
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    manifest = build(args.output, make_zip=not args.no_zip)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
