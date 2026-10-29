"""Build the 50-chart Gradio demo project (10 stations x 14-18 May 2024).

This is the demo counterpart of the 300-chart gate: same canonical names and
project/export contract, but only five days and ten stations.  It is deliberately
small enough to run on a laptop and to show in a live session.

    python demo_make_50.py

It creates ``projects/demo_storm_may2024`` and two archives in ``exports/``:
the default canonical ZIP and the custom ``storm_report_2024_demo.zip`` view.
Pass ``--compact`` to make both archives share one Plotly bundle; the project
tree and its self-contained HTML files are not changed.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

import chart_naming as cn
import geomag_analyzer as ga
import geomag_plotter as gp
import intermagnet_loader as il
import project_store as ps

STATIONS = ("IRT", "API", "BSL", "ABG", "BRW", "SHU", "FRD", "CMO", "BOU", "TUC")
DATES = ("2024-05-14", "2024-05-15", "2024-05-16", "2024-05-17", "2024-05-18")
PROJECT = "demo_storm_may2024"
CUSTOM_ARCHIVE = "storm_report_2024_demo"
CUSTOM_FOLDERS = {
    "1": ["irt", "api", "bsl", "abg"],
    "2": ["brw", "shu", "frd"],
    "3": ["cmo", "bou", "tuc"],
}
CUSTOM_SORT = ["date", "station"]
CUSTOM_SOURCE_ORDER = ["OMNI", "INTERMAGNET", "KYOTO", "SUPERMAG", "NOAA", "ISWA"]


def _day(frame, day: str):
    stamps = frame["timestamp"].dt.strftime("%Y-%m-%d")
    return frame.loc[stamps == day].copy()


def _dmy(day: str) -> str:
    return f"{day[8:10]}.{day[5:7]}.{day[0:4]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Share one Plotly bundle in the exported ZIPs (project files stay self-contained).",
    )
    args = parser.parse_args()

    store = ps.Workspace()
    created = store.create(args.project, stations=STATIONS, dates=DATES)
    if not created.get("ok"):
        print(json.dumps(created, ensure_ascii=False, indent=2))
        return 1

    problems: list[str] = []
    charts = 0
    with tempfile.TemporaryDirectory(prefix="demo50_") as tmp:
        tmp_dir = Path(tmp)
        for station in STATIONS:
            frame = il.fetch_observatory_data(
                station, DATES[0], DATES[-1], data_type="best-avail"
            )
            if isinstance(frame, dict):
                problems.append(
                    f"{station}: {frame.get('error')}: {frame.get('message')}"
                )
                continue
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            derived_all = ga.calculate_derived_components(frame)
            if ga.is_error(derived_all):
                problems.append(f"{station}: derive failed: {derived_all}")
                continue
            for day in DATES:
                raw_day = _day(frame, day)
                derived_day = _day(derived_all, day)
                if raw_day.empty or derived_day.empty:
                    problems.append(f"{station}/{day}: empty day")
                    continue
                saved_raw = store.save(station, day, raw_day, kind="raw")
                saved_derived = store.save(station, day, derived_day, kind="derived")
                if not saved_raw.get("ok") or not saved_derived.get("ok"):
                    problems.append(
                        f"{station}/{day}: save failed "
                        f"{saved_raw} / {saved_derived}"
                    )
                    continue
                handle = f"derived:{station.lower()}:{day}"
                filename = cn.components_filename(handle, ["H", "D", "I"])
                title = f"{station} {_dmy(day)} H/D/I — INTERMAGNET"
                out = gp.plot_components(
                    derived_day,
                    ["H", "D", "I"],
                    title=title,
                    output_dir=tmp_dir,
                    filename=filename,
                    include_plotlyjs=True,
                )
                if gp.is_error(out):
                    problems.append(f"{station}/{day}: plot failed: {out}")
                    continue
                written = store.save_chart(
                    out,
                    [handle],
                    kind="components",
                    metadata={
                        "components": ["H", "D", "I"],
                        "product": "components",
                        "handles": [handle],
                    },
                )
                if not written:
                    problems.append(f"{station}/{day}: chart was not filed")
                else:
                    charts += 1

    default_zip = ps.export_project(
        args.project,
        archive_name=args.project,
        sort_by=["station", "start_date", "components", "source", "file"],
        compact=args.compact,
    )
    custom_zip = ps.export_project(
        args.project,
        archive_name=CUSTOM_ARCHIVE,
        folders=CUSTOM_FOLDERS,
        sort_by=CUSTOM_SORT,
        source_order=CUSTOM_SOURCE_ORDER,
        compact=args.compact,
    )

    tree = ps.project_tree(args.project)
    html_files = [
        entry for entry in tree
        if str(entry.get("path", "")).endswith(".html")
    ]
    summary = {
        "project": args.project,
        "charts_written_this_run": charts,
        "html_in_tree": len(html_files),
        "problems": problems,
        "default_zip": default_zip,
        "custom_zip": custom_zip,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if not problems and len(html_files) == 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
