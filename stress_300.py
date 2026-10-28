# ============================================================
# FULL AGENT STRESS TEST
# 300 charts / 6 data sources / 2 component matrices / 1 comparison matrix
#
# Paste this cell AFTER the real 14B brain is loaded.  It uses the `brain`
# object from the notebook namespace and does not load or clone anything.
#
# Coverage-valid set (verified against INTERMAGNET):
#   * BOX has no May 2024 data at any publication state -> excluded.
#   * API has no data before 2024-05-13.
#   * ABG has no 2024-05-13 day, so the 20-day window starts 2024-05-14.
#   * ABG replaces BOX; the window ends 2024-06-02 to give all ten stations
#     twenty complete days.
#
# The six requested 50-chart blocks are logical blocks:
#   1-4: one 10-station x 20-date matrix = 200 component charts
#   5:   one 5-station x 10-date matrix   =  50 component charts
#   6:   one 10-station x 5-date matrix   =  50 ground-vs-solar-wind charts
#
# Final gate also verifies the naming/export contract added around the data:
#   * every cell has the deterministic canonical basename and source suffix;
#   * charts.json carries source(s), station, date, components, kind, product;
#   * the normal matrix ZIP keeps its default name and canonical tree;
#   * a prompt-named archive/folders export is a view, not a project mutation.
# ============================================================
import csv
import io
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def _find_repo():
    """Locate the directory that actually contains agent_core.py.

    Kaggle setup cells disagree about whether the repo is at
    /content/geomagnetic_agent or /kaggle/working/geomagnetic-agent.  Trusting
    a stale DIR is what produced ``ModuleNotFoundError: No module named
    'agent_core'``.  Search the known locations and the parents of the current
    directory, then use the directory that really has the module.
    """
    candidates = []
    if "DIR" in globals():
        candidates.append(Path(DIR))
    candidates.extend(
        [
            Path.cwd(),
            Path("/kaggle/working/geomagnetic-agent"),
            Path("/content/geomagnetic_agent"),
        ]
    )
    for base in list(candidates):
        candidates.extend(base.parents)
    for root in (Path("/kaggle/working"), Path("/content"), Path.cwd()):
        if root.is_dir():
            try:
                for child in root.iterdir():
                    if child.is_dir():
                        candidates.append(child)
            except OSError:
                pass

    seen = set()
    resolved = []
    for candidate in candidates:
        try:
            candidate = candidate.expanduser().resolve()
        except OSError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "agent_core.py").is_file():
            resolved.append(candidate)

    # Prefer the checkout that also contains this file.  A stale
    # /content/geomagnetic_agent clone can have agent_core.py but not the
    # coverage-valid stress cell; choosing it would silently run old code.
    for candidate in resolved:
        if (candidate / "stress_300.py").is_file():
            return candidate
    if resolved:
        return resolved[0]
    raise RuntimeError(
        "agent_core.py not found. Set DIR to the repository directory "
        "(for example /kaggle/working/geomagnetic-agent) and rerun this cell."
    )


REPO = _find_repo()
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("REPO:", REPO)

import agent_core as ac
import chart_naming as cn
import project_store as ps

print("agent_core:", ac.__file__)
assert (
    Path(ac.__file__).resolve().parent == REPO
), "Загружен agent_core.py из другого каталога; перезапусти runtime и повтори ячейку."

assert "brain" in globals(), "Сначала выполни ячейку загрузки 14B."
assert not isinstance(brain, ac.ScriptedBrain), "Нужен реальный brain, не ScriptedBrain."
ac.OFFLINE = False

# Keep this run out of the persistent projects/ and exports/ roots.  The old
# 258-chart science run stays there untouched; the final gate below must count
# only the HTML written by this run.  A stale project tree was why a perfect
# 200/50/50 matrix run was reported as 305/100/50.
RUNS_ROOT = REPO / "stress_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR = Path(tempfile.mkdtemp(prefix="stress_", dir=str(RUNS_ROOT)))
RUN_PROJECTS = RUN_DIR / "projects"
RUN_EXPORTS = RUN_DIR / "exports"
RUN_PROJECTS.mkdir(parents=True, exist_ok=True)
RUN_EXPORTS.mkdir(parents=True, exist_ok=True)

STORE = ac.FrameStore(workspace=ps.Workspace(root=RUN_PROJECTS))
_ORIGINAL_EXPORTS_ROOT = ps.EXPORTS_ROOT
ps.EXPORTS_ROOT = RUN_EXPORTS
assert STORE.workspace.root != ps.PROJECTS_ROOT, (
    "stress run must not reuse the persistent projects/ root"
)
print("RUN_DIR:", RUN_DIR)
print("RUN_PROJECTS:", RUN_PROJECTS)
print("RUN_EXPORTS:", RUN_EXPORTS)

STATIONS = ["IRT", "API", "BSL", "ABG", "BRW", "SHU", "FRD", "CMO", "BOU", "TUC"]
HIGHLAT = ["BRW", "SHU", "FRD", "CMO", "BOU"]

# A: 10 x 20 = 200 charts, four logical 5-day blocks of 50.
DATES_A = [f"2024-05-{day:02d}" for day in range(14, 32)] + [
    "2024-06-01",
    "2024-06-02",
]
# B: 5 x 10 = 50 charts.  May 1-10 does not overlap matrix A, so every
# component chart is a distinct file, not a rewrite of an A chart.
DATES_B = [f"2024-05-{day:02d}" for day in range(1, 11)]
# C: 10 x 5 = 50 ground-vs-solar-wind charts, all ten stations covered.
STORM_DATES = [f"2024-05-{day:02d}" for day in range(14, 19)]

PROJECT_A = "may2024_matrix_A_200"
PROJECT_B = "may2024_matrix_B_50"
PROJECT_C = "may2024_spaceweather_50"

# Prompt-driven custom export: these names must survive from the prompt into
# the ZIP.  The project tree itself must stay canonical.
CUSTOM_ARCHIVE = "storm_report_2024"
CUSTOM_FOLDERS = {
    "1": ["irt", "api", "bsl", "abg"],
    "2": ["brw", "shu", "frd"],
    "3": ["cmo", "bou", "tuc"],
}
CUSTOM_SORT = ["date", "station"]
CUSTOM_SOURCE_ORDER = ["OMNI", "INTERMAGNET", "KYOTO", "SUPERMAG", "NOAA", "ISWA"]

assert len(STATIONS) == 10
assert len(HIGHLAT) == 5
assert len(DATES_A) == 20
assert len(DATES_B) == 10
assert len(STORM_DATES) == 5

MASTER_PROMPT = """
Я готовлю научный разбор геомагнитной активности мая 2024 и прошу полный
аудит агента на реальных данных, а не демонстрационный ответ.

Нужно 300 разных HTML-графиков, ровно по 50 в шести логических блоках:
1. 14–18 мая, 10 станций — 50 магнитограмм.
2. 19–23 мая, 10 станций — 50 магнитограмм.
3. 24–28 мая, 10 станций — 50 магнитограмм.
4. 29 мая–2 июня, 10 станций — 50 магнитограмм.
5. 1–10 мая, 5 высокоширотных станций — 50 магнитограмм.
6. 14–18 мая, 10 станций — 50 сравнений «земля — солнечный ветер».

Станции: IRT, API, BSL, ABG, BRW, SHU, FRD, CMO, BOU, TUC.
Высокоширотные: BRW, SHU, FRD, CMO, BOU.
BOX не использовать: у него нет данных за май 2024; вместо него ABG.
ABG не отдаёт 13 мая, поэтому рабочий набор начинается 14 мая 2024.

Блоки 1–4: одна матрица 10 станций × 20 дат, действия
["fetch_observatory_data", "calculate_derived_components", "plot_components"],
проект may2024_matrix_A_200. Блок 5: одна матрица 5 станций × 10 дат, проект
may2024_matrix_B_50. В каждом матричном запуске — ровно один create_task_matrix
и один process_matrix_batch(run_until_complete=True); без циклов и повторного
создания матрицы; экспорт ZIP обязателен.

Блок 6: ОДНА матрица 10 станций × 5 дат, действия
["fetch_observatory_data", "plot_space_weather_comparison"], проект
may2024_spaceweather_50. Передай shared_args:
{"solar_wind_df_descriptor": "sw:omni:2024-05-01",
 "ground_component": "H",
 "solar_wind_components": ["Bz", "V"]}.
Матрица сама подставит ground_df_descriptor = raw:<station>:<date> и уникальное
имя файла для каждой ячейки. Не вызывай сравнения по одному в цикле.

Обязательно использовать все 6 источников: INTERMAGNET (блоки 1–5), NASA
CDAWeb/OMNI (блок 6), Kyoto WDC Dst/Kp, SuperMAG SME/SML/SMU, NOAA SWPC
alerts/forecast/solar_events, NASA ISWA forecast. Для ISWA perturbation map
ожидается штатный ответ service_decommissioned — это не скрытая ошибка.

Не скрывай частичные и пустые станции: в отчёте должны быть полные, частичные
и пустые, с причинами. В конце перечисли проекты, ZIP-архивы, число файлов и
все ожидаемые ошибки внешних сервисов.
"""

print(MASTER_PROMPT)


def run_stage(label, prompt):
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    started = time.time()
    result = ac.run_agent_with_planner(
        prompt,
        brain=brain,
        store=STORE,
        verbose=True,
        use_llm_planner=True,
    )
    elapsed = time.time() - started
    text = str(result.get("text") or "")
    print(f"[{label}] ok={result.get('ok')} elapsed={elapsed:.1f}s")
    print(text[:1500])
    return result


def project_html(project):
    path = ps.project_dir(project, root=STORE.workspace.root)
    if path is None or not path.is_dir():
        return []
    return sorted(path.rglob("*.html"))


def canonical_chart_name(label, station, event):
    """The deterministic filename the code must produce for one matrix cell.

    A/B use the H/D/I component matrix; C is the ground-vs-OMNI chart.  Both
    sides of A/B resolve to the same name whether the matrix used ``raw:`` or
    ``derived:``, so this checks the naming contract, not an implementation
    detail of the cell.
    """
    if label == "C":
        return cn.space_weather_filename(
            f"raw:{station.lower()}:{event}",
            "sw:omni:2024-05-01",
            "H",
            ["Bz", "V"],
        )
    return cn.components_filename(
        f"raw:{station.lower()}:{event}", ["H", "D", "I"]
    )


def check_chart_cells(label, project, stations, events, expected_name, expected_source):
    """Each requested station/date cell must own exactly one canonical chart.

    Uniqueness is the full project-relative path, not the basename: counting
    basenames rejects a perfectly good matrix tree where 200 cells each contain
    ``irt/2024-05-14/...html`` and so on.  The basename itself is checked
    against the deterministic naming function, including the source suffix.
    """
    problems = []
    root = ps.project_dir(project, root=STORE.workspace.root)
    if root is None or not root.is_dir():
        return [f"matrix {label} project directory is missing"]

    expected_paths = set()
    for station in stations:
        for event in events:
            cell = root / station.lower() / event
            charts = sorted(cell.glob("*.html")) if cell.is_dir() else []
            if len(charts) != 1:
                problems.append(
                    f"matrix {label} cell {station}/{event} has {len(charts)} HTML charts"
                )
                expected_paths.update(chart.resolve() for chart in charts)
                continue
            chart = charts[0]
            expected = expected_name(station, event)
            if chart.name != expected:
                problems.append(
                    f"matrix {label} cell {station}/{event}: "
                    f"{chart.name!r} != canonical {expected!r}"
                )
            if cn.source_from_filename(chart.name) != expected_source:
                problems.append(
                    f"matrix {label} cell {station}/{event}: "
                    f"source suffix {cn.source_from_filename(chart.name)!r} "
                    f"!= {expected_source!r}"
                )
            if chart.stat().st_size == 0:
                problems.append(
                    f"matrix {label} cell {station}/{event} has an empty chart"
                )
            expected_paths.add(chart.resolve())

    extras = [
        path
        for path in root.rglob("*.html")
        if path.resolve() not in expected_paths
    ]
    if extras:
        problems.append(
            f"matrix {label} has {len(extras)} HTML charts outside the requested cells"
        )
    return problems


def check_chart_metadata(
    label,
    project,
    stations,
    events,
    expected_name,
    expected_source,
    components,
    kind,
    product,
):
    """``charts.json`` must describe every cell, not just the file on disk."""
    problems = []
    root = ps.project_dir(project, root=STORE.workspace.root)
    if root is None or not root.is_dir():
        return [f"matrix {label} project directory is missing"]

    rows = ps.read_chart_index(project, root=STORE.workspace.root)
    by_key = {
        (str(row.get("relative_path") or ""), str(row.get("file") or "")): row
        for row in rows
    }
    expected_keys = set()
    expected_sources = expected_source.split("-")
    for station in stations:
        for event in events:
            filename = expected_name(station, event)
            relative = f"{station.lower()}/{event}/{filename}"
            key = (relative, filename)
            expected_keys.add(key)
            row = by_key.get(key)
            if row is None:
                problems.append(f"matrix {label} charts.json is missing {relative}")
                continue
            checks = (
                ("file", row.get("file"), filename),
                ("source", row.get("source"), expected_source),
                ("sources", list(row.get("sources") or []), expected_sources),
                ("station", row.get("station"), station),
                ("date", row.get("date"), event),
                ("components", list(row.get("components") or []), components),
                ("kind", row.get("kind"), kind),
                ("product", row.get("product"), product),
            )
            for field, got, want in checks:
                if got != want:
                    problems.append(
                        f"matrix {label} charts.json {relative}: "
                        f"{field}={got!r} != {want!r}"
                    )
            handles = list(row.get("handles") or [])
            if label == "C":
                want_handles = [
                    f"raw:{station.lower()}:{event}",
                    "sw:omni:2024-05-01",
                ]
                if handles != want_handles:
                    problems.append(
                        f"matrix {label} charts.json {relative}: "
                        f"handles={handles!r} != {want_handles!r}"
                    )
            else:
                allowed = {
                    f"raw:{station.lower()}:{event}",
                    f"derived:{station.lower()}:{event}",
                }
                if len(handles) != 1 or handles[0] not in allowed:
                    problems.append(
                        f"matrix {label} charts.json {relative}: "
                        f"handles={handles!r} do not name the cell"
                    )
            if len(str(row.get("sha256") or "")) != 64:
                problems.append(
                    f"matrix {label} charts.json {relative}: sha256 is missing"
                )
            if int(row.get("bytes") or 0) <= 0:
                problems.append(
                    f"matrix {label} charts.json {relative}: bytes is not positive"
                )

    extras = set(by_key) - expected_keys
    if extras:
        problems.append(
            f"matrix {label} charts.json has {len(extras)} unexpected rows"
        )
    return problems


def _archive_path(result):
    export = result.get("export") if isinstance(result.get("export"), dict) else {}
    archive = result.get("archive") or export.get("path") or export.get("archive")
    return Path(str(archive)) if archive else None


def check_default_archive(label, project, result, stations, events, expected_name):
    """The normal matrix export must keep its default name and canonical tree."""
    problems = []
    archive = _archive_path(result)
    if archive is None or not archive.is_file():
        return []  # check_matrix already reports a missing archive
    target = ps.project_dir(project, root=STORE.workspace.root)
    expected_archive = f"{target.name}.zip" if target is not None else ""
    if expected_archive and archive.name != expected_archive:
        problems.append(
            f"matrix {label} default ZIP name {archive.name!r} != {expected_archive!r}"
        )

    try:
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            expected_members = set()
            for station in stations:
                for event in events:
                    expected_members.add(
                        f"{station.lower()}/{event}/{expected_name(station, event)}"
                    )
            charts = [
                name
                for name in names
                if name.lower().endswith(".html") and name != "index.html"
            ]
            missing = expected_members - set(charts)
            extra = set(charts) - expected_members
            if missing:
                problems.append(
                    f"matrix {label} default ZIP is missing {len(missing)} canonical charts"
                )
            if extra:
                problems.append(
                    f"matrix {label} default ZIP has {len(extra)} charts outside "
                    "the canonical tree"
                )
            for member in charts:
                if len(member.split("/")) != 3:
                    problems.append(
                        f"matrix {label} default ZIP has a non-canonical path {member!r}"
                    )
                    break
                if bundle.getinfo(member).file_size == 0:
                    problems.append(
                        f"matrix {label} default ZIP has an empty chart {member!r}"
                    )
                    break
            for meta in (
                "manifest.json",
                "charts.json",
                "index.csv",
                "index.html",
                "export_manifest.json",
            ):
                if meta not in names:
                    problems.append(
                        f"matrix {label} default ZIP is missing {meta}"
                    )
            try:
                rows = list(
                    csv.DictReader(
                        io.StringIO(bundle.read("index.csv").decode("utf-8"))
                    )
                )
            except (KeyError, ValueError, UnicodeDecodeError) as exc:
                problems.append(
                    f"matrix {label} default index.csv is unreadable: {exc}"
                )
                rows = []
            if len(rows) != len(expected_members):
                problems.append(
                    f"matrix {label} default index.csv has {len(rows)} rows, "
                    f"expected {len(expected_members)}"
                )
            if rows:
                pairs = [(row.get("station"), row.get("date")) for row in rows]
                if pairs != sorted(pairs):
                    problems.append(
                        f"matrix {label} default index.csv is not sorted by "
                        "station, date"
                    )
                for row in rows:
                    station = str(row.get("station") or "")
                    event = str(row.get("date") or "")
                    if station not in stations or event not in events:
                        problems.append(
                            f"matrix {label} default index.csv has unknown row "
                            f"{station}/{event}"
                        )
                        continue
                    if row.get("file") != expected_name(station, event):
                        problems.append(
                            f"matrix {label} default index.csv file "
                            f"{row.get('file')!r} is not canonical"
                        )
            try:
                manifest = json.loads(
                    bundle.read("export_manifest.json").decode("utf-8")
                )
            except (KeyError, ValueError) as exc:
                problems.append(
                    f"matrix {label} export_manifest.json is unreadable: {exc}"
                )
            else:
                if expected_archive and manifest.get("archive") != expected_archive:
                    problems.append(
                        f"matrix {label} export_manifest archive "
                        f"{manifest.get('archive')!r} != {expected_archive!r}"
                    )
                if manifest.get("folders") != {"layout": "canonical"}:
                    problems.append(
                        f"matrix {label} default export unexpectedly has custom folders"
                    )
                if manifest.get("sort_by") != [
                    "station",
                    "start_date",
                    "components",
                    "source",
                    "file",
                ]:
                    problems.append(
                        f"matrix {label} default export sort_by is not the default"
                    )
                if manifest.get("counts", {}).get("charts") != len(expected_members):
                    problems.append(
                        f"matrix {label} default export chart count is wrong"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        problems.append(f"matrix {label} default ZIP is unreadable: {exc}")
    return problems


def matrix_prompt(project, events, stations):
    return (
        "Создай ровно ОДНУ подзадачу create_task_matrix и ничего больше.\n"
        "Не перечисляй ячейки и не добавляй process_matrix_batch: агент сам "
        "выполнит один process_matrix_batch(run_until_complete=True) и экспорт.\n"
        f"events = {json.dumps(events, ensure_ascii=False)}\n"
        f"stations = {json.dumps(stations, ensure_ascii=False)}\n"
        'actions = ["fetch_observatory_data", "calculate_derived_components", '
        '"plot_components"]\n'
        f'project_name = "{project}"\n'
    )


def comparison_matrix_prompt(project, events, stations, solar_wind):
    return (
        "Создай ровно ОДНУ подзадачу create_task_matrix и ничего больше.\n"
        "Не перечисляй ячейки и не добавляй process_matrix_batch: агент сам "
        "выполнит один process_matrix_batch(run_until_complete=True) и экспорт.\n"
        f"events = {json.dumps(events, ensure_ascii=False)}\n"
        f"stations = {json.dumps(stations, ensure_ascii=False)}\n"
        'actions = ["fetch_observatory_data", "plot_space_weather_comparison"]\n'
        "shared_args = "
        + json.dumps(
            {
                "solar_wind_df_descriptor": solar_wind,
                "ground_component": "H",
                "solar_wind_components": ["Bz", "V"],
            },
            ensure_ascii=False,
        )
        + "\n"
        f'project_name = "{project}"\n'
    )


def custom_export_prompt(project):
    return (
        "Выполни ровно ОДИН вызов export_project и ничего больше.\n"
        'В плане должна быть ровно одна подзадача с tool_name="export_project".\n'
        "Не создавай графики, не вызывай process_matrix_batch и не меняй проект.\n"
        f"name = {json.dumps(project)}\n"
        f"archive_name = {json.dumps(CUSTOM_ARCHIVE)}\n"
        f"folders = {json.dumps(CUSTOM_FOLDERS, ensure_ascii=False)}\n"
        f"sort_by = {json.dumps(CUSTOM_SORT)}\n"
        f"source_order = {json.dumps(CUSTOM_SOURCE_ORDER)}\n"
        "Папки 1/2/3 — это имена верхнего уровня в ZIP. Файлы внутри должны "
        "сохранить канонические имена, проект на диске менять нельзя. "
        "Верни путь архива."
    )


def check_custom_export(result, project, stations, events, before_paths):
    """Prompt-named folders must be an export view, not a project mutation."""
    problems = []
    if not result.get("ok"):
        problems.append("custom export stage not ok")

    all_calls = list(result.get("tool_calls") or [])
    unexpected_calls = [
        call.get("tool") for call in all_calls if call.get("tool") != "export_project"
    ]
    if unexpected_calls:
        problems.append(
            f"custom export stage made unexpected calls: {unexpected_calls!r}"
        )
    calls = [
        call for call in all_calls if call.get("tool") == "export_project"
    ]
    if len(calls) != 1:
        problems.append(
            f"custom export stage made {len(calls)} export_project calls, expected 1"
        )
    else:
        call = calls[0]
        if not call.get("ok"):
            problems.append("custom export_project call failed")
        args = call.get("arguments") or {}
        target = ps.project_dir(project, root=STORE.workspace.root)
        requested = str(args.get("name") or "")
        resolved = (
            ps.project_dir(requested, root=STORE.workspace.root)
            if requested
            else None
        )
        if (
            target is None
            or resolved is None
            or resolved.name != target.name
        ):
            problems.append(
                f"custom export used project {args.get('name')!r}, expected {project!r}"
            )
        archive_arg = str(args.get("archive_name") or "").lower()
        if archive_arg not in (CUSTOM_ARCHIVE, f"{CUSTOM_ARCHIVE}.zip"):
            problems.append(
                f"custom export archive_name={archive_arg!r}, "
                f"expected {CUSTOM_ARCHIVE!r}"
            )
        folders_arg = args.get("folders")
        if not isinstance(folders_arg, dict) or set(folders_arg) != set(CUSTOM_FOLDERS):
            problems.append(
                f"custom export folders={folders_arg!r}, "
                f"expected keys {sorted(CUSTOM_FOLDERS)}"
            )
        if ps._normalise_sort_by(args.get("sort_by")) != ps._normalise_sort_by(
            CUSTOM_SORT
        ):
            problems.append(
                f"custom export sort_by={args.get('sort_by')!r}, "
                f"expected {CUSTOM_SORT!r}"
            )
        source_order = args.get("source_order") or []
        if isinstance(source_order, str):
            source_order = source_order.replace(",", " ").split()
        got_order = [str(source).strip().upper() for source in source_order]
        if got_order != CUSTOM_SOURCE_ORDER:
            problems.append(
                f"custom export source_order={got_order!r}, "
                f"expected {CUSTOM_SOURCE_ORDER!r}"
            )

    archive = RUN_EXPORTS / f"{CUSTOM_ARCHIVE}.zip"
    if not archive.is_file():
        problems.append(f"custom archive is missing: {archive}")
        return problems
    if archive.stat().st_size == 0:
        problems.append("custom archive is empty")
        return problems

    station_folder = {
        station.upper(): folder
        for folder, patterns in CUSTOM_FOLDERS.items()
        for station in patterns
    }
    expected_members = set()
    for station in stations:
        for event in events:
            expected_members.add(
                f"{station_folder[station]}/{station.lower()}/{event}/"
                f"{canonical_chart_name('C', station, event)}"
            )

    try:
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            top_level = {name.split("/", 1)[0] for name in names if "/" in name}
            if top_level != set(CUSTOM_FOLDERS):
                problems.append(
                    f"custom archive top-level folders {sorted(top_level)!r} "
                    f"!= {sorted(CUSTOM_FOLDERS)!r}"
                )
            for meta in (
                "manifest.json",
                "charts.json",
                "index.csv",
                "index.html",
                "export_manifest.json",
            ):
                if meta not in names:
                    problems.append(f"custom archive is missing {meta}")

            charts = [
                name
                for name in names
                if name.lower().endswith(".html") and name != "index.html"
            ]
            missing = expected_members - set(charts)
            extra = set(charts) - expected_members
            if missing:
                problems.append(
                    f"custom archive is missing {len(missing)} canonical charts"
                )
            if extra:
                problems.append(
                    f"custom archive has {len(extra)} charts outside the requested folders"
                )
            basenames = [Path(name).name for name in charts]
            if len(set(basenames)) != len(basenames):
                problems.append("custom archive contains duplicate chart basenames")
            for member in charts:
                if bundle.getinfo(member).file_size == 0:
                    problems.append(f"custom archive has an empty chart {member!r}")
                    break

            try:
                rows = list(
                    csv.DictReader(
                        io.StringIO(bundle.read("index.csv").decode("utf-8"))
                    )
                )
            except (KeyError, ValueError, UnicodeDecodeError) as exc:
                problems.append(f"custom archive index.csv is unreadable: {exc}")
                rows = []
            if len(rows) != len(expected_members):
                problems.append(
                    f"custom archive index.csv has {len(rows)} rows, "
                    f"expected {len(expected_members)}"
                )
            if rows:
                orders = [int(row.get("order") or 0) for row in rows]
                if orders != list(range(1, len(rows) + 1)):
                    problems.append("custom archive index.csv order is not sequential")
                pairs = [(row.get("date"), row.get("station")) for row in rows]
                wanted_pairs = sorted(
                    (event, station) for station in stations for event in events
                )
                if pairs != wanted_pairs:
                    problems.append(
                        "custom archive index.csv is not sorted by date, station"
                    )
                for row in rows:
                    station = str(row.get("station") or "")
                    event = str(row.get("date") or "")
                    if station not in station_folder or event not in events:
                        problems.append(
                            f"custom archive index.csv has unknown row {station}/{event}"
                        )
                        continue
                    filename = canonical_chart_name("C", station, event)
                    wanted = (
                        f"{station_folder[station]}/{station.lower()}/{event}/{filename}"
                    )
                    if row.get("file") != filename:
                        problems.append(
                            f"custom archive index.csv file {row.get('file')!r} "
                            f"!= {filename!r}"
                        )
                    if row.get("archive_path") != wanted:
                        problems.append(
                            f"custom archive index.csv path {row.get('archive_path')!r} "
                            f"!= {wanted!r}"
                        )
                    if row.get("source") != "INTERMAGNET-OMNI":
                        problems.append(
                            "custom archive index.csv source is not INTERMAGNET-OMNI"
                        )
                    if row.get("components") != "H-Bz-V":
                        problems.append(
                            "custom archive index.csv components are not H-Bz-V"
                        )
                    if row.get("kind") != "space_weather":
                        problems.append(
                            "custom archive index.csv kind is not space_weather"
                        )
                    if row.get("product") != "space_weather":
                        problems.append(
                            "custom archive index.csv product is not space_weather"
                        )

            try:
                index_html = bundle.read("index.html").decode("utf-8")
            except (KeyError, UnicodeDecodeError) as exc:
                problems.append(f"custom archive index.html is unreadable: {exc}")
            else:
                for member in expected_members:
                    if Path(member).name not in index_html:
                        problems.append(
                            f"custom archive index.html does not list {Path(member).name}"
                        )
                        break
                if "INTERMAGNET-OMNI" not in index_html:
                    problems.append(
                        "custom archive index.html does not show INTERMAGNET-OMNI"
                    )

            try:
                manifest = json.loads(
                    bundle.read("export_manifest.json").decode("utf-8")
                )
            except (KeyError, ValueError) as exc:
                problems.append(
                    f"custom archive export_manifest.json is unreadable: {exc}"
                )
            else:
                if manifest.get("archive") != f"{CUSTOM_ARCHIVE}.zip":
                    problems.append(
                        f"custom export_manifest archive {manifest.get('archive')!r} "
                        f"!= {CUSTOM_ARCHIVE + '.zip'!r}"
                    )
                if manifest.get("sort_by") != ["start_date", "station"]:
                    problems.append(
                        f"custom export_manifest sort_by {manifest.get('sort_by')!r} "
                        "is not ['start_date', 'station']"
                    )
                if manifest.get("source_order") != CUSTOM_SOURCE_ORDER:
                    problems.append(
                        "custom export_manifest source_order is wrong"
                    )
                if set(manifest.get("folders") or {}) != set(CUSTOM_FOLDERS):
                    problems.append(
                        "custom export_manifest folders do not match the prompt"
                    )
                if manifest.get("counts", {}).get("charts") != len(expected_members):
                    problems.append(
                        "custom export_manifest chart count is wrong"
                    )
                if manifest.get("counts", {}).get("files") != len(names):
                    problems.append(
                        "custom export_manifest file count does not match the ZIP"
                    )
                actual_counts = {
                    folder: sum(1 for name in names if name.startswith(f"{folder}/"))
                    for folder in CUSTOM_FOLDERS
                }
                if manifest.get("folders") != actual_counts:
                    problems.append(
                        "custom export_manifest folder counts do not match the ZIP"
                    )
                if len(manifest.get("charts") or []) != len(expected_members):
                    problems.append(
                        "custom export_manifest charts list is incomplete"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        problems.append(f"custom archive is unreadable: {exc}")

    target = ps.project_dir(project, root=STORE.workspace.root)
    if target is not None:
        for folder in CUSTOM_FOLDERS:
            if (target / folder).exists():
                problems.append(
                    f"custom export mutated the project: folder {folder!r} exists"
                )
        after_paths = project_html(project)
        if after_paths != before_paths:
            problems.append(
                "custom export changed the canonical project chart set"
            )
        if len(ps.read_chart_index(project, root=STORE.workspace.root)) != len(
            expected_members
        ):
            problems.append(
                "custom export changed the canonical project chart metadata"
            )
    return problems


def read_matrix(result):
    path = result.get("matrix_path")
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def matrix_actions(result):
    return [
        set(task.get("actions") or [])
        for task in (read_matrix(result).get("tasks") or [])
    ]


def print_matrix_report(label, result):
    """Print full / partial / empty stations and every missing cell with reason."""
    matrix = read_matrix(result)
    tasks = matrix.get("tasks") or []
    if not tasks:
        print(f"[{label}] matrix report unavailable")
        return
    print(f"[{label}] per-station coverage:")
    for task in tasks:
        station = str(task.get("station_code") or "?")
        total = len(task.get("events") or [])
        ok = len(task.get("events_ok") or [])
        failed = len(task.get("events_failed") or [])
        status = str(task.get("status") or "?")
        print(f"  {station}: {ok}/{total} ok, {failed} failed, status={status}")
        errors = task.get("event_errors") or {}
        for date in task.get("events_failed") or []:
            reason = (errors.get(date) or ["unknown"])[0]
            print(f"    {date}: {str(reason)[:240]}")


def check_matrix_shape(label, result, stations, events):
    """The matrix must be station-grouped, with exactly the requested cells."""
    problems = []
    tasks = read_matrix(result).get("tasks") or []
    if len(tasks) != len(stations):
        problems.append(
            f"matrix {label} has {len(tasks)} tasks != {len(stations)} stations"
        )
    got_stations = [str(task.get("station_code") or "") for task in tasks]
    if len(set(got_stations)) != len(got_stations):
        problems.append(f"matrix {label} has duplicate station tasks")
    if set(got_stations) != set(stations):
        problems.append(f"matrix {label} station set mismatch")
    for task in tasks:
        if set(task.get("events") or []) != set(events):
            problems.append(
                f"matrix {label} task {task.get('station_code')} date set mismatch"
            )
            break
    return problems


def check_matrix(label, result, expected):
    problems = []
    if not result.get("ok"):
        problems.append(f"matrix {label} not ok")
    if result.get("expected_cells") != expected:
        problems.append(
            f"matrix {label} expected_cells={result.get('expected_cells')} != {expected}"
        )
    if result.get("produced_cells") != expected:
        problems.append(
            f"matrix {label} produced_cells={result.get('produced_cells')} != {expected}"
        )
    if result.get("missing_cells"):
        problems.append(
            f"matrix {label} has {len(result['missing_cells'])} missing cells"
        )
    export = result.get("export")
    if not isinstance(export, dict) or ac.is_error(export):
        problems.append(f"matrix {label} export not ok")
    archive = result.get("archive")
    if not archive and isinstance(export, dict) and not ac.is_error(export):
        archive = export.get("path") or export.get("archive")
    if not archive or not Path(str(archive)).is_file():
        problems.append(f"matrix {label} ZIP archive is missing")
    elif Path(str(archive)).stat().st_size == 0:
        problems.append(f"matrix {label} ZIP archive is empty")
    return problems


# ---------------------------------------------------------------------------
# 1. Matrix A: 10 x 20 = 200 charts
# ---------------------------------------------------------------------------
rA = run_stage(
    "1/9 Matrix A: 10 stations x 20 dates = 200 charts",
    matrix_prompt(PROJECT_A, DATES_A, STATIONS),
)
htmlA = project_html(PROJECT_A)
print("HTML A:", len(htmlA))

# ---------------------------------------------------------------------------
# 2. Matrix B: 5 high-latitude stations x 10 dates = 50 charts
# ---------------------------------------------------------------------------
rB = run_stage(
    "2/9 Matrix B: 5 high-latitude stations x 10 dates = 50 charts",
    matrix_prompt(PROJECT_B, DATES_B, HIGHLAT),
)
htmlB = project_html(PROJECT_B)
print("HTML B:", len(htmlB))

# ---------------------------------------------------------------------------
# 3. OMNI solar wind + Kyoto Dst/Kp
# ---------------------------------------------------------------------------
r_context = run_stage(
    "3/9 OMNI + Kyoto",
    "Выполни ровно два вызова, без графиков и без матрицы:\n"
    '1) fetch_solar_wind(start_date="2024-05-01", end_date="2024-05-30")\n'
    '2) fetch_geomagnetic_indices(start_date="2024-05-01", end_date="2024-05-30")\n'
    "Верни точные handles.",
)

# ---------------------------------------------------------------------------
# 4. SuperMAG SME/SML/SMU
# ---------------------------------------------------------------------------
skipped = []
if os.environ.get("SUPERMAG_LOGON"):
    r_smag = run_stage(
        "4/9 SuperMAG indices",
        "Выполни ровно два вызова fetch_supermag_data для индексов SME/SML/SMU:\n"
        '1) fetch_supermag_data(start_date="2024-05-01", '
        'end_date="2024-05-15", data_type="indices")\n'
        '2) fetch_supermag_data(start_date="2024-05-16", '
        'end_date="2024-05-30", data_type="indices")\n'
        "Не объединяй их в один запрос: сервис SuperMAG принимает "
        "не больше 24 дней за вызов.",
    )
    if not r_smag.get("ok"):
        print("\nSuperMAG failure details:")
        for call in r_smag.get("tool_calls") or []:
            if call.get("ok") is False:
                print("  tool:", call.get("tool"))
                print("  error:", call.get("error"))
                print("  message:", call.get("message"))
                print("  hint:", call.get("hint"))
                if call.get("detail"):
                    print("  detail:", str(call["detail"])[:500])
else:
    r_smag = None
    skipped.append("SuperMAG: переменная SUPERMAG_LOGON не задана")
    print("\nSKIP SuperMAG: переменная SUPERMAG_LOGON не задана.")

# ---------------------------------------------------------------------------
# 5. NOAA SWPC: alerts / forecast / solar_events
# ---------------------------------------------------------------------------
r_noaa = run_stage(
    "5/9 NOAA SWPC",
    "Выполни ровно три вызова, без графиков:\n"
    '1) get_noaa_space_weather_context(query_type="alerts")\n'
    '2) get_noaa_space_weather_context(query_type="forecast")\n'
    '3) get_noaa_space_weather_context(query_type="solar_events", '
    'start_date="2024-05-01", end_date="2024-05-30")',
)

# ---------------------------------------------------------------------------
# 6. ISWA: working forecast + expected decommissioned map
# ---------------------------------------------------------------------------
r_iswa = run_stage(
    "6/9 ISWA forecast",
    "Выполни ровно один вызов: get_iswa_forecast()",
)
iswa_map, _ = ac.HANDLERS["get_iswa_perturbation_map"](
    {"timestamp": "2024-05-10T12:00:00Z"}, STORE
)
print("\nISWA map expected service_decommissioned:", iswa_map.get("error"))
print(iswa_map.get("message"))

# ---------------------------------------------------------------------------
# 7. Comparison matrix: 10 stations x 5 dates = 50 charts
# ---------------------------------------------------------------------------
rC = run_stage(
    "7/9 Comparison matrix: 10 stations x 5 dates = 50 charts",
    comparison_matrix_prompt(
        PROJECT_C, STORM_DATES, STATIONS, "sw:omni:2024-05-01"
    ),
)
htmlC = project_html(PROJECT_C)
comp_plots = list(htmlC)
print("HTML C:", len(htmlC))

# ---------------------------------------------------------------------------
# 8. Prompt-driven custom archive/folders
# ---------------------------------------------------------------------------
r_export = run_stage(
    "8/9 Custom archive/folders export",
    custom_export_prompt(PROJECT_C),
)
htmlC_after = project_html(PROJECT_C)
print("HTML C after custom export:", len(htmlC_after))

r_list = run_stage(
    "9/9 List projects",
    "Выполни ровно один вызов list_projects().",
)

# ---------------------------------------------------------------------------
# Final gate
# ---------------------------------------------------------------------------
failures = []
MATRIX_META = {
    "A": ("INTERMAGNET", ["H", "D", "I"], "components", "components"),
    "B": ("INTERMAGNET", ["H", "D", "I"], "components", "components"),
    "C": ("INTERMAGNET-OMNI", ["H", "Bz", "V"], "space_weather", "space_weather"),
}
for label, result, expected, stations, events, project in (
    ("A", rA, 200, STATIONS, DATES_A, PROJECT_A),
    ("B", rB, 50, HIGHLAT, DATES_B, PROJECT_B),
    ("C", rC, 50, STATIONS, STORM_DATES, PROJECT_C),
):
    source, components, kind, product = MATRIX_META[label]
    expected_name = lambda station, event, _label=label: canonical_chart_name(
        _label, station, event
    )
    failures.extend(check_matrix(label, result, expected))
    failures.extend(check_matrix_shape(label, result, stations, events))
    failures.extend(
        check_chart_cells(
            label, project, stations, events, expected_name, source
        )
    )
    failures.extend(
        check_chart_metadata(
            label,
            project,
            stations,
            events,
            expected_name,
            source,
            components,
            kind,
            product,
        )
    )
    failures.extend(
        check_default_archive(
            label, project, result, stations, events, expected_name
        )
    )
    print_matrix_report(label, result)

if len(htmlA) != 200:
    failures.append(f"matrix A charts: {len(htmlA)} != 200")
if len(htmlB) != 50:
    failures.append(f"matrix B charts: {len(htmlB)} != 50")
if len(htmlC) != 50:
    failures.append(f"comparison charts: {len(htmlC)} != 50")
if len(set(comp_plots)) != len(comp_plots):
    failures.append("comparison filenames are not unique")
all_paths = htmlA + htmlB + htmlC
if len(set(all_paths)) != 300:
    failures.append("total distinct chart paths != 300")
all_names = [path.name for path in all_paths]
if len(set(all_names)) != 300:
    failures.append("total distinct canonical chart basenames != 300")
unknown_sources = [
    name for name in all_names if cn.source_from_filename(name) == "UNKNOWN"
]
if unknown_sources:
    failures.append(
        f"{len(unknown_sources)} charts have no recognized source suffix: "
        f"{unknown_sources[:3]}"
    )
run_root = RUN_PROJECTS.resolve()
for path in all_paths:
    try:
        path.resolve().relative_to(run_root)
    except ValueError:
        failures.append(f"chart outside this run's project root: {path}")
        break
if any(path.stat().st_size == 0 for path in all_paths):
    failures.append("one or more HTML charts are empty")

# Provenance: a count of 50 is not enough if the matrix used plot_components
# instead of plot_space_weather_comparison, or omitted shared_args.
for task_actions in matrix_actions(rA):
    if not {
        "fetch_observatory_data",
        "calculate_derived_components",
        "plot_components",
    }.issubset(task_actions):
        failures.append("matrix A has an unexpected action set")
        break
for task_actions in matrix_actions(rB):
    if not {
        "fetch_observatory_data",
        "calculate_derived_components",
        "plot_components",
    }.issubset(task_actions):
        failures.append("matrix B has an unexpected action set")
        break
for task_actions in matrix_actions(rC):
    if "plot_space_weather_comparison" not in task_actions:
        failures.append("comparison matrix did not use plot_space_weather_comparison")
        break
try:
    matrixC = read_matrix(rC)
    if not matrixC:
        failures.append("comparison matrix JSON is missing")
    elif (
        matrixC.get("shared_args", {}).get("solar_wind_df_descriptor")
        != "sw:omni:2024-05-01"
    ):
        failures.append("comparison matrix shared_args are wrong")
    for task in matrixC.get("tasks") or []:
        if (
            task.get("shared_args", {}).get("solar_wind_df_descriptor")
            != "sw:omni:2024-05-01"
        ):
            failures.append("comparison cell shared_args are wrong")
            break
except (KeyError, OSError, ValueError):
    failures.append("comparison matrix JSON is missing")

if not r_context.get("ok"):
    failures.append("OMNI/Kyoto stage not ok")
if "sw:omni:2024-05-01" not in STORE.names():
    failures.append("missing sw:omni:2024-05-01")
if "idx:kyoto:2024-05-01" not in STORE.names():
    failures.append("missing idx:kyoto:2024-05-01")
if r_smag is not None:
    smag_calls = [
        call
        for call in r_smag.get("tool_calls") or []
        if call.get("tool") == "fetch_supermag_data"
    ]
    expected_smag = {
        ("2024-05-01", "2024-05-15"),
        ("2024-05-16", "2024-05-30"),
    }
    got_smag = {
        (
            (call.get("arguments") or {}).get("start_date"),
            (call.get("arguments") or {}).get("end_date"),
        )
        for call in smag_calls
    }
    if len(smag_calls) != 2 or got_smag != expected_smag:
        failures.append(
            "SuperMAG stage did not run the two required 15-day windows"
        )
    for slot in ("idx:supermag:2024-05-01", "idx:supermag:2024-05-16"):
        if slot not in STORE.names():
            failures.append(f"missing {slot}")
    if not r_smag.get("ok"):
        failures.append("SuperMAG stage not ok")
if not r_noaa.get("ok"):
    failures.append("NOAA stage not ok")
if not r_iswa.get("ok"):
    failures.append("ISWA forecast stage not ok")
if iswa_map.get("error") != "service_decommissioned":
    failures.append(f"ISWA map: {iswa_map.get('error')} (expected service_decommissioned)")
if not r_list.get("ok"):
    failures.append("list_projects stage not ok")
if htmlC_after != htmlC:
    failures.append("custom export stage changed the canonical C chart set")
failures.extend(
    check_custom_export(r_export, PROJECT_C, STATIONS, STORM_DATES, htmlC)
)

exports_root = RUN_EXPORTS.resolve()
for label, result in (("A", rA), ("B", rB), ("C", rC)):
    export = result.get("export") if isinstance(result.get("export"), dict) else {}
    archive = result.get("archive") or export.get("path") or export.get("archive")
    if not archive:
        continue  # check_matrix already reported the missing ZIP
    try:
        Path(str(archive)).resolve().relative_to(exports_root)
    except ValueError:
        failures.append(f"matrix {label} ZIP outside this run's export root: {archive}")

print("\n" + "=" * 72)
print("STRESS TEST SUMMARY")
print("=" * 72)
print("run dir:", RUN_DIR)
print("projects:", ", ".join((PROJECT_A, PROJECT_B, PROJECT_C)))
print("matrix A:", len(htmlA), "charts;", rA.get("coverage"))
print("matrix B:", len(htmlB), "charts;", rB.get("coverage"))
print("comparisons:", len(htmlC), "charts;", rC.get("coverage"))
print("total:", len(htmlA) + len(htmlB) + len(htmlC))
for label, result in (("A", rA), ("B", rB), ("C", rC)):
    export = result.get("export") if isinstance(result.get("export"), dict) else {}
    archive = result.get("archive") or export.get("path") or export.get("archive")
    print(f"ZIP {label}:", archive or "missing")
print(
    "custom ZIP:",
    RUN_EXPORTS / f"{CUSTOM_ARCHIVE}.zip",
    "folders:", sorted(CUSTOM_FOLDERS),
)
print("skipped:", skipped or "none")
print(
    "coverage note: BOX excluded (no May 2024 data); ABG replaces BOX; "
    "window starts 14 May because ABG has no 13 May day and ends 2 June "
    "to give all ten stations twenty complete days."
)
print("expected service notes:")
if r_smag is None:
    print("  - SuperMAG: skipped because SUPERMAG_LOGON is not set")
else:
    print("  - SuperMAG: exercised with SUPERMAG_LOGON")
print("  - NOAA solar_events: 0 records for May 2024 (7-day current window only)")
print("  - NOAA alerts/forecast: current window only, not historical May 2024")
print("  - ISWA perturbation map: service_decommissioned by NASA")
print("failures:", failures or "none")
# Do not leak the run-local export root into later notebook work.
ps.EXPORTS_ROOT = _ORIGINAL_EXPORTS_ROOT
if failures:
    raise AssertionError(failures)
source_note = (
    "6 sources exercised"
    if r_smag is not None
    else "5 sources exercised; SuperMAG skipped"
)
print(
    f"STRESS TEST PASSED: 300 distinct canonical charts, {source_note}, "
    "3 matrix runs, 3 default ZIPs, 1 prompt-named custom ZIP"
)
