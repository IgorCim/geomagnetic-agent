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
# ============================================================
import json
import os
import sys
import tempfile
import time
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


def check_chart_cells(label, project, stations, events):
    """Each requested station/date cell must own exactly one non-empty chart.

    The chart file is deliberately called ``chart.html`` inside every cell.
    Uniqueness is the full project-relative path, not the basename: counting
    basenames rejects a perfectly good matrix tree where 200 cells each contain
    ``irt/2024-05-14/chart.html`` and so on.
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
    "1/7 Matrix A: 10 stations x 20 dates = 200 charts",
    matrix_prompt(PROJECT_A, DATES_A, STATIONS),
)
htmlA = project_html(PROJECT_A)
print("HTML A:", len(htmlA))

# ---------------------------------------------------------------------------
# 2. Matrix B: 5 high-latitude stations x 10 dates = 50 charts
# ---------------------------------------------------------------------------
rB = run_stage(
    "2/7 Matrix B: 5 high-latitude stations x 10 dates = 50 charts",
    matrix_prompt(PROJECT_B, DATES_B, HIGHLAT),
)
htmlB = project_html(PROJECT_B)
print("HTML B:", len(htmlB))

# ---------------------------------------------------------------------------
# 3. OMNI solar wind + Kyoto Dst/Kp
# ---------------------------------------------------------------------------
r_context = run_stage(
    "3/7 OMNI + Kyoto",
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
        "4/7 SuperMAG indices",
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
    "5/7 NOAA SWPC",
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
    "6/7 ISWA forecast",
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
    "7/7 Comparison matrix: 10 stations x 5 dates = 50 charts",
    comparison_matrix_prompt(
        PROJECT_C, STORM_DATES, STATIONS, "sw:omni:2024-05-01"
    ),
)
htmlC = project_html(PROJECT_C)
comp_plots = list(htmlC)
print("HTML C:", len(htmlC))

r_list = run_stage(
    "7/7 List projects",
    "Выполни ровно один вызов list_projects().",
)

# ---------------------------------------------------------------------------
# Final gate
# ---------------------------------------------------------------------------
failures = []
for label, result, expected, stations, events, project in (
    ("A", rA, 200, STATIONS, DATES_A, PROJECT_A),
    ("B", rB, 50, HIGHLAT, DATES_B, PROJECT_B),
    ("C", rC, 50, STATIONS, STORM_DATES, PROJECT_C),
):
    failures.extend(check_matrix(label, result, expected))
    failures.extend(check_matrix_shape(label, result, stations, events))
    failures.extend(check_chart_cells(label, project, stations, events))
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
print(f"STRESS TEST PASSED: 300 distinct charts, {source_note}, 3 matrix runs")
