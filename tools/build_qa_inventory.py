"""Read-only QA inventory and prompt templates; never executes an agent task.

Only schemas/introspection and AST parsing are used. Output is a test-design
artifact, not evidence of successful tool, model, supplier or scientific tests.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Dependencies identify earlier cases, not prose that should be pasted to Qwen.
# Runtime variables must be filled by the test runner from actual results.
SEEDS = {
    "fetch_observatory_data": ("Загрузи INTERMAGNET IRT за 2024-09-10. Укажи точный handle, источник, число строк и фактический тип публикации.", ["TOOL-02"], "источник и UTC; не выдумывать definitive"),
    "create_project": ("Создай проект {PROJECT}. Укажи фактическое имя каталога и путь внутри корня проектов.", [], "каталог и manifest существуют; допустимая нормализация отражена"),
    "fetch_many": ("В проект {PROJECT} загрузи IRT и BOU за 2024-09-10 одним пакетным запросом, max_downloads=2. Перечисли фактические таблицы и ошибки.", ["TOOL-02"], "два station/day либо точный partial; отдельные handle"),
    "list_projects": ("Перечисли существующие проекты через list_projects. Не создавай новые проекты.", ["TOOL-02"], "список совпадает с разрешёнными каталогами на диске"),
    "export_project": ("Экспортируй проект {PROJECT} в архив qa_report.zip. Внутри создай папку Stations для irt и bou, сортировка date,station. Укажи фактический путь архива.", ["TOOL-03", "TOOL-14"], "реальный ZIP, безопасные члены, manifest, индекс, хеши и все ожидаемые файлы"),
    "create_task_matrix": ("Создай одну матрицу IRT и BOU × 2024-09-10 и 2024-09-11, действия fetch_observatory_data, calculate_derived_components, plot_components, проект {MATRIX_PROJECT}. Верни путь матрицы.", [], "кодовый контракт создания отдельно; Gradio может запустить весь матричный workflow"),
    "process_matrix_batch": ("Продолжи обработку существующей матрицы {MATRIX_PATH}, run_until_complete=True. Матрицу заново не создавай. Укажи покрытие и архив.", ["TOOL-06"], "четыре ячейки; resume/idempotency; контролируемый partial без stale artifacts"),
    "calculate_derived_components": ("Для {RAW_IRT_D0} вычисли H/D/I, без повторной загрузки. Укажи новый handle и единицы компонентов.", ["TOOL-01"], "построчный независимый H/D/I, неизменность исходной таблицы"),
    "get_statistics": ("Получи статистику Y для BOU за 2024-09-10, без повторной загрузки. Покажи медиану, count и пропуски.", ["TOOL-03"], "один get_statistics в прямом пути; компонента Y; числа и ddof сверены независимо"),
    "detect_anomalies": ("Для {DERIVED_IRT_D0} найди статистические выбросы H с sigma_threshold=3. Укажи порог и число; не объявляй их автоматически бурями.", ["TOOL-08"], "правильная маска и краевые случаи постоянного/пустого ряда"),
    "calculate_derived_math": ("Для {DERIVED_IRT_D0} вычисли размах H через calculate_derived_math, metric=delta, без новой загрузки. Укажи окно и единицы.", ["TOOL-08"], "каждый режим delta/anomaly/dH_dt отдельно; численные эталоны и шаг времени"),
    "calculate_baseline": ("Для {DERIVED_IRT_D0} вычисли baseline H в UTC-окне 00–04, mode=night, night_hours=[0,4]. Назови выбранные строки и единицы.", ["TOOL-08"], "night/full; границы и окно через полночь; UTC не объявлен местной ночью"),
    "evaluate_custom_formula": ("Для {RAW_IRT_D0} вычисли безопасную формулу hypot(X,Y). Укажи фактический новый handle. Исходные данные не меняй.", ["TOOL-01"], "независимая численная проверка; отдельная серия разрешённых и запрещённых AST"),
    "plot_components": ("Построй H из {DERIVED_IRT_D0} без новой загрузки. Имя файла выбери автоматически. Укажи реально созданный файл.", ["TOOL-08"], "traces=значения выбранной таблицы; UTC, единицы, пропуски, фактическое имя"),
    "plot_comparison": ("Сравни H IRT за 2024-09-10 и 2024-09-11 на одном графике, используй {DERIVED_IRT_D0} и {DERIVED_IRT_D1} без новой загрузки. Нужны две подписанные кривые.", ["TOOL-08", "SETUP-IRT-D1"], "два разных дня/источника данных в traces, не две подписи одной кривой"),
    "get_station_geomagnetic_coords": ("Покажи геомагнитные координаты станции IRT через get_station_geomagnetic_coords и назови использованное приближение.", [], "сверка координат/модели; отсутствие утверждения IGRF/AACGM"),
    "calculate_mlt": ("Вычисли магнитное местное время MLT для IRT в 2024-09-10T12:00:00Z. Укажи модель и нормализацию суток.", ["TOOL-16"], "0/24, округление минут, референсный расчёт для той же модели"),
    "group_stations_by_mlt": ("Сгруппируй IRT, BOU и ABG по MLT на 2024-09-10T12:00:00Z, bin_hours=3. Укажи все группы и модель.", [], "каждая доступная станция ровно один раз; границы bins и неизвестная станция"),
    "calculate_local_time": ("Вычисли местное солнечное время IRT в 2024-09-10T23:30:00Z по долготе. Не подменяй его гражданским часовым поясом.", ["TOOL-16"], "UTC+longitude/15, корректный переход даты"),
    "apply_physics_formula": ("Вычисли apply_physics_formula metric={METRIC} для {GROUND_HANDLE}, используя {WIND_HANDLE} и {REFERENCE_HANDLE}, только когда они нужны этой метрике. Назови единицы и границы применимости.", ["SCIENCE-GATE", "METRIC-SPECIFIC-INPUTS"], "отдельная карточка на каждую из 11 метрик, независимый oracle; Ey сейчас известный FAIL"),
    "calculate_eej_index": ("Для научно проверенной пары {EEJ_EQUATORIAL_STATION} и {EEJ_REFERENCE_STATION} получи EEJ за {EEJ_DAY}. Назови пару, время, baseline и ограничения метода.", ["SETUP-EEJ-PAIR", "SCIENCE-GATE"], "пара заранее обоснована; timestamps совмещены; IRT не используется как ложный экваториальный пример"),
    "plot_overlay": ("Наложи H IRT и BOU за 2024-09-10, time_system=UT, offsets IRT=0 и BOU=100. Используй имеющиеся таблицы; объясни, что смещение только для отображения.", ["TOOL-03"], "UT/LT/MLT отдельно; offset не меняет таблицы/статистику"),
    "plot_space_weather_comparison": ("Сопоставь H из {DERIVED_IRT_D0} с Bz и V из {WIND_D0}, без новой загрузки. Построй сравнение земли и солнечного ветра и укажи время и единицы панелей.", ["TOOL-08", "TOOL-24"], "совпадение дат, правильные каналы, alignment/gaps и панели"),
    "fetch_solar_wind": ("Загрузи солнечный ветер OMNI за 2024-09-10. Укажи dataset, handle, число строк, координаты IMF и единицы.", [], "метаданные/fill, GSM/GSE, источник, фактическое временное покрытие"),
    "fetch_geomagnetic_indices": ("Получи реальные индексы Kyoto Dst/Kp за 2024-09-10. Укажи источник, handle, время и исходное разрешение Kp. Не рассчитывай dst_proxy.", [], "официальный ряд не заменён proxy; расширение Kp не названо новыми измерениями"),
    "get_noaa_space_weather_context": ("Получи текущие NOAA alerts. Отдели наблюдение, предупреждение и прогноз; укажи фактическое окно данных.", [], "три режима alerts/forecast/solar_events; архивное окно отдельно от текущих продуктов"),
    "fetch_supermag_data": ("Запроси SuperMAG indices SME/SML/SMU за 2024-09-10. При отказе покажи реальную причину, не заменяй их Kyoto и не раскрывай логин.", ["SETUP-SUPERMAG"], "sites/indices, допустимая длина окна; missing logon отдельно от HTTP; integration может быть BLOCKED_EXTERNAL"),
    "get_iswa_perturbation_map": ("Запроси карту ISWA magnetic perturbation на 2024-09-10T12:00:00Z. Если текущий клиент её не поддерживает, честно сообщи отказ без выдуманного изображения.", [], "ветка сейчас отключена; контракт ошибки может PASS, реальная карта остаётся BLOCKED/UNSUPPORTED"),
    "get_iswa_forecast": ("Получи текущий прогноз ISWA. Отдели прогноз Kp от наблюдений Dst, укажи источники и горизонт времени.", [], "проверить оба фактических HAPI-набора, kp_kind, частичный отказ и отсутствие ложного Dst-прогноза"),
}


def main():
    import agent_core as ac
    import geomag_math as gm

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sources = [p for p in ROOT.glob("*.py") if not p.name.startswith("test_")]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    schemas = {t["function"]["name"]: t["function"] for t in ac.TOOL_SCHEMAS}
    assert len(schemas) == len(ac.TOOL_SCHEMAS) == 29
    assert set(schemas) == set(ac.HANDLERS) == set(SEEDS)
    tools, cases = [], []
    for i, (name, schema) in enumerate(schemas.items(), 1):
        handler = inspect.unwrap(ac.HANDLERS[name])
        path = Path(inspect.getsourcefile(handler)).resolve()
        params = schema.get("parameters", {})
        tools.append({"id": f"TOOL-{i:02d}", "name": name,
                      "handler": handler.__name__, "handler_file": path.relative_to(ROOT).as_posix(),
                      "handler_line": inspect.getsourcelines(handler)[1],
                      "required": params.get("required", []), "parameters": params.get("properties", {})})
        prompt, deps, check = SEEDS[name]
        selected = {s["function"]["name"] for s in ac.select_tools_for_query(prompt)}
        cases.append({"id": f"TOOL-{i:02d}", "target_tool": name, "dependencies": deps,
                      "prompt_template": prompt, "result_check": check, "status": "PLANNED_NOT_EXECUTED",
                      "standalone_router_offers_target_in_template_dry_run": name in selected,
                      "template_router_tools": sorted(selected),
                      "router_note": "Dry-run only, not live evidence. Workflow/planner and substituted templates need separate validation."})
    modules = []
    for path in sorted(sources):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        functions = [{"name": node.name, "line": node.lineno, "private": node.name.startswith("_")}
                     for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        modules.append({"file": path.name, "definitions": functions})
    physics_router_variants = []
    for metric in ac._PHYSICS_METRICS:
        query = SEEDS['apply_physics_formula'][0].replace('{METRIC}', metric)
        selected = sorted(s['function']['name'] for s in ac.select_tools_for_query(query))
        physics_router_variants.append({'metric': metric, 'template': query,
                                        'offered_tools': selected,
                                        'status': 'DRY_RUN_ONLY_REVIEW_REQUIRED'})
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Read-only test-design inventory. Does not execute tools, Qwen, external services, plots or qualification tests.",
        "tool_count": len(tools), "get_statistics_included": True,
        "not_additional_tools": ["run_bound_statistics", "run_agent", "run_agent_with_planner", "Planner", "LLMPlanner", "parse_tool_calls"],
        "tools": tools, "physics_registry": list(ac._PHYSICS_METRICS),
        "physics_aliases": ac._PHYSICS_METRIC_ALIASES,
        "physics_router_variants": physics_router_variants,
        "dsl_functions": gm.formula_functions(), "modules": modules, "source_sha256": before,
    }
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output = ROOT / "docs" / "qa"
    output.mkdir(parents=True, exist_ok=True)
    (output / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "tool_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Каталог первичных случаев для 29 инструментов", "",
             "Статус всех случаев: PLANNED_NOT_EXECUTED. Это план, не успешный прогон.", "",
             "Шаблоны с {PROJECT}, {MATRIX_PATH} и другими переменными нельзя отправлять как есть. Перед живым запуском тестовый runner должен подставить уникальные имена и реальные handle из выполненных зависимостей. После этого пользователю выдаются отдельные готовые сообщения, без указаний проверяющему внутри prompt.", "",
             "Это первичные карточки, а не полное покрытие: режимы, негативные и граничные случаи раскрыты в QA_MASTER_PLAN.md. Router dry-run — структурная проверка шаблона без Qwen; она не подтверждает исполнение или поведение planner.", ""]
    for case in cases:
        lines.extend([f"## {case['id']}: {case['target_tool']}", "",
                      "Зависимости: " + (", ".join(case["dependencies"]) or "SETUP-00") + ".", "",
                      "```text", case["prompt_template"], "```", "",
                      "Проверяет runner: " + case["result_check"] + ".", "",
                      "Target доступен при standalone dry-run шаблона: " + str(case["standalone_router_offers_target_in_template_dry_run"]) + ".", ""])
    (output / "TOOL_RUNBOOK.md").write_text("\n".join(lines), encoding="utf-8")
    missing = [c["target_tool"] for c in cases if not c["standalone_router_offers_target_in_template_dry_run"]]
    print(json.dumps({"tools": len(tools), "physics_metrics": len(report["physics_registry"]),
                      "dsl_functions": len(report["dsl_functions"]), "source_files_unchanged": True,
                      "status": "DESIGN_ONLY", "template_router_gaps_to_review": missing,
                      "gap_caution": "Unresolved template variables are not proven router defects; inspect expanded variants and workflow contracts.",
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
