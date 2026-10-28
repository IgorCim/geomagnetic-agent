# ACCEPTANCE — 300-chart / six-source gate

**Дата:** 2026-10-28
**Статус:** PASS для матричного, экспортного и six-source контура.
**Кодовый baseline:** `ed46cf73c48221051319b7d49ce3aabe8eaa475e` (`ed46cf7`, 2026-10-28).
**Модель:** Qwen 2.5 14B Instruct GGUF Q4_K_M, `n_ctx=16384`, реальный `brain` (не `ScriptedBrain`).
**Прогон:** `/kaggle/working/geomagnetic-agent/stress_runs/stress_3ffd1r21`.

> Важно: `stress_300.py` не печатает `git rev-parse HEAD`. Указанный baseline — тот
> checkout, из которого готовился ноутбук. Для абсолютной воспроизводимости в
> ноутбук нужно добавить вывод `git rev-parse HEAD` перед stress-ячейкой.

---

## 1. Результат

| Блок | Проект | Сетка | HTML | Покрытие | Default ZIP |
|---|---|---|---:|---:|---|
| A | `may2024_matrix_A_200` | 10 станций × 20 дат | 200 | 200/200 | `exports/may2024_matrix_a_200.zip` |
| B | `may2024_matrix_B_50` | 5 станций × 10 дат | 50 | 50/50 | `exports/may2024_matrix_b_50.zip` |
| C | `may2024_spaceweather_50` | 10 станций × 5 дат | 50 | 50/50 | `exports/may2024_spaceweather_50.zip` |
| **Итого** | 3 матрицы | — | **300** | **300/300** | 3 ZIP |

- 300 уникальных HTML-путей.
- 300 уникальных канонических basename.
- `UNKNOWN` source-тегов нет.
- A/B: `STATION_YYYY-MM-DD_H-D-I_INTERMAGNET.html`.
- C: `STATION_YYYY-MM-DD_H_vs_OMNI_2024-05-01_Bz-V_INTERMAGNET-OMNI.html`.
- В каждом матричном запуске — один `create_task_matrix`; матричный цикл
  выполняет один `process_matrix_batch(run_until_complete=True)` и экспорт.

## 2. Покрытие станций

A (20 дат на станцию): IRT, API, BSL, ABG, BRW, SHU, FRD, CMO, BOU, TUC — все
`20/20`, `status=done`.

B (10 дат на станцию): BRW, SHU, FRD, CMO, BOU — все `10/10`, `status=done`.

C (5 дат на станцию): IRT, API, BSL, ABG, BRW, SHU, FRD, CMO, BOU, TUC — все
`5/5`, `status=done`.

Итог по статусам в этом прогоне:

- полные: A 10/10 станций, B 5/5, C 10/10;
- частичные: 0;
- пустые: 0;
- `missing_cells`: нет;
- `job_ok`: true для A/B/C.

Это не сокрытие пропусков: в выбранном окне 14 мая–2 июня 2024 и 1–10 мая
2024 все запрошенные станции отдали полные наборы. Сценарий partial/empty
проверяется отдельными тестами и демо-сценарием 2, а не этим 300-chart gate.

## 3. Custom export

- Архив: `exports/storm_report_2024.zip`.
- Размер в логе: 66.1 MB.
- Верхние папки: `1`, `2`, `3`.
- 50 строк в `index.csv`, сортировка `date, station`.
- `source_order`: `OMNI`, `INTERMAGNET`, `KYOTO`, `SUPERMAG`, `NOAA`, `ISWA`.
- Канонические имена внутри архива сохранены.
- Проект C после экспорта не изменился: 50 HTML до и 50 после.
- `charts.json` проекта C не изменился.
- `created_at` проекта C не изменился.
- Custom export — представление, не мутация проекта.

## 4. Источники

| Источник | Что подтверждено в прогоне |
|---|---|
| INTERMAGNET | 250 графиков A/B; 50 ground-серий C |
| NASA CDAWeb / OMNI | `sw:omni:2024-05-01`, 43 200 строк; 50 comparison-графиков C |
| Kyoto WDC | `idx:kyoto:2024-05-01`, 720 строк Dst/Kp |
| SuperMAG | два окна по 15 дней, по 21 600 строк; `SUPERMAG_LOGON` задан |
| NOAA SWPC | alerts 48, forecast 81, solar_events 0 |
| NASA ISWA | forecast 25 строк; perturbation map — `service_decommissioned` |

`skipped: none`. SuperMAG не пропущен.

## 5. Security / export contract

`security probe: ok`, `security notes: none`.

Проверено в этом прогоне:

- path-shaped `archive_name` отклоняется;
- path-shaped `folders` отклоняется;
- отклонённый export не меняет `manifest.json`;
- `include_index="false"` не пишет `index.csv`/`index.html`;
- `created_at` сохраняется при повторном export;
- symlink/junction не попадает в `project_tree` и ZIP;
- содержимое внешней цели symlink не утекает в ZIP.

Границы этого прогона: внешний `matrix_path`, reserved/длинные имена проектов,
`plot_overlay` как matrix action, ретрай partial/failed и неизвестный matrix
action проверяются pytest-регрессиями, а не `stress_300.py`.

## 6. Регрессии и бюджет промпта

- Полный suite перед приёмкой: `997 passed, 3 skipped`.
- Targeted matrix/console: `37 passed`.
- Projects/export/junction/created_at: `18 passed`.
- Chart naming: `8 passed`.
- `SYSTEM_PROMPT`: 3194 символа.
- `prompt_budget(SYSTEM_PROMPT, TOOL_SCHEMAS)`: `est_tokens=8187`, `n_ctx=16384`.
- В stress-логе строка `[PROMPT BUDGET] system=0ch ... est_tokens=1` относится
  к вызову планировщика без system/tools, а не к основному `SYSTEM_PROMPT`.
  Основной tool-loop с `SYSTEM_PROMPT` этим прогоном не проверялся.

## 7. Ожидаемые ограничения внешних сервисов

1. NOAA `alerts`/`forecast` — текущее окно 2026 года, не исторический май 2024.
2. NOAA `solar_events: 0` — следствие семидневного эндпоинта; это не
   доказательство отсутствия вспышек в мае 2024.
3. ISWA perturbation map — `service_decommissioned` у NASA; ожидаемый ответ.
4. ISWA forecast — текущий прогноз, не исторический май 2024.
5. Kyoto/SuperMAG/NOAA/ISWA в этом прогоне — контекстные вызовы; на 300 HTML
   наложены только INTERMAGNET и OMNI.
6. Gate проверяет наличие, имена, метаданные, ZIP, безопасность и покрытие.
   Он не проверяет научную корректность кривых, масштабов и подписей.

## 8. Артефакты

Прогон создан в отдельном каталоге и не затрагивает старый
`stress_g3solehz`:

```text
stress_runs/stress_3ffd1r21/
  projects/
  exports/
    may2024_matrix_a_200.zip
    may2024_matrix_b_50.zip
    may2024_spaceweather_50.zip
    storm_report_2024.zip
```

`stress_runs/` игнорируется git. Для приёмки нужно сохранить лог и ZIP вне
Kaggle-сессии; в репозитории остаётся только этот документ.

## 9. Вердикт

**Матричный, экспортный и six-source gate — PASS для `ed46cf7`.**

Следующие шаги:

1. Научная валидация значений и выборочная проверка графиков.
2. End-to-end demo в Gradio: 50 графиков, partial/empty, custom ZIP,
   multi-turn store, security-отказы.
3. P2: legacy stem, размеры Plotly-артефактов, per-session SuperMAG, CI.

## 10. Post-acceptance изменения

После прогона `stress_3ffd1r21` в рабочем дереве появились изменения, которые
**не входили в этот gate**:

- `geomag_plotter.py`: H/D/I и Bz/V с разными единицами теперь рисуются на
  отдельных панелях; одиночные/одноимённые единицы сохраняют прежний вид.
- `chart_naming.py`: заголовок C больше не печатает descriptor
  `sw:omni:2024-05-01` как дату данных.
- Добавлены регрессии в `test_geomag_tools.py` и `test_chart_naming.py`.
- Добавлен `peak_storm_2024.py` и 20-графиковый supplement 10–12 мая
  (`PEAK_STORM_2024.md`), вне 300-набора.

Поэтому 300 HTML в `stress_3ffd1r21` остаются валидными для `ed46cf7`, но не
для нового HEAD. Полный 300-gate на новом HEAD нужно прогнать отдельно перед
release; старый `stress_g3solehz` и `stress_3ffd1r21` при этом не трогать.
