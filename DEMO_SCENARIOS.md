# Демонстрационные сценарии

Все сценарии проверяют не только «красивый» ответ, но и отказы: частичные
данные, пустые ячейки, небезопасные пути и сетевые ошибки. Запускать после
загрузки реального `brain` (Qwen 2.5 14B), с `GEOMAG_OFFLINE=0`.

На Windows перед запуском полезно выставить UTF-8, иначе emoji в старых
консольных сообщениях могут дать `UnicodeEncodeError`:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

## 1. Реальная буря мая 2024

Промпт:

```text
Построй матрицу 10 станций (IRT, API, BSL, ABG, BRW, SHU, FRD, CMO, BOU, TUC)
на 14–18 мая 2024, действия:
["fetch_observatory_data", "calculate_derived_components", "plot_components"].
Проект: demo_storm_may2024.
Ровно один create_task_matrix и один process_matrix_batch(run_until_complete=True).
Затем export_project. Покажи полные, частичные и пустые станции.
```

Ожидается:

- 50 HTML в `demo_storm_may2024/<station>/<date>/`;
- имена `STATION_YYYY-MM-DD_H-D-I_INTERMAGNET.html`;
- `charts.json` с source/station/date/components/kind/product;
- default ZIP `demo_storm_may2024.zip`;
- `job_ok=true`, `coverage=50/50`; если API отдал не всё — `job_ok=false`,
  `missing_cells` и предупреждение, а не зелёный отчёт.

## 2. Partial / empty и повторный запуск

Промпт:

```text
Создай матрицу для ABG на 2024-05-13, действие fetch_observatory_data,
проект demo_empty_abg. Выполни process_matrix_batch(run_until_complete=True)
и покажи, что именно не пришло.
```

Ожидается:

- `ABG` не отдаёт 13 мая; ячейка попадает в `missing_cells`;
- `job_ok=false`, `empty`/`partial` не скрываются;
- ZIP может быть собран из доступных данных, но summary содержит
  «ПОКРЫТИЕ НЕПОЛНОЕ»;
- повторный `process_matrix_batch(run_until_complete=True)` увеличивает
  `attempts` и повторяет `partial`/`failed` задачи, а не возвращает
  `processed=0`.

## 3. Custom ZIP без мутации проекта

Промпт:

```text
Выполни ровно один export_project для demo_storm_may2024:
archive_name="storm_report_2024",
folders={"1":["irt","api","bsl","abg"],"2":["brw","shu","frd"],
         "3":["cmo","bou","tuc"]},
sort_by=["date","station"].
Не меняй файлы проекта.
```

Ожидается:

- `storm_report_2024.zip` с верхними папками `1/2/3`;
- внутри — те же канонические имена, без переименования;
- `index.csv` отсортирован по `date, station`;
- проект на диске не изменился, `created_at` сохранился.

## 4. Безопасность путей

### Внешний matrix JSON

```text
process_matrix_batch(matrix_path="C:\\Temp\\outside_matrix.json",
                     run_until_complete=True)
```

Ожидается `error="matrix_outside_workspace"`; файл не читается и не
перезаписывается.

### Опасное имя архива

```text
export_project(name="demo_storm_may2024", archive_name="../evil")
```

Ожидается `error="invalid_archive_name"`; `manifest.json` не меняется.

### Junction / symlink

Создайте junction внутри тестового проекта на внешнюю папку и вызовите
`project_tree`/`export_project`. Ожидается, что ссылка и её содержимое не
попадут ни в дерево, ни в ZIP.

### `plot_overlay` в матрице

```text
create_task_matrix(..., actions=["fetch", "plot_overlay"])
```

Ожидается `error="unsupported_matrix_action"`: `plot_overlay` требует
несколько станций сразу и не является per-cell действием.

## 5. Сетевые ошибки и структурированные ответы

Запросите заведомо отсутствующий день/станцию или отключите сеть. Ожидается
`no_data`/`network`/`http_error` с `message`, `hint` и без traceback. Повторный
`process_matrix_batch(run_until_complete=True)` должен ретраить partial/failed
ячейки, а `_run_matrix_job` — возвращать `ok=false` при неполном покрытии.

## Быстрая проверка перед показом

```powershell
python -m pytest -p no:nengo -q
```

Полный 300-chart gate: `stress_300.py` в Kaggle-ноутбуке с реальным `brain`.
Старый зелёный артефакт `stress_g3solehz` не перезапускать и не переименовывать.
