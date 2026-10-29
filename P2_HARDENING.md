# P2 hardening

Документ фиксирует четыре пункта P2 после demo/scientific validation. Это
инженерная подготовка к release, а не новый 300-chart gate: полный live-14B
прогон должен быть выполнен на финальном HEAD после этих изменений.

## 1. Per-session SuperMAG

Раньше Gradio-обработчик писал логин в `os.environ["SUPERMAG_LOGON"]`. Это
делало секрет процессным: следующая сессия могла унаследовать логин первой.

Теперь:

- `FrameStore` хранит `supermag_logon`;
- `_apply_session_logon()` в `gradio_app.py` пишет логин только в `gr.State`
  текущего браузера;
- пустое поле очищает session-значение;
- `_handle_fetch_supermag()` передаёт `store.supermag_logon` в
  `supermag.fetch_supermag_data(..., logon=...)`;
- CLI/скрипты по-прежнему могут использовать `SUPERMAG_LOGON` как fallback,
  если явный логин не задан;
- в историю чата, tool log и `os.environ` значение из веб-поля не попадает.

Проверки:

```text
test_handles_and_plots.py::test_gradio_logon_is_stored_on_the_session_state
test_supermag_loader.py::test_handler_passes_the_session_logon_without_touching_the_environment
```

Ограничение: если оператор намеренно задал `SUPERMAG_LOGON` в окружении
сервера, он остаётся общим fallback-логином для сессий без своего значения.
Это конфигурация развёртывания, а не утечка пользовательского ввода.

## 2. CI и зависимости

Добавлены:

- `requirements.txt` — runtime: NumPy, pandas, Plotly, requests, pyarrow,
  Jinja2, Gradio, huggingface-hub;
- `requirements-dev.txt` — runtime + pytest;
- `requirements-llm.txt` — необязательный `llama-cpp-python` (ставится
  отдельным wheel под CPU/CUDA);
- `.github/workflows/tests.yml` — Python 3.12, offline suite на push/PR.

CI запускает:

```text
python -m pytest -q -m "not network"
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` отключает случайные внешние pytest-плагины
(на рабочей машине, например, несовместимый `nengo`). Network-тесты исключены;
они остаются отдельными маркерами и запускаются вручную при наличии доступа.

Локальный эквивалент команды CI проходит на текущем дереве:

```text
985 passed, 1 skipped, 22 deselected
```

`llama-cpp-python` в CI не нужен: модель и GPU проверяются только в live-gate.

## 3. Compact Plotly export

Каждый self-contained HTML содержит копию Plotly (~4.8 MB). Для 300 графиков
это ~1.5 GB исходников; ZIP сжимает их примерно до 66–80 MB, но копии
Plotly всё равно остаются внутри каждого файла.

Добавлен **опциональный export-only** режим:

```python
project_store.export_project(
    "demo_storm_may2024",
    archive_name="demo_storm_may2024_compact",
    compact=True,
)
```

или:

```text
python demo_make_50.py --compact
```

В таком ZIP:

- один `_plotly/plotly.min.js` на архив;
- каждая HTML-копия ссылается на него относительным путём
  (например, `../../_plotly/plotly.min.js`);
- данные и `Plotly.newPlot` остаются в HTML;
- проект на диске, `charts.json` и `created_at` не меняются;
- HTML в проекте остаётся self-contained для поля и офлайна.

Замер на существующем demo (50 графиков):

```text
default: 79 668 932 B
compact:  7 475 723 B
entries: 206
charts:  50
asset:   4 815 814 B (несжатый plotly.min.js)
testzip: None
```

Опциональный артефакт: `exports/demo_storm_may2024_compact.zip`. Default
`exports/demo_storm_may2024.zip` не перезаписывался.

Проверка compact-архива: 50/50 HTML, канонические имена, корректный
`../../_plotly/plotly.min.js`, `index.html` и `export_manifest.json` на месте.
Исходный `projects/demo_storm_may2024/...html` после экспорта остался
self-contained (4 971 797 B, встроенный Plotly).

## 4. Legacy stem

Старый двухтокенный source-суффикс `..._INTERMAGNET_OMNI.html` теперь
нормализуется в канонический `..._INTERMAGNET-OMNI.html`, а не теряет
`INTERMAGNET` и не превращается в `..._OMNI.html`.

Исправление в `chart_naming.py`:

- `source_from_filename("..._INTERMAGNET_OMNI.html")` возвращает
  `INTERMAGNET-OMNI`;
- `safe_filename()` идемпотентно приводит legacy-имя к каноническому;
- уже каноническое имя не меняется;
- регрессия закреплена в `test_chart_naming.py`.

## 5. Что остаётся для release

P2 меняет код, поэтому старый `stress_3ffd1r21` (коммит `ed46cf7`) больше
не является доказательством для финального HEAD.

Нужно:

1. Запушить P2-коммит.
2. Запустить полный 300-gate на этом HEAD: 300 HTML, шесть источников,
   канонические имена, ZIP, security-пробы.
3. Проверить, что live-14B демонстрация не подменена deterministic fallback.
4. Только после этого обновлять release-документацию и тег.
