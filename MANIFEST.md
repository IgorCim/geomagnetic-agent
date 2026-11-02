> Историческое описание от 21.10.2026. Часть сведений устарела (SuperMAG limit, хранение логина, UI, тесты). Текущие документы: [README](README.md), [описание продукта](docs/PROJECT_BRIEF.md), [разбор кода](docs/CODE_REVIEW.md).

# 🧲 ТЕХНИЧЕСКИЙ МАНИФЕСТ: Геомагнитный ИИ-агент

**Версия:** Stage 4 (полный стек Stages 1–4)
**Дата аудита:** 21 октября 2026
**Ядро:** Qwen 2.5 7B (llama.cpp, N_CTX = 16384)
**Инструментов:** 29 · **Физических метрик:** 11 · **Базовых формул:** 11 · **Источников данных:** 6
**Код:** 32 .py-файла, ~28 000 строк (не-тестовый код ~18 000 строк)

---

## РАЗДЕЛ 1: АРХИТЕКТУРА СИСТЕМЫ

### 1.1. Общая схема

Система построена как **четырёхстадийный конвейер**, где каждая стадия — отдельный модуль с чистым контрактом:

| Стадия | Модуль | Роль |
|--------|--------|------|
| 1 | `intermagnet_loader.py`, `supermag_loader.py`, `space_weather_loader.py`, `noaa_swpc_loader.py`, `iswa_loader.py` | **Данные** — скачивание и нормализация |
| 2 | `geomag_math.py`, `geomag_coords.py`, `geomag_analyzer.py`, `geomag_plotter.py` | **Математика** — компоненты, статистика, графики |
| 3 | `physics_engine.py`, `agent_core.py`, `agent_state.py`, `agent_planner.py` | **Мозг** — физика, tool-calling, планирование |
| 4 | `gradio_app.py` | **Интерфейс** — чат, панель графиков, проекты |

Поток данных: `LLM → tool_call → HANDLER → loader/engine → FrameStore → frame_handle → LLM`.

### 1.2. Агент: LLM + Scratchpad + Валидация

**LLM-слой.** Модель (Qwen 2.5 7B) не имеет памяти между запросами — каждый вызов `run_agent` независим. Модель получает:

- `SYSTEM_PROMPT` (3199 символов) с нумерованными правилами 1–11 и «ПРАВИЛО 0 (ДЕКОМПОЗИЦИЯ)»;
- динамически выбранные схемы инструментов (`select_tools_for_query`, максимум `TOOLS_PER_TASK = 6` из 29);
- тексты ошибок предыдущих раундов.

**Scratchpad (класс `Scratchpad`, `agent_core.py:1566`).** Это «внешняя рабочая память» — обычный Python-объект, а не контекст модели:

```python
class Scratchpad:
    current_handle: str          # активный frame_handle (raw: → derived: → ...)
    completed_steps: list[str]   # что уже сделано (без повторов)
    errors: list[...]            # история ошибок с подсказками
    failures: dict[signature]    # счётчик одинаковых падений
```

Ключевая идея: `to_prompt_injection()` рендерит ровно те факты, которые модели нужны прямо сейчас, а не весь лог.

**Валидация.** Два уровня:

1. **Строгая проверка аргументов** (`_strict_argument_check`) — например, `apply_physics_formula` разрешает только `df_descriptor`, `metric`, `external_params`, `window_seconds`. Лишний ключ → ошибка до исполнения.
2. **Валидация внешних параметров** (`_validate_external_params`) — до диспетчеризации проверяется, что для `burton_coupling` есть `Np/V/Bz/Dst`, а для `eej_index` — референсная станция. Отсутствие называется здесь, а не тонет в недрах движка.

Хэндлы (`frame_handle`) — это единственный способ адресовать данные. Формат: `raw:<station>:<date>`, `derived:<station>:<date>`, `sw:omni:<date>`, `idx:kyoto:<date>`, `idx:supermag:<date>`, `anomalies:...`. Семейство задаёт слот; `_AUX_FRAME_FAMILIES` (`sw:`, `idx:`) не перехватывают рабочий слот.

### 1.3. Цикл рассуждений (tool calling → ошибка → исправление)

Главный цикл — `run_agent` (`agent_core.py:6091`):

```
round = 0
while round < max_rounds (18) and spent < max_tool_calls (16):
    1. brain.chat(messages, tools)  → текст + список tool-calls
    2. если вызовов нет          → стоп (stop_reason="no_tool_calls") → финальный текст
    3. если бюджет исчерпан      → refused-записи в лог, форсируем ответ
    4. для каждого вызова:
         malformed → error malformed_tool_call + образец синтаксиса
         unknown   → error unknown_tool + _self_correction_hint
         известный → HANDLER(args, store) → payload, note
         исключение → error tool_exception (traceback, но цикл жив)
    5. payload → _compact_tool_message (≤400 символов) → роль "tool"
    6. обновить Scratchpad (handle, шаги, ошибки)
```

**Самокоррекция.** `_self_correction_hint(tool_name)` превращает галлюцинацию в конкретное действие: если модель зовёт `calculate_dst_proxy`, подсказка отвечает «используй `apply_physics_formula(metric='dst_proxy')`». Если ошибка `unknown_frame` — hint заменяется на актуальный `scratchpad.current_handle`:

```python
if payload.get("error") == "unknown_frame" and scratchpad.current_handle:
    hint = f"Использовал неверный handle, актуальный: {scratchpad.current_handle}"
```

**Разрыв цикла.** `add_failure(name, args_signature(args), error)` считает идентичные падения. Повтор той же падающей пары (инструмент + аргументы) генерирует `[CRITICAL LOOP DETECTED]` — модель обязана сменить стратегию, а не долбиться в одну точку.

**Короткое замыкание для массовых задач.** Если `process_matrix_batch` вернул `completed_run=True`, цикл **не отдаёт длинный payload модели** (это переполняет контекст 7B и даёт пустой ответ «Нет текстового ответа»), а сразу возвращает `stop_reason="matrix_completed_short_circuit"` с текстом инструмента.

### 1.4. Отказоустойчивость (Graceful Degradation)

| Отказ | Поведение |
|-------|-----------|
| Тул не существует | `unknown_tool` + список доступных + подсказка |
| Невалидный JSON вызова | `malformed_tool_call` + образец `<tool_call>{...}</tool_call>` |
| Исключение в хендлере | `tool_exception`, traceback в stderr, цикл продолжается |
| Исчерпан бюджет вызовов | каждая отказанная пара пишется отдельной строкой (`executed=False`), принудительный ответ с пометкой |
| Исчерпаны раунды | `stop_reason="round_limit"`, текст + список падений (`_with_tool_failures`) |
| Ошибка движка | dict `{"ok": False, "error", "message", "missing", "available", "hint"}` |
| Сеть недоступна | `GEOMAG_OFFLINE=1` → синтетические данные; иначе ошибка с `hint` |
| График не восстановился | текст всё равно приходит, `_fallback_note` указывает файл |
| Несколько станций в `fetch_many` | упавшая станция попадает в `failed`, остальные — работают |

**Принцип:** ничто не должно «убить» сервер. Один плохой ход не роняет процесс (см. `try/except` в `chat_and_plot`).

### 1.5. Credential Isolation для SuperMAG

SuperMAG отдаёт некоторые наборы только по логину. Реализация изоляции — в `gradio_app.py`:

- В UI есть `sm_logon = gr.Textbox(type="password")` — логин не показывается и **не попадает в историю чата или лог**.
- В `submit(...)` логин пишется в окружение **на время запроса**:

```python
def submit(text, history, state, logon):
    if not _message_text(text):
        gr.Info("Введите вопрос.")
        return gr.skip()
    if logon and logon.strip():
        os.environ["SUPERMAG_LOGON"] = logon.strip()
    return chat_and_plot(text, history, state)
```

- `supermag_loader.get_logon()` читает `SUPERMAG_LOGON` при формировании запроса.
- `os.environ` в сообщениях агента **никогда не логируется**: `_format_tool_log` пишет только имя инструмента и аргументы, а логин — не аргумент.

**Честное ограничение:** переменная окружения живёт в процессе между запросами (видна другим потокам того же сервера). Для одиночного пользователя это приемлемо; строгая изоляция потребовала бы передавать логин параметром в `chat_and_plot → supermag_loader` без глобального `os.environ`.

---

## РАЗДЕЛ 2: ИСТОЧНИКИ ДАННЫХ (6 штук)

### 2.1. INTERMAGNET (Edinburgh GIN + Kyoto WDC)

- **URL:** `https://imag-data.bgs.ac.uk/GIN_V1/GINServices` (первичный), `https://wdc.kugi.kyoto-u.ac.jp/hapi` (вторичный)
- **Данные:** минутные/секундные магнитограммы X/Y/Z/F/H/D/I, IAGA-2002
- **Загрузчик:** `intermagnet_loader.py` (`fetch_observatory_data`, `get_available_stations`, `get_station_details`)
- **Покрытие:** Edinburgh GIN — 154 обсерватории (до вчера); Kyoto — 326 обсерваторий (архив ~1998–2021/22)
- **Публикационные состояния:** `definitive`, `quasi-def`, `adjusted`, `reported`, `best-avail`, `auto`
- **Примеры станций:** IRT (Иркутск), MOS (Москва), NVS (Новосибирск), SPG (СПб), YAK (Якутск), KHB (Хабаровск), MGD (Магадан), VLA (Владивосток), PET (Камчатка), ARS, BOX, TIK

### 2.2. NASA CDAWeb HAPI (OMNI + Kyoto индексы)

- **URL:** `https://cdaweb.gsfc.nasa.gov/hapi`, `https://wdc.kugi.kyoto-u.ac.jp/hapi`
- **Данные:** OMNI 1-мин солнечный ветер (`OMNI_HRO2_1MIN`): `flow_speed` (V, км/с), `proton_density` (Np, см⁻³), `T` (K), `BX_GSE`/`BY_GSM`/`BZ_GSM` (нТл); Kyoto `hour_dst` (Dst, нТл), `hour3h_kp` (Kp, 0–9)
- **Загрузчик:** `space_weather_loader.py` (`fetch_solar_wind`, `fetch_geomagnetic_indices`)
- **Что добавляет:** причина бури (южный Bz + быстрый V) и официальная мера силы

### 2.3. SuperMAG (JHU/APL)

- **URL:** `https://supermag.jhuapl.edu/services/` (`data-api.php`, `indices.php`, `inventory.php`)
- **Данные:** базово-скомпенсированные компоненты `dX/dY/dZ` (возмущение ≈ 0) и глобальные индексы `SME/SML/SMU` (1 мин)
- **Загрузчик:** `supermag_loader.py` (`fetch_supermag_data`, `fetch_supermag_inventory`, `get_logon`)
- **Параметры:** `data_type='sites'` (нужна станция) или `'indices'` (станция не нужна)
- **Лимит:** максимум 32 дня за запрос
- **Что добавляет:** *возмущение* как таковое и тайминг суббурь

### 2.4. NASA NOAA SWPC

- **URL:** `https://services.swpc.noaa.gov/products/alerts.json`, `.../noaa-planetary-k-index-forecast.json`, `.../json/goes/primary/xray-flares-7-day.json`, `.../solar_probabilities.json`
- **Данные:** алерты G1–G5, 3-дневный прогноз Kp, каталог рентгеновских вспышек GOES
- **Загрузчик:** `noaa_swpc_loader.py` (`fetch_geomagnetic_alerts`, `fetch_solar_wind_forecast`, `fetch_solar_events`)
- **Что добавляет:** официальная оперативная картина космической погоды

### 2.5. NASA ISWA (CCMC)

- **URL:** `https://iswa.ccmc.gsfc.nasa.gov/` (HAPI-часть работает)
- **Данные:** прогноз Dst/Kp на 3-часовой сетке (поле `kp_kind`: observed/estimated/predicted); Dst — nowcast Kyoto
- **Загрузчик:** `iswa_loader.py` (`get_dst_kp_forecast`, `get_ground_perturbation_map`, `build_wms_url`)
- **⚠️ ДЕКОМИССИЯ:** публичный WMS-слой `ground_mag_perturbation` **удалён** при миграции NASA CCMC (проверено 2026-10-20: в каталоге из 1351 фида нет `perturbation`/`ground magnet`; все WMS-пути возвращают 404/403). Инструмент вернёт `service_decommissioned`.

### 2.6. Kyoto WDC (Dst/Kp) — исторический архив

- **URL:** `https://wdc.kugi.kyoto-u.ac.jp/hapi`
- **Данные:** канонический Dst и Kp (первоисточник, не зеркало)
- **Роль:** восстановление definitive-данных за старые интервалы, где GIN их не отдаёт

### 2.7. Внутреннее устройство загрузчиков (построчно)

**Общий контракт.** Все пять загрузчиков возвращают либо `pandas.DataFrame` (успех), либо **стабильный error-dict** `{"ok": False, "error": <код>, "message", ..., "hint"?}`; `is_error(result)` отличает одно от другого. Хендлеры в `agent_core` не бросают исключений наружу — исключение загрузчика превращается в `tool_exception` цикла, но сам процесс не падает.

**Сеть, ретраи, байты.** Во всех модулях `DEFAULT_MAX_RETRIES = 3`, `DEFAULT_TIMEOUT` 30–90 с (INTERMAGNET — 90, SuperMAG — 60, NOAA/ISWA — 30). Запросы идут с **браузерным `User-Agent`** (иначе часть хостов отдаёт 403). Ретрай — с задержкой (`time.sleep`), а не в tight-loop.

**INTERMAGNET (`intermagnet_loader.py`, 1023 строки).**
- Константы: `GIN_URL`, `KYOTO_HAPI_URL`, `CHUNK_DAYS = 31`, `STATION_CACHE_TTL_S = 86400`, `COMPONENTS = ('X','Y','Z','F')`.
- **Цепочки публикационных состояний:** GIN — `definitive → quasi-def → adjusted → reported → best-avail`; Kyoto — `definitive → best-avail`. `data_type='auto'` берёт всю цепочку целиком; явное состояние ставится первым, остальные — фолбэк.
- **Разбиение на чанки** по 31 дню (`_chunk_dates`), затем `pd.concat` → сортировка → `drop_duplicates(subset='timestamp')` → колонки строго `timestamp,X,Y,Z,F`.
- **Парсер IAGA-2002** (`parse_iaga2002`): читает заголовок, извлекает пропуски/нули, приводит к nT; `X/Y/Z/F`.
- **Кэш:** каталог станций (TTL 24 ч) + данные (`save_to_cache`/`load_from_cache`), `clear_cache`, флаг `refresh`.
- **Коды ошибок:** `invalid_input` (дата/тип/end<start), `catalogue_unavailable`, `station_not_found` (с подсказкой соседних кодов по первой букве), `no_data` (все состояния исчерпаны, `tried=attempts`, hint про лаг definitive до года).
- Возвращаемые `attrs`: станция, разрешённое публикационное состояние, backend (`GIN`/`kyoto`), `from_cache`.

**Space Weather (`space_weather_loader.py`, 591 строка).**
- `CDAWEB_HAPI_URL`, `KYOTO_HAPI_URL`, `SOLAR_WIND_DATASETS = ('OMNI_HRO2_1MIN','OMNI_HRO_1MIN')` (второй — фолбэк), `DST_DATASET='hour_dst'`, `KP_DATASET='hour3h_kp'`.
- Порядок колонок жёстко задан: ветер `timestamp,V,Np,T,Bx,By,Bz`; индексы `timestamp,Dst,Kp`.
- Коды ошибок: `api_unavailable`, `invalid_input`. Кэш отсутствует (оперативные данные).

**SuperMAG (`supermag_loader.py`, 647 строк).**
- `BASE_URL`, `DATA_URL` (`data-api.php`), `INDICES_URL` (`indices.php`), `INVENTORY_URL`, `DATA_TYPES = ('sites','indices')`, `INDEX_COLUMNS = ('SME','SML','SMU')`, `GLOBAL_STATION = 'GLOBAL'`, `MAX_DAYS = 32`, `LOGON_ENV = 'SUPERMAG_LOGON'`.
- `data_type='sites'` требует станцию; `'indices'` — нет (станция `GLOBAL`).
- `get_logon()` читает `SUPERMAG_LOGON`; при отсутствии — подсказка о бесплатной регистрации.
- **Лимит 32 дня** с отдельной подсказкой `_PERIOD_HINT` (код станции / другой период / `data_type='indices'`).

**NOAA SWPC (`noaa_swpc_loader.py`, 571 строка).**
- Конечные точки: `ALERTS_URL`, `KP_FORECAST_URL`, `SOLAR_PROBABILITIES_URL`, `XRAY_FLARES_URL` (7 дней) и `XRAY_FLARES_LATEST_URL`.
- Из алертов время вытаскивается набором `_TIME_FIELDS` (`issue_time`/`start_time`/`end_time` × «Issue Time:»/«Valid From:»/…).
- Коды ошибок: `invalid_input`, `noaa_api_unavailable` (+ `_UNAVAILABLE_HINT`).

**ISWA (`iswa_loader.py`, 864 строки).**
- `HAPI_BASE_URL`, `WMS_ENDPOINT` (`wmts.cgi`), `DST_DATASET='dst_quicklook'`, `KP_DATASET='NOAA_KP_P3H'`, окно прогноза `FORECAST_HOURS_BACK=24 / AHEAD=48`, шаг 3 ч, `_DEFAULT_BBOX=(-180,-90,180,90)`.
- Коды ошибок: `invalid_input`, `iswa_forecast_unavailable`, `iswa_unavailable`, `no_data`, **`service_decommissioned`** (WMS-слой, см. §8.2) с `_DECOMMISSIONED_FALLBACK` (план из двух шагов).
- Утилиты `iso_z`, `parse_timestamp`, `build_wms_url`, `save_map_png`, `synthetic_map_png` (offline).

---

## РАЗДЕЛ 3: ВСЕ ИНСТРУМЕНТЫ (29)

> Все схемы — в `TOOL_SCHEMAS`; хендлеры — в `HANDLERS` (`agent_core.py`). Каждая строка «Пример» — валидный JSON вызова.

### Данные

**1. `fetch_observatory_data`** — скачать один день одной обсерватории. Первый вызов перед любым анализом.
- Аргументы: `station_code: str`, `start_date: str`, `end_date: str` (опц.), `data_type: enum` (опц.), `samples_per_day: enum["Minute","Second"]` (опц.)
- Пример: `{"name":"fetch_observatory_data","arguments":{"station_code":"IRT","start_date":"2024-09-10"}}`

**2. `fetch_many`** — скачать диапазон для НЕСКОЛЬКИХ станций за один вызов.
- Аргументы: `stations: str[]`, `start_date: str`, `end_date: str`, `data_type: enum`, `samples_per_day: enum`, `max_downloads: int` (потолок 8), `project: str`
- Пример: `{"name":"fetch_many","arguments":{"stations":["IRT","NVS","YAK"],"start_date":"2024-09-10","end_date":"2024-09-12"}}`

**3. `fetch_solar_wind`** — OMNI 1-мин солнечный ветер.
- Аргументы: `start_date: str`, `end_date: str` (опц.)
- Пример: `{"name":"fetch_solar_wind","arguments":{"start_date":"2024-09-12"}}`

**4. `fetch_geomagnetic_indices`** — Dst (часовой) и Kp (3-часовой) из Kyoto.
- Аргументы: `start_date: str`, `end_date: str`
- Пример: `{"name":"fetch_geomagnetic_indices","arguments":{"start_date":"2024-09-12"}}`

**5. `fetch_supermag_data`** — SuperMAG: возмущённые компоненты или индексы суббурь.
- Аргументы: `start_date: str`, `end_date: str`, `data_type: enum["sites","indices"]`, `station_code: str` (опц.)
- Пример: `{"name":"fetch_supermag_data","arguments":{"start_date":"2024-09-12","data_type":"indices"}}`

**6. `get_noaa_space_weather_context`** — алерты/прогноз/вспышки NOAA SWPC.
- Аргументы: `query_type: enum["alerts","forecast","solar_events"]`, `start_date: str`, `end_date: str`
- Пример: `{"name":"get_noaa_space_weather_context","arguments":{"query_type":"forecast"}}`

**7. `get_iswa_forecast`** — прогноз Dst/Kp из ISWA (без аргументов).
- Пример: `{"name":"get_iswa_forecast","arguments":{}}`

**8. `get_iswa_perturbation_map`** — карта возмущений ISWA (ДЕКОМИССИРОВАНА).
- Аргументы: `timestamp: str`, `bbox: number[]`, `width: int`, `height: int`
- Пример: `{"name":"get_iswa_perturbation_map","arguments":{"timestamp":"2024-09-12T12:00:00Z"}}`

### Физика и математика

**9. `apply_physics_formula`** — ЕДИНЫЙ мета-инструмент для всей физики.
- Аргументы: `df_descriptor: str`, `metric: enum` (12 значений), `external_params: object`, `window_seconds: int`
- Пример: `{"name":"apply_physics_formula","arguments":{"df_descriptor":"raw:irt:2024-09-10","metric":"dst_proxy"}}`

**10. `calculate_eej_index`** — экваториальный электроджет (разность H двух станций).
- Аргументы: `equatorial_station: str`, `reference_station: str`, `start_date: str`, `end_date: str`
- Пример: `{"name":"calculate_eej_index","arguments":{"equatorial_station":"AAE","reference_station":"ABG","start_date":"2023-09-12"}}`

**11. `calculate_derived_components`** — производные компоненты H/D/I.
- Аргументы: `df: str`, `components: str[]` (опц.)

**12. `calculate_derived_math`** — скалярные метрики: `delta`, `anomaly`, `dH_dt`.
- Аргументы: `df: str`, `metric: enum["delta","anomaly","dH_dt"]`, `component: enum`, `window: int`, `baseline_value: number`
- Пример: `{"name":"calculate_derived_math","arguments":{"df":"raw:irt:2024-09-10","metric":"anomaly","component":"H","baseline_value":50120.5}}`

**13. `calculate_baseline`** — тихий опорный уровень компоненты.
- Аргументы: `df: str`, `component: enum`, `mode: enum["night","full"]`, `night_hours: int[]`
- Пример: `{"name":"calculate_baseline","arguments":{"df":"raw:irt:2024-09-10","component":"H"}}`

**14. `evaluate_custom_formula`** — пользовательская арифметика над колонками (парсится, НЕ исполняется как код).
- Аргументы: `df: str`, `formula: str`
- Пример: `{"name":"evaluate_custom_formula","arguments":{"df":"raw:irt:2024-09-10","formula":"sqrt(X**2 + Y**2) + Z/10"}}`

**15. `get_statistics`** — mean/min/max/std по колонкам.

**16. `detect_anomalies`** — выбросы по сигмам.
- Аргументы: `df: str`, `component: str`, `threshold: number`

### Графики

**17. `plot_components`** — одна или несколько колонок одной серии.
- Пример: `{"name":"plot_components","arguments":{"df":"raw:irt:2024-09-10","components":["H","D"]}}`

**18. `plot_comparison`** — два дня одной станции.
- Аргументы: `df1: str`, `df2: str`, `component: enum`, `title: str`, `filename: str`

**19. `plot_overlay`** — один компонент с нескольких станций на одном графике.
- Пример: `{"name":"plot_overlay","arguments":{"stations":["IRT","NVS"],"dates":["2024-09-12"],"component":"H"}}`

**20. `plot_space_weather_comparison`** — Земля (левая ось) vs солнечный ветер (правая).
- Пример: `{"name":"plot_space_weather_comparison","arguments":{"ground_df_descriptor":"derived:irt:2024-09-12","solar_wind_df_descriptor":"sw:omni:2024-09-12","ground_component":"H","solar_wind_components":["Bz","V"]}}`

### Проекты

**21. `create_project`** — создать проект (`name: str`).
**22. `list_projects`** — список проектов.
**23. `export_project`** — упаковать проект в ZIP (`name: str` опц.).

### Массовая обработка

**24. `create_task_matrix`** — план пакетной задачи (по станциям).
- Аргументы: `events: str[]`, `stations: str[]`, `actions: str[]`, `project_name: str`
**25. `process_matrix_batch`** — исполнить матрицу.
- Аргументы: `matrix_path: str`, `run_until_complete: bool`, `batch_size: int`

### Магнетизм/координаты

**26. `get_station_geomagnetic_coords`** — геомагнитные координаты станции.
**27. `calculate_mlt`** — магнитное локальное время.
**28. `calculate_local_time`** — гражданское локальное время (`LT = UT + lon/15`).
**29. `group_stations_by_mlt`** — группировка станций по MLT-секторам.

---

## РАЗДЕЛ 4: ВСЕ ФИЗИЧЕСКИЕ ФОРМУЛЫ И МЕТРИКИ

Формулы живут **на трёх уровнях**:

| Уровень | Модуль | Что считает | Как вызывается |
|---------|--------|-------------|----------------|
| **А. Базовая математика** | `geomag_math.py` | компоненты H/D/I/F, вариации, производные, ночной базис, DSL | `calculate_derived_components`, `calculate_derived_math`, `calculate_baseline`, `evaluate_custom_formula` |
| **Б. Физика** | `physics_engine.py` | 11 физических метрик | `apply_physics_formula`, `calculate_eej_index` |
| **В. Анализ и координаты** | `geomag_analyzer.py`, `geomag_coords.py` | статистика, аномалии, производные, геомагнитные координаты, MLT | `get_statistics`, `detect_anomalies`, `get_station_geomagnetic_coords`, `calculate_mlt` |

### 4.0. ⚠️ Коллизия имени `dst_proxy`

В проекте **две разные** функции с именем `calculate_dst_proxy`:

- `geomag_math.calculate_dst_proxy(df, component)` — **вторая производная** `d²H/dt²` (кривизна поля).
- `physics_engine.calculate_dst_proxy(df, latitude)` — **упрощённый Dst** `dH·cos(широта)`.

Через инструменты доступна **только физическая** версия (`apply_physics_formula(metric='dst_proxy')`). Базовая (`geomag_math`) — внутренняя, используется для низкоуровневой математики. Не путать их: smooth-экскурсия 100 нТл за 10 мин имеет большой вклад во «вторую производную», но малый мгновенный `dH/dt`.

### 4.А. Базовая математика (`geomag_math.py`)

Функции: `horizontal_field`, `total_field`, `declination`, `inclination`, `add_derived_components`, `component_series`, `calculate_delta`, `calculate_anomaly`, `calculate_dH_dt`, `calculate_dst_proxy` (2-я производная), `get_nighttime_baseline`, `evaluate_formula`.

- **H (горизонтальная составляющая):** `H = sqrt(X² + Y²)` (через `np.hypot`, устойчиво к переполнению; ровно `0.0` при X=Y=0).
- **F (полное поле):** `F = sqrt(X² + Y² + Z²) = hypot(hypot(X,Y), Z)`; всегда `F ≥ H`.
- **D (склонение):** `D = degrees(arctan2(Y, X))` ∈ [−180°, 180°], положительно на восток; при X=Y=0 → `0.0` (счётчик таких строк в `attrs['degenerate_horizontal_rows']`).
- **I (наклонение):** `I = degrees(arctan2(Z, H))` ∈ [−90°, 90°], положительно вниз; при H=0 → ±90° (геометрический предел).
- **Ночной базис** (`get_nighttime_baseline`): среднее компоненты в окне 00:00–04:00 UT (окно **замкнутое**, `(22,6)` = 22:00–05:59). Возвращает `{baseline, n_samples, n_finite, night_hours, window}`; `baseline=None` при пустом окне (не NaN — чтобы «нет данных» не приняли за 0).
- **Дельта** (`calculate_delta`): `max − min` по всему фрейму (float) либо по блокам `window=N` (DataFrame: `block, start, end, min, max, delta`).
- **Аномалия** (`calculate_anomaly`): `B − baseline_value`; при `baseline_value=None` — **намеренная ошибка** `missing_baseline`, а не молчаливый ноль.
- **dH/dt** (`calculate_dH_dt`): первая разность `B[i] − B[i−1]` в нТл/мин, первая строка NaN; `samples_per_minute` масштабирует для субминутных данных.
- **d²H/dt²** (`calculate_dst_proxy`, базовая): `np.diff(values, n=2)`, первые две строки NaN — величина, из которой строится Dst.

**Безопасный DSL формул** (`evaluate_formula`, `available_columns`, `formula_functions`):

- **Операторы:** `+ − * / % **`, унарные `+ −`.
- **Функции (19):** `sqrt, sin, cos, tan, asin, acos, atan, atan2, abs, exp, log, log10, hypot, degrees, radians, power, minimum, maximum, sign`.
- **Константы:** `pi`, `e`, `nan`.
- **Колонки:** собственные числовые колонки фрейма + производные `H/F/D/I` + `X/Y/Z`.
- **Безопасность:** AST-whitelist — разрешены только `Expression, BinOp, UnaryOp, Call, Constant, Load, Name`. **Отсутствуют** `Attribute` (нет `os.system`), `Subscript` (нет `X['__class__']`), `Lambda`, `Compare`, `IfExp`, `Dict/List/Set`. Формула **парсится, но никогда не исполняется как код**.

### 4.Б. Физические метрики (`physics_engine.py`)

11 метрик доступны через `apply_physics_formula` (плюс алиас `dH_dt` → `gic`).

### 4.1. `sq_baseline` → `extract_sq_baseline`

- **Смысл:** тихий уровень Sq-вариации (спокойный ночной интервал).
- **Формула:** медиана (по умолчанию) компонент за 00:00–04:00 UT.
- **Вход:** X/Y/Z (или H/F строится через `hypot`).
- **Выход:** колонки `X_sq/Y_sq/Z_sq/H_sq/F_sq` (константы).
- **Пример:** `metric='sq_baseline'`.

### 4.2. `deltas` → `calculate_deltas`

- **Смысл:** отклонение от Sq-базовой линии.
- **Формула:** `dX = X − X_sq`; `dH = hypot(dX, dY)`; `F_delta = hypot(dH, dZ)`.
- **Вход:** X/Y/Z (+ `*_sq`, иначе материализуется).
- **Выход:** `dX/dY/dZ/dH/F_delta`.

### 4.3. `gic` / `dH_dt` → `calculate_gic_proxy`

- **Смысл:** стандартный прокси GIC — скорость изменения горизонтального поля.
- **Формула:** `dH/dt` в нТл/мин (первая разность на минутной сетке).
- **Вход:** H (или X,Y).
- **Выход:** колонка `dH_dt` + `attrs['dH_dt_peak_abs_nT_per_min']`.

### 4.4. `dst_proxy` → `calculate_dst_proxy`

- **Смысл:** локальный упрощённый Dst (кольцевой ток).
- **Формула:** `dH = H − H_sq`; `dst_proxy = dH · cos(latitude)`.
- **Вход:** H (+ широта станции), Sq-базис.
- **Выход:** колонка `dst_proxy` (нТл; отрицательна при буре).
- **Пример:** `{"df_descriptor":"raw:irt:2024-09-12","metric":"dst_proxy"}`.

### 4.5. `detect_ssc` → `detect_ssc`

- **Смысл:** внезапное начало (SSC) по резкому росту H.
- **Формула:** `dH/dt > threshold_nT_per_min` (по умолчанию 5 нТл/мин), рост.
- **Выход:** `ssc_flag`, `dH_dt`, `attrs['ssc_events']` (до 10 событий).

### 4.6. `ey_coupling` → `calculate_coupling_function`

- **Смысл:** межпланетное электрическое поле, движущее геомагнитную активность.
- **Формула:** `E_y = V · B_z` (мВ/м, при V в км/с и Bz в нТл).
- **Вход:** внешний источник (`external_params`: скаляры `{'V':450,'Bz':-5}`, хэндл `{'solar_wind':'sw:omni:...'}` или вектора из `align_solar_wind`).
- **Выход:** константа или колонка `E_y` + `attrs['ey_coupling']`.

### 4.7. `burton_coupling` → `calculate_burton_coupling`

- **Смысл:** модель кольцевого тока Burton et al. (1975) + одношаговый прогноз Dst.
- **Формула:**

```
dDst/dt = a · sqrt(Np) · V · Bs − Dst/τ
Bs = −Bz  (Bz < 0, иначе 0)
a = 0.005,  τ = 8 ч
```

- **Вход:** `external_params`: `Np` (см⁻³), `V` (км/с), `Bz` (нТл), `Dst` (нТл).
- **Выход:** колонка `burton_coupling` (нТл/ч) + `attrs['burton']` (прогноз).

### 4.8. `akasofu_epsilon` → `calculate_akasofu_epsilon`

- **Смысл:** параметр ε Акасофу — энерговклад солнечного ветра в магнитосферу.
- **Формула:**

```
ε = V · B_total² · l0² · sin⁴(θ/2)
B_total = sqrt(Bx² + By² + Bz²)
θ = arctan(By / Bz)
l0 = 7 RE
```

- **Вход:** `external_params`: `V` (км/с), `Bx/By/Bz` (нТл).
- **Выход:** колонка `akasofu_epsilon` (Вт) + `attrs['akasofu']`.

### 4.9. `peak_dhdt` → `calculate_peak_dhdt`

- **Смысл:** пиковое |dH/dt| в скользящем окне, градуированное для риска GIC.
- **Формула:** `peak_dhdt = max(|dH/dt|)` в окне `window_seconds` (по умолчанию 10).
- **Вход:** H (или X,Y).
- **Выход:** колонки `dH_dt`, `peak_dhdt` + `attrs['peak_dhdt']` (порог: `low`<50, `medium`≤200, `high`>200 нТл/мин).
- **Пример:** `{"df_descriptor":"raw:irt:2024-09-12","metric":"peak_dhdt","window_seconds":10}`.

### 4.10. `substorm_onset` → `detect_substorm_onset`

- **Смысл:** начало суббури по устойчивому падению SML (авроральный электроджет).
- **Формула:** первый сэмпл самого раннего пробега, где `dSML/dt < −drop_threshold_nT_per_min` (по умолчанию −50) держится ≥ `min_duration_min` (3 мин).
- **Вход:** колонка `SML` (SuperMAG `data_type='indices'`).
- **Выход:** колонка `sml_dh_dt` + `attrs['substorm']` (`onset_time`, `sml_drop`, `confidence`).

### 4.11. `eej_index` → `calculate_eej_index`

- **Смысл:** сила экваториального электроджета.
- **Формула:** `EEJ = H(экватор) − H(референс)`; референс — станция в 6–8° по широте.
- **Вход:** экваториальный фрейм + `external_params={'reference': <фрейм с H>}`.
- **Выход:** колонка `eej_index` + `attrs['eej']` (mean/peak/peak_time).
- **Пример:** `{"df_descriptor":"raw:aae:2023-09-12","metric":"eej_index","external_params":{"reference":"raw:abg:2023-09-12"}}`.

### 4.В. Анализ и координаты

**`geomag_analyzer.py`** (`get_statistics`, `detect_anomalies`, `calculate_derived_components`, `compute_derivatives`):

- **Статистика** (`get_statistics`): mean/min/max/std по колонкам (X/Y/Z/F/H), `digits=2`.
- **Аномалии** (`detect_anomalies`): выбросы по порогу сигм — `|x − mean| > sigma_threshold · std` (по умолчанию 3σ).
- **Производные** (`compute_derivatives`): dH/dt и dF/dt по времени.
- **Производные компоненты** (`calculate_derived_components`): добавляет H/D/I (делегирует в `add_derived_components`).

**`geomag_coords.py`** (`get_station_geomagnetic_coords`, `calculate_mlt`, `group_stations_by_mlt`):

- **Географические → геомагнитные** (`geographic_to_geomagnetic`): наклонный диполь, полюс 80.7°N, 72.7°W — **первый порядок**, не значение поля. Возвращает `model` в payload, чтобы аппроксимацию не приняли за IGRF/AACGM.
- **Расстояние** (`haversine_distance`): `d = 2R·arcsin(sqrt(sin²(Δφ/2) + cosφ₁cosφ₂sin²(Δλ/2)))`.
- **MLT** (`magnetic_local_time`): магнитное локальное время = UT + (geomag_lon/15), обёрнутое в [0,24).
- **Группировка** (`group_by_mlt`): станции по MLT-секторам (`bin_hours`).
- **Смещения для overlay** (`calculate_offsets_for_overlay`, `apply_offsets`): `base_offset=100` нТл на станцию; смещение — **только для отображения**, в данные не пишется.

**⚠️ Почему геомагнитная широта важнее географической:** амплитуда возмущения масштабируется с геомагнитной широтой. Две станции на 65°N и 67°N географически могут иметь геомагнитные широты 75° и 58° и вести себя по-разному — сортировка для overlay ведётся именно по геомагнитной широте.

---

## РАЗДЕЛ 5: ВОЗМОЖНОСТИ ГРАФИКОВ

### 5.1. Типы графиков

| Инструмент | Что рисует |
|-----------|-----------|
| `plot_components` | N колонок одной серии (X/Y/Z/F/H/D/I, `dst_proxy`, `Dst`, `Kp`) |
| `plot_comparison` | один компонент двух дней (два хэндла) |
| `plot_overlay` | один компонент нескольких станций, с nT-смещениями |
| `plot_space_weather_comparison` | двойная ось: Земля (H/dst_proxy/dH_dt) vs ветер (Bz/V/By/Np) |

### 5.2. Несколько линий на одном графике

- `plot_components(components=['H','D'])` — **несколько компонент** одной станции.
- `plot_overlay(stations=['IRT','NVS','YAK'], dates=['2024-09-12'])` — **несколько станций**, один компонент, с `offsets={'IRT':0,'NVS':150}`.

### 5.3. Формат вывода и интерактивность

- Графики — **Plotly HTML**, сохраняются в `plots/` (DEFAULT_OUTPUT_DIR).
- Отображаются в Gradio через **`srcdoc`-iframe**: документ инлайнится в `gr.HTML`, поэтому zoom/pan/hover работают внутри фрейма. Если бы вставляли через `innerHTML`, `<script>` бы не выполнился — потому и выбран `srcdoc`.
- Восстановление фигуры из HTML (`_figure_from_html`) — собственная сканирующая машина, распознаёт `Plotly.newPlot`, `Plotly.react`, `Plotly.animate`, `Plotly.restyle` и встроенный JSON.

### 5.4. Скачивание

- Панель отображает до `MAX_PLOTS_IN_PANEL = 12` графиков.
- Не поместившиеся — остаются на диске и видны в `gr.Files` («Файлы графиков»); `_fallback_note` сообщает, что график сохранён в `plots/`.
- Весь проект упаковывается в ZIP (`export_project`).

### 5.5. Внутреннее устройство плоттера (`geomag_plotter.py`, 1112 строк)

**Библиотека:** только **Plotly** (`go.Figure`, `go.Scatter`/`go.ScatterGl`) — matplotlib нигде не используется.

**Выход:** `DEFAULT_OUTPUT_DIR = <модуль>/plots`. Запись — `fig.write_html(path, include_plotlyjs=..., full_html=True, auto_open=False, config={"displaylogo": False, "scrollZoom": True, "toImageButtonOptions": {"format":"png","scale":2}})`; возвращает **абсолютный путь строкой**. `include_plotlyjs=True` → самодостаточный HTML (~4 МБ); `'cdn'` → ~10 КБ, но нужен интернет.

**Публичные функции плоттера (5):**

| Функция | Что рисует |
|---------|-----------|
| `plot_components` | N компонент одной серии, общая ось |
| `plot_comparison` | два дня на оси «время суток» (якорь `TIME_OF_DAY_ANCHOR = 2000-01-01`) |
| `plot_overlay` | один компонент нескольких станций, смещения, UT/LT/MLT |
| `plot_space_weather_comparison` | двойная ось: Земля (левая, `#d62728`) vs ветер (правая `yaxis2`, пунктир) |
| `plot_magnetogram` | 4 панели X/Y/Z/F, общая ось времени |

**⚠️ `plot_magnetogram` НЕ подключён как инструмент** — функция есть в модуле, но не в `TOOL_SCHEMAS`/`HANDLERS`. Агенту недоступна (только через прямой вызов из Python).

**Константы:** `GL_THRESHOLD = 5000` (порог перехода на WebGL/даунсэмплинг), палитры `_SERIES_COLOR`, `_COMPARISON_COLORS`, `_OVERLAY_COLORS`, таблицы единиц `_PLOT_UNITS` (компоненты + физические колонки + солнечный ветер).

**UT/LT/MLT в overlay:** `time_system` задаёт **только подпись оси** (`_TIME_SYSTEM_LABEL`); реальный сдвиг времени даёт `time_shifts[station]` в часах (`xs = time_of_day + timedelta(hours=shift)`). Смещения (`offsets`) — **только display**, в данные не пишутся.

---

## РАЗДЕЛ 6: МАССОВАЯ ОБРАБОТКА

### 6.1. `create_task_matrix`

Планирует большую задачу как **ОДНУ матрицу, сгруппированную ПО СТАНЦИЯМ**: одна задача на станцию со всеми её датами событий. Порог: `MATRIX_THRESHOLD = 15` юнитов; выше — матрица, ниже — обычные вызовы.

- `events: str[]`, `stations: str[]`, `actions: str[]`, `project_name: str`
- Возвращает `matrix_path`.

Слова-действия: `fetch`, `calc_HDI`, `plot` (мапятся на хендлеры; полные имена тоже работают). По умолчанию: `fetch_observatory_data, calculate_derived_components, plot_components`.

### 6.2. `process_matrix_batch`

- `matrix_path: str`, `run_until_complete: bool`, `batch_size: int` (по умолчанию 5).
- `run_until_complete=True` — обрабатывает **ВСЕ** станции за один вызов, **автоматически собирает ZIP** и возвращает сводку, начинающуюся с `МАССОВАЯ ОБРАБОТКА ЗАВЕРШЕНА`. Отдельный `export_project` не нужен.
- `run_until_complete=False` — только `batch_size` станций (отладка).

### 6.3. Экспорт ZIP

`export_project(name)` упаковывает дерево проекта `<станция>/<дата>/`, каждый CSV, каждый HTML и манифесты (`manifest.json` — версия данных, время загрузки, коммит агента).

### 6.4. Пример

```
1) create_task_matrix(events=["2024-05-10","2024-05-11"], stations=["IRT","NVS","YAK"], actions=["fetch","calc_HDI","plot"], project_name="May storm")
2) process_matrix_batch(matrix_path="<...>", run_until_complete=True)
3) Сводка: "МАССОВАЯ ОБРАБОТКА ЗАВЕРШЕНА" + путь к ZIP.
```

**Правило для модели:** НИКОГДА не вызывать `process_matrix_batch` в цикле и НИКОГДА не вызывать `create_task_matrix` повторно.

### 6.5. Внутреннее устройство матрицы (`agent_core.py`)

**Группировка (подтверждено):** `_handle_create_task_matrix` (4332–4412) создаёт **ОДНУ задачу на станцию**: `{"id": "station_<CODE>", "station_code": ..., "events": [ВСЕ даты], "actions": [все действия], "status": "pending"}`. Задача 10×26 становится матрицей из 26 задач — станция×событие **никогда не перечисляется**.

**Слова-действия** (`MATRIX_ACTION_ALIASES`, 1230–1240):

| Слово | Хендлер |
|-------|---------|
| `fetch` | `fetch_observatory_data` |
| `calc_hdi` / `calculate_hdi` / `hdi` / `derive` | `calculate_derived_components` |
| `plot` | `plot_components` |
| `stats` / `statistics` | `get_statistics` |
| `anomalies` | `detect_anomalies` |

Полные имена хендлеров тоже проходят (проверка против `HANDLERS`).

**Авто-ZIP (подтверждено, 4575–4591):** при `run_until_complete and remaining == 0` хендлер **сам вызывает `_handle_export_project`** и кладёт путь в `archive`. Комментарий в коде: после тяжёлого прогона 7B-модель надёжно забывает шаг `export_project`, поэтому ZIP собирает Python.

**`fetch_many` (3989–4109):**
- Бюджет `max_downloads` по умолчанию `MAX_BATCH_DOWNLOADS = 8` — **один потолок на весь вызов**, не на станцию; `spent` увеличивается **до** запроса (зависший запрос тоже съедает бюджет).
- Упавшая станция → в `failed` (с `station/date/error/message`), цикл `continue` — **сбой не прерывает пакет**.
- Исчерпан бюджет → остаток в `refused` с причиной `"download budget exhausted"`.
- Возвращает `ok=True` даже при частичном сбое; `summary` = «N downloaded, N failed, N refused of N attempts (budget N)».
- Ошибки только на невалидном входе: `missing_stations`, `invalid_date`, `nothing_to_fetch`.

### 6.6. Внутреннее устройство хранилища проектов (`project_store.py`, 609 строк)

**Раскладка на диске:**
```
projects/<name>/
    manifest.json
    <station>/<date>/
        manifest.json
        raw.csv  derived.csv  *.html
```
Корни: `PROJECTS_ROOT = <модуль>/projects`, `EXPORTS_ROOT = <модуль>/exports`.

**`manifest.json` проекта:** `{project, stations, dates, artifacts, created_at, updated_at, agent_commit}`.
**`manifest.json` листа:** `{project, station, date, kind, file, rows, columns, data_version, source, downloaded_at, agent_commit, geomagnetic_coords, written_at}`.

**Ключевые функции:** `create_project` (идемпотент), `list_projects`, `save_artifact`, `export_project`, `project_tree`, `iter_manifests`, `Workspace` (обёртка с `create`/`save`/`save_chart`), `parse_slot` (`"raw:irt:2024-09-10"` → `("irt","2024-09-10")`), `agent_commit` (git HEAD через subprocess, кэшируется).

**`export_project`:** обновляет манифест по диску, затем `zipfile.ZipFile(..., ZIP_DEFLATED)` — **перезаписывает, а не досыпает** (старый член не сохраняется). Пишет в `exports/<name>.zip`. В архив входит всё дерево проекта (манифесты, CSV, HTML). Возвращает `{ok, project, path, bytes, size, files, stations, dates, manifest}`.

**Ошибки:** `invalid_project_name`, `project_not_found`, `unreadable_manifest`, `invalid_artifact_path`, `artifact_write_failed`. `project_tree`/`list_projects`/`iter_manifests` при отсутствии проекта возвращают **пустую коллекцию**, не ошибку.

### 6.7. Тесты и E2E (`scenarios_e2e.py`, 857 строк)

Помимо обычного pytest-набора, есть **E2E-прогон против реальной Qwen** — каждый сценарий это естественно-языковой вопрос, модель сама выбирает инструменты.

- **Запуск:** `python scenarios_e2e.py [--scenario compare] [--model <gguf>] [--dry-run] [--verbose]`.
- **10 сценариев:** `one_day`, `compare`, `medians`, `compare_uppercase`, `three_days`, `median_F`, `range_F`, `compare_range_F`, `project_batch`, `three_days_clean`.
- **Что проверяет `check()`:** порядок `must_call`; подсчёт ошибок (`unrecovered_failures` — сбой закрыт только если позже успешен тот же субъект); различающиеся хэндлы в `df1`/`df2`; каноничность (нижний регистр) хэндлов; `min_plots` + существование файлов; «Построено графиков: N»; имена трейсов (`expect_traces`/`expect_trace_dates`, парсит `Plotly.newPlot` из HTML); запрещённые коды в ответе (`unknown_frame` и т.п.); диапазон nT в ответе.
- **`_ShoutingBrain`** имитирует модель, которая передаёт хэндлы В ВЕРХНЕМ РЕГИСТРЕ (`raw:IRT:...`), проверяя устойчивость к регистру.

### 6.8. Offline-режим (`GEOMAG_OFFLINE=1`)

Каждый сетевой хендлер имеет детерминированный синтетический аналог: `_offline_frame`, `_offline_solar_wind` (быстрый поток с южным Bz), `_offline_indices` (Dst/Kp с депрессией ~12:00 UT), `_offline_noaa`, `_offline_iswa_forecast`, `_offline_supermag`. Тесты сети (`-m network`) исключаются флагом `-m "not network"`.

### 6.9. Тестовый набор (553 теста в 12 файлах)

| Файл | Тестов | Строк | Что покрывает |
|------|-------:|------:|---------------|
| `test_agent_core.py` | 183 | 2 960 | Схемы, реестр хендлеров, парсер tool-calls, Scratchpad, матрица, проекты, `run_agent` (5 helper-двойников «мозга») |
| `test_scenarios_e2e.py` | 45 | 787 | Сценарии E2E-набора (см. §6.7) в офлайн-режиме |
| `test_handles_and_plots.py` | 44 | 793 | Жизненный цикл хэндлов и восстановление фигур из HTML |
| `test_iswa_loader.py` | 49 | 869 | ISWA HAPI/WMS, парсинг, decommissioned |
| `test_supermag_loader.py` | 37 | 513 | SuperMAG sites/indices, лимит 32 дня, логин |
| `test_physics_engine.py` | 30 | 305 | 11 физических метрик |
| `test_projects.py` | 27 | 374 | `project_store`, манифесты, ZIP |
| `test_intermagnet_loader.py` | 25 | 288 | GIN/Kyoto, чанки, состояния, коды ошибок |
| `test_geomag_tools.py` | 67 | 603 | Базовые инструменты математики/статистики |
| `test_noaa_swpc_loader.py` | 22 | 392 | Алерты/прогноз/вспышки NOAA |
| `test_space_weather_loader.py` | 14 | 241 | OMNI + Dst/Kp |
| `test_handle_boundary.py` | 10 | 212 | Границы работы с хэндлами |

**Итог офлайн-прогона:** `python -m pytest -m "not network"` → **846 passed, 1 skipped, 22 deselected** (~102 с). Сетевые тесты вынесены под маркер `network`.

---

## РАЗДЕЛ 7: ИНТЕРФЕЙС GRADIO

### 7.1. Компоненты

- **Чат:** `gr.Chatbot` (type="messages" на 4.x/5.x, адаптация `_chatbot_kwargs`), `gr.Textbox` + `gr.Button("Спросить")`.
- **Примеры:** `gr.Examples` с 3 образцами («График H», «Сравнение двух дней», «Поиск аномалий»).
- **Панель графиков:** `MAX_PLOTS_IN_PANEL = 12` слотов `gr.HTML` (srcdoc-iframe).
- **Файлы:** `gr.Files` (`files_view`) — скачивание графиков, не поместившихся в панель.
- **Журнал:** `gr.Accordion("Журнал вызовов инструментов")` + `gr.Markdown` (`tool_log`).
- **Проекты:** `gr.Dropdown`, кнопка «Обновить», read-only `gr.Dataframe`, кнопка «Собрать ZIP», `gr.DownloadButton`.
- **Безопасность:** `sm_logon = gr.Textbox(type="password")`.

### 7.2. srcdoc

Графики — самодостаточные Plotly-документы, инлайнятся в `<iframe srcdoc="...">`. Это единственный способ сохранить интерактивность: `innerHTML` не выполняет скрипты. `_iframe` экранирует payload через `html.escape(quote=True)`.

### 7.3. Безопасность (SuperMAG Logon)

- Логин вводится в поле `type="password"`.
- Устанавливается в `os.environ["SUPERMAG_LOGON"]` только на время запроса в `submit`.
- В историю чата и в журнал инструментов **не попадает**.
- Ограничение: живёт в процессе между запросами (см. §1.5).

### 7.4. Обработка хода

`chat_and_plot` возвращает плоский кортеж: `[messages, *12 plot-slots, tool_log, plots_state, file_paths, cleared_textbox]`. `split_result` разбирает его по именам (`1 + MAX_PLOTS_IN_PANEL + HANDLER_TAIL`), чтобы счёт не «протухал».

История: `USE_HISTORY_CONTEXT=True`, `HISTORY_TURNS=4` — краткая выжимка предыдущих реплик добавляется к запросу, поскольку `run_agent` stateless.

---

## РАЗДЕЛ 8: ОГРАНИЧЕНИЯ И ЧЕСТНЫЕ ПРЕДУПРЕЖДЕНИЯ

### 8.1. Недоступность данных (definitive lag)

Definitive-данные для экваториальных станций (например AAE) **отстают на месяцы**. Инструмент `calculate_eej_index` содержит явную инструкцию:

> `CRITICAL FALLBACK`: при ошибке `no_data` за текущий год — **автоматически повторить** с `data_type='auto'` или историческим годом (например `'2023-09-12'`). Не сдаваться.

Аналогично `data_type='auto'` в `fetch_many` проходит цепочку публикационных состояний.

### 8.2. ISWA WMS декомиссирован

Публичный WMS-слой `ground_mag_perturbation` удалён при миграции NASA CCMC (проверено 2026-10-20). `get_iswa_perturbation_map` вернёт `service_decommissioned`. **Замена:** `fetch_supermag_data(data_type='indices')` (SME/SML/SMU) для глобальной картины суббурь. Ответ намеренно сохраняется целиком (`FALLBACK_HINT_MAX_CHARS = 800`), потому что подсказка содержит два разных исправления.

### 8.3. Что делать, если агент не может выполнить задачу

- Ошибки самокорректируются: `unknown_tool`/`malformed_tool_call` несут образец правильного вызова; `unknown_frame` — актуальный хэндл.
- При исчерпании бюджета — ответ с пометкой «_(лимит вызовов инструментов исчерпан, ответ неполный)_», сохранившиеся графики всё равно показываются.
- Продолжение формулируется как follow-up (агент stateless, но UI добавляет контекст).

### 8.4. Ограничения LLM

- **7B-модель**: склонна к галлюцинациям имён инструментов (`calculate_dst_proxy`), их отсекает `_self_correction_hint`.
- **Контекст**: длинный payload матрицы переполняет окно → предусмотрено короткое замыкание.
- **Время ответа**: `MAX_TOOL_CALLS=16`, `MAX_ROUNDS=18`; многостанционные задачи идут через матрицу, а не через LLM-цикл.
- **Детерминизм**: `TEMPERATURE=0.1` — наука, не творчество.
- **Гарантия чисел**: правила промпта запрещают выдумывать цифры — только из инструментов.

---

## РАЗДЕЛ 9: ПРИМЕРЫ РЕАЛЬНЫХ ЗАПРОСОВ

### Пример 1. Базовый анализ (одна станция, один день)

```python
from agent_core import run_agent
run_agent("Скачай данные по станции IRT за 10 сентября 2024, посчитай H и построй график.")
```
Маршрут: `fetch_observatory_data(IRT, 2024-09-10)` → `plot_components(H)`.

### Пример 2. Сравнение двух дней

```python
run_agent("Сравни 10 и 11 сентября 2024 по станции IRT: построй наложенные графики H.")
```
Маршрут: два `fetch_observatory_data` → `plot_comparison(df1, df2, component='H')`.

### Пример 3. Оценка риска GIC

```python
run_agent("Оцени риск GIC по станции IRT за 12 сентября 2024.")
```
Маршрут: `apply_physics_formula(metric='peak_dhdt', window_seconds=10)` → уровень `low/medium/high` из `attrs['peak_dhdt']`.

### Пример 4. Сложные физические величины (Burton, Akasofu)

```python
run_agent("Посчитай функцию Бёртона и энерговклад Акасофу для бури 12 сентября 2024.")
```
Маршрут: `fetch_solar_wind` + `fetch_geomagnetic_indices` → `apply_physics_formula(metric='burton_coupling')` → `apply_physics_formula(metric='akasofu_epsilon')`.

### Пример 5. Массовая обработка с ZIP

```python
run_agent("Обработай 3 станции (IRT, NVS, YAK) за 10–12 мая 2024: скачай, посчитай H и построй графики, упакуй в ZIP.")
```
Маршрут: `create_task_matrix` → `process_matrix_batch(run_until_complete=True)` → сводка «МАССОВАЯ ОБРАБОТКА ЗАВЕРШЕНА» + ZIP.

---

## ПРИЛОЖЕНИЕ: Карта бюджета промпта

- `len(SYSTEM_PROMPT) = 3199` символов
- `N_CTX = 16384`, бюджет `est_tokens ≤ N_CTX//2 = 8192`
- Динамический выбор схем: `TOOLS_PER_TASK = 6`
- Сводка кадра: `FRAME_SUMMARY_MAX_CHARS = 400`
- Подсказка-фолбэк: `FALLBACK_HINT_MAX_CHARS = 800`
- Панель графиков: `MAX_PLOTS_IN_PANEL = 12`, `HANDLER_TAIL = 4`
- Порог матрицы: `MATRIX_THRESHOLD = 15`, `MATRIX_BATCH_SIZE = 5`

**Итог:** 29 инструментов, 11 физических метрик, 6 источников данных, 4 стадии, 1 мета-инструмент физики. Система спроектирована вокруг одного принципа — **маленькая модель не должна помнить много**: память вынесена в `Scratchpad`, физика — в один `apply_physics_formula`, объём — в матрицы.

---

## ПРИЛОЖЕНИЕ 2: Область и методология аудита (честно)

**Прочитано полностью и построчно:**
- `agent_core.py` — все 29 схем инструментов, все 29 хендлеров, промпт, `Scratchpad`, селектор, матрица, проекты, `run_agent` (6 918 строк)
- `physics_engine.py` — все 13 функций + константы
- `geomag_math.py` — компоненты, дельты, производные, базис, DSL
- `geomag_analyzer.py`, `geomag_coords.py` — статистика, аномалии, координаты, MLT
- `geomag_plotter.py` — все 5 публичных функций, механизм `write_html`
- `project_store.py` — раскладка, манифесты, ZIP, `Workspace`
- `gradio_app.py` — UI, `chat_and_plot`, srcdoc, credential isolation
- `scenarios_e2e.py` — 10 сценариев, `check()`, `_ShoutingBrain`
- `agent_state.py`, `agent_planner.py` — `SessionState`/`TaskStep`, `Planner`/`LLMPlanner`
- Все 5 загрузчиков (`intermagnet`, `space_weather`, `supermag`, `noaa_swpc`, `iswa`) — конечные точки, константы, цепочки состояний, парсер IAGA-2002, чанки, кэш, коды ошибок (§2.7)
- Тестовый набор — 12 файлов, 553 теста, структура и покрытие (§6.9)

**Проверено фактами, а не по памяти:** количество инструментов (29 = 29 хендлеров), 11 метрик `_PHYSICS_METRICS`, `len(SYSTEM_PROMPT)=3199`, `N_CTX=16384`, `MAX_PLOTS_IN_PANEL=12`, наличие `_handle_*` для каждого тула, `plot_magnetogram` отсутствует в `TOOLS_BY_NAME`; 553 теста и 846 прошедших в офлайн-прогоне; коды ошибок загрузчиков выведены из исходников, а не из docstring.

**Что изучено менее глубоко (известное ограничение):**
- Тела отдельных тестов не читались построчно — структура, число и покрытие проверены (счётчики, прогон), но не каждый ассерт.
- Точные URL отдельных конечных точек NOAA за пределами найденных констант.

**Найденные расхождения (исправлены в этом манифесте):**
1. В первой редакции Раздела 4 были **только 11 физических метрик** — пропущены 11 базовых формул `geomag_math.py` и слой анализа/координат. Добавлено.
2. Не была отмечена **коллизия `dst_proxy`** (2-я производная vs. упрощённый Dst). Добавлено.
3. `plot_magnetogram` существует, но **не подключён как инструмент** — теперь указано явно.
4. Отсутствовали плоттер, `project_store`, E2E-набор и offline-режим. Добавлено (Разделы 5.5, 6.5–6.8).

**Мёртвый код (не входит в аудит, но отмечен):** `gradio_app.py:921–927` — блок `btn_download.click(...)` после `if __name__ == "__main__"`, ссылается на несуществующие имена, никогда не исполняется.
