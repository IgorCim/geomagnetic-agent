# Каталог первичных случаев для 29 инструментов

Статус всех случаев: PLANNED_NOT_EXECUTED. Это план, не успешный прогон.

Шаблоны с {PROJECT}, {MATRIX_PATH} и другими переменными нельзя отправлять как есть. Перед живым запуском тестовый runner должен подставить уникальные имена и реальные handle из выполненных зависимостей. После этого пользователю выдаются отдельные готовые сообщения, без указаний проверяющему внутри prompt.

Это первичные карточки, а не полное покрытие: режимы, негативные и граничные случаи раскрыты в QA_MASTER_PLAN.md. Router dry-run — структурная проверка шаблона без Qwen; она не подтверждает исполнение или поведение planner.

## TOOL-01: fetch_observatory_data

Зависимости: TOOL-02.

```text
Загрузи INTERMAGNET IRT за 2024-09-10. Укажи точный handle, источник, число строк и фактический тип публикации.
```

Проверяет runner: источник и UTC; не выдумывать definitive.

Target доступен при standalone dry-run шаблона: True.

## TOOL-02: create_project

Зависимости: SETUP-00.

```text
Создай проект {PROJECT}. Укажи фактическое имя каталога и путь внутри корня проектов.
```

Проверяет runner: каталог и manifest существуют; допустимая нормализация отражена.

Target доступен при standalone dry-run шаблона: True.

## TOOL-03: fetch_many

Зависимости: TOOL-02.

```text
В проект {PROJECT} загрузи IRT и BOU за 2024-09-10 одним пакетным запросом, max_downloads=2. Перечисли фактические таблицы и ошибки.
```

Проверяет runner: два station/day либо точный partial; отдельные handle.

Target доступен при standalone dry-run шаблона: True.

## TOOL-04: list_projects

Зависимости: TOOL-02.

```text
Перечисли существующие проекты через list_projects. Не создавай новые проекты.
```

Проверяет runner: список совпадает с разрешёнными каталогами на диске.

Target доступен при standalone dry-run шаблона: True.

## TOOL-05: export_project

Зависимости: TOOL-03, TOOL-14.

```text
Экспортируй проект {PROJECT} в архив qa_report.zip. Внутри создай папку Stations для irt и bou, сортировка date,station. Укажи фактический путь архива.
```

Проверяет runner: реальный ZIP, безопасные члены, manifest, индекс, хеши и все ожидаемые файлы.

Target доступен при standalone dry-run шаблона: True.

## TOOL-06: create_task_matrix

Зависимости: SETUP-00.

```text
Создай одну матрицу IRT и BOU × 2024-09-10 и 2024-09-11, действия fetch_observatory_data, calculate_derived_components, plot_components, проект {MATRIX_PROJECT}. Верни путь матрицы.
```

Проверяет runner: кодовый контракт создания отдельно; Gradio может запустить весь матричный workflow.

Target доступен при standalone dry-run шаблона: True.

## TOOL-07: process_matrix_batch

Зависимости: TOOL-06.

```text
Продолжи обработку существующей матрицы {MATRIX_PATH}, run_until_complete=True. Матрицу заново не создавай. Укажи покрытие и архив.
```

Проверяет runner: четыре ячейки; resume/idempotency; контролируемый partial без stale artifacts.

Target доступен при standalone dry-run шаблона: True.

## TOOL-08: calculate_derived_components

Зависимости: TOOL-01.

```text
Для {RAW_IRT_D0} вычисли H/D/I, без повторной загрузки. Укажи новый handle и единицы компонентов.
```

Проверяет runner: построчный независимый H/D/I, неизменность исходной таблицы.

Target доступен при standalone dry-run шаблона: True.

## TOOL-09: get_statistics

Зависимости: TOOL-03.

```text
Получи статистику Y для BOU за 2024-09-10, без повторной загрузки. Покажи медиану, count и пропуски.
```

Проверяет runner: один get_statistics в прямом пути; компонента Y; числа и ddof сверены независимо.

Target доступен при standalone dry-run шаблона: True.

## TOOL-10: detect_anomalies

Зависимости: TOOL-08.

```text
Для {DERIVED_IRT_D0} найди статистические выбросы H с sigma_threshold=3. Укажи порог и число; не объявляй их автоматически бурями.
```

Проверяет runner: правильная маска и краевые случаи постоянного/пустого ряда.

Target доступен при standalone dry-run шаблона: True.

## TOOL-11: calculate_derived_math

Зависимости: TOOL-08.

```text
Для {DERIVED_IRT_D0} вычисли размах H через calculate_derived_math, metric=delta, без новой загрузки. Укажи окно и единицы.
```

Проверяет runner: каждый режим delta/anomaly/dH_dt отдельно; численные эталоны и шаг времени.

Target доступен при standalone dry-run шаблона: True.

## TOOL-12: calculate_baseline

Зависимости: TOOL-08.

```text
Для {DERIVED_IRT_D0} вычисли baseline H в UTC-окне 00–04, mode=night, night_hours=[0,4]. Назови выбранные строки и единицы.
```

Проверяет runner: night/full; границы и окно через полночь; UTC не объявлен местной ночью.

Target доступен при standalone dry-run шаблона: True.

## TOOL-13: evaluate_custom_formula

Зависимости: TOOL-01.

```text
Для {RAW_IRT_D0} вычисли безопасную формулу hypot(X,Y). Укажи фактический новый handle. Исходные данные не меняй.
```

Проверяет runner: независимая численная проверка; отдельная серия разрешённых и запрещённых AST.

Target доступен при standalone dry-run шаблона: True.

## TOOL-14: plot_components

Зависимости: TOOL-08.

```text
Построй H из {DERIVED_IRT_D0} без новой загрузки. Имя файла выбери автоматически. Укажи реально созданный файл.
```

Проверяет runner: traces=значения выбранной таблицы; UTC, единицы, пропуски, фактическое имя.

Target доступен при standalone dry-run шаблона: True.

## TOOL-15: plot_comparison

Зависимости: TOOL-08, SETUP-IRT-D1.

```text
Сравни H IRT за 2024-09-10 и 2024-09-11 на одном графике, используй {DERIVED_IRT_D0} и {DERIVED_IRT_D1} без новой загрузки. Нужны две подписанные кривые.
```

Проверяет runner: два разных дня/источника данных в traces, не две подписи одной кривой.

Target доступен при standalone dry-run шаблона: True.

## TOOL-16: get_station_geomagnetic_coords

Зависимости: SETUP-00.

```text
Покажи геомагнитные координаты станции IRT через get_station_geomagnetic_coords и назови использованное приближение.
```

Проверяет runner: сверка координат/модели; отсутствие утверждения IGRF/AACGM.

Target доступен при standalone dry-run шаблона: True.

## TOOL-17: calculate_mlt

Зависимости: TOOL-16.

```text
Вычисли магнитное местное время MLT для IRT в 2024-09-10T12:00:00Z. Укажи модель и нормализацию суток.
```

Проверяет runner: 0/24, округление минут, референсный расчёт для той же модели.

Target доступен при standalone dry-run шаблона: True.

## TOOL-18: group_stations_by_mlt

Зависимости: SETUP-00.

```text
Сгруппируй IRT, BOU и ABG по MLT на 2024-09-10T12:00:00Z, bin_hours=3. Укажи все группы и модель.
```

Проверяет runner: каждая доступная станция ровно один раз; границы bins и неизвестная станция.

Target доступен при standalone dry-run шаблона: True.

## TOOL-19: calculate_local_time

Зависимости: TOOL-16.

```text
Вычисли местное солнечное время IRT в 2024-09-10T23:30:00Z по долготе. Не подменяй его гражданским часовым поясом.
```

Проверяет runner: UTC+longitude/15, корректный переход даты.

Target доступен при standalone dry-run шаблона: True.

## TOOL-20: apply_physics_formula

Зависимости: SCIENCE-GATE, METRIC-SPECIFIC-INPUTS.

```text
Вычисли apply_physics_formula metric={METRIC} для {GROUND_HANDLE}, используя {WIND_HANDLE} и {REFERENCE_HANDLE}, только когда они нужны этой метрике. Назови единицы и границы применимости.
```

Проверяет runner: отдельная карточка на каждую из 11 метрик, независимый oracle; Ey сейчас известный FAIL.

Target доступен при standalone dry-run шаблона: False.

## TOOL-21: calculate_eej_index

Зависимости: SETUP-EEJ-PAIR, SCIENCE-GATE.

```text
Для научно проверенной пары {EEJ_EQUATORIAL_STATION} и {EEJ_REFERENCE_STATION} получи EEJ за {EEJ_DAY}. Назови пару, время, baseline и ограничения метода.
```

Проверяет runner: пара заранее обоснована; timestamps совмещены; IRT не используется как ложный экваториальный пример.

Target доступен при standalone dry-run шаблона: True.

## TOOL-22: plot_overlay

Зависимости: TOOL-03.

```text
Наложи H IRT и BOU за 2024-09-10, time_system=UT, offsets IRT=0 и BOU=100. Используй имеющиеся таблицы; объясни, что смещение только для отображения.
```

Проверяет runner: UT/LT/MLT отдельно; offset не меняет таблицы/статистику.

Target доступен при standalone dry-run шаблона: True.

## TOOL-23: plot_space_weather_comparison

Зависимости: TOOL-08, TOOL-24.

```text
Сопоставь H из {DERIVED_IRT_D0} с Bz и V из {WIND_D0}, без новой загрузки. Построй сравнение земли и солнечного ветра и укажи время и единицы панелей.
```

Проверяет runner: совпадение дат, правильные каналы, alignment/gaps и панели.

Target доступен при standalone dry-run шаблона: True.

## TOOL-24: fetch_solar_wind

Зависимости: SETUP-00.

```text
Загрузи солнечный ветер OMNI за 2024-09-10. Укажи dataset, handle, число строк, координаты IMF и единицы.
```

Проверяет runner: метаданные/fill, GSM/GSE, источник, фактическое временное покрытие.

Target доступен при standalone dry-run шаблона: True.

## TOOL-25: fetch_geomagnetic_indices

Зависимости: SETUP-00.

```text
Получи реальные индексы Kyoto Dst/Kp за 2024-09-10. Укажи источник, handle, время и исходное разрешение Kp. Не рассчитывай dst_proxy.
```

Проверяет runner: официальный ряд не заменён proxy; расширение Kp не названо новыми измерениями.

Target доступен при standalone dry-run шаблона: True.

## TOOL-26: get_noaa_space_weather_context

Зависимости: SETUP-00.

```text
Получи текущие NOAA alerts. Отдели наблюдение, предупреждение и прогноз; укажи фактическое окно данных.
```

Проверяет runner: три режима alerts/forecast/solar_events; архивное окно отдельно от текущих продуктов.

Target доступен при standalone dry-run шаблона: True.

## TOOL-27: fetch_supermag_data

Зависимости: SETUP-SUPERMAG.

```text
Запроси SuperMAG indices SME/SML/SMU за 2024-09-10. При отказе покажи реальную причину, не заменяй их Kyoto и не раскрывай логин.
```

Проверяет runner: sites/indices, допустимая длина окна; missing logon отдельно от HTTP; integration может быть BLOCKED_EXTERNAL.

Target доступен при standalone dry-run шаблона: True.

## TOOL-28: get_iswa_perturbation_map

Зависимости: SETUP-00.

```text
Запроси карту ISWA magnetic perturbation на 2024-09-10T12:00:00Z. Если текущий клиент её не поддерживает, честно сообщи отказ без выдуманного изображения.
```

Проверяет runner: ветка сейчас отключена; контракт ошибки может PASS, реальная карта остаётся BLOCKED/UNSUPPORTED.

Target доступен при standalone dry-run шаблона: True.

## TOOL-29: get_iswa_forecast

Зависимости: SETUP-00.

```text
Получи текущий прогноз ISWA. Отдели прогноз Kp от наблюдений Dst, укажи источники и горизонт времени.
```

Проверяет runner: проверить оба фактических HAPI-набора, kp_kind, частичный отказ и отсутствие ложного Dst-прогноза.

Target доступен при standalone dry-run шаблона: True.
