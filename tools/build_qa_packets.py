"""Publish literal QA prompts, preserving stable tool numbers and prerequisites.

Builds documentation only. Does not call Qwen, tools or providers.
"""
from pathlib import Path
import json
import re
from build_qa_inventory import SEEDS

ROOT = Path(__file__).resolve().parents[1]
VALUES = {
    'PROJECT': 'qa_tools_round2', 'MATRIX_PROJECT': 'qa_matrix_round2',
    'MATRIX_PATH': 'projects/qa_matrix_round2/qa_matrix_round2_matrix.json',
    'RAW_IRT_D0': 'raw:irt:2024-09-10', 'DERIVED_IRT_D0': 'derived:irt:2024-09-10',
    'DERIVED_IRT_D1': 'derived:irt:2024-09-11', 'WIND_D0': 'sw:omni:2024-09-10',
    'METRIC': 'sq_baseline', 'GROUND_HANDLE': 'raw:irt:2024-09-10',
    'WIND_HANDLE': 'sw:omni:2024-09-10', 'REFERENCE_HANDLE': 'raw:bou:2024-09-10',
}


def main():
    inventory = json.loads((ROOT/'docs/qa/inventory.json').read_text(encoding='utf-8'))
    lines = ['# Готовые запросы: 29 инструментов', '',
             'Это новая редакция для живого прогона, не отчёт PASS. Копируйте только текст внутри блока. Каждый блок — отдельное сообщение. Номера соответствуют прежнему каталогу, а не порядку выполнения.', '',
             '**После перезапуска Python/сессии старые таблицы в памяти не считаем сохранёнными.** Успешная матрица и её файлы не гарантируют наличие всех таблиц в FrameStore чата. Если подготовка завершилась ошибкой, зависимые запросы не запускайте.', '',
             '## Порядок и подготовка', '',
             '- Базовый пакет: **2 → 1 → 3 → 4 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 5**.',
             '- Для №15 сначала №8, затем подготовка D1 ниже, затем №15.',
             '- Матричный пакет: **6 → 7**. Это проверка существующего workflow: №6 может сразу выполнить матрицу; №7 тогда проверяет повторное продолжение без дублей.',
             '- Координаты/время: **16 → 17 → 18 → 19**; таблицы не нужны.',
             '- Физика: №20 только после №1. Это один режим из 11, не вся физика. №21 пока BLOCKED_SCIENCE_SETUP.',
             '- Наложение: №22 после №3. Земля/ветер: **8 → 24 → 23**.',
             '- №25–29 отдельно; отсутствие логина в №27 проверяет отказ, а не работающий SuperMAG API.', '',
             'Подготовка D1 (только для №15):', '', '```text',
             'Загрузи INTERMAGNET IRT за 2024-09-11 и рассчитай H/D/I. Укажи реальные raw и derived handle и число строк.',
             '```', '',
             'Для другого полного прогона меняйте имена проектов согласованно либо используйте новую изолированную рабочую папку. «Новый диалог» очищает RAM чата, но не удаляет проекты на диске.', '',
             '## Скачивание ZIP', '',
             'В Gradio откройте «Проекты» → «Обновить» → выберите qa_tools_round2 → «Собрать ZIP» → «Скачать проект (ZIP)». Сохранённый файл должен открываться архиватором. Для №5 дополнительно проверьте именно qa_report.zip: кнопка проекта собирает стандартный ZIP и сама по себе не проверяет пользовательскую структуру qa_report.zip.', '',
             '## Запросы', '']
    records = []
    for entry in inventory['tools']:
        name = entry['name']; number = entry['id']
        prompt, deps, oracle = SEEDS[name]
        if name == 'calculate_eej_index':
            lines += [f'### {number}: {name}', '',
                      '**BLOCKED_SCIENCE_SETUP. Не запускать с выдуманной парой.** Нужны обоснованные экваториальная/референсная станции, дата общего покрытия и независимый эталон. IRT не использовать как экваториальную. Это незавершённая проверка, не PASS.', '']
            records.append({'id': number, 'status': 'BLOCKED_SCIENCE_SETUP', 'prompt': None})
            continue
        if name == 'apply_physics_formula':
            prompt = 'Примени apply_physics_formula metric=sq_baseline к raw:irt:2024-09-10, без новой загрузки. Укажи фактический handle, метод, UTC-окно и ограничения baseline. Не называй фиксированное UTC-окно гарантированно спокойной местной ночью.'
        for key, value in VALUES.items():
            prompt = prompt.replace('{'+key+'}', value)
        assert not re.search(r'\{[A-Z][A-Z0-9_]*\}', prompt), (name, prompt)
        lines += [f'### {number}: {name}', '', '```text', prompt, '```', '',
                  'Проверка после ответа (не копировать в чат): ' + oracle + '.', '']
        records.append({'id': number, 'status': 'NOT_RUN', 'prompt': prompt, 'dependencies': deps})
    (ROOT/'docs/qa/TOOL_RUNBOOK.md').write_text('\n'.join(lines), encoding='utf-8')
    (ROOT/'docs/qa/ready_tool_cases.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

    functions = {
        'abs': ('abs(-3)', 3), 'acos': ('acos(1)', 0), 'asin': ('asin(0)', 0),
        'atan': ('atan(0)', 0), 'atan2': ('atan2(0,1)', 0), 'cos': ('cos(0)', 1),
        'degrees': ('degrees(pi)', 180), 'exp': ('exp(0)', 1), 'hypot': ('hypot(3,4)', 5),
        'log': ('log(e)', 1), 'log10': ('log10(100)', 2), 'maximum': ('maximum(2,3)', 3),
        'minimum': ('minimum(2,3)', 2), 'power': ('power(2,3)', 8), 'radians': ('radians(180)', 'pi'),
        'sign': ('sign(-3)', -1), 'sin': ('sin(pi/2)', 1), 'sqrt': ('sqrt(9)', 3), 'tan': ('tan(0)', 0),
    }
    assert set(functions) == set(inventory['dsl_functions'])
    text = ['# Формулы: отдельная программа живых проверок', '',
            '**Статус NOT_RUN / ORACLE_PENDING.** Промпты проверяют доступность и исполнение; численную правильность подтверждают независимые code-oracle, не текст модели. Полный перечень Python-функций хранится в inventory.json: legacy-функции без tool schema недоступны через Gradio.', '',
            '## Подготовка в новой сессии', '',
            'Выполните отдельно №1, №3, №8 и №24 из TOOL_RUNBOOK.md (перед №3 — создание проекта №2). После любой ошибки подготовки остановите зависимый пакет. Используйте те же даты и handle.', '',
            '## A. Производные компоненты, статистика и арифметика', '',
            'Каждый пункт — отдельное сообщение. Проверять фактические вызовы и результаты:', '']
    base = [
        'Для raw:irt:2024-09-10 рассчитай H/D/I. Укажи формулы, единицы и фактический derived handle.',
        'Получи инструментом статистику X/Y/Z/F/H/D/I из derived:irt:2024-09-10, без новой загрузки. Укажи число значений и пропуски.',
        'Рассчитай baseline H для derived:irt:2024-09-10, mode=night, night_hours=[0,4]. Укажи среднее и фактическое окно.',
        'Рассчитай baseline H для derived:irt:2024-09-10, mode=full. Укажи метод.',
        'Рассчитай calculate_derived_math metric=delta, component=H для derived:irt:2024-09-10.',
        'Сначала получи calculate_baseline H mode=night для derived:irt:2024-09-10. Затем рассчитай calculate_derived_math metric=anomaly для той же таблицы, передав baseline_value из результата этого вызова, а не по памяти.',
        'Рассчитай calculate_derived_math metric=dH_dt, component=H для derived:irt:2024-09-10. Укажи временной шаг и единицы.',
        'Найди detect_anomalies H для derived:irt:2024-09-10 с sigma_threshold=3. Не называй статистические выбросы доказанными бурями.',
    ]
    for n, prompt in enumerate(base, 1): text += [f'### A{n:02d}', '', '```text', prompt, '```', '']
    text += ['## B. Все 19 функций DSL', '',
             'Значение справа — эталон проверяющего для строк с конечным X. Не добавлять его в промпт. Допуск назначается отдельно; по одному примеру функция не считается полностью проверенной. Каждое вычисление может перезаписать slot formula:irt:2024-09-10, поэтому сохраняйте результат до следующего.', '']
    for n, (name, (expression, expected)) in enumerate(functions.items(), 1):
        text += [f'### B{n:02d}: {name}', '', '```text',
                 f'Через evaluate_custom_formula вычисли выражение {expression}+0*X для raw:irt:2024-09-10. Укажи точный новый handle и численный результат инструмента, не заменяй вызов устным расчётом.',
                 '```', '', f'Эталон: {expected}.', '']
    text += ['## C. Все 11 физических режимов', '',
             'Каждый — отдельный запрос. Все числа пока требуют независимой научной сверки. Ey имеет известный FAIL масштаба; его ответ не принимать за верный. EEJ без обоснованной пары заблокирован. Метрики gic/dst_proxy и детекторы нельзя объявлять измеренными токами или официальными индексами.', '']
    for n, metric in enumerate(inventory['physics_registry'], 1):
        schema_metric = 'ssc_flag' if metric == 'detect_ssc' else metric
        if metric == 'eej_index':
            text += [f'### C{n:02d}: eej_index', '', '**BLOCKED_SCIENCE_SETUP**: как №21 tool runbook. Промпт с фиктивной парой не выдаётся.', '']; continue
        wind = ' Передай solar_wind_handle=sw:omni:2024-09-10.' if metric in {'ey_coupling', 'burton_coupling', 'akasofu_epsilon'} else ''
        text += [f'### C{n:02d}: {metric}', '', '```text',
                 f'Примени apply_physics_formula metric={schema_metric} к raw:irt:2024-09-10.{wind} Укажи фактический результат инструмента, новый handle, единицы и ограничения. Не подменяй метрику другой.', '```', '']
    text += ['## D. Остальные преобразования', '',
             'Координаты/LT/MLT/binning — №16–19; UT/LT/MLT overlay — три отдельных варианта №22. Преобразование единиц, fill, resampling, временные производные, aliases и legacy Python-функции требуют code-тестов из QA_SCIENCE_LEDGER.md.', '',
             'Дополнительно раскрыть операторы +,-,*,/,%,**, унарные знаки; pi/e/nan; домены sqrt/log/acos, деление на ноль, NaN/Inf, граничные значения, запрещённые AST и лимиты. Константные DSL-примеры выше — только первый проход, не вся линейка формул.', '']
    (ROOT/'docs/qa/FORMULA_RUNBOOK.md').write_text('\n'.join(text), encoding='utf-8')
    print('Published 29 stable tool cards (28 literal prompts, EEJ blocked), 8 math, 19 DSL and 11 physics-mode cards.')


if __name__ == '__main__':
    main()
