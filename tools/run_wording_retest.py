"""Wording diagnostics for the aaebae5 combined run; no production changes."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def build_plan():
    original = runpy.run_path(str(ROOT/'tools/run_chat_acceptance.py'))['build_plan']()
    cards = {c['id']: c for p in original for c in p['cases']}
    def old(key):
        return dict(cards[key])
    def case(key, prompt, source):
        return {'id': key, 'prompt': prompt, 'original_case': source}
    setup = [old('TOOL-01'), old('A01')]
    formula_cases = []
    for key in ['D08', 'D09', 'D10', 'D11', 'D12', 'D13']:
        formula_cases.append(case(key+'-WORDING', cards[key]['prompt'] +
            ' Значения min/max/mean и n_finite возьми непосредственно из результата evaluate_custom_formula. '
            'Дополнительный get_statistics не нужен. Не используй выражение или слово result как имя колонки.', key))
    negative = []
    for number in range(1, 10):
        key = f'E{number:02d}'
        negative.append(case(key+'-WORDING', cards[key]['prompt'] +
            ' Это отрицательная проверка: ожидаемый отказ инструмента является правильным результатом. '
            'Выполни ровно одну попытку с исходным выражением. После отказа сообщи его код и причину и закончи ответ. '
            'Не исправляй выражение, не повторяй вызов и не вызывай статистику.', key))
    return [
        {'id': 'wording_plots', 'cases': [old('TOOL-01'), old('A01'), old('CONTEXT-05'),
            case('HF-WORDING', 'Из raw:bou:2024-09-10 построй компоненты H и F через plot_components. '
                 'F уже является компонентой таблицы, calculate_derived_components для F не вызывай. '
                 'Повторная загрузка не нужна.', 'CONTEXT-12'),
            case('LT-WORDING', 'Вызови plot_overlay: stations=["IRT","BOU"], dates=["2024-09-10"], '
                 'component="H", time_system="LT", offsets={"IRT":0,"BOU":100}. '
                 'Таблицы уже загружены. Не загружай их повторно и не вызывай calculate_mlt. '
                 'Построй наложение и объясни, что смещения только для отображения.', 'D06')]},
        {'id': 'wording_formulas', 'cases': setup + formula_cases + [
            case('BASELINE-WORDING', 'Для derived:irt:2024-09-10 вызови calculate_baseline: component="H", '
                 'mode="night", night_hours=[0,4]. Дождись результата. Затем вызови calculate_derived_math '
                 'для этой же таблицы: metric="anomaly", component="H", baseline_value — точное число '
                 'из только что полученного результата. Сообщи фактический handle серии H минус baseline. '
                 'Статистика и поиск выбросов не нужны; baseline не угадывай и не округляй.', 'A06/BASELINE-EXPLICIT')]},
        {'id': 'wording_negative', 'cases': [old('TOOL-01')] + negative},
        {'id': 'wording_long_context', 'cases': [old(f'CONTEXT-{i:02d}') for i in range(1,14)] + [
            case('NO-DERIVE-WORDING', 'Построй H через plot_components из существующей таблицы '
                 'derived:irt:2024-09-10. Компонента H уже рассчитана. Запрещены новые загрузки '
                 'и вызовы calculate_derived_components. Если таблицы нет, сообщи об этом.', 'CONTEXT-14')]},
        {'id': 'wording_source_refusal', 'cases': [old('TOOL-01'),
            case('REFUSAL-WORDING', 'Один раз вызови apply_physics_formula metric="substorm_onset" '
                 'для raw:irt:2024-09-10. Это проверка отказа на наземной таблице без индекса SML. '
                 'После отказа сообщи код и причину и остановись. Не вызывай загрузчики, '
                 'не меняй источник и не придумывай время события.', 'C10')]},
        {'id': 'wording_source_recovery', 'cases': [old('TOOL-01'),
            case('ALLOW-SML-WORDING', 'Проверь возможность расчёта substorm_onset за 2024-09-10. '
                 'SML здесь название геомагнитного индекса, не код станции. '
                 'Вызови fetch_supermag_data с data_type="indices", start_date="2024-09-10", '
                 'end_date="2024-09-10", без station_code. Если сервис отказал, сообщи реальный код '
                 'и причину, не повторяй запрос и не вычисляй событие. Только при успешном получении '
                 'SML передай фактический handle индексов в apply_physics_formula metric="substorm_onset". '
                 'Не используй наземную таблицу вместо индексов и не раскрывай логин.', 'ALLOW-SML')]},
    ]


def run(brain):
    plan = build_plan()
    print('Проверка уточнённых формулировок:', sum(len(p['cases']) for p in plan),
          'сообщений;', len(plan), 'независимых чатов.', flush=True)
    print('Отрицательные проверки ожидают честный отказ. RECORDED не означает PASS.', flush=True)
    return runpy.run_path(str(ROOT/'tools/run_chat_acceptance.py'))['run_acceptance'](brain, plan=plan)
