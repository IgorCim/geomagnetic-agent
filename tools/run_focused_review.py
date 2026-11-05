"""Focused live regressions, including permitted recovery and original wording.

Reuses production Gradio handler and evidence capture; no automatic PASS claims.
Each phase is an independent session. Long history is deliberately retained.
"""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def build_plan():
    helper = runpy.run_path(str(ROOT/'tools/run_chat_acceptance.py'))
    original = helper['build_plan']()
    cards = {c['id']: c for p in original for c in p['cases']}

    def card(key):
        return dict(cards[key])

    def extra(key, prompt):
        return {'id': key, 'prompt': prompt}

    raw = 'raw:irt:2024-09-10'
    derived = 'derived:irt:2024-09-10'
    setup = [card('TOOL-01'), card('A01')]
    phases = [
        {'id': '01_permissions', 'cases': setup + [card('CONTEXT-14'),
            extra('ALLOW-DERIVE', f'Для {raw} заново рассчитай H/D/I и построй H. Повторный расчёт разрешён.'),
            extra('ORDINARY-PLOT', f'Построй H из {raw}.')]},
        {'id': '02_mlt', 'cases': setup + [card('CONTEXT-05'), card('TOOL-18'), card('D04'),
                                         card('D05'), card('D06'), card('D07')]},
        {'id': '03_outliers_fresh', 'cases': setup + [card('A08')]},
        {'id': '04_context_outliers', 'cases': [card(f'CONTEXT-{i:02d}') for i in range(1, 15)] + [card('A08')]},
        {'id': '05_formula_chain', 'cases': setup + [card('D08'), card('D11'), card('D12'),
            extra('RESULT-STATS', f'Через evaluate_custom_formula вычисли выражение hypot(X,Y)-H для {derived}. '
                  'Затем получи get_statistics для колонки FORMULA из фактически полученного handle. Исходную таблицу не подменяй.'),
            card('A06'), extra('BASELINE-EXPLICIT', f'Вызови calculate_baseline для {derived}, component=H, mode=night, night_hours=[0,4]. '
                  'Затем вызови calculate_derived_math для той же таблицы, metric=anomaly, component=H, '
                  'baseline_value возьми из первого результата. Нужна серия H минус baseline, а не поиск выбросов.') ]},
        {'id': '06_source_refusal', 'cases': [card('TOOL-01'), card('C10')]},
        {'id': '07_source_recovery', 'cases': [card('TOOL-01'),
            extra('ALLOW-SML', 'Рассчитай substorm_onset за 2024-09-10. Если в загруженных наземных данных нет SML, разрешаю загрузить '
                  'реальные SML из SuperMAG за 2024-09-10 и использовать их. При отказе сервиса покажи реальную причину, не придумывай результат.')]},
    ]
    return phases


def run(brain, groups=None):
    helper = runpy.run_path(str(ROOT/'tools/run_chat_acceptance.py'))
    plan = build_plan()
    if groups is not None:
        unknown = set(groups) - {p['id'] for p in plan}
        if unknown:
            raise ValueError(f'Unknown groups: {sorted(unknown)}')
        plan = [p for p in plan if p['id'] in groups]
    print('Целевой прогон:', sum(len(p['cases']) for p in plan), 'сообщений;', len(plan), 'независимых чатов.')
    print('Отказ SuperMAG не равен отказу маршрутизации. Итог требует проверки журналов и CSV.')
    return helper['run_acceptance'](brain, plan=plan)
