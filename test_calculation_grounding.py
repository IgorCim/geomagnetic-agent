"""User-observed false calculation successes: real handlers, scripted model replies."""
import json
import numpy as np
import pandas as pd
import pytest

import agent_core as ac
from chat_session import SessionMemory


def seeded():
    store = ac.FrameStore()
    frame = pd.DataFrame({'timestamp': pd.to_datetime(['2024-09-10T00:00Z', '2024-09-10T01:00Z',
                                                     '2024-09-10T04:00Z', '2024-09-10T05:00Z']),
                          'X': [3., 0., 5., 7.], 'Y': [4., 12., 0., 0.], 'Z': [12.] * 4})
    store.put('raw:irt:2024-09-10', frame)
    ac._handle_derive({'df_descriptor': 'raw:irt:2024-09-10', 'components': ['H', 'D', 'I']}, store)
    return store


def call(name, **args):
    return {'content': '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'}


class Brain:
    def __init__(self, replies):
        self.replies = iter(replies)

    def chat(self, *a, **kw):
        return next(self.replies, {'content': 'Готово: 123.45, baseline:irt:2024-09-10; всё проверено.'})


def run(query, replies, store=None):
    store = store or seeded()
    request = SessionMemory().prepare(query, store)
    return ac.run_agent(query, brain=Brain(replies), store=store, contract=request, max_rounds=7), store


@pytest.mark.parametrize('query', [
    'Рассчитай baseline H для derived:irt:2024-09-10, mode=full.',
    'Через evaluate_custom_formula вычисли выражение cos(0)+0*X для raw:irt:2024-09-10.',
    'Примени apply_physics_formula metric=peak_dhdt к raw:irt:2024-09-10.',
    'Рассчитай H/D/I для raw:irt:2024-09-10.',
])
def test_correct_looking_prose_without_call_is_not_success(query):
    result, _ = run(query, [])
    assert not result['ok']
    assert result['error'] == 'request_incomplete'
    assert not result['tool_calls']
    assert '123.45' not in result['text']
    assert 'baseline:irt' not in result['text']


@pytest.mark.parametrize('mode,expected', [('night', 8.5), ('full', 7.25)])
def test_baseline_fact_is_real_window_mean_not_model_daily_mean(mode, expected):
    result, _ = run(f'Рассчитай baseline H для derived:irt:2024-09-10, mode={mode}.', [
        call('calculate_baseline', df_descriptor='derived:irt:2024-09-10', component='H', mode=mode)])
    assert result['ok'], result
    assert result['tool_calls'][0]['calculation']['baseline'] == expected
    assert f'baseline: {expected}' in result['text']
    assert '123.45' not in result['text'] and 'baseline:irt' not in result['text']


def test_baseline_then_anomaly_has_both_tools_and_same_current_baseline():
    query = ('Сначала получи calculate_baseline H mode=night для derived:irt:2024-09-10. '
             'Затем рассчитай calculate_derived_math metric=anomaly для той же таблицы, '
             'передав baseline_value из результата этого вызова, а не по памяти.')
    offered = {t['function']['name'] for t in ac.select_tools_for_query(query)}
    assert {'calculate_baseline', 'calculate_derived_math'} <= offered
    result, store = run(query, [
        call('calculate_baseline', df_descriptor='derived:irt:2024-09-10', component='H', mode='night'),
        call('calculate_derived_math', df_descriptor='derived:irt:2024-09-10', component='H', metric='anomaly', baseline_value=8.5)])
    assert result['ok'], result
    assert [e['tool'] for e in result['tool_calls']] == ['calculate_baseline', 'calculate_derived_math']
    np.testing.assert_allclose(store.get('anomaly:irt:2024-09-10')['ANOMALY'], [-3.5, 3.5, -3.5, -1.5])
    assert '123.45' not in result['text']


def test_anomaly_does_not_accept_forgotten_or_foreign_baseline():
    store = seeded()
    request = SessionMemory().prepare('Сначала получи calculate_baseline, затем рассчитай calculate_derived_math metric=anomaly для derived:irt:2024-09-10.', store)
    args = dict(df_descriptor='derived:irt:2024-09-10', component='H', metric='anomaly', baseline_value=8.5)
    assert request.check_call('calculate_derived_math', args, store)
    request.observe_result('calculate_baseline', args, {'ok': True, 'baseline': 8.5})
    assert request.check_call('calculate_derived_math', args, store) is None
    assert request.check_call('calculate_derived_math', dict(args, baseline_value=7.25), store)


def test_formula_cannot_be_replaced_by_previous_formula():
    store = seeded()
    request = SessionMemory().prepare('Через evaluate_custom_formula вычисли выражение cos(0)+0*X для raw:irt:2024-09-10.', store)
    args = {'df_descriptor': 'raw:irt:2024-09-10', 'formula': 'abs(-3)+0*X'}
    assert request.check_call('evaluate_custom_formula', args, store)
    assert request.check_call('evaluate_custom_formula', dict(args, formula='cos(0) + 0 * X'), store) is None


def test_dsl_result_summary_comes_from_values_even_if_final_prose_is_false():
    result, _ = run('Через evaluate_custom_formula вычисли выражение degrees(pi)+0*X для raw:irt:2024-09-10.', [
        call('evaluate_custom_formula', df_descriptor='raw:irt:2024-09-10', formula='degrees(pi)+0*X')])
    assert result['ok'], result
    assert 'mean: 180.0' in result['text'] and 'min: 180.0' in result['text']
    assert '123.45' not in result['text']


def test_delta_advertises_a_real_math_frame():
    store = seeded()
    result, _ = ac._handle_derived_math({'df_descriptor': 'derived:irt:2024-09-10', 'metric': 'delta', 'component': 'H'}, store)
    assert store.get(result['frame_handle']) is not None
    assert result['delta'] == 7.0
    np.testing.assert_allclose(store.get(result['frame_handle'])['DELTA'], 7.)


def test_explicit_metric_cannot_be_substituted_by_different_physics():
    store = seeded()
    request = SessionMemory().prepare('Примени apply_physics_formula metric=gic к raw:irt:2024-09-10.', store)
    assert request.check_call('apply_physics_formula', {'df_descriptor': 'raw:irt:2024-09-10', 'metric': 'dst_proxy'}, store)


def test_physics_facts_discard_fabricated_range_and_peak():
    result, store = run('Примени apply_physics_formula metric=peak_dhdt к raw:irt:2024-09-10.', [
        call('apply_physics_formula', df_descriptor='raw:irt:2024-09-10', metric='peak_dhdt')])
    assert result['ok'], result
    actual = store.get('derived:irt:2024-09-10').attrs['peak_dhdt']['peak_dhdt_nT_min']
    assert str(actual) in result['text'] and '123.45' not in result['text']
    assert 'отсчётов' in result['text']


def test_stats_HDI_does_not_require_derivation_to_be_run_again():
    store = seeded()
    request = SessionMemory().prepare('Получи статистику H/D/I из derived:irt:2024-09-10, без новой загрузки.', store)
    assert not request.wants_derived
    assert not request.calculation_tools


def test_explanation_of_a_tool_is_not_a_request_to_execute_it():
    request = SessionMemory().prepare('Объясни, что такое calculate_baseline. Не вызывай calculate_baseline.', seeded())
    assert not request.calculation_tools


def test_substorm_with_ground_frame_fails_without_inventing_an_onset():
    result, _ = run('Примени apply_physics_formula metric=substorm_onset к raw:irt:2024-09-10.', [
        call('apply_physics_formula', df_descriptor='raw:irt:2024-09-10', metric='substorm_onset')])
    assert not result['ok']
    assert any(e.get('error') == 'invalid_data_type' for e in result['tool_calls'])
    assert 'SML' in result['text'] and '123.45' not in result['text']


def test_baseline_explicit_window_is_not_replaced():
    store = seeded()
    request = SessionMemory().prepare('Рассчитай baseline H для derived:irt:2024-09-10, night_hours=[0,4].', store)
    assert request.check_call('calculate_baseline', {'df_descriptor': 'derived:irt:2024-09-10', 'night_hours': [1, 5]}, store)


def test_compact_baseline_result_keeps_number_needed_by_followup():
    payload = {'ok': True, 'baseline': 8.5, 'unit': 'nT', 'mode': 'night',
               'available_handles': ['raw:irt:2024-09-10'] * 1000}
    compact = ac._compact_tool_message(payload)
    assert '8.5' in compact and 'baseline' in compact and 'nT' in compact


def test_pure_interpretation_is_not_silently_replaced_by_a_calculation_receipt():
    request = SessionMemory().prepare('Проанализируй геомагнитную активность и объясни ограничения модели.', seeded())
    assert not request.should_use_facts([{'ok': True, 'tool': 'apply_physics_formula'}])
