import pytest
import agent_core as ac
from chat_session import SessionMemory
from test_chat_session import seed


def request(text):
    store = ac.FrameStore()
    raw = seed(store)
    ac._handle_derive({'df_descriptor': raw, 'components': ['H', 'D', 'I']}, store)
    return SessionMemory().prepare(text, store), store


@pytest.mark.parametrize('verb', ['Вызови', 'Выполни', 'Сначала получи'])
def test_dependent_baseline_must_be_current_and_exact(verb):
    r, store = request(verb + ' calculate_baseline H mode=night для derived:irt:2024-09-10. Затем calculate_derived_math metric=anomaly, baseline_value возьми из первого результата, а не поиск выбросов.')
    assert 'detect_anomalies' not in r.calculation_tools
    args = {'df_descriptor': 'derived:irt:2024-09-10', 'component': 'H'}
    r.observe_result('calculate_baseline', args, {'baseline': 18260.928843530255})
    assert r.check_call('calculate_derived_math', dict(args, metric='anomaly', baseline_value=15.2), store)
    assert not r.check_call('calculate_derived_math', dict(args, metric='anomaly', baseline_value=18260.928843530255), store)


@pytest.mark.parametrize('ending', ['без повторного расчёта H/D/I', 'calculate_derived_components не вызывай', 'не вызывай calculate_derived_components'])
def test_plot_does_not_require_forbidden_calculation(ending):
    r, store = request('Построй H из derived:irt:2024-09-10; ' + ending)
    assert not r.wants_derived
    assert 'calculate_derived_components' not in r.calculation_tools
    assert r.check_call('calculate_derived_components', {'df_descriptor': 'derived:irt:2024-09-10'}, store)


def test_outliers_are_not_plain_statistics():
    r, store = request('Для derived:irt:2024-09-10 найди статистические выбросы H с sigma_threshold=3.')
    assert r.bound_statistics_args(store) is None
    assert 'detect_anomalies' in r.calculation_tools


def test_derived_with_explanation_has_its_tool():
    selected = ac.select_tools_for_query('Рассчитай H/D/I и укажи формулы, единицы и handle.')
    assert any(s['function']['name'] == 'calculate_derived_components' for s in selected)


@pytest.mark.parametrize('query, tool', [
    ('Экспортируй проект science_run в архив report.zip', 'export_project'),
    ('Перечисли существующие проекты через list_projects', 'list_projects'),
    ('Вызови get_noaa_space_weather_context с query_type=alerts', 'get_noaa_space_weather_context'),
    ('Получи текущий прогноз ISWA', 'get_iswa_forecast'),
])
def test_completion_requires_real_operation(query, tool):
    r, store = request(query)
    assert tool in r.required_operations
    assert r.missing([], store)
    assert r.missing([{'tool': tool, 'ok': False}], store)
    assert not r.missing([{'tool': tool, 'ok': True, 'note': 'Actual result'}], store)
    assert r.should_use_facts([])
    assert 'Actual result' in r.facts([{'tool': tool, 'ok': True, 'note': 'Actual result'}], store)


@pytest.mark.parametrize('query', ['Экспортируй проект science_run в report.zip',
                                 'Вызови get_noaa_space_weather_context с query_type=alerts'])
def test_model_claim_without_execution_is_not_reported_as_success(query):
    r, store = request(query)
    fake = ac.ScriptedBrain([{'content': 'Готово, выдуманный результат 12345.'}] * 4)
    result = ac.run_agent(query, brain=fake, store=store, contract=r, verbose=False)
    assert not result['ok']
    assert '12345' not in result['text']


def test_negative_export_does_not_require_export():
    r, _ = request('Перечисли проекты; не экспортируй их. export_project не вызывай.')
    assert 'export_project' not in r.required_operations


@pytest.mark.parametrize('metric', ['gic', '"gic"', "'gic'"])
def test_quoted_metric_rejects_previous_metric(metric):
    r, store = request('Вызови apply_physics_formula metric=' + metric + ' для raw:irt:2024-09-10')
    assert r.check_call('apply_physics_formula', {'metric': 'deltas', 'df_descriptor': 'raw:irt:2024-09-10'}, store)


def test_export_glob_selecting_day_directories_keeps_their_files(tmp_path):
    import project_store as ps
    for station in ('irt', 'bou'):
        folder = tmp_path/station/'2025-03-02'
        folder.mkdir(parents=True)
        (folder/'raw.csv').write_text('X,Y,Z\n3,4,12\n')
    literal, error = ps._resolve_export_folders(tmp_path, {'Stations': ['irt', 'bou']})
    assert error is None
    globbed, error = ps._resolve_export_folders(tmp_path, {'Stations': ['irt/*', 'bou/*']})
    assert error is None
    assert globbed == literal
    assert len(globbed) == 2


@pytest.mark.parametrize('ending', ['без скачивания и повторного расчёта H/D/I',
                                 'без загрузки и расчета H/D/I'])
def test_coordinated_prohibition(ending):
    r, store = request('Используй derived:irt:2024-09-10. Построй H ' + ending)
    assert r.no_fetch and r.no_derive and not r.wants_derived
    assert r.check_call('calculate_derived_components', {'df_descriptor': r.handle}, store)
    assert r.check_call('fetch_observatory_data', {'station_code': 'IRT'}, store)
    assert not r.check_call('plot_components', {'df_descriptor': r.handle, 'components': ['H']}, store)


@pytest.mark.parametrize('text', ['Построй H из raw:irt:2024-09-10.',
                                'Рассчитай H/D/I для raw:irt:2024-09-10 без загрузки.'])
def test_no_implicit_ban_on_necessary_derivation(text):
    r, store = request(text)
    assert not r.no_derive
    assert not r.check_call('calculate_derived_components', {'df_descriptor': r.handle}, store)


@pytest.mark.parametrize('text,tool', [
    ('Сгруппируй IRT, BOU и ABG по MLT на 2024-09-10T12:00:00Z, bin_hours=3.', 'group_stations_by_mlt'),
    ('Через group_stations_by_mlt сгруппируй IRT и BOU.', 'group_stations_by_mlt'),
    ('Через plot_overlay наложи H IRT и BOU за 2024-09-10, time_system=MLT.', 'plot_overlay'),
])
def test_coordinate_claim_is_not_execution(text, tool):
    r, store = request(text)
    assert tool in r.required_operations
    result = ac.run_agent(text, brain=ac.ScriptedBrain([{'content': 'Готово: выдуманные группы и файл.'}] * 4),
                          store=store, contract=r, verbose=False)
    assert not result['ok']
    assert 'выдуманные' not in result['text']


def test_formula_result_is_allowed_only_after_successful_bound_computation():
    r, store = request('Через evaluate_custom_formula вычисли выражение hypot(X,Y)-H для derived:irt:2024-09-10.')
    args = {'df_descriptor': r.handle, 'formula': 'hypot(X,Y)-H'}
    payload, _ = ac._handle_custom_formula(args, store)
    result_handle = payload['frame_handle']
    followup = {'df_descriptor': result_handle, 'components': ['FORMULA']}
    assert r.check_call('get_statistics', followup, store)  # an old slot is not lineage
    r.observe_result('evaluate_custom_formula', args, {'error': 'failed', 'frame_handle': result_handle})
    assert r.check_call('get_statistics', followup, store)
    r.observe_result('evaluate_custom_formula', args, payload)
    assert not r.check_call('get_statistics', followup, store)
    assert not r.check_call('plot_components', followup, store)
    assert r.check_call('get_statistics', {'df_descriptor': 'raw:irt:2024-09-10'}, store)
    other, _ = request(r.query)
    assert other.check_call('get_statistics', followup, store)  # no cross-turn exemption


def test_source_recovery_only_blocked_when_explicitly_forbidden():
    base = 'Примени apply_physics_formula metric=substorm_onset к raw:irt:2024-09-10.'
    r, store = request(base + ' Не подменяй источник.')
    assert r.check_call('fetch_supermag_data', {'start_date': '2024-09-10'}, store)
    r, store = request('Рассчитай substorm_onset за 2024-09-10. Если нужен SML, загрузи его из SuperMAG.')
    assert not r.check_call('fetch_supermag_data', {'start_date': '2024-09-10'}, store)
    store.put('idx:supermag:2024-09-10', store.get('raw:irt:2024-09-10').copy())
    assert not r.check_call('apply_physics_formula', {'df_descriptor': 'idx:supermag:2024-09-10', 'metric': 'substorm_onset'}, store)


def test_group_payload_is_preserved_in_grounded_answer():
    from chat_session import capture_calculation
    r, store = request('Сгруппируй IRT и BOU по MLT.')
    payload = {'bins': {'00-03': ['BOU'], '18-21': ['IRT']}, 'bin_hours': 3, 'model': 'tilted_dipole'}
    entry = {'tool': 'group_stations_by_mlt', 'ok': True, 'note': 'grouped 2 stations'}
    capture_calculation(entry, entry['tool'], payload)
    text = r.facts([entry], store)
    assert '00-03' in text and 'BOU' in text and 'tilted_dipole' in text


@pytest.mark.parametrize('wording', ['get_statistics для колонки FORMULA', 'статистику FORMULA', 'медиану FORMULA'])
def test_formula_then_statistics_requires_both_real_calls(wording):
    from test_chat_session import call
    query = ('Через evaluate_custom_formula вычисли выражение hypot(X,Y)-H для derived:irt:2024-09-10. '
             f'Затем получи {wording} из фактически полученного handle.')
    r, store = request(query)
    assert r.components == ('FORMULA',)
    formula_call = call('evaluate_custom_formula', df_descriptor=r.handle, formula='hypot(X,Y)-H')
    result = ac.run_agent(query, brain=ac.ScriptedBrain([formula_call] + [{'content': 'Готово'}] * 4),
                          store=store, contract=r, verbose=False)
    assert not result['ok']
    assert 'get_statistics' in result['text']
    r, store = request(query)
    # A pre-existing same-name result does not satisfy this new computation.
    ac._handle_custom_formula({'df_descriptor': r.handle, 'formula': 'X+100'}, store)
    result = ac.run_agent(query, brain=ac.ScriptedBrain([
        call('get_statistics', df_descriptor='formula:irt:2024-09-10', components=['FORMULA']),
        formula_call,
        call('get_statistics', df_descriptor='formula:irt:2024-09-10', components=['FORMULA']),
        {'content': 'Готово'},
    ]), store=store, contract=r, verbose=False)
    assert result['tool_calls'][0]['ok'] is False
    stats = [e for e in result['tool_calls'] if e['tool'] == 'get_statistics' and e['ok']]
    assert len(stats) == 1
    assert stats[0]['statistics']['FORMULA']['median'] == 0
    assert not r.missing(result['tool_calls'], store)


def test_formula_summary_alone_does_not_require_separate_statistics():
    r, _ = request('Через evaluate_custom_formula вычисли выражение hypot(X,Y)-H для derived:irt:2024-09-10. Укажи min/max/mean.')
    assert not r.wants_stats
