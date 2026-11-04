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
