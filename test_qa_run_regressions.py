"""Regressions from the first user tool-catalog run, not live Qwen acceptance."""
import json
from pathlib import Path

import pandas as pd
import pytest

import agent_core as ac
import gradio_app as ga
import project_store as ps
from chat_session import SessionMemory


@pytest.mark.parametrize("question", [
    "Перечисли существующие проекты через list_projects. Не создавай новые проекты.",
    "Перечислить все проекты", "list_projects", "Покажи доступные проекты",
])
def test_project_listing_is_offered(question):
    assert "list_projects" in {t['function']['name'] for t in ac.select_tools_for_query(question)}


@pytest.mark.parametrize("placeholder", ["PROJECT", "RAW_IRT_D0", "METRIC", "EEJ_EQUATORIAL_STATION"])
def test_unresolved_templates_never_reach_model_or_create_data(placeholder):
    class ForbiddenBrain:
        def chat(self, *a, **kw):
            pytest.fail("Unresolved template reached model")
    store = ac.FrameStore()
    result = ga.make_chat_handler(ForbiddenBrain())(f"Вычисли для {{{placeholder}}}", [], [], store)
    messages, _, _, _, _, _ = ga.split_result(result)
    assert "незаполненные поля" in messages[-1]['content']
    assert store.names() == []


def test_literal_raw_handle_has_real_derived_postcondition():
    store = ac.FrameStore()
    frame = pd.DataFrame({'X': [3.], 'Y': [4.], 'Z': [12.]})
    store.put('raw:irt:2024-09-10', frame)
    request = SessionMemory().prepare("Для raw:irt:2024-09-10 вычисли H/D/I, без повторной загрузки", store)
    assert request.missing([], store)
    result, _ = ac._handle_derive({'df_descriptor': 'raw:irt:2024-09-10', 'components': ['H', 'D', 'I']}, store)
    assert not ac.is_error(result)
    from chat_session import capture_calculation
    entry = {'tool': 'calculate_derived_components', 'ok': True}
    capture_calculation(entry, 'calculate_derived_components', result)
    assert request.missing([], store), 'An old table alone is not proof of a new calculation call'
    assert request.missing([entry], store) == []


def test_fetch_answer_preserves_publication_metadata(monkeypatch):
    frame = pd.DataFrame({'timestamp': pd.date_range('2024-09-10', periods=2, freq='min', tz='UTC'),
                          'X': [3., 3.], 'Y': [4., 4.], 'Z': [12., 12.]})
    frame.attrs = {'source': [{'backend': 'GIN', 'state': 'provisional'}], 'publication_states': ['provisional']}
    monkeypatch.setattr(ac, 'OFFLINE', False)
    monkeypatch.setattr(ac.loader, 'fetch_observatory_data', lambda **kw: frame)
    store = ac.FrameStore()
    payload, _ = ac._handle_fetch({'station_code': 'IRT', 'start_date': '2024-09-10'}, store)
    assert payload['publication_states'] == ['provisional']
    request = SessionMemory().prepare('Загрузи INTERMAGNET IRT за 2024-09-10. Укажи тип публикации.', store)
    answer = request.facts([{'ok': True, 'frame_handle': payload['frame_handle']}], store)
    assert 'provisional' in answer and 'GIN' in answer and 'UTC' in answer
    assert 'definitive' not in answer


@pytest.mark.parametrize('lon,stamp,hm,local_day', [
    (104.45, '2024-09-10T23:30:00Z', '06:28', '2024-09-11'),
    (14.99, '2024-09-10T12:00:00Z', '13:00', '2024-09-10'),
    (-120., '2024-09-10T01:00:00Z', '17:00', '2024-09-09'),
])
def test_mean_solar_time_date_and_minute_carry(monkeypatch, lon, stamp, hm, local_day):
    monkeypatch.setattr(ac.loader, 'get_available_stations', lambda: {'IRT': ('test', 52., lon)})
    result, _ = ac._handle_calculate_local_time({'station_code': 'IRT', 'timestamp': stamp}, ac.FrameStore())
    assert result['lt_hm'] == hm
    assert result['local_mean_solar_datetime'].startswith(local_day)
    assert result['time_kind'] == 'local_mean_solar_time'


def test_solar_time_response_uses_tool_not_civil_time_prose(monkeypatch):
    monkeypatch.setattr(ac.loader, 'get_available_stations', lambda: {'IRT': ('test', 52., 104.45)})
    class Brain:
        def __init__(self): self.n = 0
        def chat(self, *a, **kw):
            self.n += 1
            if self.n == 1:
                return {'content': '<tool_call>' + json.dumps({'name': 'calculate_local_time', 'arguments': {
                    'station_code': 'IRT', 'timestamp': '2024-09-10T23:30:00Z'}}) + '</tool_call>'}
            return {'content': 'Это гражданское время с tilted_dipole.'}
    store = ac.FrameStore()
    query = 'Вычисли местное солнечное время IRT в 2024-09-10T23:30:00Z по долготе.'
    request = SessionMemory().prepare(query, store)
    result = ac.run_agent(query, brain=Brain(), store=store, contract=request)
    assert result['ok']
    assert '2024-09-11T06:27:48' in result['text']
    assert 'tilted_dipole' not in result['text']


def test_zip_download_becomes_visible_and_is_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, 'PROJECTS_ROOT', tmp_path/'projects')
    monkeypatch.setattr(ps, 'EXPORTS_ROOT', tmp_path/'exports')
    ps.create_project('qa')
    result = ga._make_zip('qa')
    assert result['visible'] is True
    import zipfile
    with zipfile.ZipFile(result['value']) as bundle:
        assert bundle.testzip() is None


def test_custom_archives_are_listed_only_for_their_project(tmp_path, monkeypatch):
    import zipfile
    monkeypatch.setattr(ps, 'PROJECTS_ROOT', tmp_path/'projects')
    monkeypatch.setattr(ps, 'EXPORTS_ROOT', tmp_path/'exports')
    ps.create_project('one')
    ps.create_project('two')
    one = ps.export_project('one', archive_name='custom_report')
    two = ps.export_project('two', archive_name='other_report')
    invalid = ps.EXPORTS_ROOT/'broken.zip'
    invalid.write_bytes(b'not a zip')
    with zipfile.ZipFile(ps.EXPORTS_ROOT/'oversized.zip', 'w') as z:
        z.writestr('export_manifest.json', '{' + ' ' * 70000 + '}')
    assert ga._project_archives('one') == [str(Path(one['path']).resolve())]
    assert str(Path(two['path']).resolve()) not in ga._project_archives('one')
    assert ga._project_archives('../one') == []
