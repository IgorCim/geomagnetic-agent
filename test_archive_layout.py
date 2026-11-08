"""Archive contents, not model prose, are the acceptance oracle."""
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import project_store as ps
import agent_core as ac
from chat_session import SessionMemory
from chat_session import Request, explicit_chart_pairs
from test_chat_session import seed


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, 'EXPORTS_ROOT', tmp_path / 'exports')
    ws = ps.Workspace(root=tmp_path / 'projects')
    ws.create('science')
    root = ps.project_dir('science', ws.root)
    paths = []
    for station, day in [('irt', '2024-09-10'), ('bou', '2024-09-11')]:
        for component in 'XY':
            relative = f'{station}/{day}/{component}_INTERMAGNET.html'
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'<html>{station} {day} {component}</html>')
            paths.append(relative)
    return ws, root, paths


def export(ws, folders, **kwargs):
    return ps.export_project(ws.project, root=ws.root, folders=folders, **kwargs)


def test_explicit_pair_export_rejects_duplicate_day_without_banning_reuse(project):
    ws, root, sources = project
    query = ('Загрузи INTERMAGNET IRT и BOU за 2024-09-10 и 2024-09-11. '
             'Для каждой из четырёх пар станция-дата построй отдельный HTML-график X.')
    request = Request(query)
    store = ac.FrameStore(workspace=ws)
    folders = {'Обзор': [sources[0], sources[2]], 'Обзор/Детали': [sources[2]]}
    assert request.check_call('export_project', {'folders': folders}, store)
    for station, day in [('bou', '2024-09-10'), ('irt', '2024-09-11')]:
        relative = f'{station}/{day}/X.html'
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('chart')
        folders['Обзор'].append(relative)
    assert request.check_call('export_project', {'folders': folders}, store) is None
    assert Request('Положи один график в две папки').check_call('export_project', {'folders': folders}, store) is None
    assert not explicit_chart_pairs('Загрузи INTERMAGNET IRT с 2024-09-10 по 2024-09-11. Для каждой пары построй график.')
    assert not explicit_chart_pairs('Загрузи INTERMAGNET IRT и BOU за 2024-09-10 и 2024-09-11.')


def test_handler_exact_files_go_directly_to_numbered_folders(project):
    ws, root, sources = project
    store = ac.FrameStore(workspace=ws)
    result, _ = ac._handle_export_project({'folders': {'1': [sources[0]], '2': [sources[1]], '3': [sources[2]]}}, store)
    assert result['ok'], result
    with ZipFile(result['path']) as z:
        assert {p for p in z.namelist() if '/' in p} == {'1/X_INTERMAGNET.html', '2/Y_INTERMAGNET.html', '3/X_INTERMAGNET.html'}
        assert z.read('3/X_INTERMAGNET.html') == (root / sources[2]).read_bytes()


@pytest.mark.parametrize('compact', [False, True])
def test_nested_mixed_files_and_subfolders_keep_index_provenance(project, compact):
    ws, root, sources = project
    result = export(ws, {'.': [sources[0]], 'Report': [sources[1]], 'Report/Comparison': [sources[2], sources[3]]}, folder_layout='direct', compact=compact)
    assert result['ok'], result
    expected = {'X_INTERMAGNET.html': sources[0], 'Report/Y_INTERMAGNET.html': sources[1],
                'Report/Comparison/X_INTERMAGNET.html': sources[2], 'Report/Comparison/Y_INTERMAGNET.html': sources[3]}
    with ZipFile(result['path']) as z:
        manifest = json.loads(z.read('export_manifest.json'))
        assert manifest['source_files'] == expected
        assert set(manifest['files']) == set(z.namelist())
        rows = list(csv.DictReader(io.StringIO(z.read('index.csv').decode())))
        assert {r['archive_path'] for r in rows} == set(expected)
        for destination, source in expected.items():
            assert z.read(destination) == (root / source).read_bytes()


def test_same_basename_from_different_stations_is_not_lost(project):
    ws, root, sources = project
    selected = [sources[0], sources[2]]
    first = export(ws, {'Together': selected}, folder_layout='direct', archive_name='first')
    second = export(ws, {'Together': selected[::-1]}, folder_layout='direct', archive_name='second')
    assert first['ok'] and second['ok']
    assert first['source_files'] == second['source_files']
    assert len(first['source_files']) == 2
    with ZipFile(first['path']) as z:
        for destination, source in first['source_files'].items():
            assert destination.count('/') == 1
            assert z.read(destination) == (root / source).read_bytes()


def test_unicode_folder_names_preserve_spaces_and_mixed_children(project):
    ws, root, sources = project
    result = export(ws, {'Обзор': [sources[0]], 'Обзор/Второй день': [sources[2]]}, folder_layout='direct')
    assert result['ok'], result
    with ZipFile(result['path']) as z:
        assert z.read('Обзор/X_INTERMAGNET.html') == (root / sources[0]).read_bytes()
        assert z.read('Обзор/Второй день/X_INTERMAGNET.html') == (root / sources[2]).read_bytes()


@pytest.mark.parametrize('folder', ['../bad', '/absolute', 'A/../../B', 'A/CON', 'A//B', 'A/./B', 'C:/bad', '_plotly/scripts'])
def test_unsafe_nested_destinations_fail_before_mutation(project, folder):
    ws, root, sources = project
    before = (root / 'manifest.json').read_bytes()
    result = export(ws, {folder: [sources[0]]}, folder_layout='direct')
    assert ps.is_error(result), result
    assert (root / 'manifest.json').read_bytes() == before


def test_file_folder_conflict_and_missing_selection_do_not_replace_zip(project):
    ws, root, sources = project
    initial = export(ws, {'ok': [sources[0]]}, folder_layout='direct')
    original = Path(initial['path']).read_bytes()
    for folders in ({'Report': [sources[0]], 'Report/X_INTERMAGNET.html/Sub': [sources[1]]},
                    {'Report': [sources[0], 'missing.html']}):
        result = export(ws, folders, folder_layout='direct')
        assert ps.is_error(result), result
        assert Path(initial['path']).read_bytes() == original


def test_same_table_followup_uses_confirmed_table_without_model():
    store = ac.FrameStore()
    handle = seed(store, 'bou')
    memory = SessionMemory(last_good=handle, last_inputs=[handle])
    query = 'Из той же таблицы получи минимум, максимум и медиану X инструментом. Не скачивай повторно.'
    request = memory.prepare(query, store)
    assert request.bound_statistics_args(store) == {'df_descriptor': handle, 'components': ['X']}
    assert ac.run_bound_statistics(request, store, verbose=False)['ok']
    memory.last_inputs.append(seed(store, 'irt'))
    assert memory.prepare(query, store).clarification


def test_export_recovery_is_scoped_to_same_archive_and_destinations():
    failed = {'tool': 'export_project', 'ok': False, 'error': 'empty_export_folder', 'project': 'p',
              'arguments': {'name': 'p', 'archive_name': 'report.zip', 'folders': {'1': ['X.html']}}}
    success = {'tool': 'export_project', 'ok': True, 'export_result': {'project': 'p', 'filename': 'report.zip',
               'folders': {'1': 1}, 'source_files': {'1/X.html': 'irt/day/X.html'}}}
    text = ac._with_tool_failures('ZIP создан.', [failed, success])
    assert 'Не удалось выполнить' not in text and 'после исправления' in text
    for changes in ({'project': 'other'}, {'filename': 'other.zip'}, {'folders': {'other': 1}}, {'source_files': {}}):
        unrelated = {**success, 'export_result': {**success['export_result'], **changes}}
        assert 'Не удалось выполнить' in ac._with_tool_failures('ZIP создан.', [failed, unrelated])
    assert 'Не удалось выполнить' in ac._with_tool_failures('Нет ZIP.', [success, failed])
