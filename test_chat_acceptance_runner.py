"""Runner orchestration tests. Fixture runs are never labelled live acceptance."""
import io
import json
from pathlib import Path
import re
import runpy

import pandas as pd
import pytest

import agent_core as ac
import gradio_app as ga
import project_store as ps

RUNNER = runpy.run_path(str(Path(__file__).parent/'tools/run_chat_acceptance.py'))


def test_plan_reuses_verbatim_prompts_with_dependencies_and_three_sessions():
    plan = RUNNER['build_plan']()
    assert [p['id'] for p in plan] == ['01_tools', '02_context', '03_formulas']
    tools = [c for c in plan[0]['cases'] if not c.get('setup')]
    assert len(tools) == 29
    assert len({c['id'] for c in tools}) == 29
    ready = json.loads((Path(__file__).parent/'docs/qa/ready_tool_cases.json').read_text(encoding='utf-8'))
    assert {c['id']: c['prompt'] for c in tools} == {c['id']: c['prompt'] for c in ready}
    order = [c['id'] for c in plan[0]['cases']]
    for before, after in [('TOOL-02', 'TOOL-03'), ('TOOL-03', 'TOOL-05'), ('TOOL-08', 'TOOL-14'),
                          ('SETUP-IRT-D1', 'TOOL-15'), ('TOOL-24', 'TOOL-23')]:
        assert order.index(before) < order.index(after)
    assert len(plan[1]['cases']) == 20
    formulas = [c for c in plan[2]['cases'] if not c.get('setup')]
    assert len(formulas) == 61
    assert sum(c['id'].startswith('D') for c in formulas) == 13
    assert sum(c['id'].startswith('E') for c in formulas) == 10
    assert sum(c.get('prompt') is None for p in plan for c in p['cases']) == 2
    for phase in plan:
        for card in phase['cases']:
            assert not re.search(r'\{[A-Z][A-Z0-9_]*\}', card.get('prompt') or '')


def test_public_entry_refuses_fixture_and_offline_without_creating_a_run(tmp_path, monkeypatch):
    with pytest.raises(TypeError, match='LlamaBrain'):
        RUNNER['run_acceptance'](object(), output_parent=tmp_path)
    fake = object.__new__(ac.LlamaBrain)
    monkeypatch.setattr(ac, 'OFFLINE', True)
    with pytest.raises(RuntimeError, match='OFFLINE'):
        RUNNER['run_acceptance'](fake, output_parent=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_redaction_across_write_boundaries_and_json():
    console, file = io.StringIO(), io.StringIO()
    redact = RUNNER['Redactor'](['secret_user'])
    tee = RUNNER['Tee'](console, file, redact)
    tee.write('logon=secret_')
    tee.write('user https://example.invalid/?logon=secret_user&n=1\n')
    tee.finish()
    assert 'secret_user' not in file.getvalue()
    assert '[REDACTED]' in file.getvalue()
    assert console.getvalue() == file.getvalue()


def _call(name, **args):
    return {'content': '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'}


def test_real_handler_preserves_history_isolates_sessions_and_saves_actual_results(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, 'OFFLINE', False)
    monkeypatch.setattr(ac.loader, 'fetch_observatory_data', lambda **kw: pd.DataFrame({
        'timestamp': pd.date_range('2024-09-10', periods=2, freq='min'),
        'X': [3., 3.], 'Y': [4., 4.], 'Z': [12., 12.]}))
    class Fixture:
        def __init__(self):
            self.calls = 0
            self.replies = iter([
                _call('fetch_observatory_data', station_code='IRT', start_date='2024-09-10', end_date='2024-09-10'),
                {'content': 'fixture final'},
                _call('calculate_derived_components', df_descriptor='raw:irt:2024-09-10', components=['H','D','I']),
                {'content': 'fixture final'},
            ])
        def chat(self, *a, **kw):
            self.calls += 1
            return next(self.replies)
    full = RUNNER['build_plan']()
    by_id = {c['id']: c for c in full[0]['cases']}
    plan = [{'id': 'first', 'cases': [by_id['TOOL-01'], by_id['TOOL-08']]},
            {'id': 'second', 'cases': [by_id['TOOL-08']]}]
    old = (Path.cwd(), ps.PROJECTS_ROOT, ps.EXPORTS_ROOT, ac.plotter.DEFAULT_OUTPUT_DIR,
           ga.PLOTS_DIR, ga.VERBOSE, ga.run_agent, ga.run_agent_with_planner, ac.run_bound_statistics)
    fixture = Fixture()
    report = RUNNER['_execute_plan'](fixture, plan, tmp_path/'run', scope='FIXTURE_TEST_ONLY')
    assert fixture.calls == 4, 'Second session must reject the unavailable raw handle before model call'
    assert report['status'] == 'RECORDED' and report['acceptance_verdict'] == 'REVIEW_REQUIRED'
    assert 'unknown_frame' in report['phases'][1]['turns'][0]['answer']
    assert report['phases'][0]['turns'][0]['target_observed'] is True
    assert report['phases'][0]['turns'][1]['frame_snapshots']
    history = json.loads((tmp_path/'run/first/history.json').read_text(encoding='utf-8'))
    assert len(history) == 4
    csvs = list((tmp_path/'run/first').rglob('*.csv'))
    assert csvs and any('H' in pd.read_csv(p).columns for p in csvs)
    assert old == (Path.cwd(), ps.PROJECTS_ROOT, ps.EXPORTS_ROOT, ac.plotter.DEFAULT_OUTPUT_DIR,
                   ga.PLOTS_DIR, ga.VERBOSE, ga.run_agent, ga.run_agent_with_planner, ac.run_bound_statistics)


def test_exception_is_saved_next_prompt_runs_and_interruption_restores_globals(tmp_path, monkeypatch):
    count = []
    def factory(brain):
        def handler(message, history, shown, store):
            count.append(message)
            if len(count) == 1:
                raise ValueError('fixture exception')
            raise KeyboardInterrupt()
        return handler
    monkeypatch.setattr(ga, 'make_chat_handler', factory)
    cwd, roots, original = Path.cwd(), (ps.PROJECTS_ROOT, ps.EXPORTS_ROOT), ga.run_agent
    existing = RUNNER['build_plan']()[1]['cases'][:2]
    with pytest.raises(KeyboardInterrupt):
        RUNNER['_execute_plan'](object(), [{'id':'fixture', 'cases':existing}], tmp_path/'run', scope='FIXTURE_TEST_ONLY')
    progress = json.loads((tmp_path/'run/progress.json').read_text(encoding='utf-8'))
    assert progress['status'] == 'INTERRUPTED'
    assert progress['phases'][0]['turns'][0]['execution_status'] == 'RUNNER_EXCEPTION'
    assert progress['phases'][0]['turns'][1]['execution_status'] == 'INTERRUPTED'
    assert len(count) == 2
    assert ga.run_agent is original and Path.cwd() == cwd
    assert roots == (ps.PROJECTS_ROOT, ps.EXPORTS_ROOT)


def test_blocked_case_is_not_sent_or_counted_as_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(ga, 'make_chat_handler', lambda brain: lambda *a: pytest.fail('Blocked prompt sent'))
    blocked = next(c for c in RUNNER['build_plan']()[0]['cases'] if c['id']=='TOOL-21')
    report = RUNNER['_execute_plan'](object(), [{'id':'blocked', 'cases':[blocked]}], tmp_path/'run', scope='FIXTURE_TEST_ONLY')
    record = report['phases'][0]['turns'][0]
    assert record['execution_status']=='BLOCKED_SCIENCE_SETUP'
    assert record['acceptance_verdict']=='NOT_RUN'


def test_source_guard_ignores_generated_zip_but_refuses_changed_python(tmp_path):
    import hashlib
    source = tmp_path/'agent_core.py'
    source.write_text('old source', encoding='utf-8')
    (tmp_path/'SOURCE_MANIFEST.json').write_text(json.dumps({'sha256': {
        'agent_core.py': hashlib.sha256(source.read_bytes()).hexdigest(),
        'exports/p.zip': 'outdated-output-hash',
    }}), encoding='utf-8')
    RUNNER['_verify_frozen_source'](tmp_path)
    source.write_text('changed source', encoding='utf-8')
    with pytest.raises(RuntimeError, match='SOURCE VERIFIED'):
        RUNNER['_verify_frozen_source'](tmp_path)


@pytest.mark.parametrize('bad_hash', [False, True])
def test_kaggle_cell_downloads_only_pinned_helper_reuses_brain_and_checks_hash(tmp_path, monkeypatch, bad_hash):
    import urllib.request
    source = Path(__file__).parent/'tools/run_chat_acceptance.py'
    cell = (Path(__file__).parent/'tools/kaggle_chat_acceptance.py').read_text(encoding='utf-8')
    compile(cell, 'kaggle-chat-acceptance', 'exec')
    monkeypatch.setattr(ac, '__file__', str(tmp_path/'agent_core.py'))
    # The public launcher deliberately downloads a historical immutable runner,
    # which is not necessarily the current checkout's runner.
    pinned = Path(__file__).parent/'tests/qa_pins/3b8666c00127e241daaf15190e637ce325f0c9b8/run_chat_acceptance.py'
    if pinned.is_file():
        pinned_bytes = pinned.read_bytes()
    else:
        import subprocess
        pinned_bytes = subprocess.check_output(['git', 'show', '3b8666c00127e241daaf15190e637ce325f0c9b8:tools/run_chat_acceptance.py'])
    payload = b'changed' if bad_hash else pinned_bytes
    downloaded, calls = [], []
    def open_fixture(url, **kw):
        downloaded.append(url)
        return io.BytesIO(payload)
    monkeypatch.setattr(urllib.request, 'urlopen', open_fixture)
    fixture_brain = object.__new__(ac.LlamaBrain)
    class Demo:
        def close(self): calls.append('close')
    def path_fixture(path, **kw):
        assert Path(path).parent == tmp_path/'verification'
        assert '3b8666c00127e241daaf15190e637ce325f0c9b8' in kw['init_globals']['RUNNER_ORIGIN']
        def execute(brain):
            assert brain is fixture_brain
            calls.append('run')
            return {'run_dir': 'FIXTURE', 'archive': 'FIXTURE.zip'}
        return {'_verify_frozen_source': lambda root: None, 'run_acceptance': execute}
    monkeypatch.setattr(runpy, 'run_path', path_fixture)
    scope = {'brain': fixture_brain, 'demo': Demo()}
    if bad_hash:
        with pytest.raises(AssertionError, match='SHA256'):
            exec(cell, scope)
        assert not calls and not (tmp_path/'verification').exists()
    else:
        exec(cell, scope)
        assert calls == ['close', 'run']
        assert (tmp_path/'verification').is_dir()
    assert len(downloaded) == 1 and '/master/' not in downloaded[0]
