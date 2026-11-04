"""Test diagnostic scheduling and evidence gates, not Qwen's live quality."""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import requests

ROOT = Path(__file__).resolve().parent
D = runpy.run_path(str(ROOT/'tools/run_prompt_diagnostics.py'))
H = runpy.run_path(str(ROOT/'tools/run_chat_acceptance.py'))


def test_pairs_keep_originals_and_vary_context_separately():
    pairs = D['build_pairs'](H)
    assert len(pairs) == 22
    assert sum(len(c['setup'])+1 for c in pairs)+1 == 57
    assert len({c['id'] for c in pairs}) == 22
    by_id = {c['id']: c for c in pairs}
    assert by_id['gic_fresh_original']['prompt'] == by_id['gic_after_deltas_original']['prompt']
    assert by_id['gic_fresh_explicit']['prompt'] == by_id['gic_after_deltas_explicit']['prompt']
    assert len(by_id['gic_after_deltas_original']['setup']) == len(by_id['gic_fresh_original']['setup'])+1
    originals = {c['prompt'] for p in H['build_plan']() for c in p['cases']}
    assert all(c['prompt'] in originals for c in pairs if c['variant'] == 'original')
    assert all('IgorCim' not in c['prompt'] for c in pairs)


def test_missing_setup_checks_actual_frames_columns_and_project(tmp_path):
    frames = {'derived:irt:2024-09-10': SimpleNamespace(columns=['H'])}
    card = {'requires': ['raw:irt:2024-09-10', 'derived:irt:2024-09-10'],
            'requires_projects': ['qa_tools_round2']}
    result = D['missing_setup'](card, frames, tmp_path)
    assert result == (['raw:irt:2024-09-10'], ['qa_tools_round2'], {'derived:irt:2024-09-10': ['D', 'I']})
    frames['raw:irt:2024-09-10'] = SimpleNamespace(columns=['X', 'Y', 'Z'])
    frames['derived:irt:2024-09-10'].columns = ['H', 'D', 'I']
    project = tmp_path/'qa_tools_round2'
    project.mkdir()
    (project/'manifest.json').write_text('{}')
    assert D['missing_setup'](card, frames, tmp_path) == ([], [], {})


def test_http_probe_records_external_evidence_without_claiming_decommission(tmp_path, monkeypatch):
    def get(url, **kwargs):
        if '/api/info' in url:
            body = b'{"name":"ground magnetic perturbation candidate"}'
            return SimpleNamespace(status_code=200, ok=True, json=lambda: json.loads(body), content=body, text=body.decode(),
                                   headers={'Content-Type': 'application/json'}, url=url)
        raise requests.Timeout('fixture timeout')
    monkeypatch.setattr(requests, 'get', get)
    D['probe_iswa'](tmp_path)
    result = json.loads((tmp_path/'iswa_http.json').read_text())
    assert result['observations'][0]['keyword_snippets']
    assert result['observations'][1]['network_error'] == 'fixture timeout'
    assert 'cannot establish decommissioning' in result['verdict']
    assert (tmp_path/'current_catalog.body').is_file()


def test_no_fixture_model_accepted_by_public_entry(tmp_path):
    with pytest.raises(RuntimeError, match='LlamaBrain'):
        D['run_diagnostics'](object(), helper_path=tmp_path/'absent.py')


def test_continuation_only_unfinished_pair_and_preserves_verbatim_setup():
    full = D['build_pairs'](H)
    selected = D['select_pairs'](H, ['gic_after_deltas'])
    assert selected == [p for p in full if p['case'] == 'gic_after_deltas']
    assert [p['variant'] for p in selected] == ['original', 'explicit']
    assert sum(len(p['setup'])+1 for p in selected)+1 == 7
    assert all([c['id'] for c in p['setup']] == ['TOOL-01', 'C02'] for p in selected)
    for bad in ([], 'gic_after_deltas', ['unknown']):
        with pytest.raises(ValueError):
            D['select_pairs'](H, bad)
    assert D['select_pairs'](H) == full


def test_launcher_pins_both_scripts_without_updating_agent_or_embedding_login():
    import ast
    source = (ROOT/'tools/kaggle_prompt_diagnostics.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'qa_files' for t in n.targets))
    pins = ast.literal_eval(assignment.value)
    assert pins[0] == ('3b8666c00127e241daaf15190e637ce325f0c9b8', 'run_chat_acceptance.py', D['HELPER_SHA'])
    import subprocess
    import hashlib
    for commit, name, digest in pins:
        assert hashlib.sha256(subprocess.check_output(['git', 'show', commit+':tools/'+name], cwd=ROOT)).hexdigest() == digest
    assert 'git pull' not in source and 'load_brain(' not in source
    assert 'IgorCim' not in source.replace('https://raw.githubusercontent.com/IgorCim/', '')


def test_remaining_launcher_pins_scripts_and_selects_only_unfinished_work():
    import ast
    import hashlib
    import subprocess
    source = (ROOT/'tools/kaggle_prompt_diagnostics_remaining.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    values = {n.targets[0].id: n.value for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    for commit, name, digest in ast.literal_eval(values['qa_files']):
        assert hashlib.sha256(subprocess.check_output(['git', 'show', commit+':tools/'+name], cwd=ROOT)).hexdigest() == digest
    assert ast.literal_eval(values['qa_remaining_cases']) == ['gic_after_deltas']
    call = values['qa_diagnostic_result']
    assert any(k.arg == 'probe_services' and ast.literal_eval(k.value) is False for k in call.keywords)
    assert 'git pull' not in source and 'load_brain(' not in source
