"""Orchestration tests only; no network or real model inference."""
from pathlib import Path
import runpy
import pytest
import agent_core as ac
import project_store as ps


@pytest.mark.parametrize('fail_stress', [False, True])
def test_full_run_credentials_and_restoration(tmp_path, monkeypatch, fail_stress):
    import os
    runner = runpy.run_path(str(Path(__file__).parent/'tools/run_full_evidence.py'))
    helper = runpy.run_path(str(Path(__file__).parent/'tools/run_chat_acceptance.py'))
    fn = runner['run']
    fn.__globals__['ROOT'] = tmp_path
    monkeypatch.setattr(ac, 'OFFLINE', False)
    monkeypatch.setenv('SUPERMAG_LOGON', 'previous-credential')
    seen = []
    def acceptance(brain, **kwargs):
        seen.append((os.environ.get('SUPERMAG_LOGON'), kwargs['plan']))
        return {'run_dir': 'fixture', 'archive': 'fixture.zip'}
    helper.update(run_acceptance=acceptance, _verify_frozen_source=lambda root: None)
    def load(path):
        return {'build_plan': lambda: [{'id': 'original', 'cases': []}]} if 'combined' in str(path) else helper
    monkeypatch.setattr(runpy, 'run_path', load)
    (tmp_path/'stress_300.py').write_text(
        "import os\nimport project_store as ps\nassert os.environ['SUPERMAG_LOGON']=='test-credential'\n"
        "ps.EXPORTS_ROOT='changed'\nr_smag={'ok':True}\nRUN_DIR='fixture_stress'\n" +
        ("raise RuntimeError('fixture failure')\n" if fail_stress else ''), encoding='utf-8')
    cwd, exports = Path.cwd(), ps.EXPORTS_ROOT
    result = fn(object.__new__(ac.LlamaBrain), 'test-credential')
    assert [s[0] for s in seen] == ['test-credential', None]
    assert seen[-1][1][0]['id'] == 'no_supermag_logon'
    assert os.environ['SUPERMAG_LOGON'] == 'previous-credential'
    assert Path.cwd() == cwd and ps.EXPORTS_ROOT == exports
    assert result['report']['stages']['01_stress']['status'] == ('FAILED' if fail_stress else 'RETURNED')
    assert (Path(result['run_dir'])/'summary.json').is_file()
