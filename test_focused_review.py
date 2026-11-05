from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent


def test_focus_plan_keeps_original_failures_and_dependencies():
    runner = runpy.run_path(str(ROOT/'tools/run_focused_review.py'))
    plan = runner['build_plan']()
    phases = {p['id']: p['cases'] for p in plan}
    assert len(phases) == 7
    long = phases['04_context_outliers']
    assert [c['id'] for c in long] == [f'CONTEXT-{i:02d}' for i in range(1, 15)] + ['A08']
    assert phases['03_outliers_fresh'][-1]['prompt'] == long[-1]['prompt']
    assert phases['01_permissions'][2]['prompt'] == long[13]['prompt']
    for p in plan:
        ids = [c['id'] for c in p['cases']]
        assert len(ids) == len(set(ids))
        assert all(c['prompt'] for c in p['cases'])
    assert phases['06_source_refusal'][-1]['id'] == 'C10'
    assert 'разрешаю' in phases['07_source_recovery'][-1]['prompt']
