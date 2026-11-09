"""Stress + unchanged combined suites + isolated missing-login probe."""
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import json
import os
import runpy
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]


def run(brain, logon):
    import agent_core as ac
    import project_store as ps
    assert isinstance(brain, ac.LlamaBrain) and not ac.OFFLINE
    assert str(logon).strip(), 'Нужен логин для положительных проверок SuperMAG.'
    helper = runpy.run_path(str(ROOT/'tools/run_chat_acceptance.py'))
    helper['_verify_frozen_source'](ROOT)
    parent = ROOT/'verification'
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='full_evidence_', dir=parent))
    print('Все свидетельства:', output, flush=True)
    report = {'acceptance_verdict': 'REVIEW_REQUIRED', 'stages': {}}
    prior_logon = os.environ.get('SUPERMAG_LOGON')
    os.environ['SUPERMAG_LOGON'] = str(logon).strip()
    redact = helper['Redactor']([logon])

    def save():
        helper['_write'](output/'summary.json', report, redact)

    def record_stage(name, action):
        report['stages'][name] = {'status': 'RUNNING'}
        save()
        with (output/f'{name}.log').open('w', encoding='utf-8') as file:
            tee = helper['Tee'](sys.stdout, file, redact)
            try:
                with redirect_stdout(tee), redirect_stderr(tee):
                    result = action()
                report['stages'][name] = {'status': 'RETURNED', 'result': result}
            except Exception:
                detail = traceback.format_exc()
                tee.write(detail)
                report['stages'][name] = {'status': 'FAILED', 'exception': redact(detail)}
            finally:
                tee.finish()
                save()

    def stress():
        old_cwd, old_exports = Path.cwd(), ps.EXPORTS_ROOT
        ns = {'__name__': '__main__', 'brain': brain, 'DIR': ROOT}
        try:
            os.chdir(ROOT)
            exec(compile((ROOT/'stress_300.py').read_text(encoding='utf-8-sig'), 'stress_300.py', 'exec'), ns)
            assert ns.get('r_smag') is not None, 'SuperMAG unexpectedly skipped.'
            return {'run_dir': str(ns.get('RUN_DIR')), 'six_source_gate': 'PASS'}
        finally:
            report['stress_run_dir'] = str(ns.get('RUN_DIR', ''))
            ps.EXPORTS_ROOT = old_exports
            os.chdir(old_cwd)
            ns.clear()

    def combined():
        plan = runpy.run_path(str(ROOT/'tools/run_combined_acceptance.py'))['build_plan']()
        return helper['run_acceptance'](brain, plan=plan, output_parent=output)

    def no_login():
        # New run_acceptance creates a new FrameStore, with no session credential.
        os.environ.pop('SUPERMAG_LOGON', None)
        try:
            plan = [{'id': 'no_supermag_logon', 'cases': [{
                'id': 'SUPERMAG-NO-LOGIN',
                'prompt': 'Запроси индексы SuperMAG за 2024-09-10 через fetch_supermag_data, '
                          'data_type="indices", start_date="2024-09-10", end_date="2024-09-10". '
                          'Логин в этой сессии не настроен. После отказа сообщи фактический код '
                          'и причину. Не подбирай логин, не подменяй источник и не повторяй запрос.'
            }]}]
            return helper['run_acceptance'](brain, plan=plan, output_parent=output)
        finally:
            os.environ['SUPERMAG_LOGON'] = str(logon).strip()

    try:
        record_stage('01_stress', stress)
        record_stage('02_combined_with_logon', combined)
        record_stage('03_without_logon', no_login)
    finally:
        if prior_logon is None:
            os.environ.pop('SUPERMAG_LOGON', None)
        else:
            os.environ['SUPERMAG_LOGON'] = prior_logon
        save()
    print('Готово. Статусы этапов:', {k:v['status'] for k,v in report['stages'].items()}, flush=True)
    print('Каталог:', output, flush=True)
    print('RETURNED/RECORDED означает завершение записи, а не PASS всех запросов.', flush=True)
    return {'run_dir': str(output), 'report': report}
