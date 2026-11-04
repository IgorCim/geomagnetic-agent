"""Paired prompt diagnostics. Production agent, schemas and prompts stay frozen.

Run with the existing real Kaggle brain. A clearer request succeeding does not
excuse failure of a valid ordinary request. Single pairs do not prove causality.
"""
from pathlib import Path
import hashlib
import json
import runpy
import sys
import tempfile
import time
import traceback
import zipfile
from contextlib import redirect_stdout, redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
HELPER_SHA = 'a6a0422e161fcf8f77f9cbb071e9864b26f3073806d9679a2aaebcc96c5bf7eb'
HELPER_URL = 'https://raw.githubusercontent.com/IgorCim/geomagnetic-agent/3b8666c00127e241daaf15190e637ce325f0c9b8/tools/run_chat_acceptance.py'


def build_pairs(helper):
    plan = helper['build_plan']()
    tools = {c['id']: c for c in plan[0]['cases']}
    formulas = {c['id']: c for c in plan[2]['cases']}
    raw, derived = 'raw:irt:2024-09-10', 'derived:irt:2024-09-10'
    load = dict(tools['TOOL-01'], requires=[])
    derive = dict(tools['TOOL-08'], requires=[raw])
    batch = dict(tools['TOOL-03'], requires_projects=['qa_tools_round2'])
    plot = dict(tools['TOOL-14'], requires=[derived])
    setup = [load, derive]
    # Only requests which failed or recovered in a supplied run are paired.
    rows = [
        ('derive', [load], formulas['A01']['prompt'],
         f'Вызови calculate_derived_components для {raw}, components=["H","D","I"], без загрузки данных. Укажи формулы, единицы и фактический derived handle.',
         ['calculate_derived_components'], [raw]),
        ('baseline_full', setup, formulas['A04']['prompt'],
         f'Через calculate_baseline вычисли baseline H для {derived}, mode=full. Укажи фактический результат и метод.',
         ['calculate_baseline'], [derived]),
        ('baseline_anomaly', setup, formulas['A06']['prompt'],
         f'Вызови calculate_baseline для {derived}, component=H, mode=night, night_hours=[0,4]. Затем вызови calculate_derived_math для той же таблицы, metric=anomaly, component=H, baseline_value возьми из первого результата. Нужна серия H минус baseline, а не поиск выбросов. Укажи реальные результаты обоих вызовов.',
         ['calculate_baseline', 'calculate_derived_math'], [derived]),
        ('anomalies', setup, tools['TOOL-10']['prompt'],
         f'Вызови detect_anomalies для {derived}, component=H, sigma_threshold=3. Укажи число отмеченных строк и handle. get_statistics не заменяет поиск выбросов; выбросы не называй доказанными бурями.',
         ['detect_anomalies'], [derived]),
        ('list_projects', [tools['TOOL-02']], tools['TOOL-04']['prompt'],
         'Выполни list_projects() сейчас. Перечисли результат этого вызова; проекты не создавай и не экспортируй.',
         ['list_projects'], []),
        ('export', [tools['TOOL-02'], batch, derive, plot], tools['TOOL-05']['prompt'],
         'Вызови export_project для qa_tools_round2: archive_name=qa_report, folders={"Stations":["irt","bou"]}, sort_by=["date","station"]. Нужны каталоги обеих станций целиком внутри Stations. Укажи реально созданный ZIP.',
         ['export_project'], [raw, 'raw:bou:2024-09-10', derived]),
        ('no_recalculate', setup, plan[1]['cases'][13]['prompt'],
         f'Вызови только plot_components для {derived}, components=["H"]. Таблица уже рассчитана. Загрузку и calculate_derived_components не вызывай. Укажи созданный файл.',
         ['plot_components'], [derived]),
        ('forecast', [], tools['TOOL-29']['prompt'],
         'Выполни get_iswa_forecast() сейчас и сообщи фактический результат. Отдели прогноз Kp от наблюдений Dst, укажи источники и горизонт времени. Описание возможностей не заменяет вызов.',
         ['get_iswa_forecast'], []),
        ('noaa', [], tools['TOOL-26']['prompt'],
         'Вызови get_noaa_space_weather_context с query_type=alerts. По реальному результату укажи число записей и доступное временное покрытие. Не называй WARNING прогнозом и не смешивай время последней записи с полным окном; если метаданных нет, скажи об этом.',
         ['get_noaa_space_weather_context'], []),
    ]
    # 2 x 2: wording and preceding metric vary independently.
    for context in ('fresh', 'after_deltas'):
        rows.append(('gic_' + context,
                     [load] + ([dict(formulas['C02'], requires=[raw])] if context == 'after_deltas' else []),
                     formulas['C03']['prompt'],
                     f'Вызови apply_physics_formula для {raw}, metric="gic". Нужна именно gic, не deltas. Укажи фактический результат, handle, единицы и ограничения.',
                     ['apply_physics_formula'], [raw]))
    pairs = []
    for name, prep, original, explicit, targets, requires in rows:
        for variant, prompt in [('original', original), ('explicit', explicit)]:
            pairs.append({'id': name + '_' + variant, 'case': name, 'variant': variant,
                          'setup': prep, 'prompt': prompt, 'targets': targets, 'requires': requires})
            if name in {'export', 'list_projects'}:
                pairs[-1]['requires_projects'] = ['qa_tools_round2']
    return pairs


def probe_iswa(directory):
    """Independent HTTP evidence; a local disabled stub cannot certify NASA."""
    import requests
    import iswa_loader as il
    from datetime import datetime, timezone
    reports = []
    urls = [('current_catalog', 'https://iswa.ccmc.gsfc.nasa.gov/api/info'),
            ('legacy_getmap', il.build_wms_url('2024-09-10T12:00:00Z'))]
    for label, url in urls:
        row = {'id': label, 'requested_url': url, 'utc': datetime.now(timezone.utc).isoformat()}
        try:
            response = requests.get(url, timeout=30)
            body = response.content
            (directory/(label + '.body')).write_bytes(body)
            row.update(status=response.status_code, final_url=response.url,
                       content_type=response.headers.get('Content-Type'), bytes=len(body),
                       sha256=hashlib.sha256(body).hexdigest(), png=body.startswith(b'\x89PNG\r\n\x1a\n'))
            if label == 'current_catalog':
                if response.ok:
                    try:
                        feeds = response.json().get('datafeeds', [])
                        row['datafeed_count'] = len(feeds)
                        row['protocols'] = sorted({str(f.get('dataSource', {}).get('protocol')) for f in feeds})
                    except (ValueError, AttributeError, TypeError):
                        row['catalog_shape_unrecognized'] = True
                row['keyword_snippets'] = []
                import re
                for m in re.finditer(r'perturbation|ground.mag|\bwms\b', response.text, re.I):
                    row['keyword_snippets'].append(response.text[max(0, m.start()-100):m.end()+150])
                    if len(row['keyword_snippets']) == 30:
                        break
        except requests.RequestException as exc:
            row['network_error'] = str(exc)
        reports.append(row)
    (directory/'iswa_http.json').write_text(json.dumps({
        'observations': reports,
        'verdict': 'REVIEW_REQUIRED: 404 establishes only that URL; 403/timeout cannot establish decommissioning. Catalog keywords are candidates, not proof of a usable map.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')


def missing_setup(card, store, projects_root):
    missing = [handle for handle in card.get('requires', []) if store.get(handle) is None]
    projects = [name for name in card.get('requires_projects', [])
                if not (projects_root/name/'manifest.json').is_file()]
    columns = {name: sorted({'H', 'D', 'I'} - set(store.get(name).columns))
               for name in card.get('requires', [])
               if name.startswith('derived:') and store.get(name) is not None}
    return missing, projects, {name: cols for name, cols in columns.items() if cols}


def select_pairs(helper, case_ids=None):
    pairs = build_pairs(helper)
    if case_ids is None:
        return pairs
    if not isinstance(case_ids, (list, tuple)) or not case_ids:
        raise ValueError('case_ids должен быть непустым списком имён проверок.')
    unknown = set(case_ids) - {p['case'] for p in pairs}
    if unknown:
        raise ValueError(f'Неизвестные проверки: {sorted(unknown)}')
    return [p for p in pairs if p['case'] in case_ids]


def run_diagnostics(brain, *, helper_path, logon='', repeats=1, case_ids=None, probe_services=True):
    import agent_core as ac
    import gradio_app as ga
    import project_store as ps
    from datetime import datetime, timezone
    if not isinstance(brain, ac.LlamaBrain) or isinstance(brain, ac.ScriptedBrain) or ac.OFFLINE:
        raise RuntimeError('Нужна уже загруженная настоящая LlamaBrain и живые загрузчики.')
    assert isinstance(repeats, int) and 1 <= repeats <= 3
    for module in (ac, ga, ps, __import__('chat_session'), __import__('physics_engine'), __import__('geomag_plotter')):
        assert Path(module.__file__).resolve().parent == ROOT, 'Смешаны checkout модулей.'
    helper_path = Path(helper_path)
    assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == HELPER_SHA
    h = runpy.run_path(str(helper_path))
    h['_verify_frozen_source'](ROOT)
    pairs = select_pairs(h, case_ids)
    parent = ROOT/'verification'
    parent.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='prompt_diagnostics_', dir=parent))
    redact = h['Redactor']([logon])
    files = list(ROOT.glob('*.py'))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Real Gradio Python handler, independent fresh chats per variant; not browser UI. No agent edits, no fixed seed, no causal PASS.',
              'supermag_configured': bool(logon), 'source_sha256': before,
              'python': sys.version, 'model_repo': ac.MODEL_REPO, 'model_stem': ac.MODEL_STEM,
              'n_ctx': getattr(brain, '_n_ctx', None),
              'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'selected_cases': sorted({p['case'] for p in pairs}),
              'independent_iswa_probe': 'REQUESTED' if probe_services else 'NOT_REPEATED',
              'comparisons': []}
    write = lambda: h['_write'](directory/'results.json', report, redact)
    console = sys.stdout
    try:
        if probe_services:
            probe_iswa(directory)
        # SuperMAG: real authenticated tool call, in a separate session.
        pairs.append({'id': 'supermag_authenticated', 'case': 'service_check', 'variant': 'authenticated',
                      'setup': [], 'prompt': 'Вызови fetch_supermag_data с data_type=indices за 2024-09-10 для SME/SML/SMU. Используй логин сессии, не выводи его. Укажи реальный handle, число строк либо реальную причину отказа.',
                      'targets': ['fetch_supermag_data'], 'requires': []})
        h['_write'](directory/'plan.json', pairs, redact)
        for repeat in range(repeats):
            # Reverse pair order on alternate repetitions to expose order bias.
            ordered = pairs if repeat % 2 == 0 else list(reversed(pairs))
            for pair in ordered:
                phase = directory/f'{repeat+1:02d}_{pair["id"]}'
                with h['_environment'](phase, ga, ac, ps):
                    history, shown = [], []
                    store = ac.FrameStore(workspace=ps.Workspace(root=phase/'projects'))
                    if logon:
                        store = ga._apply_session_logon(store, logon)
                    handler = ga.make_chat_handler(brain)
                    result = {'id': pair['id'], 'repeat': repeat+1, 'case': pair['case'],
                              'variant': pair['variant'], 'turns': [], 'verdict': 'REVIEW_REQUIRED'}
                    report['comparisons'].append(result)
                    cards = [dict(c, role='setup') for c in pair['setup']] + [dict(pair, role='target')]
                    for i, card in enumerate(cards):
                        record = {'role': card['role'], 'prompt': card['prompt'], 'status': 'STARTED'}
                        result['turns'].append(record)
                        missing, missing_projects, missing_columns = missing_setup(card, store, ps.PROJECTS_ROOT)
                        if missing or missing_projects or missing_columns:
                            record.update(status='BLOCKED_SETUP', missing_handles=missing,
                                          missing_projects=missing_projects, missing_columns=missing_columns)
                            print(f'{pair["id"]}: BLOCKED_SETUP; см. results.json', flush=True)
                            write()
                            break
                        calls, started = [], time.monotonic()
                        with (phase/f'{i:02d}.log').open('w', encoding='utf-8') as log:
                            tee = h['Tee'](console, log, redact)
                            try:
                                with redirect_stdout(tee), redirect_stderr(tee), h['_record_results'](ga, ac, calls):
                                    print(f'\n===== {pair["id"]} / {card["role"]} =====\n{card["prompt"]}', flush=True)
                                    write()
                                    out = handler(card['prompt'], history, shown, store)
                                    history, _, tool_log, shown, downloads, _ = ga.split_result(out)
                                    record.update(status='RETURNED', answer=history[-1]['content'], tool_log=tool_log, downloads=downloads)
                                    print(record['answer'], '\n', tool_log, flush=True)
                            except BaseException:
                                record.update(status='INTERRUPTED_OR_EXCEPTION', exception=traceback.format_exc())
                                raise
                            finally:
                                tee.finish()
                                flat = [c for r in calls for c in r['result'].get('tool_calls', [])]
                                record.update(engine_results=calls, elapsed_seconds=round(time.monotonic()-started, 3),
                                              successful_tools=[c.get('tool') for c in flat if c.get('ok')],
                                              errors=[c for c in flat if c.get('error') or c.get('ok') is False],
                                              tables=[{'handle': n, 'rows': len(store.get(n))} for n in store.names()])
                                snapshots = phase/f'{i:02d}_frames'
                                snapshots.mkdir(exist_ok=True)
                                try:
                                    record['snapshots'] = h['_snapshot'](store, flat, snapshots)
                                except Exception as exc:
                                    record['snapshot_error'] = f'{type(exc).__name__}: {exc}'
                                if card['role'] == 'target':
                                    record['requested_tools'] = card['targets']
                                    record['successful_requested_tools'] = sorted(set(card['targets']) & set(record['successful_tools']))
                                h['_write'](phase/f'{i:02d}.json', record, redact)
                                h['_write'](phase/'history.json', history, redact)
                                write()
        report['status'] = 'RECORDED'
    finally:
        report['unchanged_agent_sources'] = all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == v for n, v in before.items())
        if report['status'] == 'RUNNING':
            report['status'] = 'INCOMPLETE'
        write()
        archive = directory.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for p in directory.rglob('*'):
                if p.is_file() and p.suffix in {'.json', '.log', '.csv', '.body'}:
                    bundle.write(p, p.relative_to(directory).as_posix())
        print(f'Результаты: {directory}\nАрхив: {archive}\nЭто запись наблюдений, не автоматический PASS.', flush=True)
    return {'run_dir': str(directory), 'archive': str(archive)}
