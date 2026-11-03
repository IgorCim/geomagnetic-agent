"""Replay the existing acceptance prompts through the production Gradio handler.

No new prompts, model loading, source updates, automatic retries or PASS claims.
Public entry requires a real LlamaBrain and live loaders. Private orchestration
helpers are tested with explicitly labelled fixtures.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['docs/qa/ready_tool_cases.json', 'docs/qa/TOOL_RUNBOOK.md',
           'docs/LONG_CHAT.md', 'docs/qa/FORMULA_RUNBOOK.md']


def build_plan(root: Path = ROOT) -> list[dict]:
    """Extract verbatim text blocks; schedule existing dependencies explicitly."""
    root = Path(root)
    tools = json.loads((root/SOURCES[0]).read_text(encoding='utf-8'))
    inventory = json.loads((root/'docs/qa/inventory.json').read_text(encoding='utf-8'))
    names = {t['id']: t['name'] for t in inventory['tools']}
    by_id = {c['id']: dict(c, target_tool=names[c['id']]) for c in tools}
    assert len(by_id) == 29
    tool_doc = (root/SOURCES[1]).read_text(encoding='utf-8')
    prep = re.search(r'Подготовка D1.*?```text\s*\n(.*?)```', tool_doc, re.S)
    assert prep, 'Missing existing D1 setup prompt'
    order = [2, 1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 5, 6, 7,
             15, 16, 17, 18, 19, 20, 21, 22, 24, 23, 25, 26, 27, 28, 29]
    tool_cases = []
    for number in order:
        if number == 15:
            tool_cases.append({'id': 'SETUP-IRT-D1', 'prompt': prep[1].strip(), 'setup': True})
        tool_cases.append(by_id[f'TOOL-{number:02d}'])
    context = (root/SOURCES[2]).read_text(encoding='utf-8').split(
        '## Живая приёмка Qwen: 20 отдельных сообщений', 1)[1].split('\n## ', 1)[0]
    context_prompts = re.findall(r'```text\s*\n(.*?)```', context, re.S)
    assert len(context_prompts) == 20
    context_cases = [{'id': f'CONTEXT-{n:02d}', 'prompt': p.strip()}
                     for n, p in enumerate(context_prompts, 1)]
    formula_doc = (root/SOURCES[3]).read_text(encoding='utf-8')
    headings = list(re.finditer(r'^### ([A-E]\d{2})[^\n]*$', formula_doc, re.M))
    formula_cases = []
    for heading in headings:
        tail = formula_doc[heading.end():]
        body = re.split(r'^#{2,3} ', tail, maxsplit=1, flags=re.M)[0]
        block = re.search(r'```text\s*\n(.*?)```', body, re.S)
        if heading[1] == 'D01':
            formula_cases.append(dict(by_id['TOOL-08'], id='SETUP-D-HD I', setup=True))
        if block:
            formula_cases.append({'id': heading[1], 'prompt': block[1].strip()})
        else:
            assert 'BLOCKED_SCIENCE_SETUP' in body, heading[1]
            formula_cases.append({'id': heading[1], 'prompt': None, 'status': 'BLOCKED_SCIENCE_SETUP'})
    assert len(headings) == 61, f'Unexpected formula catalog: {len(headings)}'
    # Reuse existing tool prompts to prepare this independent formula session.
    formula_setup = [dict(by_id[f'TOOL-{n:02d}'], id=f'SETUP-FORMULA-{n:02d}', setup=True)
                     for n in [2, 1, 3, 24]]
    plan = [{'id': '01_tools', 'cases': tool_cases},
            {'id': '02_context', 'cases': context_cases},
            {'id': '03_formulas', 'cases': formula_setup + formula_cases}]
    for phase in plan:
        for card in phase['cases']:
            if card.get('prompt'):
                assert not re.search(r'\{[A-Z][A-Z0-9_]*\}', card['prompt']), card['id']
    return plan


class Redactor:
    def __init__(self, secrets=()):
        self.secrets = sorted((str(s) for s in secrets if s), key=len, reverse=True)

    def __call__(self, text):
        text = str(text)
        for secret in self.secrets:
            text = text.replace(secret, '[REDACTED]')
        return re.sub(r'(?i)([?&](?:logon|token|api_key)=)[^&\s\"\']+', r'\1[REDACTED]', text)


class Tee:
    """Line buffering also redacts secrets split across successive writes."""
    def __init__(self, console, file, redact):
        self.console, self.file, self.redact = console, file, redact
        self.pending = ''

    def write(self, text):
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            self._emit(line + '\n')
        return len(text)

    def _emit(self, text):
        text = self.redact(text)
        self.console.write(text)
        self.file.write(text)
        self.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

    def finish(self):
        if self.pending:
            self._emit(self.pending)
            self.pending = ''


def _plain(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items() if str(k) not in {'store', 'brain', 'supermag_logon'}}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return f'<{type(value).__name__}>'


def _write(path, value, redact):
    encoded = redact(json.dumps(_plain(value), ensure_ascii=False, indent=2, default=str))
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(encoded, encoding='utf-8')
    temp.replace(path)


@contextmanager
def _environment(phase_dir, ga, ac, ps):
    """Temporary output roots; never alter loaders, schemas or model settings."""
    previous = (Path.cwd(), ps.PROJECTS_ROOT, ps.EXPORTS_ROOT, ac.plotter.DEFAULT_OUTPUT_DIR, ga.PLOTS_DIR, ga.VERBOSE)
    try:
        phase_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(phase_dir)
        ps.PROJECTS_ROOT = phase_dir/'projects'
        ps.EXPORTS_ROOT = phase_dir/'exports'
        ac.plotter.DEFAULT_OUTPUT_DIR = phase_dir/'plots'
        ga.PLOTS_DIR = phase_dir/'plots'
        ga.VERBOSE = True
        yield
    finally:
        os.chdir(previous[0])
        ps.PROJECTS_ROOT, ps.EXPORTS_ROOT, ac.plotter.DEFAULT_OUTPUT_DIR, ga.PLOTS_DIR, ga.VERBOSE = previous[1:]


@contextmanager
def _record_results(ga, ac, records):
    """Forward calls unchanged and retain structured results for the evidence."""
    originals = []
    fields = ('ok', 'error', 'stop_reason', 'text', 'tool_calls', 'plots', 'matrix_path', 'coverage', 'missing_cells', 'archive', 'rounds')
    try:
        for obj, name in [(ga, 'run_agent'), (ga, 'run_agent_with_planner'), (ac, 'run_bound_statistics')]:
            original = getattr(obj, name)
            originals.append((obj, name, original))

            def wrap(fn, label):
                @wraps(fn)
                def recorded(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    if isinstance(result, dict):
                        records.append(dict(path=label, result={k: _plain(result[k]) for k in fields if k in result}))
                    return result
                return recorded
            setattr(obj, name, wrap(original, name))
        yield
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


def _snapshot(store, calls, directory):
    produced = set()
    for entry in calls:
        if not entry.get('ok'):
            continue
        produced.update(h for h in [entry.get('frame_handle'), *(entry.get('frame_handles') or []), *(entry.get('resolved_handles') or [])] if isinstance(h, str))
    saved = []
    for handle in sorted(produced):
        frame = store.get(handle)
        if frame is None:
            continue
        filename = hashlib.sha256(handle.encode()).hexdigest()[:16] + '.csv'
        path = directory/filename
        frame.to_csv(path, index=False)
        saved.append({'handle': handle, 'path': str(path), 'rows': len(frame),
                      'columns': list(frame.columns), 'attrs': _plain(frame.attrs),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return saved


def _execute_plan(brain, plan, run_dir, *, scope, logon=''):
    """Private orchestration; tests label fixture runs explicitly, never LIVE."""
    import agent_core as ac
    import gradio_app as ga
    import project_store as ps
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    redact = Redactor([logon])
    manifest = {'scope': scope, 'status': 'RUNNING', 'acceptance_verdict': 'REVIEW_REQUIRED', 'phases': []}
    _write(run_dir/'plan.json', plan, redact)
    console = sys.stdout
    try:
        for phase in plan:
            phase_dir = run_dir/phase['id']
            phase_state = {'id': phase['id'], 'status': 'RUNNING', 'turns': []}
            manifest['phases'].append(phase_state)
            _write(run_dir/'progress.json', manifest, redact)
            with _environment(phase_dir, ga, ac, ps):
                history, shown = [], []
                store = ac.FrameStore(workspace=ps.Workspace(root=phase_dir/'projects'))
                if logon:
                    store = ga._apply_session_logon(store, logon)
                assert not store.names() and store.workspace.project is None
                handler = ga.make_chat_handler(brain)
                print(f'\n===== НОВЫЙ ЧАТ: {phase["id"]} =====', flush=True)
                for index, card in enumerate(phase['cases'], 1):
                    record = {'id': card['id'], 'setup': bool(card.get('setup')), 'prompt': card.get('prompt'),
                              'execution_status': 'STARTED', 'acceptance_verdict': 'REVIEW_REQUIRED'}
                    phase_state['turns'].append(record)
                    turn_dir = phase_dir/f'{index:03d}_{card["id"].replace(" ", "_")}'
                    turn_dir.mkdir()
                    _write(turn_dir/'turn.json', record, redact)
                    _write(run_dir/'progress.json', manifest, redact)
                    if not card.get('prompt'):
                        record.update(execution_status=card.get('status', 'BLOCKED'), acceptance_verdict='NOT_RUN')
                        print(f'[{phase["id"]} {card["id"]}] {record["execution_status"]}; не засчитывается как PASS.', flush=True)
                        _write(turn_dir/'turn.json', record, redact)
                        _write(run_dir/'progress.json', manifest, redact)
                        continue
                    results = []
                    start = time.monotonic()
                    with (turn_dir/'console.log').open('w', encoding='utf-8') as logfile:
                        tee = Tee(console, logfile, redact)
                        try:
                            with redirect_stdout(tee), redirect_stderr(tee), _record_results(ga, ac, results):
                                print(f'\n[{phase["id"]} {card["id"]}] ЗАПРОС:\n{card["prompt"]}', flush=True)
                                out = handler(card['prompt'], history, shown, store)
                                history, _slots, tool_log, shown, files, _ = ga.split_result(out)
                                answer = history[-1]['content'] if history else ''
                                print(f'\nОТВЕТ:\n{answer}\n\nЖУРНАЛ ИНСТРУМЕНТОВ:\n{tool_log}', flush=True)
                                record.update(answer=answer, tool_log=tool_log, download_files=files,
                                              execution_status='HANDLER_RETURNED', engine_results=results)
                        except KeyboardInterrupt:
                            record.update(execution_status='INTERRUPTED', engine_results=results)
                            raise
                        except Exception:
                            detail = traceback.format_exc()
                            record.update(execution_status='RUNNER_EXCEPTION', exception=detail, engine_results=results)
                            tee.write(detail + '\n')
                            history = list(history) + [{'role': 'user', 'content': card['prompt']},
                                                       {'role': 'assistant', 'content': 'Ошибка выполнения обработчика; детали сохранены в журнале.'}]
                        finally:
                            tee.finish()
                            record['elapsed_seconds'] = round(time.monotonic() - start, 3)
                            calls = [c for item in results for c in item['result'].get('tool_calls', [])]
                            target = card.get('target_tool')
                            record['target_tool'] = target
                            record['target_observed'] = any(c.get('tool') == target for c in calls) if target else None
                            record['tool_errors'] = [c for c in calls if c.get('error') or c.get('ok') is False]
                            record['tables'] = [{'handle': h, 'rows': len(store.get(h))} for h in store.names()]
                            try:
                                record['frame_snapshots'] = _snapshot(store, calls, turn_dir)
                            except Exception as exc:
                                record['snapshot_error'] = f'{type(exc).__name__}: {exc}'
                            _write(turn_dir/'turn.json', record, redact)
                            _write(phase_dir/'history.json', history, redact)
                            _write(run_dir/'progress.json', manifest, redact)
                phase_state['status'] = 'RECORDED'
        manifest['status'] = 'RECORDED'
    except KeyboardInterrupt:
        manifest['status'] = 'INTERRUPTED'
        raise
    except BaseException:
        manifest['status'] = 'ABORTED'
        raise
    finally:
        _write(run_dir/'progress.json', manifest, redact)
        transcript = []
        for phase in manifest['phases']:
            transcript.extend([f'# {phase["id"]}', ''])
            for turn in phase['turns']:
                transcript.extend([f'## {turn["id"]}', '', turn.get('prompt') or '(BLOCKED)', '',
                                   str(turn.get('answer') or turn['execution_status']), '',
                                   str(turn.get('tool_log') or ''), ''])
        (run_dir/'transcript.md').write_text(redact('\n'.join(transcript)), encoding='utf-8')
        print(f'\nСохранено: {run_dir}; статус записи: {manifest["status"]}. Научная/живая приёмка: REVIEW_REQUIRED.', flush=True)
    return manifest


def _verify_frozen_source(root):
    manifest_path = root/'SOURCE_MANIFEST.json'
    if not manifest_path.is_file():
        return
    source = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name, digest in source.get('sha256', {}).items():
        # Output archives may be tracked in older checkouts. A newly generated
        # ZIP is not a source-code update and must not trigger a false refusal.
        if name not in SOURCES and not name.endswith('.py'):
            continue
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Файлы отличаются от SOURCE VERIFIED. Нужен новый чистый запуск ячеек 1–5; код после brain не обновлять.')
    runtime_path = root/'portfolio_evidence/runtime.json'
    if runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
        loaded = runtime.get('source_manifest_sha256')
        if loaded and loaded != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise RuntimeError('SOURCE_MANIFEST изменился после загрузки brain. Перезапустите Python и ячейки 1–5.')


def run_acceptance(brain, *, output_parent=None):
    """Run live acceptance using the already-loaded real model. Never reload it."""
    import agent_core as ac
    if not isinstance(brain, ac.LlamaBrain) or isinstance(brain, ac.ScriptedBrain):
        raise TypeError('Нужна настоящая brain=LlamaBrain из ячейки 5. Тестовые модели не принимаются.')
    if ac.OFFLINE:
        raise RuntimeError('OFFLINE включён. Нужна живая конфигурация из Kaggle-ячейки 5.')
    if Path(ac.__file__).resolve().parent != ROOT:
        raise RuntimeError('Модель/модули и runner взяты из разных checkout. Перезапустите Python и выполните ячейки 1–5.')
    for name in ['gradio_app', 'project_store', 'chat_session', 'physics_engine', 'geomag_plotter']:
        module = __import__(name)
        if Path(module.__file__).resolve().parent != ROOT:
            raise RuntimeError(f'Модуль {name} загружен из другого checkout. Нужен свежий Python.')
    plan = build_plan()
    _verify_frozen_source(ROOT)
    parent = Path(output_parent or ROOT/'verification').resolve()
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='chat_acceptance_', dir=parent))
    files = list(ROOT.glob('*.py')) + [ROOT/p for p in SOURCES] + [Path(__file__).resolve()]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    versions = {}
    for package in ['gradio', 'numpy', 'pandas', 'plotly', 'requests', 'llama-cpp-python']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {'scope': 'Real loaded LlamaBrain + production Gradio Python handler; no browser/HTTP automation.',
                'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
                'head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'runner_origin': globals().get('RUNNER_ORIGIN', 'local_checkout'),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'model_repo': ac.MODEL_REPO, 'model_stem': ac.MODEL_STEM,
                'n_ctx': getattr(brain, '_n_ctx', None), 'versions': versions, 'source_sha256': hashes,
                'supermag_configured': bool(os.environ.get('SUPERMAG_LOGON'))}
    redact = Redactor([os.environ.get('SUPERMAG_LOGON', '')])
    _write(run_dir/'runtime.json', metadata, redact)
    try:
        report = _execute_plan(brain, plan, run_dir, scope=metadata['scope'], logon=os.environ.get('SUPERMAG_LOGON', ''))
    finally:
        changed = [name for name, h in hashes.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != h]
        _write(run_dir/'source_check.json', {'changed_during_run': changed, 'unchanged': not changed}, redact)
        archive_path = run_dir.with_suffix('.zip')
        # Lightweight evidence bundle. HTML charts and produced project ZIPs stay
        # in their phase folders; do not duplicate hundreds of MB into this ZIP.
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(run_dir.rglob('*')):
                if path.is_file() and path.suffix in {'.json', '.md', '.log', '.csv'}:
                    bundle.write(path, path.relative_to(run_dir).as_posix())
        print(f'Архив ответов, журналов и CSV: {archive_path}', flush=True)
    return {'run_dir': str(run_dir), 'archive': str(archive_path), 'report': report}


if __name__ == '__main__':
    print(json.dumps([{'phase': p['id'], 'cards': len(p['cases']),
                       'prompts': sum(bool(c.get('prompt')) for c in p['cases'])} for p in build_plan()], indent=2))
