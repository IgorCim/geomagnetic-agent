# Приёмка текущей версии на Kaggle

Цель — получить доказательства именно текущего набора файлов, настоящей Qwen и поведения UI. Старый six-source PASS не переносится на новую версию. Из этого документа собран `kaggle_portfolio_acceptance.ipynb`; код ниже — последовательные ячейки.

Нужны Kaggle GPU и Internet ON. Код скачивается из GitHub; Dataset/Input и source ZIP не нужны. Новая папка запуска не удаляет прежние models, проекты и исторические прогоны. Все действия ниже выполняет пользователь в своей Kaggle-сессии; локально эти GPU-ячейки не запускались.

Версия последовательности: 02.11.2026. Используйте её целиком в свежем kernel. Сопоставление со старыми ячейками и порядок проверки контекста: [KAGGLE_MIGRATION.md](KAGGLE_MIGRATION.md). `brain` загружается один раз; после этого никаких clone/pull/reset или удаления модулей.

## 1. Получить текущую версию из GitHub

Первый запуск — в свежем kernel. Вставьте всю ячейку целиком. Она клонирует master в новую отдельную папку, печатает полный HEAD и сохраняет SHA-256 отслеживаемых файлов. Старые папки, проекты и веса не удаляются. Каждая повторная попытка до загрузки модели использует новое имя папки автоматически.

После загрузки brain эту ячейку не повторять: начните свежий kernel для новой версии кода. SOURCE_MANIFEST.json создаётся самой ячейкой из Git checkout; искать его в /kaggle/input не нужно.

```python
# Kaggle cell 1: obtain one clean GitHub revision without deleting old work.
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile

PROJECT_MODULES = {
    'agent_core', 'agent_planner', 'agent_state', 'project_store', 'gradio_app',
    'supermag_loader', 'iswa_loader', 'intermagnet_loader', 'space_weather_loader',
    'noaa_swpc_loader', 'geomag_analyzer', 'geomag_math', 'geomag_plotter',
    'geomag_coords', 'physics_engine', 'chart_naming', 'chat_session',
}
loaded = sorted(PROJECT_MODULES.intersection(sys.modules))
assert not loaded and 'brain' not in globals(), (
    'Перезапусти kernel перед ячейкой 1: уже загружены модули/brain. '
    'Файлы проекта и веса удалять не нужно. Модули: ' + ', '.join(loaded)
)

REPO_URL = 'https://github.com/IgorCim/geomagnetic-agent.git'
BRANCH = 'master'
WORKING = Path('/kaggle/working')
assert WORKING.is_dir(), 'Эта ячейка предназначена для Kaggle'
# Every attempt has its own directory. Existing checkouts, models and outputs stay intact.
DIR = Path(tempfile.mkdtemp(prefix='geomagnetic_github_', dir=WORKING))
cloned = subprocess.run(
    ['git', 'clone', '--branch', BRANCH, '--single-branch', REPO_URL, str(DIR)],
    capture_output=True, text=True,
)
if cloned.returncode:
    raise RuntimeError('Не удалось скачать GitHub. Проверь Internet ON в Kaggle.\n' + cloned.stderr)

def checkout_git(*args):
    return subprocess.check_output(['git', '-C', str(DIR), *args], text=True).strip()

HEAD = checkout_git('rev-parse', 'HEAD')
required = [
    'agent_core.py', 'stress_300.py', 'gradio_app.py', 'requirements-dev.txt',
    'tools/verify_offline.py', 'tools/verify_gradio.py', 'docs/acceptance_protocol.json',
]
missing = [name for name in required if not (DIR / name).is_file()]
assert not missing, f'GitHub содержит другую поставку. HEAD={HEAD}; отсутствуют: {missing}'
protocol = json.loads((DIR / 'docs/acceptance_protocol.json').read_text(encoding='utf-8'))
assert protocol.get('version') == 'github-20261102-v1', f'Другая версия протокола: {protocol}'
assert not checkout_git('status', '--porcelain'), 'Новый checkout неожиданно содержит изменения'

tracked = subprocess.check_output(['git', '-C', str(DIR), 'ls-files', '-z']).decode('utf-8').split('\0')
hashes = {}
for name in filter(None, tracked):
    path = DIR / name
    assert path.is_file() and path.resolve().is_relative_to(DIR.resolve()), name
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
source = {
    'source': 'github', 'git_revision': HEAD, 'git_branch': BRANCH,
    'protocol_version': protocol['version'], 'dirty_tree': False, 'sha256': hashes,
}
(DIR / 'SOURCE_MANIFEST.json').write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')

stem = 'qwen2.5-14b-instruct-q4_k_m'
model_candidates = []
if os.environ.get('GEOMAG_MODELS_DIR'):
    model_candidates.append(Path(os.environ['GEOMAG_MODELS_DIR']))
model_candidates.extend([
    WORKING / 'geomagnetic-agent/models', WORKING / 'geomagnetic_agent/models',
    WORKING / 'geomagnetic-models',
])
MODELS = next((p.resolve() for p in model_candidates if p.is_dir() and any(p.glob(stem + '*.gguf'))),
              WORKING / 'geomagnetic-models')
MODELS.mkdir(parents=True, exist_ok=True)
os.environ['GEOMAG_MODELS_DIR'] = str(MODELS)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GEOMAG_OFFLINE'] = '0'
os.environ['GEOMAG_DEBUG_PARSER'] = '0'
os.environ['GEOMAG_DEBUG_SCRATCHPAD'] = '0'
os.chdir(DIR)
sys.path.insert(0, str(DIR))
print('SOURCE VERIFIED:', len(hashes), 'tracked files')
print('GitHub branch:', BRANCH)
print('HEAD:', HEAD)
print('DIR:', DIR)
print('MODELS:', MODELS)
print('Дальше выполняй ячейку 2. После загрузки brain код не обновлять.')
```

## 2. Зависимости и оборудование

Wheel — официальный адрес из прежнего Kaggle/Colab сценария проекта. Если wheel не подходит обновлённому образу Kaggle, остановиться и зафиксировать ошибку; CPU fallback не выдавать за GPU-прогон.

```python
import subprocess, sys, json
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt'], check=True)
probe = subprocess.run([sys.executable, '-c',
    'import json, llama_cpp; from llama_cpp import llama_cpp as lc; '
    'print(json.dumps({"version": llama_cpp.__version__, "gpu": bool(lc.llama_supports_gpu_offload())}))'],
    capture_output=True, text=True)
try:
    existing = json.loads(probe.stdout.strip()) if probe.returncode == 0 else {}
except ValueError:
    existing = {}
if existing.get('version') == '0.3.35' and existing.get('gpu'):
    print('CUDA llama-cpp 0.3.35 уже установлен; повторная загрузка wheel не нужна')
else:
    assert 'llama_cpp' not in sys.modules, 'Перезапусти kernel перед заменой бинарного llama-cpp'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
        'https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35-cu121/llama_cpp_python-0.3.35-py3-none-manylinux_2_35_x86_64.whl'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'diskcache'], check=True)
import torch
from llama_cpp import llama_cpp as lc
assert torch.cuda.is_available(), 'Включите GPU в настройках Kaggle'
assert lc.llama_supports_gpu_offload(), 'llama.cpp не поддерживает GPU: остановить приёмку'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GB:', torch.cuda.get_device_properties(0).total_memory / 1e9)
```

## 3. Офлайн-регрессии именно этой поставки

```python
import subprocess, sys
subprocess.run([sys.executable, '-B', 'tools/verify_offline.py'], check=True)
subprocess.run([sys.executable, '-B', 'tools/verify_gradio.py'], check=True)
subprocess.run([sys.executable, '-B', 'tools/probe_sources.py'], check=True)
```

Если регрессии не прошли, видео успешного продукта не записывать. HTTP 200 каталога не считать data-test. Новый HTTP-отчёт из Kaggle нужен отдельно от локального: сеть и маршруты отличаются.

## 4. SuperMAG: показать текущий реальный ответ без логина в кадре

Если логин сохранён в Kaggle Secrets под именем SUPERMAG_LOGON, его можно передать серверу; значение не печатать. Если секрета нет, missing_logon означает отсутствие настройки, а не воспроизведение 404. Ниже выбран один день, а не 30-дневное окно, превышающее лимит клиента.

```python
import os, json
if not os.environ.get('SUPERMAG_LOGON', '').strip():
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ['SUPERMAG_LOGON'] = UserSecretsClient().get_secret('SUPERMAG_LOGON')
    except Exception:
        pass
configured = bool(os.environ.get('SUPERMAG_LOGON', '').strip())
print('SuperMAG logon configured; value hidden' if configured else 'SuperMAG not configured; authenticated request skipped')
import supermag_loader as sm
if os.environ.get('SUPERMAG_LOGON'):
    sm_result = sm.fetch_supermag_data('2024-05-10', '2024-05-10', data_type='indices', logon=os.environ['SUPERMAG_LOGON'], timeout=20, retries=1)
    if sm.is_error(sm_result):
        print(json.dumps(sm_result, ensure_ascii=False, indent=2))
    else:
        print('SuperMAG rows:', len(sm_result), 'columns:', list(sm_result.columns))
        print('UTC coverage:', sm_result['timestamp'].min(), sm_result['timestamp'].max())
```

Текущий результат не предполагать заранее: записать точный error/status либо реальные данные. В новой версии URL-логин маскируется, но перед публикацией видео всё равно просмотреть кадры. Не выводить переменные окружения целиком.

## 5. Загрузить настоящую модель и записать окружение

```python
import agent_core as ac
import importlib.metadata as metadata
import hashlib, json, platform, logging, os
from pathlib import Path
ac.OFFLINE = False
os.environ['GEOMAG_DEBUG_PARSER'] = '0'
os.environ['GEOMAG_DEBUG_SCRATCHPAD'] = '0'
brain = ac.load_brain(n_gpu_layers=-1, n_ctx=16384, verbose=False)
logging.getLogger('llama-cpp-python').setLevel(logging.WARNING)
assert not ac.is_error(brain), brain
assert isinstance(brain, ac.LlamaBrain) and not isinstance(brain, ac.ScriptedBrain)
assert ac.MODEL_REPO == 'Qwen/Qwen2.5-14B-Instruct-GGUF'
evidence = Path('portfolio_evidence')
evidence.mkdir(exist_ok=True)
runtime = {'python': platform.python_version(), 'model_repo': ac.MODEL_REPO,
           'model_stem': ac.MODEL_STEM, 'brain': type(brain).__name__,
           'n_ctx_requested': 16384, 'n_ctx_actual': brain._n_ctx,
           'source_manifest_sha256': hashlib.sha256(Path('SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
           'versions': {name: metadata.version(name) for name in ['gradio','numpy','pandas','plotly','requests','llama-cpp-python']}}
(evidence / 'runtime.json').write_text(json.dumps(runtime, indent=2))
print(json.dumps(runtime, indent=2))
```

## 6. Короткая живая проверка до UI

Это настоящий run_agent; ScriptedBrain и подменённые загрузчики запрещены. Записать stdout и результат. После результата проверьте сам HTML и таблицы, а не только ok=True.

```python
import json
import project_store as ps
store = ac.FrameStore(workspace=ps.Workspace(root=evidence / 'preflight_projects'))
prompt = ('Станция IRT, 2024-09-10. Загрузи реальные данные INTERMAGNET, вычисли H, D, I. '
          'Посчитай min/max/mean/median/std H и построй H. Укажи источник, дату и точный handle.')
result = ac.run_agent(prompt, brain=brain, store=store, verbose=True)
(evidence / 'live_preflight.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
print(result.get('text'))
required = {'fetch_observatory_data', 'calculate_derived_components', 'get_statistics', 'plot_components'}
successful = {c['tool'] for c in result.get('tool_calls', []) if c.get('ok')}
assert required <= successful, result
assert result.get('plots') and all(Path(p).is_file() for p in result['plots'])
assert result.get('text', '').strip()
```

## 7. Полный 300-chart gate отдельной ячейкой

Эта ячейка потенциально длительная. В stress_300.py три матрицы, девять стадий, контекстные источники и security/export проверки. Каталог запуска уникальный. При нынешнем 404 SuperMAG strict six-source gate может завершиться FAIL; это корректно показывается как текущий результат. Не заменять его старой зелёной строкой. Пропуск SuperMAG также нельзя называть six-source PASS.

```python
import contextlib, os, sys, traceback
from pathlib import Path

class EvidenceTee:
    def __init__(self, stream, file): self.stream, self.file = stream, file
    def write(self, text):
        original_length = len(text)
        secret = os.environ.get('SUPERMAG_LOGON', '')
        if secret: text = text.replace(secret, '[redacted]')
        self.stream.write(text); self.file.write(text); self.file.flush()
        return original_length
    def flush(self): self.stream.flush(); self.file.flush()

exports_before_stress = ps.EXPORTS_ROOT
stress_ns = {'__name__': '__main__', 'brain': brain, 'DIR': DIR}
stress_status = {'status': 'RUNNING', 'run_dir': None}
with (evidence / 'stress_300.log').open('w', encoding='utf-8') as log:
    with contextlib.redirect_stdout(EvidenceTee(sys.stdout, log)), contextlib.redirect_stderr(EvidenceTee(sys.stderr, log)):
        try:
            exec(compile(Path('stress_300.py').read_text(encoding='utf-8-sig'), 'stress_300.py', 'exec'), stress_ns)
            assert stress_ns.get('r_smag') is not None, 'Six-source gate: SuperMAG пропущен; настрой SUPERMAG_LOGON и повтори. 300 графиков сами по себе не означают проверку шести источников.'
            stress_status['status'] = 'PASS'
        except BaseException as exc:
            stress_status.update(status='FAIL', exception=type(exc).__name__)
            traceback.print_exc()
            print('CURRENT STRICT GATE: FAIL; inspect per-source and matrix results above')
            raise
        finally:
            ps.EXPORTS_ROOT = exports_before_stress
            if 'RUN_DIR' in stress_ns:
                RUN_DIR = Path(stress_ns['RUN_DIR'])
                stress_status['run_dir'] = str(RUN_DIR)
            (evidence / 'stress_status.json').write_text(json.dumps(stress_status, indent=2))
            stress_ns.clear()  # release stress-only tables before starting Gradio
```

Отдельно проверить: каждый график соответствует station/day; нет stale/дубликатов; raw/derived CSV и встроенные Plotly values совпадают; UTC/оси/единицы корректны; ZIP содержит ожидаемые 300 HTML. Минимальная научная выборка — A/B/C, высокие/средние широты, штормовой/спокойный день, место с NaN. Для утверждения обо всех 300 нужны автоматическая численная сверка всех traces и сохранённые отчёты, а не только имена/счётчики. Исходный stress gate такой сверки не делает.

## 8. Gradio с настоящей Qwen

Для автоматического последовательного прогона инструментов → 20 ходов → формул A–E вместо ручной отправки используйте отдельную [одну ячейку QA_AUTORUN](QA_AUTORUN.md) после ячейки 5. Она не требует запуска стресс-ячейки 7 или браузерной ячейки 8 и не обновляет код поверх brain.

По просьбе пользователя для проверки в Kaggle включена временная публичная ссылка PUBLIC_DEMO=True. Открывать напечатанный адрес в отдельной вкладке; localhost Kaggle недоступен с компьютера пользователя. Ссылка не имеет встроенной авторизации. Не оставлять публичный сервер после показа; не вводить в чат логины или приватные сведения.

```python
import gradio_app as ga
PUBLIC_DEMO = True  # Временная публичная ссылка для проверки; закрыть после показа.
if 'demo' in globals():
    demo.close()
demo = ga.build_demo(brain)
_, local_url, public_url = demo.launch(share=PUBLIC_DEMO, debug=False, prevent_thread_lock=True, inline=False)
print('Открой Gradio в отдельной вкладке:', public_url or local_url)
if PUBLIC_DEMO:
    assert public_url, 'Публичный туннель не создан; смотри ошибку launch выше'
```

Выполнить [VIDEO_RUNBOOK.md](VIDEO_RUNBOOK.md) и 20-ходовую проверку [LONG_CHAT.md](LONG_CHAT.md). Внешние данные реальные; результаты не заменять локальной TestBrain-проверкой. Для нового независимого сценария используйте новую браузерную сессию, не полагайтесь только на очистку transcript.

## 9. Сохранить доказательства

Скачать `portfolio_evidence/`, `verification/` нового прогона и ZIP из конкретного нового RUN_DIR. Если stress дошёл до создания RUN_DIR, его значение есть в stdout. Старые файлы не смешивать с новыми.

```python
import json, zipfile
from pathlib import Path
from IPython.display import FileLink, display
proof_zip = Path('/kaggle/working/portfolio_evidence_20261102.zip')
with zipfile.ZipFile(proof_zip, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for base in [Path('portfolio_evidence'), Path('verification')]:
        if base.is_dir():
            for p in base.rglob('*'):
                if p.is_file(): bundle.write(p, p.as_posix())
    bundle.write('SOURCE_MANIFEST.json')
    if 'RUN_DIR' in globals():
        for p in Path(RUN_DIR).rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.json', '.zip'}:
                bundle.write(p, p.relative_to(DIR).as_posix())
print('Download:', proof_zip, 'bytes:', proof_zip.stat().st_size)
with zipfile.ZipFile(proof_zip) as bundle:
    assert bundle.testzip() is None
display(FileLink(str(proof_zip)))
if 'RUN_DIR' in globals():
    for path in sorted(Path(RUN_DIR).rglob('*.zip')):
        with zipfile.ZipFile(path) as bundle:
            assert bundle.testzip() is None, path
            charts = [n for n in bundle.namelist() if n.endswith('.html') and not n.endswith('index.html')]
            assert len(charts) == len(set(charts)), path
            print(path.name, 'charts:', len(charts), 'bytes:', path.stat().st_size)
        display(FileLink(str(path)))
status_file = evidence / 'stress_status.json'
print('Current strict gate:', json.loads(status_file.read_text()) if status_file.exists() else 'NOT RUN')
```

Перед публикацией проверить архив и видео на секреты. Не отправлять его автоматически третьим лицам. Полный успех портфолио определяется по матрице требований и сохранённым результатам; универсальной гарантии «100% всех возможных запросов» этот протокол не даёт.
