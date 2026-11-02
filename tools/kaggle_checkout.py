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
    'geomag_coords', 'physics_engine', 'chart_naming',
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
os.chdir(DIR)
sys.path.insert(0, str(DIR))
print('SOURCE VERIFIED:', len(hashes), 'tracked files')
print('GitHub branch:', BRANCH)
print('HEAD:', HEAD)
print('DIR:', DIR)
print('MODELS:', MODELS)
print('Дальше выполняй ячейку 2. После загрузки brain код не обновлять.')
