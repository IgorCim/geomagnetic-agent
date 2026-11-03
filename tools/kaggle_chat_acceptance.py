# Одна дополнительная ячейка ПОСЛЕ загрузки brain (ячейка 5).
# Получает только проверенный вспомогательный runner, не обновляет код агента.
from pathlib import Path
import hashlib, runpy
from urllib.request import urlopen
import agent_core as ac

assert 'brain' in globals() and isinstance(brain, ac.LlamaBrain), 'Сначала загрузите brain в ячейке 5.'
qa_repo = Path(ac.__file__).resolve().parent
qa_url = 'https://raw.githubusercontent.com/IgorCim/geomagnetic-agent/3b8666c00127e241daaf15190e637ce325f0c9b8/tools/run_chat_acceptance.py'
qa_sha256 = 'a6a0422e161fcf8f77f9cbb071e9864b26f3073806d9679a2aaebcc96c5bf7eb'
qa_script = qa_repo / 'tools' / 'run_chat_acceptance.py'
if not qa_script.is_file() or hashlib.sha256(qa_script.read_bytes()).hexdigest() != qa_sha256:
    with urlopen(qa_url, timeout=30) as qa_response:
        qa_code = qa_response.read()
    assert hashlib.sha256(qa_code).hexdigest() == qa_sha256, 'SHA256 runner не совпал; запуск отменён.'
    qa_script = qa_repo / 'verification' / ('qa_runner_' + qa_sha256[:12] + '.py')
    qa_script.parent.mkdir(parents=True, exist_ok=True)
    qa_script.write_bytes(qa_code)
qa_module = runpy.run_path(str(qa_script), init_globals={'RUNNER_ORIGIN': qa_url})
qa_module['_verify_frozen_source'](qa_repo)
if 'demo' in globals():
    demo.close()
qa_result = qa_module['run_acceptance'](brain)
print('Все серии записаны. Это не автоматический PASS. Результаты:', qa_result['run_dir'])
print('ZIP с ответами, журналами и CSV:', qa_result['archive'])
