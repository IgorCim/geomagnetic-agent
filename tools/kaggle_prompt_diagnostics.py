# Дополнительная ячейка после загрузки brain. Код агента не обновляет.
# Перед запуском задайте QA_SUPERMAG_LOGON в самой Kaggle-ячейке.
from pathlib import Path
import hashlib
import runpy
from urllib.request import urlopen
import agent_core as ac

assert 'brain' in globals() and isinstance(brain, ac.LlamaBrain), 'Сначала загрузите brain в ячейке 5.'
qa_repo = Path(ac.__file__).resolve().parent
qa_files = [
    ('3b8666c00127e241daaf15190e637ce325f0c9b8', 'run_chat_acceptance.py',
     'a6a0422e161fcf8f77f9cbb071e9864b26f3073806d9679a2aaebcc96c5bf7eb'),
    ('e09655bfbfbf86d662f35efa4e936715f7d36911', 'run_prompt_diagnostics.py',
     '0fb53c93e10d9d8ba27d00c5503191b150828f86a6043080d4c2635e2bbd7abc'),
]
qa_paths = []
for qa_revision, qa_name, qa_hash in qa_files:
    qa_url = f'https://raw.githubusercontent.com/IgorCim/geomagnetic-agent/{qa_revision}/tools/{qa_name}'
    with urlopen(qa_url, timeout=30) as qa_response:
        qa_bytes = qa_response.read()
    assert hashlib.sha256(qa_bytes).hexdigest() == qa_hash, 'SHA256 не совпал; запуск отменён.'
    qa_path = qa_repo/'verification'/('diagnostic_' + qa_hash[:12] + '.py')
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_bytes(qa_bytes)
    qa_paths.append(qa_path)
qa_helper = runpy.run_path(str(qa_paths[0]))
qa_helper['_verify_frozen_source'](qa_repo)
qa_runner = runpy.run_path(str(qa_paths[1]))
qa_pairs = qa_runner['build_pairs'](qa_helper)
print('Пар запросов:', len(qa_pairs)//2, '; сообщений с подготовкой и SuperMAG:',
      sum(len(p['setup'])+1 for p in qa_pairs)+1, flush=True)
if 'demo' in globals():
    demo.close()
qa_diagnostic_result = qa_runner['run_diagnostics'](
    brain, helper_path=qa_paths[0], logon=globals().get('QA_SUPERMAG_LOGON', ''), repeats=1)

# Настоящая кнопка Gradio для скачивания ZIP, без ссылки на страницу Kaggle.
import gradio as gr
with gr.Blocks() as qa_download:
    gr.File(value=qa_diagnostic_result['archive'], label='Скачать ZIP: ответы, журналы и CSV')
qa_download.launch(share=True, inline=False, allowed_paths=[qa_diagnostic_result['archive']])
