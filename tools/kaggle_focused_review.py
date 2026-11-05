# Run after cells 1–5 from the same checkout. Does not reload brain or update code.
from pathlib import Path
import runpy
import agent_core as ac
import gradio as gr

assert 'brain' in globals() and isinstance(brain, ac.LlamaBrain), 'Сначала выполните ячейки 1–5 в свежем Python.'
qa_root = Path(ac.__file__).resolve().parent
qa_script = qa_root/'tools/run_focused_review.py'
assert qa_script.is_file(), 'Нужна версия проекта с tools/run_focused_review.py; после brain код не обновлять.'
if 'demo' in globals():
    demo.close()
if 'qa_download' in globals():
    qa_download.close()
qa_focus = runpy.run_path(str(qa_script))['run'](brain, groups=globals().get('QA_FOCUS_GROUPS'))
with gr.Blocks() as qa_download:
    gr.File(value=qa_focus['archive'], label='Скачать ответы, журналы и CSV целевого прогона')
qa_download.launch(share=True, inline=False, allowed_paths=[qa_focus['archive']])
