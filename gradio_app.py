"""Stage 4 -- Gradio chat interface for the geomagnetic agent.

Talks to the Stage 3 brain through :func:`agent_core.run_agent` and draws the
figures it produced straight into the page.

Four things this file gets right that a naive version does not
----------------------------------------------------------------

1. **The model is loaded exactly once.** ``load_brain()`` runs in :func:`main`
   before the server exists, and the resulting object is closed over by the chat
   handler. Loading it per message would push 4.7 GB into VRAM on every turn and
   OOM the T4 on the second click.

2. **Plots go through ``gr.Plot``, not ``gr.HTML``.** This one is worth reading.
   The obvious approach -- read the Plotly HTML file and hand it to ``gr.HTML``
   -- silently does nothing: Gradio injects component content with ``innerHTML``,
   and browsers refuse to execute ``<script>`` tags inserted that way, so the
   plotly.js bundle never boots and the user gets a blank pane. Gradio even
   warns about it at runtime. Instead the figure is recovered from the saved
   document and handed to ``gr.Plot`` as a real Plotly object, which Gradio
   renders natively in the browser.

3. **The payload is 80x smaller.** A saved figure is a ~4.9 MB HTML file
   (plotly.js is inlined). Only its ``data``/``layout`` are extracted, about
   0.06 MB, so a turn costs 60 KB over the wire instead of 5 MB.

4. **The handler never touches global state,** so the whole loop is testable
   with a scripted brain and no GPU.

``gr.Blocks`` is used rather than ``gr.ChatInterface`` because a Gradio 6
``Chatbot`` sanitises HTML and accepts only text, images and files -- there is
no way to put a live figure inside the conversation at all.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

# The banner below uses emoji, and a stock Windows console is cp1251 -- encoding
# one raises UnicodeEncodeError and kills the script before any real work
# happens. Colab is already UTF-8, so this is a no-op there.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

# Allow `python gradio_app.py` from the project root as well as from a Colab
# cell that only put the directory on sys.path.
_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import agent_core as core  # noqa: E402  (import after the sys.path fix)

run_agent = core.run_agent
load_brain = core.load_brain
is_error = core.is_error


def _plotly():
    """Import plotly lazily -- it is only needed once a figure exists."""
    import plotly.graph_objects as go

    return go


# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #
#: How many saved figures to remember for the download list.
MAX_PLOTS_IN_PANEL = 3

#: ``run_agent`` is stateless -- every call is an independent query with a fresh
#: message list. To make follow-ups like "and the same for 11 September" work, a
#: short recap of the previous turns is prepended to the query. Set False for
#: strict one-shot behaviour.
USE_HISTORY_CONTEXT = True
HISTORY_TURNS = 4

#: `GRADIO_SHARE=0` keeps the server local (useful for debugging in Colab);
#: `GRADIO_DEBUG=0` quiets the launch banner.
SHARE = os.environ.get("GRADIO_SHARE", "1").strip().lower() not in ("0", "false", "no")
DEBUG = os.environ.get("GRADIO_DEBUG", "1").strip().lower() not in ("0", "false", "no")
VERBOSE = os.environ.get("GRADIO_VERBOSE", "0").strip().lower() not in ("0", "false", "no", "")

_EXAMPLES = [
    "Скачай данные по станции Иркутск (IRT) за 10 сентября 2024, посчитай H и построй график.",
    "Сравни 10 и 11 сентября 2024 по станции IRT: построй наложенные графики H.",
    "Найди аномалии по станции NVS за 10 сентября 2024 с порогом 3 сигмы.",
]


# --------------------------------------------------------------------------- #
# figure recovery
# --------------------------------------------------------------------------- #
def _scan_json_value(text: str, index: int) -> tuple[str, int]:
    """Return the JSON value starting at ``text[index]`` and the index after it.

    A hand-rolled scanner is needed because the figure inside a Plotly document
    is a single ``Plotly.newPlot("id", [...], {...}, {...})`` call, not a clean
    JSON file, and it contains braces inside strings (hovertemplates and the
    like) that defeat a regex.
    """
    while index < len(text) and text[index] in " \t\r\n,":
        index += 1
    start = index
    if index >= len(text):
        raise ValueError("no JSON value at offset")

    char = text[index]
    if char == '"':
        index += 1
        while index < len(text):
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == '"':
                return text[start : index + 1], index + 1
            index += 1
        raise ValueError("unterminated string")

    if char in "[{":
        depth = 0
        while index < len(text):
            current = text[index]
            if current == '"':
                _, index = _scan_json_value(text, index)
                continue
            if current in "[{":
                depth += 1
            elif current in "]}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1], index + 1
            index += 1
        raise ValueError("unbalanced brackets")

    while index < len(text) and text[index] not in ",]}) \t\r\n":
        index += 1
    return text[start:index], index


def figure_from_plot_file(path: str) -> Any | None:
    """Rebuild a Plotly figure from a saved ``.html`` document.

    Returns ``None`` (never raises) if the file is missing or unparseable, so a
    broken figure degrades to a download link instead of killing the turn.
    """
    try:
        html = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    marker = html.find("Plotly.newPlot(")
    if marker < 0:
        return None

    try:
        cursor = marker + len("Plotly.newPlot(")
        _, cursor = _scan_json_value(html, cursor)  # the div id
        data_text, cursor = _scan_json_value(html, cursor)  # traces
        layout_text, _ = _scan_json_value(html, cursor)  # layout
        figure = json.loads(data_text)
        layout = json.loads(layout_text)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None

    if not isinstance(figure, list) or not figure:
        return None

    return _plotly().Figure(data=figure, layout=layout)


def _format_size(path: str) -> str:
    try:
        return f"{Path(path).stat().st_size / 1e6:.1f} MB"
    except OSError:
        return "?"


def _files_markdown(entries: list[tuple[str, str]]) -> str:
    """List every figure produced so far, newest first, for later download."""
    if not entries:
        return ""
    lines = ["### Сохранённые графики", "", "| Вопрос | Файл | Размер |", "|---|---|---|"]
    for question, path in entries:
        short = question.replace("|", "/")[:70]
        name = Path(path).name
        lines.append(f"| {short} | `{name}` | {_format_size(path)} |")
    lines.append("")
    lines.append(f"Файлы лежат в папке `plots/` рядом с проектом.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def _format_tool_log(tool_calls: list[dict[str, Any]] | None) -> str:
    """Render the agent's tool calls as a readable markdown log.

    Steps look like ``{"round", "tool", "arguments", "ok"}``.
    """
    if not tool_calls:
        return ""
    lines = ["### Журнал вызовов инструментов", ""]
    for step in tool_calls:
        name = step.get("tool") or "(без имени)"
        mark = "OK" if step.get("ok") else "**ошибка**"
        lines.append(f"- раунд {step.get('round', '?')} · `{name}` — {mark}")
        arguments = step.get("arguments") or {}
        if isinstance(arguments, dict) and arguments:
            rendered = ", ".join(f"`{k}={v}`" for k, v in arguments.items())
            lines.append(f"  - аргументы: {rendered}")
    return "\n".join(lines)


def _message_text(value: Any) -> str:
    """Flatten a Gradio message body to plain text.

    A ``Chatbot`` value arrives in OpenAI shape, but Gradio 6 also normalises
    ``content`` into a list of blocks like ``[{"text": "...", "type": "text"}]``.
    Both forms have to be handled or the follow-up recap degenerates into the
    literal text ``[{'text': ...}]``.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _message_text(value.get("text", ""))
    if isinstance(value, (list, tuple)):
        return " ".join(_message_text(block) for block in value).strip()
    return ""


def _context_from_history(history: list[Any] | None) -> str:
    """Build a short recap of previous turns for the stateless agent."""
    if not USE_HISTORY_CONTEXT or not history:
        return ""
    turns: list[str] = []
    for item in history[-HISTORY_TURNS * 2 :]:
        question = answer = ""
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:  # (user, assistant)
                question, answer = _message_text(item[0]), _message_text(item[1])
            elif isinstance(item, dict):  # {"role", "content"}
                content = _message_text(item.get("content", ""))
                if item.get("role") == "user":
                    question = content
                else:
                    answer = content
        except (AttributeError, TypeError):
            continue
        if question:
            turns.append(f"- учёный спросил: {question[:300]}")
        if answer:
            turns.append(f"- ты ответил: {answer[:300]}")
    if not turns:
        return ""
    return (
        "Контекст предыдущих реплик (для справки, данные из него не годятся — "
        "любые числа всё равно нужно получить инструментом):\n"
        + "\n".join(turns)
        + "\n\nНовый запрос:\n"
    )


# --------------------------------------------------------------------------- #
# the chat handler
# --------------------------------------------------------------------------- #
def make_chat_handler(brain: Any):
    """Bind ``brain`` into a Gradio handler. The brain is captured once."""

    def chat_and_plot(
        message: str,
        history: list[Any] | None,
        plots_state: list[tuple[str, str]] | None,
    ) -> tuple[list[Any], Any, str, list[tuple[str, str]], str]:
        history = list(history or [])
        plots_state = list(plots_state or [])

        question = (message or "").strip()
        if not question:
            return history, None, "", plots_state, ""

        if VERBOSE:
            print(f"[web] запрос: {question[:120]}")

        query = _context_from_history(history) + question

        def untouched(extra_log: str = "") -> tuple:
            panel = _files_markdown(plots_state)
            return history, None, extra_log, plots_state, ""

        try:
            result = run_agent(query, brain=brain)
        except Exception as exc:  # one bad turn must not kill the server
            traceback.print_exc()
            detail = f"{type(exc).__name__}: {exc}"
            history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"Внутренняя ошибка агента: {detail}"},
            ]
            return history, None, f"**Ошибка:** {detail}", plots_state, ""

        if is_error(result):
            code = result.get("error", "unknown")
            detail = result.get("message", "агент вернул ошибку")
            history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"Ошибка `{code}`: {detail}"},
            ]
            return history, None, f"**{code}**: {detail}", plots_state, ""

        answer = str(result.get("text", "")).strip() or "(пустой ответ)"
        if result.get("stop_reason") == "tool_budget_exhausted":
            answer += "\n\n_(лимит вызовов инструментов исчерпан, ответ неполный)_"

        # Record every figure, then draw the newest one.
        for path in result.get("plots", []) or []:
            plots_state.insert(0, (question[:80], str(path)))
        plots_state = plots_state[:MAX_PLOTS_IN_PANEL]

        newest = plots_state[0][1] if plots_state else None
        figure = figure_from_plot_file(newest) if newest else None
        if newest and figure is None:
            # The file exists but could not be parsed -- still offer the link.
            print(f"[web] не удалось разобрать график: {newest}")

        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return history, figure, _format_tool_log(result.get("tool_calls")), plots_state, ""

    return chat_and_plot


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_demo(brain: Any) -> gr.Blocks:
    """Assemble the interface around an already-loaded brain."""
    chat_and_plot = make_chat_handler(brain)

    with gr.Blocks(title="Геомагнитный агент", fill_height=True) as demo:
        gr.Markdown(
            "# 🧲 Геомагнитный ИИ-агент\n"
            "Спросите про данные INTERMAGNET любой станции. Агент сам вызовет "
            "инструменты, посчитает компоненты и построит графики."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Диалог", height=460)
                with gr.Row():
                    textbox = gr.Textbox(
                        label="Ваш вопрос",
                        placeholder="Например: построй график H по станции IRT за 10 сентября 2024",
                        lines=2,
                        scale=5,
                    )
                    send = gr.Button("Спросить", variant="primary", scale=1)
                gr.Examples(
                    _EXAMPLES,
                    inputs=textbox,
                    label="Примеры вопросов",
                    example_labels=["График H", "Сравнение двух дней", "Поиск аномалий"],
                )

            with gr.Column(scale=2):
                gr.Markdown("### График")
                plot_view = gr.Plot(label="")
                plot_files = gr.Markdown(
                    "Здесь появятся ссылки на сохранённые графики. Спросите агента "
                    "о данных по станции — например, про Иркутск (IRT) за 10 сентября 2024."
                )

        with gr.Accordion("Журнал вызовов инструментов", open=False):
            tool_log = gr.Markdown(
                "Здесь появится журнал: какие инструменты вызвал агент "
                "и с какими аргументами."
            )

        plots_state = gr.State([])

        def submit(text, history, state):
            if not (text or "").strip():
                gr.Info("Введите вопрос.")
                return gr.skip()
            history, figure, log, state, _ = chat_and_plot(text, history, state)
            return history, figure, log, state, _files_markdown(state), ""

        send.click(
            submit,
            inputs=[textbox, chatbot, plots_state],
            outputs=[chatbot, plot_view, tool_log, plots_state, plot_files, textbox],
        )
        textbox.submit(
            submit,
            inputs=[textbox, chatbot, plots_state],
            outputs=[chatbot, plot_view, tool_log, plots_state, plot_files, textbox],
        )

    return demo


def main() -> None:
    # Load the brain exactly once, before the server exists.
    print("⏳ Загружаю Qwen 2.5 7B в VRAM (один раз на весь сеанс)...")
    brain = load_brain()
    if is_error(brain):
        print(f"\n❌ Мозг не загрузился: {brain.get('error')}: {brain.get('message')}")
        print(f"   совет: {brain.get('hint', 'проверьте ячейку установки CUDA и наличие GPU')}")
        raise SystemExit(1)
    print("✅ Мозг в VRAM.")

    demo = build_demo(brain)
    print("🌐 Запускаю интерфейс...")
    demo.launch(
        share=SHARE,
        debug=DEBUG,
        server_name="0.0.0.0",
        prevent_thread_lock=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
