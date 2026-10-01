"""Stage 4 -- Gradio chat interface for the geomagnetic agent.

Talks to the Stage 3 brain through :func:`agent_core.run_agent` and draws the
figures it produced straight into the page.

Version compatibility
---------------------
This file is expected to run on whatever Gradio happens to be installed -- 4.x
and 5.x on Kaggle, 6.x in some Colab images. The one place that matters is
``gr.Chatbot``: on 4.x/5.x it defaults to ``type="tuples"`` and rejects
OpenAI-shaped messages outright with

    Error: 'Data incompatible with tuples format. Each message should be a
    list of length 2.'

so ``type="messages"`` is passed there. Gradio 6 removed the parameter
entirely (messages is the only format), and passing it anyway is a TypeError,
so the kwarg is added only when the installed version actually accepts it.
See :func:`_chatbot_kwargs`.

Design notes
------------

* **The model is loaded exactly once.** ``load_brain()`` runs in :func:`main`
  before the server exists and the object is closed over by the chat handler.
  Loading it per message would push 4.7 GB into VRAM on every turn and OOM.

* **Plots go through ``gr.Plot``, not ``gr.HTML``.** Gradio injects component
  content with ``innerHTML`` and browsers refuse to run ``<script>`` tags
  inserted that way, so inlining a Plotly document renders nothing at all. The
  figure is instead recovered from the saved document and handed over as a real
  Plotly object.

* **A figure that cannot be recovered is never fatal.** The text answer still
  arrives and names the file; see :func:`figure_from_plot_file`.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

# The banner uses emoji and a stock Windows console is cp1251, where encoding
# one raises UnicodeEncodeError and kills the script before any real work.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

# Allow `python gradio_app.py` from the project root as well as from a Colab or
# Kaggle cell that only put the directory on sys.path.
_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import agent_core as core  # noqa: E402  (import after the sys.path fix)

run_agent = core.run_agent
load_brain = core.load_brain
is_error = core.is_error

GRADIO_VERSION = getattr(gr, "__version__", "unknown")


# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #
#: How many saved figures to remember for the download list.
MAX_PLOTS_IN_PANEL = 3

#: ``run_agent`` is stateless -- every call is an independent query with a fresh
#: message list. To make follow-ups like "and the same for 11 September" work, a
#: short recap of the previous turns is prepended. Set False for one-shot mode.
USE_HISTORY_CONTEXT = True
HISTORY_TURNS = 4

#: `GRADIO_SHARE=0` keeps the server local (handy for debugging);
#: `GRADIO_DEBUG=0` quiets the banner; `GEOMAG_DEBUG_PARSER=0` hides parser logs.
SHARE = os.environ.get("GRADIO_SHARE", "1").strip().lower() not in ("0", "false", "no")
DEBUG = os.environ.get("GRADIO_DEBUG", "1").strip().lower() not in ("0", "false", "no")
VERBOSE = os.environ.get("GRADIO_VERBOSE", "0").strip().lower() not in ("0", "false", "no", "")

_EXAMPLES = [
    "Скачай данные по станции Иркутск (IRT) за 10 сентября 2024, посчитай H и построй график.",
    "Сравни 10 и 11 сентября 2024 по станции IRT: построй наложенные графики H.",
    "Найди аномалии по станции NVS за 10 сентября 2024 с порогом 3 сигмы.",
]

#: Told to the user when a figure exists on disk but cannot be drawn inline.
PLOT_FALLBACK = (
    "📊 Интерактивный график сохранён в файл: {name}. "
    "(Скачивание доступно в разделе файлов)."
)


# --------------------------------------------------------------------------- #
# Gradio version compatibility
# --------------------------------------------------------------------------- #
def _chatbot_kwargs() -> dict[str, Any]:
    """Kwargs for ``gr.Chatbot`` that work on Gradio 4, 5 and 6.

    ``type="messages"`` is mandatory on 4.x/5.x, where the default is
    ``"tuples"`` and OpenAI-shaped messages are rejected. Gradio 6 dropped the
    parameter, and passing it raises TypeError.
    """
    try:
        accepted = inspect.signature(gr.Chatbot.__init__).parameters
    except (TypeError, ValueError):
        return {}
    if "type" in accepted:
        return {"type": "messages"}
    return {}


def _plotly():
    """Import plotly lazily -- only needed once a figure exists."""
    import plotly.graph_objects as go

    return go


# --------------------------------------------------------------------------- #
# message helpers
# --------------------------------------------------------------------------- #
def _message_text(value: Any) -> str:
    """Flatten a message body to plain text.

    A ``Chatbot`` value is OpenAI-shaped, but Gradio also normalises ``content``
    into a list of blocks such as ``[{"text": "...", "type": "text"}]``. Both
    forms must be handled or the recap degrades into the literal text
    ``[{'text': ...}]``.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if "text" in value:
            return _message_text(value.get("text"))
        return _message_text(value.get("content", ""))
    if isinstance(value, (list, tuple)):
        return " ".join(_message_text(block) for block in value).strip()
    if value is None:
        return ""
    return str(value).strip()


def normalize_history(history: Any) -> list[dict[str, str]]:
    """Coerce any incoming chatbot value into ``[{"role", "content": str}]``.

    Tolerates tuples-mode input ``[(user, assistant), ...]``, messages-mode
    input, and Gradio's block-list ``content``, so the handler cannot be broken
    by whatever shape the installed Gradio hands it.
    """
    if not history:
        return []
    if isinstance(history, (str, bytes)):
        return [{"role": "user", "content": _message_text(history)}]

    messages: list[dict[str, str]] = []
    for item in history:
        try:
            if isinstance(item, dict):
                role = str(item.get("role") or "assistant")
                if role not in ("user", "assistant", "system"):
                    role = "assistant"
                text = _message_text(item.get("content", item.get("text", "")))
            elif isinstance(item, (list, tuple)):
                # tuples mode: (user, assistant) or a bare message
                if len(item) >= 2 and all(
                    isinstance(part, str) for part in item[:2]
                ):
                    role, text = "user", _message_text(item[0])
                    messages.append({"role": role, "content": text})
                    role, text = "assistant", _message_text(item[1])
                else:
                    role, text = "user", _message_text(item)
            else:
                role, text = "user", _message_text(item)
        except (AttributeError, TypeError, ValueError):
            continue
        if text:
            messages.append({"role": role, "content": text})
    return messages


def _context_from_history(history: list[dict[str, str]] | None) -> str:
    """Build a short recap of previous turns for the stateless agent."""
    if not USE_HISTORY_CONTEXT or not history:
        return ""
    turns: list[str] = []
    for message in history[-HISTORY_TURNS * 2 :]:
        if message.get("role") == "user":
            turns.append(f"- учёный спросил: {message['content'][:300]}")
        elif message.get("role") == "assistant" and message.get("content"):
            turns.append(f"- ты ответил: {message['content'][:300]}")
    if not turns:
        return ""
    return (
        "Контекст предыдущих реплик (для справки, данные из него не годятся — "
        "любые числа всё равно нужно получить инструментом):\n"
        + "\n".join(turns)
        + "\n\nНовый запрос:\n"
    )


def _as_text(result: Any) -> str:
    """Pull the assistant's answer out of whatever ``run_agent`` returned.

    ``bot_response_text`` must always be a string -- a ``None`` reaching
    ``gr.Chatbot`` is a crash, not a cosmetic bug.
    """
    if result is None:
        return "(агент ничего не вернул)"
    if isinstance(result, str):
        return result.strip() or "(пустой ответ)"
    if isinstance(result, dict):
        for key in ("text", "message", "answer", "response", "content", "reason"):
            if key not in result:
                continue
            value = result.get(key)
            if isinstance(value, str):
                if value.strip():
                    return value.strip()
                continue
            if value is not None and not isinstance(value, (dict, list, tuple)):
                return str(value).strip()
        return "(агент не вернул текст)"
    return _message_text(result) or "(пустой ответ)"


def _plot_paths(result: Any) -> list[str]:
    """Best-effort list of figure paths from an agent result."""
    if not isinstance(result, dict):
        return []
    plots = result.get("plots") or result.get("plot") or []
    if isinstance(plots, (str, bytes)):
        return [_message_text(plots)] if plots else []
    if isinstance(plots, (list, tuple)):
        return [_message_text(item) for item in plots if _message_text(item)]
    return []


def _format_tool_log(tool_calls: Any) -> str:
    """Render the agent's tool calls as a readable markdown log."""
    if not isinstance(tool_calls, (list, tuple)) or not tool_calls:
        return ""
    lines = ["### Журнал вызовов инструментов", ""]
    for step in tool_calls:
        if not isinstance(step, dict):
            continue
        name = step.get("tool") or "(без имени)"
        mark = "OK" if step.get("ok") else "**ошибка**"
        lines.append(f"- раунд {step.get('round', '?')} · `{name}` — {mark}")
        arguments = step.get("arguments")
        if isinstance(arguments, dict) and arguments:
            rendered = ", ".join(f"`{k}={v}`" for k, v in arguments.items())
            lines.append(f"  - аргументы: {rendered}")
    return "\n".join(lines) if len(lines) > 2 else ""


# --------------------------------------------------------------------------- #
# figure recovery
# --------------------------------------------------------------------------- #
def _scan_json_value(text: str, index: int) -> tuple[str, int]:
    """Return the JSON value starting at ``text[index]`` and the index after it.

    A scanner is needed because the figure inside a Plotly document is a single
    JavaScript call rather than a clean JSON file, and it contains braces inside
    strings (hovertemplates and the like) that defeat a regular expression.
    """
    while index < len(text) and text[index] in " \t\r\n,":
        index += 1
    start = index
    if index >= len(text):
        raise ValueError("no JSON value at offset")

    char = text[index]
    if char in "\"'":
        quote = char
        index += 1
        while index < len(text):
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == quote:
                return text[start : index + 1], index + 1
            index += 1
        raise ValueError("unterminated string")

    if char in "[{":
        depth = 0
        while index < len(text):
            current = text[index]
            if current in "\"'":
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

    while index < len(text) and text[index] not in ",;()[]{} \t\r\n":
        index += 1
    return text[start:index], index


#: JavaScript entry points that may carry a figure, with the argument position
#: of the layout. ``Plotly.animate`` takes frames before the layout, which is
#: the usual reason a naive two-argument extraction returns nothing.
_PLOT_CALLS = (
    ("Plotly.newPlot(", 2),
    ("Plotly.newPlot2(", 2),
    ("Plotly.react(", 2),
    ("Plotly.animate(", 3),
    ("Plotly.restyle(", -1),
)


def _figure_from_html(html: str) -> Any | None:
    """Try every known way a Plotly figure can be embedded in a document."""
    if not html:
        return None

    # 1. A JavaScript plotting call.
    for marker, layout_position in _PLOT_CALLS:
        cursor = 0
        while True:
            found = html.find(marker, cursor)
            if found < 0:
                break
            cursor = found + len(marker)
            if layout_position < 0:
                continue
            try:
                position = cursor
                _, position = _scan_json_value(html, position)  # target id (arg 0)
                data_text, position = _scan_json_value(html, position)  # traces (arg 1)
                # Skip whatever sits between the traces and the layout, then read
                # the layout itself. `layout_position` counts arguments from the
                # target id, and two of them have already been consumed.
                for _ in range(max(0, layout_position - 2)):
                    _, position = _scan_json_value(html, position)
                layout_text, _ = _scan_json_value(html, position)
                data = json.loads(data_text)
                layout = json.loads(layout_text) if layout_text.strip() else {}
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(data, list) and data:
                return _plotly().Figure(
                    data=data, layout=layout if isinstance(layout, dict) else {}
                )

    # 2. A plain JSON figure, e.g. someone passed a .json to write_html.
    stripped = html.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            candidate = json.loads(stripped)
            if isinstance(candidate, dict) and isinstance(candidate.get("data"), list):
                return _plotly().Figure(data=candidate["data"],
                                        layout=candidate.get("layout") or {})
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    # 3. An embedded {"data": ..., "layout": ...} JSON block.
    for match in ("{\"data\"", "{'data'"):
        found = html.find(match)
        while found >= 0:
            try:
                blob, _ = _scan_json_value(html, found)
                candidate = json.loads(blob)
                if isinstance(candidate, dict) and isinstance(candidate.get("data"), list):
                    return _plotly().Figure(data=candidate["data"],
                                            layout=candidate.get("layout") or {})
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
            found = html.find(match, found + 1)

    return None


def figure_from_plot_file(path: str) -> Any | None:
    """Rebuild a Plotly figure from a saved ``.html`` document.

    Returns ``None`` and never raises for any reason whatsoever -- a missing,
    unreadable, truncated or unfamiliar file must not be able to take down the
    chat, so every failure path collapses to ``None`` and the caller degrades
    to a text answer.
    """
    try:
        html = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - OSError, ValueError, even a bad path type
        return None
    if not html.strip():
        return None
    try:
        return _figure_from_html(html)
    except Exception:  # noqa: BLE001 - a figure must never break the answer
        return None


# --------------------------------------------------------------------------- #
# the chat handler
# --------------------------------------------------------------------------- #
def make_chat_handler(brain: Any):
    """Bind ``brain`` into a Gradio handler. The brain is captured once."""

    def chat_and_plot(
        message: str,
        history: Any,
        plots_state: Any,
    ) -> tuple[list[dict[str, str]], Any, str, list[dict[str, str]], list[str], str]:
        messages = normalize_history(history)
        saved: list[dict[str, str]] = [
            entry for entry in (plots_state or []) if isinstance(entry, dict) and entry.get("path")
        ]

        question = _message_text(message)
        if not question:
            return messages, None, "", saved, [], ""

        if VERBOSE:
            print(f"[web] запрос: {question[:120]}")

        def answer_with(text: str, figure: Any = None, log: str = "") -> tuple:
            text = text if isinstance(text, str) and text.strip() else "(пустой ответ)"
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": text})
            return messages, figure, log, saved, [entry["path"] for entry in saved], ""

        try:
            result = run_agent(_context_from_history(messages) + question, brain=brain)
        except Exception as exc:  # one bad turn must not kill the server
            traceback.print_exc()
            detail = f"{type(exc).__name__}: {exc}"
            return answer_with(f"Внутренняя ошибка агента: {detail}",
                               None, f"**Ошибка:** {detail}")

        if isinstance(result, dict) and is_error(result):
            code = result.get("error", "unknown")
            detail = result.get("message", "агент вернул ошибку")
            return answer_with(f"Ошибка `{code}`: {detail}", None, f"**{code}**: {detail}")

        text = _as_text(result)
        if isinstance(result, dict) and result.get("stop_reason") == "tool_budget_exhausted":
            text += "\n\n_(лимит вызовов инструментов исчерпан, ответ неполный)_"

        # Record the figures, newest first, capped.
        for path in reversed(_plot_paths(result)):
            saved.insert(0, {"question": question[:80], "path": path})
        saved = saved[:MAX_PLOTS_IN_PANEL]

        # Draw the newest figure if one can be recovered, otherwise say where it
        # is on disk and carry on with the text answer.
        figure = None
        note = ""
        for entry in saved:
            if not Path(entry["path"]).is_file():
                continue
            figure = figure_from_plot_file(entry["path"])
            if figure is not None:
                break
            name = Path(entry["path"]).name
            print(f"[web] не удалось разобрать график: {entry['path']}")
            note = "\n\n" + PLOT_FALLBACK.format(name=name)
            break

        return answer_with(text + note, figure, _format_tool_log(
            result.get("tool_calls") if isinstance(result, dict) else None))

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
                chatbot = gr.Chatbot(label="Диалог", height=460, **_chatbot_kwargs())
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
                gr.Markdown("### Файлы графиков")
                files_view = gr.Files(label="", height=140)
                gr.Markdown(
                    "Если график не отрисован выше, он всё равно сохранён на диске "
                    "в папке `plots/` — скачайте его здесь."
                )

        with gr.Accordion("Журнал вызовов инструментов", open=False):
            tool_log = gr.Markdown(
                "Здесь появится журнал: какие инструменты вызвал агент "
                "и с какими аргументами."
            )

        plots_state = gr.State([])

        def submit(text, history, state):
            if not _message_text(text):
                gr.Info("Введите вопрос.")
                return gr.skip()
            return chat_and_plot(text, history, state)

        outputs = [chatbot, plot_view, tool_log, plots_state, files_view, textbox]
        send.click(submit, inputs=[textbox, chatbot, plots_state], outputs=outputs)
        textbox.submit(submit, inputs=[textbox, chatbot, plots_state], outputs=outputs)

    return demo


def main() -> None:
    print(f"🎛 Gradio {GRADIO_VERSION}")
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
