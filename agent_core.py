"""Agent Core -- the brain of the geomagnetic AI assistant.

A local Qwen 2.5 7B Instruct (GGUF, Q4_K_M) is driven through
``llama-cpp-python`` and given access to the Stage 1 + Stage 2 Python tools in
a bounded tool-calling loop. No paid API, no API keys: the weights are pulled
once from HuggingFace and everything afterwards runs inside the notebook.

Three design points that this module exists to get right
--------------------------------------------------------

1. **Tool calls are parsed out of the text, not read from ``message.tool_calls``.**
   ``llama-cpp-python`` renders the tool schemas into the prompt correctly, but
   its default Jinja2 response path (``_convert_text_completion_to_chat``) only
   ever sets ``content`` -- it never populates ``tool_calls``. Qwen therefore
   emits a literal ``<tool_call>{...}</tool_call>`` block *inside* the
   text. A loop that checks ``if message["tool_calls"]`` never fires, silently
   executes nothing and hands raw tool-call JSON to the scientist as the final
   answer. ``parse_tool_calls()`` below is therefore the heart of the module.

2. **DataFrames never enter the JSON context.** ``analyze_derived`` returns a
   1440-row frame; dumping that into the prompt would blow the context window
   and cost seconds of prefill. Tools exchange *string handles* ("raw",
   "derived", "anomalies") and the dispatcher swaps in the real object.

3. **Everything is bounded and total.** At most ``MAX_TOOL_CALLS`` tool
   executions and ``MAX_ROUNDS`` model round-trips per user query. Every
   failure path -- unknown tool, bad arguments, tool exception, upstream
   ``is_error`` payload, unparseable model output -- returns a
   JSON-serialisable dict instead of raising.

Run it
------
    python agent_core.py                 # real model + real data (needs the model)
    python agent_core.py --offline       # hermetic: no model, no network
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

import geomag_analyzer as analyzer
import geomag_math as geomath
import geomag_plotter as plotter
import intermagnet_loader as loader

#: Bumped whenever the tool-call parser changes shape. ``colab_run.ipynb``
#: prints this next to ``git rev-parse HEAD``; a notebook that shows an older
#: value is running a cached copy of this module, not this file.
PARSER_VERSION = "bulletproof-3"

__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_SCHEMAS",
    "FrameStore",
    "LlamaBrain",
    "ScriptedBrain",
    "download_model",
    "load_brain",
    "parse_tool_calls",
    "run_agent",
    "is_error",
]


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
# NOTE: this quantisation is *sharded* upstream -- there is no single-file
# "qwen2.5-7b-instruct-q4_k_m.gguf". We match the stem and hand llama.cpp
# shard 00001; it picks up the siblings from the same directory.
MODEL_STEM = "qwen2.5-7b-instruct-q4_k_m"
MODELS_DIR = Path(os.environ.get("GEOMAG_MODELS_DIR", "models"))

N_CTX = 8192
N_BATCH = 512
TEMPERATURE = 0.1          # a scientist wants determinism, not creativity
MAX_TOKENS = 1200

# A three-day request costs three fetches, three derive calls and three plots
# before a single sentence of the answer, and a two-day comparison that first
# tries a bad handle costs a failed call plus a retry. 16 leaves room for both
# without letting a runaway loop run away.
MAX_TOOL_CALLS = 16        # tool executions per user query
MAX_ROUNDS = 18            # model round-trips per user query

OFFLINE = os.environ.get("GEOMAG_OFFLINE", "").strip() not in ("", "0", "false")

# Authoritative, read out of the live INTERMAGNET registry (Edinburgh GIN +
# Kyoto WDC, 327 stations). Hand-written guesses are wrong more often than not:
# ABK is Abisko/Sweden, SPB is not St Petersburg, Moscow is MOS not MOW.
STATION_HINT = (
    "IRT=Иркутск, MOS=Москва, NVS=Новосибирск, SPG=Санкт-Петербург, "
    "YAK=Якутск, KHB=Хабаровск, MGD=Магадан, VLA=Владивосток, "
    "PET=Паратунка (Камчатка), ARS=Арти, BOX=Борок, TIK=Тикси"
)

#: Station codes are stored lower-case in the slot name, so a model that echoes
#: back the upper-case code it typed ("IRT") must still resolve. The store lower-
#: cases every key, and this rule tells the model to do the same so that the
#: handles it prints match the handles it reads back out of the tool results.
HANDLE_CASE_RULE = (
    "ВАЖНО: хэндл фрейма всегда пиши строчными буквами. Например, если инструмент "
    "вернул frame_handle='raw:IRT:2024-09-10', передавай его в следующий вызов как "
    "'raw:irt:2024-09-10' — код станции в хэндле всегда в нижнем регистре. "
    "Сравнение двух дней требует два РАЗНЫХ хэндла, например "
    "'raw:irt:2024-09-10' и 'raw:irt:2024-09-11'."
)

SYSTEM_PROMPT = (
    "Ты — профессиональный геофизический ИИ-ассистент. У тебя есть инструменты. "
    "Никогда не выдумывай цифры, вызывай инструменты.\n\n"
    "Правила работы:\n"
    "1. Отвечай на русском языке, кратко и по делу, как коллега-геофизик.\n"
    "2. Любое число в твоём ответе должно прийти из результата инструмента. "
    "Нет результата — нет числа.\n"
    "3. Данные бери инструментом fetch_observatory_data. Код станции — трёхбуквенный "
    f"код IAGA. Известны: {STATION_HINT}.\n"
    "4. Стандартный порядок для графика: сначала fetch_observatory_data, потом "
    "calculate_derived_components, потом get_statistics / detect_anomalies, "
    "потом plot_components / plot_comparison. Не перескакивай шаги.\n"
"4a. ВЫБОР ИНСТРУМЕНТА — строго по этой таблице:\n"
    "    • X, Y, Z и F — это ГОТОВЫЕ колонки в сырых данных. Медиана, среднее, "
    "минимум, максимум, размах, стандартное отклонение: вызывай get_statistics "
    "сразу с хэндлом raw и components=['F']. НЕ вызывай перед этим "
    "calculate_derived_components — она считает только H, D, I, и ответит "
    "unknown_component.\n"
    "    • H, D, I — их считает calculate_derived_components, и больше ничего.\n"
    "    • Размах (max минус min) — это metric='delta'. Метрики 'range' НЕ "
    "СУЩЕСТВУЕТ: calculate_derived_math знает ровно три — 'delta', 'dH_dt' "
    "и 'anomaly'. Скорость изменения — 'dH_dt', отклонение от базы — "
    "'anomaly'. component='F' или 'H'.\n"
    "    • Своя формула — evaluate_custom_formula по колонкам X, Y, Z, F. "
    "Формула НИКОГДА не исполняется как код, а разбирается в безопасный список "
    "операций, поэтому пиши математику свободно, но не имена файлов, не текст "
    "и не вызовы функций.\n"
    "    • calculate_baseline — тихая ночная база. Нужен для metric='anomaly', "
    "и тогда это два вызова: сначала baseline, потом его значение в "
    "baseline_value.\n"
    "    • ПЕРЕД ЛЮБЫМ из них — fetch_observatory_data. Без созданного хэндла raw "
    "инструменты математики ответят unknown_frame, а не посчитают. Хэндл "
    "бери из frame_handle предыдущего результата, никогда не выдумывай.\n"
    "5. Вместо DataFrame инструментам передавай строковый хэндл, скопированный из "
    'поля "frame_handle" предыдущего результата. Это либо полный слот на конкретный '
    "день вида \"raw:irt:2024-09-10\" / \"derived:irt:2024-09-10\", либо короткое "
    'имя семейства "raw" / "derived" / "anomalies" (это всегда самый свежий '
    "элемент семейства). Не выдумывай хэндл: полный список доступных имён приходит "
    'в поле "available_handles" результата инструмента.\n'
    "5a. ПРАВИЛО СЛОТОВ: передавай в инструмент только тот хэндл, который сам "
    "создал в этом диалоге. 'raw:...' появляется после fetch_observatory_data. "
    "'derived:...' — только после calculate_derived_components или "
    "calculate_derived_math. 'anomalies:...' — после detect_anomalies. Никогда "
    "не передавай 'derived:...' или 'anomalies:...' в инструмент, который ты "
    "ещё не вызывал: получишь unknown_frame и потратишь вызов впустую. "
    "Порядок всегда такой: сначала 'raw:...', потом производные компоненты, "
    "потом статистика, потом графики.\n"
    "6. Если инструмент вернул {\"ok\": false} и ты НЕ смог получить ответ — "
    "сообщи пользователю причину из поля message и предложи, что делать. Но "
    "если ошибка была промежуточной и ты нашёл другой путь и ответил — просто "
    "ответь, не пересказывай в ответе коды ошибок. Не подставляй свои цифры "
    "вместо ошибки.\n"
    "7. В финальном ответе перечисли построенные графики и их файлы. Единицы "
    "измерения всегда пиши международным сокращением 'nT' — не «нанотесла» и "
    "не «нТл»: от этого зависят проверка твоего ответа и фильтрация ошибок.\n"
    "8. Ты можешь вызвать несколько инструментов подряд, прежде чем ответить. "
    "Если нужно сравнить два дня — вызови fetch_observatory_data дважды, по разу на "
    "день, и передай в plot_comparison два РАЗНЫХ хэндла.\n"
    "9. ВАЖНО: Ты НЕ являешься API-сервером. НИКОГДА не отвечай JSON-объектом, "
    "Python-словарём или строкой вида {'content': ..., 'tool_calls': ...}. "
    "Ответ такого вида считается ошибкой, данные не появятся.\n"
    "10. ФОРМАТ ОТВЕТА СТРОГО ТАК, ДВА ВАРИАНТА И НИКАКИХ ДРУГИХ:\n"
    "    (а) чтобы вызвать инструмент — выведи ТОЛЬКО блок тегов, без "
    "слов вокруг, без пояснений, без markdown:\n"
    '        <tool_call>{"name": "fetch_observatory_data", "arguments": '
    '{"station_code": "IRT", "start_date": "2024-09-10", "end_date": "2024-09-10"}}</tool_call>\n'
    '        <tool_call>{"name": "calculate_derived_math", "arguments": '
    '{"df": "raw:irt:2024-09-10", "metric": "delta", "component": "F"}}</tool_call>\n'
    "    Ровно одна пара фигурных скобок { } вокруг всего объекта и { } вокруг "
    "arguments. Никаких двойных скобок вида {\"name\": ...}} — это не "
    "валидный JSON и такой вызов не выполнится.\n"
    "    (б) когда все данные получены и инструменты больше не нужны — обычный "
    "текст на русском, БЕЗ тегов, БЕЗ фигурных скобок, БЕЗ кавычек вокруг "
    "имён инструментов.\n"
    "    Если сомневаешься между (а) и (б) — бери (а): лишний вызов "
    "инструмента дешевле, чем выдуманный ответ.\n\n"
    f"11. {HANDLE_CASE_RULE}"
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def is_error(value: Any) -> bool:
    """True when a payload is a failure dict. Same contract as stages 1 and 2."""
    return isinstance(value, dict) and value.get("ok") is False


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"ok": False, "error": code, "message": message}
    payload.update(extra)
    return payload


def _jsonable(value: Any, digits: int = 2) -> Any:
    """numpy/pandas -> plain Python, NaN/inf -> None. Keeps payloads JSON-safe."""
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if (value != value or value in (float("inf"), float("-inf"))) else round(value, digits)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item(), digits)
        except Exception:
            pass
    return str(value)


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- #
# tool registry -- OpenAI function-calling schemas understood by Qwen 2.5
# --------------------------------------------------------------------------- #
_FRAME_ARG = {
    "type": "string",
    "description": (
        "A data descriptor returned by a previous tool, copied EXACTLY from its "
        "'frame_handle' field. It is either a full per-day slot such as "
        "'raw:irt:2024-09-10' or 'derived:irt:2024-09-10', or one of the short "
        "family names 'raw' / 'derived' / 'anomalies', which always mean the most "
        "recent member of that family. Never invent a handle: the valid ones are "
        "listed in 'available_handles' of the previous tool result."
    ),
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_observatory_data",
            "description": (
                "Download geomagnetic data for one INTERMAGNET observatory. "
                "Call this first, before any analysis. Returns the row count, the "
                "available columns, and a frame_handle such as 'raw:irt:2024-09-10'. "
                "Call it once per day you need and keep those distinct handles: the "
                "short name 'raw' only ever points at the most recent fetch. "
                "Values are in nT, timestamps are UTC."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "station_code": {
                        "type": "string",
                        "description": f"Three-letter IAGA station code, e.g. IRT. Known: {STATION_HINT}",
                    },
                    "start_date": {"type": "string", "description": "Start date, YYYY-MM-DD."},
                    "end_date": {"type": "string", "description": "End date, YYYY-MM-DD (inclusive)."},
                    "data_type": {
                        "type": "string",
                        "enum": ["definitive", "quasi-def", "adjusted", "reported", "best-avail", "auto"],
                        "description": "Publication state. 'auto' walks a fallback chain. Default 'definitive'.",
                    },
                    "samples_per_day": {
                        "type": "string",
                        "enum": ["Minute", "Second"],
                        "description": "Temporal resolution. Default 'Minute'.",
                    },
                },
                "required": ["station_code", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_derived_components",
            "description": (
                "Compute the horizontal field H, declination D and inclination I from "
                "the Cartesian components X/Y/Z. Use this whenever H is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "components": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["H", "D", "I"]},
                        "description": "Subset to compute. Default all three.",
                    },
                },
                "required": ["df"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_statistics",
            "description": (
                "Descriptive statistics (min, max, mean, median, std, count) for one "
                "or more components. Returns exact numbers you may quote directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "components": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["X", "Y", "Z", "F", "H", "D", "I"]},
                        "description": "Components to summarise.",
                    },
                    "digits": {"type": "integer", "description": "Decimal places. Default 2."},
                },
                "required": ["df"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": (
                "Flag samples that deviate from the series mean by more than N standard "
                "deviations. Use for outlier and disturbance screening."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "component": {"type": "string", "enum": ["X", "Y", "Z", "F", "H", "D", "I"]},
                    "sigma_threshold": {
                        "type": "number",
                        "description": "Threshold in sigmas. Default 3.0.",
                    },
                },
                "required": ["df"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_derived_math",
            "description": (
                "Measure how a component behaved over time, as a scalar you may quote: "
                "the peak-to-peak variation ('range', 'how much did H swing'), the "
                "rate of change in nT per minute, or the distance from a baseline. "
                "Use metric='delta' for a storm's total excursion, 'dH_dt' for how "
                "fast the field was changing, and 'anomaly' to subtract a quiet "
                "reference obtained from calculate_baseline. Returns a new "
                "frame_handle holding the per-sample series, plus the scalar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "metric": {
                        "type": "string",
                        "enum": ["delta", "anomaly", "dH_dt"],
                        "description": (
                            "'delta' = max - min over the window; 'dH_dt' = rate of "
                            "change in nT/min; 'anomaly' = value minus baseline_value."
                        ),
                    },
                    "component": {
                        "type": "string",
                        "enum": ["X", "Y", "Z", "F", "H", "D", "I"],
                        "description": "Component to measure. H is the usual choice.",
                    },
                    "window": {
                        "type": "integer",
                        "description": (
                            "Optional block size in samples for metric='delta'. Omit it "
                            "to get one variation for the whole day; set it to get the "
                            "largest excursion within any block."
                        ),
                    },
                    "baseline_value": {
                        "type": "number",
                        "description": (
                            "Required for metric='anomaly'. The quiet reference to "
                            "subtract, e.g. the 'baseline' from calculate_baseline."
                        ),
                    },
                },
                "required": ["df", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_baseline",
            "description": (
                "Establish the quiet reference level for a component, so that departures "
                "from it can be reported as anomalies. mode='night' averages the quietest "
                "window of the day (00:00-04:00 UTC by default), when the ring current is "
                "minimal. Returns the baseline value; pass it to calculate_derived_math "
                "with metric='anomaly'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "component": {"type": "string", "enum": ["X", "Y", "Z", "F", "H", "D", "I"]},
                    "mode": {
                        "type": "string",
                        "enum": ["night", "full"],
                        "description": (
                            "'night' = mean over the quiet night window (default). "
                            "'full' = mean over every sample, which includes the storm "
                            "and so is a poor quiet reference."
                        ),
                    },
                    "night_hours": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Night window as [start_hour, end_hour] UTC, default [0, 4].",
                    },
                },
                "required": ["df"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_custom_formula",
            "description": (
                "Compute a custom arithmetic expression over the columns of a frame and "
                "get the result as a new plottable series. Use it for a quantity the "
                "built-in tools do not cover, e.g. 'sqrt(X**2 + Y**2) + Z/10' or "
                "'degrees(atan2(Z, sqrt(X**2 + Y**2)))' for inclination in degrees. "
                f"Allowed columns: {', '.join(geomath.available_columns())} plus the frame's own "
                f"numeric columns. Allowed operators: + - * / % **. Allowed functions: "
                f"{', '.join(sorted(geomath.DSL_FUNCTIONS))}, plus the constants pi and e. "
                "Anything else is refused: the formula is parsed, never executed, so no "
                "Python code can run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "formula": {
                        "type": "string",
                        "description": (
                            "The expression, e.g. 'sqrt(X**2 + Y**2) + Z/10'. Column names "
                            "must be upper case."
                        ),
                    },
                },
                "required": ["df", "formula"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_components",
            "description": (
                "Plot one or more components of a single time series as an interactive "
                "Plotly HTML chart. Returns the absolute path of the saved file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "components": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["X", "Y", "Z", "F", "H", "D", "I"]},
                        "description": "Components to plot.",
                    },
                    "title": {"type": "string", "description": "Chart title."},
                    "filename": {"type": "string", "description": "Output file name, .html."},
                },
                "required": ["df", "components"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_comparison",
            "description": (
                "Overlay two different days of one component on a shared time-of-day "
                "axis. Use it to compare, for example, a storm day against a quiet day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df1": {
                        **_FRAME_ARG,
                        "description": (
                            "First day: a frame_handle copied from a previous tool result, "
                            "e.g. 'raw:irt:2024-09-10'."
                        ),
                    },
                    "df2": {
                        **_FRAME_ARG,
                        "description": (
                            "Second day: a frame_handle from a *different* slot than df1, "
                            "e.g. 'raw:irt:2024-09-11'."
                        ),
                    },
                    "component": {"type": "string", "enum": ["X", "Y", "Z", "F", "H", "D", "I"]},
                    "title": {"type": "string", "description": "Chart title."},
                    "filename": {"type": "string", "description": "Output file name, .html."},
                },
                "required": ["df1", "df2", "component"],
            },
        },
    },
]

TOOLS_BY_NAME = {t["function"]["name"] for t in TOOL_SCHEMAS}
PLOT_TOOLS = {"plot_components", "plot_comparison"}


# --------------------------------------------------------------------------- #
# model download
# --------------------------------------------------------------------------- #
def _shard_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"-(\d{5})-of-(\d{5})", path.name)
    return (int(m.group(1)) if m else 0, path.name)


def _first_shard(files: Sequence[Path]) -> Path:
    """llama.cpp must be pointed at shard 00001; it loads the rest itself."""
    return sorted(files, key=_shard_sort_key)[0]


def _gguf_files(directory: Path, stem: str) -> list[Path]:
    return sorted(directory.glob(f"{stem}*.gguf"), key=_shard_sort_key)


def _entry_shard(directory: Path, stem: str) -> Path | None:
    """Return shard 00001 only if it is present *and* finished.

    An interrupted download can leave shard 00002 complete while 00001 is still
    ``.incomplete``. Returning shard 2 would hand llama.cpp a file that starts
    in the middle of the model, so the entry shard is required explicitly.
    """
    for path in _gguf_files(directory, stem):
        match = re.search(r"-(\d{5})-of-\d{5}", path.name)
        if match is None or int(match.group(1)) == 1:
            return path if path.is_file() and path.stat().st_size > 0 else None
    return None


def _purge_incomplete(directory: Path) -> int:
    """Drop half-written shards and their lock files left by a killed download."""
    removed = 0
    for pattern in ("*.incomplete", "*.lock", "*.metadata"):
        for path in directory.glob(pattern):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def model_is_downloaded(stem: str = MODEL_STEM, target_dir: str | Path | None = None) -> bool:
    """True when the weights are already on disk. Never touches the network."""
    return _entry_shard(Path(target_dir or MODELS_DIR), stem) is not None


def download_model(
    repo_id: str = MODEL_REPO,
    stem: str = MODEL_STEM,
    target_dir: str | Path | None = None,
    verbose: bool = True,
) -> str | dict[str, Any]:
    """Ensure the GGUF weights are on disk and return the path of shard 1.

    Returns an absolute path string on success, an error dict on failure.
    """
    target = Path(target_dir or MODELS_DIR)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _error("model_dir_unwritable", f"Cannot create {target}: {exc}")

    _purge_incomplete(target)

    local = _entry_shard(target, stem)
    if local is not None:
        if verbose:
            print(f"[agent] model already present: {local}")
        return str(local)

    if OFFLINE:
        return _error(
            "offline",
            "GEOMAG_OFFLINE is set, so the model cannot be downloaded.",
            expected_dir=str(target.resolve()),
        )

    def say(msg: str) -> None:
        if verbose:
            print(f"[agent] {msg}", flush=True)

    # huggingface_hub gives us resume + retries; fall back to plain urllib so the
    # module works in a bare Kaggle image without the extra dependency.
    try:
        from huggingface_hub import snapshot_download

        say(f"downloading {repo_id} ({stem}*) via huggingface_hub ...")
        path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{stem}*.gguf"],
            local_dir=str(target),
            max_workers=4,
        )
        found = _gguf_files(Path(path), stem)
        entry = _entry_shard(Path(path), stem)
        if entry is None or not found:
            return _error("model_not_found", f"No complete {stem}*.gguf in {path}")
        return str(entry)
    except ImportError:
        say("huggingface_hub not installed, falling back to urllib")
    except Exception as exc:  # network hiccup, gated repo, disk full ...
        if verbose:
            print(f"[agent] huggingface_hub failed: {exc}", file=sys.stderr)
            traceback.print_exc()

    import urllib.error
    import urllib.request

    api = f"https://huggingface.co/api/models/{repo_id}"
    try:
        with urllib.request.urlopen(api, timeout=60) as resp:
            siblings = json.load(resp).get("siblings", [])
    except Exception as exc:
        return _error("model_listing_failed", f"Cannot list {repo_id}: {exc}")

    names = [s["rfilename"] for s in siblings if s.get("rfilename", "").startswith(stem)]
    names = [n for n in names if n.endswith(".gguf")]
    if not names:
        return _error(
            "model_not_found",
            f"{repo_id} publishes no {stem}*.gguf",
            available=[s.get("rfilename") for s in siblings],
        )
    names.sort(key=lambda n: _shard_sort_key(Path(n)))

    for name in names:
        destination = target / name
        if destination.exists() and destination.stat().st_size > 0:
            say(f"skip {name} (already there)")
            continue
        url = f"https://huggingface.co/{repo_id}/resolve/main/{name}"
        say(f"downloading {name} ...")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(
                destination, "wb"
            ) as fh:
                shutil_copyfileobj(resp, fh)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            return _error("model_download_failed", f"{name}: {exc}", url=url)

    found = _gguf_files(target, stem)
    entry = _entry_shard(target, stem)
    if entry is None or not found:
        return _error("model_not_found", f"Download left no complete {stem}*.gguf in {target}")
    if verbose:
        total = sum(p.stat().st_size for p in found) / 1e9
        say(f"done -> {entry} ({len(found)} shard(s), {total:.1f} GB)")
    return str(entry)


def shutil_copyfileobj(src, dst, length: int = 1 << 20) -> None:
    while True:
        chunk = src.read(length)
        if not chunk:
            break
        dst.write(chunk)


# --------------------------------------------------------------------------- #
# brains
# --------------------------------------------------------------------------- #
class LlamaBrain:
    """Adapter around ``llama_cpp.Llama`` exposing a single ``chat`` method."""

    def __init__(self, llm: Any, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE):
        self._llm = llm
        self._max_tokens = max_tokens
        self._temperature = temperature

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        response = self._llm.create_chat_completion(
            messages=messages,
            tools=tools,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        message = response["choices"][0]["message"]
        return {
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls"),
        }


class ScriptedBrain:
    """Deterministic stand-in for the LLM.

    Lets the whole loop -- parsing, dispatch, budgeting, error handling -- be
    exercised on a laptop with no model, no GPU and no network.
    """

    def __init__(self, script: Sequence[dict[str, Any]]):
        self._script = list(script)
        self.calls = 0

    def chat(self, messages, tools) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        if not self._script:
            return {"content": "Готово.", "tool_calls": None}
        return self._script.pop(0)


def _resolve_gpu_layers(requested: int) -> int:
    """Honour ``n_gpu_layers=-1`` but never ask for GPU offload without a GPU."""
    if requested != -1 or requested > 0:
        return requested
    try:
        from llama_cpp import llama_cpp as _lc

        if not _lc.llama_supports_gpu_offload():
            print(
                "[agent] WARNING: no CUDA device detected, falling back to n_gpu_layers=0. "
                "The agent will run, but slowly.",
                file=sys.stderr,
            )
            return 0
    except Exception:
        return 0
    return -1


def load_brain(
    model_path: str | Path | None = None,
    n_gpu_layers: int = -1,
    n_ctx: int = N_CTX,
    verbose: bool = True,
) -> Any:
    """Download (if needed) and load the brain. Returns a brain or an error dict."""
    path = Path(model_path) if model_path else download_model(verbose=verbose)
    if is_error(path):
        return path

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        return _error(
            "llama_cpp_unavailable",
            "llama-cpp-python is not installed. On Kaggle/Colab install the "
            "prebuilt CUDA wheel -- see the report section 'Owner instructions'.",
            hint="pip install --pre https://abetlen.github.io/llama-cpp-python/whl/cu121",
            cause=str(exc),
        )

    layers = _resolve_gpu_layers(n_gpu_layers)
    if verbose:
        print(f"[agent] loading {path} (n_gpu_layers={layers}, n_ctx={n_ctx}) ...", flush=True)
    try:
        llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_gpu_layers=layers,
            n_batch=N_BATCH,
            n_threads=max(1, (os.cpu_count() or 4)),
            verbose=verbose,
        )
    except Exception as exc:
        return _error("model_load_failed", f"llama.cpp refused {path}: {exc}", path=str(path))
    return LlamaBrain(llm)


# --------------------------------------------------------------------------- #
# tool-call parsing -- the critical part
# --------------------------------------------------------------------------- #
_INVISIBLE = "\u200b\u200c\u200d\ufeff"
_TOOL_CALL_RE = re.compile(
    r"<\s*\??\s*tool_call\s*>\s*(?P<body>.*?)\s*<\s*/\s*\??\s*tool_call\s*>",
    re.DOTALL,
)
_OPEN_TAG_RE = re.compile(r"<\s*\??\s*tool_call\s*>", re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(?P<body>\{.*?})\s*```", re.DOTALL)


def _clean(text: str) -> str:
    for ch in _INVISIBLE:
        text = text.replace(ch, "")
    return text


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    """Qwen emits ``arguments`` as a JSON *string*; the chat template wants a dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {"_value": raw}


def _reply_text(reply: Any) -> str:
    """Normalise whatever the backend handed back into plain text.

    llama-cpp-python is inconsistent about the shape of a chat message: some
    code paths return an OpenAI-style ``{"content": ...}`` dict, others return
    the bare ``str``. Anything else is stringified rather than trusted.
    """
    if isinstance(reply, str):
        return reply
    if isinstance(reply, dict):
        content = reply.get("content")
        return content if isinstance(content, str) else ("" if content is None else str(content))
    return "" if reply is None else str(reply)


def _iter_json_spans(text: str):
    """Yield every balanced ``{...}`` / ``[...]`` span in *text*, outermost first.

    A regex such as ``\\{.*?\\}`` cannot be used here: ``arguments`` routinely
    nests objects and arrays, and a non-greedy match stops at the *first*
    closing brace and hands back unparseable JSON. This scanner tracks depth
    while skipping brackets that sit inside JSON strings, and honours backslash
    escapes, so a literal ``"}"`` inside a value cannot unbalance it.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            if depth == 0:
                start = index
            depth += 1
        elif char in "}]" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start : index + 1]
                start = -1


# Backwards-compatible alias: the Stage 1/2 callers only ever needed objects.
_iter_json_objects = _iter_json_spans


def _straighten_quotes(text: str) -> str:
    """Replace curly quotes, which models sometimes emit, with ASCII ones."""
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u201e", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _single_quotes_to_double(text: str) -> str:
    """Rewrite bare ``'`` as ``"`` while leaving double-quoted spans alone.

    Only valid JSON ever contains ``"``, so anything else outside a string is
    a single-quote typo and can be swapped safely.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            out.append(char)
        elif char == "'":
            out.append('"')
        else:
            out.append(char)
    return "".join(out)


def _drop_trailing_commas(text: str) -> str:
    """Remove ``,`` that sits immediately before ``}`` or ``]`` outside strings."""
    out: list[str] = []
    in_string = False
    escaped = False
    length = len(text)
    for index, char in enumerate(text):
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            out.append(char)
            continue
        if char == ",":
            look = index + 1
            while look < length and text[look] in " \t\r\n":
                look += 1
            if look < length and text[look] in "}]":
                continue  # trailing comma: drop it
        out.append(char)
    return "".join(out)


def _try_loads(raw: str) -> Any:
    """``json.loads`` with a small ladder of repairs, or ``None`` on failure."""
    raw = raw.strip()
    if not raw:
        return None
    for variant in (raw, _straighten_quotes(raw)):
        try:
            return json.loads(variant)
        except (json.JSONDecodeError, ValueError):
            pass
    for base in (_straighten_quotes(raw), raw):
        repaired = _drop_trailing_commas(_single_quotes_to_double(base))
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _json_payloads(text: str) -> list[Any]:
    """Parse *text* as JSON, progressively more forgiving.

    Attempts, in order: the text as given; the ``{`` .. last ``}`` slice (this
    discards any prose the model wrapped around the payload); then every
    balanced span, so two calls emitted back to back are both recovered.
    """
    out: list[Any] = []

    def _add(raw: str) -> None:
        if not raw.strip():
            return
        value = _try_loads(raw)
        if value is not None and value not in out:
            out.append(value)

    _add(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        _add(text[start : end + 1])
    for span in _iter_json_spans(text):
        _add(span)
    return out


# Keys a model may plausibly use for the function name and the argument bag.
_NAME_KEYS = ("name", "tool_name", "function_name", "tool")
_ARG_KEYS = ("arguments", "parameters", "args", "input")
_ENVELOPE_KEYS = ("function", "tool_call", "call")

# A runaway or looping model must not be able to fan out unbounded tool calls.
_MAX_CALLS_PER_BODY = 8


def _normalize_call(value: Any, depth: int = 0) -> list[tuple[str, dict[str, Any]]]:
    """Pull ``(name, arguments)`` pairs out of one parsed JSON value.

    Handles the shapes a 7B model actually emits: a bare ``{"name", ...}``,
    the OpenAI-style ``{"function": {"name", ...}}`` envelope, ``parameters``
    instead of ``arguments``, and a top-level array of calls.
    """
    if depth > 4:
        return []
    if isinstance(value, list):
        out: list[tuple[str, dict[str, Any]]] = []
        for item in value:
            out.extend(_normalize_call(item, depth + 1))
        return out
    if not isinstance(value, dict):
        return []

    if not any(key in value for key in _NAME_KEYS):
        for key in _ENVELOPE_KEYS:
            inner = value.get(key)
            if isinstance(inner, dict):
                return _normalize_call(inner, depth + 1)

    name = next(
        (value[k] for k in _NAME_KEYS if isinstance(value.get(k), str) and value[k].strip()),
        None,
    )
    if not name:
        return []
    arguments = next((value[k] for k in _ARG_KEYS if k in value), {})
    return [(str(name).strip(), _coerce_arguments(arguments))]


def _calls_from_json(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Every call recoverable from one candidate body, tags or no tags.

    All parsed payloads are visited, not just the first that works, so a model
    that emits two calls back to back without a tag between them is not
    truncated to the first. ``_iter_json_spans`` only ever yields outermost
    spans, so a nested ``arguments`` object is never double-counted.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for payload in _json_payloads(body):
        for name, arguments in _normalize_call(payload):
            if (name, json.dumps(arguments, sort_keys=True, default=str)) not in [
                (n, json.dumps(a, sort_keys=True, default=str)) for n, a in out
            ]:
                out.append((name, arguments))
        if len(out) >= _MAX_CALLS_PER_BODY:
            break
    return out


def _api_envelope(text: str) -> tuple[str, Any] | None:
    """Detect the OpenAI response envelope Qwen sometimes hallucinates.

    Instead of emitting a tool call, the model writes a literal
    ``{'content': '...', 'tool_calls': None}`` -- it is role-playing the API
    server. Returns ``(content, raw_tool_calls)`` when such a dict is present,
    where ``raw_tool_calls`` is ``None`` for a refusal to call any tool, and
    ``None`` (the outer value) when the text is not an envelope at all.
    """
    if "tool_calls" not in text:
        return None
    for span in _iter_json_spans(text):
        parsed: Any = None
        # Two syntaxes, two parsers. The model writes Python reprs with single
        # quotes and `None`; it also writes JSON with `null`. `literal_eval`
        # chokes on `null`, `json.loads` chokes on `None` and on bare single
        # quotes, so both are tried on both spellings.
        as_json = _single_quotes_to_double(_straighten_quotes(span))
        for candidate in (span, as_json):
            for loader in (ast.literal_eval, _try_loads):
                try:
                    parsed = loader(candidate)
                except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                    continue
                if parsed is not None:
                    break
            if parsed is not None:
                break
        if not isinstance(parsed, dict) or "tool_calls" not in parsed:
            continue
        content = parsed.get("content")
        text_part = content if isinstance(content, str) else ""
        return text_part, parsed.get("tool_calls")
    return None


# Sent back to the model when it answers with an API envelope instead of a
# tool call. Names the exact mistake and shows the exact expected shape.
_FORMAT_CORRECTION = (
    "Твой предыдущий ответ был неверным: это был JSON/Python-словарь вида "
    "{'content': ..., 'tool_calls': ...}. Ты не API-сервер, так отвечать нельзя.\n"
    "Чтобы получить данные, выведи РОВНО один блок тегов и ничего больше:\n"
    '<tool_call>{"name": "fetch_observatory_data", "arguments": '
    '{"station_code": "IRT", "start_date": "2024-09-10", "end_date": "2024-09-10"}}</tool_call>\n'
    "Не добавляй пояснений до или после блока тегов."
)


def _why_unparsed(body: str) -> str:
    """A short, human-readable reason a body yielded no call."""
    stripped = body.strip()
    if not stripped:
        return "empty tool call"
    try:
        json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        return f"not valid JSON ({exc})"
    return "valid JSON, but no usable 'name'/'arguments' pair: " + _short(stripped, 200)


def parse_tool_calls(
    reply: Any, debug: bool | None = None
) -> tuple[list[dict[str, Any]], str]:
    # 1. Unconditional trace of the raw model output, first thing, before any
    #    normalisation. If this line is missing from a Colab log, the runtime
    #    is executing a cached copy of this module and NOT this file.
    #    Silence it with GEOMAG_DEBUG_PARSER=0 once the agent is stable.
    if os.environ.get("GEOMAG_DEBUG_PARSER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    ):
        print(f"\n[DEBUG] RAW LLM OUTPUT FOR PARSING:\n{repr(str(reply)[:1000])}\n")

    """Split a model reply into tool calls and the leftover prose.

    Accepts ``str`` or an OpenAI-style ``dict`` (any backend shape; text is
    normalised via :func:`_reply_text`), then in order of preference:

    1. a structured ``tool_calls`` list (set by llama-cpp-python only for a few
       chat handlers, so it cannot be relied on);
    2. ``<tool_call>{...}</tool_call>`` blocks, which is what Qwen 2.5
       actually emits through llama.cpp's default Jinja2 path;
    3. a bare JSON object ``{"name": ..., "arguments": ...}``, which small
       models produce when they forget the tags.

    Bodies are salvaged rather than rejected: prose around the JSON, trailing
    commas, single or curly quotes, stray brackets, a ``function`` envelope and
    a top-level array are all tolerated, because a 7B model emits all of them.
    A body that truly cannot be read becomes one ``malformed`` call carrying a
    ``reason``.

    Never raises: unparseable payloads are reported, not propagated.

    Returns ``(calls, leftover_text)`` where each call is
    ``{"name": str, "arguments": dict, "malformed": bool}``.
    """
    content = _reply_text(reply)
    structured = reply.get("tool_calls") if isinstance(reply, dict) else None

    calls: list[dict[str, Any]] = []
    for tc in structured or []:
        fn = tc.get("function", tc) if isinstance(tc, dict) else {}
        name = fn.get("name")
        if name:
            calls.append({"name": str(name), "arguments": _coerce_arguments(fn.get("arguments"))})
    if calls:
        return calls, content

    text = _clean(content)

    found = [m.group("body") for m in _TOOL_CALL_RE.finditer(text)]
    remainder = _TOOL_CALL_RE.sub("", text)

    if not found and "<tool_call" in text.lower():
        # Unbalanced tags: the model was cut off mid-call. Try to recover.
        chunks = re.split(r"<\s*\??\s*tool_call\s*>", text)[1:]
        found = chunks
        remainder = text[: text.lower().find("<tool_call")]

    if not found:
        for match in _FENCE_RE.finditer(text):
            if _calls_from_json(match.group("body")):
                found.append(match.group("body"))
                remainder = _FENCE_RE.sub("", remainder)

    if not found and '"name"' in text and '"arguments"' in text:
        for span in _iter_json_spans(text):
            if _calls_from_json(span):
                found.append(span)
                # Drop the consumed span, as the fenced branch does, so a
                # forgotten tag cannot leak tool JSON into the final answer.
                remainder = remainder.replace(span, "", 1)

    for body in found:
        recovered = _calls_from_json(body)
        if recovered:
            for name, arguments in recovered:
                calls.append({"name": name, "arguments": arguments, "malformed": False})
        else:
            calls.append(
                {
                    "name": "",
                    "arguments": _coerce_arguments(body),
                    "malformed": True,
                    "reason": _why_unparsed(body),
                }
            )

    if not calls:
        # The model answered with an API envelope instead of a tool call. Any
        # real calls inside it are still honoured; the prose is handed back as
        # the leftover so the loop can read it, and a refusal (``tool_calls:
        # None``) surfaces as an empty call list rather than as a fake success.
        envelope = _api_envelope(text)
        if envelope is not None:
            content_text, raw_calls = envelope
            recovered: list[tuple[str, dict[str, Any]]] = []
            if raw_calls is not None:
                recovered = _normalize_call(raw_calls)
            for name, arguments in recovered[:_MAX_CALLS_PER_BODY]:
                calls.append({"name": name, "arguments": arguments, "malformed": False})
            return calls, content_text

    return calls, remainder


# --------------------------------------------------------------------------- #
# frame handles
# --------------------------------------------------------------------------- #
class FrameStore:
    """Holds real DataFrames so they never have to travel through JSON.

    Each stored frame gets a permanent, descriptive *slot* such as
    ``raw:irt:2024-09-10``, so fetching a second day no longer overwrites the
    first. The short legacy names ``raw``/``derived``/``anomalies`` stay usable
    as *family aliases* and always resolve to the most recently stored member of
    that family, which keeps every single-day flow working unchanged.
    """

    def __init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        # family -> slots in write order; the last one is what the alias points at
        self._families: dict[str, list[str]] = {}

    @staticmethod
    def _normalise(handle: str) -> str:
        return handle.strip().lower()

    def put(self, name: str, df: pd.DataFrame) -> str:
        """Store ``df`` under the slot ``name`` and return the canonical slot."""
        slot = self._normalise(name)
        self._frames[slot] = df
        family = slot.split(":", 1)[0]
        slots = self._families.setdefault(family, [])
        if slot in slots:
            slots.remove(slot)
        slots.append(slot)
        return slot

    def derive_name(self, source: Any, family: str) -> str:
        """Derive a slot name for a new family, keeping the source's suffix.

        ``raw:irt:2024-09-10`` -> ``derived:irt:2024-09-10``, so each day keeps
        its own derived/anomalies frame instead of fighting over one slot.
        """
        slot = self._normalise(str(source)) if isinstance(source, str) else ""
        suffix = slot.split(":", 1)[1] if ":" in slot else ""
        return f"{family}:{suffix}" if suffix else family

    def get(self, handle: Any) -> pd.DataFrame | None:
        if isinstance(handle, pd.DataFrame):
            return handle
        if not isinstance(handle, str):
            return None
        key = self._normalise(handle)
        frame = self._frames.get(key)
        if frame is not None:
            return frame
        # bare family name -> the most recent member of that family
        for slot in reversed(self._families.get(key, [])):
            frame = self._frames.get(slot)
            if frame is not None:
                return frame
        return None

    def names(self) -> list[str]:
        """Every live slot name -- this is what the model is told it can use."""
        return sorted(self._frames)

    def aliases(self) -> dict[str, str]:
        """Short name -> the slot it currently points at, for the model's benefit."""
        out: dict[str, str] = {}
        for family, slots in self._families.items():
            for slot in reversed(slots):
                if self._frames.get(slot) is not None:
                    out[family] = slot
                    break
        return out

    def resolve(self, handle: Any) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
        """Normalise first, then look up, then describe what *is* available.

        Normalising here (rather than inside ``get`` alone) is what makes the
        failure path trustworthy: ``available`` is built from real keys, and the
        message quotes the same canonical string the store holds, so a model that
        copies the hint lands on a valid handle on the very next attempt.
        """
        frame = self.get(handle)
        if frame is not None:
            return frame, None
        available = self.names()
        canonical = (
            self._normalise(handle) if isinstance(handle, str) else None
        )
        message = f"No data under handle {handle!r}."
        if canonical and canonical not in available:
            # Tell the model precisely how its spelling differs, which is the one
            # detail that turns a silent mismatch into an obvious fix.
            message += (
                f" Did you mean one of: {', '.join(available)}?"
                if available
                else ""
            )
        return None, _error(
            "unknown_frame",
            message,
            available=available or ["(none yet - call fetch_observatory_data first)"],
            aliases=self.aliases(),
            requested=handle if isinstance(handle, str) else None,
            hint=(
                "Use one of the 'available' handles verbatim, copied from a previous "
                "tool result's 'frame_handle'. They are all lower-case. A bare family "
                "name ('raw', 'derived', 'anomalies') always means the most recent member "
                "of that family, so to compare two days pass the two full slot names, "
                "e.g. 'raw:irt:2024-09-10' and 'raw:irt:2024-09-11'."
                if available
                else "Call fetch_observatory_data first; it returns a frame_handle."
            ),
        )


def _summarise_frame(df: pd.DataFrame, store: FrameStore | None = None, name: str = "") -> dict[str, Any]:
    """A compact, JSON-safe description of a frame -- never the frame itself."""
    numeric = df.select_dtypes("number")
    stats: dict[str, Any] = {}
    for column in list(numeric.columns)[:6]:
        series = numeric[column]
        valid = series.dropna()
        if valid.empty:
            stats[column] = {"count": 0}
            continue
        stats[column] = {
            "count": int(valid.size),
            "min": _jsonable(valid.min()),
            "max": _jsonable(valid.max()),
            "mean": _jsonable(valid.mean()),
        }

    summary: dict[str, Any] = {
        "ok": True,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "n_missing": int(df.isna().sum().sum()),
    }
    if "timestamp" in df.columns and len(df):
        summary["start"] = _jsonable(df["timestamp"].iloc[0])
        summary["end"] = _jsonable(df["timestamp"].iloc[-1])
    if stats:
        summary["statistics"] = stats
    if store is not None and name:
        summary["frame_handle"] = name
        summary["available_handles"] = store.names()
        summary["handle_aliases"] = store.aliases()
    return summary


def _offline_frame() -> pd.DataFrame:
    """Two synthetic days, so the offline demo needs no network at all."""
    return plotter._synthetic_day(10, seed=7)


# --------------------------------------------------------------------------- #
# tool handlers
# --------------------------------------------------------------------------- #
Handler = Callable[[dict[str, Any], FrameStore], tuple[Any, str]]


def _first_present(df: pd.DataFrame, store: FrameStore, requested: str) -> str:
    """Pick the richest handle that actually has the requested column.

    Returns a concrete slot name, never a bare family alias: an alias now points
    at the most recent day, so returning "derived" here would silently hand the
    caller the wrong day once two days are loaded.
    """
    if df is not None and requested in df.columns:
        for slot in store.names():
            if store.get(slot) is df:
                return slot
    # Compare case-insensitively: geomag_plotter upper-cases the requested
    # component names before matching columns, so 'formula' and 'FORMULA' resolve
    # the same way there. Matching exactly one way here made the two disagree, and
    # a handle was silently downgraded to 'raw' for a column it did contain.
    wanted = requested.strip().upper()
    if df is not None and any(str(c).upper() == wanted for c in df.columns):
        for slot in store.names():
            if store.get(slot) is df:
                return slot
    for family in ("derived", "raw"):
        for slot in reversed(store._families.get(family, [])):
            candidate = store.get(slot)
            if candidate is not None and any(
                str(c).upper() == wanted for c in candidate.columns
            ):
                return slot
    return "raw"


def _handle_fetch(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    station = (args.get("station_code") or "").upper()
    start = args.get("start_date") or ""
    end = args.get("end_date") or start

    # Build the slot already normalised. Storing under the raw upper-case string
    # made every handle the tools advertise differ from the keys listed in
    # 'available_handles', so the model copied a name that was never in the list
    # and only resolved because FrameStore.get() happened to lower-case it.
    slot = f"raw:{station.lower()}:{start}" if station and start else "raw"
    slot = FrameStore._normalise(slot)

    if OFFLINE:
        frame = _offline_frame()
        # No extra put("raw", ...): the bare family name already resolves to the
        # most recent slot, and a redundant entry would shadow it in the alias map.
        store.put(slot, frame)
        summary = _summarise_frame(frame, store, slot)
        summary["offline"] = True
        return summary, f"Offline mode: synthesised {len(frame)} rows as handle {slot!r}."

    result = loader.fetch_observatory_data(
        station_code=station or args.get("station_code"),
        start_date=start,
        end_date=end,
        data_type=args.get("data_type") or "definitive",
        samples_per_day=args.get("samples_per_day") or "Minute",
    )
    if loader.is_error(result):
        return result, f"fetch failed: {result.get('error')}"
    store.put(slot, result)
    summary = _summarise_frame(result, store, slot)
    summary["station"] = station or args.get("station_code")
    summary["publication_state"] = (result.attrs or {}).get("publication_state")
    summary["source"] = (result.attrs or {}).get("source")
    return summary, f"Fetched {len(result)} rows for {station or args.get('station_code')} -> handle {slot!r}."


def _handle_derive(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    frame, err = store.resolve(args.get("df"))
    if err:
        return err, f"calculate_derived_components failed: {err['error']}"

    components = args.get("components") or ["H", "D", "I"]
    result = analyzer.calculate_derived_components(frame, components=components)
    if analyzer.is_error(result):
        return result, f"calculate_derived_components failed: {result.get('error')}"

    source_slot = args.get("df") or "raw"
    slot = store.derive_name(source_slot, "derived")
    store.put(slot, result)
    summary = _summarise_frame(result, store, slot)
    summary["computed"] = [c for c in ("H", "D", "I") if c in result.columns]
    return summary, f"Computed {', '.join(summary['computed'])} -> handle {slot!r}."


def _handle_stats(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    frame, err = store.resolve(args.get("df"))
    if err:
        return err, f"get_statistics failed: {err['error']}"
    components = args.get("components") or ["X", "Y", "Z", "F", "H"]
    digits = args.get("digits")
    result = analyzer.get_statistics(
        frame, components=components, **({"digits": int(digits)} if digits is not None else {})
    )
    if analyzer.is_error(result):
        return result, f"get_statistics failed: {result.get('error')}"
    summary = dict(result)
    summary["available_handles"] = store.names()
    summary["handle_aliases"] = store.aliases()
    return summary, f"Statistics for {', '.join(result)}."


def _handle_anomalies(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    frame, err = store.resolve(args.get("df"))
    if err:
        return err, f"detect_anomalies failed: {err['error']}"
    component = args.get("component") or "H"
    threshold = args.get("sigma_threshold")
    result = analyzer.detect_anomalies(
        frame,
        component=component,
        **({"sigma_threshold": float(threshold)} if threshold is not None else {}),
    )
    if analyzer.is_error(result):
        return result, f"detect_anomalies failed: {result.get('error')}"
    source_slot = args.get("df") or "raw"
    slot = store.derive_name(source_slot, "anomalies")
    store.put(slot, result)
    summary = _summarise_frame(result, store, slot)
    summary["component"] = component
    summary["n_flagged"] = int(len(result))
    return summary, f"Flagged {len(result)} samples of {component} -> handle {slot!r}."


def _handle_derived_math(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Measure a component's behaviour: variation, rate of change, or anomaly.

    Every branch writes a real column into a fresh slot, so the result is
    plottable and can itself be the input to the next tool. The scalar is
    returned alongside the handle because the number is what the model is
    actually asked for -- a handle alone would force an extra round trip.
    """
    frame, err = store.resolve(args.get("df"))
    if err:
        return err, f"calculate_derived_math failed: {err['error']}"

    metric = str(args.get("metric") or "delta").strip().lower()
    component = str(args.get("component") or "H").strip().upper()

    if metric == "delta":
        window = args.get("window")
        result = geomath.calculate_delta(
            frame,
            component,
            **({"window": int(window)} if window is not None else {}),
        )
        if geomath.is_error(result):
            return result, f"calculate_derived_math failed: {result.get('error')}"
        # windowed results are a per-block table; the whole-frame case is a float
        if isinstance(result, pd.DataFrame):
            finite = result["delta"].dropna()
            slot = store.derive_name(args.get("df") or "raw", "math")
            store.put(slot, result)
            summary = _summarise_frame(result, store, slot)
            summary.update(
                {
                    "metric": metric,
                    "component": component,
                    "window": int(args["window"]),
                    "n_blocks": int(len(result)),
                    "max_delta": _jsonable(finite.max()) if not finite.empty else None,
                    "mean_delta": _jsonable(finite.mean()) if not finite.empty else None,
                }
            )
            return summary, (
                f"{component} variation over {summary['window']}-sample windows: "
                f"max {summary['max_delta']}, mean {summary['mean_delta']} "
                f"-> handle {slot!r}."
            )
        slot = store.derive_name(args.get("df") or "raw", "math")
        summary = _summarise_frame(frame, store, slot)
        summary.update(
            {
                "metric": metric,
                "component": component,
                "delta": _jsonable(result),
                "unit": geomath.component_units(component),
                "available_handles": store.names(),
                "handle_aliases": store.aliases(),
            }
        )
        return summary, (
            f"{component} varied by {_jsonable(result)} "
            f"{geomath.component_units(component)} over the whole period."
        )

    if metric == "anomaly":
        baseline = args.get("baseline_value")
        if baseline is None:
            return _error(
                "missing_baseline",
                "metric='anomaly' needs baseline_value. Call calculate_baseline first.",
                hint="Take the quiet reference with calculate_baseline(mode='night'), "
                "then pass its 'baseline' here.",
            ), "calculate_derived_math needs a baseline for metric='anomaly'."
        series = geomath.calculate_anomaly(frame, component, float(baseline))
        if geomath.is_error(series):
            return series, f"calculate_derived_math failed: {series.get('error')}"
        slot = store.derive_name(args.get("df") or "raw", "anomaly")
        frame_out = frame.copy()
        # Upper case to match the X/Y/Z/H/D/I convention: geomag_plotter
        # upper-cases the requested component names before matching columns, so a
        # lowercase 'anomaly' column would be unreachable from plot_components.
        frame_out["ANOMALY"] = series
        store.put(slot, frame_out)
        summary = _summarise_frame(frame_out, store, slot)
        summary.update(
            {
                "metric": metric,
                "component": component,
                "baseline": _jsonable(baseline),
                "unit": geomath.component_units(component),
                "available_handles": store.names(),
                "handle_aliases": store.aliases(),
            }
        )
        return summary, (
            f"{component} minus the baseline {_jsonable(baseline)} is stored in the "
            f"'ANOMALY' column -> handle {slot!r}."
        )

    if metric == "dh_dt":
        series = geomath.calculate_dH_dt(frame, component=component)
        if geomath.is_error(series):
            return series, f"calculate_derived_math failed: {series.get('error')}"
        slot = store.derive_name(args.get("df") or "raw", "math")
        frame_out = frame.copy()
        frame_out["DH_DT"] = series
        store.put(slot, frame_out)
        finite = series.dropna()
        peak = _jsonable(finite.abs().max()) if not finite.empty else None
        summary = _summarise_frame(frame_out, store, slot)
        summary.update(
            {
                "metric": metric,
                "component": component,
                "unit": "nT/min",
                "peak_rate": peak,
                "available_handles": store.names(),
                "handle_aliases": store.aliases(),
            }
        )
        return summary, (
            f"Rate of change of {component} stored in the 'DH_DT' column; peak "
            f"{peak} nT/min -> handle {slot!r}."
        )

    return _error(
        "unknown_metric",
        f"metric={metric!r} is not supported.",
        available=["delta", "anomaly", "dH_dt"],
        hint="Use 'delta' for a range, 'dH_dt' for a rate, 'anomaly' for a baseline difference.",
    ), f"calculate_derived_math refused metric {metric!r}."


def _handle_baseline(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Establish the quiet reference level for a component."""
    frame, err = store.resolve(args.get("df"))
    if err:
        return err, f"calculate_baseline failed: {err['error']}"

    component = str(args.get("component") or "H").strip().upper()
    mode = str(args.get("mode") or "night").strip().lower()

    if mode == "night":
        hours = args.get("night_hours")
        if hours is None:
            result = geomath.get_nighttime_baseline(frame, component)
        else:
            if not isinstance(hours, (list, tuple)) or len(hours) != 2:
                return _error(
                    "invalid_input",
                    f"night_hours={hours!r} must be a pair of hours.",
                    hint="For example [22, 6] for a night window that wraps midnight.",
                ), "calculate_baseline needs night_hours as a pair of hours."
            result = geomath.get_nighttime_baseline(frame, component, tuple(hours))
    elif mode == "full":
        series = geomath.component_series(frame, component)
        if geomath.is_error(series):
            return series, f"calculate_baseline failed: {series.get('error')}"
        finite = series[np.isfinite(series)]
        if finite.empty:
            return _error(
                "no_finite_data",
                f"{component} has no finite samples.",
            ), f"calculate_baseline failed for {component}."
        result = {
            "ok": True,
            "baseline": _jsonable(finite.mean()),
            "n_samples": int(len(finite)),
            "n_finite": int(finite.size),
            "window": "full period",
        }
    else:
        return _error(
            "unknown_mode",
            f"mode={mode!r} is not supported.",
            available=["night", "full"],
            hint="'night' is the quiet reference; 'full' includes the storm and is "
            "usually a poor baseline.",
        ), f"calculate_baseline refused mode {mode!r}."

    if geomath.is_error(result):
        return result, f"calculate_baseline failed: {result.get('error')}"

    payload = dict(result)
    payload["component"] = component
    payload["mode"] = mode
    payload["unit"] = geomath.component_units(component)
    if mode == "night":
        payload["available_handles"] = store.names()
        payload["handle_aliases"] = store.aliases()
    note = ""
    if payload.get("baseline") is None:
        note = " No finite sample fell inside the window, so there is no baseline."
    return payload, (
        f"Quiet {mode} baseline for {component} is {payload.get('baseline')} "
        f"{payload.get('unit')}.{note}"
    )


def _handle_custom_formula(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Evaluate a user-supplied formula and store the result as a new column.

    The formula is parsed, never executed: see :mod:`geomag_math`. A refused
    formula is reported back with the accepted grammar so the model can correct
    it on the next round instead of guessing.
    """
    frame, err = store.resolve(args.get("df"))
    if err:
        return err, f"evaluate_custom_formula failed: {err['error']}"

    formula = args.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        return _error(
            "missing_formula",
            "The 'formula' argument is required and must be a non-empty string.",
            hint="For example: 'sqrt(X**2 + Y**2) + Z/10'.",
        ), "evaluate_custom_formula was called without a formula."

    series = geomath.evaluate_formula(frame, formula)
    if geomath.is_error(series):
        payload = dict(series)
        # Spell the grammar out on failure: the model has no other way to learn it.
        payload["allowed_functions"] = sorted(geomath.DSL_FUNCTIONS)
        payload["allowed_columns"] = geomath.available_columns(frame)
        payload["allowed_operators"] = ["+", "-", "*", "/", "%", "**"]
        return payload, (
            f"evaluate_custom_formula refused the formula: {series.get('message')}"
        )

    slot = store.derive_name(args.get("df") or "raw", "formula")
    frame_out = frame.copy()
    # Upper case for the same reason as ANOMALY: the plotter upper-cases the
    # requested names, so 'FORMULA' is reachable from plot_components while a
    # lowercase 'formula' column would be rejected as an unknown component.
    frame_out["FORMULA"] = series
    store.put(slot, frame_out)

    finite = series[np.isfinite(series)]
    summary = _summarise_frame(frame_out, store, slot)
    summary.update(
        {
            "formula": formula,
            "column": "FORMULA",
            "min": _jsonable(finite.min()) if not finite.empty else None,
            "max": _jsonable(finite.max()) if not finite.empty else None,
            "mean": _jsonable(finite.mean()) if not finite.empty else None,
            "n_finite": int(finite.size),
            "available_handles": store.names(),
            "handle_aliases": store.aliases(),
        }
    )
    return summary, (
        f"Evaluated {formula!r} into the 'FORMULA' column -> handle {slot!r}. "
        f"It is now plottable with plot_components(components=['FORMULA'])."
    )


def _handle_plot_components(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    components = args.get("components") or ["H"]
    handle = _first_present(store.resolve(args.get("df"))[0], store, components[0])
    frame, err = store.resolve(handle)
    if err:
        return err, f"plot_components failed: {err['error']}"
    result = plotter.plot_components(
        frame,
        components=components,
        title=args.get("title") or "Geomagnetic components",
        filename=args.get("filename"),
    )
    if plotter.is_error(result):
        return result, f"plot_components failed: {result.get('error')}"
    return {"ok": True, "path": result, "components": components, "rows": int(len(frame))}, (
        f"Saved chart to {result}"
    )


def _handle_plot_comparison(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    first, err = store.resolve(args.get("df1"))
    if err:
        return err, f"plot_comparison failed (df1): {err['error']}"
    second, err = store.resolve(args.get("df2"))
    if err:
        return err, f"plot_comparison failed (df2): {err['error']}"
    if first is second:
        return _error(
            "same_dataframe",
            "Both handles point at the same data; a comparison needs two different days.",
            requested={"df1": args.get("df1"), "df2": args.get("df2")},
            available=store.names(),
            hint=(
                "Fetch each day separately and pass the two distinct slot names, "
                "e.g. 'raw:irt:2024-09-10' and 'raw:irt:2024-09-11'. A bare 'raw' only ever "
                "points at the most recent fetch, so it can never represent both days."
            ),
        ), "plot_comparison refused: both handles resolved to the same data."
    result = plotter.plot_comparison(
        first,
        second,
        component=args.get("component") or "H",
        title=args.get("title") or "Day comparison",
        filename=args.get("filename"),
    )
    if plotter.is_error(result):
        return result, f"plot_comparison failed: {result.get('error')}"
    return {"ok": True, "path": result, "component": args.get("component") or "H"}, (
        f"Saved comparison to {result}"
    )


HANDLERS: dict[str, Handler] = {
    "fetch_observatory_data": _handle_fetch,
    "calculate_derived_components": _handle_derive,
    "get_statistics": _handle_stats,
    "detect_anomalies": _handle_anomalies,
    "calculate_derived_math": _handle_derived_math,
    "calculate_baseline": _handle_baseline,
    "evaluate_custom_formula": _handle_custom_formula,
    "plot_components": _handle_plot_components,
    "plot_comparison": _handle_plot_comparison,
}


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def run_agent(
    user_query: str,
    brain: Any = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_rounds: int = MAX_ROUNDS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Answer ``user_query`` by letting the model call the Stage 1/2 tools.

    Returns ``{"ok", "text", "plots", "tool_calls", "rounds", "stop_reason"}``.
    ``text`` and ``plots`` are the stable contract; the rest is diagnostics.
    """
    if not isinstance(user_query, str) or not user_query.strip():
        return _error("invalid_input", "user_query must be a non-empty string")

    if brain is None:
        # Deliberately *not* load_brain(): asking a question must never pull
        # 5 GB over the network as a side effect. Download once, up front.
        if not model_is_downloaded():
            return _error(
                "brain_not_loaded",
                "No model on disk and run_agent will not download one implicitly. "
                "Call load_brain() once to fetch and load it, then pass the "
                "returned brain to run_agent().",
                models_dir=str(MODELS_DIR.resolve()),
                expected=f"{MODEL_REPO} / {MODEL_STEM}*.gguf",
            )
        brain = load_brain()
    if is_error(brain):
        return brain
    if brain is None or not hasattr(brain, "chat"):
        return _error("invalid_brain", "brain must expose a .chat(messages, tools) method")

    store = FrameStore()
    plots: list[str] = []
    log: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    spent = 0
    stop_reason = "no_tool_calls"
    format_retries = 0

    for round_no in range(1, max_rounds + 1):
        try:
            reply = brain.chat(messages, TOOL_SCHEMAS)
        except Exception as exc:
            return _error("brain_failed", f"{type(exc).__name__}: {exc}", round=round_no)

        content = _reply_text(reply)
        calls, leftover = parse_tool_calls(reply)

        if not calls:
            # The model role-played the API server and returned
            # {'content': ..., 'tool_calls': None}. Accepting that as the final
            # answer would be exactly the "hallucinated success" we are trying
            # to eliminate, so correct it and let it retry -- a couple of times,
            # then give up honestly rather than inventing an answer.
            envelope = _api_envelope(content)
            if (
                envelope is not None
                and envelope[1] is None
                and format_retries < 2
                and round_no < max_rounds
            ):
                format_retries += 1
                if verbose:
                    print(
                        f"[agent] round {round_no}: model returned an API envelope, "
                        f"correcting (attempt {format_retries}/2)"
                    )
                messages.append({"role": "user", "content": _FORMAT_CORRECTION})
                continue

            text = leftover.strip() or content.strip()
            if verbose:
                print(f"[agent] round {round_no}: final answer ({len(text)} chars)")
            return {
                "ok": True,
                "text": _with_tool_failures(
                    text or "Не удалось получить текстовый ответ от модели.", log
                ),
                "plots": plots,
                "tool_calls": log,
                "rounds": round_no,
                "stop_reason": stop_reason,
            }

        if spent + len(calls) > max_tool_calls:
            stop_reason = "tool_budget_exhausted"
            leftover = leftover.strip()
            messages.append(
                {
                    "role": "assistant",
                    "content": leftover or None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": c["name"] or "unknown",
                                "arguments": c["arguments"],
                            },
                        }
                        for index, c in enumerate(calls)
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "content": _dumps(
                        _error(
                            "tool_budget_exhausted",
                            f"Only {max_tool_calls} tool calls are allowed per question. "
                            "Answer now with what you already have.",
                        )
                    ),
                }
            )
            # The refused calls are recorded one-by-one, not collapsed into a
            # single "(refused)" row: a refused plot_components means a chart the
            # user asked for does not exist, and hiding which calls were dropped
            # made the truncated run look complete.
            for skipped in calls:
                name = skipped.get("name") or "unknown"
                log.append(
                    {
                        "round": round_no,
                        "tool": name,
                        "arguments": skipped.get("arguments", {}),
                        "ok": False,
                        "executed": False,
                        "error": "tool_budget_exhausted",
                        "message": f"Not executed: the budget of {max_tool_calls} tool "
                        "calls was already spent.",
                        "hint": "Ask for the remaining charts in a follow-up message.",
                    }
                )
            if verbose:
                print(
                    f"[agent] round {round_no}: budget exhausted, refused "
                    f"{len(calls)} call(s), forcing an answer"
                )
            continue

        spent += len(calls)
        messages.append(
            {
                "role": "assistant",
                "content": leftover.strip() or None,
                # `arguments` must stay a dict: the Qwen template renders it with
                # Jinja `tojson`, which would double-encode a JSON string.
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for index, c in enumerate(calls)
                ],
            }
        )

        for call in calls:
            name, args = call["name"], call["arguments"]
            entry: dict[str, Any] = {"round": round_no, "tool": name, "arguments": args}

            if call.get("malformed") or not name:
                # Checked before unknown_tool: a truncated or unparsable call has
                # an empty name, and "no tool named ''" tells the model nothing.
                payload = _error(
                    "malformed_tool_call",
                    "The tool call could not be parsed. Emit exactly "
                    '<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call> '
                    "with valid JSON and nothing after the closing tag.",
                    received=_short(args),
                    reason=call.get("reason", "unrecognised payload"),
                )
                entry["error"] = payload["error"]
            elif name not in HANDLERS:
                payload = _error(
                    "unknown_tool",
                    f"No tool named {name!r}.",
                    available=sorted(TOOLS_BY_NAME),
                )
                entry["error"] = payload["error"]
            else:
                try:
                    payload, note = HANDLERS[name](args, store)
                except Exception as exc:
                    traceback.print_exc()
                    payload = _error(
                        "tool_exception",
                        f"{type(exc).__name__}: {exc}",
                        tool=name,
                    )
                    note = f"{name} raised {type(exc).__name__}"

            entry["ok"] = not is_error(payload)
            entry["executed"] = True
            # Surface the diagnostic fields, not just the code: without these a
            # malformed call in Colab reports "malformed_tool_call" and nothing
            # else, which is what sent the previous two fixes down blind paths.
            for key in ("message", "received", "reason", "hint"):
                if isinstance(payload, dict) and payload.get(key) is not None:
                    entry[key] = payload[key]
            # Record the handle the tool advertised. It is the only way an
            # outside checker can confirm the name the model is told to copy is
            # the name the store really holds.
            if isinstance(payload, dict) and isinstance(
                payload.get("frame_handle"), str
            ):
                entry["frame_handle"] = payload["frame_handle"]
                entry["available"] = payload.get("available")
            if entry["ok"] and name in PLOT_TOOLS and isinstance(payload.get("path"), str):
                plots.append(payload["path"])
                entry["plot"] = payload["path"]
            if is_error(payload):
                entry["error"] = payload.get("error")
                entry["message"] = payload.get("message")

            log.append(entry)
            if verbose:
                status = "ok" if entry["ok"] else f"FAILED ({entry.get('error')})"
                print(f"[agent] round {round_no}: {name}({_short(args)}) -> {status}")

            messages.append({"role": "tool", "content": _dumps(payload)})

    return {
        "ok": stop_reason != "no_tool_calls",
        "text": _with_tool_failures(
            (leftover or content or "").strip()
            or "Модель не сформулировала ответ в пределах отведённого числа шагов.",
            log,
        ),
        "plots": plots,
        "tool_calls": log,
        "rounds": max_rounds,
        "stop_reason": "round_limit",
    }


def _with_plot_tally(text: str, log: list[dict[str, Any]]) -> str:
    """State plainly how many charts exist, and list the files.

    When the budget cuts a run short the model's own sentence is optimistic --
    it says "the charts are ready" while two of them were never drawn -- and the
    user is left hunting for a file that does not exist. The count is computed
    from the executed plot calls, so it cannot disagree with ``result["plots"]``.
    """
    built = [e for e in log if e.get("ok") and e.get("plot")]
    dropped = [
        e for e in log
        if not e.get("ok", True) and e.get("tool") in PLOT_TOOLS
    ]
    if not built and not dropped:
        return text

    names = [Path(str(e["plot"])).name for e in built]
    line = f"Построено графиков: {len(built)}"
    if dropped:
        line += f" из {len(built) + len(dropped)} запрошенных"
    line += "."
    if names:
        line += "\nФайлы: " + ", ".join(names)
    if dropped:
        line += (
            f"\nНе построено: {len(dropped)} — лимит вызовов инструментов исчерпан. "
            "Попросите оставшиеся графики отдельным сообщением."
        )
    if "Построено графиков" in text:
        return text
    return f"{text}\n\n{line}" if text.strip() else line


def _failure_entries(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The failed calls a user could be shown.

    ``(refused)`` entries are loop control flow, not a tool the model called,
    so they are never user-facing failures.
    """
    return [
        e
        for e in log
        if not e.get("ok", True)
        and e.get("error")
        and e.get("tool") != "(refused)"
    ]


#: Every spelling of "nanotesla" an answer may plausibly use. One definition,
#: because two regexes disagreeing on this is how a correct answer gets treated
#: as a failure: the recovery filter below decides whether a number was reported,
#: and the E2E checker decides whether it can be verified. A model that writes
#: "60631.39 нанотесла" has reported a number, and both must agree.
_NANO_TESLA = r"(?:нТл|nT\b|нТ|нанотесл\w*)"

_NT_VALUE_RE = re.compile(
    rf"(\d+(?:[.,]\d+)?)\s*{_NANO_TESLA}",
    flags=re.IGNORECASE,
)


def find_nt_values(text: Any) -> list[float]:
    """Every nanotesla quantity stated in *text*, as floats.

    A comma is a decimal separator in Russian, so "60631,39 нТ" has to read as
    60631.39. Handing the captured string straight to ``float()`` raises on it,
    which would turn a correct answer into a checker crash.
    """
    if not isinstance(text, str):
        return []
    values: list[float] = []
    for raw in _NT_VALUE_RE.findall(text):
        try:
            values.append(float(raw.replace(",", ".")))
        except ValueError:  # pragma: no cover - the regex only yields digits
            continue
    return values


#: A figure or a quantity the user can check. The agent is told to state both,
#: and both are what distinguishes "answered the question" from "hit a wall".
_ANSWERED_RE = re.compile(
    rf"\d[\d\s.,]*\s*{_NANO_TESLA}"       # a number carrying a unit, any spelling
    r"|[\w.\-/]+\.html?"                    # a saved chart, by filename
    r"|построил|построен|график",
    flags=re.IGNORECASE,
)


def run_recovered(text: str, log: list[dict[str, Any]]) -> bool:
    """True when the agent failed a call but still answered the question.

    The failure footer exists because a model reports a failed tool as a vague
    "the attempt failed", which hid a real bug once: the actionable reason never
    reached the user. That reasoning still holds when the run ended in nothing.

    It stops holding when the agent recovered. A wrong first tool call is an
    internal detail of how the answer was found; printing it under a red
    "Не удалось выполнить часть операций" heading made a correct answer look
    like a broken system, and a user who reads that cannot tell which of the two
    it was.

    Recovery requires all three, so silence is never bought by hiding a real
    failure: there was a failure, at least one call actually succeeded, and the
    answer carries a checkable result -- a quantity with a unit, or a chart.
    """
    failures = _failure_entries(log)
    if not failures:
        return False
    if not any(e.get("ok") for e in log):
        return False  # nothing worked: the footer is the only signal there is
    return bool(_ANSWERED_RE.search(text or ""))


def _with_tool_failures(text: str, log: list[dict[str, Any]]) -> str:
    """Append any unrecovered tool failure to the final answer.

    The model routinely reports a failed tool call as a vague "the attempt
    failed", which is exactly what hid this bug: the actionable `reason`/`hint`
    never reached the user. Anything the model did not already mention gets
    appended verbatim, so the contract is visible whatever the model says.

    When the agent recovered -- see :func:`run_recovered` -- the footer is
    suppressed. It is still written to the console, so debugging loses nothing;
    only the user-facing surface gets quiet.
    """
    text = _with_plot_tally(text, log)
    failures = _failure_entries(log)
    if not failures:
        return text
    if run_recovered(text, log):
        print(
            "[agent] промежуточные ошибки скрыты из ответа (агент "
            f"восстановился): {sorted({str(e.get('error')) for e in failures})}"
        )
        for entry in failures:
            print(
                f"[agent]   {entry.get('tool')}: {entry.get('error')} -- "
                f"{entry.get('message') or entry.get('reason') or ''}"
            )
        return text
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for entry in failures:
        error = str(entry.get("error"))
        reason = entry.get("message") or entry.get("reason") or ""
        if error == "tool_budget_exhausted":
            # A refused call is bookkeeping, not a failure the user must read.
            # _with_plot_tally already says how many charts exist and what is
            # missing, and repeating every skipped call buries that.
            continue
        if error in text and (not reason or reason in text):
            continue
        key = (error, str(reason))
        if key in seen:
            continue
        seen.add(key)
        tool = entry.get("tool")
        prefix = f"{tool}: " if tool and tool != "(refused)" else ""
        lines.append(f"- {prefix}{error}: {reason}")
        if entry.get("hint"):
            lines.append(f"  подсказка: {entry['hint']}")
    if not lines:
        return text
    return f"{text}\n\nНе удалось выполнить часть операций:\n" + "\n".join(lines)


def _short(args: dict[str, Any], limit: int = 90) -> str:
    text = _dumps(args)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
DEMO_QUERY = "Скачай данные по станции Иркутск за 10 сентября 2024. Посчитай H и построй график."


def _demo_script() -> list[dict[str, Any]]:
    """What a well-behaved Qwen would emit for :data:`DEMO_QUERY`."""
    return [
        {
            "content": "",
            "tool_calls": None,
            "_text": '<tool_call>{"name": "fetch_observatory_data", '
            '"arguments": {"station_code": "IRT", "start_date": "2024-09-10", '
            '"end_date": "2024-09-10", "data_type": "definitive"}}</tool_call>',
        },
        {
            "content": "",
            "_text": '<tool_call>{"name": "calculate_derived_components", '
            '"arguments": {"df": "raw", "components": ["H"]}}</tool_call>',
        },
        {
            "content": "",
            "_text": '<tool_call>{"name": "get_statistics", '
            '"arguments": {"df": "derived", "components": ["H"]}}</tool_call>',
        },
        {
            "content": "",
            "_text": '<tool_call>{"name": "plot_components", '
            '"arguments": {"df": "derived", "components": ["H"], '
            '"title": "IRT 2024-09-10"}}</tool_call>',
        },
        {"content": "Готово: данные Иркутска за 10.09.2024 обработаны, H посчитан, график построен."},
    ]


def _unwrap(script: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """ScriptedBrain returns replies verbatim; ``_text`` is shorthand for content."""
    out = []
    for item in script:
        item = dict(item)
        if "_text" in item:
            item["content"] = item.pop("_text")
        out.append(item)
    return out


def _main() -> int:
    # The demo speaks Russian; Windows consoles default to a legacy code page and
    # would otherwise render every answer as mojibake.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Agent Core demo")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="no model download, no network: exercise the loop with a scripted brain",
    )
    parser.add_argument("--query", default=DEMO_QUERY)
    args = parser.parse_args()

    if args.offline:
        os.environ["GEOMAG_OFFLINE"] = "1"
        print("=" * 78)
        print("OFFLINE MODE -- scripted brain, synthetic data, no downloads")
        print("=" * 78)
        brain = ScriptedBrain(_unwrap(_demo_script()))
    else:
        print("=" * 78)
        print(f"QUERY: {args.query}")
        print("=" * 78)
        brain = load_brain()

    if is_error(brain):
        print(f"[agent] brain unavailable: {brain['error']}: {brain['message']}")
        return 1

    result = run_agent(args.query, brain=brain)

    print()
    print("=" * 78)
    print("TOOL CALL LOG")
    print("=" * 78)
    for index, entry in enumerate(result.get("tool_calls", []), start=1):
        mark = "ok " if entry.get("ok") else "ERR"
        print(f"{index}. [{mark}] {entry['tool']}({_short(entry['arguments'])})")
        if entry.get("error"):
            print(f"        error: {entry['error']} -- {entry.get('message', '')}")
        if entry.get("plot"):
            print(f"        plot : {entry['plot']}")

    print()
    print("=" * 78)
    print("FINAL ANSWER")
    print("=" * 78)
    print(result.get("text", ""))
    print()
    print(f"plots  : {result.get('plots', [])}")
    print(f"rounds : {result.get('rounds')}  stop_reason: {result.get('stop_reason')}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())
