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
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd

import geomag_analyzer as analyzer
import geomag_math as geomath
import geomag_plotter as plotter
import intermagnet_loader as loader
import project_store as projects
import geomag_coords as gcoords

# Global persistent workspace across run_agent() invocations
_WORKSPACE = projects.Workspace()

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
    "run_agent_with_planner",
    "select_tools_for_task",
    "build_dynamic_prompt",
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

#: Ceiling on downloads inside one ``fetch_many`` call, whatever it asks for.
#: The agent's own budget bounds how many times it may *call* a tool, not how much
#: work one call may do -- without this a single call could download a year of
#: data for every known station and swamp both the network and the run.
MAX_BATCH_DOWNLOADS = 8

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

BASE_PROMPT_COMPACT = (
    "Ты — профессиональный геофизический ИИ-ассистент. Никогда не выдумывай цифры — "
    "вызывай инструменты.\n\n"
    "Правила:\n"
    "1. Отвечай на русском, кратко, как коллега-геофизик.\n"
    "2. Любое число в ответе — только из результата инструмента.\n"
    "3. Данные бери fetch_observatory_data. Код станции — трёхбуквенный IAGA. "
    f"Известны: {STATION_HINT}.\n"
    "4. Для графика: fetch_observatory_data → calculate_derived_components "
    "→ plot_components / plot_comparison / plot_overlay. Шаги не перескакивай.\n"
    "4a. ПЕРЕД ЛЮБЫМ из них — fetch_observatory_data: без хэндла raw математика "
    "и графики ответят unknown_frame. Хэндл бери из frame_handle, не выдумывай.\n"
    "5. Инструменты:\n"
    "    • Данные: fetch_observatory_data, fetch_many, list_projects, "
    "create_project, export_project.\n"
    "    • Пакетная работа: create_task_matrix (план \"события×станции\" одной "
    "матрицей), process_matrix_batch (выполнить батч из матрицы напрямую), "
    "export_project. При большом количестве станций/дат (>15 комбинаций) НЕ "
    "перечисляй задачи по одной — создай create_task_matrix c lists events, "
    "stations, actions и project_name, затем process_matrix_batch, пока в "
    "сводке не будет 'Осталось в матрице: 0', потом export_project.\n"
    "    • Математика: calculate_derived_components (считает только H, D, I; "
    "X, Y, Z, F уже есть в raw), get_statistics, detect_anomalies, "
    "calculate_derived_math — метрики 'delta', 'dH_dt', 'anomaly'; Метрики "
    "'range' НЕ СУЩЕСТВУЕТ; calculate_baseline — база для 'anomaly', значение "
    "в baseline_value; evaluate_custom_formula — формула не исполняется как код.\n"
    "    • Магнетизм: get_station_geomagnetic_coords, calculate_mlt (магнитное "
    "время), calculate_local_time (гражданское время), group_stations_by_mlt.\n"
    "    • Графики: plot_components, plot_comparison, plot_overlay (несколько "
    "станций; данные и H/D/I достаёт сам).\n"
    "6. Вместо DataFrame передавай строковый хэндл из frame_handle: "
    "\"raw:irt:2024-09-10\" / \"derived:irt:2024-09-10\" либо семейство "
    "\"raw\" / \"derived\" / \"anomalies\".\n"
    "7. Ошибку инструмента перескажи пользователю и предложи, что делать; свои "
    "цифры не подставляй.\n"
    "8. В ответе перечисли графики и файлы. Единицы — 'nT', не «нанотесла».\n"
    "9. Ты НЕ API-сервер: не отвечай {'content': ..., 'tool_calls': ...}.\n"
    "10. ФОРМАТ ОТВЕТА: (а) вызов инструмента — только теги, без слов:\n"
    '        <tool_call>{"name": "fetch_observatory_data", "arguments": '
    '{"station_code": "IRT", "start_date": "2024-09-10", "end_date": "2024-09-10"}}</tool_call>\n'
    "    (б) данных достаточно — обычный текст без тегов. Сомневаешься — бери (а).\n\n"
    f"11. {HANDLE_CASE_RULE}"
)

#: The base prompt the ordinary single-question loop uses. ``run_agent_with_planner``
#: builds narrower prompts for long-running jobs from :data:`BASE_PROMPT_COMPACT`.
SYSTEM_PROMPT = BASE_PROMPT_COMPACT


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
            "name": "create_project",
            "description": (
                "Start a named project. Every later artifact is filed under it as "
                "<station>/<date>/ with a manifest recording the observatory, the "
                "data version, the download time and the agent commit. Call this "
                "before fetch_many or export_project. Calling it again with the same "
                "name reuses the existing project rather than failing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Short human-readable project name, e.g. 'Boreal storm "
                            "2024'. Letters, digits, spaces, '-' and '_' only; the name "
                            "is normalised into a folder name."
                        ),
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_many",
            "description": (
                "Download one date range for SEVERAL observatories in a single call. "
                "Use this instead of repeating fetch_observatory_data when more than "
                "one station is needed: the stations share one budget of downloads, "
                "and each returns its own frame_handle such as 'raw:irt:2024-09-10'. "
                "A station that cannot be downloaded is reported in 'failed' while the "
                "others still succeed -- partial results are normal, so report what "
                "arrived and what did not. When a project is open, each day's data is "
                "also written into it as CSV."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            f"Three-letter IAGA codes. Known: {STATION_HINT}"
                        ),
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
                    "max_downloads": {
                        "type": "integer",
                        "description": (
                            "Ceiling on downloads for this call, shared by all stations. "
                            "Defaults to 8. Windows beyond the ceiling are refused and "
                            "listed under 'refused' rather than fetched."
                        ),
                    },
                    "project": {
                        "type": "string",
                        "description": "Project to file the data under. Defaults to the open project.",
                    },
                },
                "required": ["stations", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "???????? ?????? ???? ???????? ? ???????????",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "export_project",
            "description": (
                "Pack a whole project into a single ZIP for download: the folder tree, "
                "every CSV and HTML chart, and the manifests. Returns the archive path. "
                "Call this once the work is done and tell the user the file name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Project to export. Defaults to the open project.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task_matrix",
            "description": (
                "Plan a large batch job as ONE matrix: every combination of the "
                "given events and stations becomes a task in a JSON file, so the "
                "model never has to enumerate hundreds of tasks in its own reply. "
                "Use this INSTEAD of generating a long list of tool calls when the "
                "job spans many stations/dates (more than ~15 units). Then loop "
                "process_matrix_batch until its summary says 'Осталось в матрице: 0', "
                "and finally export_project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Event dates, YYYY-MM-DD, e.g. [\"2024-10-10\", \"2024-09-08\"].",
                    },
                    "stations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Three-letter IAGA codes. Known: {STATION_HINT}",
                    },
                    "actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "What to do for each (event, station) cell, in order. "
                            "Words 'fetch', 'calc_HDI', 'plot' map to the handlers; "
                            "full handler names also work. Default: fetch, calc_HDI, plot."
                        ),
                    },
                    "project_name": {
                        "type": "string",
                        "description": "Project that stores the results and the matrix.",
                    },
                },
                "required": ["events", "stations", "actions", "project_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_matrix_batch",
            "description": (
                "Execute the next batch of pending cells of a task matrix. Reads the "
                "JSON file created by create_task_matrix and runs each action (fetch, "
                "derive, plot) internally, without extra model round-trips. The summary "
                "reports progress; keep calling with 'Осталось в матрице: 0' (or "
                "remaining: 0) tells you all cells are done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "matrix_path": {
                        "type": "string",
                        "description": "The matrix_path returned by create_task_matrix.",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Pending cells to run this call. Default 5 (optimal 3-5).",
                    },
                },
                "required": ["matrix_path"],
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
    {
        "type": "function",
        "function": {
            "name": "get_station_geomagnetic_coords",
            "description": (
                "Tilted-dipole geomagnetic coordinates (geomag_lat, geomag_lon) of one station. "
                "USE when the user asks how far north a station is in geomagnetic terms, or before "
                "comparing storm amplitudes across stations. DO NOT USE to get the station's "
                "geographic latitude/longitude, and do not call it for data download."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "station_code": {"type": "string", "description": "IAGA station code"}
                },
                "required": ["station_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_mlt",
            "description": (
                "Magnetic Local Time (MLT) of one station at a UTC instant; mlt_hours in [0,24), "
                "MLT=12 means magnetic noon. USE for 'what magnetic time is it at X'. DO NOT USE "
                "when the user wants ordinary civil local time -- that is calculate_local_time; "
                "MLT and LT differ by the magnetic-vs-geographic longitude."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "station_code": {"type": "string", "description": "IAGA station code"},
                    "timestamp": {"type": "string", "description": "ISO 8601 UTC timestamp"}
                },
                "required": ["station_code", "timestamp"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "group_stations_by_mlt",
            "description": (
                "Group several stations into MLT bins at one UTC instant. USE to see which stations "
                "sit in the same magnetic-time sector. DO NOT USE for local/civil time, and do not "
                "call it once per station -- it takes the whole list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stations": {"type": "array", "items": {"type": "string"}, "description": "IAGA station codes"},
                    "timestamp": {"type": "string", "description": "ISO 8601 UTC timestamp"},
                    "bin_hours": {"type": "integer", "description": "Bin width in hours (default 1)"}
                },
                "required": ["stations", "timestamp"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_local_time",
            "description": (
                "Civil local time (LT) at a station: LT = UT + longitude/15, wrapped to [0,24), "
                "returned as lt_hours and lt_hm. USE for 'what local time is it at X'. DO NOT USE "
                "for magnetic local time -- that is calculate_mlt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "station_code": {"type": "string", "description": "IAGA station code"},
                    "timestamp": {"type": "string", "description": "ISO 8601 UTC timestamp"}
                },
                "required": ["station_code", "timestamp"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_overlay",
            "description": (
                "Draw one component from SEVERAL stations on ONE chart, each curve shifted by a "
                "per-station nT offset, with the x axis in UT, LT or MLT. Stations are ordered "
                "north-first by geomagnetic latitude. AUTO-FETCHES any station/day that is not "
                "loaded yet and derives H, D, I on the spot, so no fetch_observatory_data call "
                "is needed first. USE to compare several observatories, or to stack them on a "
                "local/magnetic clock. DO NOT USE to compare two days of a single station -- "
                "that is plot_comparison."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IAGA station codes to overlay, e.g. ['IRT','API','BSL']"
                    },
                    "dates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dates as YYYY-MM-DD; one date shared by all stations."
                    },
                    "component": {"type": "string", "enum": ["X", "Y", "Z", "F", "H", "D", "I"]},
                    "offsets": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "nT shift per station for visual separation, e.g. {'IRT': 0, 'API': 150}. Missing stations get 0."
                    },
                    "time_system": {"type": "string", "enum": ["UT", "LT", "MLT"], "description": "X-axis time system, default UT."},
                    "title": {"type": "string", "description": "Chart title."},
                    "filename": {"type": "string", "description": "Output file name, .html."}
                },
                "required": ["stations", "dates", "component"]
            }
        }
    },
]

TOOLS_BY_NAME = {t["function"]["name"] for t in TOOL_SCHEMAS}
PLOT_TOOLS = {"plot_components", "plot_comparison", "plot_overlay"}

# Generator-Executor batch jobs. A 10-events x 26-stations request is 260 units
# of work; asking the model to enumerate them in one JSON reply overflows its
# context, so anything above MATRIX_THRESHOLD units is executed through a task
# matrix instead: create_task_matrix writes the plan once, and
# process_matrix_batch grinds through it in small chunks calling the handlers
# directly (no per-step LLM round-trip).
MATRIX_THRESHOLD = 15
MATRIX_BATCH_SIZE = 5
_DEFAULT_MATRIX_ACTIONS = [
    "fetch_observatory_data",
    "calculate_derived_components",
    "plot_components",
]
# Friendly action words a model may use instead of handler names.
MATRIX_ACTION_ALIASES = {
    "fetch": "fetch_observatory_data",
    "calc_hdi": "calculate_derived_components",
    "calculate_hdi": "calculate_derived_components",
    "hdi": "calculate_derived_components",
    "derive": "calculate_derived_components",
    "plot": "plot_components",
    "stats": "get_statistics",
    "statistics": "get_statistics",
    "anomalies": "detect_anomalies",
}
# Handler names that do one unit of per-(station, date) work. When a generated
# plan is made *entirely* of these the runner can factor it back into a matrix
# without asking the model to re-enumerate the combos.
_MATRIX_PER_UNIT_TOOLS = {
    "fetch_observatory_data",
    "calculate_derived_components",
    "get_statistics",
    "detect_anomalies",
    "calculate_derived_math",
    "plot_components",
    "plot_overlay",
    "calculate_mlt",
    "calculate_local_time",
}

#: Which tools a planner subtask may need. ``run_agent`` gets everything; a
#: focused tool (say ``calculate_mlt``) gets only the two-step chain that can
#: possibly serve it, so the model sees a short schema list instead of all 18.
TOOL_SELECTION: dict[str, list[str]] = {
    "plot_components": ["fetch_observatory_data", "calculate_derived_components", "plot_components"],
    "plot_comparison": ["fetch_observatory_data", "calculate_derived_components", "plot_comparison"],
    "plot_overlay": ["fetch_observatory_data", "calculate_derived_components", "plot_overlay"],
    "get_statistics": ["fetch_observatory_data", "get_statistics"],
    "calculate_derived_math": ["fetch_observatory_data", "calculate_derived_math"],
    "detect_anomalies": ["fetch_observatory_data", "detect_anomalies"],
    "calculate_local_time": ["calculate_local_time"],
    "calculate_mlt": ["get_station_geomagnetic_coords", "calculate_mlt"],
    "group_stations_by_mlt": ["get_station_geomagnetic_coords", "group_stations_by_mlt"],
    "create_project": ["create_project"],
    "fetch_many": ["fetch_many"],
    "export_project": ["export_project"],
    "create_task_matrix": [
        "create_task_matrix",
        "process_matrix_batch",
        "fetch_observatory_data",
        "calculate_derived_components",
        "plot_components",
        "export_project",
    ],
    "process_matrix_batch": [
        "process_matrix_batch",
        "fetch_observatory_data",
        "calculate_derived_components",
        "plot_components",
        "export_project",
    ],
    "run_agent": list(TOOLS_BY_NAME),
}


def select_tools_for_task(tool_name: str) -> list[dict[str, Any]]:
    """Return only the schemas a subtask with *tool_name* may need.

    The full 18-tool schema list goes into every ordinary ``run_agent`` call; a
    subtask is the opposite -- one job, one tool, and just enough context to
    fetch what it needs. Unknown names fall back to the full list, because a
    tool the planner has never heard of must not be hidden from the model.
    """
    needed = TOOL_SELECTION.get(str(tool_name or ""), list(TOOLS_BY_NAME))
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] in needed]


def build_dynamic_prompt(
    base_prompt: str, tools: list[dict[str, Any]], context: str
) -> str:
    """Assemble a short system prompt for one focused subtask.

    From the compact base prompt it keeps the behavioural rules (lowercase
    handles, no invented numbers, the nT unit) and appends only the schemas
    relevant to the current step plus the session's TODO context. Long example
    tables that belong to single-shot use are not repeated here.
    """
    tools_text = "\n".join(
        f"- {t['function']['name']}: {t['function'].get('description', '')[:100]}"
        for t in tools
    )
    return f"""{base_prompt}

Доступные инструменты для текущей задачи:
{tools_text}

Контекст выполнения:
{context}

Выполни текущую подзадачу. Если нужны данные — сначала вызови инструменты для их получения.
"""


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

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        response = self._llm.create_chat_completion(
            messages=messages,
            tools=tools,
            temperature=self._temperature,
            max_tokens=self._max_tokens if max_tokens is None else max_tokens,
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


def _unwrap_doubled_braces(text: str) -> Iterator[str]:
    """Yield *text* with one redundant outer brace layer peeled off.

    Asked for a tool call, a model sometimes wraps the object in a second pair
    of braces: ``{{"name": ..., "arguments": {...}}}``. ``json.loads`` rejects
    that outright, and the span scanner cannot rescue it either -- it balances
    the braces it sees and yields the same doubled shape back, which is why
    ``{{...}}}`` outside a tag produced no call at all rather than a bad one.

    How many braces are spare varies: the wrapper sometimes closes (``{{...}}``)
    and sometimes not (``{{...}}}``), so trailing candidates are generated too
    and the first one that parses wins. Removing only from the ends is the point.
    A blanket ``text.replace("{{", "{")`` would rewrite braces inside string
    values, and this agent lets a model pass arbitrary text through verbatim.

    Yields nothing unless the text actually opens with a doubled brace, so a
    well-formed call never enters this path.
    """
    stripped = text.strip()
    if not stripped.startswith("{{"):
        return
    opens = 2
    while opens < 4 and stripped[opens : opens + 1] == "{":
        opens += 1
    for drop_open in range(1, opens):
        head = stripped[drop_open:]
        for drop_close in (0, 1, 2):
            body = head[: len(head) - drop_close] if drop_close else head
            body = body.rstrip()
            if body and body != stripped:
                yield body


def _load_bases(raw: str) -> Iterator[str]:
    """The quote- and comma-repaired readings of *raw*, lazier than eager."""
    for base in (_straighten_quotes(raw), raw):
        yield _drop_trailing_commas(_single_quotes_to_double(base))


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
    bases = list(_load_bases(raw))
    for repaired in bases:
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            pass
    for repaired in bases:
        for unwrapped in _unwrap_doubled_braces(repaired):
            try:
                return json.loads(unwrapped)
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
    return [(str(name
).strip(), _coerce_arguments(arguments))]


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

    def __init__(self, workspace: "projects.Workspace | None" = None) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        # family -> slots in write order; the last one is what the alias points at
        self._families: dict[str, list[str]] = {}
        # The project this run is filling, if the user opened one. Kept here so
        # every tool keeps the same (args, store) signature: a second channel
        # into run_agent would have to be threaded past every existing handler and
        # every test that calls one directly.
        self.workspace = workspace if workspace is not None else _WORKSPACE

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


def _is_derived_handle(handle: Any) -> bool:
    """Whether *handle* explicitly names the derived family, not a bare alias."""
    return isinstance(handle, str) and handle.strip().lower().startswith("derived:")


def _auto_derive(
    store: FrameStore, handle: Any, components: Sequence[str]
) -> tuple[str | None, str]:
    """Materialise a missing ``derived:`` slot from its ``raw:`` sibling.

    A model asked to plot a derived component straight after fetching usually
    infers the slot name from the naming convention rather than reading it off a
    tool result, so it asks for ``derived:irt:2024-09-11`` while only
    ``raw:irt:2024-09-11`` is loaded. The convention is right and the slot simply
    has not been created yet; reporting ``unknown_frame`` there spends a round
    teaching the model something the code can just do.

    Scoped deliberately. Only the ``derived`` family qualifies, because deriving
    is a pure function of raw columns. Only a slot whose raw sibling is already
    stored qualifies, so the fallback can never invent data the model never
    fetched. ``anomalies`` is excluded: those depend on a baseline and a
    detection window, so they stay an explicit call.

    Derives only the components the caller actually asked for, and only those
    that are derivable -- X, Y, Z and F are already on the raw frame, and
    ``calculate_derived_components`` rejects a request it cannot satisfy.

    Returns ``(slot, note)``; ``slot`` is None when nothing was derived, leaving
    the caller to report the miss normally.
    """
    if not _is_derived_handle(handle):
        return None, ""
    key = handle.strip().lower()
    raw = store.get("raw:" + key.split(":", 1)[1])
    if raw is None:
        return None, ""
    wanted = [
        name
        for name in (str(c).strip().upper() for c in components)
        if name in analyzer.DERIVED
    ]
    result = analyzer.calculate_derived_components(raw, components=wanted)
    if analyzer.is_error(result):
        # Fall through to the normal unknown_frame error rather than surfacing a
        # compute failure the model cannot act on -- it usually means the raw
        # frame lacks the Cartesian columns, and the handles are what it needs.
        return None, ""
    slot = store.put(key, result)
    return slot, f"auto-derived: {slot} (из raw)"


def _resolve_plottable(
    store: FrameStore, handle: Any, components: Sequence[str], notes: list[str]
) -> tuple[Any, dict[str, Any] | None]:
    """Resolve a handle for plotting, deriving a missing derived slot from raw.

    A derived handle that cannot be derived is returned as a miss rather than
    falling back to a family scan. Substituting whichever other frame happened
    to be loaded would plot the wrong day under the name the model asked for,
    which is far harder to notice than an error.
    """
    frame, err = store.resolve(handle)
    if frame is not None:
        return frame, None
    if not _is_derived_handle(handle):
        return None, err
    slot, note = _auto_derive(store, handle, components)
    if slot is None:
        return None, err
    notes.append(note)
    return store.resolve(slot)


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


def _file_into_project(
    store: FrameStore, chart: str, handles: Sequence[Any], kind: str
) -> list[str]:
    """Copy a saved chart into the project days its handles point at.

    Returns an empty list when no project is open, which is every pre-project
    flow: the plotter's own path is still what those runs report, so nothing about
    the existing behaviour changes.
    """
    try:
        return store.workspace.save_chart(chart, handles, kind=kind)
    except OSError as exc:
        # A project that cannot be written must not sink the chart: the chart is
        # the answer, the filing is bookkeeping.
        print(f"[agent] could not file {chart} into the project: {exc}")
        return []


def _handle_plot_components(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    components = args.get("components") or ["H"]
    notes: list[str] = []
    frame, err = _resolve_plottable(store, args.get("df"), components, notes)
    if frame is None and not _is_derived_handle(args.get("df")):
        # No handle, or a bare family alias: search for the richest slot that
        # actually carries the requested column. An explicit derived handle is
        # excluded on purpose -- see _resolve_plottable.
        handle = _first_present(frame, store, components[0])
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
    filed = _file_into_project(store, result, [args.get("df")], "components")
    payload = {"ok": True, "path": result, "components": components, "rows": int(len(frame))}
    if filed:
        payload["filed_into_project"] = filed
    return payload, "; ".join([*notes, f"Saved chart to {result}"])


def _handle_plot_comparison(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    component = args.get("component") or "H"
    notes: list[str] = []
    first, err = _resolve_plottable(store, args.get("df1"), [component], notes)
    if err:
        return err, f"plot_comparison failed (df1): {err['error']}"
    second, err = _resolve_plottable(store, args.get("df2"), [component], notes)
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
        component=component,
        title=args.get("title") or "Day comparison",
        filename=args.get("filename"),
    )
    if plotter.is_error(result):
        return result, f"plot_comparison failed: {result.get('error')}"
    filed = _file_into_project(
        store, result, [args.get("df1"), args.get("df2")], "comparison"
    )
    payload = {"ok": True, "path": result, "component": component}
    if filed:
        payload["filed_into_project"] = filed
    return payload, "; ".join([*notes, f"Saved comparison to {result}"])


# --------------------------------------------------------------------------- #
# projects: a folder tree the output actually lives in
# --------------------------------------------------------------------------- #
def _date_range(start: str, end: str) -> list[str]:
    """Every date from ``start`` to ``end`` inclusive, as ``YYYY-MM-DD``.

    Parsed by hand rather than with ``date.fromisoformat`` so a malformed date
    fails with the model's own string in the message: the loop below reports a
    bad date against the day it came from, and a bare ``ValueError`` would not
    say which of eight windows was wrong.
    """
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(start or "").strip())
    if not match:
        return []
    try:
        first = date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return []
    last = first
    end_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(end or start).strip())
    if end_match:
        try:
            last = date(int(end_match[1]), int(end_match[2]), int(end_match[3]))
        except ValueError:
            last = first
    if last < first:
        first, last = last, first
    out: list[str] = []
    current = first
    # Bounded so a reversed or absurd window cannot spin: the download budget
    # caps real work anyway, and an unbounded date loop is a hang.
    while current <= last and len(out) < 366:
        out.append(current.isoformat())
        current += timedelta(days=1)
    return out


def _handle_create_project(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    result = store.workspace.create(args.get("name"))
    if projects.is_error(result):
        return result, f"create_project failed: {result['error']}"
    return {
        "ok": True,
        "project": result["project"],
        "path": result["path"],
        "layout": "projects/<project>/<station>/<date>/",
        "available": store.workspace.known(),
        "hint": (
            "Now use fetch_many or fetch_observatory_data; each result is filed "
            "under this project automatically, and export_project zips the tree."
        ),
    }, result["note"]


def _handle_fetch_many(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Fetch one date range for several stations under a single download budget.

    Partial failure is the expected case, not an exception: an observatory that
    has not published a definitive value for a date will fail, and the run should
    still be able to answer about the stations that worked. So every window is
    attempted, each outcome is recorded separately, and only a malformed request
    returns an error payload.
    """
    raw_stations = args.get("stations")
    if isinstance(raw_stations, str):
        raw_stations = [raw_stations]
    stations = [str(s).strip().upper() for s in (raw_stations or []) if str(s).strip()]
    if not stations:
        return _error(
            "missing_stations",
            "fetch_many needs at least one station code.",
            received=raw_stations,
            hint=f"Pass 'stations' as a list of IAGA codes. Known: {STATION_HINT}",
        ), "fetch_many refused: no station given."

    days = _date_range(args.get("start_date"), args.get("end_date"))
    if not days:
        return _error(
            "invalid_date",
            "start_date must be YYYY-MM-DD and end_date the same or later.",
            received={"start_date": args.get("start_date"), "end_date": args.get("end_date")},
        ), "fetch_many refused: the date range could not be read."

    # One ceiling for the whole call, not one per station. Three stations over ten
    # days is thirty downloads; a per-station budget would quietly promise ninety.
    wanted = args.get("max_downloads")
    budget = MAX_BATCH_DOWNLOADS
    if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted > 0:
        budget = min(wanted, MAX_BATCH_DOWNLOADS)

    project = store.workspace.resolve(args.get("project"))
    data_type = args.get("data_type") or "definitive"
    samples = args.get("samples_per_day") or "Minute"

    fetched: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    spent = 0

    for station in stations:
        for day in days:
            if spent >= budget:
                refused.append(
                    {"station": station, "date": day, "reason": "download budget exhausted"}
                )
                continue
            spent += 1
            # Counted before the call, not after: a request that hangs or raises
            # still consumed the attempt, and a budget that only decrements on
            # success is not a budget.
            payload, _ = _handle_fetch(
                {
                    "station_code": station,
                    "start_date": day,
                    "end_date": day,
                    "data_type": data_type,
                    "samples_per_day": samples,
                },
                store,
            )
            if projects.is_error(payload):
                failed.append(
                    {
                        "station": station,
                        "date": day,
                        "error": payload.get("error"),
                        "message": payload.get("message"),
                    }
                )
                continue
            handle = payload.get("frame_handle") or f"raw:{station.lower()}:{day}"
            frame = store.get(handle)
            saved = (
                store.workspace.save(station, day, frame, kind="raw", project=project)
                if frame is not None
                else {}
            )
            fetched.append(
                {
                    "station": station,
                    "date": day,
                    "frame_handle": handle,
                    "rows": payload.get("rows"),
                    "publication_state": payload.get("publication_state"),
                    "offline": bool(payload.get("offline")),
                    "saved_to_project": saved.get("path"),
                }
            )

    if not fetched and not failed and not refused:
        return _error(
            "nothing_to_fetch",
            "fetch_many had no station/day window to download.",
            stations=stations, dates=days,
        ), "fetch_many had nothing to do."

    result: dict[str, Any] = {
        "ok": True,
        "requested": {"stations": stations, "dates": days, "project": project},
        "downloaded": spent,
        "budget": budget,
        "fetched": fetched,
        "failed": failed,
        "refused": refused,
        "available_handles": store.names(),
        "frame_handles": [item["frame_handle"] for item in fetched],
    }
    # A partial batch is still ok:true -- the stations that worked are usable. The
    # counts are what let the model say "two of eight" instead of either
    # overclaiming or reporting total failure.
    result["summary"] = (
        f"{len(fetched)} downloaded, {len(failed)} failed, {len(refused)} refused "
        f"of {spent} attempts (budget {budget})"
    )
    return result, f"fetch_many: {result['summary']}."


def _handle_export_project(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    name = store.workspace.resolve(args.get("name"))
    if not name:
        return _error(
            "no_project",
            "No project is open. Call create_project first, or pass 'name'.",
            available=store.workspace.known(),
        ), "export_project refused: no project to export."
    result = projects.export_project(name, root=store.workspace.root)
    if projects.is_error(result):
        return result, f"export_project failed: {result['error']}"
    return {
        "ok": True,
        "project": result["project"],
        "path": result["path"],
        "filename": Path(result["path"]).name,
        "size": result["size"],
        "files": result["files"],
        "tree": store.workspace.tree_markdown(name),
    }, f"Exported project {result['project']!r} to {result['path']} ({result['size']})."


def _matrix_dir(workspace: Any) -> Path:
    """Directory holding task matrices, next to the project they belong to."""
    return Path(workspace.root) if workspace else Path(".matrices")


def _matrix_path(project_name: str, workspace: Any) -> Path:
    """``<project>/<project_name>_matrix.json`` -- inside the project, so the
    archive that export_project builds carries the job ledger with it."""
    target = projects.project_dir(project_name, workspace.root)
    return target / f"{project_name}_matrix.json"


def _normalise_matrix_events(raw: Any) -> list[str]:
    """Accept a list (or single) of ``YYYY-MM-DD`` event dates, in order."""
    items = [raw] if isinstance(raw, str) else list(raw or [])
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            try:
                date(int(text[:4]), int(text[5:7]), int(text[8:10]))
                out.append(text)
            except ValueError:
                continue
    return out


def _normalise_matrix_stations(raw: Any) -> list[str]:
    items = [raw] if isinstance(raw, str) else list(raw or [])
    out: list[str] = []
    for item in items:
        code = str(item or "").strip().upper()
        if re.fullmatch(r"[A-Z]{2,4}", code) and code not in out:
            out.append(code)
    return out


def _normalise_matrix_actions(raw: Any) -> list[str] | None:
    """Map friendly action words to handler names. None means "malformed"."""
    if raw is None or (isinstance(raw, list) and not raw):
        return list(_DEFAULT_MATRIX_ACTIONS)
    items = [raw] if isinstance(raw, str) else list(raw or [])
    resolved: list[str] = []
    for item in items:
        word = str(item or "").strip()
        tool = MATRIX_ACTION_ALIASES.get(word.lower(), word)
        if tool in HANDLERS and tool not in resolved:
            resolved.append(tool)
    return resolved or None


def _matrix_task_args(
    tool: str, station: str, event: str, store: FrameStore
) -> dict[str, Any]:
    """Build the handler arguments for one (station, event) cell of a matrix."""
    raw_slot = f"raw:{station.lower()}:{event}"
    derived_slot = f"derived:{station.lower()}:{event}"
    if tool == "fetch_observatory_data":
        return {"station_code": station, "start_date": event, "end_date": event}
    if tool == "calculate_derived_components":
        return {"df": raw_slot, "components": ["H", "D", "I"]}
    if tool in ("plot_components", "plot_overlay"):
        frame = store.get(derived_slot)
        df = derived_slot if frame is not None else raw_slot
        return {
            "df": df,
            "components": ["H", "D", "I"],
            "filename": f"{station}_{event}_components.html",
        }
    if tool == "get_statistics":
        return {"df": raw_slot, "components": ["X", "Y", "Z", "F"]}
    if tool == "detect_anomalies":
        frame = store.get(derived_slot)
        return {"df": derived_slot if frame is not None else raw_slot}
    if tool == "calculate_mlt":
        return {"station_code": station, "df": raw_slot}
    if tool == "calculate_local_time":
        return {"station_code": station, "df": raw_slot}
    return {}


def _ensure_matrix_fetch(
    station: str, event: str, store: FrameStore
) -> tuple[bool, str]:
    """Make sure the cell's raw frame exists, fetching it if not."""
    slot = f"raw:{station.lower()}:{event}"
    if store.get(slot) is not None:
        return True, ""
    payload, note = _handle_fetch(
        {"station_code": station, "start_date": event, "end_date": event}, store
    )
    if is_error(payload):
        return False, f"fetch: {payload.get('error')}: {str(payload.get('message'))[:200]}"
    frame = store.get(slot)
    if frame is not None:
        store.workspace.save(station, event, frame, kind="raw")
    return True, note


def _ensure_matrix_derived(
    station: str, event: str, store: FrameStore
) -> tuple[bool, str]:
    """Make sure the cell's derived H/D/I frame exists, deriving it if not.

    Mirrors what the single-agent loop does for you: a plot request with no
    explicit derive step still gets its H/D/I components.
    """
    slot = f"derived:{station.lower()}:{event}"
    if store.get(slot) is not None:
        return True, ""
    ok, note = _ensure_matrix_fetch(station, event, store)
    if not ok:
        return False, note
    payload, _ = _handle_derive(
        {"df": f"raw:{station.lower()}:{event}", "components": ["H", "D", "I"]}, store
    )
    if is_error(payload):
        return False, (
            f"calculate_derived_components: {payload.get('error')}: "
            f"{str(payload.get('message'))[:200]}"
        )
    frame = store.get(slot)
    if frame is not None:
        store.workspace.save(station, event, frame, kind="derived")
    return True, ""


def _execute_matrix_task(task: dict[str, Any], store: FrameStore) -> tuple[bool, str]:
    """Run every action of one matrix cell by calling handlers directly.

    This is the whole point of the Generator-Executor pattern: the batch tool
    executes ``fetch -> derive -> plot`` as plain Python calls, so a 260-cell job
    costs 260/5 = 52 tool calls instead of hundreds of model round-trips.
    """
    station = task.get("station", "")
    event = task.get("event", "")
    for tool in task.get("actions") or _DEFAULT_MATRIX_ACTIONS:
        if tool not in HANDLERS:
            return False, f"Неизвестное действие {tool!r} в матрице."
        # Data prerequisites: a cell whose actions omit fetch/derive must not
        # fail on an unknown_frame -- the plot just gets its data implicitly.
        if tool != "fetch_observatory_data":
            ok, note = _ensure_matrix_fetch(station, event, store)
            if not ok:
                return False, note
        if tool in ("plot_components", "plot_overlay"):
            derived_slot = f"derived:{station.lower()}:{event}"
            if store.get(derived_slot) is None:
                ok, note = _ensure_matrix_derived(station, event, store)
                if not ok:
                    return False, note
        try:
            payload, _ = HANDLERS[tool](
                _matrix_task_args(tool, station, event, store), store
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return False, f"{tool} raised {type(exc).__name__}: {exc}"
        if is_error(payload):
            return (
                False,
                f"{tool}: {payload.get('error')}: {str(payload.get('message'))[:200]}",
            )
        if tool == "fetch_observatory_data" and payload.get("frame_handle"):
            frame = store.get(payload["frame_handle"])
            if frame is not None:
                store.workspace.save(station, event, frame, kind="raw")
        if tool == "calculate_derived_components":
            derived = store.get(f"derived:{station.lower()}:{event}")
            if derived is not None:
                store.workspace.save(station, event, derived, kind="derived")
    return True, ""


def _handle_create_task_matrix(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Write the full events x stations plan to ``<project>/_matrix.json``.

    The matrix is the *generator* half of the pattern: one JSON file holds every
    pending unit, so the model never has to enumerate them in its own reply.
    """
    project_name = str(args.get("project_name") or "").strip()
    if not project_name:
        return _error(
            "missing_project_name",
            "create_task_matrix needs a project_name.",
            hint="Name the project so export_project can later archive the results.",
        ), "create_task_matrix refused: project_name is required."

    events = _normalise_matrix_events(args.get("events"))
    stations = _normalise_matrix_stations(args.get("stations"))
    if not events:
        return _error(
            "invalid_events",
            "events must be a list of YYYY-MM-DD dates.",
            received=args.get("events"),
        ), "create_task_matrix refused: no valid event dates."
    if not stations:
        return _error(
            "invalid_stations",
            "stations must be a list of IAGA codes.",
            received=args.get("stations"),
        ), "create_task_matrix refused: no valid station codes."
    actions = _normalise_matrix_actions(args.get("actions"))
    if not actions:
        return _error(
            "invalid_actions",
            "actions must be handler names or known words (fetch, calc_HDI, plot).",
            received=args.get("actions"),
        ), "create_task_matrix refused: no usable actions."

    # Open (or reuse) the project so data and charts are filed under it and the
    # final export_project has a tree to zip. create_project is idempotent.
    if project_name not in store.workspace.known():
        created = store.workspace.create(project_name)
        if is_error(created):
            return created, f"create_task_matrix failed: {created['error']}"
    else:
        store.workspace.project = project_name

    tasks = [
        {
            "event": event,
            "station": station,
            "status": "pending",
            "attempts": 0,
            "error": None,
            "actions": list(actions),
        }
        for event in events
        for station in stations
    ]
    matrix: dict[str, Any] = {
        "project": project_name,
        "total": len(tasks),
        "completed": 0,
        "failed": 0,
        "pending": len(tasks),
        "tasks": tasks,
    }
    path = _matrix_path(project_name, store.workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = f"Матрица создана. Всего задач: {len(tasks)}. Файл: {path}"
    return {
        "ok": True,
        "project": project_name,
        "total": len(tasks),
        "matrix_path": str(path),
        "actions": actions,
        "sample_tasks": tasks[:3],
        "summary": summary,
    }, summary


def _handle_process_matrix_batch(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Execute the next chunk of pending matrix cells, handling tools directly.

    Reads ``matrix_path``, runs the first ``batch_size`` pending tasks through
    :func:`_execute_matrix_task` -- direct Python handler calls, no model in the
    loop -- then persists the updated matrix. The summary line is the *executor*
    contract: "Осталось в матрице: 0" is what tells a driver loop it is done.
    """
    value = str(args.get("matrix_path") or "").strip()
    path: Path | None = None
    if value:
        candidate = Path(os.path.expanduser(value))
        if candidate.is_file():
            path = candidate
        elif "/" in value or "\\" in value:
            path = candidate if candidate.is_file() else None
        else:
            # A bare file name: look inside every project under this workspace.
            for known in projects.list_projects(store.workspace.root):
                probe = projects.project_dir(known, store.workspace.root) / value
                if probe.is_file():
                    path = probe
                    break
    if path is None or not path.is_file():
        return _error(
            "matrix_not_found",
            "No matrix at the given path.",
            received=value,
            hint="Pass the matrix_path returned by create_task_matrix.",
        ), f"process_matrix_batch refused: {value!r} is not a matrix file."

    try:
        matrix = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _error("matrix_unreadable", f"Cannot read matrix: {exc}"), "matrix unreadable."

    raw_size = args.get("batch_size", MATRIX_BATCH_SIZE)
    try:
        batch_size = max(1, min(int(raw_size), 20))
    except (TypeError, ValueError):
        batch_size = MATRIX_BATCH_SIZE

    tasks = matrix.get("tasks") or []
    pending = [t for t in tasks if t.get("status") == "pending"]
    chunk = pending[:batch_size]

    succeeded = 0
    failed_here = 0
    processed: list[dict[str, Any]] = []
    for task in chunk:
        task["attempts"] = int(task.get("attempts") or 0) + 1
        ok, reason = _execute_matrix_task(task, store)
        if ok:
            task["status"] = "done"
            task["error"] = None
            succeeded += 1
        else:
            task["status"] = "failed"
            task["error"] = reason
            failed_here += 1
        processed.append({"event": task["event"], "station": task["station"], "ok": ok, "error": None if ok else reason})

    done = sum(1 for t in tasks if t.get("status") == "done")
    failed = sum(1 for t in tasks if t.get("status") == "failed")
    remaining = sum(1 for t in tasks if t.get("status") == "pending")
    matrix["completed"] = done
    matrix["failed"] = failed
    matrix["pending"] = remaining
    path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = (
        f"Обработано {len(chunk)} задач. Успешно: {succeeded}, Ошибок: {failed_here}. "
        f"Осталось в матрице: {remaining}. Текущий прогресс: {done}/{matrix.get('total', 0)}."
    )
    return {
        "ok": True,
        "processed": len(chunk),
        "succeeded": succeeded,
        "failed": failed_here,
        "remaining": remaining,
        "total": matrix.get("total", 0),
        "progress": f"{done}/{matrix.get('total', 0)}",
        "matrix_path": str(path),
        "results": processed,
        "summary": summary,
    }, summary


def _factor_plan_for_matrix(
    tasks: Sequence[Any],
) -> tuple[list[str], list[str], list[str], str] | None:
    """Re-derive (events, stations, actions, project) from a >threshold plan.

    A deterministic planner that expands "26 станций" already produced one task
    per station; instead of running 26 model round-trips we fold the plan back
    into a matrix. Returns None when the plan mixes tools a matrix cell cannot
    express, in which case the caller executes the plan as generated.
    """
    events: set[str] = set()
    stations: set[str] = set()
    actions: list[str] = []
    project_name = ""
    for task in tasks:
        name = getattr(task, "tool_name", None)
        tool_args = getattr(task, "tool_args", None) or {}
        if name == "create_project":
            project_name = str(
                tool_args.get("project_name") or tool_args.get("name") or project_name
            )
            continue
        if name == "export_project":
            continue
        if name in ("create_task_matrix", "process_matrix_batch"):
            return None
        if name not in _MATRIX_PER_UNIT_TOOLS:
            return None
        station = str(
            tool_args.get("station_code") or tool_args.get("station") or ""
        ).strip().upper()
        event = str(
            tool_args.get("start_date")
            or tool_args.get("end_date")
            or tool_args.get("date")
            or ""
        ).strip()
        if station:
            stations.add(station)
        if event:
            events.add(event)
        if name not in actions:
            actions.append(name)
    if not events or not stations or not actions:
        return None
    return sorted(events), sorted(stations), actions, project_name or "batch_matrix"


def _run_matrix_job(
    events: Sequence[str],
    stations: Sequence[str],
    actions: Sequence[str],
    project_name: str,
    store: FrameStore,
    verbose: bool = False,
) -> dict[str, Any]:
    """create_task_matrix -> loop process_matrix_batch -> export_project.

    The full Generator-Executor run, called by :func:`run_agent_with_planner`
    whenever a plan degenerates into hundreds of per-unit tasks. Each batch is
    executed by handlers directly, so the whole job runs in Python.
    """
    if verbose:
        print(
            f"🏭 Матрица: {len(events)} событий x {len(stations)} станций = "
            f"{len(events) * len(stations)} задач (проект {project_name!r})."
        )
    create_payload, create_note = _handle_create_task_matrix(
        {
            "events": list(events),
            "stations": list(stations),
            "actions": list(actions),
            "project_name": project_name,
        },
        store,
    )
    if is_error(create_payload):
        return {
            "ok": False,
            "text": create_payload.get("message", create_note),
            "results": [],
            "matrix_path": None,
            "export": create_payload,
            "summary": create_note,
            "store": store,
        }
    path = create_payload["matrix_path"]

    batches: list[dict[str, Any]] = []
    done = 0
    failed = 0
    while True:
        payload, note = _handle_process_matrix_batch(
            {"matrix_path": path, "batch_size": MATRIX_BATCH_SIZE}, store
        )
        batches.append({"ok": not is_error(payload), "payload": payload, "note": note})
        if is_error(payload):
            return {
                "ok": False,
                "text": payload.get("message", note),
                "results": batches,
                "matrix_path": path,
                "export": None,
                "summary": note,
                "store": store,
            }
        done += int(payload["succeeded"])
        failed += int(payload["failed"])
        if verbose:
            print(f"   {payload['summary']}")
        if payload["processed"] == 0 or payload["remaining"] == 0:
            break

    export_payload, export_note = _handle_export_project({"name": project_name}, store)
    archive = export_payload.get("path") if not is_error(export_payload) else export_payload.get("message")
    text = (
        f"Матрица {project_name}: выполнено {done}, ошибок {failed} "
        f"из {create_payload['total']} задач. Архив: {archive}."
    )
    return {
        "ok": failed == 0,
        "text": text,
        "results": batches,
        "matrix_path": path,
        "export": export_payload,
        "summary": text,
        "store": store,
    }


def _handle_list_projects(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    projects_list = store.workspace.known()
    details = []
    for name in projects_list:
        manifest = projects.read_project_manifest(name, root=store.workspace.root)
        if projects.is_error(manifest):
            details.append({"name": name, "manifest_available": False})
        else:
            details.append({
                "name": name,
                "manifest_available": True,
                "stations": manifest.get("stations", []),
                "dates": manifest.get("dates", []),
                "created_at": manifest.get("created_at"),
                "updated_at": manifest.get("updated_at"),
            })
    return {
        "ok": True,
        "projects": projects_list,
        "details": details,
        "current": store.workspace.project,
    }, f"list_projects: {len(projects_list)} projects found."



def _handle_get_station_geomagnetic_coords(args, store):
    code = str(args.get('station_code') or '').strip().upper()
    if not code:
        return _error('invalid_input', 'station_code is required'), 'station_code is required'
    try:
        from intermagnet_loader import get_available_stations
        rec = get_available_stations().get(code)
    except Exception as exc:
        return _error('registry_error', str(exc)), 'registry error'
    if not rec:
        return (
            _error(
                'station_not_found',
                f'Station {code} is not in the INTERMAGNET registry',
                hint='Check the IAGA code',
            ),
            f'station_not_found: {code}',
        )
    if not (isinstance(rec, tuple) and len(rec) >= 3):
        return _error('registry_error', 'unexpected registry format'), 'registry error'
    _name, lat, lon = rec[0], rec[1], rec[2]
    try:
        res = gcoords.geographic_to_geomagnetic(float(lat), float(lon))
    except Exception as exc:
        return _error('invalid_input', str(exc)), str(exc)
    if isinstance(res, dict) and res.get('ok') is False:
        return res, 'conversion failed'
    out = dict(res)
    out['station_code'] = code
    out['geo_lat'] = out.get('geomag_lat')
    out['geo_lon'] = out.get('geomag_lon')
    return out, f'geomagnetic coordinates for {code}'


def _handle_calculate_mlt(args, store):
    code = str(args.get('station_code') or '').strip().upper()
    ts = args.get('timestamp')
    if not code or not ts:
        return _error('invalid_input', 'station_code and timestamp are required'), 'missing arguments'
    coords, _ = _handle_get_station_geomagnetic_coords({'station_code': code}, store)
    if isinstance(coords, dict) and coords.get('error'):
        return coords, coords.get('message')
    res = gcoords.magnetic_local_time(float(coords['geo_lon']), str(ts))
    if isinstance(res, dict) and res.get('ok') is False:
        return res, 'MLT calculation failed'
    return res, f'MLT for {code} at {ts}: {res.get("mlt_hours")} h'


def _handle_group_stations_by_mlt(args, store):
    stations = args.get('stations') or []
    ts = args.get('timestamp')
    if not stations or not ts:
        return _error('invalid_input', 'stations and timestamp are required'), 'missing arguments'
    bin_hours = args.get('bin_hours') or 1
    try:
        bin_hours = int(bin_hours)
    except Exception:
        return _error('invalid_input', 'bin_hours must be an integer'), 'bad bin_hours'
    items = []
    for sc in stations:
        coords, _ = _handle_get_station_geomagnetic_coords({'station_code': str(sc)}, store)
        if isinstance(coords, dict) and coords.get('error'):
            continue  # a missing station must not abort the group
        mlt = gcoords.magnetic_local_time(float(coords['geo_lon']), str(ts))
        if isinstance(mlt, dict) and mlt.get('mlt_hours') is not None:
            items.append((str(sc).upper(), mlt['mlt_hours']))
    res = gcoords.group_by_mlt(items, bin_hours)
    if isinstance(res, dict) and res.get('ok') is False:
        return res, 'grouping failed'
    return res, f'grouped {len(items)} stations by MLT'


def _handle_calculate_local_time(args, store):
    code = str(args.get('station_code') or '').strip().upper()
    ts = args.get('timestamp')
    if not code or not ts:
        return _error('invalid_input', 'station_code and timestamp are required'), 'missing arguments'
    try:
        from intermagnet_loader import get_available_stations
        rec = get_available_stations().get(code)
    except Exception as exc:
        return _error('registry_error', str(exc)), 'registry error'
    if not rec or rec[2] is None:
        return (
            _error(
                'station_not_found',
                f'No usable longitude for {code} in the INTERMAGNET registry',
                hint='Check the IAGA code',
            ),
            f'station_not_found: {code}',
        )
    try:
        longitude = float(rec[2])
    except (TypeError, ValueError):
        return _error('registry_error', f'longitude for {code} is not a number'), 'registry error'
    try:
        stamp = pd.to_datetime(str(ts), utc=True, errors='raise')
    except Exception as exc:
        return _error('invalid_input', f'Cannot parse timestamp {ts!r}: {exc}'), 'bad timestamp'
    ut_hours = (stamp.hour + stamp.minute / 60.0 + stamp.second / 3600.0) % 24.0
    lt = (ut_hours + longitude / 15.0) % 24.0
    h = int(lt)
    m = int((lt - h) * 60.0 + 0.5) % 60
    if m == 60:
        h = (h + 1) % 24
        m = 0
    result = {
        'ok': True,
        'station_code': code,
        'timestamp_utc': stamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'lt_hours': float(round(lt, 4)),
        'lt_hm': f'{h:02d}:{m:02d}',
        'longitude': longitude,
        'model': 'lt_ut_plus_lon_over_15',
    }
    return result, f"local time for {code} at {result['timestamp_utc']}: {result['lt_hm']}"


def _station_registry_coords(code):
    """Return ``(lat, lon)`` for a station, or ``None`` when unavailable."""
    try:
        from intermagnet_loader import get_available_stations
        rec = get_available_stations().get(code)
    except Exception:
        return None
    if not rec or rec[1] is None or rec[2] is None:
        return None
    try:
        return float(rec[1]), float(rec[2])
    except (TypeError, ValueError):
        return None


def _handle_plot_overlay(args, store):
    stations = [str(s).strip().upper() for s in (args.get('stations') or []) if str(s).strip()]
    dates = args.get('dates') or ([args['date']] if args.get('date') else [])
    if isinstance(dates, str):
        dates = [dates]
    dates = [str(d).strip() for d in dates if str(d).strip()]
    component = str(args.get('component') or 'H').strip().upper()
    time_system = str(args.get('time_system') or 'UT').strip().upper()
    if not stations:
        return _error('invalid_input', 'stations must list at least one IAGA code'), 'missing stations'
    if not dates:
        return _error('invalid_input', 'dates must list at least one YYYY-MM-DD date'), 'missing dates'
    if time_system not in ('UT', 'LT', 'MLT'):
        return _error('invalid_input', "time_system must be 'UT', 'LT' or 'MLT'"), 'bad time_system'

    # A single shared day keeps every trace on the same clock; the schema says so.
    date = dates[0]
    offsets_in = args.get('offsets') or {}
    if not isinstance(offsets_in, dict):
        return _error('invalid_input', 'offsets must be an object of {station: nT}'), 'bad offsets'

    # Data first: overlay must survive being called in the first round, before
    # any data was fetched. The agent often jumps straight to the chart and the
    # useful behaviour is to fetch the missing days on its behalf (mirroring the
    # auto-derive in the plotting path) rather than spend a round teaching it
    # about fetch_observatory_data.
    notes: list[str] = []
    for code in stations:
        if store.resolve(f"raw:{code.lower()}:{date}")[0] is None:
            try:
                payload, note = _handle_fetch(
                    {"station_code": code, "start_date": date, "end_date": date}, store
                )
            except Exception as exc:
                payload = _error('auto_fetch_failed', str(exc))
            if is_error(payload):
                return (
                    _error(
                        'auto_fetch_failed',
                        f"Could not fetch {code} for {date}: {payload.get('error')} "
                        f"({payload.get('message', '')})".strip(),
                        requested={'stations': stations, 'date': date},
                        hint=(
                            "plot_overlay tried to load the day itself; the download "
                            "failed, so retry fetch_observatory_data manually."
                        ),
                    ),
                    f"plot_overlay failed: auto-fetch of {code}/{date} failed",
                )
            notes.append(f"auto-fetched {code} for {date}")

    # Derive the derived components so the sibling slots exist for later tools.
    if component in analyzer.DERIVED:
        for code in stations:
            slot, note = _auto_derive(
                store, f"derived:{code.lower()}:{date}", [component]
            )
            if slot is not None:
                notes.append(note)

    frames: dict[str, Any] = {}
    time_shifts: dict[str, float] = {}
    offsets: dict[str, float] = {}
    missing: list[str] = []
    for code in stations:
        handle = f"raw:{code.lower()}:{date}"
        frame, err = store.resolve(handle)
        if frame is None:
            missing.append(handle)
            continue
        frames[code] = frame
        try:
            offsets[code] = float(offsets_in.get(code, offsets_in.get(code.lower(), 0.0)))
        except (TypeError, ValueError):
            return _error('invalid_input', f'offset for {code} is not a number'), 'bad offset'
        time_shifts[code] = _overlay_time_shift(code, frame, time_system)

    if not frames:
        return (
            _error(
                'unknown_frame',
                f"No fetched data for {date}; nothing to overlay.",
                requested={'stations': stations, 'date': date},
                available=store.names(),
                hint=(
                    "Fetch each station first, e.g. raw:irt:%s, then call plot_overlay "
                    "with the same date." % date
                ),
            ),
            'plot_overlay failed: no frames resolved',
        )
    if missing:
        return (
            _error(
                'unknown_frame',
                f"Missing data for: {', '.join(missing)}.",
                requested={'stations': stations, 'date': date},
                available=store.names(),
                hint=(
                    "Fetch the missing stations for %s first (fetch_observatory_data "
                    "or fetch_many), then retry plot_overlay." % date
                ),
            ),
            'plot_overlay failed: some frames unresolved',
        )

    # North on top: sort by geomagnetic latitude, descending.
    order = _sort_stations_north_first(list(frames))
    ordered = {code: frames[code] for code in order}

    result = plotter.plot_overlay(
        ordered,
        component=component,
        offsets=offsets,
        time_system=time_system,
        time_shifts=time_shifts,
        title=args.get('title') or f"Overlay {', '.join(order)} {date}",
        filename=args.get('filename'),
    )
    if plotter.is_error(result):
        return result, f"plot_overlay failed: {result.get('error')}"
    filed = _file_into_project(
        store, result, [f"raw:{code.lower()}:{date}" for code in order], "overlay"
    )
    payload = {
        'ok': True,
        'plot_path': result,
        'path': result,
        'stations': order,
        'component': component,
        'date': date,
        'time_system': time_system,
        'offsets_applied': offsets,
    }
    if notes:
        payload['auto'] = notes
    if filed:
        payload['filed_into_project'] = filed
    summary = notes + [f"overlay of {', '.join(order)} saved to {result}"]
    return payload, " ".join(summary)


def _sort_stations_north_first(codes: list[str]) -> list[str]:
    """Order stations by geomagnetic latitude, northern first."""
    scored: list[tuple[float, str]] = []
    for index, code in enumerate(codes):
        coords = _station_registry_coords(code)
        lat_m = None
        if coords is not None:
            conv = gcoords.geographic_to_geomagnetic(coords[0], coords[1])
            if isinstance(conv, dict) and conv.get('ok'):
                lat_m = conv.get('geomag_lat')
        # Stations without coordinates keep their input order, below the located
        # ones, so the chart never silently reorders what the user asked for.
        scored.append((lat_m if lat_m is not None else -1000.0 - index, code))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [code for _, code in scored]


def _overlay_time_shift(code: str, frame: pd.DataFrame, time_system: str) -> float:
    """Hours to add to a station's time-of-day for the chosen clock."""
    if time_system == 'UT':
        return 0.0
    coords = _station_registry_coords(code)
    if coords is None:
        return 0.0
    lat, lon = coords
    if time_system == 'LT':
        return lon / 15.0
    # MLT: advance at the same rate as UT, so a single offset taken at the first
    # sample slides the whole trace onto the magnetic clock.
    conv = gcoords.geographic_to_geomagnetic(lat, lon)
    if not (isinstance(conv, dict) and conv.get('ok') and conv.get('geomag_lon') is not None):
        return 0.0
    try:
        first = pd.to_datetime(frame['timestamp'], errors='coerce').dropna().iloc[0]
    except (KeyError, IndexError):
        return 0.0
    mlt = gcoords.magnetic_local_time(conv['geomag_lon'], first.to_pydatetime())
    if not (isinstance(mlt, dict) and mlt.get('mlt_hours') is not None):
        return 0.0
    ut_hours = (first.hour + first.minute / 60.0 + first.second / 3600.0) % 24.0
    return float(mlt['mlt_hours']) - ut_hours


HANDLERS: dict[str, Handler] = {
    "fetch_observatory_data": _handle_fetch,
    "create_project": _handle_create_project,
    "list_projects": _handle_list_projects,
    "fetch_many": _handle_fetch_many,
    "export_project": _handle_export_project,
    "create_task_matrix": _handle_create_task_matrix,
    "process_matrix_batch": _handle_process_matrix_batch,
    "calculate_derived_components": _handle_derive,
    "get_statistics": _handle_stats,
    "detect_anomalies": _handle_anomalies,
    "calculate_derived_math": _handle_derived_math,
    "calculate_baseline": _handle_baseline,
    "evaluate_custom_formula": _handle_custom_formula,
    "plot_components": _handle_plot_components,
    "plot_comparison": _handle_plot_comparison,
    "get_station_geomagnetic_coords": _handle_get_station_geomagnetic_coords,
    "calculate_mlt": _handle_calculate_mlt,
    "group_stations_by_mlt": _handle_group_stations_by_mlt,
    "calculate_local_time": _handle_calculate_local_time,
    "plot_overlay": _handle_plot_overlay,
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
    system_prompt: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    store: FrameStore | None = None,
) -> dict[str, Any]:
    """Answer ``user_query`` by letting the model call the Stage 1/2 tools.

    ``system_prompt`` / ``tools`` override the defaults for planner subtasks:
    a narrow task sees a short prompt and only the schemas that can serve it.
    Defaults to :data:`SYSTEM_PROMPT` and :data:`TOOL_SCHEMAS`.

    ``store`` lets the external brain share one :class:`FrameStore` across many
    subtasks; when None a fresh store is created for this call.

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

    prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    schemas = tools if tools is not None else TOOL_SCHEMAS
    store = store or FrameStore()
    plots: list[str] = []
    log: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query},
    ]

    spent = 0
    stop_reason = "no_tool_calls"
    format_retries = 0

    for round_no in range(1, max_rounds + 1):
        try:
            reply = brain.chat(messages, schemas)
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
            note = ""

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
            # Handlers describe what they actually did in their note -- which
            # handle they derived, what they saved. That note used to be returned
            # and dropped on the floor, so work the agent did silently was
            # invisible in the log; an auto-derived slot in particular is
            # something the user needs to see to trust the chart.
            if note:
                entry["note"] = note
            # Surface the diagnostic fields, not just the code: without these a
            # malformed call in Colab reports "malformed_tool_call" and nothing
            # else, which is what sent the previous two fixes down blind paths.
            for key in ("message", "received", "reason", "hint"):
                if isinstance(payload, dict) and payload.get(key) is not None:
                    entry[key] = payload[key]
            # The project a project tool acted on. Without this the log says only
            # that a ZIP exists, and an outside checker has no way to find the tree
            # the run claims to have written.
            if isinstance(payload, dict) and isinstance(payload.get("project"), str):
                entry["project"] = payload["project"]
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


def run_agent_with_planner(
    query: str,
    brain: Any = None,
    store: FrameStore | None = None,
    max_rounds: int = 100,
    verbose: bool = False,
    use_llm_planner: bool = True,
) -> dict[str, Any]:
    """Run a possibly-large request through the external planner.

    One :class:`~agent_state.TaskStep` per unit ("26 станций x 10 событий"
    expands into hundreds of steps) and each step is executed by its own
    ``run_agent`` call. With ``use_llm_planner=True`` (default) a real request is
    split by the model itself (:class:`~agent_planner.LLMPlanner`), which beats
    the regex planner on free-form phrasing; a parse failure falls back to the
    deterministic :class:`~agent_planner.Planner`. When ``brain`` is absent the
    model cannot plan, so the deterministic planner is used.

    CRITICAL: every subtask shares the same :class:`FrameStore` (created here if
    not passed). Frames fetched by step 1 are visible to step 260; without this
    a batch job would re-download every day for every chart. The shared store is
    returned so callers can inspect or reuse it.

    Generator-Executor for huge jobs: when the plan is a single
    ``create_task_matrix`` step, or when it degenerated into more than
    :data:`MATRIX_THRESHOLD` per-unit tasks, the matrix path is taken instead of
    hundreds of model round-trips -- :func:`_run_matrix_job` writes the matrix
    once, grinds through it via :func:`process_matrix_batch` (handlers called
    directly, no LLM for each cell) and exports the project.
    """
    if use_llm_planner and brain is not None and not is_error(brain) and hasattr(brain, "chat"):
        from agent_planner import LLMPlanner

        planner = LLMPlanner(brain)
        state = planner.plan(query, verbose=verbose)
    else:
        from agent_planner import Planner

        state = Planner().plan(query)
    shared = store if store is not None else FrameStore()

    if verbose:
        print(f"📋 План: {len(state.tasks)} подзадач")
        for task in state.tasks[:10]:
            print(f"   - {task.id}: {task.description}")
        if len(state.tasks) > 10:
            print(f"   ... и ещё {len(state.tasks) - 10} подзадач")

    # A single "create_task_matrix" task means the model already switched to the
    # Generator-Executor pattern: run the whole matrix job (create -> batch loop
    # -> export) without ever making the model enumerate the cells.
    if len(state.tasks) == 1 and state.tasks[0].tool_name == "create_task_matrix":
        args = state.tasks[0].tool_args or {}
        if verbose:
            print("🏭 План — одна задача create_task_matrix: запускаю матричный цикл...")
        return _run_matrix_job(
            _normalise_matrix_events(args.get("events")),
            _normalise_matrix_stations(args.get("stations")),
            _normalise_matrix_actions(args.get("actions")) or _DEFAULT_MATRIX_ACTIONS,
            str(args.get("project_name") or "").strip() or "batch_matrix",
            shared,
            verbose=verbose,
        )

    # A plan that degenerated into > MATRIX_THRESHOLD per-unit tasks (e.g. a
    # deterministic "26 станций" expansion, or an LLM plan that listed them all
    # against instructions) is folded back into a matrix: the model is neither
    # asked to enumerate the combo list nor to round-trip for every cell.
    if len(state.tasks) > MATRIX_THRESHOLD:
        factored = _factor_plan_for_matrix(state.tasks)
        if factored is not None:
            events, stations, actions, project_name = factored
            if verbose:
                print(
                    f"🔄 План из {len(state.tasks)} подзадач свернут в матрицу "
                    f"({len(events)}x{len(stations)}, действия: {actions})."
                )
            return _run_matrix_job(events, stations, actions, project_name, shared, verbose=verbose)

    # A single "run_agent" task means the planner did not recognise the request:
    # fall back to the ordinary single-question loop.
    if len(state.tasks) == 1 and state.tasks[0].tool_name == "run_agent":
        result = run_agent(
            query,
            brain=brain,
            max_rounds=max_rounds,
            verbose=verbose,
            store=shared,
        )
        result = dict(result) if isinstance(result, dict) else {"ok": False, "text": str(result)}
        result["store"] = shared
        return result

    results: list[dict[str, Any]] = []
    for task in state.tasks:
        if verbose:
            print(f"\n🔧 Выполняю: {task.id} — {task.description}")
        schemas = select_tools_for_task(task.tool_name)
        sub_prompt = build_dynamic_prompt(
            BASE_PROMPT_COMPACT, schemas, state.get_context_for_prompt()
        )
        try:
            result = run_agent(
                task.description,
                brain=brain,
                max_rounds=max_rounds,
                verbose=verbose,
                system_prompt=sub_prompt,
                tools=schemas,
                store=shared,
            )
            ok = bool(result.get("ok")) if isinstance(result, dict) else False
            state.mark_done(task.id, result if ok else None)
            if not ok:
                state.mark_failed(task.id, str(result.get("text", result)))
            results.append(
                {"task_id": task.id, "ok": ok, "result": result if ok else None,
                 "error": None if ok else str(result.get("text", result))}
            )
        except Exception as exc:
            state.mark_failed(task.id, str(exc))
            results.append({"task_id": task.id, "ok": False, "error": str(exc)})

    done = sum(1 for r in results if r["ok"])
    failed = len(results) - done
    return {
        "ok": failed == 0,
        "text": f"Выполнено {done} из {len(results)} подзадач. Ошибок: {failed}.",
        "results": results,
        "state": state.to_dict(),
        "summary": state.get_progress_summary(),
        "store": shared,
    }


def _unique(values: list[str]) -> list[str]:
    """Order-preserving de-duplication, for counting distinct things."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _attempted_plot_name(entry: dict[str, Any]) -> str:
    """The chart file one plot call was trying to produce.

    A call that failed has no ``plot`` path, so the filename it asked for is read
    out of its arguments. Counting attempts needs that name: three retries of one
    chart are one requested chart, and treating them as three is how a run ends up
    claiming "3 of 6" for three finished files.
    """
    path = entry.get("plot")
    if isinstance(path, str) and path:
        return Path(path).name
    arguments = entry.get("arguments")
    if isinstance(arguments, dict):
        requested = arguments.get("filename")
        if isinstance(requested, str) and requested.strip():
            return Path(requested.strip()).name
    return ""


def _with_plot_tally(text: str, log: list[dict[str, Any]]) -> str:
    """State plainly how many charts exist, and list the files.

    When the budget cuts a run short the model's own sentence is optimistic --
    it says "the charts are ready" while two of them were never drawn -- and the
    user is left hunting for a file that does not exist. The count is computed
    from the executed plot calls, so it cannot disagree with ``result["plots"]``.

    Both sides of "N из M" count *distinct charts*, not tool calls. A model that
    tries the same chart several times, or gets one chart right on the third
    attempt, asked for one file and got one file; counting the attempts turned a
    finished three-chart run into "3 из 6", which reads as three failures.
    """
    built = _unique(
        [
            Path(str(entry["plot"])).name
            for entry in log
            if entry.get("ok") and isinstance(entry.get("plot"), str) and entry["plot"]
        ]
    )
    requested = _unique(
        [_attempted_plot_name(entry) for entry in log if entry.get("tool") in PLOT_TOOLS]
    )
    if not built and not requested:
        return text

    missing = [name for name in requested if name not in set(built)]
    line = f"Построено графиков: {len(built)}"
    if len(requested) > len(built):
        line += f" из {len(requested)} запрошенных"
    line += "."
    if built:
        line += "\nФайлы: " + ", ".join(built)
    if missing:
        # The old wording blamed the tool budget for every failure. A chart can
        # also be missing because a handler refused it, and telling the user the
        # budget ran out sends them to ask for charts twice.
        cause = (
            "лимит вызовов инструментов исчерпан"
            if any(e.get("error") == "tool_budget_exhausted" for e in log)
            else "вызов не удался"
        )
        line += (
            f"\nНе построено: {len(missing)} ({cause}): "
            + ", ".join(missing)
            + ". Попросите оставшиеся графики отдельным сообщением."
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
