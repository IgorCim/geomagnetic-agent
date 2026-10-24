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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd

import geomag_analyzer as analyzer
import geomag_math as geomath
import geomag_plotter as plotter
import intermagnet_loader as loader
import space_weather_loader as space_weather
import noaa_swpc_loader as noaa
import supermag_loader as supermag
import iswa_loader as iswa
import project_store as projects
import geomag_coords as gcoords
import physics_engine as pengine

# Global persistent workspace across run_agent() invocations
_WORKSPACE = projects.Workspace()

#: Bumped whenever the tool-call parser changes shape. ``colab_run.ipynb``
#: prints this next to ``git rev-parse HEAD``; a notebook that shows an older
#: value is running a cached copy of this module, not this file.
PARSER_VERSION = "bulletproof-3"

__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_SCHEMAS",
    "Scratchpad",
    "FrameStore",
    "LlamaBrain",
    "ScriptedBrain",
    "download_model",
    "load_brain",
    "parse_tool_calls",
    "code_revision",
    "run_agent",
    "run_agent_with_planner",
    "select_tools_for_task",
    "select_tools_for_query",
    "explain_tool_selection",
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

#: Context window handed to llama.cpp. 8192 left ~200 free tokens once the full
#: 21-schema list was appended to the system prompt, so the model produced an
#: empty answer ("Нет текстового ответа"). Doubling it to 16384 keeps a wide
#: safety margin while the schema diet (dynamic selection + trimmed
#: descriptions) cuts what we actually send. ``load_brain(n_ctx=...)`` overrides.
N_CTX = 16384
N_CTX_FALLBACK = 12288
N_BATCH = 512
TEMPERATURE = 0.1          # a scientist wants determinism, not creativity
MAX_TOKENS = 1200

# A three-day request costs three fetches, three derive calls and three plots
# before a single sentence of the answer, and a two-day comparison that first
# tries a bad handle costs a failed call plus a retry. A multi-step 14B-model
# task chains more of these before it answers, so the budget is raised to 28
# calls / 25 rounds -- still bounded, so a runaway loop still terminates.
MAX_TOOL_CALLS = 28        # tool executions per user query
MAX_ROUNDS = 25            # model round-trips per user query

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
    "Хэндлы пиши строчными (в нижнем регистре): "
    "frame_handle='raw:IRT:2024-09-10' передавай как 'raw:irt:2024-09-10'. "
    "Два дня — два хэндла."
)

BASE_PROMPT_COMPACT = (
    "Ты — профессиональный геофизический ассистент. Не выдумывай цифры — "
    "вызывай инструменты.\n\n"
    "БАЗОВЫЕ ПРАВИЛА:\n"
    "ПРАВИЛО 0 (ДЕКОМПОЗИЦИЯ): ЕСЛИ запрос содержит несколько разных задач или "
    "источников — СНАЧАЛА напиши нумерованный ПЛАН (1, 2, 3...), затем выполняй "
    "инструменты строго по одному пункту. НИКОГДА не объединяй разные метрики или "
    "источники в один вызов.\n"
    "1. Отвечай по-русски, кратко. Числа — только из инструмента.\n"
    "2. Данные бери fetch_observatory_data, код IAGA. "
    f"Известны: {STATION_HINT}.\n"
    "3. Хэндл — только из frame_handle. "
    f"{HANDLE_CASE_RULE}\n"
    "4. Физику — ТОЛЬКО через apply_physics_formula metric. calculate_dst_proxy "
    "НЕ существует. Для солнечных метрик передавай solar_wind_handle с "
    "хэндлом фрейма. EEJ — calculate_eej_index.\n"
    "5. Метрики calculate_derived_math: 'delta', 'dH_dt', 'anomaly'. "
    "Метрики 'range' НЕ СУЩЕСТВУЕТ.\n"
    "6. Ошибку перескажи, предложи шаг; файлы; "
    "единицы 'nT'.\n\n"
    "ГРАФИКИ (ЕСЛИ → ТО):\n"
    "4a. ОДНА СТАНЦИЯ: ЕСЛИ нужен график станции → fetch_observatory_data, "
    "затем plot_components/plot_comparison/plot_overlay.\n"
    "4b. ЗЕМЛЯ vs КОСМОС: ЕСЛИ сравниваешь станцию с ветром → "
    "fetch_observatory_data + fetch_solar_wind, затем "
    "plot_space_weather_comparison(ground_component='H', "
    "solar_wind_components=['Bz','V']). ground_df_descriptor — хэндл СТАНЦИИ "
    "('raw:irt:2024-09-12'), НЕ idx:supermag/idx:kyoto.\n"
    "4c. ИНДЕКСЫ: ЕСЛИ работаешь с индексами → fetch_supermag_data 'indices' "
    "(SME/SML/SMU) или fetch_geomagnetic_indices, затем plot_components.\n"
    "4d. ПЕРЕД ЛЮБЫМ из них — fetch_observatory_data, иначе unknown_frame.\n"
    "4f. idx:... как ground_df_descriptor НЕЛЬЗЯ — только raw:...\n\n"
    "КОСМИЧЕСКАЯ ПОГОДА:\n"
    "7. Бури, алерты — get_noaa_space_weather_context "
    "('alerts'|'forecast'|'solar_events').\n"
    "4e. Dst/Kp — get_iswa_forecast; карта get_iswa_perturbation_map "
    "ДЕКОМИССИРОВАНА → fetch_supermag_data('indices').\n\n"
    "ИНСТРУМЕНТЫ: Данные — fetch_observatory_data, fetch_many, list_projects, "
    "create_project, export_project, fetch_solar_wind, "
    "fetch_geomagnetic_indices. Пакетно — create_task_matrix, "
    "process_matrix_batch. Математика — calculate_derived_components, "
    "get_statistics, detect_anomalies, calculate_baseline (baseline_value), "
    "evaluate_custom_formula (не исполняется как код), apply_physics_formula. "
    "Магнетизм — get_station_geomagnetic_coords, calculate_mlt, "
    "calculate_local_time, group_stations_by_mlt.\n\n"
    "ФОРМАТ ОТВЕТА:\n"
    "8. Ты НЕ API-сервер: не отвечай {'content':..., 'tool_calls':...}. "
    "Вызов — теги:\n"
    '      <tool_call>{"name": "fetch_observatory_data", "arguments": '
    '{"station_code": "IRT", "start_date": "2024-09-10"}}</tool_call>\n'
    "Данных достаточно — текст без тегов.\n\n"
    "АЛГОРИТМ ДЛЯ МАССОВЫХ ЗАДАЧ:\n"
    "1) create_task_matrix; 2) ОДИН process_matrix_batch с "
    "run_until_complete=True; 3) проверь 'МАССОВАЯ ОБРАБОТКА ЗАВЕРШЕНА' "
    "(экспорт делается автоматически); 4) сводка. НЕ вызывай больше никаких "
    "инструментов. НИКОГДА не вызывай process_matrix_batch в цикле и НИКОГДА не "
    "вызывай create_task_matrix повторно."
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


def _args_signature(args: Any) -> str:
    """A stable fingerprint of a tool call's arguments.

    Two calls with the same signature are the *same* call however the model
    ordered its keys; the loop breaker in :class:`Scratchpad` uses this to tell
    "retrying a broken call" apart from "making progress with new arguments".
    """
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(args)


# --------------------------------------------------------------------------- #
# tool registry -- OpenAI function-calling schemas understood by Qwen 2.5
# --------------------------------------------------------------------------- #
_FRAME_ARG = {
    "type": "string",
    "description": (
        "A descriptor from a previous tool, copied EXACTLY from its "
        "'frame_handle': a slot like 'raw:irt:2024-09-10' or a family name "
        "'raw' / 'derived' / 'anomalies'. Never invent one; see "
        "'available_handles' in the last result."
    ),
}

#: The math tools predate ``df_descriptor``: they take ``df``. Both spellings
#: are accepted so a model that just used ``apply_physics_formula`` does not
#: bounce off a schema mismatch and loop.
_FRAME_ALIAS_NOTE = (
    " Note: This tool accepts both 'df' and 'df_descriptor' as arguments for "
    "backward compatibility."
)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_observatory_data",
            "description": (
                "Download one day of minute data for one INTERMAGNET observatory "
                "(three-letter IAGA code). Call this first, before any analysis. "
                "Returns df_descriptor 'raw:<STATION>:<date>'; reuse it verbatim."
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
                "Download one date range for SEVERAL observatories in one call. Use "
                "instead of repeating fetch_observatory_data. Stations share one "
                "download budget; each returns its own df_descriptor. A station that "
                "fails is listed under 'failed' while the rest still succeed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Three-letter IAGA codes.",
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
            "description": "List the projects currently on disk.",
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
                "Plan a large batch job as ONE matrix grouped BY STATION: one task per "
                "station carrying all its event dates. Use INSTEAD of a long list of "
                "tool calls when the job spans many stations/dates (>~15 units). Then "
                "call process_matrix_batch once with run_until_complete=True."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Event dates, YYYY-MM-DD.",
                    },
                    "stations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Three-letter IAGA codes.",
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
                "Execute the pending STATION tasks of a task matrix (read from the "
                "matrix_path create_task_matrix returned). run_until_complete=True "
                "finishes ALL pending stations in this one call, builds the project "
                "archive automatically, and returns a summary starting "
                "'МАССОВАЯ ОБРАБОТКА ЗАВЕРШЕНА'; no separate export_project call is "
                "needed. False processes only batch_size stations (debugging)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "matrix_path": {
                        "type": "string",
                        "description": "The matrix_path returned by create_task_matrix.",
                    },
                    "run_until_complete": {
                        "type": "boolean",
                        "description": (
                            "True = process every pending station to completion "
                            "inside this one call (recommended). False = process "
                            "only batch_size stations (debugging)."
                        ),
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Stations per call when run_until_complete=False. Default 5. Ignored when run_until_complete=True.",
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
                + _FRAME_ALIAS_NOTE
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "df_descriptor": {
                        "type": "string",
                        "description": "Alias of 'df': the same handle from 'frame_handle'.",
                    },
                    "components": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["H", "D", "I"]},
                        "description": "Subset to compute. Default all three.",
                    },
                },
                "required": [],
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
                + _FRAME_ALIAS_NOTE
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "df_descriptor": {
                        "type": "string",
                        "description": "Alias of 'df': the same handle from 'frame_handle'.",
                    },
                    "components": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["X", "Y", "Z", "F", "H", "D", "I"]},
                        "description": "Components to summarise.",
                    },
                    "digits": {"type": "integer", "description": "Decimal places. Default 2."},
                },
                "required": [],
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
                + _FRAME_ALIAS_NOTE
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "df_descriptor": {
                        "type": "string",
                        "description": "Alias of 'df': the same handle from 'frame_handle'.",
                    },
                    "component": {"type": "string", "enum": ["X", "Y", "Z", "F", "H", "D", "I"]},
                    "sigma_threshold": {
                        "type": "number",
                        "description": "Threshold in sigmas. Default 3.0.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_derived_math",
            "description": (
                "Measure a component over time as a scalar you may quote: 'delta' "
                "(max-min variation), 'dH_dt' (rate of change in nT/min), or 'anomaly' "
                "(value minus baseline_value from calculate_baseline). Returns a new "
                "derived df_descriptor holding the series, plus the scalar."
                + _FRAME_ALIAS_NOTE
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "df_descriptor": {
                        "type": "string",
                        "description": "Alias of 'df': the same handle from 'frame_handle'.",
                    },
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
                "required": ["metric"],
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
                + _FRAME_ALIAS_NOTE
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "df_descriptor": {
                        "type": "string",
                        "description": "Alias of 'df': the same handle from 'frame_handle'.",
                    },
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
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_custom_formula",
            "description": (
                "Compute a custom arithmetic expression over a frame's columns; returns "
                "a new plottable series. Use for a quantity the built-in tools do not "
                "cover, e.g. 'sqrt(X**2 + Y**2) + Z/10'. Columns: "
                f"{', '.join(geomath.available_columns())} plus the frame's own numeric "
                f"columns. Operators: + - * / % **. Functions: "
                f"{', '.join(sorted(geomath.DSL_FUNCTIONS))}, plus pi and e. The formula "
                "is parsed, never executed."
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
                "Plotly HTML chart. Any real column is accepted, case-insensitively: "
                "the standard X/Y/Z/F/H/D/I, derived columns such as 'dst_proxy', and "
                "the Kyoto indices 'Dst'/'Kp' (frame 'idx:kyoto:<дата>'). Returns the "
                "absolute path of the saved file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df": _FRAME_ARG,
                    "components": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["X", "Y", "Z", "F", "H", "D", "I", "dst_proxy", "Dst", "Kp"]},
                        "description": (
                            "Components/columns to plot, e.g. ['Dst','Kp'] for the "
                            "Kyoto indices frame or ['H','dst_proxy'] for a derived "
                            "observatory frame."
                        ),
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
                "axis. Use it to compare, for example, a storm day against a quiet day. "
                "ЗАПРЕЩЕНО использовать для сравнения данных солнечного ветра и "
                "магнитного поля Земли: инструмент умеет только две даты одной "
                "компоненты одного типа данных. Для Земли vs космоса используй "
                "ТОЛЬКО plot_space_weather_comparison."
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
                    "component": {
                        "type": "string",
                        "enum": ["X", "Y", "Z", "F", "H", "D", "I", "dst_proxy", "Dst", "Kp"],
                        "description": (
                            "Component to compare across the two frames; any real "
                            "column, including the Kyoto indices 'Dst'/'Kp'."
                        ),
                    },
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
            "name": "apply_physics_formula",
            "description": (
                "Apply a physics formula to a stored frame and return a derived frame "
                "with the new column. For solar-wind metrics, pass 'solar_wind_handle': "
                "the system aligns solar wind data with the ground station frame "
                "automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "df_descriptor": {
                        "type": "string",
                        "description": "Handle of the ground station or indices frame, from 'frame_handle'.",
                    },
                    "metric": {
                        "type": "string",
                        "enum": [
                            "ey_coupling", "burton_coupling", "akasofu_epsilon",
                            "peak_dhdt", "substorm_onset", "eej_index", "dst_proxy",
                            "ssc_flag", "sq_baseline", "deltas", "gic",
                        ],
                        "description": (
                            "Name of the metric to calculate. Available: ey_coupling, "
                            "burton_coupling, akasofu_epsilon, peak_dhdt, "
                            "substorm_onset, eej_index, dst_proxy, ssc_flag, "
                            "sq_baseline, deltas, gic."
                        ),
                    },
                    "solar_wind_handle": {
                        "type": "string",
                        "description": (
                            "Handle of solar wind frame. Required for: ey_coupling, "
                            "burton_coupling, akasofu_epsilon."
                        ),
                    },
                    "reference_handle": {
                        "type": "string",
                        "description": "Handle of reference station frame. Required for: eej_index.",
                    },
                    "window_seconds": {
                        "type": "integer",
                        "description": (
                            "Window size in samples for 'peak_dhdt' (default 10 = 10 "
                            "minutes on a 1-minute grid)."
                        ),
                    },
                },
                "required": ["df_descriptor", "metric"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_eej_index",
            "description": (
                "Use for the equatorial electrojet, EEJ, or the H-difference "
                "between an equatorial and an off-equatorial station. Downloads "
                "BOTH stations for the date range, aligns them and returns "
                "H_equatorial - H_reference as a derived frame. Do NOT use "
                "apply_physics_formula for EEJ (one frame only). REQUIRED: "
                "'equatorial_station', 'reference_station' (IAGA codes) and "
                "'start_date'. CRITICAL FALLBACK: definitive data for equatorial "
                "stations (like AAE) often lags by months. If you get a 'no_data' "
                "error for the current year, YOU MUST AUTOMATICALLY RETRY the tool "
                "call with data_type='auto' OR a historical year (e.g. "
                "'2023-09-12'). Do not give up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equatorial_station": {
                        "type": "string",
                        "description": "Three-letter IAGA code on the dip equator, e.g. IRT.",
                    },
                    "reference_station": {
                        "type": "string",
                        "description": (
                            "Three-letter IAGA code 6-8 degrees away in latitude, "
                            "e.g. ABG."
                        ),
                    },
                    "start_date": {"type": "string", "description": "Start date, YYYY-MM-DD."},
                    "end_date": {
                        "type": "string",
                        "description": "End date, YYYY-MM-DD (inclusive). Defaults to start_date.",
                    },
                },
                "required": ["equatorial_station", "reference_station", "start_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_overlay",
            "description": (
                "Draw ONE component from SEVERAL stations on ONE chart (per-station nT "
                "offset; x axis UT/LT/MLT; north-first; auto-fetches missing days). "
                "REQUIRED: 'stations' = LIST of IAGA codes, e.g. ['IRT'] (NOT a frame "
                "handle); 'dates' = list of YYYY-MM-DD, e.g. ['2024-09-12']; "
                "'component' = one string, e.g. 'H'. 'df1'/'df2' DO NOT EXIST here. "
                "Correct call: {\"stations\": [\"IRT\"], \"dates\": [\"2024-09-12\"], "
                "\"component\": \"H\"}. Two days of one station → plot_comparison. "
                "Earth-vs-solar-wind → ONLY plot_space_weather_comparison."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "REQUIRED. List of IAGA station codes to overlay, "
                            "e.g. ['IRT'] or ['IRT','API','BSL']. A list of codes, "
                            "never a df1/df2 frame handle."
                        )
                    },
                    "dates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dates as YYYY-MM-DD; one date shared by all stations, e.g. ['2024-09-12']."
                    },
                    "component": {
                        "type": "string",
                        "enum": ["X", "Y", "Z", "F", "H", "D", "I"],
                        "description": "Single component to overlay, e.g. 'H'. Default 'H'."
                    },
                    "components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Alias for 'component': a list, e.g. ['H']. First entry "
                            "is drawn; must be a frame column (X/Y/Z/F/H/D/I)."
                        )
                    },
                    "offsets": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "nT shift per station, e.g. {'IRT': 0, 'API': 150}."
                    },
                    "time_system": {"type": "string", "enum": ["UT", "LT", "MLT"], "description": "X-axis time system, default UT."},
                    "title": {"type": "string", "description": "Chart title."},
                    "filename": {"type": "string", "description": "Output file name, .html."}
                },
                "required": ["stations", "dates"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_space_weather_comparison",
            "description": (
                "КРИТИЧЕСКИ ВАЖНО: используй для сравнения Земли и космоса -- "
                "компонент Земли (H, dst_proxy) с солнечным ветром (Bz, V, By, Np). "
                "Dual-axis: GROUND (left) vs SOLAR-WIND (right); L1 aligned to the "
                "observatory clock. НЕ заменяй plot_overlay/plot_comparison -- они не "
                "смешивают Землю и ветер. Перед вызовом: (1) fetch_observatory_data "
                "(raw:<station>:<дата>); (2) apply_physics_formula(metric='dst_proxy'); "
                "(3) fetch_solar_wind (sw:omni:<дата>). ground_component: H, dst_proxy "
                "или dH_dt. НЕ 'Dst' — это глобальный индекс Kyoto, не компонент.\n"
                "EXAMPLE: {\"ground_df_descriptor\": \"derived:irt:2024-09-12\", "
                "\"solar_wind_df_descriptor\": \"sw:omni:2024-09-12\", "
                "\"ground_component\": \"H\", \"solar_wind_components\": [\"Bz\",\"V\"]}\n"
                "WARNING: NOT 'frame_handle', 'df1', 'df2', 'stations' -- use EXACTLY "
                "'ground_df_descriptor' and 'solar_wind_df_descriptor'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ground_df_descriptor": {
                        **_FRAME_ARG,
                        "description": (
                            "Ground handle, e.g. 'raw:irt:2024-09-12' or a "
                            "'derived:...' frame with 'dst_proxy'."
                        ),
                    },
                    "solar_wind_df_descriptor": {
                        **_FRAME_ARG,
                        "description": "Handle from fetch_solar_wind, e.g. 'sw:omni:2024-09-12'.",
                    },
                    "ground_component": {
                        "type": "string",
                        "description": "LEFT axis: 'H', 'dst_proxy' or 'dH_dt'. NOT 'Dst'.",
                    },
                    "solar_wind_components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "RIGHT axis, e.g. ['Bz','V'] (also Bx, By, Np).",
                    },
                    "title": {"type": "string", "description": "Chart title."},
                    "filename": {"type": "string", "description": "Output file name, .html."}
                },
                "required": [
                    "ground_df_descriptor",
                    "solar_wind_df_descriptor",
                    "ground_component",
                    "solar_wind_components"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_solar_wind",
            "description": (
                "Download L1 solar-wind data (NASA/OMNI, 1-minute: speed V, density Np, "
                "temperature T, IMF Bx/By/Bz in GSM) for a date range. USE this to find "
                "the CAUSE of a geomagnetic storm (southward Bz + fast V). Returns "
                "frame_handle 'sw:omni:<START_DATE>' to pass to apply_physics_formula "
                "(metric='ey_coupling'). DO NOT USE for observatory data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date, YYYY-MM-DD (UTC)."},
                    "end_date": {"type": "string", "description": "End date, YYYY-MM-DD (UTC), inclusive. Defaults to start_date."}
                },
                "required": ["start_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_geomagnetic_indices",
            "description": (
                "Download geomagnetic activity indices (Kyoto WDC: hourly Dst, 3-hourly "
                "Kp) for a date range -- the storm SEVERITY record. USE this to VALIDATE "
                "a local proxy: compare the station's 'dst_proxy'/'dH_dt' against the "
                "official Dst and Kp. USE alongside fetch_solar_wind to describe a storm. "
                "Returns frame_handle 'idx:kyoto:<START_DATE>' whose plottable columns "
                "are 'Dst' (nT) and 'Kp' (0-9); draw them with plot_components. DO NOT "
                "USE for observatory magnetograms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date, YYYY-MM-DD (UTC)."},
                    "end_date": {"type": "string", "description": "End date, YYYY-MM-DD (UTC), inclusive. Defaults to start_date."}
                },
                "required": ["start_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_noaa_space_weather_context",
            "description": (
                "Official NOAA SWPC layer: current geomagnetic alerts (G1-G5), "
                "the 3-day planetary Kp outlook, the GOES X-ray flare catalogue. "
                "Correlate these official figures with local observatory data and "
                "Kyoto indices. NOT for raw solar-wind measurements (V, Bz, ...): "
                "that is fetch_solar_wind. query_type picks one of 'alerts' | "
                "'forecast' | 'solar_events'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["alerts", "forecast", "solar_events"],
                        "description": (
                            "'alerts' = active geomagnetic alerts/warnings (G1-G5); "
                            "'forecast' = 3-day Kp/G-scale forecast; "
                            "'solar_events' = X-ray flare catalogue for a date range."
                        )
                    },
                    "start_date": {"type": "string", "description": "Start date, YYYY-MM-DD (UTC). Used by 'solar_events'."},
                    "end_date": {"type": "string", "description": "End date, YYYY-MM-DD (UTC), inclusive. Defaults to start_date."}
                },
                "required": ["query_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_supermag_data",
            "description": (
                "Скачивает данные с базы SuperMAG. Используйте это для получения "
                "возмущенных компонент (dX, dY, dZ) с уже удаленной базовой линией, "
                "или для получения глобальных индексов суббурь (SME, SML, SMU), "
                "указав data_type='indices'. station_code НЕОБЯЗАТЕЛЕН: при "
                "data_type='indices' его передавать не нужно."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "station_code": {
                        "type": "string",
                        "description": (
                            "НЕОБЯЗАТЕЛЕН. Нужен только при data_type='sites' "
                            "(напр. 'IRT'); при 'indices' игнорируется."
                        ),
                    },
                    "start_date": {"type": "string", "description": "Начальная дата, YYYY-MM-DD (UTC)."},
                    "end_date": {
                        "type": "string",
                        "description": "Конечная дата, YYYY-MM-DD (UTC), включительно. Максимум 32 дня.",
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["sites", "indices"],
                        "description": (
                            "'sites' = минутные данные станции dX/dY/dZ (базовая линия уже "
                            "удалена); 'indices' = глобальные SME/SML/SMU, станция не нужна."
                        ),
                    },
                },
                "required": ["start_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_iswa_perturbation_map",
            "description": (
                "Карта наземных возмущений ISWA (WMS ground_mag_perturbation) — "
                "ДЕКОМИССИРОВАНА: NASA CCMC удалила публичный WMS-слой при миграции, "
                "инструмент вернёт service_decommissioned. Для глобальной картины "
                "суббурь используй fetch_supermag_data(data_type='indices') (SME/SML/SMU)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "Момент, ISO 8601 UTC, напр. '2024-09-12T12:00:00Z'.",
                    },
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[min_lon, min_lat, max_lon, max_lat]; по умолчанию весь мир.",
                    },
                    "width": {"type": "integer", "description": "Ширина PNG, px (по умолчанию 1024)."},
                    "height": {"type": "integer", "description": "Высота PNG, px (по умолчанию 512)."},
                },
                "required": ["timestamp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_iswa_forecast",
            "description": (
                "Прогноз Dst/Kp из ISWA (HAPI): Kp на 3-часовой сетке — "
                "наблюдённый/оценённый/предсказанный (поле kp_kind); Dst — nowcast "
                "Kyoto, вперёд не прогнозируется. Аргументов не требует."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOLS_BY_NAME = {t["function"]["name"] for t in TOOL_SCHEMAS}
PLOT_TOOLS = {
    "plot_components",
    "plot_comparison",
    "plot_overlay",
    "plot_space_weather_comparison",
}

# Generator-Executor batch jobs. A 10-events x 26-stations request is 260 units
# of work; asking the model to enumerate them in one JSON reply overflows its
# context, so anything above MATRIX_THRESHOLD units is executed through a task
# matrix instead. The matrix is grouped BY STATION (one task per station with the
# full list of its event dates), so a 260-cell job is a 26-task matrix; a single
# process_matrix_batch(run_until_complete=True) grinds it in Python -- no per-unit
# LLM round-trip. Matrix generation and batch execution must both land in the
# graph to keep the "one matrix task per station" invariant true for new players.
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
    "plot_space_weather_comparison": [
        "fetch_observatory_data",
        "fetch_solar_wind",
        "apply_physics_formula",
        "plot_space_weather_comparison",
    ],
    "apply_physics_formula": [
        "fetch_observatory_data",
        "fetch_solar_wind",
        "fetch_geomagnetic_indices",
        "apply_physics_formula",
    ],
    "fetch_solar_wind": ["fetch_solar_wind", "fetch_geomagnetic_indices", "apply_physics_formula"],
    "fetch_geomagnetic_indices": ["fetch_geomagnetic_indices", "apply_physics_formula"],
    "calculate_eej_index": ["calculate_eej_index"],
    "get_noaa_space_weather_context": ["get_noaa_space_weather_context"],
    "fetch_supermag_data": [
        "fetch_supermag_data",
        "detect_anomalies",
        "plot_components",
    ],
    "get_iswa_perturbation_map": ["get_iswa_perturbation_map"],
    "get_iswa_forecast": ["get_iswa_forecast"],
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


#: Upper bound on the *optional* schemas a single free-form ``run_agent``
#: question may see. The full 29-schema list is what filled the 8192-token window
#: to ~200 tokens of headroom; a question only ever needs a handful. A tool the
#: question makes mandatory (its metric's loader or its decisive computer) is
#: never dropped to satisfy this budget -- correctness first, then the cap.
TOOLS_PER_TASK = 6

#: The frame summary handed back to the model is capped at this many characters,
#: so a 1440-row frame cannot dump columns/statistics into the context.
FRAME_SUMMARY_MAX_CHARS = 400

#: A payload whose ``fallback_hint`` is the point (the decommissioned ISWA map)
#: is kept whole up to this cap instead of being truncated mid-instruction. The
#: hint names two distinct fixes, so cutting it in half would leave the model
#: with exactly the ambiguity that made it loop.
FALLBACK_HINT_MAX_CHARS = 800

#: Structural evidence that a request is a bulk job. The old trigger list
#: contained the bare word "станции", which in Russian is also the dative /
#: prepositional singular ("по станции IRT"). Every ordinary station question
#: was therefore read as a 260-cell matrix job, five matrix schemas filled the
#: cap, and the schema the question actually needed was evicted: fetch_solar_wind
#: for Burton, get_statistics for a median, apply_physics_formula for a substorm.
#: Bulk now needs evidence, not a preposition.
_MATRIX_TOOLS = (
    "create_task_matrix",
    "process_matrix_batch",
    "fetch_observatory_data",
    "calculate_derived_components",
    "plot_components",
)

#: Tools that bring a frame into the store.
_SOURCE_TOOLS = frozenset({
    "fetch_observatory_data",
    "fetch_solar_wind",
    "fetch_geomagnetic_indices",
    "fetch_supermag_data",
})

#: Tools that consume a frame but never create one. A question that asks for one
#: of these without naming a source is given the ground loader as a default.
_COMPUTE_TOOLS = frozenset({
    "calculate_derived_components",
    "calculate_derived_math",
    "calculate_baseline",
    "calculate_eej_index",
    "detect_anomalies",
    "evaluate_custom_formula",
    "get_statistics",
    "plot_components",
    "plot_comparison",
    "plot_overlay",
    "plot_space_weather_comparison",
})

#: Which frame(s) a physics metric reads. This is the same knowledge that
#: ``_METRIC_DATA_REQUIREMENTS`` encodes for validation; keeping one registry per
#: concern means a new metric cannot be routable-but-unfetchable. Solar-wind
#: metrics also need the ground frame: it is the primary ``df_descriptor``.
_METRIC_SOURCE_TOOLS: dict[str, tuple[str, ...]] = {
    "sq_baseline": ("fetch_observatory_data",),
    "deltas": ("fetch_observatory_data",),
    "gic": ("fetch_observatory_data",),
    "dst_proxy": ("fetch_observatory_data",),
    "ey_coupling": ("fetch_observatory_data", "fetch_solar_wind"),
    "detect_ssc": ("fetch_observatory_data",),
    "burton_coupling": ("fetch_observatory_data", "fetch_solar_wind"),
    "akasofu_epsilon": ("fetch_observatory_data", "fetch_solar_wind"),
    "peak_dhdt": ("fetch_observatory_data",),
    "substorm_onset": ("fetch_supermag_data",),
    "eej_index": ("fetch_observatory_data",),
}

#: The tool that computes each metric. EEJ is the one metric with a dedicated
#: handler; everything else goes through apply_physics_formula.
_METRIC_DECISIVE_TOOL: dict[str, str] = {"eej_index": "calculate_eej_index"}

#: Phrasings that name a metric without its identifier. Latin identifiers and
#: aliases are added automatically from the registries; only genuinely new
#: Russian/English wording belongs here.
_METRIC_QUERY_EXTRA: dict[str, tuple[str, ...]] = {
    "sq_baseline": (r"спокойн[а-я]*\s+вариац",),
    "deltas": (r"дельт",),
    "gic": (r"трансформатор", r"\bpower\s+grid\b"),
    "dst_proxy": (r"dst-?прокси", r"\bdst\s+proxy\b"),
    "ey_coupling": (r"конвекц",),
    "detect_ssc": (
        r"внезапн[а-я]*\s+начал[а-я]*\s+бури",
        r"внезапное\s+начало",
    ),
    "burton_coupling": (r"бёртон", r"бертон"),
    "akasofu_epsilon": (r"акасофу", r"эпсилон"),
    "peak_dhdt": (r"peak\s*dh\s*/?\s*dt", r"пик[а-я]*\s+dh"),
    "substorm_onset": (r"суббур",),
    "eej_index": (
        r"экваториальн[а-я]*\s+электроджет",
        r"h-?разниц",
    ),
}


def _build_metric_query_patterns() -> dict[str, tuple[str, ...]]:
    """One regex list per metric, derived from the metric registry itself.

    Called after the registry is declared, so a metric added later is picked up
    without touching the router.
    """
    patterns: dict[str, list[str]] = {
        metric: [r"\b" + re.escape(metric) + r"\b"] for metric in _PHYSICS_METRICS
    }
    for alias, metric in _PHYSICS_METRIC_ALIASES.items():
        patterns.setdefault(metric, []).append(r"\b" + re.escape(alias) + r"\b")
    for metric, extra in _METRIC_QUERY_EXTRA.items():
        patterns.setdefault(metric, []).extend(extra)
    return {metric: tuple(items) for metric, items in patterns.items()}


#: Non-metric intents, most specific first. Cyrillic stems are matched without
#: word boundaries because Russian inflects; short Latin tokens stay anchored so
#: "ey" cannot match "key". Order matters: the cap is filled top-down, so the
#: decisive product of a question must appear before its presentation.
_CAPABILITY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        r"\bnoaa\b|\bswpc\b|алерт|\balert\b|предупрежд|\bwarning\b|вспышк|"
        r"\bflare\b|шкал[аы]?\s*g[1-5]\b|\bg[1-5]\b",
        ("get_noaa_space_weather_context",),
    ),
    (
        r"\biswa\b[^.]{0,40}(прогноз|forecast|предсказ)|"
        r"(прогноз|forecast)[^.]{0,40}\biswa\b|\bkp_kind\b",
        ("get_iswa_forecast",),
    ),
    (
        r"\biswa\b|возмущен|perturbation|глобальн[а-я]*\s+карт",
        ("get_iswa_perturbation_map", "get_iswa_forecast"),
    ),
    (
        r"\bsupermag\b|\bsme\b|\bsml\b|\bsmu\b|аврор|\bauror|электроджет|"
        r"\belectrojet",
        ("fetch_supermag_data",),
    ),
    (
        r"\bkyoto\b|\bdst\b|\bkp\b|индекс",
        ("fetch_geomagnetic_indices",),
    ),
    (
        r"\bomni\b|солнечн[а-я]*\s+ветер|\bsolar\s+wind\b|\bimf\b",
        ("fetch_solar_wind",),
    ),
    (
        r"станц|обсерватор|магнитометр|наземн|"
        r"горизонтальн[а-я]*\s+составляющ|\b[xyz]\b\s*-?\s*компонент",
        ("fetch_observatory_data",),
    ),
    (
        r"статист|\bstatistic|средн|медиан|\bmedian|минимум|максимум|\bmean\b|"
        r"\bmin\b|\bmax\b|разброс|\bstd\b|квартил|перцентил",
        ("get_statistics",),
    ),
    (
        r"аномали|\banomal|выброс|базов[а-я]*\s+лини|\bbaseline\b",
        ("detect_anomalies", "calculate_baseline"),
    ),
    (
        r"размах|дельт|\bdelta\b|d[hz]_?dt|производн|скорость\s+изменен",
        ("calculate_derived_math",),
    ),
    (
        r"формул|\bformula\b|выражен|\bexpression\b",
        ("calculate_derived_math", "evaluate_custom_formula"),
    ),
    (
        r"\bmlt\b|магнитное\s+время|местное\s+время|гражданск|координ|"
        r"широт|долгот",
        (
            "get_station_geomagnetic_coords",
            "calculate_mlt",
            "calculate_local_time",
            "group_stations_by_mlt",
        ),
    ),
    (
        r"сравн|\bcomparison\b|два\s+дня",
        ("calculate_derived_components", "plot_comparison"),
    ),
    (
        r"наложи|\boverlay\b|несколько\s+станц",
        ("calculate_derived_components", "plot_overlay"),
    ),
    (
        r"(список|какие|покажи)\s+проект|\blist\s+projects?\b",
        ("list_projects",),
    ),
    (
        r"проект|\bproject\b|\bzip\b|архив",
        ("create_project", "fetch_many", "export_project"),
    ),
    (
        r"график|\bplot\b|\bchart\b|построй|нарисуй|визуализ",
        ("calculate_derived_components", "plot_components"),
    ),
)


def _ordered_unique(names: Sequence[str]) -> list[str]:
    """De-duplicate while preserving priority order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name in TOOLS_BY_NAME and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _bulk_units(text: str) -> int:
    """Estimate (station x date) work units from an explicit count in the query."""
    stations = [int(n) for n in re.findall(r"(\d{1,3})\s*(?:станц|обсерватор)", text)]
    dates = [int(n) for n in re.findall(r"(\d{1,3})\s*(?:дат|дней|суток|событи)", text)]
    return (max(stations) if stations else 1) * (max(dates) if dates else 1)


def _is_bulk_query(text: str) -> bool:
    """True only when the request itself asks for matrix-scale work.

    ``MATRIX_THRESHOLD`` is the same cut-off the matrix generator uses, so the
    router cannot send a four-cell project down the 260-cell path.
    """
    if re.search(r"\bматриц|\bmatrix\b", text):
        return True
    if _bulk_units(text) >= MATRIX_THRESHOLD:
        return True
    if re.search(r"\b(все|всех|несколько|много)\s+(станц|обсерватор)", text):
        return True
    if re.search(r"\bbatch\b|\bпакет", text) and not re.search(
        r"проект|\bproject\b|архив|\bzip\b", text
    ):
        return True
    return False


def _selection_plan(
    query: str, cap: int = TOOLS_PER_TASK
) -> tuple[list[str], dict[str, Any]]:
    """Decide which tool names a free-form question gets, and why.

    Two sets, not one list: ``mandatory`` is derived from the physics-metric
    registry and is never truncated -- a decisive tool that loses a race with a
    presentation tool is exactly the bug this replaces. ``optional`` fills the
    remaining room up to *cap*.
    """
    text = str(query or "").lower()

    metrics = [
        metric
        for metric, patterns in _METRIC_QUERY_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    ]

    mandatory: list[str] = []
    for metric in metrics:
        mandatory.extend(_METRIC_SOURCE_TOOLS.get(metric, ()))
        mandatory.append(_METRIC_DECISIVE_TOOL.get(metric, "apply_physics_formula"))

    optional: list[str] = []
    for pattern, tools in _CAPABILITY_RULES:
        if re.search(pattern, text):
            optional.extend(tools)

    bulk = _is_bulk_query(text)
    if bulk:
        mandatory = list(_MATRIX_TOOLS) + mandatory

    mandatory_unique = _ordered_unique(mandatory)
    optional_unique = [
        name for name in _ordered_unique(optional) if name not in mandatory_unique
    ]

    fallback = not mandatory_unique and not optional_unique
    if fallback:
        mandatory_unique = [
            "fetch_observatory_data",
            "calculate_derived_components",
            "plot_components",
            "get_statistics",
        ]
    elif not (set(mandatory_unique) | set(optional_unique)) & _SOURCE_TOOLS:
        if set(optional_unique) & _COMPUTE_TOOLS:
            optional_unique.insert(0, "fetch_observatory_data")

    room = max(cap, len(mandatory_unique)) - len(mandatory_unique)
    selected = mandatory_unique + optional_unique[:room]

    info: dict[str, Any] = {
        "metrics": metrics,
        "mandatory": mandatory_unique,
        "optional": optional_unique[:room],
        "bulk": bulk,
        "units": _bulk_units(text),
        "fallback": fallback,
    }
    return selected, info


def select_tools_for_query(query: str, cap: int = TOOLS_PER_TASK) -> list[dict[str, Any]]:
    """Pick the schemas a free-form question can need.

    The selector answers one question: *which tool must the model be able to
    call for this sentence?* It is a router, not a keyword dump:

    * a named physics metric pulls in the frame sources the metric actually
      reads (:data:`_METRIC_SOURCE_TOOLS`) plus the tool that computes it;
    * non-metric intents are matched by :data:`_CAPABILITY_RULES`, most
      specific first;
    * a station question is not a bulk job: the matrix pair only appears when
      the request itself asks for matrix-scale work (:func:`_is_bulk_query`);
    * the mandatory set is never truncated by the cap, so the decisive tool
      cannot be evicted by presentation tools;
    * when nothing matches, a safe computational core is returned instead of
      the first *cap* schemas in catalog order (which were project tools).

    The returned list stays in catalog order so the prompt is byte-stable for a
    given selection.
    """
    ordered, _ = _selection_plan(query, cap)
    keep = set(ordered)
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] in keep]


def explain_tool_selection(query: str, cap: int = TOOLS_PER_TASK) -> dict[str, Any]:
    """The selection plus the evidence behind it, for verbose logs and tests."""
    ordered, info = _selection_plan(query, cap)
    keep = set(ordered)
    schemas = [t for t in TOOL_SCHEMAS if t["function"]["name"] in keep]
    info["tools"] = [t["function"]["name"] for t in schemas]
    info["est_tokens"] = prompt_budget(SYSTEM_PROMPT, schemas)["est_tokens"]
    return info



def estimate_tokens(text: str) -> int:
    """Rough token count: ~3 chars/token for Cyrillic, ~4 for Latin.

    Good enough to fail a prompt that has out-grown its window; the real
    tokenizer is not guaranteed to be importable in every deployment.
    """
    cyrillic = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    other = len(text) - cyrillic
    return (cyrillic + 2) // 3 + (other + 3) // 4


def prompt_budget(
    system_prompt: str,
    schemas: list[dict[str, Any]],
    n_ctx: int = N_CTX,
) -> dict[str, Any]:
    """One budget line for a call: system + schemas, estimated tokens vs n_ctx."""
    schemas_json = _dumps(schemas)
    est = estimate_tokens(system_prompt) + estimate_tokens(schemas_json)
    return {
        "system_chars": len(system_prompt),
        "schemas_chars": len(schemas_json),
        "tools": len(schemas),
        "est_tokens": est,
        "n_ctx": n_ctx,
    }


#: Fields a tool result must keep when it is too big to send whole: the model
#: needs the outcome, the handle to reuse, and any number it is meant to quote.
_MODEL_PAYLOAD_KEYS = (
    "ok", "error", "message", "hint", "rows", "n_missing", "frame_handle",
    "available_handles", "handle_aliases", "path", "plot", "metric", "result",
    "station", "date", "computed", "offline", "publication_state", "source",
    "query_type", "count", "summary", "max_kp", "max_scale", "max_scale_time",
    "map_path", "metadata", "warnings", "fallback_hint",
)


#: An identical failing call repeated this many times in a row is a loop, not
#: progress. The third identical failure is what triggers the warning.
LOOP_THRESHOLD = 3

#: How many recent failures the Scratchpad keeps for the loop check.
LOOP_WINDOW = 8

#: Tool-specific "what to fix" advice shown when a loop is detected. A model
#: that re-sends a broken call needs the concrete argument, not "try again".
_LOOP_FIX_HINT: dict[str, str] = {
    "plot_overlay": (
        "plot_overlay requires 'stations' (a list of IAGA codes, e.g. ['IRT']) "
        "and a component (e.g. \"component\": \"H\"); it does NOT take 'df1'/'df2'."
    ),
    "plot_comparison": (
        "plot_comparison requires 'df1' and 'df2' (two frame handles) plus "
        "'component'; it does NOT take 'stations'."
    ),
    "plot_space_weather_comparison": (
        "call fetch_solar_wind FIRST (this tool needs a 'sw:omni:<date>' frame); "
        "then pass 'ground_df_descriptor' (e.g. 'raw:irt:2024-09-12', or a derived "
        "frame carrying 'dst_proxy') and 'solar_wind_df_descriptor' (e.g. "
        "'sw:omni:2024-09-12'). ground_component must be H/dst_proxy/dH_dt, NOT "
        "'Dst' (global Kyoto index). It does NOT take 'stations'/'df1'/'df2'."
    ),
}

_LOOP_FIX_HINT_DEFAULT = (
    "the argument names or values are wrong -- re-read the tool description, "
    "fix the arguments, then call it once."
)


#: How many live handles the scratchpad shows. A long matrix run can create
#: dozens; the newest ones are what the next step needs. The active handle is
#: always repeated on its own line, so it stays visible even when it is older
#: than the display window.
HANDLE_DISPLAY_LIMIT = 12

#: How many recent tool exchanges stay verbatim in the LLM history. Older ones
#: are elided; their handles, completed steps and errors survive in the
#: Scratchpad, so the model keeps the state without re-reading old payloads.
HISTORY_KEEP_EXCHANGES = 6


class Scratchpad:
    """Persistent working memory for the agent during a single user query.

    Tracks a registry of every live frame handle, the active handle, completed
    steps, encountered errors (with hints), and an optional plan. Injected into
    the context before each LLM call to prevent losing state between rounds and
    to enable lightweight self-learning.

    The registry matters because one ``current_handle`` slot is not enough: a
    solar-wind fetch creates ``sw:omni:...``, then a ground fetch makes
    ``raw:...`` active and the wind handle disappears from the injected state.
    The model may need it again for a second coupling metric; guessing the name
    happens to work while naming is deterministic, but it is not state.
    """

    def __init__(self) -> None:
        self.current_handle: str = ""
        #: Exact store handle -> compact provenance (family, tool, rows, metrics).
        self.handles: dict[str, dict[str, Any]] = {}
        self._handle_seq = 0
        #: Count of old chat messages dropped by the history cap. Kept as a
        #: visible state note so the model does not hunt for elided results.
        self.elided_messages = 0
        self.completed_steps: list[str] = []
        self.errors: list[dict] = []
        self.plan: list[str] = []
        #: ``(tool, args_signature, error)`` for every failed call, newest last.
        #: The loop breaker reads the tail to catch an identical failing call
        #: repeated round after round.
        self.failed_calls: list[tuple[str, str, str]] = []

    def reset(self) -> None:
        self.current_handle = ""
        self.handles.clear()
        self._handle_seq = 0
        self.elided_messages = 0
        self.completed_steps.clear()
        self.errors.clear()
        self.plan.clear()
        self.failed_calls.clear()

    def update_handle(self, new_handle: str) -> None:
        if not isinstance(new_handle, str):
            return
        new_handle = new_handle.strip()
        if new_handle:
            self.current_handle = new_handle

    def register_handle(
        self,
        handle: Any,
        *,
        tool: str = "",
        payload: dict[str, Any] | None = None,
        metric: str = "",
        round_no: int = 0,
    ) -> None:
        """Remember a live frame handle and what produced it.

        Keyed by the exact store key, so a shared ``derived:`` slot accumulates
        the metrics written into it instead of spawning duplicate entries.
        """
        if not isinstance(handle, str):
            return
        handle = handle.strip()
        if not handle:
            return

        family = handle.split(":", 1)[0].lower() if ":" in handle else ""
        info = self.handles.get(handle)
        if info is None:
            self._handle_seq += 1
            info = {
                "handle": handle,
                "family": family,
                "tool": "",
                "order": self._handle_seq,
                "last_round": 0,
                "metrics": [],
                "rows": None,
            }
            self.handles[handle] = info

        if tool:
            info["tool"] = str(tool)
        if round_no:
            info["last_round"] = int(round_no)
        if isinstance(payload, dict):
            rows = payload.get("rows")
            if isinstance(rows, int):
                info["rows"] = rows
        if metric:
            metric = str(metric).strip().lower()
            if metric and metric not in info["metrics"]:
                info["metrics"].append(metric)

    def note_elided(self, count: int) -> None:
        """Record how many old chat messages the history cap removed."""
        if count > 0:
            self.elided_messages += int(count)

    @staticmethod
    def _describe_handle(info: dict[str, Any]) -> str:
        """One compact, copyable handle line: exact name first, provenance after."""
        handle = str(info.get("handle") or "")
        facts: list[str] = []
        rows = info.get("rows")
        if isinstance(rows, int):
            facts.append(f"{rows}r")

        metrics = [str(m) for m in (info.get("metrics") or [])]
        family = str(info.get("family") or "")
        if metrics:
            shown = ", ".join(metrics[-4:])
            if len(metrics) > 4:
                shown = f"+{len(metrics) - 4} more: {shown}"
            facts.append(shown)
        elif family == "sw":
            facts.append("OMNI")
        elif family == "idx":
            facts.append("Dst/Kp")
        elif family == "raw":
            parts = handle.split(":")
            if len(parts) > 1 and parts[1]:
                facts.append(parts[1].upper())
        else:
            tool = str(info.get("tool") or "")
            if tool:
                facts.append(_SCRATCHPAD_STEP_BY_TOOL.get(tool, tool))

        return f"{handle} ({'; '.join(facts)})" if facts else handle

    def _displayed_handles(
        self, limit: int = HANDLE_DISPLAY_LIMIT
    ) -> list[dict[str, Any]]:
        """The newest handles, one per produced slot, capped for the prompt."""
        ordered = sorted(
            self.handles.values(),
            key=lambda info: (
                int(info.get("last_round") or 0),
                int(info.get("order") or 0),
            ),
            reverse=True,
        )
        return ordered[: max(1, int(limit))]

    def add_completed_step(self, step: str) -> None:
        if not isinstance(step, str):
            return
        step = step.strip()
        if not step:
            return
        if step not in self.completed_steps:
            self.completed_steps.append(step)

    def add_error(self, tool: str, error: str, hint: str = "") -> None:
        entry: dict[str, Any] = {
            "tool": str(tool) if tool is not None else "",
            "error": str(error) if error is not None else "",
        }
        if hint:
            entry["hint"] = str(hint)
        self.errors.append(entry)

    def add_failure(self, tool: str, signature: str, error: str) -> None:
        """Record one failed call as ``(tool, argument fingerprint, error)``.

        Only the tail matters to the loop breaker, so the list is trimmed to the
        last :data:`LOOP_WINDOW` entries instead of growing for the whole run.
        """
        self.failed_calls.append(
            (
                str(tool) if tool is not None else "",
                str(signature) if signature is not None else "",
                str(error) if error is not None else "",
            )
        )
        if len(self.failed_calls) > LOOP_WINDOW:
            del self.failed_calls[:-LOOP_WINDOW]

    def repeated_failure(self) -> tuple[str, str] | None:
        """The identical failing call seen :data:`LOOP_THRESHOLD` times in a row.

        Returns ``(tool, error)`` when the last ``LOOP_THRESHOLD`` entries share
        the same tool, argument fingerprint and error -- the signature of a model
        stuck re-emitting one broken call -- and ``None`` otherwise.
        """
        if len(self.failed_calls) < LOOP_THRESHOLD:
            return None
        tail = self.failed_calls[-LOOP_THRESHOLD:]
        if any(entry != tail[0] for entry in tail):
            return None
        return tail[0][0], tail[0][2]

    def to_prompt_injection(self) -> str:
        """One compact line injected into the system prompt before every LLM call.

        Built only from fields that carry information, so an untouched scratchpad
        adds a single short line rather than a block of empties -- the schema diet
        keeps the window small and this must not eat into it.
        """
        parts: list[str] = []
        if self.handles:
            shown = self._displayed_handles()
            rendered = "; ".join(self._describe_handle(info) for info in shown)
            hidden = len(self.handles) - len(shown)
            if hidden > 0:
                rendered += f"; +{hidden} older"
            parts.append(f"Handles: {rendered}")
        if self.current_handle:
            parts.append(f"Active handle: {self.current_handle}")
        if self.elided_messages:
            parts.append(
                f"Context: {self.elided_messages} older tool messages elided "
                "(live handles above)"
            )
        if self.completed_steps:
            parts.append(f"Completed: {', '.join(self.completed_steps)}")
        if self.errors:
            recent: list[str] = []
            for err in self.errors[-3:]:  # only the freshest mistakes matter
                line = f"{err.get('tool', '?')}: {err.get('error', '')}"
                if err.get("hint"):
                    line += f" ({err['hint']})"
                recent.append(line)
            parts.append(f"Errors: {'; '.join(recent)}")
        if self.plan:
            parts.append(f"Plan: {', '.join(self.plan)}")

        body = ". ".join(parts) if parts else "No state yet"
        loop = self.repeated_failure()
        if loop:
            tool, error = loop
            fix = _LOOP_FIX_HINT.get(tool, _LOOP_FIX_HINT_DEFAULT)
            return (
                f"[SCRATCHPAD] {body}. "
                f"[CRITICAL LOOP DETECTED]: Вы вызвали '{tool}' {LOOP_THRESHOLD} раза "
                f"подряд с теми же аргументами и той же ошибкой ({error}). Остановитесь. "
                f"Прочитайте описание инструмента: {fix} "
                f"NEXT STEP: change the arguments before calling {tool} again."
            )
        return f"[SCRATCHPAD] {body}. NEXT STEP: {self._next_step()}"

    def _next_step(self) -> str:
        """A short, state-derived hint about what to do next."""
        loop = self.repeated_failure()
        if loop:
            return (
                f"STOP calling {loop[0]} with the same arguments -- the identical "
                "call has failed repeatedly. Fix the arguments from its schema, "
                "then call it once."
            )
        if self.errors:
            if self.current_handle:
                return (
                    f"the last call failed -- retry it with the Active handle "
                    f"{self.current_handle!r} (do not reuse the rejected name)."
                )
            # No handle yet: the call was refused before it produced one (an
            # invented tool name, a malformed call). Point at the fix instead of
            # "call the data source", which sent the model off to plot whatever
            # it happened to have instead of correcting the rejected call.
            return (
                "the last call failed -- apply the fix in Errors above before "
                "calling anything else."
            )
        if not self.completed_steps:
            return "call the data source first, then analyse and plot."
        if "plot" not in self.completed_steps and not any(
            s.startswith("plot") for s in self.completed_steps
        ):
            return "continue towards the requested result (analyse / plot)."
        return "summarise the result for the user; do not repeat completed steps."


def _trim_history(
    messages: list[dict[str, Any]],
    keep_exchanges: int = HISTORY_KEEP_EXCHANGES,
) -> int:
    """Drop the oldest tool exchanges from the LLM context.

    The system + user preamble is never touched. Only whole exchanges are
    removed: an ``assistant`` message carrying ``tool_calls`` plus the ``tool``
    messages that follow it must stay together, otherwise the Qwen chat template
    renders a result with no call it belongs to. Old payloads are redundant --
    the Scratchpad's handle registry and completed-step list are the durable
    memory -- and dropping them keeps a 25-call run from filling the window with
    results the model has already used.

    Returns the number of messages removed.
    """
    if len(messages) <= 2:
        return 0
    keep = max(1, int(keep_exchanges))
    starts = [
        index
        for index, message in enumerate(messages)
        if index >= 2
        and message.get("role") == "assistant"
        and message.get("tool_calls")
    ]
    if len(starts) <= keep:
        return 0
    cut = starts[-keep]
    if cut <= 2:
        return 0
    removed = cut - 2
    del messages[2:cut]
    return removed


#: Friendly progress label recorded on the scratchpad for each successful tool.
#: Keeps the "Completed:" line short and readable instead of echoing tool names.
_SCRATCHPAD_STEP_BY_TOOL: dict[str, str] = {
    "fetch_observatory_data": "fetch",
    "fetch_many": "fetch_many",
    "fetch_solar_wind": "solar_wind",
    "fetch_geomagnetic_indices": "indices",
    "get_noaa_space_weather_context": "noaa_context",
    "get_iswa_perturbation_map": "iswa_map",
    "get_iswa_forecast": "iswa_forecast",
    "fetch_supermag_data": "supermag",
    "plot_space_weather_comparison": "space_weather_plot",
    "calculate_derived_components": "calc_hdi",
    "get_statistics": "stats",
    "detect_anomalies": "anomalies",
    "calculate_derived_math": "derived_math",
    "calculate_baseline": "baseline",
    "evaluate_custom_formula": "formula",
    "plot_components": "plot",
    "plot_comparison": "plot_comparison",
    "plot_overlay": "plot_overlay",
    "create_project": "create_project",
    "export_project": "export_project",
}

#: Frame families that are auxiliary inputs (space weather) rather than the
#: working magnetometer frame. Kept out of the Scratchpad's "Active handle".
_AUX_FRAME_FAMILIES = {"sw", "idx"}

#: Hallucinated physics tool names -> the metric that actually serves the request.
#: A small model invents "calculate_dst_proxy" and friends; instead of a bare
#: "unknown_tool" it gets told the one correct call to make next.
_PHYSICS_TOOL_TO_METRIC: dict[str, str] = {
    "calculate_dst_proxy": "dst_proxy",
    "calculate_dst": "dst_proxy",
    "compute_dst": "dst_proxy",
    "dst_proxy": "dst_proxy",
    "calculate_gic": "gic",
    "calculate_gic_proxy": "gic",
    "compute_gic": "gic",
    "gic": "gic",
    "dH_dt": "gic",
    "calculate_dh_dt": "gic",
    "calculate_ssc": "detect_ssc",
    "detect_ssc": "detect_ssc",
    "ssc": "detect_ssc",
    "calculate_ssc_detection": "detect_ssc",
    "calculate_sq_baseline": "sq_baseline",
    "sq_baseline": "sq_baseline",
    "calculate_baseline_sq": "sq_baseline",
    "calculate_deltas": "deltas",
    "compute_deltas": "deltas",
    "deltas": "deltas",
    "calculate_coupling": "ey_coupling",
    "calculate_ey": "ey_coupling",
    "calculate_coupling_function": "ey_coupling",
    "coupling_function": "ey_coupling",
    "solar_wind_coupling": "ey_coupling",
    "ey_coupling": "ey_coupling",
}


def _self_correction_hint(tool_name: Any) -> str:
    """The [SELF-CORRECTION] line the Scratchpad shows after a bad tool name.

    Kept to one short line: a 7B model scans rather than reads, and a long hint
    competes with the Scratchpad's own NEXT STEP line for its attention. A
    physics-looking name is answered with the exact ``metric`` to pass next.
    """
    name = str(tool_name or "").strip()
    metric = _PHYSICS_TOOL_TO_METRIC.get(name) or _PHYSICS_TOOL_TO_METRIC.get(name.lower())
    if metric:
        return f"[SELF-CORRECTION]: Используй apply_physics_formula с metric='{metric}'."
    return f"[SELF-CORRECTION]: Инструмент '{name}' не существует. Выбери доступный."


def _compact_tool_message(payload: Any, max_chars: int = FRAME_SUMMARY_MAX_CHARS) -> str:
    """The string a tool result turns into for the model, capped at *max_chars*.

    A 1440-row frame summarised with per-column statistics is legitimately
    large, but the model does not need the column dump to act -- it needs the
    handle and the headline number. Handlers still return the full payload to
    their callers; only what crosses into the context window is trimmed here.

    A payload carrying a ``fallback_hint`` is the exception: that key *is* the
    instruction the model must follow (the ISWA map is decommissioned, use
    SuperMAG indices for the global picture and a station handle for
    plot_space_weather_comparison). Truncating it mid-sentence is worse than
    sending it whole, so such a payload is kept intact up to
    :data:`FALLBACK_HINT_MAX_CHARS` -- still a hard cap, just a roomier one.
    """
    text = _dumps(payload)
    if isinstance(payload, dict) and payload.get("fallback_hint"):
        limit = max(max_chars, FALLBACK_HINT_MAX_CHARS)
        return text if len(text) <= limit else text[:limit]
    if len(text) <= max_chars or not isinstance(payload, dict):
        return text if len(text) <= max_chars else text[:max_chars]

    compact = {k: payload[k] for k in _MODEL_PAYLOAD_KEYS if k in payload}
    # Keep the newest handles only; older ones rarely matter for the next step.
    handles = compact.get("available_handles")
    while isinstance(handles, list) and len(handles) > 3 and len(_dumps(compact)) > max_chars:
        handles = handles[-3:]
        compact["available_handles"] = handles
    text = _dumps(compact)
    if len(text) <= max_chars:
        return text

    compact.pop("handle_aliases", None)
    text = _dumps(compact)
    return text if len(text) <= max_chars else text[:max_chars]


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

    def __init__(
        self,
        llm: Any,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        n_ctx: int = N_CTX,
    ):
        self._llm = llm
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._n_ctx = n_ctx

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        # One line per call so an over-full window is visible before the model
        # returns an empty answer, not after.
        system = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"), ""
        )
        budget = prompt_budget(system or "", list(tools or []), self._n_ctx)
        print(
            f"[PROMPT BUDGET] system={budget['system_chars']}ch "
            f"schemas={budget['schemas_chars']}ch tools={budget['tools']} "
            f"est_tokens={budget['est_tokens']} / {budget['n_ctx']}",
            flush=True,
        )
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

    def _try_load(ctx: int) -> Any:
        return Llama(
            model_path=str(path),
            n_ctx=ctx,
            n_gpu_layers=layers,
            n_batch=N_BATCH,
            n_threads=max(1, (os.cpu_count() or 4)),
            verbose=verbose,
        )

    try:
        llm = _try_load(n_ctx)
    except Exception as exc:
        # A 16384 context is the target, but on a tight GPU llama.cpp can refuse
        # to allocate it. Step down to 12288 once rather than fall over; the
        # schema diet already keeps the live prompt far under either window.
        if n_ctx > N_CTX_FALLBACK:
            if verbose:
                print(
                    f"[agent] n_ctx={n_ctx} failed ({type(exc).__name__}); "
                    f"retrying with {N_CTX_FALLBACK}",
                    flush=True,
                )
            try:
                llm = _try_load(N_CTX_FALLBACK)
                n_ctx = N_CTX_FALLBACK
            except Exception as exc2:
                return _error(
                    "model_load_failed",
                    f"llama.cpp refused {path}: {exc2}",
                    path=str(path),
                    n_ctx=N_CTX_FALLBACK,
                )
        else:
            return _error(
                "model_load_failed", f"llama.cpp refused {path}: {exc}", path=str(path), n_ctx=n_ctx
            )
    return LlamaBrain(llm, n_ctx=n_ctx)


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


def _offline_solar_wind(start: str) -> pd.DataFrame:
    """A deterministic storm driver for offline runs (no network at all).

    Reproduces the shape that matters physically: a fast stream carrying a
    southward IMF through the middle of the day, so the coupling tool returns a
    strong positive Ey.
    """
    try:
        day = pd.Timestamp(str(start))
        if pd.isna(day):
            raise ValueError(start)
    except (TypeError, ValueError):
        day = pd.Timestamp("2024-09-12")
    stamps = pd.date_range(day.normalize(), periods=1440, freq="min", tz="UTC")
    t = np.linspace(0.0, 1.0, len(stamps))
    bump = np.exp(-((t - 0.5) ** 2) / (2 * 0.05 ** 2))
    frame = pd.DataFrame({
        "timestamp": stamps,
        "V": 380.0 + 320.0 * bump,
        "Np": 4.0 + 12.0 * bump,
        "T": 8.0e4 + 9.0e5 * bump,
        "Bx": 1.0,
        "By": -2.0,
        "Bz": 3.0 - 21.0 * bump,
    })
    frame.attrs = {"dataset": "offline-synthetic (OMNI shape)", "offline": True}
    return frame


def _offline_indices(start: str) -> pd.DataFrame:
    """A deterministic Dst/Kp day with a storm-time depression around 12:00 UT."""
    try:
        day = pd.Timestamp(str(start))
        if pd.isna(day):
            raise ValueError(start)
    except (TypeError, ValueError):
        day = pd.Timestamp("2024-09-12")
    stamps = pd.date_range(day.normalize(), periods=24, freq="h", tz="UTC")
    t = np.linspace(0.0, 1.0, len(stamps))
    bump = np.exp(-((t - 0.5) ** 2) / (2 * 0.08 ** 2))
    frame = pd.DataFrame({
        "timestamp": stamps,
        "Dst": 5.0 - 130.0 * bump,
        "Kp": 1.0 + 6.0 * bump,
    })
    frame.attrs = {"source": "offline-synthetic (Kyoto shape)", "offline": True}
    return frame


def _offline_noaa(query_type: str) -> dict[str, Any]:
    """Deterministic NOAA SWPC context for offline runs (no network at all).

    Mirrors the live loader payload so the handler and the model see the same
    fields: a single K5/G1 alert, a 3-day Kp forecast peaking at G1, and two
    X-ray flares. Values are the documented 2026-10-10 G3 / quiet-day picture.
    """
    if query_type == "alerts":
        data = [{
            "product_id": "K05A", "kind": "ALERT", "k_index": 5.0, "g_scale": "G1",
            "severity": "Minor", "issue_time": "2026-10-10T21:45:00+00:00",
            "start_time": "2026-10-10T21:45:00+00:00", "end_time": None,
            "active_warning": True,
            "summary": "ALERT: Geomagnetic K-index of 5 (G1 - Minor)",
        }]
        return {"ok": True, "data": data, "count": len(data),
                "source": "NOAA SWPC (offline synthetic)"}
    if query_type == "forecast":
        data = [
            {"time_tag": "2026-10-16T18:00:00+00:00", "kp": 2.0,
             "status": "observed", "g_scale": None},
            {"time_tag": "2026-10-17T06:00:00+00:00", "kp": 5.0,
             "status": "predicted", "g_scale": "G1"},
            {"time_tag": "2026-10-17T12:00:00+00:00", "kp": 3.7,
             "status": "predicted", "g_scale": None},
        ]
        return {"ok": True, "data": data, "count": len(data),
                "source": "NOAA SWPC (offline synthetic)",
                "max_kp": 5.0, "max_scale": "G1",
                "max_scale_time": "2026-10-17T06:00:00+00:00"}
    data = [
        {"begin_time": "2026-10-17T06:22:00+00:00", "max_time": "2026-10-17T06:30:00+00:00",
         "end_time": "2026-10-17T06:40:00+00:00", "begin_class": "B6.7",
         "max_class": "C1.2", "class_letter": "C", "satellite": 18},
        {"begin_time": "2026-10-10T13:18:00+00:00", "max_time": "2026-10-10T13:25:00+00:00",
         "end_time": "2026-10-10T13:30:00+00:00", "begin_class": "B9.5",
         "max_class": "M2.7", "class_letter": "M", "satellite": 18},
    ]
    return {"ok": True, "data": data, "count": len(data),
            "source": "NOAA SWPC (offline synthetic) (GOES X-ray flare catalogue)"}


def _handle_fetch_solar_wind(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Fetch OMNI L1 solar wind and store it under a ``sw:omni:<day>`` handle.

    Solar wind is the *cause* half of a storm: the coupling function needs V and
    Bz on the observatory clock, and the handle is what carries those columns to
    ``apply_physics_formula`` without a DataFrame ever entering the JSON channel.
    """
    start = str(args.get("start_date") or "").strip()
    end = str(args.get("end_date") or start).strip()
    if not start:
        return _error("invalid_input", "start_date is required (YYYY-MM-DD)."), "missing start_date"

    slot = f"sw:omni:{start}"
    if OFFLINE:
        frame = _offline_solar_wind(start)
        store.put(slot, frame)
        summary = _summarise_frame(frame, store, slot)
        summary["offline"] = True
        summary["source"] = "OMNI (offline synthetic)"
        return summary, f"Offline: солнечный ветер синтезирован, {len(frame)} строк -> {slot!r}."

    result = space_weather.fetch_solar_wind(start, end)
    if space_weather.is_error(result):
        return result, f"fetch_solar_wind failed: {result.get('error')}"
    store.put(slot, result)
    summary = _summarise_frame(result, store, slot)
    summary["source"] = (result.attrs or {}).get("dataset") or "OMNI (L1)"
    summary["start_date"] = start
    summary["end_date"] = end
    return summary, f"OMNI солнечный ветер: {len(result)} строк ({start}..{end}) -> {slot!r}."


def _handle_fetch_geomagnetic_indices(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Fetch Kyoto Dst/Kp and store them under an ``idx:kyoto:<day>`` handle.

    Indices are the *severity* half of a storm: Dst quantifies the ring-current
    depression and Kp the planetary activity, so they answer "how strong was it"
    while the solar wind answers "why".
    """
    start = str(args.get("start_date") or "").strip()
    end = str(args.get("end_date") or start).strip()
    if not start:
        return _error("invalid_input", "start_date is required (YYYY-MM-DD)."), "missing start_date"

    slot = f"idx:kyoto:{start}"
    if OFFLINE:
        frame = _offline_indices(start)
        store.put(slot, frame)
        summary = _summarise_frame(frame, store, slot)
        summary["offline"] = True
        summary["source"] = "Kyoto WDC (offline synthetic)"
        return summary, f"Offline: индексы Dst/Kp синтезированы, {len(frame)} строк -> {slot!r}."

    result = space_weather.fetch_geomagnetic_indices(start, end)
    if space_weather.is_error(result):
        return result, f"fetch_geomagnetic_indices failed: {result.get('error')}"
    store.put(slot, result)
    summary = _summarise_frame(result, store, slot)
    summary["source"] = "Kyoto WDC (Dst hourly, Kp 3-hourly)"
    summary["start_date"] = start
    summary["end_date"] = end
    return summary, f"Индексы Dst/Kp: {len(result)} строк ({start}..{end}) -> {slot!r}."


#: How many NOAA records the model sees before only the digest survives the cap.
_NOAA_MAX_ROWS = 8

#: Flare class letter -> rank, for "strongest flare" comparisons.
_NOAA_FLARE_RANK = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}


def _summarize_noaa(query_type: str, result: dict[str, Any]) -> str:
    """One-line, quotable digest of a NOAA payload.

    The handler payload is capped by ``_compact_tool_message``, so this line must
    carry the facts a storm report needs even when the raw rows are dropped.
    """
    data = result.get("data") or []
    if query_type == "alerts":
        if not data:
            return "Активных геомагнитных алертов NOAA нет."
        scales = [a.get("g_scale") for a in data if a.get("g_scale")]
        worst = max(scales, key=lambda g: int(str(g)[1:])) if scales else None
        top = data[0]
        parts = [f"Алертов NOAA: {len(data)}"]
        if worst:
            parts.append(f"максимум {worst}")
        parts.append(f"последний {top.get('summary')} ({top.get('issue_time')})")
        return "; ".join(parts) + "."
    if query_type == "forecast":
        if not data:
            return "Прогноз Kp NOAA недоступен."
        tail = f" = {result['max_scale']}" if result.get("max_scale") else ""
        if result.get("max_scale_time"):
            tail += f" к {result['max_scale_time']}"
        return (
            f"Прогноз Kp (3 дня): максимум Kp {result.get('max_kp')}{tail}; "
            f"строк {len(data)}."
        )
    if not data:
        return "Вспышек в выбранном окне NOAA не зарегистрировано."
    strongest = max(
        data, key=lambda f: _NOAA_FLARE_RANK.get(str(f.get("class_letter") or ""), -1)
    )
    cmx = sum(1 for f in data if str(f.get("class_letter") or "") in ("C", "M", "X"))
    return (
        f"Вспышек NOAA: {len(data)} (C/M/X: {cmx}); "
        f"сильнейшая {strongest.get('max_class')} в {strongest.get('max_time')}."
    )


def _handle_noaa_context(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Return official NOAA SWPC context (alerts / forecast / flare catalogue).

    This is the *forecast and warning* layer: unlike the magnetogram and solar
    wind loaders it does not produce a frame handle, because what NOAA publishes
    is already a digested list of warnings, predicted Kp values and flares. The
    payload is trimmed so a large catalogue cannot be silently compacted away.
    """
    query_type = str(args.get("query_type") or args.get("type") or "").strip().lower()
    if query_type not in ("alerts", "forecast", "solar_events"):
        return _error(
            "invalid_input",
            "query_type must be one of 'alerts', 'forecast', 'solar_events'.",
            allowed=["alerts", "forecast", "solar_events"],
        ), "bad query_type"

    start = str(args.get("start_date") or "").strip()
    end = str(args.get("end_date") or start).strip()

    if OFFLINE:
        result = _offline_noaa(query_type)
    elif query_type == "alerts":
        result = noaa.fetch_geomagnetic_alerts()
    elif query_type == "forecast":
        result = noaa.fetch_solar_wind_forecast()
    else:
        result = noaa.fetch_solar_events(start or None, end or None)

    if noaa.is_error(result):
        return result, f"get_noaa_space_weather_context failed: {result.get('error')}"

    data = list(result.get("data") or [])
    payload: dict[str, Any] = {
        "ok": True,
        "query_type": query_type,
        "source": result.get("source"),
        "count": len(data),
        "summary": _summarize_noaa(query_type, result),
        "data": data[:_NOAA_MAX_ROWS],
    }
    for key in ("max_kp", "max_scale", "max_scale_time", "message"):
        if result.get(key) is not None:
            payload[key] = result[key]
    return payload, f"NOAA {query_type}: {len(data)} записей ({payload['summary']})"


def _offline_iswa_map(
    timestamp: str,
    bbox: Any = None,
    width: Any = 1024,
    height: Any = 512,
) -> dict[str, Any]:
    """The offline answer for the retired map layer (mirrors the live one).

    The WMS layer is decommissioned for real, so offline must not pretend a
    picture exists -- a synthetic PNG here would teach the model that the map
    works, which is exactly the mistake this phase removes. Offline reports the
    same ``service_decommissioned`` payload the live loader returns, with the
    same ``fallback_hint``, after validating the arguments so a malformed call
    is still ``invalid_input``.
    """
    try:
        iso = iswa.iso_z(iswa.parse_timestamp(timestamp))
        iswa._norm_bbox(bbox)
        iswa._norm_size(width, "width")
        iswa._norm_size(height, "height")
    except ValueError as exc:
        return _error(
            "invalid_input",
            str(exc),
            hint=(
                "Pass 'timestamp' as ISO 8601 (e.g. '2024-09-12T12:00:00Z'), "
                "bbox as [minx, miny, maxx, maxy], width/height as pixel counts."
            ),
        )
    return iswa._decommissioned()


def _handle_iswa_perturbation_map(
    args: dict[str, Any], store: FrameStore
) -> tuple[Any, str]:
    """Report the ISWA ground magnetic perturbation map layer's decommission.

    NASA CCMC retired the public WMS map layer in the ISWA migration, so this
    tool no longer draws a picture: it returns ``service_decommissioned`` with a
    ``fallback_hint`` naming the SuperMAG indices that cover the same science
    and the Dst/Kp forecast. The arguments are still validated (they are what
    the model fills in), but no map path is produced.
    """
    strict = _strict_argument_check(
        "get_iswa_perturbation_map",
        args,
        allowed=("timestamp", "bbox", "width", "height"),
        required=("timestamp",),
        missing_hint=(
            "CRITICAL: You forgot the mandatory argument 'timestamp'. It MUST be "
            "an ISO 8601 instant like '2024-09-12T12:00:00Z'."
        ),
    )
    if strict is not None:
        return strict, "bad iswa map arguments"

    timestamp = str(args.get("timestamp")).strip()
    bbox = args.get("bbox")
    width = args.get("width", 1024)
    height = args.get("height", 512)

    if OFFLINE:
        result = _offline_iswa_map(timestamp, bbox, width, height)
    else:
        result = iswa.get_ground_perturbation_map(
            timestamp, bbox=bbox, width=width, height=height
        )
    if iswa.is_error(result):
        return result, (
            f"get_iswa_perturbation_map ({timestamp}): {result.get('error')} — "
            f"{result.get('fallback_hint') or result.get('message')}"
        )

    meta = result.get("metadata") or {}
    return result, (
        f"Карта возмущений ISWA на {meta.get('timestamp')} -> {result.get('map_path')}"
    )


def _offline_iswa_forecast() -> dict[str, Any]:
    """A deterministic Dst/Kp outlook for offline runs (no network at all).

    Window and shape mirror the live loader: 24 h back / 48 h ahead on a
    three-hour grid. The past half carries "observed" Kp values that peak at
    5.7 (a G1 storm) with Dst bottoms out at -66 nT, and the future half
    carries "predicted" Kp peaking at 5.3 -- and ``dst: None``, because no Dst
    forecast exists anywhere in ISWA.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=24)
    stop = now + timedelta(hours=48)
    observed = [3.0, 4.3, 5.7, 4.7, 3.3, 2.7, 2.3, 2.0, 1.7]
    predicted = [2.0, 2.7, 3.3, 4.0, 4.7, 5.3, 4.7, 4.0, 3.3]
    dst_past = [-20.0, -35.0, -52.0, -66.0, -58.0, -44.0, -33.0, -24.0, -18.0]

    forecast: list[dict[str, Any]] = []
    stamp = start
    index = 0
    while stamp <= stop:
        if stamp <= now:
            kp = observed[index % len(observed)]
            dst: float | None = dst_past[index % len(dst_past)]
            kind = "observed"
        else:
            kp = predicted[index % len(predicted)]
            dst = None
            kind = "predicted"
        forecast.append(
            {
                "timestamp": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dst": dst,
                "kp": kp,
                "kp_kind": kind,
            }
        )
        index += 1
        stamp += timedelta(hours=3)

    return {
        "ok": True,
        "forecast": forecast,
        "count": len(forecast),
        "source": "ISWA HAPI (offline synthetic forecast)",
        "window": {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stop": stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "step_hours": 3,
            "now": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "sources": {
            "kp": "ISWA HAPI NOAA_KP_P3H (offline synthetic)",
            "dst": "ISWA HAPI dst_quicklook (offline synthetic)",
        },
    }


#: How many forecast rows the model sees; ``summary`` carries the rest.
_ISWA_MAX_ROWS = 8

_KP_KIND_LABEL = {
    "observed": "наблюдение",
    "predicted": "прогноз",
    "estimated": "оценка",
    "reported": "Kp-3H",
}


def _summarize_iswa_forecast(result: dict[str, Any]) -> str:
    """One-line, quotable digest of a Dst/Kp outlook.

    ``_compact_tool_message`` will drop the ``forecast`` array itself, so the
    peak, its kind and the horizon must live in this line. The kind is spelled
    out because a Kp of 5.7 that is *predicted* and one that was *observed* mean
    different things to the user.
    """
    rows = list(result.get("forecast") or [])
    valued = [r for r in rows if r.get("kp") is not None]
    peak = max(valued, key=lambda r: r["kp"]) if valued else None
    dsts = [r["dst"] for r in rows if r.get("dst") is not None]
    stop = (result.get("window") or {}).get("stop") or (rows[-1]["timestamp"] if rows else "?")
    parts = [f"Прогноз ISWA: строк {len(rows)}, до {stop}"]
    if peak is not None:
        kind = _KP_KIND_LABEL.get(str(peak.get("kp_kind")), str(peak.get("kp_kind")))
        parts.append(f"Kp пик {peak['kp']} ({kind}) в {peak['timestamp']}")
    else:
        parts.append("Kp в окне нет")
    if dsts:
        parts.append(
            f"Dst минимум {min(dsts)} nT (наблюдения; вперёд не прогнозируется)"
        )
    else:
        parts.append("Dst недоступен")
    return "; ".join(parts) + "."


def _handle_iswa_forecast(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Return the ISWA Dst/Kp outlook: predicted Kp, quicklook Dst.

    Takes no arguments -- the loader owns the window (24 h back / 48 h ahead),
    because "what comes next" is not a question the model should have to phrase
    in dates. Like the NOAA handler it returns a digest rather than a frame: the
    rows feed ``summary``, and the newest ``_ISWA_MAX_ROWS`` ride along.
    """
    strict = _strict_argument_check(
        "get_iswa_forecast", args, allowed=(), required=()
    )
    if strict is not None:
        return strict, "bad iswa forecast arguments"

    result = _offline_iswa_forecast() if OFFLINE else iswa.get_dst_kp_forecast()
    if iswa.is_error(result):
        return result, f"get_iswa_forecast failed: {result.get('error')}"

    rows = list(result.get("forecast") or [])
    now = (result.get("window") or {}).get("now")
    horizon = [r for r in rows if now and r.get("timestamp", "") >= str(now)]
    payload: dict[str, Any] = {
        "ok": True,
        "source": result.get("source"),
        "count": len(rows),
        "summary": _summarize_iswa_forecast(result),
        "window": result.get("window"),
        "sources": result.get("sources"),
        "forecast": (horizon or rows)[:_ISWA_MAX_ROWS],
    }
    if result.get("warnings"):
        payload["warnings"] = result["warnings"]
    return payload, f"ISWA Dst/Kp: {len(rows)} строк ({payload['summary']})"



def _offline_supermag(kind: str, station: str, start: str) -> pd.DataFrame:
    """A deterministic substorm for offline runs (no network at all).

    Reproduces the shape that matters physically: an electrojet enhancement
    around 12:00 UT, so SMU rises while SML drops sharply and the station sees
    a negative bay -- the signature a substorm study asks about.
    """
    try:
        day = pd.Timestamp(str(start))
        if pd.isna(day):
            raise ValueError(start)
    except (TypeError, ValueError):
        day = pd.Timestamp("2024-09-12")
    stamps = pd.date_range(day.normalize(), periods=1440, freq="min", tz="UTC")
    t = np.linspace(0.0, 1.0, len(stamps))
    bump = np.exp(-((t - 0.5) ** 2) / (2 * 0.04 ** 2))
    if kind == "indices":
        frame = pd.DataFrame({
            "timestamp": stamps,
            "SME": 20.0 + 640.0 * bump,
            "SML": -15.0 - 520.0 * bump,
            "SMU": 15.0 + 90.0 * bump,
        })
        frame.attrs = {
            "source": "SuperMAG (offline synthetic)",
            "data_type": "indices",
            "units": "nT",
        }
        return frame
    frame = pd.DataFrame({
        "timestamp": stamps,
        "dX": 4.0 - 120.0 * bump,
        "dY": 2.0 - 35.0 * bump,
        "dZ": 1.0 + 12.0 * bump,
        "X_geo": 4.0 - 118.0 * bump,
        "Y_geo": 2.0 - 33.0 * bump,
        "Z_geo": 1.0 + 12.0 * bump,
    })
    frame.attrs = {
        "source": "SuperMAG (offline synthetic)",
        "station": station,
        "data_type": "sites",
        "units": "nT",
    }
    return frame


def _handle_fetch_supermag(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Fetch SuperMAG minute data and store it under its own handle family.

    Two products, one tool, because a substorm study needs both halves: the
    station frame (``dX/dY/dZ``, main field already removed) says what the
    ground saw, and the global SME/SML/SMU indices say when the auroral
    electrojet broke loose -- which is what dates the onset.

    ``data_type='indices'`` lands in the auxiliary ``idx:`` family, so the
    global product cannot hijack the Active handle a follow-up plot call
    reaches for; the station frame lands in ``smag:`` and becomes active, like
    any other working magnetometer frame.
    """
    strict = _strict_argument_check(
        "fetch_supermag_data",
        args,
        allowed=("station_code", "start_date", "end_date", "data_type"),
        required=("start_date",),
        missing_hint=(
            "CRITICAL: The mandatory argument is 'start_date', a YYYY-MM-DD string "
            "(e.g. '2024-09-12'). 'end_date' is optional and defaults to start_date. "
            "With data_type='sites' also pass station_code, e.g. 'IRT'; with "
            "data_type='indices' the station is ignored."
        ),
    )
    if strict is not None:
        return strict, f"fetch_supermag_data refused: {strict['error']}"

    station = str(args.get("station_code") or "").strip().upper()
    start = str(args.get("start_date") or "").strip()
    end = str(args.get("end_date") or start).strip()
    kind = str(args.get("data_type") or "sites").strip().lower()
    if kind not in supermag.DATA_TYPES:
        return _error(
            "invalid_input",
            "data_type must be 'sites' (станция) or 'indices' (SME/SML/SMU).",
            allowed=list(supermag.DATA_TYPES),
        ), "bad data_type"
    if kind == "sites" and not station:
        return _error(
            "invalid_input",
            "station_code is required when data_type='sites' (напр. 'IRT').",
        ), "missing station_code"

    slot = (
        f"idx:supermag:{start}" if kind == "indices" else f"smag:{station.lower()}:{start}"
    )
    slot = FrameStore._normalise(slot)

    if OFFLINE:
        frame = _offline_supermag(kind, station, start)
        store.put(slot, frame)
        summary = _summarise_frame(frame, store, slot)
        summary["offline"] = True
        summary["source"] = "SuperMAG (offline synthetic)"
        summary["data_type"] = kind
        if station:
            summary["station"] = station
        return summary, (
            f"Offline: SuperMAG {kind} синтезированы, {len(frame)} строк -> {slot!r}."
        )

    result = supermag.fetch_supermag_data(
        start, end, data_type=kind, station_code=station or None
    )
    if supermag.is_error(result):
        return result, f"fetch_supermag_data failed: {result.get('error')}"
    store.put(slot, result)
    summary = _summarise_frame(result, store, slot)
    summary["source"] = (result.attrs or {}).get("source")
    summary["data_type"] = kind
    summary["start_date"] = start
    summary["end_date"] = end
    if station:
        summary["station"] = station
    return summary, f"SuperMAG {kind}: {len(result)} строк ({start}..{end}) -> {slot!r}."


def _legacy_frame_handle(tool: str, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Resolve the frame argument of a pre-``df_descriptor`` tool.

    The physics tool speaks ``df_descriptor``; the math tools were written
    against ``df``. A model that has just used one reaches for the other, so
    both spellings are accepted here. ``df`` wins when a caller supplies both,
    which keeps the old call sites byte-for-byte compatible.
    """
    handle = args.get("df") or args.get("df_descriptor")
    if not handle:
        return "", _error(
            "missing_frame",
            f"{tool} needs a frame handle.",
            missing=["df"],
            hint=(
                "Pass 'df' (or the equivalent 'df_descriptor') with a handle from a "
                "previous tool's 'frame_handle', e.g. 'raw:irt:2024-09-12'."
            ),
        )
    return str(handle), None


def _handle_derive(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    handle, missing = _legacy_frame_handle("calculate_derived_components", args)
    if missing:
        return missing, "calculate_derived_components failed: missing_frame"
    frame, err = store.resolve(handle)
    if err:
        return err, f"calculate_derived_components failed: {err['error']}"

    components = args.get("components") or ["H", "D", "I"]
    result = analyzer.calculate_derived_components(frame, components=components)
    if analyzer.is_error(result):
        return result, f"calculate_derived_components failed: {result.get('error')}"

    source_slot = handle
    slot = store.derive_name(source_slot, "derived")
    store.put(slot, result)
    summary = _summarise_frame(result, store, slot)
    summary["computed"] = [c for c in ("H", "D", "I") if c in result.columns]
    return summary, f"Computed {', '.join(summary['computed'])} -> handle {slot!r}."


def _handle_stats(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    handle, missing = _legacy_frame_handle("get_statistics", args)
    if missing:
        return missing, "get_statistics failed: missing_frame"
    frame, err = store.resolve(handle)
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
    handle, missing = _legacy_frame_handle("detect_anomalies", args)
    if missing:
        return missing, "detect_anomalies failed: missing_frame"
    frame, err = store.resolve(handle)
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
    source_slot = handle
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
    handle, missing = _legacy_frame_handle("calculate_derived_math", args)
    if missing:
        return missing, "calculate_derived_math failed: missing_frame"
    frame, err = store.resolve(handle)
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
            slot = store.derive_name(handle, "math")
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
        slot = store.derive_name(handle, "math")
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
        slot = store.derive_name(handle, "math")
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
    handle, missing = _legacy_frame_handle("calculate_baseline", args)
    if missing:
        return missing, "calculate_baseline failed: missing_frame"
    frame, err = store.resolve(handle)
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
    """Run one STATION task: every event of the station, every action per event.

    The station loop is the outer dimension and the events list the inner one;
    both live here in Python, far from the model. Errors inside a single event
    are logged and do NOT abort the rest of the station's events. Per-event
    outcomes are written back onto the task as ``events_ok`` / ``events_failed``
    so the batch summary can tell a fully successful station from a partial one
    and from one that yielded no data at all. An all-key success returns
    ``(True, "")``; any failed event fails the station with an aggregated
    description, but the events that did work are still recorded as succeeded.
    """
    station = str(task.get("station_code") or "").strip().upper()
    events = [str(e) for e in (task.get("events") or [])]
    tools = task.get("actions") or _DEFAULT_MATRIX_ACTIONS
    event_failures: dict[str, list[str]] = {event: [] for event in events}
    for event in events:
        for tool in tools:
            if tool not in HANDLERS:
                event_failures[event].append(f"неизвестное действие {tool!r}")
                break
            # Data prerequisites: a cell whose actions omit fetch/derive must
            # not fail on an unknown_frame -- the plot just gets data implicitly.
            if tool != "fetch_observatory_data":
                ok, note = _ensure_matrix_fetch(station, event, store)
                if not ok:
                    event_failures[event].append(note)
                    break
            if tool in ("plot_components", "plot_overlay"):
                derived_slot = f"derived:{station.lower()}:{event}"
                if store.get(derived_slot) is None:
                    ok, note = _ensure_matrix_derived(station, event, store)
                    if not ok:
                        event_failures[event].append(note)
                        break
            try:
                payload, _ = HANDLERS[tool](
                    _matrix_task_args(tool, station, event, store), store
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                event_failures[event].append(
                    f"{tool} raised {type(exc).__name__}: {exc}"
                )
                break
            if is_error(payload):
                event_failures[event].append(
                    f"{tool}: {payload.get('error')}: "
                    f"{str(payload.get('message'))[:200]}"
                )
                break
            if tool == "fetch_observatory_data" and payload.get("frame_handle"):
                frame = store.get(payload["frame_handle"])
                if frame is not None:
                    store.workspace.save(station, event, frame, kind="raw")
            if tool == "calculate_derived_components":
                derived = store.get(f"derived:{station.lower()}:{event}")
                if derived is not None:
                    store.workspace.save(station, event, derived, kind="derived")
    failed_events = [e for e in events if event_failures[e]]
    ok_events = [e for e in events if not event_failures[e]]
    task["events_ok"] = ok_events
    task["events_failed"] = failed_events
    if not failed_events:
        return True, ""
    details = [f"{e}: {event_failures[e][0]}" for e in failed_events]
    for detail in details:
        print(f"[matrix] {station}/{detail} -- обработка станции продолжается.")
    return False, f"события: {'; '.join(details)}"


def _handle_create_task_matrix(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Write the station-grouped plan to ``<project>/<project>_matrix.json``.

    The matrix is the *generator* half of the pattern: ONE task per station,
    carrying the full list of its event dates, so the model never has to
    enumerate the events x stations product -- a 10x26 job is a 26-task matrix.
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
            "id": f"station_{station}",
            "station_code": station,
            "events": list(events),
            "actions": list(actions),
            "status": "pending",
            "attempts": 0,
            "error": None,
        }
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

    summary = (
        f"✅ Матрица создана. Всего задач: {len(tasks)}. Файл: {path}. "
        f"СЛЕДУЮЩИЙ ШАГ: Вызови process_matrix_batch с run_until_complete=True."
    )
    return {
        "ok": True,
        "project": project_name,
        "total": len(tasks),
        "matrix_path": str(path),
        "summary": summary,
    }, summary


def _matrix_station_class(task: dict[str, Any]) -> str:
    """Classify a processed station task as full / partial / empty / pending.

    ``full``  -- every event succeeded;
    ``partial`` -- at least one event succeeded and at least one had no data;
    ``empty`` -- no event yielded data (all no_data / station_not_found);
    ``pending`` -- not run yet. Falls back to the persisted ``status`` for
    matrices written before per-event tracking existed.
    """
    events_ok = task.get("events_ok")
    events_failed = task.get("events_failed")
    if events_ok is None and events_failed is None:
        status = task.get("status")
        if status == "done":
            return "full"
        if status == "partial":
            return "partial"
        if status == "failed":
            return "empty"
        return "pending"
    if not events_failed:
        return "full"
    if events_ok:
        return "partial"
    return "empty"


def _handle_process_matrix_batch(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Execute matrix station-tasks, handling tools directly, no model in the loop.

    Reads ``matrix_path`` and runs pending stations through
    :func:`_execute_matrix_task` -- direct Python handler calls. With
    ``run_until_complete=True`` (the production path) *every* pending station --
    and, inside it, every event of that station -- is processed inside this one
    call; ``run_until_complete=False`` (debugging) processes only ``batch_size``
    stations so the flow can be inspected step by step. Event errors inside a
    station are logged and recorded but never abort the station.
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
    if path is None:
        # The model guessed a wrong file name (e.g. "task_matrix.json"). Rather
        # than fail and burn another round-trip, fall back to the matrix of the
        # currently-open project, or to the only matrix in the workspace.
        current = store.workspace.project
        if current:
            probe = _matrix_path(current, store.workspace)
            if probe.is_file():
                path = probe
        if path is None:
            found = sorted(
                p
                for known in projects.list_projects(store.workspace.root)
                for p in projects.project_dir(known, store.workspace.root).glob(
                    "*_matrix.json"
                )
                if p.is_file()
            )
            if len(found) == 1:
                path = found[0]
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

    raw_flag = args.get("run_until_complete", False)
    if isinstance(raw_flag, bool):
        run_until_complete = raw_flag
    else:
        run_until_complete = str(raw_flag or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    raw_size = args.get("batch_size", MATRIX_BATCH_SIZE)
    try:
        batch_size = max(1, min(int(raw_size), 20))
    except (TypeError, ValueError):
        batch_size = MATRIX_BATCH_SIZE

    tasks = matrix.get("tasks") or []
    pending = [t for t in tasks if t.get("status") == "pending"]
    chunk = pending if run_until_complete else pending[:batch_size]

    succeeded = 0
    failed_here = 0
    partial_here = 0
    processed: list[dict[str, Any]] = []
    for task in chunk:
        task["attempts"] = int(task.get("attempts") or 0) + 1
        ok, reason = _execute_matrix_task(task, store)
        events_ok = list(task.get("events_ok") or [])
        events_failed = list(task.get("events_failed") or [])
        if not events_failed:
            task["status"] = "done"
            task["error"] = None
            succeeded += 1
        elif events_ok:
            task["status"] = "partial"
            task["error"] = reason
            partial_here += 1
        else:
            task["status"] = "failed"
            task["error"] = reason
            failed_here += 1
        processed.append(
            {
                "station": task.get("station_code", ""),
                "events": len(task.get("events") or []),
                "events_ok": len(events_ok),
                "events_failed": len(events_failed),
                "ok": ok,
                "error": None if ok else reason,
            }
        )

    full = sum(1 for t in tasks if _matrix_station_class(t) == "full")
    partial = sum(1 for t in tasks if _matrix_station_class(t) == "partial")
    empty = sum(1 for t in tasks if _matrix_station_class(t) == "empty")
    remaining = sum(1 for t in tasks if t.get("status") == "pending")
    empty_stations = [
        t.get("station_code", "")
        for t in tasks
        if _matrix_station_class(t) == "empty"
    ]
    partial_stations = [
        t.get("station_code", "")
        for t in tasks
        if _matrix_station_class(t) == "partial"
    ]
    total = matrix.get("total", 0)
    matrix["completed"] = full + partial
    matrix["failed"] = empty
    matrix["partial"] = partial
    matrix["pending"] = remaining
    path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    archive: str | None = None
    export_payload: Any = None
    export_error: str | None = None
    if run_until_complete and remaining == 0:
        # The whole job is done, so Python collects the ZIP itself. After a heavy
        # 260-cell run the 7B model reliably drops the follow-up export_project
        # step and answers with nothing; folding the archive into this one tool
        # call removes that step from the model's plate entirely.
        project_name = str(matrix.get("project") or "")
        if project_name:
            export_payload, export_note = _handle_export_project(
                {"name": project_name}, store
            )
            if not is_error(export_payload):
                archive = export_payload.get("path") or export_payload.get("archive")
            else:
                export_error = export_payload.get("message") or export_note

    if run_until_complete:
        if archive:
            summary = (
                f"✅ МАССОВАЯ ОБРАБОТКА ЗАВЕРШЕНА И СОХРАНЕНА В АРХИВ. "
                f"Станций всего: {total}. Полностью успешных: {full}. "
                f"Частичных: {partial}. Полностью пустых: {empty}. "
                f"Список пустых станций: [{', '.join(empty_stations)}]. "
                f"Архив с данными и графиками успешно собран: {archive}. "
                f"СЛЕДУЮЩИЙ ШАГ: Просто напиши пользователю краткий итоговый "
                f"отчет на основе этих данных. Больше не нужно вызывать никакие "
                f"инструменты."
            )
        else:
            summary = (
                f"✅ МАССОВАЯ ОБРАБОТКА ЗАВЕРШЕНА. Станций всего: {total}. "
                f"Полностью успешных: {full}. Частичных: {partial}. "
                f"Полностью пустых: {empty}. "
                f"Список пустых станций: [{', '.join(empty_stations)}]. "
                f"Не удалось собрать архив: {export_error}. "
                f"СЛЕДУЮЩИЙ ШАГ: сообщи пользователю итог и причину."
            )
    else:
        summary = (
            f"Обработано {len(chunk)}. Успешно: {succeeded}, Ошибок: {failed_here}. "
            f"Осталось задач: {remaining}."
        )
    return {
        "ok": True,
        "completed_run": run_until_complete,
        "processed": len(chunk),
        "succeeded": succeeded,
        "failed": failed_here,
        "partial": partial_here,
        "full": full,
        "empty": empty,
        "empty_stations": empty_stations,
        "partial_stations": partial_stations,
        "remaining": remaining,
        "total": total,
        "progress": f"{full + partial}/{total}",
        "archive": archive,
        "export": export_payload,
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
    """create_task_matrix -> one run_until_complete=True batch -> export_project.

    The full Generator-Executor run, called by :func:`run_agent_with_planner`
    whenever a plan degenerates into per-unit tasks. The batch tool grinds every
    station (and every event inside it) in Python; the model is not in the loop.
    """
    if verbose:
        print(
            f"🏭 Матрица: {len(events)} событий x {len(stations)} станций = "
            f"{len(stations)} задач (проект {project_name!r})."
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

    payload, note = _handle_process_matrix_batch(
        {"matrix_path": path, "run_until_complete": True, "batch_size": MATRIX_BATCH_SIZE},
        store,
    )
    batches = [{"ok": not is_error(payload), "payload": payload, "note": note}]
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
    full = int(payload.get("full", payload.get("succeeded", 0)))
    partial = int(payload.get("partial", 0))
    empty = int(payload.get("empty", payload.get("failed", 0)))
    empty_stations = payload.get("empty_stations") or []
    if verbose:
        print(f"   {payload['summary']}")

    # process_matrix_batch now collects the archive itself once every station is
    # done, so only export here if it somehow did not (keeps the ZIP path in the
    # payload stable for callers that read ``export``).
    export_payload = payload.get("export")
    if not isinstance(export_payload, dict) or is_error(export_payload):
        export_payload, export_note = _handle_export_project({"name": project_name}, store)
    archive = payload.get("archive")
    if not archive:
        archive = export_payload.get("path") if not is_error(export_payload) else export_payload.get("message")
    text = (
        f"Матрица {project_name}: станций {create_payload['total']} — "
        f"полностью успешных {full}, частичных {partial}, пустых {empty}"
        + (f" ({', '.join(empty_stations)})" if empty_stations else "")
        + f". Архив: {archive}."
    )
    return {
        "ok": empty == 0,
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


_PHYSICS_METRICS: dict[str, Any] = {
    "sq_baseline": pengine.extract_sq_baseline,
    "deltas": pengine.calculate_deltas,
    "gic": pengine.calculate_gic_proxy,
    "dst_proxy": pengine.calculate_dst_proxy,
    "ey_coupling": pengine.calculate_coupling_function,
    "detect_ssc": pengine.detect_ssc,
    "burton_coupling": pengine.calculate_burton_coupling,
    "akasofu_epsilon": pengine.calculate_akasofu_epsilon,
    "peak_dhdt": pengine.calculate_peak_dhdt,
    "substorm_onset": pengine.detect_substorm_onset,
    "eej_index": pengine.calculate_eej_index,
}

#: Metrics that read the solar wind and therefore REQUIRE ``solar_wind_handle``.
#: The handler resolves that handle, aligns the frame to the observatory clock
#: with :func:`physics_engine.align_solar_wind`, and hands the aligned frame to
#: the formula as ``solar_wind``. There is no scalar path any more: a number is
#: not a frame and could not be aligned.
_SOLAR_WIND_METRICS = ("ey_coupling", "burton_coupling", "akasofu_epsilon")

#: Metrics that need a second frame (the off-equator reference station), supplied
#: through the flat ``reference_handle`` argument.
_TWO_FRAME_METRICS = ("eej_index",)

#: Spellings a model reaches for that mean the same metric. 'dH_dt' is the
#: physical name of the GIC proxy, so it is accepted and folded onto 'gic'
#: rather than refused as an unknown metric.
_PHYSICS_METRIC_ALIASES: dict[str, str] = {
    "dh_dt": "gic",
    "dh/dt": "gic",
    "gic_proxy": "gic",
    "sq": "sq_baseline",
    "delta": "deltas",
    "ssc": "detect_ssc",
    "ssc_flag": "detect_ssc",
    "dst": "dst_proxy",
    "ey": "ey_coupling",
    "burton": "burton_coupling",
    "burton_dst": "burton_coupling",
    "dst_prediction": "burton_coupling",
    "akasofu": "akasofu_epsilon",
    "epsilon": "akasofu_epsilon",
    "peak_dh_dt": "peak_dhdt",
    "peak_dhdt": "peak_dhdt",
    "substorm": "substorm_onset",
    "substorm_detection": "substorm_onset",
    "eej": "eej_index",
    "equatorial_electrojet": "eej_index",
}

#: Query triggers for every physics metric, built from the registry above so a
#: metric cannot be nameable by the model but invisible to the router.
_METRIC_QUERY_PATTERNS = _build_metric_query_patterns()


def _strict_argument_check(
    tool: str,
    args: dict[str, Any],
    allowed: tuple[str, ...],
    required: tuple[str, ...],
    *,
    handle_example: str = "raw:irt:2024-09-12",
    missing_hint: str | None = None,
) -> dict[str, Any] | None:
    """Refuse hallucinated or missing arguments before a handler reads them.

    A small model invents argument names (``formula``, ``ground_component``,
    ``df1``) for these two tools and then loops: the resulting generic
    ``unknown_frame`` never names the offending key, so the next attempt repeats
    the mistake. This is an allow-list gate, run first:

    * any key not in the tool's schema -> ``unknown_arguments`` listing the keys
      that ARE valid, which is the one hint that lets the model repair the call;
    * a frame descriptor that is missing/empty -> ``missing_required_argument``
      naming it and showing a concrete handle, instead of a downstream
      ``unknown_frame`` from ``store.resolve("")``.

    ``missing_hint`` swaps that second message for a tool whose mandatory
    argument is not a handle at all (a date, say): the generic "copy it from
    frame_handle" wording would send the model looking for a slot that never
    existed.

    Returns ``None`` when the call is well-formed enough to proceed.
    """
    unknown = sorted(k for k in args if k not in allowed)
    if unknown:
        shown = ", ".join(repr(u) for u in unknown)
        return _error(
            "unknown_arguments",
            f"ОШИБКА: инструмент {tool} не принимает аргументы: {shown}.",
            received={k: args.get(k) for k in unknown},
            allowed=list(allowed),
            hint=(
                f"CRITICAL: Arguments like {shown} DO NOT EXIST for this tool. "
                f"The ONLY allowed arguments are: {list(allowed)}. Remove the invalid "
                f"arguments and provide the required '{required[0]}'."
            ),
        )

    missing = [
        k for k in required
        if not (isinstance(args.get(k), str) and args.get(k).strip())
    ]
    if missing:
        named = " / ".join(f"'{k}'" for k in missing)
        return _error(
            "missing_required_argument",
            f"CRITICAL: You forgot the mandatory argument {named}.",
            missing=missing,
            required=list(required),
            allowed=list(allowed),
            hint=missing_hint
            or (
                f"CRITICAL: You forgot the mandatory argument {named}. It MUST be a "
                f"valid frame handle like '{handle_example}'. Copy it EXACTLY from the "
                "'frame_handle' field of the previous tool result; never invent one."
            ),
        )
    return None


#: Columns each metric needs on the frame it actually reads, and the dataset
#: kind that frame is supposed to be. Mirrors the engine's own ``_require_columns``
#: calls -- a looser check here would only postpone the same failure by one round.
#: Solar-wind metrics are absent on purpose: their primary frame is the ground
#: station, and the columns that matter (V, Bz, Np) live in the frame named by
#: ``solar_wind_handle``, so they are checked there instead.
_METRIC_DATA_REQUIREMENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "sq_baseline": ("ground station", ("X", "Y", "Z")),
    "deltas": ("ground station", ("X", "Y", "Z")),
    "gic": ("ground station", ("X", "Y")),
    "dst_proxy": ("ground station", ("X", "Y")),
    "detect_ssc": ("ground station", ("X", "Y")),
    "peak_dhdt": ("ground station", ("X", "Y")),
    "eej_index": ("ground station", ("H",)),
    "substorm_onset": ("indices", ("SML",)),
}

#: Columns a solar-wind frame must carry for the coupling metrics. ``Np`` is
#: only read by Burton, so it is required for that metric alone.
_SOLAR_WIND_COLUMNS = ("V", "Bz")
_SOLAR_WIND_EXTRA = {"burton_coupling": ("Np",)}


def _frame_kind(columns: Any) -> str:
    """Name the dataset a frame looks like, from its columns alone."""
    lowered = {str(c).strip().lower() for c in columns}
    if lowered & {"x", "y", "h", "d", "z"}:
        return "ground station"
    if lowered & {"v", "bz", "np", "bx", "by"}:
        return "solar wind"
    if lowered & {"sml", "smu", "dst", "kp"}:
        return "indices"
    return "unrecognised"


def _data_type_error(
    metric: str,
    descriptor: str,
    actual: str,
    needed: tuple[str, ...],
    hint: str,
) -> dict[str, Any]:
    """The one error that says "wrong dataset", not "wrong column"."""
    return _error(
        "invalid_data_type",
        (
            f"{metric} needs {', '.join(needed)}; the frame {descriptor!r} looks like "
            f"{actual} data and does not have them."
        ),
        metric=metric,
        frame=descriptor,
        data_type=actual,
        expected=needed,
        hint=hint,
    )


def _validate_data_type(
    metric: str, frame: pd.DataFrame, df_descriptor: str, *, role: str = "primary"
) -> dict[str, Any] | None:
    """Refuse a frame of the wrong kind BEFORE the formula reads it.

    The engine's ``missing_columns`` is correct but arrives one round too late and
    reads like a data-quality problem ("missing X"), when the real mistake is a
    contract one: SSC was pointed at a solar-wind frame. Naming the dataset lets
    the model repair the call in one step instead of retrying variants of it.

    ``role`` says which side of the contract ``frame`` is:

    * ``"primary"`` -- the frame named by ``df_descriptor`` (the default). For the
      solar-wind metrics it must be a ground station; the wind frame is checked
      separately with ``role="solar_wind"``.
    * ``"solar_wind"`` -- the frame named by ``solar_wind_handle``.

    Returns ``None`` when the metric can proceed.
    """
    if metric in _SOLAR_WIND_METRICS:
        if role == "solar_wind":
            needed = _SOLAR_WIND_COLUMNS + _SOLAR_WIND_EXTRA.get(metric, ())
            present = {str(c).strip().lower() for c in frame.columns}
            missing = [c for c in needed if c.lower() not in present]
            if not missing:
                return None
            actual = _frame_kind(frame.columns)
            return _data_type_error(
                metric,
                df_descriptor,
                actual,
                needed,
                hint=(
                    f"Solar wind metrics require {', '.join(needed)}. "
                    f"Current frame is {actual}. Pass a solar wind frame as "
                    "'solar_wind_handle' (e.g. 'sw:omni:2024-09-12')."
                ),
            )
        # Primary frame of a coupling metric is the ground station. A wind frame
        # here means the two handles were swapped, which alignment alone cannot
        # explain to the model.
        present = {str(c).strip() for c in frame.columns}
        if {"X", "Y"} & present or "H" in present:
            return None
        actual = _frame_kind(frame.columns)
        return _data_type_error(
            metric,
            df_descriptor,
            actual,
            ("X", "Y"),
            hint=(
                f"{metric} takes the ground station frame as 'df_descriptor' and the "
                f"solar wind as 'solar_wind_handle'. Current frame is {actual}. "
                "Pass a ground station frame."
            ),
        )

    requirement = _METRIC_DATA_REQUIREMENTS.get(metric)
    if requirement is None:
        return None
    expected_kind, needed = requirement
    present = {str(c) for c in frame.columns}
    missing = [c for c in needed if c not in present]
    # EEJ is happy with the horizontal field built from X and Y.
    if metric == "eej_index" and missing and {"X", "Y"} <= present:
        missing = []
    if not missing:
        return None

    actual = _frame_kind(frame.columns)
    if expected_kind == "indices":
        hint = (
            "substorm_onset requires SuperMAG indices (SML). "
            f"Current frame is {actual}. Pass an indices frame, e.g. "
            "fetch_supermag_data(data_type='indices')."
        )
    elif metric == "detect_ssc":
        hint = (
            "SSC requires ground station data. "
            f"Current frame is {actual}. Pass a ground station frame."
        )
    else:
        hint = (
            f"{metric} requires ground station data. "
            f"Current frame is {actual}. Pass a ground station frame."
        )
    return _data_type_error(metric, df_descriptor, actual, needed, hint)


def _resolve_solar_wind_frame(
    args: dict[str, Any], frame: pd.DataFrame, store: FrameStore, metric: str = ""
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Resolve ``solar_wind_handle`` and align it to ``frame``'s clock.

    This is the whole "Automatic Alignment" contract: the caller names a stored
    solar-wind frame, the handler resolves it and calls
    :func:`physics_engine.align_solar_wind`, so the formula never sees a handle
    and never has to guess whether a value is a number or a column name.

    Returns ``(aligned_frame, None)`` or ``(None, error_dict)``. A missing handle
    is reported as ``missing_solar_wind_handle`` -- the one error that names the
    fix, instead of a downstream "V is missing" from deep in the engine.
    """
    handle = args.get("solar_wind_handle")
    if not (isinstance(handle, str) and handle.strip()):
        return None, _error(
            "missing_solar_wind_handle",
            "This metric needs solar-wind data, but 'solar_wind_handle' was not passed.",
            missing=["solar_wind_handle"],
            required=["solar_wind_handle"],
            hint=(
                "Pass 'solar_wind_handle' argument with the handle of solar wind data "
                "(e.g., 'sw:omni:2024-09-12')."
            ),
        )

    sw_frame, sw_err = store.resolve(handle)
    if sw_err:
        return None, sw_err
    if not isinstance(sw_frame, pd.DataFrame):
        return None, _error(
            "unknown_frame",
            f"Handle {handle!r} does not hold a solar-wind frame.",
            available=store.names(),
        )

    # Same pre-flight as the primary frame, but on the wind frame: a ground
    # station passed as 'solar_wind_handle' is named as such here, before
    # alignment turns it into a confusing "missing V" from the engine.
    if metric:
        problem = _validate_data_type(metric, sw_frame, handle, role="solar_wind")
        if problem is not None:
            return None, problem

    aligned = pengine.align_solar_wind(frame, sw_frame)
    if pengine.is_error(aligned):
        return None, aligned
    return aligned, None


def _resolve_reference_frame(
    args: dict[str, Any], store: FrameStore
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Resolve ``reference_handle`` for the two-frame metrics (EEJ).

    Same shape as :func:`_resolve_solar_wind_frame`: the handle is resolved here
    so the engine only ever receives real frames.
    """
    handle = args.get("reference_handle")
    if not (isinstance(handle, str) and handle.strip()):
        return None, _error(
            "missing_reference_handle",
            "This metric needs a second (reference) frame, but 'reference_handle' "
            "was not passed.",
            missing=["reference_handle"],
            required=["reference_handle"],
            hint=(
                "Pass 'reference_handle' with the reference station's frame handle "
                "(e.g., 'raw:abg:2023-09-12')."
            ),
        )
    ref_frame, ref_err = store.resolve(handle)
    if ref_err:
        return None, ref_err
    return ref_frame, None


def _handle_apply_physics_formula(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Run one physical formula and store the enriched frame as a derived slot.

    ONE meta-tool for the whole scientific layer: a small model cannot be trusted
    to remember six signatures, and a new metric should be a table row here, not
    a new schema in the prompt budget. The result is always a frame with the new
    physical columns, so it is plottable and can feed the next tool.

    The contract is flat on purpose. There is no polymorphic ``external_params``
    mapping: a metric that needs another dataset names it with a frame handle
    (``solar_wind_handle`` / ``reference_handle``), this handler resolves it and
    aligns it, and the engine reads named columns. One input shape, one code path.
    """
    strict = _strict_argument_check(
        "apply_physics_formula",
        args,
        allowed=(
            "df_descriptor", "metric", "solar_wind_handle",
            "reference_handle", "window_seconds",
        ),
        required=("df_descriptor",),
    )
    if strict is not None:
        return strict, f"apply_physics_formula refused: {strict['error']}"

    descriptor = str(args.get("df_descriptor") or "").strip()
    metric = str(args.get("metric") or "").strip().lower()
    metric = _PHYSICS_METRIC_ALIASES.get(metric, metric)

    func = _PHYSICS_METRICS.get(metric)
    if func is None:
        return _error(
            "unknown_metric",
            f"metric={metric!r} is not supported.",
            available=sorted(_PHYSICS_METRICS),
            hint="Pick one of the listed metrics.",
        ), f"apply_physics_formula refused metric {metric!r}."

    frame, err = store.resolve(descriptor)
    if err:
        return err, f"apply_physics_formula failed: {err.get('error')}"

    # Pre-flight the frame kind before any formula runs: a wrong dataset is a
    # one-round fix, a "missing X" from the engine is a retry loop.
    problem = _validate_data_type(metric, frame, descriptor)
    if problem is not None:
        return problem, f"apply_physics_formula refused: {problem['error']}"

    if metric in _SOLAR_WIND_METRICS:
        # Every solar-wind metric takes the same aligned frame. Burton also reads
        # the current Dst from a stored indices frame when one is available; the
        # engine falls back to Dst=0 (a quiet start) when it is not.
        solar_wind, sw_err = _resolve_solar_wind_frame(args, frame, store, metric)
        if sw_err is not None:
            return sw_err, f"apply_physics_formula failed: {sw_err.get('error')}"
        alignment = dict(solar_wind.attrs.get("solar_wind_alignment", {}))
        if metric == "burton_coupling":
            indices = None
            for candidate in store.names():
                if candidate.startswith("idx:"):
                    resolved, _ = store.resolve(candidate)
                    if isinstance(resolved, pd.DataFrame) and pengine._column_ci(
                        resolved, "Dst"
                    ):
                        indices = resolved
                        break
            result = func(frame, solar_wind, indices)
        else:
            result = func(frame, solar_wind)
        if not pengine.is_error(result):
            # Keep the alignment facts on the result so the model (and a test)
            # can see that the two clocks really were married, not assumed equal.
            result.attrs["solar_wind_alignment"] = alignment
    elif metric in _TWO_FRAME_METRICS:
        reference, ref_err = _resolve_reference_frame(args, store)
        if ref_err is not None:
            return ref_err, f"apply_physics_formula failed: {ref_err.get('error')}"
        result = func(frame, reference)
    elif metric == "peak_dhdt" and args.get("window_seconds") is not None:
        result = func(frame, args.get("window_seconds"))
    else:
        result = func(frame)
    if pengine.is_error(result):
        return result, f"apply_physics_formula failed: {result.get('error')}"

    slot = store.derive_name(descriptor or "raw", "derived")
    store.put(slot, result)
    note = _physics_note(metric, result, slot)
    # No column dump: the metric plus its headline number is all the model needs
    # to quote, and the derived handle lets it plot or chain the next formula.
    payload = {
        "ok": True,
        "metric": metric,
        "frame_handle": slot,
        "available_handles": store.names(),
        "result": note,
    }
    if "solar_wind_alignment" in result.attrs:
        payload["solar_wind_alignment"] = result.attrs["solar_wind_alignment"]
    return payload, note


#: The column each metric adds, for the summary and the model's benefit.
_PHYSICS_COLUMNS: dict[str, list[str]] = {
    "sq_baseline": ["X_sq", "Y_sq", "Z_sq", "H_sq", "F_sq"],
    "deltas": ["dX", "dY", "dZ", "dH", "F_delta"],
    "gic": ["dH_dt"],
    "dst_proxy": ["dst_proxy"],
    "ey_coupling": ["E_y"],
    "detect_ssc": ["dH_dt", "ssc_flag"],
    "burton_coupling": ["burton_coupling"],
    "akasofu_epsilon": ["akasofu_epsilon"],
    "peak_dhdt": ["dH_dt", "peak_dhdt"],
    "substorm_onset": ["sml_dh_dt"],
    "eej_index": ["eej_index"],
}


def _physics_note(metric: str, result: pd.DataFrame, slot: str) -> str:
    """A one-line, quotable Russian result -- the only thing the LLM must read."""
    attrs = getattr(result, "attrs", {})
    if metric == "sq_baseline":
        base = attrs.get("sq_baseline", {})
        return (
            f"✅ Sq-базовая линия (ночная медиана 00:00–04:00 UT) рассчитана: "
            f"H_sq={base.get('H')} нТ, X_sq={base.get('X')} нТ -> хэндл {slot!r}."
        )
    if metric == "deltas":
        peak = _jsonable(result["F_delta"].abs().max())
        return (
            f"✅ Дельты (dX, dY, dZ, dH, F_delta) рассчитаны; максимальное "
            f"отклонение F_delta={peak} нТ -> хэндл {slot!r}."
        )
    if metric == "gic":
        return (
            f"✅ GIC-прокси рассчитан. Максимальный градиент dH/dt: "
            f"{attrs.get('dH_dt_peak_abs_nT_per_min')} нТ/мин -> хэндл {slot!r}."
        )
    if metric == "dst_proxy":
        low = _jsonable(result["dst_proxy"].min())
        return (
            f"✅ Упрощённый Dst-прокси рассчитан; минимум (пик бури) "
            f"{low} нТ -> хэндл {slot!r}."
        )
    if metric == "ey_coupling":
        info = attrs.get("ey_coupling", {})
        if info.get("mode") == "vector":
            return (
                f"✅ Ey по выровненному солнечному ветру: пик |E_y|="
                f"{info.get('E_y_mV_m_peak_abs')} мВ/м, среднее E_y="
                f"{info.get('E_y_mV_m_mean')} мВ/м (N={info.get('n_finite')}) "
                f"-> хэндл {slot!r}."
            )
        return f"✅ Функция связи Ey рассчитана: E_y = {attrs.get('ey_coupling', {}).get('E_y_mV_m')} мВ/м -> хэндл {slot!r}."
    if metric == "detect_ssc":
        events = attrs.get("ssc_events") or []
        head = ""
        tail = ""
        if events:
            first = events[0]
            when = first.get("timestamp", "время не указано")
            step = first.get("step_nT")
            step_txt = f", шаг {step} нТ" if step is not None else ""
            head = (
                f" Первое: {when} "
                f"({first.get('dH_dt_nT_per_min')} нТ/мин{step_txt})."
            )
            strongest = max(
                events,
                key=lambda e: float(e.get("step_nT") or 0.0),
            )
            if strongest is not first and strongest.get("step_nT") is not None:
                tail = (
                    f" Максимум: {strongest.get('timestamp')} "
                    f"({strongest.get('dH_dt_nT_per_min')} нТ/мин, "
                    f"шаг {strongest.get('step_nT')} нТ)."
                )
        trunc = " Показаны первые 10." if attrs.get("ssc_events_truncated") else ""
        return (
            f"✅ Импульсы H (SI/SSC-кандидаты): {attrs.get('ssc_count', 0)} "
            f"(порог {attrs.get('ssc_threshold_nT_per_min')} нТ/мин)."
            f"{head}{tail}{trunc} "
            f"SSC подтверждается только по солнечному ветру/сети станций. "
            f"-> хэндл {slot!r}."
        )
    if metric == "burton_coupling":
        info = attrs.get("burton", {})
        return (
            f"✅ Функция связи Бёртона: инжекция coupling="
            f"{info.get('coupling_function')} нТ/ч; "
            f"dDst/dt={info.get('dDst_dt_nT_per_h')} нТ/ч; "
            f"Dst через 1 ч = {info.get('predicted_dst')} нТ "
            f"(старт Dst={info.get('Dst_input')} нТ) -> хэндл {slot!r}."
        )
    if metric == "akasofu_epsilon":
        info = attrs.get("akasofu", {})
        return (
            f"✅ Акасофу: epsilon={info.get('epsilon_W')} Вт "
            f"({info.get('epsilon_erg_s')} эрг/с), V={info.get('V_kms')} км/с, "
            f"B={info.get('B_total_nT')} нТ -> хэндл {slot!r}."
        )
    if metric == "peak_dhdt":
        info = attrs.get("peak_dhdt", {})
        return (
            f"✅ Пиковый dH/dt за {info.get('window_seconds')} с: "
            f"{info.get('peak_dhdt_nT_min')} нТ/мин, уровень "
            f"{info.get('warning_level')} -> хэндл {slot!r}."
        )
    if metric == "substorm_onset":
        info = attrs.get("substorm", {})
        if info.get("onset_time"):
            return (
                f"✅ Начало суббури: {info.get('onset_time')}, падение SML "
                f"{info.get('sml_drop')} нТ, уверенность {info.get('confidence')} "
                f"-> хэндл {slot!r}."
            )
        return f"✅ Суббуря не обнаружена в окне -> хэндл {slot!r}."
    if metric == "eej_index":
        info = attrs.get("eej", {})
        return (
            f"✅ EEJ-индекс: пик {info.get('eej_nT')} нТ, среднее "
            f"{info.get('eej_mean_nT')} нТ -> хэндл {slot!r}."
        )
    return f"✅ {metric} рассчитан -> хэндл {slot!r}."


def _handle_calculate_eej_index(
    args: dict[str, Any], store: FrameStore
) -> tuple[Any, str]:
    """Fetch both stations and return the equatorial-electrojet H difference.

    EEJ is a *two-station* product: the equatorial station's H minus a station
    6-8 degrees away in latitude, so the Sq variation largely cancels. Feeding one
    frame to ``apply_physics_formula`` could never express that, so this tool owns
    the whole job -- it downloads both days, aligns them, and calls the engine.
    """
    strict = _strict_argument_check(
        "calculate_eej_index",
        args,
        allowed=("equatorial_station", "reference_station", "start_date", "end_date"),
        required=("equatorial_station", "reference_station", "start_date"),
        handle_example="IRT",
        missing_hint=(
            "CRITICAL: calculate_eej_index needs 'equatorial_station' and "
            "'reference_station' (three-letter IAGA codes, e.g. 'IRT' and 'ABG') "
            "plus 'start_date' (YYYY-MM-DD)."
        ),
    )
    if strict is not None:
        return strict, f"calculate_eej_index refused: {strict['error']}"

    equator = str(args.get("equatorial_station") or "").strip().upper()
    reference = str(args.get("reference_station") or "").strip().upper()
    start = str(args.get("start_date") or "").strip()
    end = str(args.get("end_date") or start).strip() or start

    def _fetch(code: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
        if OFFLINE:
            return _offline_frame(), None
        fetched = loader.fetch_observatory_data(
            station_code=code,
            start_date=start,
            end_date=end,
            data_type=args.get("data_type") or "definitive",
            samples_per_day=args.get("samples_per_day") or "Minute",
        )
        if loader.is_error(fetched):
            return None, fetched
        return fetched, None

    eq_frame, eq_err = _fetch(equator)
    if eq_err is not None:
        return eq_err, f"calculate_eej_index failed fetching {equator}: {eq_err.get('error')}"
    ref_frame, ref_err = _fetch(reference)
    if ref_err is not None:
        return ref_err, f"calculate_eej_index failed fetching {reference}: {ref_err.get('error')}"

    result = pengine.calculate_eej_index(eq_frame, ref_frame)
    if pengine.is_error(result):
        return result, f"calculate_eej_index failed: {result.get('error')}"

    slot = FrameStore._normalise(f"derived:eej:{equator.lower()}_{reference.lower()}:{start}")
    store.put(slot, result)
    info = (result.attrs or {}).get("eej", {})
    note = (
        f"✅ EEJ-индекс ({equator} − {reference}): пик {info.get('eej_nT')} нТ, "
        f"среднее {info.get('eej_mean_nT')} нТ -> хэндл {slot!r}."
    )
    payload = {
        "ok": True,
        "metric": "eej_index",
        "equatorial_station": equator,
        "reference_station": reference,
        "frame_handle": slot,
        "available_handles": store.names(),
        "result": note,
    }
    return payload, note


#: Components plot_overlay can draw straight from an observatory frame (H/D/I
#: are derived on demand). Used to pick the usable entry out of a mixed list.
_OVERLAY_COMPONENTS = ("X", "Y", "Z", "F", "H", "D", "I")


def _overlay_component(args: dict) -> str:
    """The single component to overlay, from 'component' or the 'components' alias.

    The field log showed a model sending ``components: ['Bz_south','H']`` (the
    plot_comparison/plot_components shape). Accept the list and prefer the first
    name this tool can actually draw, so a stray solar-wind label next to a real
    component (``['Bz_south','H']`` -> ``H``) is not a dead end.
    """
    candidates: list[Any] = []
    raw = args.get('component')
    if isinstance(raw, (list, tuple)):
        candidates.extend(raw)
    elif raw not in (None, ""):
        candidates.append(raw)
    alias = args.get('components')
    if isinstance(alias, str):
        candidates.append(alias)
    elif isinstance(alias, (list, tuple)):
        candidates.extend(alias)
    for candidate in candidates:
        text = str(candidate or '').strip().upper()
        if text in _OVERLAY_COMPONENTS:
            return text
    for candidate in candidates:
        text = str(candidate or '').strip().upper()
        if text:
            return text
    return 'H'


def _handle_plot_overlay(args, store):
    raw_stations = args.get('stations')
    if isinstance(raw_stations, str):  # a lone code, not a list
        raw_stations = [raw_stations]
    stations = [
        str(s).strip().upper() for s in (raw_stations or []) if str(s).strip()
    ]
    dates = args.get('dates') or ([args['date']] if args.get('date') else [])
    if isinstance(dates, str):
        dates = [dates]
    dates = [str(d).strip() for d in dates if str(d).strip()]
    component = _overlay_component(args)
    time_system = str(args.get('time_system') or 'UT').strip().upper()
    if not stations:
        # A model that copied plot_comparison's signature passes df1/df2 here.
        # Say so explicitly, or it re-sends the same broken call until the
        # budget runs out -- exactly the loop seen in the field log.
        wrong = [
            key
            for key in ('df1', 'df2', 'df', 'handle', 'frame_handle')
            if args.get(key) not in (None, "", [], {})
        ]
        example = {
            "stations": ["IRT"],
            "dates": ["2024-09-12"],
            "component": "H",
            "title": "Сравнение",
        }
        if wrong:
            message = (
                "ОШИБКА: Отсутствует или неверен аргумент 'stations'. Вы передали "
                f"{'/'.join(wrong)}, но этот инструмент принимает 'stations' "
                "(список кодов станций, например ['IRT']) и компонент "
                "('component', например 'H'; 'components' = ['H'] тоже "
                "принимается). Пожалуйста, исправь аргументы и вызови инструмент "
                "снова."
            )
        else:
            message = (
                "ОШИБКА: Аргумент 'stations' обязателен. Это должен быть список "
                "кодов станций, например ['IRT']. Пример вызова: "
                "{\"stations\": [\"IRT\"], \"dates\": [\"2024-09-12\"], "
                "\"component\": \"H\"}."
            )
        return (
            _error(
                'invalid_input',
                message,
                received={
                    key: args.get(key)
                    for key in ('stations', 'dates', 'df1', 'df2', 'components')
                    if key in args
                },
                required=['stations', 'dates'],
                example=example,
                hint=(
                    "plot_overlay takes 'stations' (a list of IAGA codes) and "
                    "'component'; it does NOT take 'df1'/'df2'."
                ),
            ),
            'missing stations',
        )
    if not dates:
        return (
            _error(
                'invalid_input',
                "ОШИБКА: Аргумент 'dates' обязателен. Это должен быть список дат "
                "YYYY-MM-DD, например ['2024-09-12'].",
                required=['stations', 'dates'],
                hint="Pass 'dates': ['2024-09-12'] (or 'date': '2024-09-12').",
            ),
            'missing dates',
        )
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


def _resolve_column_name(df: pd.DataFrame, name: Any) -> str | None:
    """The exact column in *df* matching *name* ignoring case, else ``None``."""
    if name in df.columns:
        return str(name)
    target = str(name).lower()
    for column in df.columns:
        if str(column).lower() == target:
            return str(column)
    return None


def _merge_solar_wind_columns(
    aligned: pd.DataFrame, sw_frame: pd.DataFrame, wanted: Sequence[str]
) -> pd.DataFrame:
    """Carry extra solar-wind columns onto the aligned observatory clock.

    :func:`physics_engine.align_solar_wind` aligns the coupling pair (``V`` and
    ``Bz``) and nothing else, because that pair is all ``E_y`` needs. A comparison
    chart may also want ``By``, ``Bx`` or ``Np``, so those columns are resampled
    with the same rule the engine uses -- linear in time, never extrapolated past
    the solar-wind coverage -- purely for drawing.
    """
    stamps = pd.to_datetime(aligned["timestamp"], errors="coerce", utc=True)
    valid = stamps.dropna()
    if valid.empty:
        return aligned
    target = pd.DatetimeIndex(stamps)
    unique = pd.DatetimeIndex(sorted(set(valid)))
    out = aligned.copy()
    for raw in wanted:
        column = _resolve_column_name(sw_frame, raw)
        if column is None or _resolve_column_name(out, column) is not None:
            continue
        series = pd.Series(
            pd.to_numeric(sw_frame[column], errors="coerce").to_numpy(dtype="float64"),
            index=pd.DatetimeIndex(
                pd.to_datetime(sw_frame["timestamp"], errors="coerce", utc=True)
            ),
        )
        series = series[series.index.notna()]
        if series.empty:
            continue
        series = series.groupby(level=0).last()
        union = unique.union(series.index).sort_values()
        filled = series.reindex(union).interpolate(
            method="time", limit=30, limit_area="inside"
        )
        out[column] = filled.reindex(target).to_numpy(dtype="float64")
    return out


def _handle_plot_space_weather_comparison(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    """Compare a ground magnetometer component with the solar wind on one chart.

    This is the tool the field log was missing: the agent tried to pass two
    different frames (ground and OMNI) to ``plot_overlay``, which only knows
    stations, and looped. Here the two handles are resolved, the L1 data is
    aligned to the observatory clock by :func:`physics_engine.align_solar_wind`,
    and the aligned frame is handed to the dual-axis plotter.
    """
    strict = _strict_argument_check(
        "plot_space_weather_comparison",
        args,
        allowed=(
            "ground_df_descriptor",
            "solar_wind_df_descriptor",
            "ground_component",
            "solar_wind_components",
            "title",
            "filename",
        ),
        required=("ground_df_descriptor", "solar_wind_df_descriptor"),
    )
    if strict is not None:
        return strict, f"plot_space_weather_comparison refused: {strict['error']}"

    ground_desc = str(args.get("ground_df_descriptor") or "").strip()
    sw_desc = str(args.get("solar_wind_df_descriptor") or "").strip()
    ground_component = str(args.get("ground_component") or "H").strip() or "H"
    raw_sw = args.get("solar_wind_components")
    if isinstance(raw_sw, str):
        raw_sw = [raw_sw]
    sw_components = [str(c).strip() for c in (raw_sw or ["Bz", "V"]) if str(c).strip()]
    if not sw_components:
        sw_components = ["Bz", "V"]

    # Dst is the *global* Kyoto ring-current index, not a station component; a
    # silent substitution would give a correct-looking but physically wrong chart.
    if ground_component.strip().lower() == "dst":
        return (
            _error(
                "invalid_input",
                "ОШИБКА: Dst — это глобальный индекс Kyoto WDC. Для локального "
                "сравнения используйте dst_proxy (рассчитывается через "
                "apply_physics_formula(metric='dst_proxy')).",
                requested=ground_component,
                allowed=["H", "dst_proxy", "dH_dt"],
                hint=(
                    "fetch_observatory_data -> apply_physics_formula("
                    "metric='dst_proxy'), then call this tool with "
                    "ground_component='dst_proxy' and the derived handle."
                ),
            ),
            "ground_component='Dst' is the global index, not a local component.",
        )

    # The solar wind must be downloaded BEFORE this tool can align anything; the
    # field log showed the model calling the plot first and inventing a handle.
    if not any(name.startswith("sw:") for name in store.names()):
        return (
            _error(
                "missing_dependency",
                "ОШИБКА: Для построения графика сравнения с солнечным ветром "
                "необходимо сначала скачать данные через fetch_solar_wind. "
                "Вызовите этот инструмент перед plot_space_weather_comparison.",
                hint=(
                    "fetch_solar_wind(start_date='2024-09-12') returns the "
                    "'sw:omni:2024-09-12' handle this tool expects."
                ),
                available=store.names(),
            ),
            "no solar-wind frame in the store; call fetch_solar_wind first.",
        )

    if ground_desc == sw_desc:
        return (
            _error(
                "same_dataframe",
                "The ground and solar-wind handles are the same frame; a comparison "
                "needs a magnetometer frame and a separate solar-wind frame.",
                requested={"ground": ground_desc, "solar_wind": sw_desc},
                available=store.names(),
            ),
            "plot_space_weather_comparison refused: both handles are the same frame.",
        )

    notes: list[str] = []
    ground, err = _resolve_plottable(store, ground_desc, [ground_component], notes)
    if err:
        return err, f"plot_space_weather_comparison failed (ground): {err.get('error')}"
    sw_frame, err = store.resolve(sw_desc)
    if err:
        return err, f"plot_space_weather_comparison failed (solar wind): {err.get('error')}"

    # align_solar_wind puts V and Bz on the observatory clock; anything else the
    # chart was asked for is resampled the same way for display.
    aligned = pengine.align_solar_wind(ground, sw_frame)
    if pengine.is_error(aligned):
        return aligned, f"plot_space_weather_comparison failed: {aligned.get('error')}"
    aligned = _merge_solar_wind_columns(aligned, sw_frame, sw_components)

    result = plotter.plot_space_weather_comparison(
        aligned,
        ground_component=ground_component,
        solar_wind_components=sw_components,
        title=args.get("title") or f"{ground_component} vs {'/'.join(sw_components)}",
        filename=args.get("filename"),
    )
    if plotter.is_error(result):
        return result, f"plot_space_weather_comparison failed: {result.get('error')}"

    filed = _file_into_project(store, result, [ground_desc, sw_desc], "space_weather")
    payload = {
        "ok": True,
        "path": result,
        "plot_path": result,
        "ground_component": ground_component,
        "solar_wind_components": sw_components,
        "ground_handle": ground_desc,
        "solar_wind_handle": sw_desc,
        "aligned_rows": int(len(aligned)),
    }
    visible = [c for c in sw_components if _resolve_column_name(aligned, c) is not None]
    if visible:
        payload["solar_wind_drawn"] = visible
    if notes:
        payload["auto"] = notes
    if filed:
        payload["filed_into_project"] = filed
    summary = notes + [
        f"Earth-vs-solar-wind chart ({ground_component} vs {', '.join(sw_components)}) "
        f"saved to {result}"
    ]
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
    "fetch_solar_wind": _handle_fetch_solar_wind,
    "fetch_geomagnetic_indices": _handle_fetch_geomagnetic_indices,
    "get_noaa_space_weather_context": _handle_noaa_context,
    "get_iswa_perturbation_map": _handle_iswa_perturbation_map,
    "get_iswa_forecast": _handle_iswa_forecast,
    "fetch_supermag_data": _handle_fetch_supermag,
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
    "plot_space_weather_comparison": _handle_plot_space_weather_comparison,
    "apply_physics_formula": _handle_apply_physics_formula,
    "calculate_eej_index": _handle_calculate_eej_index,
}


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def code_revision() -> str:
    """Short git hash of the checkout that is actually loaded, or ``"unknown"``.

    A notebook kernel caches imported modules in ``sys.modules``. After a
    ``git pull`` the files on disk can be new while the running ``agent_core``
    and ``physics_engine`` are still the old ones, and the log then looks like a
    physics regression rather than a stale process. Printing this at the start
    of a verbose run makes that failure visible: the hash must match the commit
    that was just pulled.
    """
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


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

    if verbose:
        print(f"[CODE] revision={code_revision()} module={Path(__file__).resolve()}")

    prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    # Never hand a small model all 29 schemas: a free-form question gets the
    # tools its intent needs. Planner subtasks pass an explicit, already narrow
    # list; the free-form path routes by query here. Under verbose the routing
    # evidence is printed, so a failed run can be diagnosed from the log alone
    # (which metrics were recognised, whether it was read as a bulk job, how
    # many tokens the schemas cost).
    if tools is not None:
        schemas = tools
        selection = None
    else:
        schemas = select_tools_for_query(user_query)
        selection = explain_tool_selection(user_query)
    if verbose and selection is not None:
        print(
            "[TOOL SELECTION] "
            f"{selection['tools']} metrics={selection['metrics']} "
            f"bulk={selection['bulk']} est_tokens={selection['est_tokens']}"
        )
    store = store or FrameStore()
    plots: list[str] = []
    log: list[dict[str, Any]] = []
    scratchpad = Scratchpad()
    scratchpad.reset()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query},
    ]

    spent = 0
    stop_reason = "no_tool_calls"
    format_retries = 0

    for round_no in range(1, max_rounds + 1):
        # Drop the oldest whole tool exchanges before the next call. The
        # Scratchpad keeps every live handle and completed step, so the model
        # loses stale payload text, not the state it needs to continue.
        removed = _trim_history(messages)
        if removed:
            scratchpad.note_elided(removed)
            print(
                f"[HISTORY] elided {removed} older messages; "
                f"{len(messages)} remain in the context"
            )
        # Inject the scratchpad into the system message before every LLM call.
        # Rebuilt from the immutable base prompt each round, so the dynamic hint
        # replaces the previous one instead of accumulating and growing the
        # context window (the schema diet exists precisely to keep it small).
        injection = scratchpad.to_prompt_injection()
        messages[0] = {"role": "system", "content": f"{prompt}\n\n{injection}"}
        # Unconditional: the field log showed the model ignoring a self-correction
        # we believed was injected, so the exact injected text is printed to tell
        # "not injected" apart from "injected and ignored".
        print(f"\n[DEBUG] SCRATCHPAD INJECTION:\n{injection}\n")
        if verbose:
            print(
                f"[SCRATCHPAD] Step {round_no}/{max_rounds}. "
                f"Handle: {scratchpad.current_handle or 'none'}. "
                f"Handles: {len(scratchpad.handles)}. "
                f"Completed: {', '.join(scratchpad.completed_steps) or 'none'}."
            )
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
                    # Turns "unknown_tool" into an actionable next call: a
                    # hallucinated physics name is answered with the exact metric.
                    hint=_self_correction_hint(name),
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
            # Keep every live handle, not just the active one. A wind fetch is
            # auxiliary and never becomes active, yet a second coupling metric
            # needs its exact name; the registry is what carries it to the next
            # round instead of relying on the model guessing the date.
            if entry["ok"] and isinstance(payload, dict):
                produced: list[str] = []
                frame_handle = payload.get("frame_handle")
                if isinstance(frame_handle, str):
                    produced.append(frame_handle)
                frame_handles = payload.get("frame_handles")
                if isinstance(frame_handles, list):
                    produced.extend(h for h in frame_handles if isinstance(h, str))
                metric = str(args.get("metric") or payload.get("metric") or "")
                for produced_handle in dict.fromkeys(produced):
                    scratchpad.register_handle(
                        produced_handle,
                        tool=name,
                        payload=payload,
                        metric=metric,
                        round_no=round_no,
                    )
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
                # The newest handle becomes the active one, so the model always
                # sees where the pipeline currently stands (raw -> derived -> ...).
                # Space-weather frames (sw:/idx:) are auxiliary *inputs*, not the
                # working magnetometer frame, so they must not hijack the slot a
                # follow-up sq_baseline/plot call will reach for.
                family = payload["frame_handle"].split(":", 1)[0]
                if family not in _AUX_FRAME_FAMILIES:
                    scratchpad.update_handle(payload["frame_handle"])

            if entry["ok"] and name in PLOT_TOOLS and isinstance(payload.get("path"), str):
                plots.append(payload["path"])
                entry["plot"] = payload["path"]

            # Track completed actions and any failure so the next round sees the
            # progress and can self-correct instead of repeating a dead handle.
            if entry["ok"]:
                step = _SCRATCHPAD_STEP_BY_TOOL.get(name)
                if step:
                    scratchpad.add_completed_step(step)
                elif name == "apply_physics_formula":
                    metric = str(args.get("metric") or "").strip()
                    if metric:
                        scratchpad.add_completed_step(f"physics_{metric}")
            else:
                entry["error"] = payload.get("error")
                entry["message"] = payload.get("message")
                # Repair the error hint with the concrete active handle, so a
                # "derived:" hallucination is answered with the real "raw:" name.
                hint = payload.get("hint") or ""
                if payload.get("error") == "unknown_frame" and scratchpad.current_handle:
                    hint = f"Использовал неверный handle, актуальный: {scratchpad.current_handle}"
                scratchpad.add_error(name, payload.get("error", ""), hint)
                # Feed the loop breaker: an identical failing call repeated on
                # the next rounds trips the [CRITICAL LOOP DETECTED] warning.
                scratchpad.add_failure(name, _args_signature(args), entry["error"])

            log.append(entry)
            if verbose:
                status = "ok" if entry["ok"] else f"FAILED ({entry.get('error')})"
                print(f"[agent] round {round_no}: {name}({_short(args)}) -> {status}")
                # The note is the payload the model actually reads. Without it
                # the log shows only "ok", and a wrong magnitude (epsilon in kW,
                # a positive Dst forecast) is invisible until the model repeats
                # it in prose. Print the evidence, not just the status.
                note = entry.get("note")
                if not note and not entry["ok"]:
                    note = entry.get("message")
                if note:
                    print(f"[NOTE] {str(note)[:500]}")
                print(
                    f"[SCRATCHPAD] Step {round_no}/{max_rounds}. "
                    f"Handle: {scratchpad.current_handle or 'none'}. "
                    f"Handles: {len(scratchpad.handles)}. "
                    f"Completed: {', '.join(scratchpad.completed_steps) or 'none'}. "
                    f"Errors: {len(scratchpad.errors)}."
                )

            # Short-circuit: a finished matrix run already carries the final
            # human-readable summary and the archive path. Feeding that long
            # payload back to a 7B model overflows its context and yields an
            # empty answer (the "Нет текстового ответа" seen in Kaggle), so hand
            # the tool's own summary to the user directly instead of asking the
            # model to rephrase it.
            if (
                name == "process_matrix_batch"
                and entry["ok"]
                and isinstance(payload, dict)
                and (payload.get("completed_run") is True or payload.get("archive"))
            ):
                text = str(payload.get("summary") or "").strip()
                if not text and isinstance(payload.get("export"), dict):
                    text = str(payload["export"].get("message") or "").strip()
                if not text:
                    text = "Массовая обработка завершена."
                if verbose:
                    print(
                        f"[agent] round {round_no}: matrix completed, "
                        "short-circuiting to the tool summary (no extra LLM call)"
                    )
                return {
                    "ok": True,
                    "text": text,
                    "plots": plots,
                    "tool_calls": log,
                    "rounds": round_no,
                    "stop_reason": "matrix_completed_short_circuit",
                }

            messages.append({"role": "tool", "content": _compact_tool_message(payload)})

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



