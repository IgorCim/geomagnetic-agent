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

import pandas as pd

import geomag_analyzer as analyzer
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

MAX_TOOL_CALLS = 7         # tool executions per user query
MAX_ROUNDS = 9             # model round-trips per user query

OFFLINE = os.environ.get("GEOMAG_OFFLINE", "").strip() not in ("", "0", "false")

# Authoritative, read out of the live INTERMAGNET registry (Edinburgh GIN +
# Kyoto WDC, 327 stations). Hand-written guesses are wrong more often than not:
# ABK is Abisko/Sweden, SPB is not St Petersburg, Moscow is MOS not MOW.
STATION_HINT = (
    "IRT=Иркутск, MOS=Москва, NVS=Новосибирск, SPG=Санкт-Петербург, "
    "YAK=Якутск, KHB=Хабаровск, MGD=Магадан, VLA=Владивосток, "
    "PET=Паратунка (Камчатка), ARS=Арти, BOX=Борок, TIK=Тикси"
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
    "4. Порядок работы: сначала fetch_observatory_data, потом "
    "calculate_derived_components, потом get_statistics / detect_anomalies, "
    "потом plot_components / plot_comparison. Не перескакивай шаги.\n"
    '5. Вместо DataFrame инструментам передавай строковый дескриптор: "raw" — данные '
    'из fetch, "derived" — данные с посчитанными H/D/I, "anomalies" — результат '
    "detect_anomalies.\n"
    "6. Если инструмент вернул {\"ok\": false} — сообщи пользователю причину из поля "
    "message и предложи, что делать. Не подставляй свои цифры вместо ошибки.\n"
    "7. В финальном ответе перечисли построенные графики и их файлы.\n"
    "8. Ты можешь вызвать несколько инструментов подряд, прежде чем ответить.\n"
    "9. ВАЖНО: Ты НЕ являешься API-сервером. НИКОГДА не отвечай JSON-объектом, "
    "Python-словарём или строкой вида {'content': ..., 'tool_calls': ...}. "
    "Ответ такого вида считается ошибкой, данные не появятся.\n"
    "10. ФОРМАТ ОТВЕТА СТРОГО ТАК, ДВА ВАРИАНТА И НИКАКИХ ДРУГИХ:\n"
    "    (а) чтобы вызвать инструмент — выведи ТОЛЬКО блок тегов, без "
    "слов вокруг, без пояснений, без markdown:\n"
    '        <tool_call>{"name": "fetch_observatory_data", "arguments": '
    '{"station_code": "IRT", "start_date": "2024-09-10", "end_date": "2024-09-10"}}</tool_call>\n'
    "    (б) когда все данные получены и инструменты больше не нужны — обычный "
    "текст на русском, БЕЗ тегов, БЕЗ фигурных скобок, БЕЗ кавычек вокруг "
    "имён инструментов.\n"
    "    Если сомневаешься между (а) и (б) — бери (а): лишний вызов "
    "инструмента дешевле, чем выдуманный ответ."
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
    "enum": ["raw", "derived", "anomalies"],
    "description": (
        "Data descriptor returned by a previous tool: 'raw' = data from "
        "fetch_observatory_data, 'derived' = raw + H/D/I, 'anomalies' = result of "
        "detect_anomalies."
    ),
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_observatory_data",
            "description": (
                "Download geomagnetic data for one INTERMAGNET observatory. "
                "Call this first, before any analysis. Returns the row count and "
                "the available columns. Values are in nT, timestamps are UTC."
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
                    "df1": {**_FRAME_ARG, "description": "Descriptor of the first day."},
                    "df2": {**_FRAME_ARG, "description": "Descriptor of the second day."},
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
    """Holds real DataFrames so they never have to travel through JSON."""

    def __init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}

    def put(self, name: str, df: pd.DataFrame) -> None:
        self._frames[name] = df

    def get(self, handle: Any) -> pd.DataFrame | None:
        if isinstance(handle, pd.DataFrame):
            return handle
        if isinstance(handle, str):
            frame = self._frames.get(handle.strip().lower())
            if frame is not None:
                return frame
        return None

    def names(self) -> list[str]:
        return sorted(self._frames)

    def resolve(self, handle: Any) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
        frame = self.get(handle)
        if frame is None:
            return None, _error(
                "unknown_frame",
                f"No data under handle {handle!r}.",
                available=self.names() or ["(none yet - call fetch_observatory_data first)"],
                hint='Use "raw" after fetch_observatory_data, "derived" after calculate_derived_components.',
            )
        return frame, None


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
    return summary


def _offline_frame() -> pd.DataFrame:
    """Two synthetic days, so the offline demo needs no network at all."""
    return plotter._synthetic_day(10, seed=7)


# --------------------------------------------------------------------------- #
# tool handlers
# --------------------------------------------------------------------------- #
Handler = Callable[[dict[str, Any], FrameStore], tuple[Any, str]]


def _first_present(df: pd.DataFrame, store: FrameStore, requested: str) -> str:
    """Pick the richest handle that actually has the requested column."""
    if df is not None and requested in df.columns:
        return "anomalies" if store.get("anomalies") is df else "derived"
    for handle in ("derived", "raw"):
        candidate = store.get(handle)
        if candidate is not None and requested in candidate.columns:
            return handle
    return "raw"


def _handle_fetch(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    if OFFLINE:
        frame = _offline_frame()
        store.put("raw", frame)
        summary = _summarise_frame(frame, store, "raw")
        summary["offline"] = True
        return summary, f"Offline mode: synthesised {len(frame)} rows as handle 'raw'."

    result = loader.fetch_observatory_data(
        station_code=args.get("station_code"),
        start_date=args.get("start_date"),
        end_date=args.get("end_date"),
        data_type=args.get("data_type") or "definitive",
        samples_per_day=args.get("samples_per_day") or "Minute",
    )
    if loader.is_error(result):
        return result, f"fetch failed: {result.get('error')}"
    store.put("raw", result)
    summary = _summarise_frame(result, store, "raw")
    summary["station"] = args.get("station_code")
    summary["publication_state"] = (result.attrs or {}).get("publication_state")
    summary["source"] = (result.attrs or {}).get("source")
    return summary, f"Fetched {len(result)} rows for {args.get('station_code')} -> handle 'raw'."


def _handle_derive(args: dict[str, Any], store: FrameStore) -> tuple[Any, str]:
    frame, err = store.resolve(args.get("df"))
    if err:
        return err, f"calculate_derived_components failed: {err['error']}"

    components = args.get("components") or ["H", "D", "I"]
    result = analyzer.calculate_derived_components(frame, components=components)
    if analyzer.is_error(result):
        return result, f"calculate_derived_components failed: {result.get('error')}"

    store.put("derived", result)
    summary = _summarise_frame(result, store, "derived")
    summary["computed"] = [c for c in ("H", "D", "I") if c in result.columns]
    return summary, f"Computed {', '.join(summary['computed'])} -> handle 'derived'."


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
    return result, f"Statistics for {', '.join(result)}."


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
    store.put("anomalies", result)
    summary = _summarise_frame(result, store, "anomalies")
    summary["component"] = component
    summary["n_flagged"] = int(len(result))
    return summary, f"Flagged {len(result)} samples of {component} -> handle 'anomalies'."


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
            hint="Call fetch_observatory_data twice for two different dates and keep both handles.",
        )
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
                "text": text or "Не удалось получить текстовый ответ от модели.",
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
            log.append(
                {
                    "round": round_no,
                    "tool": "(refused)",
                    "arguments": {},
                    "ok": False,
                    "error": "tool_budget_exhausted",
                    "message": f"Budget of {max_tool_calls} tool calls reached; "
                    "the model was told to answer instead.",
                }
            )
            if verbose:
                print(f"[agent] round {round_no}: budget exhausted, forcing an answer")
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
            # Surface the diagnostic fields, not just the code: without these a
            # malformed call in Colab reports "malformed_tool_call" and nothing
            # else, which is what sent the previous two fixes down blind paths.
            for key in ("message", "received", "reason", "hint"):
                if isinstance(payload, dict) and payload.get(key) is not None:
                    entry[key] = payload[key]
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
        "text": (leftover or content or "").strip()
        or "Модель не сформулировала ответ в пределах отведённого числа шагов.",
        "plots": plots,
        "tool_calls": log,
        "rounds": max_rounds,
        "stop_reason": "round_limit",
    }


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
