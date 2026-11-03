"""Bounded session facts and request checks, independent of model prose.

No network or model access. Facts enter only through successful tool results.
This is in-memory browser-session state, not a durable conversation database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re
import math
import ast
from typing import Any

HANDLE = re.compile(r"\b(?:raw|derived|anomalies|anomaly|math|formula|sw|idx):[a-z0-9_-]+:\d{4}-\d{2}-\d{2}\b", re.I)
DATES = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
FRAME_ARGS = {"df_descriptor", "df1", "df2", "ground_df_descriptor", "solar_wind_df_descriptor", "solar_wind_handle"}
FETCH = {"fetch_observatory_data", "fetch_solar_wind", "fetch_geomagnetic_indices", "fetch_supermag_data"}
EXCLUDE_CODES = {"OMNI", "NOAA", "ISWA", "DST", "IAGA", "HTML", "GSM", "UTC", "ZIP", "APIX"}
CALCULATION_TOOLS = {"calculate_derived_components", "calculate_baseline", "calculate_derived_math",
                     "evaluate_custom_formula", "apply_physics_formula", "detect_anomalies"}


def capture_calculation(entry: dict, name: str, payload: Any) -> None:
    """Preserve only known calculation-result fields, never model prose or credentials."""
    if name not in CALCULATION_TOOLS or not isinstance(payload, dict):
        return
    keys = ("metric", "component", "computed", "definitions", "baseline", "mode", "window", "night_hours",
            "n_samples", "n_finite", "unit", "delta", "max_delta", "mean_delta", "peak_rate",
            "formula", "min", "max", "mean", "column", "n_flagged", "sigma_threshold", "result", "limitations")
    entry["calculation"] = {k: payload[k] for k in keys if k in payload}


def station_in(question: str) -> str | None:
    handles = HANDLE.findall(question)
    ground = [h.split(":")[1] for h in handles if h.lower().startswith(("raw:", "derived:"))]
    prefixed = re.findall(r"\b(?:станц\w*|обсерватор\w*|station|загрузи|загрузить)\s+([a-z]{3,4})\b", question, re.I)
    codes = ground or [c.lower() for c in re.findall(r"\b[A-Z]{3,4}\b", question) + prefixed if c.upper() not in EXCLUDE_CODES]
    return codes[0] if len(set(codes)) == 1 else None


@dataclass
class Request:
    query: str
    date: str | None = None
    fetch_date: str | None = None
    station: str | None = None
    handle: str | None = None
    no_fetch: bool = False
    clarification: str = ""
    wants_stats: bool = False
    wants_plot: bool = False
    target: str | None = None
    components: tuple[str, ...] = ()
    wants_derived: bool = False
    routine: bool = False
    wants_local_time: bool = False
    calculation_tools: tuple[str, ...] = ()
    requested_metric: str | None = None
    baseline_mode: str | None = None
    observed_baselines: dict = field(default_factory=dict)
    requested_formula: str | None = None
    baseline_hours: tuple[int, int] | None = None

    def should_use_facts(self, log: list[dict]) -> bool:
        if self.routine or self.calculation_tools:
            return True
        interpretation = re.search(r"проанализ|объясн|интерпрет|почему|\b(?:interpret|explain|analy[sz])", self.query, re.I)
        return not interpretation and any(
            e.get("tool") in CALCULATION_TOOLS or e.get("tool") == "get_statistics" for e in log)

    def observe_result(self, name: str, args: dict, payload: Any) -> None:
        if name == "calculate_baseline" and isinstance(payload, dict) and not payload.get("error") and payload.get("baseline") is not None:
            key = (str(args.get("df_descriptor", "")).lower(), str(args.get("component") or "H").upper())
            self.observed_baselines[key] = payload["baseline"]

    def bound_statistics_args(self, store: Any) -> dict | None:
        """Resolve only a simple complete request against existing tables.

        Execution still goes through get_statistics. No network, new frame,
        model-generated argument, or stored answer is used here.
        """
        if (not self.routine or not self.wants_stats or not self.components
                or self.wants_plot or self.wants_derived or self.clarification):
            return None
        if len(set(DATES.findall(self.query))) > 1:
            return None
        if re.search(r"\b\d{1,2}:\d{2}|час|минут|интервал|окн|фильтр|после|перед|между|услов|округл|знак|точност|precision|round|digits|decimal|экспорт|сохран|обнов|\b(?:csv|json|fetch|download)\b", self.query, re.I):
            return None
        if re.search(r"\b(?:загрузи|загрузить|скачай|скачать)\b", self.query, re.I):
            return None
        candidates = [self.handle] if self.handle else [self.target]
        if not self.handle and self.station and self.date:
            candidates.extend((f"raw:{self.station}:{self.date}", f"derived:{self.station}:{self.date}"))
        for handle in dict.fromkeys(h for h in candidates if h):
            if store.get(handle) is None:
                continue
            args = {"df_descriptor": handle, "components": list(self.components)}
            if self.check_call("get_statistics", args, store) is None:
                return args
        return None

    def check_call(self, name: str, args: dict, store: Any) -> str | None:
        if name == "evaluate_custom_formula" and self.requested_formula:
            try:
                expected = ast.dump(ast.parse(self.requested_formula, mode="eval"))
                actual = ast.dump(ast.parse(str(args.get("formula") or ""), mode="eval"))
            except SyntaxError:
                expected, actual = self.requested_formula.strip(), str(args.get("formula") or "").strip()
            if expected != actual:
                return "Используй выражение текущего запроса; не подменяй его формулой предыдущей реплики."
        if name == "calculate_baseline" and self.baseline_hours is not None:
            hours = args.get("night_hours", [0, 4])
            if not isinstance(hours, (list, tuple)) or tuple(hours) != self.baseline_hours:
                return f"Запрошено UTC-окно night_hours={list(self.baseline_hours)}."
        if name in {"calculate_derived_math", "apply_physics_formula"} and self.requested_metric:
            actual = str(args.get("metric") or "").lower()
            aliases = {"ssc_flag": "detect_ssc"}
            if aliases.get(actual, actual) != aliases.get(self.requested_metric, self.requested_metric):
                return f"Запрошена metric={self.requested_metric}; передана metric={actual}."
        if name == "calculate_baseline" and self.baseline_mode and str(args.get("mode") or "night").lower() != self.baseline_mode:
            return f"Запрошен baseline mode={self.baseline_mode}."
        if (name == "calculate_derived_math" and str(args.get("metric")).lower() == "anomaly"
                and "calculate_baseline" in self.calculation_tools):
            key = (str(args.get("df_descriptor", "")).lower(), str(args.get("component") or "H").upper())
            baseline = self.observed_baselines.get(key)
            try:
                matches = baseline is not None and math.isclose(float(args.get("baseline_value")), float(baseline), rel_tol=1e-12, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                return "Сначала получи baseline для этой таблицы/компоненты в текущем запросе и передай его точное значение в baseline_value."
        if self.wants_local_time and name == "calculate_local_time":
            requested = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", self.query)
            if requested and str(args.get("timestamp")) != requested.group():
                return "Используй точный UTC timestamp текущего запроса."
            if self.station and str(args.get("station_code", "")).lower() != self.station:
                return "Используй станцию текущего запроса."
        if name == "get_statistics" and self.components:
            actual = args.get("components")
            expected = {c.casefold() for c in self.components}
            if (not isinstance(actual, list) or not actual
                    or any(not isinstance(c, str) or c.strip().casefold() not in expected for c in actual)):
                return (f"Для текущей статистики нужны components={list(self.components)!r}. "
                        f"Передано components={actual!r}. Используй запрошенные компоненты, а не компоненты прошлых реплик.")
        if self.no_fetch and name in FETCH:
            return "Запрошено использование прежних данных без повторной загрузки."
        fetch_date = self.fetch_date or self.date
        if fetch_date and name in FETCH:
            for key in ("start_date", "end_date"):
                if args.get(key) and str(args[key])[:10] != fetch_date:
                    return f"Дата запроса {fetch_date}; инструмент получил {key}={args[key]}."
        if self.station and name == "fetch_observatory_data" and str(args.get("station_code", "")).lower() != self.station:
            return f"Запрошена станция {self.station.upper()}, а не {args.get('station_code')}."
        for key in FRAME_ARGS:
            handle = args.get(key)
            if not isinstance(handle, str):
                continue
            canonical = store.aliases().get(handle.lower(), handle.lower())
            match = HANDLE.fullmatch(canonical)
            if not match:
                continue
            family, source, day = canonical.split(":")
            if store.get(canonical) is None:
                raw_sibling = f"raw:{source}:{day}"
                if family != "derived" or store.get(raw_sibling) is None:
                    return f"unknown_frame: таблица {canonical} отсутствует. Сначала получи данные нужного источника; подмена другой таблицей запрещена."
            if self.target and self.target.startswith(("idx:", "sw:")) and key == "df_descriptor":
                if canonical.split(":")[:2] != self.target.split(":")[:2]:
                    return f"Запрошен источник таблицы {self.target}; передан {canonical}."
            if self.date and day != self.date:
                return f"Запрошена дата {self.date}; передан handle {canonical}."
            if self.station and family in {"raw", "derived", "anomalies"} and source != self.station:
                return f"Запрошена станция {self.station}; передан handle {canonical}."
            if self.handle and key == "df_descriptor" and canonical != self.handle:
                return f"Используй явный handle {self.handle}; подмена на {canonical} не разрешена."
        return None

    def missing(self, log: list[dict], store: Any) -> list[str]:
        missing = []
        for tool in self.calculation_tools:
            if not any(e.get("ok") and e.get("tool") == tool and e.get("calculation") for e in log):
                missing.append(f"выполнить {tool} и получить фактический результат")
        if self.wants_local_time and not any(e.get("ok") and e.get("tool") == "calculate_local_time" for e in log):
            missing.append("вычислить местное среднее солнечное время инструментом calculate_local_time")
        if self.target and store.get(self.target) is None:
            missing.append(f"получить таблицу {self.target}")
        if self.wants_derived:
            raw = self.target or self.handle
            derived = "derived:" + raw.split(":", 1)[1] if raw and raw.startswith(("raw:", "derived:")) else None
            frame = store.get(derived) if derived else None
            if frame is None or not {"H", "D", "I"} <= set(frame.columns):
                missing.append("рассчитать H/D/I в соответствующей derived-таблице")
        stats = [e for e in log if e.get("ok") and e.get("tool") == "get_statistics" and e.get("statistics")]
        if self.wants_stats:
            done = {c.casefold() for e in stats for c in e["statistics"]}
            if not stats or (self.components and not set(c.casefold() for c in self.components) <= done):
                missing.append("получить статистику через get_statistics, components="
                               + (repr(list(self.components)) if self.components else "запрошенные компоненты"))
        if self.wants_plot and not any(e.get("ok") and e.get("plot") and Path(e["plot"]).is_file() for e in log):
            missing.append("построить запрошенный график")
        return missing

    def facts(self, log: list[dict], store: Any) -> str:
        lines = ["Подтверждённые результаты:"]
        seen_statistics = set()
        handles = list(dict.fromkeys([self.target, self.handle] + [e.get("frame_handle") for e in log if e.get("ok")]
                                     + [h for e in log if e.get("ok") for h in e.get("resolved_handles", [])]))
        for handle in handles:
            if handle and store.get(handle) is not None:
                frame = store.get(handle)
                lines.append(f"- `{handle}`: {len(frame)} строк.")
                attrs = frame.attrs or {}
                for key, label in (("source", "Источник"), ("dataset", "Dataset"), ("datasets", "Datasets"),
                                   ("publication_states", "Типы публикации"), ("publication_state", "Тип публикации"),
                                   ("note", "Метаданные")):
                    if attrs.get(key) is not None:
                        lines.append(f"  {label}: {attrs[key]}.")
                if handle.startswith("raw:") and not (attrs.get("publication_states") or attrs.get("publication_state")):
                    lines.append("  Тип публикации: не подтверждён метаданными таблицы.")
                if "timestamp" in frame and len(frame):
                    lines.append(f"  Временное покрытие: {frame['timestamp'].min()} … {frame['timestamp'].max()}; время загрузчиков — UTC.")
                if handle.startswith("sw:omni:"):
                    lines.append("  Единицы OMNI: Bx/By/Bz — nT; V — km/s; Np — cm^-3; T — K. By/Bz — GSM; Bx общий для GSE/GSM.")
                elif handle.startswith("idx:kyoto:"):
                    lines.append("  Dst — nT, исходный шаг 1 час; Kp — безразмерный, исходный шаг 3 часа; часовая таблица не означает часовые измерения Kp.")
        for entry in log:
            if not entry.get("ok"):
                continue
            calculation = entry.get("calculation")
            if calculation:
                lines.append(f"- {entry['tool']}:")
                for key, value in calculation.items():
                    lines.append(f"  {key}: {value}.")
            if entry.get("local_time"):
                result = entry["local_time"]
                lines.append(f"- Местное среднее солнечное время: {result['lt_hm']}; дата и время: {result['local_mean_solar_datetime']}; "
                             f"географическая долгота: {result['longitude']}°; модель: {result['model']}. "
                             "Это не гражданский часовой пояс и не истинное солнечное время; поправка уравнения времени не применяется.")
            for component, values in (entry.get("statistics") or {}).items():
                key = (tuple(entry.get("resolved_handles", [])), component, repr(sorted(values.items())))
                if key in seen_statistics:
                    continue
                seen_statistics.add(key)
                facts = ", ".join(f"{label}: {values[k]}" for k, label in (
                    ("min", "минимум"), ("max", "максимум"), ("mean", "среднее"),
                    ("median", "медиана"), ("std", "стандартное отклонение"),
                    ("count", "число значений"), ("missing", "пропуски")) if k in values)
                lines.append(f"- {component}: {facts}; единицы: {values.get('unit') or 'не указаны'}.")
            if entry.get("plot") and Path(entry["plot"]).is_file():
                lines.append(f"- График: `{Path(entry['plot']).name}`.")
        return "\n".join(lines) if len(lines) > 1 else "Подтверждённых результатов для этого запроса нет."


@dataclass
class SessionMemory:
    # Anchors are bounded separately: old prose is not used to reconstruct them.
    ground: list[str] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)
    last_good: str | None = None
    first_data: str | None = None
    last_stats: bool = False
    last_plot: bool = False
    last_inputs: list[str] = field(default_factory=list)

    def record(self, log: list[dict], store: Any) -> None:
        if any(e.get("ok") for e in log):
            self.last_stats = any(e.get("ok") and e.get("statistics") for e in log)
            self.last_plot = any(e.get("ok") and e.get("plot") for e in log)
        for entry in log:
            if not entry.get("ok"):
                continue
            produced = entry.get("frame_handle")
            inputs = [h for h in entry.get("resolved_handles", []) if store.get(h) is not None]
            for handle in inputs:
                self.touch(handle)
            if produced and store.get(produced) is not None:
                if self.first_data is None:
                    self.first_data = produced
                if entry.get("tool") == "fetch_observatory_data" and produced not in self.ground:
                    if len(self.ground) < 128:
                        self.ground.append(produced)
                self.touch(produced)
            if produced or inputs:
                self.last_inputs = ([produced] if produced else inputs)[-12:]

    def touch(self, handle: str) -> None:
        self.last_good = handle
        self.recent = [h for h in self.recent if h != handle][-11:] + [handle]

    @staticmethod
    def summary_requested(question: str) -> bool:
        q = question.lower()
        assets = bool(re.search(r"станц|дат|файл|результат|handle|files|dates|results", q))
        summary = bool(re.search(r"сводк|\bsummary\b", q))
        enumerate_session = (bool(re.search(r"перечисл|подведи\s+итог|\blist\b", q))
                             and bool(re.search(r"использован|разговор|диалог|контекст|подтверд", q)))
        return assets and (summary or enumerate_session)

    def prepare(self, question: str, store: Any) -> Request:
        q = question.lower()
        request = Request(question)
        placeholders = list(dict.fromkeys(re.findall(r"\{[A-Z][A-Z0-9_]*\}", question)))
        if placeholders:
            request.clarification = ("В запросе остались незаполненные поля: " + ", ".join(placeholders)
                                     + ". Укажите реальные имена, handle или параметры. Данные и станции по догадке не выбираю; инструменты не вызывались.")
            return request
        explicit = list(dict.fromkeys(h.lower() for h in HANDLE.findall(question)))
        days = list(dict.fromkeys(DATES.findall(question)))
        station = station_in(question)
        first = bool(re.search(r"перв\w*\s+(?:день|дн[яюе]|дат\w*)|first\s+day", q))
        same = bool(re.search(r"тех\s+же\s+данн|того\s+же\s+дня|same\s+(?:day|data)", q))
        correct = bool(re.search(r"корректн\w*\s+handle|правильн\w*\s+handle", q))
        if first and not days:
            candidates = ([h for h in self.ground if h.split(":")[1] == station]
                          if station else ([self.first_data] if self.first_data else []))
            candidates = [h for h in candidates[:1] if store.get(h) is not None]
            if not candidates:
                request.clarification = "Не могу подтвердить первый день по результатам этой сессии. Укажите дату YYYY-MM-DD и станцию."
                return request
            anchor = candidates[0]
            days = [anchor.split(":")[2]]
            if station:
                request.station = station
            request.query += f"\nПодтверждённая дата первого дня: {days[0]}; исходная таблица {anchor}."
        elif same and not days and self.last_good:
            if len(self.last_inputs) > 1 and not explicit:
                request.clarification = "Предыдущая операция использовала несколько таблиц. Укажите точный handle для продолжения."
                return request
            explicit = explicit or [self.last_good]
            request.query += f"\nПодтверждённая таблица предыдущей успешной операции: {self.last_good}."
        elif correct and not explicit:
            if len(self.last_inputs) > 1:
                request.clarification = "Есть несколько таблиц предыдущей успешной операции. Укажите точный корректный handle."
                return request
            if self.last_good and store.get(self.last_good) is not None:
                explicit = [self.last_good]
                request.query += f"\nПоследний подтверждённый handle: {self.last_good}."
            else:
                request.clarification = "Укажите точный корректный handle: успешной таблицы в этой сессии нет."
                return request
        # A single explicit table is a binding reference, not a hint to guess a sibling.
        if len(explicit) == 1:
            request.handle = explicit[0]
            if store.get(request.handle) is None:
                request.clarification = f"unknown_frame: таблица `{request.handle}` отсутствует в этой сессии. Подмена и загрузка другого набора не выполнены."
                return request
            days = days or [request.handle.split(":")[2]]
            if request.handle.startswith(("raw:", "derived:")):
                station = request.handle.split(":")[1]
        if len(days) == 1:
            try:
                date.fromisoformat(days[0])
                request.fetch_date = days[0]
                # Two-day comparisons deliberately retain both independent dates.
                if not re.search(r"два\s+дня|двух\s+дн|two\s+days|plot_comparison", q):
                    request.date = days[0]
            except ValueError:
                request.clarification = "Укажите существующую дату YYYY-MM-DD."
                return request
        request.station = request.station or station
        request.no_fetch = bool(re.search(r"не\s+скачивай|без\s+(?:новой|повторной)\s+загруз|no\s+refetch", q))
        request.wants_stats = bool(re.search(r"статист|медиан|\bstatistics\b|\bmedian\b", q))
        request.wants_plot = bool(re.search(r"постро|построй|график|сопостав|сравн|\bcompar|\bplot\b|\bchart\b", q))
        if same and not (request.wants_stats or request.wants_plot) and re.search(r"\b[XYZFHDI]\b", question):
            request.wants_stats = request.wants_stats or self.last_stats
            request.wants_plot = request.wants_plot or self.last_plot
            request.query += ("\nПолучи статистику." if request.wants_stats else "") + (" Построй график." if request.wants_plot else "")
        stats_start = re.search(r"статист\w*|медиан\w*|\bstatistics\b|\bmedian\b", question, re.I)
        stats_text = question[stats_start.end():] if stats_start else question
        stats_text = re.split(r"постро|график|\bplot\b", stats_text, maxsplit=1, flags=re.I)[0]
        request.components = tuple(dict.fromkeys(re.findall(r"\b(?:Dst|Kp|[XYZFHDI])\b", stats_text, re.I))) if request.wants_stats else ()
        request.wants_derived = bool(re.search(r"\bh\s*[/,]\s*d\s*[/,]\s*i\b", q)) and not request.wants_stats
        action_text = re.sub(r"не\s+(?:рассчитывай|вычисляй|считай|применяй|вызывай|используй)\b", "", q)
        computing = bool(re.search(r"рассчита|вычисл|посчита|получи|найди|определ|примени|\b(?:calculate|compute|evaluate|apply|find)\b", action_text))
        required = [name for name in sorted(CALCULATION_TOOLS) if (computing or q.lstrip().startswith(name)) and re.search(r"\b" + name + r"\b", q)
                    and not re.search(r"не\s+(?:вызывай|используй|применяй|рассчитывай)\s+" + name + r"\b", q)]
        if computing and re.search(r"\bbaseline\b|базов\w*\s+лини", q) and not required:
            required.append("calculate_baseline")
        if request.wants_derived and "calculate_derived_components" not in required:
            required.append("calculate_derived_components")
        metric = re.search(r"\bmetric\s*=\s*([a-z_]+)", q)
        if metric:
            request.requested_metric = metric[1]
        baseline_mode = re.search(r"\bmode\s*=\s*(night|full)\b", q)
        if baseline_mode:
            request.baseline_mode = baseline_mode[1]
        request.calculation_tools = tuple(required)
        formula = re.search(r"выражение\s+(.+?)\s+для\s+(?:raw|derived|formula|math):", question, re.I)
        if formula:
            request.requested_formula = formula[1].strip().strip('`')
        hours = re.search(r"night_hours\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", q)
        if hours:
            request.baseline_hours = (int(hours[1]), int(hours[2]))
        fetch_asked = bool(re.search(r"загруз|скача|получ[иь]\w*|\bfetch\b|\bload\b", q))
        if fetch_asked and request.fetch_date and not request.handle:
            if re.search(r"\bomni\b|солнечн\w*\s+вет", q):
                request.target = f"sw:omni:{request.fetch_date}"
            elif re.search(r"\bkyoto\b|\bdst\b|\bkp\b", q):
                request.target = f"idx:kyoto:{request.fetch_date}"
            elif station:
                request.target = f"raw:{station}:{request.fetch_date}"
        request.routine = bool(request.target or request.wants_stats or request.wants_plot) and not re.search(
            r"проанализ|объясн|почему|\b(?:ssc|gic|burton|proxy|epsilon|formula)\b|формул|метрик|аномал|размах|производн|mlt|налож", q)
        request.wants_local_time = bool(re.search(r"calculate_local_time|(?:местн\w*\s+солнечн|солнечн\w*\s+врем)|mean\s+solar\s+time", q))
        if request.wants_local_time:
            request.routine = True
        if request.calculation_tools or request.wants_derived:
            request.routine = True
        if request.wants_stats and request.components:
            request.query += (f"\nПараметры статистики текущего запроса: components={list(request.components)!r}. "
                              "Не переноси компоненты предыдущих реплик в этот вызов get_statistics.")
        return request

    def context(self) -> str:
        return ("Подтверждённое состояние сессии (не заменяет вызов статистики):\n"
                + "Первая успешная таблица: " + str(self.first_data or "не задана") + ". "
                + "Первые успешные наземные загрузки в порядке получения: " + ", ".join(self.ground[:12])
                + ". Последние использованные таблицы: " + ", ".join(self.recent) + ".\n")

    def summary(self, store: Any, saved: list[dict]) -> str:
        lines = ["Сводка по фактическому состоянию этой сессии:"]
        for handle in store.names():
            lines.append(f"- Таблица `{handle}`: {len(store.get(handle))} строк.")
        for item in saved:
            path = Path(item["path"])
            lines.append(f"- Файл `{path.name}`: " + ("существует" if path.is_file() else "отсутствует на диске") + ".")
        if len(lines) == 1:
            lines.append("Подтверждённых таблиц и графиков нет.")
        lines.append("Это реестр доступных таблиц и зарегистрированных файлов; он не является полным архивом текста разговора или научной проверкой содержимого.")
        return "\n".join(lines)
