"""Planner: splits a large request into one task per station / day / batch.

A model defeated by "build 300 charts" is not a failure of tooling but a
failure of scope: one prompt, one conversation window, ten or twenty tool calls.
The :class:`Planner` handles that deterministically (regex shapes like
``N станций``); the :class:`LLMPlanner` delegates the split to the model itself
for free-form requests the regexes cannot see through, then falls back to the
deterministic planner when the model's answer does not parse.
"""
from __future__ import annotations

import json
import re
from typing import Any

from agent_state import SessionState, TaskStep

__all__ = ["Planner", "LLMPlanner"]


class Planner:
    """Deterministic rule-based planner for batch geomagnetic requests."""

    PATTERNS: dict[str, str] = {
        # Most specific first: a statistics request also contains "станций",
        # which the multi-station plot (_loose_) pattern would otherwise swallow.
        "batch_statistics": r"(посчитай|статистик[ау]|медиан[ау]).*(\d+)\s*станци[яий]",
        "multi_station_plot": r"(построй|графики?|сравни).*(\d+)\s*станци[яий]|станци[яий].*(\d+)",
        "multi_date_plot": r"(построй|графики?).*(\d+)\s*(день|дня|дней|дата|даты)",
    }

    _MONTHS_RU = {
        "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
        "мая": "05", "июня": "06", "июля": "07", "августа": "08",
        "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12",
    }

    KNOWN_STATIONS = [
        "IRT", "MOS", "NVS", "SPG", "YAK", "KHB", "MGD", "VLA", "PET",
        "ARS", "BOX", "TIK", "API", "BSL",
    ]

    def plan(self, query: str) -> SessionState:
        """Break *query* into :class:`TaskStep` objects.

        A batch task is deliberately two steps per station/date: an explicit
        fetch with canonical arguments, then the plot/statistics step with the
        exact ``df_descriptor`` that fetch will create. The old single task
        carried only ``station``/``date``/``component`` -- arguments no handler
        accepts -- so the planner's structured data was dead and every subtask
        had to rediscover the handle through the model.
        """
        state = SessionState(original_query=query)
        task_type = self._detect_task_type(query)

        if task_type in ("multi_station_plot", "batch_statistics", "multi_date_plot"):
            stations = self._extract_stations(query)
            component = self._extract_component(query)
            if task_type == "multi_date_plot":
                dates = self._extract_dates(query)
            else:
                dates = [self._extract_date(query)]
            if not dates:
                # "3 дня" without dates is not a plan the deterministic parser
                # can make safely; let the agent ask or infer from context.
                state.add_task(
                    TaskStep(
                        id="step_000",
                        description=query,
                        tool_name="run_agent",
                        tool_args={"query": query},
                    )
                )
                return state
            tool = "get_statistics" if task_type == "batch_statistics" else "plot_components"
            verb = "Посчитай статистику" if tool == "get_statistics" else "Построй график"
            index = 0
            for station in stations:
                for day in dates:
                    handle = f"raw:{station.lower()}:{day}"
                    state.add_task(
                        TaskStep(
                            id=f"step_{index:03d}",
                            description=f"Скачай данные {station} за {day}",
                            tool_name="fetch_observatory_data",
                            tool_args={
                                "station_code": station,
                                "start_date": day,
                                "end_date": day,
                            },
                        )
                    )
                    index += 1
                    state.add_task(
                        TaskStep(
                            id=f"step_{index:03d}",
                            description=(
                                f"{verb} {component} для станции {station} за {day}; "
                                f"используй handle {handle}"
                            ),
                            tool_name=tool,
                            tool_args={
                                "df_descriptor": handle,
                                "components": [component],
                            },
                        )
                    )
                    index += 1
        else:
            # Unrecognised shape: a single task the model handles itself.
            state.add_task(
                TaskStep(
                    id="step_000",
                    description=query,
                    tool_name="run_agent",
                    tool_args={"query": query},
                )
            )
        return state

    def _detect_task_type(self, query: str) -> str | None:
        for task_type, pattern in self.PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                return task_type
        return None

    def _extract_stations(self, query: str) -> list[str]:
        query_lower = query.lower()
        found = [
            (query_lower.find(station.lower()), station)
            for station in self.KNOWN_STATIONS
            if station.lower() in query_lower
        ]
        if found:
            # Preserve the order the scientist wrote, not the registry order:
            # "API, IRT" must produce API's tasks first.
            return [station for _, station in sorted(found)]
        match = re.search(r"(\d+)\s*станци", query, re.IGNORECASE)
        if match:
            count = min(int(match.group(1)), len(self.KNOWN_STATIONS))
            return self.KNOWN_STATIONS[:count]
        return ["IRT"]

    def _extract_component(self, query: str) -> str:
        query_lower = query.lower()
        for comp in ["H", "D", "I", "F", "X", "Y", "Z"]:
            if f"компонент{comp}".lower() in query_lower or f"компонента {comp}".lower() in query_lower:
                return comp
            if f" {comp.lower()} " in f" {query_lower} ":
                return comp
        return "H"

    def _extract_dates(self, query: str) -> list[str]:
        """Every explicit date in *query*, preserving the order written.

        Handles ISO dates and Russian forms including "10 и 11 сентября 2024"
        and "10, 11, 12 сентября 2024". A bare "3 дня" yields no dates; the
        caller falls back to the model rather than inventing a window.
        """
        dates: list[str] = []
        for iso in re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", query):
            if iso not in dates:
                dates.append(iso)
        pattern = re.compile(
            r"((?:\d{1,2}\s*(?:,|и|/)?\s*)+)\s*"
            r"(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
            r"\s*(\d{4})",
            re.IGNORECASE,
        )
        for match in pattern.finditer(query):
            days_text, month_ru, year = match.groups()
            month = self._MONTHS_RU.get(month_ru.lower(), "01")
            for day in re.findall(r"\d{1,2}", days_text):
                value = f"{year}-{month}-{int(day):02d}"
                if value not in dates:
                    dates.append(value)
        return dates

    def _extract_date(self, query: str) -> str:
        dates = self._extract_dates(query)
        return dates[0] if dates else "2024-09-10"


class LLMPlanner:
    """Planner that asks the model to split a complex request into subtasks."""

    # A 260-chart job needs a big output budget: the default LlamaBrain cap is
    # MAX_TOKENS=1200, which fits ~20 tasks, not hundreds.
    PLAN_MAX_TOKENS = 4096
    PLAN_TEMPERATURE = 0.1

    PLANNER_PROMPT = (
        "Ты — планировщик задач для геомагнитного агента.\n"
        "Разбей запрос пользователя на конкретные подзадачи.\n\n"
        "ФОРМАТ ОТВЕТА (JSON):\n"
        '{"tasks": [{"id": "step_001", "description": "Скачай данные для станции '
        'BRW за 2024-10-10", "tool_name": "fetch_observatory_data", "tool_args": '
        '{"station_code": "BRW", "start_date": "2024-10-10", "end_date": "2024-10-10"}}, '
        '{"id": "step_002", "description": "Вычисли компоненты H, D, I для BRW", '
        '"tool_name": "calculate_derived_components", "tool_args": '
        '{"df_descriptor": "raw:brw:2024-10-10", "components": ["H", "D", "I"]}}, '
        '...]}\n\n'
        "ПРАВИЛА:\n"
        "1. Каждая подзадача — ОДИН вызов инструмента.\n"
        "2. Порядок важен: сначала fetch, потом derive, потом plot.\n"
        "3. Для каждой станции и каждой даты — отдельные подзадачи.\n"
        "4. Если нужно создать проект — добавь create_project первой.\n"
        "5. Если нужен экспорт — добавь export_project последней.\n"
        "6. Ответь ТОЛЬКО одним JSON-объектом, без пояснений и markdown.\n"
        "7. НЕ объединяй станции или даты в одной подзадаче. Мелко дроби: "
        "одна станция + одна дата + один инструмент на каждую подзадачу. "
        "Подзадач может быть сотни — это нормально.\n"
        "8. НЕ выводи списком больше 15 задач. Если комбинаций станций×дат "
        "больше 15 — верни ОДНУ подзадачу create_task_matrix с events, stations, "
        "actions, project_name; остальное (однократный process_matrix_batch "
        "с run_until_complete=True, который сам собирает архив) сделает агент.\n"
        "9. tool_args — строго по схеме инструмента: df_descriptor (НЕ df), "
        "station_code + start_date + end_date, components (список). "
        "Для create_project — name (НЕ project_name).\n"
        "10. Если подзадача использует результат предыдущей, укажи точный "
        "handle в df_descriptor."
    )

    def __init__(self, brain: Any):
        self.brain = brain

    def plan(self, query: str, verbose: bool = False) -> SessionState:
        """Ask the model for the plan; fall back to :class:`Planner` on garbage.

        ``verbose`` prints the raw model output and the extracted JSON, so a
        live run can be diagnosed when the split is suspicious (e.g. exactly
        one task where hundreds were expected).
        """
        prompt = (
            f"{self.PLANNER_PROMPT}\n\nЗапрос пользователя: {query}\n\nОтвет (JSON):"
        )
        raw = self._ask(prompt)

        if verbose:
            print("=" * 70)
            print("🤖 LLM-ПЛАНИРОВЩИК: Сырой ответ модели")
            print("=" * 70)
            print((raw or "<пусто>")[:1000])
            print("=" * 70)

        extracted = self._extract_json(raw)
        if verbose:
            print("📋 Извлечённый JSON:")
            print((extracted or "<нет>")[:500])
            print("=" * 70)

        tasks = self._parse_json(raw)
        if tasks is None:
            if verbose:
                print("❌ Ошибка парсинга — Fallback на детерминированный Planner()")
            return Planner().plan(query)

        state = SessionState(original_query=query)
        for index, item in enumerate(tasks, start=1):
            state.add_task(self._coerce_task(item, index))
        if not state.tasks:
            if verbose:
                print("❌ Пустой список задач — Fallback на детерминированный Planner()")
            return Planner().plan(query)
        if verbose:
            print(f"✅ Распарсено {len(state.tasks)} подзадач")
        return state

    def _ask(self, prompt: str) -> str:
        brain = self.brain
        if brain is None:
            return ""
        if hasattr(brain, "create_completion"):
            try:
                response = brain.create_completion(
                    prompt=prompt,
                    max_tokens=self.PLAN_MAX_TOKENS,
                    temperature=self.PLAN_TEMPERATURE,
                )
            except Exception:
                return ""
            choices = response.get("choices") or []
            if choices and isinstance(choices[0], dict):
                return str(choices[0].get("text") or choices[0].get("message") or "")
            return ""
        if hasattr(brain, "chat"):
            messages = [{"role": "user", "content": prompt}]
            try:
                # Bigger output budget for a long plan; a brain that does not
                # accept the kwarg (ScriptedBrain, test doubles) is retried.
                reply = brain.chat(messages, None, max_tokens=self.PLAN_MAX_TOKENS)
            except TypeError:
                try:
                    reply = brain.chat(messages, None)
                except Exception:
                    return ""
            except Exception:
                return ""
            if isinstance(reply, dict):
                return str(reply.get("content") or reply.get("text") or "")
            return ""
        return ""

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Return the first balanced ``{...}`` object out of *text*.

        The model often pads the JSON with prose or markdown fences, so a bare
        ``json.loads`` is not enough -- the braces are matched by hand first,
        handling nested objects and string literals.
        """
        if not text:
            return None
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        end = -1
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            ch = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end < 0:
            return None
        return text[start : end + 1]

    @classmethod
    def _parse_json(cls, text: str) -> list[dict[str, Any]] | None:
        """Pull the tasks out of the first balanced JSON object in *text*."""
        snippet = cls._extract_json(text)
        if snippet is None:
            return None
        try:
            data = json.loads(snippet)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        tasks = data.get("tasks")
        return tasks if isinstance(tasks, list) else None

    @staticmethod
    def _coerce_task(item: Any, index: int) -> TaskStep:
        """Normalise a raw JSON task into a :class:`TaskStep`.

        The planner's JSON is model output, not schema-validated data. Normalise
        the tool name and arguments here, before the task ever reaches the
        executor, so an unknown tool becomes a free-form task and a legacy
        ``df``/``station``/``project_name`` spelling becomes the canonical
        argument the handler accepts.
        """
        if isinstance(item, dict):
            description = str(item.get("description") or item.get("tool_name") or "")
            raw_name = str(item.get("tool_name") or "run_agent")
            raw_args = item.get("tool_args")
            try:
                import agent_core as ac

                tool_name = ac.normalize_tool_name(raw_name)
                args = ac.normalize_tool_args(tool_name, raw_args)
            except Exception:
                # A planner must never take the agent down because a helper
                # import failed; keep the raw values and let the executor fall
                # back to run_agent on its own.
                tool_name = raw_name
                args = raw_args if isinstance(raw_args, dict) else {}
            if tool_name == "run_agent":
                args = {"query": description}
            return TaskStep(
                id=str(item.get("id") or f"step_{index:03d}"),
                description=description,
                tool_name=tool_name,
                tool_args=args,
            )
        return TaskStep(
            id=f"step_{index:03d}",
            description=str(item),
            tool_name="run_agent",
            tool_args={"query": str(item)},
        )
