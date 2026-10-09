"""Planner: splits a large request into one task per station / day / batch.

A 7B model defeated by "build 300 charts" is not a failure of tooling but a
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
        """Break *query* into :class:`TaskStep` objects."""
        state = SessionState(original_query=query)
        task_type = self._detect_task_type(query)

        if task_type in ("multi_station_plot", "batch_statistics"):
            stations = self._extract_stations(query)
            component = self._extract_component(query)
            date = self._extract_date(query)
            tool = "plot_components" if task_type == "multi_station_plot" else "get_statistics"
            verb = "Построй график" if tool == "plot_components" else "Посчитай статистику"
            for i, station in enumerate(stations):
                state.add_task(
                    TaskStep(
                        id=f"step_{i:03d}",
                        description=f"{verb} {component} для станции {station} за {date}",
                        tool_name=tool,
                        tool_args={"station": station, "component": component, "date": date},
                    )
                )
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
        found = [s for s in self.KNOWN_STATIONS if s.lower() in query_lower]
        if found:
            return found
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

    def _extract_date(self, query: str) -> str:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
        if match:
            return match.group(1)
        match = re.search(
            r"(\d{1,2})\s*(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s*(\d{4})",
            query,
            re.IGNORECASE,
        )
        if match:
            day, month_ru, year = match.groups()
            month = self._MONTHS_RU.get(month_ru.lower(), "01")
            return f"{year}-{month}-{int(day):02d}"
        return "2024-09-10"


class LLMPlanner:
    """Planner that asks the model to split a complex request into subtasks."""

    PLANNER_PROMPT = (
        "Ты — планировщик задач для геомагнитного агента.\n"
        "Разбей запрос пользователя на конкретные подзадачи.\n\n"
        "ФОРМАТ ОТВЕТА (JSON):\n"
        '{"tasks": [{"id": "step_001", "description": "Скачай данные для станции '
        'BRW за 2024-10-10", "tool_name": "fetch_observatory_data", "tool_args": '
        '{"station_code": "BRW", "start_date": "2024-10-10", "end_date": "2024-10-10"}}, '
        '{"id": "step_002", "description": "Вычисли компоненты H, D, I для BRW", '
        '"tool_name": "calculate_derived_components", "tool_args": {"df": '
        '"raw:brw:2024-10-10", "components": ["H", "D", "I"]}}, ...]}\n\n'
        "ПРАВИЛА:\n"
        "1. Каждая подзадача — ОДИН вызов инструмента.\n"
        "2. Порядок важен: сначала fetch, потом derive, потом plot.\n"
        "3. Для каждой станции и каждой даты — отдельные подзадачи.\n"
        "4. Если нужно создать проект — добавь create_project первой.\n"
        "5. Если нужен экспорт — добавь export_project последней.\n"
        "6. Ответь ТОЛЬКО одним JSON-объектом, без пояснений и markdown."
    )

    def __init__(self, brain: Any):
        self.brain = brain

    def plan(self, query: str) -> SessionState:
        """Ask the model for the plan; fall back to :class:`Planner` on garbage."""
        prompt = (
            f"{self.PLANNER_PROMPT}\n\nЗапрос пользователя: {query}\n\nОтвет (JSON):"
        )
        tasks = self._parse_json(self._ask(prompt))
        fallback = Planner()

        if tasks is None:
            return fallback.plan(query)

        state = SessionState(original_query=query)
        for index, item in enumerate(tasks, start=1):
            state.add_task(self._coerce_task(item, index))
        return state if state.tasks else fallback.plan(query)

    def _ask(self, prompt: str) -> str:
        brain = self.brain
        if brain is None:
            return ""
        if hasattr(brain, "create_completion"):
            try:
                response = brain.create_completion(
                    prompt=prompt, max_tokens=4096, temperature=0.1, stop=["}"]
                )
            except Exception:
                return ""
            choices = response.get("choices") or []
            if choices and isinstance(choices[0], dict):
                return str(choices[0].get("text") or choices[0].get("message") or "")
            return ""
        if hasattr(brain, "chat"):
            try:
                reply = brain.chat([{"role": "user", "content": prompt}], None)
            except Exception:
                return ""
            if isinstance(reply, dict):
                return str(reply.get("content") or reply.get("text") or "")
            return ""
        return ""

    @staticmethod
    def _parse_json(text: str) -> list[dict[str, Any]] | None:
        """Pull the first balanced {...} object out of *text* and read tasks.

        The model often pads the JSON with prose or markdown fences, and a
        ``stop`` token can cut it mid-object, so a bare ``json.loads`` is not
        enough -- the braces are matched by hand first.
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
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        tasks = data.get("tasks")
        return tasks if isinstance(tasks, list) else None

    @staticmethod
    def _coerce_task(item: Any, index: int) -> TaskStep:
        """Normalise a raw JSON task into a :class:`TaskStep`."""
        if isinstance(item, dict):
            args = item.get("tool_args")
            return TaskStep(
                id=str(item.get("id") or f"step_{index:03d}"),
                description=str(item.get("description") or item.get("tool_name") or ""),
                tool_name=str(item.get("tool_name") or "run_agent"),
                tool_args=args if isinstance(args, dict) else {},
            )
        return TaskStep(
            id=f"step_{index:03d}",
            description=str(item),
            tool_name="run_agent",
            tool_args={},
        )