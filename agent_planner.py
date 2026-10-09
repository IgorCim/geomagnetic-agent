"""Planner: splits a large request into one task per station / day / batch.

A 7B model defeated by "build 300 charts" is not a failure of tooling but a
failure of scope: one prompt, one conversation window, ten or twenty tool calls.
The planner is deliberately deterministic Python, not text the model has to obey:
it recognises the request shape (``N станций``, ``N дней``, ``N станций`` for
statistics), expands it into one :class:`~agent_state.TaskStep` per unit, and
hands the session to the runner which executes them one at a time with a narrow
toolset and a short prompt.
"""
from __future__ import annotations

import re
from typing import Any

from agent_state import SessionState, TaskStep

__all__ = ["Planner"]


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