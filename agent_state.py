"""External brain state: a session, its tasks and its progress.

The LLM loses context on long jobs ("build 300 charts") because everything it
knows lives in the conversation window. The planner moves that knowledge out of
the window and into ordinary Python objects: a :class:`SessionState` tracks the
plan, each :class:`TaskStep` records one piece of work, and the bits the model
actually needs to see are rendered on demand (:meth:`SessionState.get_context_for_prompt`).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["TaskStep", "SessionState"]


@dataclass
class TaskStep:
    """One step in an execution plan."""

    id: str
    description: str
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | running | done | failed
    result: Any = None
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionState:
    """The external-memory state of one agent session."""

    session_id: str = "default"
    original_query: str = ""
    tasks: list[TaskStep] = field(default_factory=list)
    current_task_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_task(self, task: TaskStep) -> None:
        self.tasks.append(task)

    def get_current_task(self) -> TaskStep | None:
        if 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None

    def mark_done(self, task_id: str, result: Any = None) -> None:
        for task in self.tasks:
            if task.id == task_id:
                task.status = "done"
                task.result = result
                break
        # Move to the next task only when the done one is the current one, so a
        # re-opened session can resume from where it stopped.
        if (
            self.tasks
            and 0 <= self.current_task_index < len(self.tasks)
            and self.tasks[self.current_task_index].id == task_id
        ):
            self.current_task_index += 1

    def mark_failed(self, task_id: str, error: str) -> None:
        for task in self.tasks:
            if task.id == task_id:
                task.status = "failed"
                task.error = error
                break
        # A failure is still progress: the next task must see this one's error
        # and its own pending description. Without advancing here, a failed
        # step pinned the context window to itself for the rest of the plan.
        if (
            self.tasks
            and 0 <= self.current_task_index < len(self.tasks)
            and self.tasks[self.current_task_index].id == task_id
        ):
            self.current_task_index += 1

    def get_progress_summary(self) -> str:
        total = len(self.tasks)
        done = sum(1 for t in self.tasks if t.status == "done")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        return f"Всего задач: {total}, выполнено: {done}, ошибок: {failed}"

    def get_context_for_prompt(self) -> str:
        """Render the last few tasks with the handles they produced.

        Descriptions alone were not enough to chain a plan: subtask 2 was told
        "plot BRW" but not the ``raw:brw:...`` handle subtask 1 had just
        created, so it had to guess the naming convention. The compact result
        line carries the live handle (and the note) without dumping a whole
        tool payload into the prompt.
        """
        recent = self.tasks[max(0, self.current_task_index - 3): self.current_task_index + 1]
        icons = {"done": "✅", "failed": "❌", "pending": "⏳", "running": "▶️"}
        lines: list[str] = []
        for task in recent:
            icon = icons.get(task.status, "⏳")
            line = f"{icon} [{task.id}] {task.description}"
            if task.status == "done":
                summary = _compact_task_result(task)
                if summary:
                    line += f" -> {summary}"
            elif task.status == "failed" and task.error:
                line += f" -> ошибка: {str(task.error)[:180]}"
            if task.tool_args:
                try:
                    args = json.dumps(task.tool_args, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    args = str(task.tool_args)
                line += f" (args: {args[:220]})"
            lines.append(line)
        text = "\n".join(lines) if lines else "Начинаем выполнение."
        return text[:1800]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        state = cls()
        state.session_id = data.get("session_id", "default")
        state.original_query = data.get("original_query", "")
        state.current_task_index = data.get("current_task_index", 0)
        state.metadata = data.get("metadata", {})
        for task_data in data.get("tasks", []):
            task = TaskStep(**task_data)
            state.tasks.append(task)
        return state


def _compact_task_result(task: TaskStep, limit: int = 260) -> str:
    """One line of the evidence a later subtask needs from *task*."""
    result = task.result
    if result is None:
        return ""
    if not isinstance(result, dict):
        return str(result)[:limit]
    parts: list[str] = []
    handles: list[str] = []
    for call in result.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        handle = call.get("frame_handle")
        if isinstance(handle, str) and handle:
            handles.append(handle)
        for extra in call.get("frame_handles") or []:
            if isinstance(extra, str) and extra:
                handles.append(extra)
        plot = call.get("plot")
        if isinstance(plot, str) and plot:
            parts.append(f"plot={plot}")
        note = call.get("note")
        if isinstance(note, str) and note:
            parts.append(note)
    if handles:
        parts.insert(0, "handles: " + ", ".join(dict.fromkeys(handles)))
    text = result.get("text")
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    return " | ".join(parts)[:limit]
