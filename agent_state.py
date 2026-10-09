"""External brain state: a session, its tasks and its progress.

The LLM loses context on long jobs ("build 300 charts") because everything it
knows lives in the conversation window. The planner moves that knowledge out of
the window and into ordinary Python objects: a :class:`SessionState` tracks the
plan, each :class:`TaskStep` records one piece of work, and the bits the model
actually needs to see are rendered on demand (:meth:`SessionState.get_context_for_prompt`).
"""
from __future__ import annotations

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
        if self.tasks and self.tasks[self.current_task_index].id == task_id:
            self.current_task_index += 1

    def mark_failed(self, task_id: str, error: str) -> None:
        for task in self.tasks:
            if task.id == task_id:
                task.status = "failed"
                task.error = error
                break

    def get_progress_summary(self) -> str:
        total = len(self.tasks)
        done = sum(1 for t in self.tasks if t.status == "done")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        return f"Всего задач: {total}, выполнено: {done}, ошибок: {failed}"

    def get_context_for_prompt(self) -> str:
        recent = self.tasks[max(0, self.current_task_index - 3): self.current_task_index + 1]
        icons = {"done": "✅", "failed": "❌", "pending": "⏳", "running": "▶️"}
        lines = []
        for task in recent:
            icon = icons.get(task.status, "⏳")
            lines.append(f"{icon} [{task.id}] {task.description}")
        return "\n".join(lines) if lines else "Начинаем выполнение."

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