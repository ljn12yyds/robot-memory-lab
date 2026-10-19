from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import ReportStatus, StepStatus, Subtask, SubtaskReport, TaskPlan, to_dict


@dataclass
class TaskMemory:
    plan: TaskPlan
    current_step: int = 0
    world_state: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.current_step >= len(self.plan.subtasks)

    @property
    def blocked(self) -> bool:
        return any(step.status is StepStatus.BLOCKED for step in self.plan.subtasks)

    @property
    def current_subtask(self) -> Subtask | None:
        if self.finished:
            return None
        return self.plan.subtasks[self.current_step]

    def begin_current(self) -> Subtask:
        step = self.current_subtask
        if step is None:
            raise RuntimeError("task is already complete")
        if step.status is StepStatus.BLOCKED:
            raise RuntimeError("current subtask is blocked")
        step.status = StepStatus.RUNNING
        step.attempts += 1
        self._record("subtask_started", step=step.name, attempt=step.attempts)
        return step

    def apply_report(self, report: SubtaskReport) -> None:
        step = self.current_subtask
        if step is None:
            raise RuntimeError("cannot apply a report to a completed task")
        self.world_state.update(report.state_updates)
        self._record(
            "subtask_reported",
            step=step.name,
            status=report.status.value,
            message=report.message,
            state_updates=report.state_updates,
        )

        if report.status is ReportStatus.SUCCESS:
            step.status = StepStatus.COMPLETED
            self.current_step += 1
            return
        if report.status is ReportStatus.BLOCKED:
            step.status = StepStatus.BLOCKED
            return
        if step.attempts > step.max_retries:
            step.status = StepStatus.BLOCKED
            self._record("retry_budget_exhausted", step=step.name)
        else:
            step.status = StepStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": to_dict(self.plan),
            "current_step": self.current_step,
            "world_state": self.world_state,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskMemory":
        return cls(
            plan=TaskPlan.from_dict(data["plan"]),
            current_step=data.get("current_step", 0),
            world_state=dict(data.get("world_state", {})),
            events=list(data.get("events", [])),
        )

    def _record(self, event: str, **payload: Any) -> None:
        self.events.append(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **payload,
            }
        )


class JsonTaskMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> TaskMemory:
        with self.path.open("r", encoding="utf-8") as handle:
            return TaskMemory.from_dict(json.load(handle))

    def save(self, memory: TaskMemory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(memory.to_dict(), handle, indent=2, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

