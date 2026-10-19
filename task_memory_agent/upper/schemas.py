from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class ReportStatus(str, Enum):
    SUCCESS = "success"
    RETRYABLE_FAILURE = "retryable_failure"
    BLOCKED = "blocked"


@dataclass
class Subtask:
    id: int
    name: str
    success_condition: str
    max_retries: int = 2
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Subtask":
        value = dict(data)
        value["status"] = StepStatus(value.get("status", StepStatus.PENDING))
        return cls(**value)


@dataclass
class TaskPlan:
    task: str
    subtasks: list[Subtask]

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if not self.subtasks:
            raise ValueError("a plan must contain at least one subtask")
        ids = [step.id for step in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("subtask ids must be unique")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskPlan":
        return cls(
            task=data["task"],
            subtasks=[Subtask.from_dict(item) for item in data["subtasks"]],
        )


@dataclass
class SubtaskReport:
    status: ReportStatus
    state_updates: dict[str, str] = field(default_factory=dict)
    message: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtaskReport":
        return cls(
            status=ReportStatus(data["status"]),
            state_updates=dict(data.get("state_updates", {})),
            message=data.get("message", ""),
        )


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)

