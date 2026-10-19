from __future__ import annotations

from typing import Protocol

from .schemas import Subtask, TaskPlan


class Planner(Protocol):
    def plan(self, task: str, observation: str | None = None) -> TaskPlan: ...


class CabinetCupPlanner:
    """Deterministic V1 planner used to test orchestration without a VLM."""

    def plan(self, task: str, observation: str | None = None) -> TaskPlan:
        normalized = task.lower()
        if "cup" not in normalized or "cabinet" not in normalized:
            raise ValueError(
                "The deterministic planner only supports the cabinet/cup smoke task"
            )
        names = [
            ("open the cabinet", "cabinet is open"),
            ("grasp the cup", "cup is held by the gripper"),
            ("take the cup out", "cup is outside the cabinet"),
            ("place the cup on the table", "cup is on the table"),
            ("close the cabinet", "cabinet is closed"),
        ]
        return TaskPlan(
            task=task,
            subtasks=[
                Subtask(id=index, name=name, success_condition=condition)
                for index, (name, condition) in enumerate(names)
            ],
        )

