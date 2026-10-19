from __future__ import annotations

import json
from collections.abc import Callable

from .schemas import TaskPlan


SYSTEM_PROMPT = """You are the task-level planner for a robot.
Return JSON only. Decompose the user task into ordered semantic subtasks.
Each subtask must contain id, name, success_condition, and max_retries.
Never emit joint commands, poses, trajectories, grasp points, or action advice.
"""


class StructuredQwenPlanner:
    """Adapter for a local or remote Qwen3-VL text-generation callable."""

    def __init__(self, generate: Callable[[str, str, str | None], str]) -> None:
        self.generate = generate

    def plan(self, task: str, observation: str | None = None) -> TaskPlan:
        user_prompt = json.dumps(
            {"task": task, "initial_observation": observation}, ensure_ascii=False
        )
        raw = self.generate(SYSTEM_PROMPT, user_prompt, observation)
        payload = _extract_json(raw)
        return TaskPlan.from_dict(payload)


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1])
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("planner output does not contain a JSON object")
    return json.loads(cleaned[start : end + 1])

