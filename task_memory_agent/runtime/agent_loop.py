from __future__ import annotations

from typing import Protocol

from upper.planner import Planner
from upper.schemas import SubtaskReport
from upper.task_memory import JsonTaskMemoryStore, TaskMemory


class LowerAdapter(Protocol):
    def execute(self, subtask: str) -> SubtaskReport: ...


class AgentLoop:
    def __init__(
        self,
        planner: Planner,
        lower: LowerAdapter,
        store: JsonTaskMemoryStore,
    ) -> None:
        self.planner = planner
        self.lower = lower
        self.store = store

    def initialize(self, task: str, observation: str | None = None) -> TaskMemory:
        if self.store.exists():
            memory = self.store.load()
            if memory.plan.task != task:
                raise ValueError("memory file belongs to a different task")
            return memory
        memory = TaskMemory(plan=self.planner.plan(task, observation))
        self.store.save(memory)
        return memory

    def run(
        self,
        task: str,
        observation: str | None = None,
        max_dispatches: int | None = None,
    ) -> TaskMemory:
        memory = self.initialize(task, observation)
        dispatches = 0
        while not memory.finished and not memory.blocked:
            if max_dispatches is not None and dispatches >= max_dispatches:
                break
            step = memory.begin_current()
            self.store.save(memory)

            # This string is the complete upper-to-lower payload in V1.
            report = self.lower.execute(step.name)
            memory.apply_report(report)
            self.store.save(memory)
            dispatches += 1
        return memory

