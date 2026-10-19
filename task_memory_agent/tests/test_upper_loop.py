from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lower_adapter.mock_lower import MockLowerAdapter
from runtime.agent_loop import AgentLoop
from upper.planner import CabinetCupPlanner
from upper.schemas import ReportStatus, StepStatus
from upper.task_memory import JsonTaskMemoryStore


TASK = "take the cup out, place it on the table, and close the cabinet"


class UpperLoopTests(unittest.TestCase):
    def make_loop(self, root: Path, lower: MockLowerAdapter) -> AgentLoop:
        return AgentLoop(
            planner=CabinetCupPlanner(),
            lower=lower,
            store=JsonTaskMemoryStore(root / "task.json"),
        )

    def test_happy_path_completes_and_updates_world_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = self.make_loop(Path(directory), MockLowerAdapter()).run(TASK)
            self.assertTrue(memory.finished)
            self.assertFalse(memory.blocked)
            self.assertEqual(memory.world_state["cabinet"], "closed")
            self.assertEqual(memory.world_state["cup"], "on_table")
            self.assertTrue(
                all(step.status is StepStatus.COMPLETED for step in memory.plan.subtasks)
            )

    def test_retryable_failure_retries_same_subtask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lower = MockLowerAdapter(
                {
                    "grasp the cup": [
                        ReportStatus.RETRYABLE_FAILURE,
                        ReportStatus.SUCCESS,
                    ]
                }
            )
            memory = self.make_loop(Path(directory), lower).run(TASK)
            self.assertTrue(memory.finished)
            self.assertEqual(lower.calls["grasp the cup"], 2)

    def test_blocked_report_stops_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lower = MockLowerAdapter(
                {"grasp the cup": [ReportStatus.BLOCKED]}
            )
            memory = self.make_loop(Path(directory), lower).run(TASK)
            self.assertTrue(memory.blocked)
            self.assertEqual(memory.current_subtask.name, "grasp the cup")
            self.assertEqual(lower.calls["take the cup out"], 0)

    def test_resume_uses_persisted_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_lower = MockLowerAdapter()
            first = self.make_loop(root, first_lower).run(TASK, max_dispatches=2)
            self.assertEqual(first.current_step, 2)

            second_lower = MockLowerAdapter()
            resumed = self.make_loop(root, second_lower).run(TASK)
            self.assertTrue(resumed.finished)
            self.assertEqual(second_lower.calls["open the cabinet"], 0)
            self.assertEqual(second_lower.calls["grasp the cup"], 0)


if __name__ == "__main__":
    unittest.main()

