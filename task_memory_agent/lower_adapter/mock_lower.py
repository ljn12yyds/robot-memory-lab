from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from upper.schemas import ReportStatus, SubtaskReport


DEFAULT_UPDATES = {
    "open the cabinet": {"cabinet": "open"},
    "grasp the cup": {"cup": "held", "gripper": "holding_cup"},
    "take the cup out": {"cup": "outside_cabinet"},
    "place the cup on the table": {"cup": "on_table", "gripper": "empty"},
    "close the cabinet": {"cabinet": "closed"},
}


class MockLowerAdapter:
    """Scriptable lower layer. It receives only a subtask string."""

    def __init__(
        self,
        scripted_statuses: dict[str, Iterable[ReportStatus]] | None = None,
    ) -> None:
        self._statuses = {
            name: deque(values) for name, values in (scripted_statuses or {}).items()
        }
        self.calls: dict[str, int] = defaultdict(int)

    def execute(self, subtask: str) -> SubtaskReport:
        self.calls[subtask] += 1
        queue = self._statuses.get(subtask)
        status = queue.popleft() if queue else ReportStatus.SUCCESS
        updates = DEFAULT_UPDATES.get(subtask, {}) if status is ReportStatus.SUCCESS else {}
        return SubtaskReport(
            status=status,
            state_updates=updates,
            message=f"mock result for {subtask}",
        )

