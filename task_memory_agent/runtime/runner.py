from __future__ import annotations

import argparse
import json

from lower_adapter.mock_lower import MockLowerAdapter
from upper.planner import CabinetCupPlanner
from upper.task_memory import JsonTaskMemoryStore

from .agent_loop import AgentLoop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the upper-layer V1 smoke task")
    parser.add_argument("--task", required=True)
    parser.add_argument("--memory", default="runs/task_memory.json")
    parser.add_argument("--max-dispatches", type=int)
    args = parser.parse_args()

    loop = AgentLoop(
        planner=CabinetCupPlanner(),
        lower=MockLowerAdapter(),
        store=JsonTaskMemoryStore(args.memory),
    )
    memory = loop.run(args.task, max_dispatches=args.max_dispatches)
    print(json.dumps(memory.to_dict(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

