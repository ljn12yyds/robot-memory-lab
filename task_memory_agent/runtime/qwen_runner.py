from __future__ import annotations

import argparse
import json

from upper.qwen_agent import StructuredQwenPlanner
from upper.qwen_transformers import LocalQwen3VLGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Qwen3-VL planning request")
    parser.add_argument("--task", required=True)
    parser.add_argument("--image")
    parser.add_argument(
        "--model", default="/home/user/ljn/models/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    generator = LocalQwen3VLGenerator(args.model, args.max_new_tokens)
    planner = StructuredQwenPlanner(generator)
    plan = planner.plan(args.task, args.image)
    print(json.dumps({"task": plan.task, "subtasks": [vars(s) for s in plan.subtasks]}, default=str, indent=2))


if __name__ == "__main__":
    main()

