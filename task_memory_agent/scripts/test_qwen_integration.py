from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from upper.qwen_agent import StructuredQwenPlanner
from upper.qwen_transformers import LocalQwen3VLGenerator


def run_case(
    planner: StructuredQwenPlanner,
    name: str,
    task: str,
    image: str | None,
) -> dict:
    started = time.monotonic()
    plan = planner.plan(task, image)
    return {
        "name": name,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image": image,
        "plan": asdict(plan),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL upper-layer integration test")
    parser.add_argument(
        "--model", default="/home/user/ljn/models/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--image",
        default=(
            "/home/user/ljn/RoboTwin/XPolicyLab/policy/LingBot_VA/"
            "lingbot_va/example/robotwin/observation.images.cam_high.png"
        ),
    )
    parser.add_argument("--output", default="logs/qwen_integration_result.json")
    args = parser.parse_args()

    generator = LocalQwen3VLGenerator(args.model)
    planner = StructuredQwenPlanner(generator)
    cases = [
        run_case(
            planner,
            "text_plan",
            "take the cup out, place it on the table, and close the cabinet",
            None,
        ),
        run_case(
            planner,
            "visual_plan",
            "inspect the scene, pick up the target object, and place it safely",
            args.image,
        ),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"status": "success", "cases": cases}, indent=2, default=str),
        encoding="utf-8",
    )
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
