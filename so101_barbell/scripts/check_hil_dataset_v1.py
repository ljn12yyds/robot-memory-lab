#!/usr/bin/env python3
"""Validate one collected HIL Dataset v1 directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.utils.hil_dataset_v1 import (
    repair_verified_uncommitted_episodes,
    validate_hil_dataset_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--check-videos",
        action="store_true",
        help="Decode the first frame of every MP4 in addition to metadata checks.",
    )
    parser.add_argument(
        "--allow-uncommitted",
        action="store_true",
        help="Useful only for crash forensics; normal acceptance requires committed episodes.",
    )
    parser.add_argument(
        "--repair-uncommitted",
        action="store_true",
        help=(
            "After a full successful post-finalize check, append only missing validated/committed "
            "journal states whose latest state is lerobot_saved."
        ),
    )
    args = parser.parse_args()

    if args.repair_uncommitted:
        repaired = repair_verified_uncommitted_episodes(
            args.dataset_root,
            check_videos=args.check_videos,
        )
        print(json.dumps({"repaired_episode_uuids": repaired}, ensure_ascii=False, indent=2))

    result = validate_hil_dataset_v1(
        args.dataset_root,
        require_all_committed=not args.allow_uncommitted,
        check_videos=args.check_videos,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors"]:
        print(f"FAILED: {len(result['errors'])} integrity error(s).")
        return 1
    print(f"PASSED with {len(result['warnings'])} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
