#!/usr/bin/env python3
"""Build a validated, non-destructive training view over all PI0.5 HIL data.

The source datasets remain canonical.  This script writes compact Parquet
indices which point back to their source dataset/episode/frame and distinguish
positive demonstrations from policy risk/failure data.  It deliberately does
not flatten a failure followed by recovery into ordinary behavior cloning:
policy mistakes and successful human actions have different training roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import av
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVORL_ROOT = PROJECT_ROOT.parent / "Evo-RL"
DEFAULT_SOURCE_ROOTS = (
    PROJECT_ROOT / "data/rl_rollouts/pi05_hil_v1",
    PROJECT_ROOT / "data/rl_rollouts/pi05_human_inloop",
    PROJECT_ROOT / "data/rl_rollouts/pi05_human_inloop_30k_v2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        dest="source_roots",
        help="May be repeated; defaults to all known PI0.5 HIL roots.",
    )
    parser.add_argument(
        "--risk-window-frames",
        type=int,
        default=90,
        help="Policy frames immediately before intervention marked negative.",
    )
    parser.add_argument(
        "--legacy-object-id",
        default="blue_barbell_01",
        help="Object label inferred for legacy batches which predate HIL v1.",
    )
    parser.add_argument(
        "--legacy-setup-id",
        default="barbell_left_tray_right_v1",
        help="Setup label inferred for legacy batches which predate HIL v1.",
    )
    parser.add_argument(
        "--skip-video-decode-check",
        action="store_true",
        help="Only for debugging; normal preparation decodes every source video.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_parquet_rows(paths: list[Path], columns: list[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows


def scalar_bool(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else 0
    return bool(float(value)) if value is not None else False


def decode_first_video_frame(path: Path) -> None:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if next(container.decode(stream), None) is None:
            raise ValueError(f"video has no decodable frame: {path}")


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for rel in ("meta/info.json", "hil/manifest.json", "hil/episodes.jsonl"):
        path = root / rel
        if path.is_file():
            digest.update(rel.encode())
            digest.update(path.read_bytes())
    for path in sorted((root / "meta/episodes").glob("**/*.parquet")):
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def validate_source(root: Path, *, check_videos: bool) -> tuple[str, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        return "unknown", ["missing meta/info.json"], warnings
    info = read_json(info_path)
    if int(info.get("total_episodes", 0)) <= 0:
        return "empty", ["zero episodes"], warnings

    is_v1 = (root / "hil/manifest.json").is_file()
    generation = "hil_v1" if is_v1 else "legacy"
    data_files = sorted((root / "data").glob("**/*.parquet"))
    episode_files = sorted((root / "meta/episodes").glob("**/*.parquet"))
    if not data_files:
        errors.append("missing data parquet")
    if not episode_files:
        errors.append("missing episode metadata parquet")
    try:
        frame_rows = sum(pq.read_metadata(path).num_rows for path in data_files)
        if frame_rows != int(info.get("total_frames", -1)):
            errors.append(f"frame count mismatch: parquet={frame_rows}, info={info.get('total_frames')}")
    except Exception as error:
        errors.append(f"unreadable data parquet: {error}")
    try:
        episode_rows = sum(pq.read_metadata(path).num_rows for path in episode_files)
        if episode_rows != int(info.get("total_episodes", -1)):
            errors.append(
                f"episode count mismatch: parquet={episode_rows}, info={info.get('total_episodes')}"
            )
    except Exception as error:
        errors.append(f"unreadable episode parquet: {error}")

    if is_v1:
        sys.path.insert(0, str(EVORL_ROOT / "src"))
        from lerobot.utils.hil_dataset_v1 import validate_hil_dataset_v1

        result = validate_hil_dataset_v1(
            root,
            require_all_committed=True,
            check_videos=check_videos,
        )
        errors.extend(result["errors"])
        warnings.extend(result["warnings"])
    elif check_videos:
        for video in sorted((root / "videos").glob("**/*.mp4")):
            try:
                decode_first_video_frame(video)
            except Exception as error:
                errors.append(str(error))
    return generation, errors, warnings


def normalize_outcome(value: Any) -> str:
    if isinstance(value, bool):
        return "success" if value else "failure"
    text = str(value).strip().lower()
    return "success" if text in {"success", "true", "1"} else "failure"


def frame_role(
    *,
    outcome: str,
    assistance: str,
    is_human: bool,
    frame_index: int,
    intervention_start: int | None,
    terminal_frame: int,
    risk_window: int,
) -> tuple[str, bool, bool, float]:
    """Return role, BC eligibility, value eligibility and initial advantage."""
    success = outcome == "success"
    if assistance == "none" and success:
        return "autonomous_success_positive", True, True, 1.0
    if is_human:
        if not success:
            return "failed_human_recovery_negative", False, True, -1.0
        if assistance == "human_preventive":
            return "human_preventive_positive", True, True, 1.0
        return "human_recovery_positive", True, True, 1.0

    boundary = intervention_start
    if boundary is None and not success:
        boundary = terminal_frame + 1
    if boundary is not None and frame_index >= max(0, boundary - risk_window):
        return "policy_risk_negative", False, True, -0.5 if success else -1.0
    if not success:
        return "failed_policy_context", False, True, -1.0
    return "policy_context_neutral", False, True, 0.0


def build_source_rows(
    root: Path,
    generation: str,
    *,
    global_episode_start: int,
    risk_window: int,
    legacy_object_id: str,
    legacy_setup_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    info = read_json(root / "meta/info.json")
    episode_meta = read_parquet_rows(sorted((root / "meta/episodes").glob("**/*.parquet")))
    episode_meta_by_id = {int(row["episode_index"]): row for row in episode_meta}
    data_files = sorted((root / "data").glob("**/*.parquet"))
    data_schema = pq.read_schema(data_files[0])
    frame_columns = ["index", "episode_index", "frame_index", "complementary_info.is_intervention"]
    if generation == "hil_v1":
        frame_columns.extend(
            ["hil.episode_uuid", "hil.control_source", "hil.continuity_segment_id"]
        )
        hil_episodes = read_parquet_rows([root / "hil/episodes.parquet"])
        hil_episode_by_id = {int(row["episode_index"]): row for row in hil_episodes}
        manifest = read_json(root / "hil/manifest.json")
    else:
        hil_episode_by_id = {}
        manifest = {}
    missing = [name for name in frame_columns if name not in data_schema.names]
    if missing:
        raise ValueError(f"{root} lacks required frame columns: {missing}")
    frames = read_parquet_rows(data_files, frame_columns)
    frames_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in frames:
        frames_by_episode.setdefault(int(row["episode_index"]), []).append(row)

    source_key = root.name
    source_hash = source_fingerprint(root)
    episode_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for source_episode in sorted(frames_by_episode):
        source_frames = sorted(frames_by_episode[source_episode], key=lambda row: int(row["frame_index"]))
        meta = episode_meta_by_id[source_episode]
        if generation == "hil_v1":
            hmeta = hil_episode_by_id[source_episode]
            episode_uuid = str(hmeta["episode_uuid"])
            session_id = str(hmeta["session_id"])
            outcome = normalize_outcome(hmeta["task_outcome"])
            assistance = str(hmeta["assistance"])
            object_id = str(hmeta["object_id"])
            setup_id = str(hmeta["setup_id"])
            metadata_provenance = "recorded"
        else:
            episode_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{root.resolve()}#{source_episode}"))
            session_id = source_key
            outcome = normalize_outcome(meta.get("episode_success", "failure"))
            any_human = any(scalar_bool(row["complementary_info.is_intervention"]) for row in source_frames)
            assistance = "human_recovery" if any_human else "none"
            object_id = legacy_object_id
            setup_id = legacy_setup_id
            metadata_provenance = "inferred_legacy_scope"

        human_flags: list[bool] = []
        for frame in source_frames:
            if generation == "hil_v1":
                human_flags.append(str(frame["hil.control_source"]).startswith("human_"))
            else:
                human_flags.append(scalar_bool(frame["complementary_info.is_intervention"]))
        intervention_positions = [
            int(frame["frame_index"])
            for frame, is_human in zip(source_frames, human_flags, strict=True)
            if is_human
        ]
        intervention_start = min(intervention_positions) if intervention_positions else None
        terminal_frame = int(source_frames[-1]["frame_index"])
        global_episode = global_episode_start + len(episode_rows)
        episode_rows.append(
            {
                "global_episode_index": global_episode,
                "episode_uuid": episode_uuid,
                "session_id": session_id,
                "source_generation": generation,
                "source_dataset": source_key,
                "source_root": str(root.resolve()),
                "source_fingerprint": source_hash,
                "source_episode_index": source_episode,
                "frames": len(source_frames),
                "task_outcome": outcome,
                "assistance": assistance,
                "intervention_start_frame": intervention_start,
                "object_id": object_id,
                "setup_id": setup_id,
                "metadata_provenance": metadata_provenance,
                "task_text": (meta.get("tasks") or [""])[0],
                "bc_has_positive_frames": outcome == "success" and (
                    assistance == "none" or intervention_start is not None
                ),
                "value_eligible": True,
            }
        )

        previous_continuity_segment: int | None = None
        for frame, is_human in zip(source_frames, human_flags, strict=True):
            local_frame = int(frame["frame_index"])
            continuity_segment = (
                int(frame["hil.continuity_segment_id"])
                if generation == "hil_v1"
                else 0
            )
            role, bc_eligible, value_eligible, advantage = frame_role(
                outcome=outcome,
                assistance=assistance,
                is_human=is_human,
                frame_index=local_frame,
                intervention_start=intervention_start,
                terminal_frame=terminal_frame,
                risk_window=risk_window,
            )
            frame_rows.append(
                {
                    "global_episode_index": global_episode,
                    "episode_uuid": episode_uuid,
                    "source_generation": generation,
                    "source_dataset": source_key,
                    "source_root": str(root.resolve()),
                    "source_episode_index": source_episode,
                    "source_frame_index": local_frame,
                    "source_global_index": int(frame["index"]),
                    "control_source": "human" if is_human else "policy",
                    "continuity_segment_id": continuity_segment,
                    "transition_contiguous": local_frame > 0
                    and (
                        generation == "legacy"
                        or continuity_segment == previous_continuity_segment
                    ),
                    "training_role": role,
                    "bc_eligible": bc_eligible,
                    "value_eligible": value_eligible,
                    "initial_advantage": advantage,
                    "task_outcome": outcome,
                    "assistance": assistance,
                }
            )
            previous_continuity_segment = continuity_segment
    if len(episode_rows) != int(info["total_episodes"]):
        raise ValueError(f"{root}: constructed episode count mismatch")
    return episode_rows, frame_rows


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def main() -> None:
    args = parse_args()
    if args.risk_window_frames < 0:
        raise ValueError("--risk-window-frames must be non-negative")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source_roots = tuple(args.source_roots or DEFAULT_SOURCE_ROOTS)

    candidates = [
        path
        for base in source_roots
        for path in sorted(base.resolve().glob("eval_pi05_barbell_hil_*"))
        if path.is_dir()
    ]
    validations: list[dict[str, Any]] = []
    usable: list[tuple[Path, str]] = []
    for root in candidates:
        generation, errors, warnings = validate_source(
            root,
            check_videos=not args.skip_video_decode_check,
        )
        validations.append(
            {
                "source_root": str(root),
                "generation": generation,
                "usable": not errors,
                "errors": errors,
                "warnings": warnings,
            }
        )
        if not errors:
            usable.append((root, generation))

    if not usable:
        raise RuntimeError("No usable HIL datasets found")

    all_episodes: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    for root, generation in usable:
        episodes, frames = build_source_rows(
            root,
            generation,
            global_episode_start=len(all_episodes),
            risk_window=args.risk_window_frames,
            legacy_object_id=args.legacy_object_id,
            legacy_setup_id=args.legacy_setup_id,
        )
        all_episodes.extend(episodes)
        all_frames.extend(frames)

    role_counts = Counter(row["training_role"] for row in all_frames)
    outcome_counts = Counter(row["task_outcome"] for row in all_episodes)
    assistance_counts = Counter(row["assistance"] for row in all_episodes)
    generation_counts = Counter(row["source_generation"] for row in all_episodes)
    bc_frames = [row for row in all_frames if row["bc_eligible"]]
    value_frames = [row for row in all_frames if row["value_eligible"]]

    output.mkdir(parents=True)
    write_table(all_episodes, output / "episodes.parquet")
    write_table(all_frames, output / "frame_roles.parquet")
    write_table(bc_frames, output / "bc_positive_frames.parquet")
    write_table(value_frames, output / "value_frames.parquet")
    write_table(validations, output / "source_validation.parquet")

    manifest = {
        "schema": "pi05_hil_training_view_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output),
        "source_roots": [str(Path(path).resolve()) for path in source_roots],
        "source_dataset_count": len(usable),
        "excluded_source_count": len(candidates) - len(usable),
        "episodes": len(all_episodes),
        "frames": len(all_frames),
        "bc_positive_frames": len(bc_frames),
        "value_frames": len(value_frames),
        "risk_window_frames": args.risk_window_frames,
        "episode_counts_by_generation": dict(sorted(generation_counts.items())),
        "episode_counts_by_outcome": dict(sorted(outcome_counts.items())),
        "episode_counts_by_assistance": dict(sorted(assistance_counts.items())),
        "frame_counts_by_role": dict(sorted(role_counts.items())),
        "semantics": {
            "bc_positive_frames": "Only autonomous-success or successful human intervention frames.",
            "policy_risk_negative": "Policy frames in the risk window before intervention/failure; never BC targets.",
            "policy_context_neutral": "Earlier policy prefix in assisted success; retained for world/value context.",
            "legacy_assistance": "Legacy intervention is conservatively classified as human_recovery.",
            "source_of_truth": "Original LeRobot datasets and videos; this directory is an immutable index view.",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Final referential integrity checks on the materialized view.
    episode_ids = {row["global_episode_index"] for row in all_episodes}
    dangling = {row["global_episode_index"] for row in all_frames} - episode_ids
    if dangling:
        raise RuntimeError(f"Dangling frame episode references: {sorted(dangling)[:10]}")
    if len(episode_ids) != len(all_episodes):
        raise RuntimeError("Duplicate global episode indices")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
