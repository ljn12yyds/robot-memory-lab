#!/usr/bin/env python3
"""Read-only preflight audit for the RoboTwin PI0.5 training dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def joint_vector(group: h5py.Group) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(group["left_arm_joint_states"], dtype=np.float32),
            np.asarray(group["left_ee_joint_states"], dtype=np.float32),
            np.asarray(group["right_arm_joint_states"], dtype=np.float32),
            np.asarray(group["right_ee_joint_states"], dtype=np.float32),
        ],
        axis=1,
    )


def check(name: str, passed: bool, details: dict | None = None) -> dict:
    return {"name": name, "status": "PASS" if passed else "FAIL", "details": details or {}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--scene-info", type=Path, required=True)
    parser.add_argument("--raw-episodes", type=int, default=30)
    parser.add_argument("--subtasks-per-episode", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-episode-max-exclusive", type=int)
    args = parser.parse_args()

    info = json.loads((args.dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    reported_stats = json.loads(
        (args.dataset_root / "meta/stats.json").read_text(encoding="utf-8")
    )
    parquet_paths = sorted((args.dataset_root / "data").glob("**/*.parquet"))
    table = pa.concat_tables([pq.read_table(path) for path in parquet_paths])
    frame = table.to_pandas().sort_values("index").reset_index(drop=True)
    actions = np.asarray(frame["action"].tolist(), dtype=np.float32)
    states = np.asarray(frame["observation.state"].tolist(), dtype=np.float32)
    episode_ids = frame["episode_index"].to_numpy(dtype=np.int64)
    task_ids = frame["task_index"].to_numpy(dtype=np.int64)

    checks: list[dict] = []
    expected_episode_ids = np.arange(info["total_episodes"])
    observed_episode_ids = np.unique(episode_ids)
    checks.append(
        check(
            "metadata_counts",
            len(frame) == info["total_frames"]
            and len(observed_episode_ids) == info["total_episodes"]
            and np.array_equal(observed_episode_ids, expected_episode_ids),
            {
                "frames": len(frame),
                "reported_frames": info["total_frames"],
                "episodes": len(observed_episode_ids),
                "reported_episodes": info["total_episodes"],
            },
        )
    )
    checks.append(
        check(
            "finite_shapes",
            actions.shape == (len(frame), 14)
            and states.shape == (len(frame), 14)
            and np.isfinite(actions).all()
            and np.isfinite(states).all(),
            {"action_shape": list(actions.shape), "state_shape": list(states.shape)},
        )
    )

    episode_lengths: list[int] = []
    episode_errors: list[str] = []
    temporal_errors: list[float] = []
    for episode_id in observed_episode_ids:
        rows = frame[episode_ids == episode_id].sort_values("frame_index")
        length = len(rows)
        episode_lengths.append(length)
        if not np.array_equal(rows["frame_index"].to_numpy(), np.arange(length)):
            episode_errors.append(f"episode {episode_id}: non-contiguous frame_index")
        if rows["task_index"].nunique() != 1:
            episode_errors.append(f"episode {episode_id}: multiple task indices")
        ep_actions = np.asarray(rows["action"].tolist(), dtype=np.float32)
        ep_states = np.asarray(rows["observation.state"].tolist(), dtype=np.float32)
        if length > 1:
            temporal_errors.append(float(np.max(np.abs(ep_actions[:-1] - ep_states[1:]))))
    checks.append(
        check(
            "episode_boundaries_and_labels",
            not episode_errors and min(episode_lengths) >= 50,
            {
                "min_length": min(episode_lengths),
                "max_length": max(episode_lengths),
                "episodes_shorter_than_chunk": int(np.sum(np.asarray(episode_lengths) < 50)),
                "errors": episode_errors[:20],
            },
        )
    )
    max_temporal_error = max(temporal_errors)
    checks.append(
        check(
            "state_action_t_plus_one_alignment",
            max_temporal_error < 1e-6,
            {"max_abs_error": max_temporal_error},
        )
    )

    raw_state_errors: list[float] = []
    raw_action_errors: list[float] = []
    raw_temporal_errors: list[float] = []
    raw_length_errors: list[str] = []
    for raw_episode in range(args.raw_episodes):
        path = args.raw_data / f"episode_{raw_episode:07d}.hdf5"
        with h5py.File(path, "r") as source:
            raw_states = joint_vector(source["state"])
            raw_actions = joint_vector(source["action"])
        converted_ids = range(
            raw_episode * args.subtasks_per_episode,
            (raw_episode + 1) * args.subtasks_per_episode,
        )
        converted = frame[frame["episode_index"].isin(converted_ids)].sort_values("index")
        converted_states = np.asarray(converted["observation.state"].tolist(), dtype=np.float32)
        converted_actions = np.asarray(converted["action"].tolist(), dtype=np.float32)
        if len(converted) != len(raw_states):
            raw_length_errors.append(
                f"raw {raw_episode}: converted={len(converted)} raw={len(raw_states)}"
            )
            continue
        raw_state_errors.append(float(np.max(np.abs(converted_states - raw_states))))
        raw_action_errors.append(float(np.max(np.abs(converted_actions - raw_actions))))
        raw_temporal_errors.append(float(np.max(np.abs(raw_actions[:-1] - raw_states[1:]))))
    raw_max_state_error = max(raw_state_errors, default=float("inf"))
    raw_max_action_error = max(raw_action_errors, default=float("inf"))
    checks.append(
        check(
            "raw_to_lerobot_lossless_alignment",
            not raw_length_errors and raw_max_state_error < 1e-6 and raw_max_action_error < 1e-6,
            {
                "max_state_error": raw_max_state_error,
                "max_action_error": raw_max_action_error,
                "raw_t_plus_one_max_error": max(raw_temporal_errors, default=float("inf")),
                "length_errors": raw_length_errors[:20],
            },
        )
    )

    stats_actions = actions
    if args.stats_episode_max_exclusive is not None:
        stats_actions = actions[episode_ids < args.stats_episode_max_exclusive]
    quantiles = np.quantile(stats_actions, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    keys = ["q01", "q10", "q50", "q90", "q99"]
    reported = np.asarray([reported_stats["action"][key] for key in keys], dtype=np.float64)
    quantile_error = np.abs(reported - quantiles)
    checks.append(
        check(
            "global_action_quantiles",
            float(quantile_error.max()) < 1e-3,
            {
                "max_abs_error": float(quantile_error.max()),
                "bad_dimensions_q01": np.flatnonzero(quantile_error[0] >= 1e-3).astype(int).tolist(),
                "bad_dimensions_q99": np.flatnonzero(quantile_error[-1] >= 1e-3).astype(int).tolist(),
                "empirical_q01": quantiles[0].astype(float).tolist(),
                "empirical_q99": quantiles[-1].astype(float).tolist(),
                "stats_frame_count": len(stats_actions),
            },
        )
    )

    unique_tasks = np.unique(task_ids)
    tasks_table = pq.read_table(args.dataset_root / "meta/tasks.parquet").to_pandas()
    task_records = [
        {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}
        for row in tasks_table.reset_index().to_dict(orient="records")
    ]
    checks.append(
        check(
            "task_index_coverage",
            np.array_equal(unique_tasks, np.arange(info["total_tasks"]))
            and len(tasks_table) == info["total_tasks"],
            {
                "observed_task_indices": unique_tasks.astype(int).tolist(),
                "reported_tasks": info["total_tasks"],
                "task_rows": len(tasks_table),
                "tasks": task_records,
                "frames_per_task": {
                    str(int(task_id)): int(np.sum(task_ids == task_id)) for task_id in unique_tasks
                },
            },
        )
    )

    scene_info = json.loads(args.scene_info.read_text(encoding="utf-8"))
    annotation_errors: list[str] = []
    source_instruction_mismatches: list[dict] = []
    for raw_episode in range(args.raw_episodes):
        scene_episode = scene_info[f"episode_{raw_episode}"]
        with h5py.File(args.raw_data / f"episode_{raw_episode:07d}.hdf5", "r") as source:
            raw_length = len(source["state/left_arm_joint_states"])
        for stage, segment in enumerate(scene_episode["subtask_segments"]):
            converted_episode = raw_episode * args.subtasks_per_episode + stage
            rows = frame[frame["episode_index"] == converted_episode]
            task_id = int(rows["task_index"].iloc[0])
            actual_instruction = str(tasks_table.index[task_id])
            expected_length = min(int(segment["end"]), raw_length) - int(segment["start"])
            if actual_instruction != segment["instruction"] or len(rows) != expected_length:
                annotation_errors.append(
                    f"episode {converted_episode}: instruction={actual_instruction!r}, "
                    f"expected={segment['instruction']!r}, length={len(rows)}, "
                    f"expected_length={expected_length}"
                )

        instruction_path = args.scene_info.parent / "instruction" / f"episode_{raw_episode:07d}.json"
        generated = json.loads(instruction_path.read_text(encoding="utf-8"))
        expected_order = scene_episode["stack_order"]
        for split in ("seen", "unseen"):
            for sentence in generated[split]:
                observed_order = sorted(
                    ("red", "green", "blue"), key=lambda color: sentence.find(f"{color} block")
                )
                if observed_order != expected_order:
                    source_instruction_mismatches.append(
                        {
                            "raw_episode": raw_episode,
                            "split": split,
                            "expected_order": expected_order,
                            "observed_mention_order": observed_order,
                            "instruction": sentence,
                        }
                    )
    checks.append(
        check(
            "scene_info_subtask_annotation_alignment",
            not annotation_errors,
            {"errors": annotation_errors[:20]},
        )
    )
    source_instruction_check = check(
        "source_episode_instruction_order",
        not source_instruction_mismatches,
        {
            "mismatch_count": len(source_instruction_mismatches),
            "mismatches": source_instruction_mismatches[:12],
            "used_by_current_subtask_dataset": False,
        },
    )
    if source_instruction_mismatches:
        source_instruction_check["status"] = "WARN"
    checks.append(source_instruction_check)

    hdf5_inventory: dict[str, list[int]] = {}
    with h5py.File(args.raw_data / "episode_0000000.hdf5", "r") as source:
        source.visititems(
            lambda name, obj: hdf5_inventory.__setitem__(name, list(obj.shape))
            if isinstance(obj, h5py.Dataset)
            else None
        )

    video_files = sorted((args.dataset_root / "videos").glob("**/*.mp4"))
    expected_video_keys = {
        key for key, value in info["features"].items() if value["dtype"] == "video"
    }
    found_video_keys = {path.parent.parent.name for path in video_files}
    checks.append(
        check(
            "video_files_present",
            expected_video_keys == found_video_keys and len(video_files) == len(expected_video_keys),
            {
                "expected_keys": sorted(expected_video_keys),
                "found_keys": sorted(found_video_keys),
                "video_files": [str(path) for path in video_files],
            },
        )
    )

    payload = {
        "dataset_root": str(args.dataset_root),
        "raw_data": str(args.raw_data),
        "raw_episode0_hdf5_datasets": hdf5_inventory,
        "status": "FAIL" if any(item["status"] == "FAIL" for item in checks) else "PASS",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
