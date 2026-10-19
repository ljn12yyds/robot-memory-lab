#!/usr/bin/env python3
"""Split every recorded RoboTwin trajectory into subtask-level HDF5 episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def copy_group(
    source: h5py.Group,
    target: h5py.Group,
    start: int,
    end: int,
    sample_count: int,
    instruction: str,
) -> None:
    for name, item in source.items():
        if isinstance(item, h5py.Group):
            child = target.create_group(name)
            for key, value in item.attrs.items():
                child.attrs[key] = value
            copy_group(item, child, start, end, sample_count, instruction)
            continue
        if item.name == "/instructions":
            data = json.dumps([instruction]).encode("utf-8")
        elif item.shape and item.shape[0] == sample_count:
            data = item[start:end]
        else:
            data = item[()]
        dataset = target.create_dataset(name, data=data)
        for key, value in item.attrs.items():
            dataset.attrs[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    scene_info = json.loads((args.source_root / "scene_info.json").read_text(encoding="utf-8"))
    output_data = args.output_root / "data"
    output_data.mkdir(parents=True, exist_ok=False)
    manifest = []
    output_index = 0

    for source_index in range(len(scene_info)):
        episode_key = f"episode_{source_index}"
        metadata = scene_info[episode_key]
        source_path = args.source_root / "data" / f"episode_{source_index:07d}.hdf5"
        with h5py.File(source_path, "r") as source:
            sample_count = len(source["action/left_arm_joint_states"])
            segments = metadata["subtask_segments"]
            previous_end = 0
            for segment in segments:
                start = int(segment["start"])
                end = min(int(segment["end"]), sample_count)
                if start != previous_end or end <= start:
                    raise ValueError(f"invalid segments for {episode_key}: {segments}")
                output_path = output_data / f"episode_{output_index:07d}.hdf5"
                with h5py.File(output_path, "w") as target:
                    for key, value in source.attrs.items():
                        target.attrs[key] = value
                    copy_group(
                        source,
                        target,
                        start,
                        end,
                        sample_count,
                        str(segment["instruction"]),
                    )
                    target.attrs["source_episode"] = source_index
                    target.attrs["subtask_id"] = int(segment["subtask_id"])
                    target.attrs["stack_order"] = json.dumps(metadata["stack_order"])
                    target.attrs["full_task"] = str(metadata["task"])
                    target.attrs["source_start"] = start
                    target.attrs["source_end"] = end
                manifest.append(
                    {
                        "episode": output_index,
                        "source_episode": source_index,
                        "subtask_id": int(segment["subtask_id"]),
                        "stack_order": metadata["stack_order"],
                        "full_task": metadata["task"],
                        "instruction": segment["instruction"],
                        "samples": end - start,
                    }
                )
                output_index += 1
                previous_end = end
            if previous_end != sample_count:
                raise ValueError(
                    f"segments for {episode_key} end at {previous_end}, data ends at {sample_count}"
                )

    payload = {
        "source_episodes": len(scene_info),
        "subtask_episodes": len(manifest),
        "total_samples": sum(item["samples"] for item in manifest),
        "episodes": manifest,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in payload if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
