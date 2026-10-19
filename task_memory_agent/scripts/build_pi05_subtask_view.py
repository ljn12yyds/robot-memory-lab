#!/usr/bin/env python3
"""Create a LeRobot dataset view conditioned on the active language subtask."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.subtask_annotations import SubtaskAnnotations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def read_episode_lengths(source: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    for path in sorted((source / "meta/episodes").glob("**/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "length"])
        for row in table.to_pylist():
            result[int(row["episode_index"])] = int(row["length"])
    if not result:
        raise ValueError(f"no episode metadata found in {source}")
    return result


def validate_source(source: Path, annotations: SubtaskAnnotations) -> dict[int, int]:
    lengths = read_episode_lengths(source)
    missing = sorted(set(lengths) - set(annotations.episodes))
    extra = sorted(set(annotations.episodes) - set(lengths))
    if missing:
        raise ValueError(f"missing annotations for episodes: {missing[:20]}")
    if extra:
        raise ValueError(f"annotations reference absent episodes: {extra[:20]}")
    for episode, length in lengths.items():
        annotations.validate_episode(episode, length)
    return lengths


def write_index_table(path: Path, instructions: list[str], column: str) -> None:
    frame = pd.DataFrame({column: range(len(instructions))}, index=instructions)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def rewrite_data(source: Path, output: Path, annotations: SubtaskAnnotations) -> None:
    for path in sorted((source / "data").glob("**/*.parquet")):
        frame = pq.read_table(path).to_pandas()
        indices = pd.Series(index=frame.index, dtype="int64")
        for episode, rows in frame.groupby("episode_index", sort=False).groups.items():
            positions = list(rows)
            frames = frame.loc[positions, "frame_index"].astype(int).tolist()
            indices.loc[positions] = annotations.indices_for_frames(int(episode), frames)
        frame["subtask_index"] = indices.astype("int64")
        # PI0.5 already consumes the LeRobot task string. Making task_index equal the
        # active subtask preserves the strict upper-to-lower subtask-only boundary.
        frame["task_index"] = frame["subtask_index"]
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False, compression="zstd")


def rewrite_episode_metadata(
    source: Path, output: Path, annotations: SubtaskAnnotations
) -> None:
    by_id = annotations.subtask_by_id
    for path in sorted((source / "meta/episodes").glob("**/*.parquet")):
        frame = pq.read_table(path).to_pandas()
        frame["tasks"] = frame["episode_index"].map(
            lambda episode: [
                by_id[segment.subtask_id].instruction
                for segment in annotations.episodes[int(episode)]
            ]
        )
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False, compression="zstd")


def copy_metadata(source: Path, output: Path, annotations: SubtaskAnnotations) -> None:
    output_meta = output / "meta"
    output_meta.mkdir(parents=True)
    for path in (source / "meta").iterdir():
        if path.name in {"episodes", "tasks.parquet", "subtasks.parquet", "info.json"}:
            continue
        destination = output_meta / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)

    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    task_feature = dict(info["features"]["task_index"])
    info["features"]["subtask_index"] = task_feature
    info["total_tasks"] = len(annotations.subtasks)
    info["total_subtasks"] = len(annotations.subtasks)
    (output_meta / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    instructions = [subtask.instruction for subtask in annotations.subtasks]
    write_index_table(output_meta / "tasks.parquet", instructions, "task_index")
    write_index_table(output_meta / "subtasks.parquet", instructions, "subtask_index")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    annotations = SubtaskAnnotations.load(args.annotations)
    lengths = validate_source(source, annotations)
    if args.validate_only:
        print(json.dumps({**annotations.to_summary(), "frames": sum(lengths.values())}, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    output.mkdir(parents=True)
    try:
        copy_metadata(source, output, annotations)
        rewrite_episode_metadata(source, output, annotations)
        rewrite_data(source, output, annotations)
        os.symlink(source / "videos", output / "videos", target_is_directory=True)
        provenance = {
            "schema": "pi05_subtask_dataset_v1",
            "source": str(source),
            "annotations": str(args.annotations.resolve()),
            **annotations.to_summary(),
            "episodes": len(lengths),
            "frames": sum(lengths.values()),
            "video_storage": "symlink",
        }
        (output / "subtask_provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output)
        raise
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

