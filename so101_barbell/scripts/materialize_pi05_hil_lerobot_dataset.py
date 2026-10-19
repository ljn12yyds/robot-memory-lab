#!/usr/bin/env python3
"""Materialize all validated PI0.5 HIL episodes as one LeRobotDataset.

The immutable training-view manifest determines which source datasets are
usable.  HIL v1 has additional telemetry columns which legacy sessions do not
have, so this script first creates lightweight, common-schema source views and
then delegates the actual merge to LeRobot's official aggregate_datasets().
Source videos are symlinked into the temporary views and copied byte-for-byte
by the aggregator; source datasets and HIL sidecars are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVORL_ROOT = PROJECT_ROOT.parent / "Evo-RL"
DEFAULT_VIEW = PROJECT_ROOT / "data/rl_rollouts/pi05_hil_training_v1_128ep_20260814"
DEFAULT_COMMON_ROOT = PROJECT_ROOT / "data/rl_rollouts/pi05_hil_common_sources_v1_20260814"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/rl_rollouts/pi05_hil_lerobot_v1_128ep_20260814"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--common-root", type=Path, default=DEFAULT_COMMON_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="barbell/pi05_hil_lerobot_v1_128ep_20260814")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_roots_from_view(training_view: Path) -> list[Path]:
    table = pq.read_table(training_view / "episodes.parquet", columns=["source_root"])
    roots: list[Path] = []
    for value in table.column("source_root").to_pylist():
        root = Path(value).resolve()
        if root not in roots:
            roots.append(root)
    return roots


def common_features(roots: list[Path]) -> tuple[list[str], dict[str, Any]]:
    infos = [read_json(root / "meta/info.json") for root in roots]
    names = set(infos[0]["features"])
    for info in infos[1:]:
        names.intersection_update(info["features"])
    ordered = [name for name in infos[0]["features"] if name in names]
    features = {name: infos[0]["features"][name] for name in ordered}
    for root, info in zip(roots, infos, strict=True):
        mismatched = [name for name in ordered if info["features"][name] != features[name]]
        if mismatched:
            raise ValueError(f"Common feature definitions differ in {root}: {mismatched}")
    return ordered, features


def materialize_common_source(
    source: Path,
    destination: Path,
    *,
    feature_names: list[str],
    features: dict[str, Any],
) -> None:
    destination.mkdir(parents=True)
    info = read_json(source / "meta/info.json")
    info["features"] = features
    write_json(destination / "meta/info.json", info)

    stats = read_json(source / "meta/stats.json")
    write_json(destination / "meta/stats.json", {key: stats[key] for key in feature_names if key in stats})
    shutil.copy2(source / "meta/tasks.parquet", destination / "meta/tasks.parquet")

    data_columns = set(pq.read_schema(next((source / "data").glob("**/*.parquet"))).names)
    selected_data_columns = [name for name in feature_names if name in data_columns]
    for src_path in sorted((source / "data").glob("**/*.parquet")):
        dst_path = destination / src_path.relative_to(source)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pq.read_table(src_path, columns=selected_data_columns), dst_path, compression="zstd")

    allowed_stats_prefixes = tuple(f"stats/{name}/" for name in feature_names)
    for src_path in sorted((source / "meta/episodes").glob("**/*.parquet")):
        schema_names = pq.read_schema(src_path).names
        selected = [
            name
            for name in schema_names
            if not name.startswith("stats/") or name.startswith(allowed_stats_prefixes)
        ]
        dst_path = destination / src_path.relative_to(source)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pq.read_table(src_path, columns=selected), dst_path, compression="zstd")

    os.symlink(source / "videos", destination / "videos", target_is_directory=True)
    write_json(
        destination / "source.json",
        {
            "source_root": str(source),
            "schema": "pi05_hil_common_source_v1",
            "retained_features": feature_names,
        },
    )


def main() -> None:
    args = parse_args()
    training_view = args.training_view.resolve()
    common_root = args.common_root.resolve()
    output_root = args.output_root.resolve()
    reuse_common = common_root.exists()
    if output_root.exists():
        raise FileExistsError(f"Output dataset already exists: {output_root}")

    roots = source_roots_from_view(training_view)
    feature_names, features = common_features(roots)
    common_roots: list[Path] = []
    for index, source in enumerate(roots):
        destination = common_root / f"source-{index:03d}-{source.name}"
        if reuse_common:
            provenance = read_json(destination / "source.json")
            if Path(provenance["source_root"]).resolve() != source:
                raise ValueError(f"Common source provenance mismatch: {destination}")
        else:
            materialize_common_source(
                source,
                destination,
                feature_names=feature_names,
                features=features,
            )
        common_roots.append(destination)

    sys.path.insert(0, str(EVORL_ROOT / "src"))
    from lerobot.datasets.aggregate import aggregate_datasets

    repo_ids = [f"barbell/pi05_hil_source_{index:03d}" for index in range(len(common_roots))]
    aggregate_datasets(
        repo_ids=repo_ids,
        roots=common_roots,
        aggr_repo_id=args.repo_id,
        aggr_root=output_root,
        # Rotate instead of repeatedly concatenating/re-encoding source videos.
        data_files_size_in_mb=0.001,
        video_files_size_in_mb=0.001,
    )

    output_info = read_json(output_root / "meta/info.json")
    expected = read_json(training_view / "manifest.json")
    if int(output_info["total_episodes"]) != int(expected["episodes"]):
        raise RuntimeError("Aggregated episode count does not match the validated training view")
    if int(output_info["total_frames"]) != int(expected["frames"]):
        raise RuntimeError("Aggregated frame count does not match the validated training view")
    write_json(
        output_root / "training_view_provenance.json",
        {
            "schema": "pi05_hil_lerobot_training_dataset_v1",
            "training_view": str(training_view),
            "common_source_root": str(common_root),
            "source_dataset_count": len(roots),
            "total_episodes": int(output_info["total_episodes"]),
            "total_frames": int(output_info["total_frames"]),
            "retained_features": feature_names,
        },
    )
    print(json.dumps(read_json(output_root / "training_view_provenance.json"), indent=2))


if __name__ == "__main__":
    main()
