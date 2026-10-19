#!/usr/bin/env python3
"""Create a zero-copy dataset view with exact global vector statistics."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def exact_stats(values: np.ndarray) -> dict:
    result = {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).astype(float).tolist(),
        "std": values.std(axis=0, dtype=np.float64).astype(float).tolist(),
        "count": [len(values)],
    }
    result.update(
        {
            key: np.quantile(values, quantile, axis=0).astype(float).tolist()
            for key, quantile in QUANTILES.items()
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-episode-max-exclusive", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    tables = [pq.read_table(path) for path in sorted((args.source / "data").glob("**/*.parquet"))]
    table = pa.concat_tables(tables)
    stats_table = table
    if args.stats_episode_max_exclusive is not None:
        episode_indices = np.asarray(table["episode_index"].to_numpy())
        stats_table = table.filter(
            pa.array(episode_indices < args.stats_episode_max_exclusive)
        )
    corrected = json.loads((args.source / "meta/stats.json").read_text(encoding="utf-8"))
    for feature in ("action", "observation.state"):
        values = np.asarray(stats_table[feature].to_pylist(), dtype=np.float32)
        corrected[feature] = exact_stats(values)

    args.output.mkdir(parents=True)
    shutil.copytree(args.source / "meta", args.output / "meta")
    for directory in ("data", "videos"):
        os.symlink(args.source / directory, args.output / directory, target_is_directory=True)
    (args.output / "meta/stats.json").write_text(
        json.dumps(corrected, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "schema": "exact_global_vector_stats_v1",
        "source": str(args.source.resolve()),
        "corrected_features": ["action", "observation.state"],
        "stats_episode_max_exclusive": args.stats_episode_max_exclusive,
        "stats_frame_count": len(stats_table),
        "data_storage": "symlink",
        "video_storage": "symlink",
    }
    (args.output / "corrected_stats_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
