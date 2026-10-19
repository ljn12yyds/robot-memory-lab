#!/usr/bin/env python3
"""Decode the first frame of each training video with PyAV."""
from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for video_path in sorted((args.dataset_root / "videos").glob("**/*.mp4")):
        camera = video_path.parent.parent.name.removeprefix("observation.images.")
        with av.open(str(video_path)) as container:
            frame = next(container.decode(video=0), None)
        if frame is None:
            raise RuntimeError(f"could not decode {video_path}")
        np.save(args.output / f"{camera}.npy", frame.to_ndarray(format="rgb24"))


if __name__ == "__main__":
    main()
