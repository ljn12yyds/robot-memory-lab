"""Validate the public pilot table, optionally against private artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--media-dir", type=Path)
    args = parser.parse_args()

    with (ROOT / "trials.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {int(row["init_index"]) for row in rows} == set(range(5))

    memory_correct = 0
    ablated_correct = 0
    by_index = {}
    for row in rows:
        name = row["trial"]
        assert row["cue"] in {"RED", "GREEN"}
        expected = "STOP" if row["cue"] == "RED" else "CONTINUE"
        assert row["expected_next"] == expected
        assert row["cue_read_correct"] == "true"
        assert row["stage1_success"] == "true"
        assert row["continuation_success"] == "true"
        assert float(row["replay_qpos_error"]) == 0.0
        assert row["stage1_mode"] in {"online", "replay"}
        for condition in ("memory", "ablated"):
            correct = row[f"{condition}_choice"] == expected
            assert row[f"{condition}_task_success"] == str(correct).lower()

        if args.raw_dir:
            with (args.raw_dir / (name + ".json")).open(encoding="utf-8") as stream:
                raw = json.load(stream)
            assert raw["init_index"] == int(row["init_index"])
            assert raw["cue"] == row["cue"]
            assert raw["cue_read_correct"]
            assert raw["stage1_success"]
            assert raw["expected_next"] == expected
            assert raw["continuation"]["success"]
            assert raw["continuation"]["replay_qpos_error"] == 0.0
            for condition in ("memory", "ablated"):
                assert raw["decisions"][condition]["choice"] == row[f"{condition}_choice"]
                assert raw["decisions"][condition]["task_success"] == (row[f"{condition}_task_success"] == "true")

        if args.media_dir:
            image = args.media_dir / f"after_cue_init{row['init_index']}.jpg"
            assert hashlib.sha256(image.read_bytes()).hexdigest() == row["post_cue_image_sha256"]

        pair = by_index.setdefault(int(row["init_index"]), {})
        assert row["cue"] not in pair
        pair[row["cue"]] = row
        memory_correct += row["memory_task_success"] == "true"
        ablated_correct += row["ablated_task_success"] == "true"

    for pair in by_index.values():
        assert set(pair) == {"RED", "GREEN"}
        assert pair["RED"]["post_cue_image_sha256"] == pair["GREEN"]["post_cue_image_sha256"]
        assert pair["GREEN"]["stage1_mode"] in {"online", "replay"}
        assert pair["RED"]["stage1_mode"] == "replay"

    assert (memory_correct, ablated_correct) == (10, 5)
    audit = " with private artifacts" if args.raw_dir or args.media_dir else " from public summary"
    print(f"Verified {len(by_index)} pairs{audit}: memory {memory_correct}/10, ablated {ablated_correct}/10")


if __name__ == "__main__":
    main()
