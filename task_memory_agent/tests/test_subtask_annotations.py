from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.subtask_annotations import SubtaskAnnotations


class SubtaskAnnotationsTest(unittest.TestCase):
    def load(self, segments: list[dict]) -> SubtaskAnnotations:
        payload = {
            "schema": "pi05_subtask_annotations_v1",
            "task": "move the object",
            "subtasks": [
                {"id": "reach", "instruction": "reach object", "success_condition": "aligned"},
                {"id": "place", "instruction": "place object", "success_condition": "placed"},
            ],
            "episodes": {"0": segments},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return SubtaskAnnotations.load(path)

    def test_maps_contiguous_segments(self) -> None:
        annotations = self.load(
            [
                {"subtask_id": "reach", "start_frame": 0, "end_frame": 2},
                {"subtask_id": "place", "start_frame": 2, "end_frame": 4},
            ]
        )
        self.assertEqual(annotations.indices_for_frames(0, [0, 1, 2, 3]), [0, 0, 1, 1])

    def test_rejects_gap(self) -> None:
        annotations = self.load(
            [
                {"subtask_id": "reach", "start_frame": 0, "end_frame": 2},
                {"subtask_id": "place", "start_frame": 3, "end_frame": 4},
            ]
        )
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            annotations.validate_episode(0, 4)

    def test_rejects_incomplete_episode(self) -> None:
        annotations = self.load(
            [{"subtask_id": "reach", "start_frame": 0, "end_frame": 2}]
        )
        with self.assertRaisesRegex(ValueError, "expected 4"):
            annotations.validate_episode(0, 4)


if __name__ == "__main__":
    unittest.main()

