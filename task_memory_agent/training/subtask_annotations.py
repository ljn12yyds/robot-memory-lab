from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SubtaskDefinition:
    id: str
    instruction: str
    success_condition: str


@dataclass(frozen=True)
class Segment:
    subtask_id: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class SubtaskAnnotations:
    task: str
    subtasks: tuple[SubtaskDefinition, ...]
    episodes: dict[int, tuple[Segment, ...]]

    @classmethod
    def load(cls, path: str | Path) -> "SubtaskAnnotations":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != "pi05_subtask_annotations_v1":
            raise ValueError("unsupported annotation schema")
        subtasks = tuple(SubtaskDefinition(**item) for item in payload["subtasks"])
        episodes = {
            int(episode): tuple(Segment(**item) for item in segments)
            for episode, segments in payload["episodes"].items()
        }
        result = cls(task=payload["task"], subtasks=subtasks, episodes=episodes)
        result.validate_definitions()
        return result

    @property
    def subtask_by_id(self) -> dict[str, SubtaskDefinition]:
        return {subtask.id: subtask for subtask in self.subtasks}

    @property
    def index_by_id(self) -> dict[str, int]:
        return {subtask.id: index for index, subtask in enumerate(self.subtasks)}

    def validate_definitions(self) -> None:
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if not self.subtasks:
            raise ValueError("at least one subtask is required")
        ids = [subtask.id for subtask in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("subtask ids must be unique")
        for subtask in self.subtasks:
            if not subtask.id.strip() or not subtask.instruction.strip():
                raise ValueError("subtask id and instruction must not be empty")

    def validate_episode(self, episode: int, length: int) -> None:
        segments = self.episodes.get(episode)
        if not segments:
            raise ValueError(f"episode {episode} has no subtask annotations")
        cursor = 0
        known = self.subtask_by_id
        for segment in segments:
            if segment.subtask_id not in known:
                raise ValueError(
                    f"episode {episode} references unknown subtask {segment.subtask_id!r}"
                )
            if segment.start_frame != cursor:
                raise ValueError(
                    f"episode {episode} has a gap or overlap at frame {cursor}"
                )
            if segment.end_frame <= segment.start_frame:
                raise ValueError(f"episode {episode} contains an empty segment")
            cursor = segment.end_frame
        if cursor != length:
            raise ValueError(
                f"episode {episode} annotations end at {cursor}, expected {length}"
            )

    def indices_for_frames(self, episode: int, frames: list[int]) -> list[int]:
        if not frames:
            return []
        segments = self.episodes.get(episode)
        if not segments:
            raise ValueError(f"episode {episode} has no subtask annotations")
        result: list[int] = []
        segment_index = 0
        for frame in frames:
            while (
                segment_index < len(segments)
                and frame >= segments[segment_index].end_frame
            ):
                segment_index += 1
            if segment_index >= len(segments):
                raise ValueError(f"episode {episode} frame {frame} is out of range")
            segment = segments[segment_index]
            if frame < segment.start_frame:
                raise ValueError(f"episode {episode} frame order is not monotonic")
            result.append(self.index_by_id[segment.subtask_id])
        return result

    def to_summary(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "subtasks": [subtask.instruction for subtask in self.subtasks],
            "annotated_episodes": len(self.episodes),
        }
