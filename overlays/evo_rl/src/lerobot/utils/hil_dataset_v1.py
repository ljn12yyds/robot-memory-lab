"""Crash-recoverable sidecar logging for HIL Dataset v1.

LeRobot remains the canonical frame/video store.  This module journals the
decision and event facts that do not naturally fit the LeRobot frame schema.
JSONL files are the online write-ahead logs; Parquet files are query-friendly
materializations rebuilt from those logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


HIL_SCHEMA_VERSION = "1.0"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _materialize_jsonl(source: Path, target: Path) -> None:
    rows = _read_jsonl(source)
    if not rows:
        return
    tmp = target.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    os.replace(tmp, target)


def validate_hil_dataset_v1(
    dataset_root: Path,
    *,
    require_all_committed: bool = True,
    check_videos: bool = False,
    check_lerobot_files: bool = True,
) -> dict[str, list[str]]:
    """Validate referential, temporal, outcome, and optional media integrity.

    ``LeRobotDataset.save_episode()`` deliberately keeps its Parquet writer
    open across episodes.  An open Parquet file has no footer yet and cannot be
    read by PyArrow.  Online per-episode validation must therefore set
    ``check_lerobot_files=False`` and validate the durable HIL WALs only.  The
    default remains a full validation for offline use after
    ``LeRobotDataset.finalize()``.
    """
    root = Path(dataset_root)
    hil_root = root / "hil"
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = hil_root / "manifest.json"
    if not manifest_path.is_file():
        return {"errors": [f"Missing {manifest_path}"], "warnings": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("hil_schema_version") != HIL_SCHEMA_VERSION:
        errors.append(f"Unsupported hil_schema_version={manifest.get('hil_schema_version')!r}")
    if manifest.get("object_id") in {None, "", "unknown"}:
        warnings.append("object_id is unknown")
    if manifest.get("setup_id") in {None, "", "unknown"}:
        warnings.append("setup_id is unknown")

    events = _read_jsonl(hil_root / "events.jsonl")
    decisions = _read_jsonl(hil_root / "decisions.jsonl")
    episodes = _read_jsonl(hil_root / "episodes.jsonl")
    commits = _read_jsonl(hil_root / "episode_commits.jsonl")
    telemetry = _read_jsonl(hil_root / "frame_telemetry.jsonl")

    def ensure_unique(rows: list[dict[str, Any]], key: str) -> None:
        values = [row.get(key) for row in rows if row.get(key)]
        if len(values) != len(set(values)):
            errors.append(f"Duplicate {key} values")

    ensure_unique(events, "event_id")
    ensure_unique(decisions, "decision_id")
    ensure_unique(episodes, "episode_uuid")
    decisions_by_id = {row.get("decision_id"): row for row in decisions}
    decision_ids = set(decisions_by_id)

    last_telemetry_time: dict[str | None, int] = {}
    telemetry_keys: set[tuple[str | None, int]] = set()
    for row in telemetry:
        episode_uuid = row.get("episode_uuid")
        frame_index = int(row.get("frame_index", -1))
        key = (episode_uuid, frame_index)
        if key in telemetry_keys:
            errors.append(f"Duplicate telemetry frame {key}")
        telemetry_keys.add(key)
        decision_id = row.get("decision_id")
        if decision_id and decision_id not in decision_ids:
            errors.append(f"Telemetry frame {key} references missing decision {decision_id}")
        if decision_id and decision_id in decisions_by_id:
            decision = decisions_by_id[decision_id]
            if decision.get("episode_uuid") != episode_uuid:
                errors.append(
                    f"Telemetry episode {episode_uuid} references decision from "
                    f"episode {decision.get('episode_uuid')}"
                )
            if int(row.get("decision_step", -1)) >= int(decision.get("generated_steps", 0)):
                errors.append(f"Telemetry decision_step exceeds decision {decision_id}")
        policy_action_valid = bool(row.get("policy_action_valid", False))
        if row.get("control_source") in {"human_recovery", "human_preventive"} and policy_action_valid:
            errors.append(
                f"Human recovery telemetry frame in {episode_uuid} has a valid executed policy action"
            )
        if policy_action_valid and not decision_id:
            errors.append(f"Policy telemetry frame in {episode_uuid} lacks decision_id")
        current_time = int(row.get("t_mono_ns", 0))
        if current_time < last_telemetry_time.get(episode_uuid, current_time):
            errors.append(f"Episode {episode_uuid} telemetry time is not monotonic")
        last_telemetry_time[episode_uuid] = current_time

    committed = {row.get("episode_uuid") for row in commits if row.get("status") == "committed"}
    episode_ids = {row.get("episode_uuid") for row in episodes}
    if require_all_committed:
        for episode_uuid in sorted(str(item) for item in episode_ids - committed):
            errors.append(f"Episode {episode_uuid} has no committed journal state")

    for decision in decisions:
        executed = int(decision.get("actual_executed_steps", 0))
        generated = int(decision.get("generated_steps", 0))
        planned = int(decision.get("planned_execution_steps", 0))
        if executed > generated or executed > planned:
            errors.append(
                f"Decision {decision.get('decision_id')} executed={executed} exceeds "
                f"generated={generated} or planned={planned}"
            )

    for event in events:
        decision_id = event.get("decision_id")
        if decision_id and decision_id not in decision_ids:
            errors.append(f"Event {event.get('event_id')} references missing decision {decision_id}")
        occurred = event.get("occurred_mono_ns")
        handled = event.get("handled_mono_ns")
        if occurred is not None and handled is not None and int(handled) < int(occurred):
            errors.append(f"Event {event.get('event_id')} handled before it occurred")

    events_by_episode: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("episode_uuid"):
            events_by_episode.setdefault(event["episode_uuid"], []).append(event)
    for episode_uuid, rows in events_by_episode.items():
        # Keyboard callbacks may append an event after a synchronous collector
        # event even though its occurred time is earlier. WAL order is append
        # order, not physical-time order, so reconstruct semantics by timestamp.
        rows = sorted(rows, key=lambda row: int(row.get("occurred_mono_ns", 0)))
        types = [row.get("event_type") for row in rows]
        if "human_takeover_start" in types:
            takeover_index = types.index("human_takeover_start")
            if not {"failure_notice", "preventive_notice"}.intersection(types[:takeover_index]):
                errors.append(
                    f"Episode {episode_uuid} takeover lacks an earlier failure/preventive notice"
                )
        if "recovery_complete" in types and (
            "human_takeover_start" not in types
            or types.index("recovery_complete") < types.index("human_takeover_start")
        ):
            errors.append(f"Episode {episode_uuid} recovery_complete precedes takeover")
        if "preventive_complete" in types and (
            "human_takeover_start" not in types
            or types.index("preventive_complete") < types.index("human_takeover_start")
        ):
            errors.append(f"Episode {episode_uuid} preventive_complete precedes takeover")

    for episode in episodes:
        episode_uuid = episode.get("episode_uuid")
        event_types = {
            row.get("event_type") for row in events_by_episode.get(episode_uuid, [])
        }
        outcome = episode.get("task_outcome")
        assistance = episode.get("assistance")
        if outcome == "success" and "task_success" not in event_types:
            errors.append(f"Successful episode {episode_uuid} lacks task_success event")
        if assistance == "human_recovery":
            for required in ("failure_notice", "human_takeover_start"):
                if required not in event_types:
                    errors.append(
                        f"Human-recovery episode {episode_uuid} lacks {required} event"
                    )
        if assistance == "human_preventive":
            for required in ("preventive_notice", "human_takeover_start"):
                if required not in event_types:
                    errors.append(
                        f"Human-preventive episode {episode_uuid} lacks {required} event"
                    )

    if check_lerobot_files:
        data_files = sorted((root / "data").glob("**/*.parquet"))
        frame_episode_ids: set[str] = set()
        for data_file in data_files:
            try:
                schema = pq.read_schema(data_file)
                if "hil.episode_uuid" not in schema.names:
                    errors.append(f"{data_file} lacks HIL v1 frame columns")
                    continue
                columns = [
                    "frame_index",
                    "hil.episode_uuid",
                    "hil.decision_id",
                    "hil.decision_step",
                    "hil.t_mono_ns",
                    "hil.control_source",
                    "hil.policy_action_valid",
                ]
                table = pq.read_table(data_file, columns=columns).to_pylist()
            except Exception as error:
                errors.append(f"Cannot read LeRobot Parquet {data_file}: {error}")
                continue
            last_time_by_episode: dict[str, int] = {}
            for row in table:
                episode_uuid = row["hil.episode_uuid"]
                frame_episode_ids.add(episode_uuid)
                frame_key = (episode_uuid, int(row["frame_index"]))
                if frame_key not in telemetry_keys:
                    errors.append(f"LeRobot frame {frame_key} lacks telemetry WAL record")
                decision_id = row["hil.decision_id"]
                if decision_id and decision_id not in decision_ids:
                    errors.append(f"Frame references missing decision {decision_id}")
                if decision_id and decision_id in decisions_by_id:
                    decision = decisions_by_id[decision_id]
                    if decision.get("episode_uuid") != episode_uuid:
                        errors.append(
                            f"Frame episode {episode_uuid} references decision from "
                            f"episode {decision.get('episode_uuid')}"
                        )
                    if int(row["hil.decision_step"]) >= int(decision["generated_steps"]):
                        errors.append(f"Frame decision_step exceeds decision {decision_id}")
                if row["hil.control_source"] in {"human_recovery", "human_preventive"} and float(
                    row["hil.policy_action_valid"]
                ) != 0:
                    errors.append(
                        f"Human recovery frame in {episode_uuid} has a valid executed policy action"
                    )
                if float(row["hil.policy_action_valid"]) != 0 and not decision_id:
                    errors.append(f"Policy frame in {episode_uuid} lacks decision_id")
                current_time = int(row["hil.t_mono_ns"])
                if current_time < last_time_by_episode.get(episode_uuid, current_time):
                    errors.append(f"Episode {episode_uuid} frame time is not monotonic")
                last_time_by_episode[episode_uuid] = current_time

        missing_frame_episodes = episode_ids - frame_episode_ids
        for episode_uuid in sorted(str(item) for item in missing_frame_episodes):
            errors.append(f"HIL episode {episode_uuid} has no LeRobot frames")

    if check_videos:
        try:
            import av

            for video in sorted((root / "videos").glob("**/*.mp4")):
                try:
                    with av.open(str(video)) as container:
                        stream = container.streams.video[0]
                        next(container.decode(stream))
                except Exception as error:  # pragma: no cover - depends on external codec data
                    errors.append(f"Cannot decode {video}: {error}")
        except ImportError:
            warnings.append("PyAV unavailable; video decode checks skipped")

    return {"errors": sorted(set(errors)), "warnings": sorted(set(warnings))}


def repair_verified_uncommitted_episodes(
    dataset_root: Path, *, check_videos: bool = True
) -> list[str]:
    """Commit episodes whose only missing fact is the post-validation journal state.

    This is intentionally conservative: a full, post-finalize integrity check
    must pass first, and an episode is repairable only when its latest journal
    state is ``lerobot_saved``.  No frame, action, event, video, or outcome data
    is changed.
    """
    root = Path(dataset_root)
    validation = validate_hil_dataset_v1(
        root,
        require_all_committed=False,
        check_videos=check_videos,
        check_lerobot_files=True,
    )
    if validation["errors"]:
        raise RuntimeError(
            "Refusing HIL commit repair because full integrity validation failed: "
            + "; ".join(validation["errors"])
        )

    hil_root = root / "hil"
    episodes = _read_jsonl(hil_root / "episodes.jsonl")
    commits = _read_jsonl(hil_root / "episode_commits.jsonl")
    statuses: dict[str, list[str]] = {}
    for row in commits:
        episode_uuid = row.get("episode_uuid")
        if episode_uuid:
            statuses.setdefault(episode_uuid, []).append(str(row.get("status")))

    candidates: list[dict[str, Any]] = []
    for episode in episodes:
        episode_uuid = episode.get("episode_uuid")
        episode_statuses = statuses.get(episode_uuid, [])
        if "committed" in episode_statuses:
            continue
        if not episode_statuses or episode_statuses[-1] != "lerobot_saved":
            raise RuntimeError(
                f"Refusing HIL commit repair for {episode_uuid}: latest state is "
                f"{episode_statuses[-1] if episode_statuses else 'missing'}, not lerobot_saved"
            )
        candidates.append(episode)

    if not candidates:
        return []

    wal = _JsonlWal(hil_root / "episode_commits.jsonl")
    repaired: list[str] = []
    try:
        for episode in candidates:
            episode_uuid = str(episode["episode_uuid"])
            common = {
                "hil_schema_version": HIL_SCHEMA_VERSION,
                "session_id": episode.get("session_id"),
                "episode_uuid": episode_uuid,
                "episode_index": episode.get("episode_index"),
                "t_mono_ns": monotonic_ns(),
            }
            wal.append(
                {
                    **common,
                    "status": "validated",
                    "details_json": json.dumps(
                        {
                            "repair": "post_finalize_validation",
                            "warnings": validation["warnings"],
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            wal.append(
                {
                    **common,
                    "t_mono_ns": monotonic_ns(),
                    "status": "committed",
                    "details_json": json.dumps(
                        {"repair": "post_finalize_validation"}, ensure_ascii=False
                    ),
                }
            )
            repaired.append(episode_uuid)
    finally:
        wal.close()

    _materialize_jsonl(
        hil_root / "episode_commits.jsonl", hil_root / "episode_commits.parquet"
    )
    return repaired


def monotonic_ns() -> int:
    return time.monotonic_ns()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = [
            item.decode(errors="surrogateescape")
            for item in untracked_output.split(b"\0")
            if item
        ]
        working_tree_digest = hashlib.sha256(diff)
        for relative_path in sorted(untracked):
            working_tree_digest.update(relative_path.encode(errors="surrogateescape"))
            path = root / relative_path
            if path.is_file():
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        working_tree_digest.update(block)
        return {
            "root": str(root),
            "commit": commit,
            "dirty": bool(diff or untracked),
            "diff_sha256": working_tree_digest.hexdigest(),
            "untracked_files": untracked,
        }
    except (OSError, subprocess.SubprocessError):
        return {"root": str(root), "commit": None, "dirty": None, "diff_sha256": None}


class _JsonlWal:
    def __init__(self, path: Path, *, fsync_every: int = 1):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("a", encoding="utf-8")
        self._fsync_every = max(int(fsync_every), 1)
        self._pending = 0

    def append(self, record: dict[str, Any]) -> None:
        self._stream.write(json.dumps(_jsonable(record), ensure_ascii=False, separators=(",", ":")) + "\n")
        self._stream.flush()
        self._pending += 1
        if self._pending >= self._fsync_every:
            self.flush(durable=True)

    def flush(self, *, durable: bool) -> None:
        self._stream.flush()
        if durable:
            os.fsync(self._stream.fileno())
            self._pending = 0

    def close(self) -> None:
        if not self._stream.closed:
            self.flush(durable=True)
            self._stream.close()


class HILDatasetV1Recorder:
    """Owns HIL v1 identities, WALs, decision lifecycle, and commit records."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_config: dict[str, Any],
        object_id: str,
        setup_id: str,
        policy_path: Path | None,
        calibration_paths: list[Path],
        code_roots: list[Path],
        source_paths: list[Path] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.hil_root = self.dataset_root / "hil"
        self.hil_root.mkdir(parents=True, exist_ok=True)
        self.session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:12]}"
        self.object_id = object_id or "unknown"
        self.setup_id = setup_id or "unknown"
        self.session_start_wall_ns = time.time_ns()
        self.session_start_mono_ns = monotonic_ns()
        self.events_wal = _JsonlWal(self.hil_root / "events.jsonl")
        self.decisions_wal = _JsonlWal(self.hil_root / "decisions.jsonl")
        self.safety_wal = _JsonlWal(self.hil_root / "safety_events.jsonl")
        # At 30 Hz, fsync once per second bounds crash loss without injecting a
        # disk barrier into every real-time control iteration.
        self.telemetry_wal = _JsonlWal(self.hil_root / "frame_telemetry.jsonl", fsync_every=30)
        self.episodes_wal = _JsonlWal(self.hil_root / "episodes.jsonl")
        self.commits_wal = _JsonlWal(self.hil_root / "episode_commits.jsonl")
        self.episode_uuid: str | None = None
        self.episode_index: int | None = None
        self.episode_start_mono_ns: int | None = None
        self.current_decision: dict[str, Any] | None = None
        self.current_decision_step = 0
        self.continuity_segment_id = 0
        self._last_chunk_serial: int | None = None
        self._closed = False

        checkpoint_file = None
        if policy_path is not None:
            checkpoint_file = policy_path / "model.safetensors" if policy_path.is_dir() else policy_path
        manifest = {
            "hil_schema_version": HIL_SCHEMA_VERSION,
            "session_id": self.session_id,
            "session_start_wall_ns": self.session_start_wall_ns,
            "session_start_mono_ns": self.session_start_mono_ns,
            "clock_source": "CLOCK_MONOTONIC",
            "object_id": self.object_id,
            "setup_id": self.setup_id,
            "policy_path": str(policy_path) if policy_path is not None else None,
            "policy_sha256": _sha256_file(checkpoint_file) if checkpoint_file is not None else None,
            "calibrations": [
                {"path": str(path), "sha256": _sha256_file(path)} for path in calibration_paths
            ],
            "git": [_git_state(path) for path in code_roots],
            "source_files": [
                {"path": str(path), "sha256": _sha256_file(path)}
                for path in (source_paths or [])
            ],
            "config": _jsonable(manifest_config),
        }
        manifest_tmp = self.hil_root / "manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(manifest_tmp, self.hil_root / "manifest.json")
        self.record_event("session_start", source="collector")

    def start_episode(self, episode_index: int, *, seed: int | None, task_text: str | None) -> str:
        if self.episode_uuid is not None:
            raise RuntimeError("Cannot start a new HIL episode before committing or abandoning the current one.")
        self.episode_uuid = str(uuid.uuid4())
        self.episode_index = int(episode_index)
        self.episode_start_mono_ns = monotonic_ns()
        self.current_decision = None
        self.current_decision_step = 0
        self.continuity_segment_id = 0
        self._last_chunk_serial = None
        self.commit_status("recording")
        self.record_event(
            "episode_start",
            source="collector",
            details={"seed": seed, "task_text": task_text},
        )
        return self.episode_uuid

    def record_event(
        self,
        event_type: str,
        *,
        occurred_mono_ns: int | None = None,
        handled_mono_ns: int | None = None,
        frame_index: int | None = None,
        decision_id: str | None = None,
        decision_step: int | None = None,
        phase: str | None = None,
        source: str = "collector",
        reason_code: str | None = None,
        confidence: str | None = None,
        note: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        self.events_wal.append(
            {
                "hil_schema_version": HIL_SCHEMA_VERSION,
                "session_id": self.session_id,
                "episode_uuid": self.episode_uuid,
                "episode_index": self.episode_index,
                "event_id": event_id,
                "event_type": event_type,
                "occurred_mono_ns": occurred_mono_ns or monotonic_ns(),
                "handled_mono_ns": handled_mono_ns or monotonic_ns(),
                "frame_index": frame_index,
                "decision_id": decision_id,
                "decision_step": decision_step,
                "phase": phase,
                "source": source,
                "reason_code": reason_code,
                "confidence": confidence,
                "note": note,
                "details_json": json.dumps(_jsonable(details or {}), ensure_ascii=False),
            }
        )
        return event_id

    def drain_keyboard_events(self, events: dict[str, Any], *, frame_index: int) -> None:
        pending = events.get("hil_pending_events", [])
        while pending:
            item = pending.pop(0)
            event_type = item["event_type"]
            if event_type in {"pause_start", "pause_end", "hold_start", "human_takeover_start"}:
                self.continuity_segment_id += 1
            self.record_event(
                event_type,
                occurred_mono_ns=item.get("occurred_mono_ns"),
                handled_mono_ns=monotonic_ns(),
                frame_index=frame_index,
                decision_id=self.current_decision_id,
                decision_step=self.current_decision_step if self.current_decision_id else None,
                phase=item.get("phase"),
                source=item.get("source", "human"),
                reason_code=item.get("reason_code"),
            )

    @property
    def current_decision_id(self) -> str | None:
        return None if self.current_decision is None else self.current_decision["decision_id"]

    def start_decision(
        self,
        *,
        chunk_serial: int,
        source_frame_index: int,
        source_observation_t_mono_ns: int,
        inference_start_ns: int,
        inference_end_ns: int,
        raw_normalized_chunk: list,
        clipped_normalized_chunk: list,
        denormalized_chunk: list,
        planned_execution_steps: int,
        task_text: str | None,
    ) -> str:
        if self.current_decision is not None:
            self.finish_decision("scheduled_replan")
        decision_id = str(uuid.uuid4())
        self.current_decision = {
            "hil_schema_version": HIL_SCHEMA_VERSION,
            "session_id": self.session_id,
            "episode_uuid": self.episode_uuid,
            "episode_index": self.episode_index,
            "decision_id": decision_id,
            "chunk_serial": int(chunk_serial),
            "source_frame_index": int(source_frame_index),
            "source_observation_t_mono_ns": int(source_observation_t_mono_ns),
            "inference_start_ns": int(inference_start_ns),
            "inference_end_ns": int(inference_end_ns),
            "inference_latency_ns": int(inference_end_ns - inference_start_ns),
            "raw_normalized_chunk": raw_normalized_chunk,
            "clipped_normalized_chunk": clipped_normalized_chunk,
            "denormalized_chunk": denormalized_chunk,
            "generated_steps": len(raw_normalized_chunk),
            "planned_execution_steps": int(planned_execution_steps),
            "actual_executed_steps": 0,
            "decision_status": "accepted",
            "chunk_end_reason": None,
            "task_text": task_text,
        }
        self.current_decision_step = 0
        self._last_chunk_serial = int(chunk_serial)
        self.record_event(
            "policy_replan",
            frame_index=source_frame_index,
            decision_id=decision_id,
            decision_step=0,
            source="policy",
        )
        return decision_id

    def mark_decision_step_executed(self) -> int:
        step = self.current_decision_step
        if self.current_decision is not None:
            self.current_decision_step += 1
            self.current_decision["actual_executed_steps"] = self.current_decision_step
        return step

    def finish_decision(self, reason: str, *, status: str | None = None) -> None:
        if self.current_decision is None:
            return
        if status is None:
            executed = int(self.current_decision["actual_executed_steps"])
            planned = int(self.current_decision["planned_execution_steps"])
            status = "fully_executed" if executed >= planned else "partially_executed"
        self.current_decision["decision_status"] = status
        self.current_decision["chunk_end_reason"] = reason
        self.current_decision["ended_mono_ns"] = monotonic_ns()
        self.decisions_wal.append(self.current_decision)
        self.current_decision = None
        self.current_decision_step = 0

    def record_safety_event(
        self,
        event_type: str,
        *,
        frame_index: int,
        reason: str,
        policy_action: dict[str, float] | None = None,
        safety_action: dict[str, float] | None = None,
        details: dict[str, Any] | None = None,
        decision_step: int | None = None,
    ) -> None:
        linked_decision_step = (
            decision_step
            if decision_step is not None
            else (self.current_decision_step if self.current_decision_id else None)
        )
        record = {
            "hil_schema_version": HIL_SCHEMA_VERSION,
            "session_id": self.session_id,
            "episode_uuid": self.episode_uuid,
            "episode_index": self.episode_index,
            "safety_event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "t_mono_ns": monotonic_ns(),
            "frame_index": frame_index,
            "decision_id": self.current_decision_id,
            "decision_step": linked_decision_step,
            "reason": reason,
            "policy_action_json": json.dumps(_jsonable(policy_action or {})),
            "safety_action_json": json.dumps(_jsonable(safety_action or {})),
            "details_json": json.dumps(_jsonable(details or {})),
        }
        self.safety_wal.append(record)
        self.record_event(
            event_type,
            frame_index=frame_index,
            decision_id=self.current_decision_id,
            decision_step=linked_decision_step,
            source="safety_layer",
            reason_code=reason,
        )

    def record_frame_telemetry(
        self,
        *,
        frame_index: int,
        t_mono_ns: int,
        observation_start_mono_ns: int,
        observation_end_mono_ns: int,
        action_send_start_mono_ns: int,
        action_send_end_mono_ns: int,
        control_source: str,
        decision_id: str | None,
        decision_step: int,
        policy_action_valid: bool,
        policy_step_physical: dict[str, float] | None,
        safety_limited_action: dict[str, float],
        commanded_action: dict[str, float],
        safety_modified: bool,
        observation: dict[str, Any],
    ) -> None:
        self.telemetry_wal.append(
            {
                "hil_schema_version": HIL_SCHEMA_VERSION,
                "session_id": self.session_id,
                "episode_uuid": self.episode_uuid,
                "episode_index": self.episode_index,
                "frame_index": int(frame_index),
                "t_mono_ns": int(t_mono_ns),
                "observation_start_mono_ns": int(observation_start_mono_ns),
                "observation_end_mono_ns": int(observation_end_mono_ns),
                "action_send_start_mono_ns": int(action_send_start_mono_ns),
                "action_send_end_mono_ns": int(action_send_end_mono_ns),
                "continuity_segment_id": self.continuity_segment_id,
                "control_source": control_source,
                "decision_id": decision_id,
                "decision_step": int(decision_step),
                "policy_action_valid": bool(policy_action_valid),
                "policy_step_physical_json": json.dumps(_jsonable(policy_step_physical)),
                "safety_limited_action_json": json.dumps(_jsonable(safety_limited_action)),
                "commanded_action_json": json.dumps(_jsonable(commanded_action)),
                "safety_modified": bool(safety_modified),
                "observation_json": json.dumps(_jsonable(observation)),
            }
        )

    def flush_wals(self) -> None:
        for wal in (
            self.events_wal,
            self.decisions_wal,
            self.safety_wal,
            self.telemetry_wal,
            self.episodes_wal,
            self.commits_wal,
        ):
            wal.flush(durable=True)

    def commit_status(self, status: str, *, details: dict[str, Any] | None = None) -> None:
        self.commits_wal.append(
            {
                "hil_schema_version": HIL_SCHEMA_VERSION,
                "session_id": self.session_id,
                "episode_uuid": self.episode_uuid,
                "episode_index": self.episode_index,
                "status": status,
                "t_mono_ns": monotonic_ns(),
                "details_json": json.dumps(_jsonable(details or {}), ensure_ascii=False),
            }
        )

    def write_episode_record(self, record: dict[str, Any]) -> None:
        self.episodes_wal.append(
            {
                "hil_schema_version": HIL_SCHEMA_VERSION,
                "session_id": self.session_id,
                "episode_uuid": self.episode_uuid,
                "episode_index": self.episode_index,
                "object_id": self.object_id,
                "setup_id": self.setup_id,
                **_jsonable(record),
            }
        )

    def complete_episode(self) -> None:
        self.episode_uuid = None
        self.episode_index = None
        self.episode_start_mono_ns = None
        self.current_decision = None
        self.current_decision_step = 0

    def materialize_parquet(self) -> None:
        mappings = {
            "events.jsonl": "events.parquet",
            "decisions.jsonl": "decisions.parquet",
            "safety_events.jsonl": "safety_events.parquet",
            "frame_telemetry.jsonl": "frame_telemetry.parquet",
            "episodes.jsonl": "episodes.parquet",
            "episode_commits.jsonl": "episode_commits.parquet",
        }
        for jsonl_name, parquet_name in mappings.items():
            source = self.hil_root / jsonl_name
            if not source.is_file():
                continue
            _materialize_jsonl(source, self.hil_root / parquet_name)

    def close(self, *, session_complete: bool) -> None:
        if self._closed:
            return
        if self.current_decision is not None:
            self.finish_decision("session_stop", status="partially_executed")
        self.record_event("session_complete" if session_complete else "session_stop", source="collector")
        for wal in (
            self.events_wal,
            self.decisions_wal,
            self.safety_wal,
            self.telemetry_wal,
            self.episodes_wal,
            self.commits_wal,
        ):
            wal.close()
        self.materialize_parquet()
        self._closed = True
