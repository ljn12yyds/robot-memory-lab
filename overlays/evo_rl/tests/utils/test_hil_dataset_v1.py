import json

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.utils.hil_dataset_v1 import (
    HILDatasetV1Recorder,
    repair_verified_uncommitted_episodes,
    validate_hil_dataset_v1,
)


def _make_recorder(tmp_path):
    return HILDatasetV1Recorder(
        dataset_root=tmp_path,
        manifest_config={"fps": 30},
        object_id="blue_barbell_01",
        setup_id="barbell_left_tray_right_v1",
        policy_path=None,
        calibration_paths=[],
        code_roots=[],
    )


def test_hil_recorder_writes_global_ids_and_complete_decision(tmp_path):
    recorder = _make_recorder(tmp_path)
    episode_uuid = recorder.start_episode(0, seed=7, task_text="lift")
    decision_id = recorder.start_decision(
        chunk_serial=1,
        source_frame_index=0,
        source_observation_t_mono_ns=10,
        inference_start_ns=11,
        inference_end_ns=21,
        raw_normalized_chunk=[[0.0], [0.1]],
        clipped_normalized_chunk=[[0.0], [0.1]],
        denormalized_chunk=[[1.0], [2.0]],
        planned_execution_steps=2,
        task_text="lift",
    )
    assert recorder.mark_decision_step_executed() == 0
    recorder.finish_decision("failure_notice")
    recorder.write_episode_record({"task_outcome": "failure"})
    recorder.commit_status("committed")
    recorder.complete_episode()
    recorder.close(session_complete=True)

    decision = json.loads((tmp_path / "hil/decisions.jsonl").read_text().splitlines()[0])
    assert decision["episode_uuid"] == episode_uuid
    assert decision["decision_id"] == decision_id
    assert decision["generated_steps"] == 2
    assert decision["actual_executed_steps"] == 1
    assert (tmp_path / "hil/decisions.parquet").is_file()


def test_frame_telemetry_is_materialized(tmp_path):
    recorder = _make_recorder(tmp_path)
    recorder.start_episode(0, seed=None, task_text="lift")
    recorder.record_frame_telemetry(
        frame_index=0,
        t_mono_ns=20,
        observation_start_mono_ns=10,
        observation_end_mono_ns=20,
        action_send_start_mono_ns=21,
        action_send_end_mono_ns=22,
        control_source="human_recovery",
        decision_id=None,
        decision_step=-1,
        policy_action_valid=False,
        policy_step_physical=None,
        safety_limited_action={"joint": 1.0},
        commanded_action={"joint": 1.0},
        safety_modified=False,
        observation={"joint": 0.5},
    )
    recorder.complete_episode()
    recorder.close(session_complete=True)

    row = json.loads((tmp_path / "hil/frame_telemetry.jsonl").read_text().splitlines()[0])
    assert row["control_source"] == "human_recovery"
    assert json.loads(row["observation_json"]) == {"joint": 0.5}
    assert (tmp_path / "hil/frame_telemetry.parquet").is_file()


def test_hil_integrity_checker_accepts_minimal_committed_episode(tmp_path):
    recorder = _make_recorder(tmp_path)
    episode_uuid = recorder.start_episode(0, seed=None, task_text="lift")
    decision_id = recorder.start_decision(
        chunk_serial=1,
        source_frame_index=0,
        source_observation_t_mono_ns=10,
        inference_start_ns=11,
        inference_end_ns=21,
        raw_normalized_chunk=[[0.0]],
        clipped_normalized_chunk=[[0.0]],
        denormalized_chunk=[[1.0]],
        planned_execution_steps=1,
        task_text="lift",
    )
    recorder.mark_decision_step_executed()
    recorder.record_frame_telemetry(
        frame_index=0,
        t_mono_ns=30,
        observation_start_mono_ns=22,
        observation_end_mono_ns=23,
        action_send_start_mono_ns=24,
        action_send_end_mono_ns=25,
        control_source="policy",
        decision_id=decision_id,
        decision_step=0,
        policy_action_valid=True,
        policy_step_physical={"joint": 1.0},
        safety_limited_action={"joint": 1.0},
        commanded_action={"joint": 1.0},
        safety_modified=False,
        observation={"joint": 0.0},
    )
    recorder.finish_decision("task_success")
    recorder.record_event("task_success", frame_index=0, decision_id=decision_id)
    recorder.record_event("episode_end", frame_index=0)
    recorder.write_episode_record({"task_outcome": "success"})
    recorder.commit_status("committed")
    recorder.complete_episode()
    recorder.close(session_complete=True)

    data_dir = tmp_path / "data/chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "frame_index": 0,
                    "hil.episode_uuid": episode_uuid,
                    "hil.decision_id": decision_id,
                    "hil.decision_step": 0,
                    "hil.t_mono_ns": 30,
                    "hil.control_source": "policy",
                    "hil.policy_action_valid": 1.0,
                }
            ]
        ),
        data_dir / "file-000.parquet",
    )

    result = validate_hil_dataset_v1(tmp_path)
    assert result["errors"] == []


def test_hil_integrity_checker_rejects_uncommitted_episode(tmp_path):
    recorder = _make_recorder(tmp_path)
    recorder.start_episode(0, seed=None, task_text="lift")
    recorder.write_episode_record({"task_outcome": "failure"})
    recorder.complete_episode()
    recorder.close(session_complete=False)

    result = validate_hil_dataset_v1(tmp_path)
    assert any("no committed journal state" in error for error in result["errors"])


def test_online_hil_validation_does_not_read_open_lerobot_parquet(tmp_path):
    recorder = _make_recorder(tmp_path)
    episode_uuid = recorder.start_episode(0, seed=None, task_text="lift")
    decision_id = recorder.start_decision(
        chunk_serial=1,
        source_frame_index=0,
        source_observation_t_mono_ns=10,
        inference_start_ns=11,
        inference_end_ns=21,
        raw_normalized_chunk=[[0.0]],
        clipped_normalized_chunk=[[0.0]],
        denormalized_chunk=[[1.0]],
        planned_execution_steps=1,
        task_text="lift",
    )
    recorder.mark_decision_step_executed()
    recorder.record_frame_telemetry(
        frame_index=0,
        t_mono_ns=30,
        observation_start_mono_ns=22,
        observation_end_mono_ns=23,
        action_send_start_mono_ns=24,
        action_send_end_mono_ns=25,
        control_source="policy",
        decision_id=decision_id,
        decision_step=0,
        policy_action_valid=True,
        policy_step_physical={"joint": 1.0},
        safety_limited_action={"joint": 1.0},
        commanded_action={"joint": 1.0},
        safety_modified=False,
        observation={"joint": 0.0},
    )
    recorder.finish_decision("task_success")
    recorder.record_event("task_success", frame_index=0, decision_id=decision_id)
    recorder.write_episode_record({"task_outcome": "success"})
    recorder.flush_wals()

    data_dir = tmp_path / "data/chunk-000"
    data_dir.mkdir(parents=True)
    (data_dir / "open-writer.parquet").write_bytes(b"PAR1footer-not-written-yet")

    online = validate_hil_dataset_v1(
        tmp_path,
        require_all_committed=False,
        check_lerobot_files=False,
    )
    assert online["errors"] == []

    offline = validate_hil_dataset_v1(tmp_path, require_all_committed=False)
    assert any("Cannot read LeRobot Parquet" in error for error in offline["errors"])

    recorder.complete_episode()
    recorder.close(session_complete=False)


def test_repair_verified_uncommitted_episode_only_appends_commit_states(tmp_path):
    recorder = _make_recorder(tmp_path)
    episode_uuid = recorder.start_episode(0, seed=None, task_text="lift")
    recorder.record_frame_telemetry(
        frame_index=0,
        t_mono_ns=30,
        observation_start_mono_ns=22,
        observation_end_mono_ns=23,
        action_send_start_mono_ns=24,
        action_send_end_mono_ns=25,
        control_source="human_recovery",
        decision_id=None,
        decision_step=-1,
        policy_action_valid=False,
        policy_step_physical=None,
        safety_limited_action={"joint": 1.0},
        commanded_action={"joint": 1.0},
        safety_modified=False,
        observation={"joint": 0.0},
    )
    recorder.record_event("failure_notice", frame_index=0)
    recorder.record_event("human_takeover_start", frame_index=0)
    recorder.record_event("task_success", frame_index=0)
    recorder.write_episode_record(
        {"task_outcome": "success", "assistance": "human_recovery"}
    )
    recorder.commit_status("lerobot_saved")
    recorder.complete_episode()
    recorder.close(session_complete=True)

    data_dir = tmp_path / "data/chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "frame_index": 0,
                    "hil.episode_uuid": episode_uuid,
                    "hil.decision_id": "",
                    "hil.decision_step": -1,
                    "hil.t_mono_ns": 30,
                    "hil.control_source": "human_recovery",
                    "hil.policy_action_valid": 0.0,
                }
            ]
        ),
        data_dir / "file-000.parquet",
    )

    repaired = repair_verified_uncommitted_episodes(tmp_path, check_videos=False)
    assert repaired == [episode_uuid]
    assert validate_hil_dataset_v1(tmp_path)["errors"] == []
    assert repair_verified_uncommitted_episodes(tmp_path, check_videos=False) == []


def test_hil_integrity_checker_accepts_preventive_intervention(tmp_path):
    recorder = _make_recorder(tmp_path)
    episode_uuid = recorder.start_episode(0, seed=None, task_text="lift")
    recorder.record_frame_telemetry(
        frame_index=0,
        t_mono_ns=30,
        observation_start_mono_ns=22,
        observation_end_mono_ns=23,
        action_send_start_mono_ns=24,
        action_send_end_mono_ns=25,
        control_source="human_preventive",
        decision_id=None,
        decision_step=-1,
        policy_action_valid=False,
        policy_step_physical=None,
        safety_limited_action={"joint": 1.0},
        commanded_action={"joint": 1.0},
        safety_modified=False,
        observation={"joint": 0.0},
    )
    recorder.record_event("preventive_notice", frame_index=0)
    recorder.record_event("human_takeover_start", frame_index=0)
    recorder.record_event("preventive_complete", frame_index=0)
    recorder.record_event("task_success", frame_index=0)
    recorder.write_episode_record(
        {"task_outcome": "success", "assistance": "human_preventive"}
    )
    recorder.commit_status("committed")
    recorder.complete_episode()
    recorder.close(session_complete=True)

    data_dir = tmp_path / "data/chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "frame_index": 0,
                    "hil.episode_uuid": episode_uuid,
                    "hil.decision_id": "",
                    "hil.decision_step": -1,
                    "hil.t_mono_ns": 30,
                    "hil.control_source": "human_preventive",
                    "hil.policy_action_valid": 0.0,
                }
            ]
        ),
        data_dir / "file-000.parquet",
    )

    assert validate_hil_dataset_v1(tmp_path)["errors"] == []
