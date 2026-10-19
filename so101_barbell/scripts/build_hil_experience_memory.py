#!/usr/bin/env python3
"""从 HIL training view 构建可审计的 Experience 与可检索 Memory Bank。

第一版不假装已经拥有 WAM latent。它先用机器人低维状态构造可验证的
State--Action--Effect--Outcome--Correction 经验；如果提供由 WAM 导出的 latent
Parquet，则 Effect 自动增加 ``delta_z``。原始 HIL 数据始终保持只读。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STATE = "observation.state"
ACTION = "action"
POLICY_ACTION = "complementary_info.policy_action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="构建 experiences.parquet 和 memory_bank.parquet")
    build.add_argument("--training-view", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--risk-window-frames", type=int, default=90)
    build.add_argument("--correction-window-frames", type=int, default=90)
    build.add_argument(
        "--latent-parquet",
        type=Path,
        help="可选 WAM latent，须含 source_root/source_episode_index/source_frame_index/z",
    )
    query = sub.add_parser("query", help="按一个 Experience 的前置状态和坏动作检索记忆")
    query.add_argument("--memory-root", type=Path, required=True)
    query.add_argument("--experience-id", required=True)
    query.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def read_rows(path: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    return pq.read_table(path, columns=columns).to_pylist()


def vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def mean_vector(rows: list[dict[str, Any]], column: str) -> np.ndarray:
    return np.stack([vector(row[column]) for row in rows]).mean(axis=0)


def load_source(root: Path) -> dict[int, list[dict[str, Any]]]:
    columns = [STATE, ACTION, POLICY_ACTION, "episode_index", "frame_index", "hil.policy_action_valid"]
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((root / "data").glob("**/*.parquet")):
        schema = pq.read_schema(path)
        available = [name for name in columns if name in schema.names]
        if "episode_index" not in available:
            raise ValueError(f"{path} lacks episode_index")
        for row in read_rows(path, available):
            episodes[int(row["episode_index"])].append(row)
    return {
        episode: sorted(rows, key=lambda row: int(row["frame_index"]))
        for episode, rows in episodes.items()
    }


def load_latents(path: Path | None) -> dict[tuple[str, int, int], np.ndarray]:
    if path is None:
        return {}
    required = {"source_root", "source_episode_index", "source_frame_index", "z"}
    schema = pq.read_schema(path)
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"latent parquet lacks columns: {sorted(missing)}")
    return {
        (str(row["source_root"]), int(row["source_episode_index"]), int(row["source_frame_index"])):
        vector(row["z"])
        for row in read_rows(path, sorted(required))
    }


def safe_unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-8 else np.zeros_like(value)


def retrieval_features(pre_state: np.ndarray, bad_action: np.ndarray) -> np.ndarray:
    """检索键只编码决策时可知的信息，避免把结果或人工答案泄漏给查询。"""
    return np.concatenate([pre_state, bad_action]).astype(np.float32)


def abstract_memory(assistance: str, outcome: str) -> tuple[str, str]:
    if assistance == "human_preventive":
        situation = "policy action approached a human-identified risk boundary"
        correction = "apply the demonstrated preventive action before continuing"
    else:
        situation = "policy action produced or confirmed a recoverable failure state"
        correction = "apply the demonstrated human recovery action before resuming the task"
    if outcome != "success":
        correction = "do not treat this correction as a reusable positive action"
    return situation, correction


def build(args: argparse.Namespace) -> None:
    view = args.training_view.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if args.risk_window_frames <= 0 or args.correction_window_frames <= 0:
        raise ValueError("window sizes must be positive")
    episodes = read_rows(view / "episodes.parquet")
    latents = load_latents(args.latent_parquet)
    experiences: list[dict[str, Any]] = []
    skipped = defaultdict(int)
    source_cache: dict[str, dict[int, list[dict[str, Any]]]] = {}

    for episode in episodes:
        start = episode.get("intervention_start_frame")
        if start is None:
            skipped["no_intervention"] += 1
            continue
        source_root = Path(str(episode["source_root"]))
        source_episode = int(episode["source_episode_index"])
        source_key = str(source_root)
        if source_key not in source_cache:
            source_cache[source_key] = load_source(source_root)
        rows = source_cache[source_key].get(source_episode, [])
        if not rows:
            skipped["missing_source_episode"] += 1
            continue
        start = int(start)
        by_frame = {int(row["frame_index"]): row for row in rows}
        risk = [by_frame[i] for i in range(max(0, start - args.risk_window_frames), start) if i in by_frame]
        correction = [
            by_frame[i]
            for i in range(start, start + args.correction_window_frames)
            if i in by_frame
        ]
        if not risk or not correction:
            skipped["empty_window"] += 1
            continue

        pre_state = mean_vector(risk[-min(10, len(risk)):], STATE)
        post_state = vector(correction[-1][STATE])
        bad_action_column = POLICY_ACTION if POLICY_ACTION in risk[0] else ACTION
        bad_action = mean_vector(risk, bad_action_column)
        correction_action = mean_vector(correction, ACTION)
        state_effect = post_state - pre_state
        key_start = (str(source_root), source_episode, int(risk[0]["frame_index"]))
        key_end = (str(source_root), source_episode, int(correction[-1]["frame_index"]))
        delta_z = None
        if key_start in latents and key_end in latents:
            if latents[key_start].shape != latents[key_end].shape:
                raise ValueError(f"latent shape mismatch for {episode['episode_uuid']}")
            delta_z = latents[key_end] - latents[key_start]
        outcome = str(episode["task_outcome"])
        assistance = str(episode["assistance"])
        situation, correction_rule = abstract_memory(assistance, outcome)
        exp_id = f"{episode['episode_uuid']}:{start}"
        experiences.append(
            {
                "experience_id": exp_id,
                "episode_uuid": str(episode["episode_uuid"]),
                "source_root": str(source_root),
                "source_episode_index": source_episode,
                "intervention_frame": start,
                "risk_start_frame": int(risk[0]["frame_index"]),
                "correction_end_frame": int(correction[-1]["frame_index"]),
                "goal": str(episode["task_text"]),
                "object_id": str(episode["object_id"]),
                "setup_id": str(episode["setup_id"]),
                "assistance": assistance,
                "outcome": outcome,
                "pre_state": pre_state.tolist(),
                "bad_action": bad_action.tolist(),
                "state_effect": state_effect.tolist(),
                "correction_action": correction_action.tolist(),
                "corrected_effect": state_effect.tolist(),
                "delta_z": None if delta_z is None else delta_z.tolist(),
                "value_target": 1.0 if outcome == "success" else 0.0,
                "preference_valid": outcome == "success",
                "memory_situation": situation,
                "memory_correction": correction_rule,
                "retrieval_features": retrieval_features(pre_state, bad_action).tolist(),
            }
        )

    if not experiences:
        raise RuntimeError("No intervention experiences could be constructed")
    # SO-101 各关节取值范围不同。先按整个 Experience 库逐维标准化，否则绝对值
    # 较大的肩/肘关节会不成比例地主导余弦相似度。
    feature_matrix = np.stack([vector(row["retrieval_features"]) for row in experiences])
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    for row, features in zip(experiences, feature_matrix, strict=True):
        row["retrieval_embedding"] = safe_unit((features - feature_mean) / feature_std).tolist()
    output.mkdir(parents=True)
    table = pa.Table.from_pylist(experiences)
    pq.write_table(table, output / "experiences.parquet", compression="zstd")
    # v1 中一个成功纠正事件对应一条 Memory；后续可在 embedding 空间聚类去重。
    memories = [row for row in experiences if row["preference_valid"]]
    pq.write_table(pa.Table.from_pylist(memories), output / "memory_bank.parquet", compression="zstd")
    manifest = {
        "schema": "hil_experience_memory_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_view": str(view),
        "experiences": len(experiences),
        "memories": len(memories),
        "skipped": dict(sorted(skipped.items())),
        "risk_window_frames": args.risk_window_frames,
        "correction_window_frames": args.correction_window_frames,
        "effect_representation": "delta_z+state_delta" if latents else "state_delta_only",
        "latent_source": str(args.latent_parquet.resolve()) if args.latent_parquet else None,
        "retrieval_feature_mean": feature_mean.tolist(),
        "retrieval_feature_std": feature_std.tolist(),
        "retrieval_contract": "cosine(pre_state || bad_action); outcome/correction excluded from query key",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def query(args: argparse.Namespace) -> None:
    rows = read_rows(args.memory_root.resolve() / "memory_bank.parquet")
    source = next((row for row in rows if row["experience_id"] == args.experience_id), None)
    if source is None:
        raise KeyError(f"Unknown experience_id: {args.experience_id}")
    query_vector = safe_unit(vector(source["retrieval_embedding"]))
    scored = []
    for row in rows:
        if row["experience_id"] == args.experience_id:
            continue
        score = float(np.dot(query_vector, safe_unit(vector(row["retrieval_embedding"]))))
        scored.append((score, row))
    result = [
        {
            "score": score,
            "experience_id": row["experience_id"],
            "assistance": row["assistance"],
            "outcome": row["outcome"],
            "memory_situation": row["memory_situation"],
            "memory_correction": row["memory_correction"],
            "correction_action": row["correction_action"],
        }
        for score, row in sorted(scored, key=lambda item: item[0], reverse=True)[: args.top_k]
    ]
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    build(args) if args.command == "build" else query(args)


if __name__ == "__main__":
    main()
