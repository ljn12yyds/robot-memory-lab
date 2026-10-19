#!/usr/bin/env python3
"""合并 BC 示范与策略 rollout，并生成第一轮 AWR 所需的 Episode 标签。

输入：
    1. 已确认质量的人工示范数据集（默认 100 条 BC 数据）。
    2. ``record_barbell_policy_rollouts.sh`` 采集的策略数据集。
       其 ``meta/rl_episode_labels.json`` 必须包含每条轨迹的成功/失败结果。

输出：
    一个新的、可直接训练的 LeRobot 数据集。原始数据不会被修改。
    新数据集的前半部分是 BC Episode，后半部分是策略 rollout Episode。
    BC Episode 使用固定正 advantage；rollout 保留 success 标签，后续再计算 advantage。

注意：
    这是纯数据处理脚本，不连接相机或机械臂。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BC_ROOT = PROJECT_ROOT / "data/barbell/so101_barbell_100_merged_v1"


def parse_args() -> argparse.Namespace:
    """解析来源数据、输出数据和 BC 固定 advantage。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True, help="策略 rollout 数据集目录")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="合并数据集输出目录；必须尚不存在",
    )
    parser.add_argument(
        "--output-repo-id",
        required=True,
        help="合并后的逻辑名称，例如 barbell/so101_barbell_rl_round1",
    )
    parser.add_argument("--bc-root", type=Path, default=DEFAULT_BC_ROOT)
    parser.add_argument(
        "--bc-repo-id",
        default="barbell/so101_barbell_100_merged_v1",
    )
    parser.add_argument(
        "--rollout-repo-id",
        help="策略数据集逻辑名称；默认从 rollout-root/meta/info.json 读取",
    )
    parser.add_argument(
        "--bc-advantage",
        type=float,
        default=0.3,
        help="人工示范的固定 advantage，LeHome 风格默认值为 0.3",
    )
    return parser.parse_args()


def main() -> None:
    """校验两个数据集与标签，完成合并和 Episode 重编号。"""
    # ---------- 1. 防止覆盖已有数据 ----------
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"为避免覆盖已有数据，输出目录必须不存在：{args.output_root}")

    # ---------- 2. 解析 rollout 的逻辑 repo_id ----------
    rollout_info = json.loads((args.rollout_root / "meta/info.json").read_text())
    rollout_repo_id = args.rollout_repo_id or rollout_info.get("repo_id")
    if not rollout_repo_id:
        # LeRobot v3 的 info.json 通常不保存 repo_id，因此允许从目录名推导。
        rollout_repo_id = f"barbell/{args.rollout_root.name}"

    # ---------- 3. 加载元数据与人工结果标签 ----------
    bc_meta = LeRobotDatasetMetadata(args.bc_repo_id, root=args.bc_root)
    rollout_meta = LeRobotDatasetMetadata(rollout_repo_id, root=args.rollout_root)
    rollout_labels_path = args.rollout_root / "meta/rl_episode_labels.json"
    rollout_payload = json.loads(rollout_labels_path.read_text())
    rollout_labels = rollout_payload.get("episodes", [])

    # ---------- 4. 标签必须完整覆盖所有 rollout Episode ----------
    expected_rollout_episodes = set(range(rollout_meta.total_episodes))
    labeled_rollout_episodes = {int(row["episode_index"]) for row in rollout_labels}
    if labeled_rollout_episodes != expected_rollout_episodes:
        missing = sorted(expected_rollout_episodes - labeled_rollout_episodes)
        extra = sorted(labeled_rollout_episodes - expected_rollout_episodes)
        raise ValueError(f"rollout 标签与数据不一致：missing={missing}, extra={extra}")

    # LeRobot 官方聚合器负责重写全局 frame/episode 索引并拼接两路视频。
    # ---------- 5. 用 LeRobot 官方聚合器拼接数据和两路视频 ----------
    aggregate_datasets(
        repo_ids=[args.bc_repo_id, rollout_repo_id],
        roots=[args.bc_root, args.rollout_root],
        aggr_repo_id=args.output_repo_id,
        aggr_root=args.output_root,
    )

    # 合并后 rollout 的 Episode 编号整体后移 BC Episode 数量。
    # ---------- 6. 为合并后的 Episode 生成新标签 ----------
    labels = [
        {"episode_index": episode, "advantage": args.bc_advantage, "source": "bc"}
        for episode in range(bc_meta.total_episodes)
    ]
    for row in rollout_labels:
        labels.append(
            {
                "episode_index": bc_meta.total_episodes + int(row["episode_index"]),
                "success": bool(row["success"]),
                "source": "policy_rollout",
                "termination": row.get("termination", "unknown"),
                # 保留人工接管边界。后续逐帧 advantage 生成器可以把接管前的
                # 错误策略片段和接管后的人工纠正片段赋予不同训练权重。
                "human_intervention": bool(row.get("human_intervention", False)),
                "intervention_start_frame": row.get("intervention_start_frame"),
            }
        )

    # ---------- 7. 保存 AWR 下一阶段会读取的标签 ----------
    output_labels = args.output_root / "meta/rl_episode_labels.json"
    output_labels.write_text(
        json.dumps({"schema_version": 1, "episodes": labels}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"合并完成：{args.output_root}")
    print(f"BC Episodes：{bc_meta.total_episodes}")
    print(f"Rollout Episodes：{rollout_meta.total_episodes}")
    print(f"RL 标签：{output_labels}")


if __name__ == "__main__":
    main()
