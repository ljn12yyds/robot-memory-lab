# HIL Dataset v1.0

这套协议用于 PI0.5 真机 HIL 采集。LeRobot 仍保存标准帧、状态、动作和视频；`hil/` sidecar 保存决策、事件、安全修改、真实时间和提交状态。

## 操作流程

- `Enter`：确认初始场景或 RESET 完成，开始下一轮策略推理。
- `←`：记录 `failure_notice`，停止策略；扶住仍有力矩的主臂，再按 `Enter` 进入人工恢复。
- `→`：当前任务成功。立即结束并保存当前 Episode，然后进入 RESET 遥操作。
- `↓`：人工恢复也无法完成，保存当前 Episode 为最终失败，然后进入 RESET。
- `Space`：暂停/继续全部策略和遥操作执行。
- `Esc`：停止 Session；正在进行的轨迹按 `aborted` 保存，尽量保留已有证据。

RESET 期间用主臂带动从臂回原位并布置环境；完成后按 `Enter`。当前没有 Episode 或 RESET 超时。

## 每个新 Dataset 的内容

```text
data/                         LeRobot逐帧状态、实际动作及HIL帧字段
videos/                       Base/Wrist原始视频
meta/                         LeRobot元数据
hil/manifest.json             Session、配置、checkpoint/calibration/git哈希
hil/events.jsonl              事件WAL（含occurred/handled双时间）
hil/decisions.jsonl           每次PI0.5完整50-step提案与实际执行前缀
hil/frame_telemetry.jsonl     每帧动作生命周期和低维状态WAL
hil/safety_events.jsonl       clip/reject/tracking等安全事实
hil/episodes.jsonl            object/setup/outcome/assistance
hil/episode_commits.jsonl     Episode提交状态机
hil/*.parquet                 上述WAL的查询用镜像
```

完整策略 Chunk 只在 `decisions` 保存一次。逐帧数据通过 `decision_id + decision_step` 引用它，并另外保存策略物理动作、安全层动作和实际命令。人工恢复帧的策略动作标为无效，不用零动作冒充监督标签。

HOLD/Pause 不写普通训练帧，但会写事件并增加 `continuity_segment_id`。构建训练 transition 时不得跨越不同 continuity segment。

## 持久化与验收

稀疏事件、Decision、Safety 和 Commit 每条立即 `fsync`。30 Hz telemetry 每 30 帧 `fsync`，Episode 结束时强制同步。因此异常断电时，事件事实立即保存，逐帧 sidecar 最多约损失一秒；未出现最终 `committed` 的 Episode 不进入正式训练。

每轮顺序固定为：

```text
停止控制 → flush WAL → LeRobot save_episode → 写Episode记录
→ 引用/时间/结果完整性检查 → committed → RESET
```

Session 完成后运行：

```bash
cd /home/user/ljn/Evo-RL
.venv/bin/python /home/user/ljn/barbell_task/scripts/check_hil_dataset_v1.py \
  /path/to/dataset --check-videos
```

只有输出 `PASSED` 的 Dataset 才进入训练清单。`--allow-uncommitted` 只用于崩溃取证，不表示数据通过验收。

## Pilot

正式批量采集前先采 5–10 个 Episode，至少覆盖：自主成功、失败后人工恢复成功、人工恢复最终失败、一次 Pause。完成后运行完整检查，并人工抽查一条 `decision → executed prefix → video consequence → failure_notice → recovery` 是否可完整重建。
