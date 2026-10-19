# SO-101 蓝色杠铃任务

本项目用于快速完成一条真实机器人研究流水线：

1. 左主臂遥操作左从臂采集示范；
2. 使用两路相机和关节状态训练 SmolVLA；
3. 在真实机械臂上进行闭环推理；
4. 保存策略成功/失败 rollout；
5. 使用 LeHome 风格的 segment advantage / AWR 继续微调。

任务文本固定为：

> Lift the blue barbell with the left arm and place it in the orange tray

## 1. 目录说明

| 目录 | 用途 | 是否可以自动重建 |
| --- | --- | --- |
| `data/barbell/` | 正式 BC 数据和已经合并、可直接训练的数据集 | 否，重要资产 |
| `data/rl_rollouts/` | 原始策略 rollout，每轮使用独立时间戳子目录 | 否，重要资产 |
| `models/` | SmolVLA 基础模型及双相机任务配置 | 基础模型可重新下载 |
| `outputs/` | 训练 checkpoint。当前正式模型位于 `smolvla_barbell_10k_v1` | 可重训，但成本高 |
| `logs/` | 训练终端日志 | 可以 |
| `reports/` | 数据审计、推理诊断和侧车日志 | 可以 |
| `scripts/` | 采集、训练、推理和强化学习的全部入口 | 否，核心代码 |
| `docs/` | 架构规范和人工介入研究记录 | 否，项目文档 |

项目的重要修改记录在 [`CHANGELOG.md`](CHANGELOG.md)，代码组织和维护规则见
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

外部依赖：

- LeRobot 源码：`/home/user/ljn/lerobot`
- Python 环境：`/home/user/miniconda3/envs/lerobot`
- SO-101 校准：`~/.cache/huggingface/lerobot/calibration/`

为支持 AWR，LeRobot 源码中还有三处配套修改：

- `src/lerobot/utils/rabc.py`：读取直接 advantage 并计算样本权重；
- `src/lerobot/configs/train.py`：增加 `rabc_beta` 温度参数；
- `src/lerobot/scripts/lerobot_train.py`：把温度参数传入加权器。

如果以后重新安装或替换 `/home/user/ljn/lerobot`，需要同步保留这三处修改，
否则 BC 训练仍可运行，但 `train_smolvla_barbell_rl_v1.sh` 无法使用 advantage 模式。

## 2. 已确认的硬件映射

| 设备 | 稳定标识 |
| --- | --- |
| 左从臂 `SO101_FOLLOWER_ID` | `usb-1a86_USB_Single_Serial_FOLLOWER_SERIAL-if00` |
| 左主臂 `SO101_LEADER_ID` | `usb-1a86_USB_Single_Serial_LEADER_SERIAL-if00` |
| 顶部相机 | `pci-0000:05:00.0-usb-0:2:1.0-video-index0`，软件旋转 180° |
| 左腕相机 | `pci-0000:00:14.0-usb-0:12.4:1.0-video-index0` |

USB 重新插拔后，`by-path` 相机路径可能变化。正式运行前先执行：

```bash
cd /home/user/ljn/barbell_task
./scripts/check_collection_devices.sh
```

## 3. scripts 文件说明

### 正式流水线

| 文件 | 作用 | 是否控制机械臂 |
| --- | --- | --- |
| `record_barbell_dataset.sh` | 人工遥操作采集 BC 示范 | 是 |
| `train_smolvla_barbell_v1.sh` | 使用示范数据训练 SmolVLA | 否 |
| `train_pi05_barbell_expert_v2.sh` | Quantile 有界、有效 batch 8 的 π0.5 动作专家训练 | 否 |
| `run_smolvla_rollout.py` | 带空格死手开关的连续真实推理 | 是 |
| `record_barbell_policy_rollouts.sh` | 执行策略并保存成功/失败 RL rollout | 是 |
| `record_pi05_human_inloop.sh` | Evo-RL 原生 π0.5 单臂人工介入采集 | 是 |
| `prepare_barbell_rl_dataset.py` | 合并 BC 与 rollout 数据并重编号标签 | 否 |
| `compute_barbell_segment_advantages.py` | 把 Episode 结果转换为逐帧 advantage | 否 |
| `train_smolvla_barbell_rl_v1.sh` | advantage 加权 AWR 微调 | 否 |
| `safe_lerobot_cli.py` | 采集入口共用的安全、标签及实际动作记录层 | 间接控制 |

### 检查与诊断

| 文件 | 作用 | 是否控制机械臂 |
| --- | --- | --- |
| `check_collection_devices.sh` | 查看串口与相机路径 | 否 |
| `check_start_alignment.py` | 只读比较主从臂起始角度 | 否 |
| `check_left_teleop.sh` | 30 秒遥操作连通测试 | 是 |
| `diagnose_left_follower_readonly.py` | 读取从臂电压、温度和通信状态 | 否 |
| `diagnose_left_leader_readonly.py` | 读取主臂电压和状态 | 否 |
| `audit_barbell_dataset.py` | 检查数据、动作和两路视频完整性 | 否 |
| `analyze_raw_leader_sidecar.py` | 分析主臂目标、实际指令与从臂延迟 | 否 |
| `probe_smolvla_checkpoint_offline.py` | 使用历史数据离线检查模型 | 否 |
| `probe_smolvla_live_no_motion.py` | 使用真实观测预测但不发送动作 | 否 |

## 4. 行为克隆流水线

### 4.1 采集

```bash
cd /home/user/ljn/barbell_task

BARBELL_MAX_EPISODES=100 \
BARBELL_REPO_ID=barbell/so101_barbell_bc_round2 \
./scripts/record_barbell_dataset.sh
```

按键：

- Recording 阶段 `→`：保存成功 Episode；
- Recording 阶段 `←`：丢弃失败 Episode 并重录；
- Reset 阶段 `→`：复位完成，进入下一条；
- `Esc`：安全结束。

数据默认写入：

```text
data/barbell/<数据集名称>/
```

### 4.2 训练

当前正式训练数据：

```text
data/barbell/so101_barbell_100_merged_v1
```

训练新实验时必须使用新的输出名：

```bash
cd /home/user/ljn/barbell_task

SMOLVLA_RUN_NAME=smolvla_barbell_bc_round2 \
SMOLVLA_STEPS=10000 \
./scripts/train_smolvla_barbell_v1.sh
```

#### 使用 π0.5 训练（Evo-RL 代码链路）

推荐使用 v2 入口。它从干净的 π0.5 Base 重新训练动作专家，继续冻结
PaliGemma 视觉语言主干；训练与推理都在 quantile 反归一化前把状态和动作
限制到 `[-1,1]`。默认 physical batch 为 1、梯度累积 8 次，因此有效 batch
为 8；训练 30000 个真实优化器更新，共消费 240000 个样本，约覆盖当前
79508 帧数据 3.02 遍：

```bash
cd /home/user/ljn/barbell_task

CUDA_VISIBLE_DEVICES=2 \
PI05_RUN_NAME=pi05_barbell_expert_quantileclip_effb8_30k_v2 \
PI05_STEPS=30000 \
PI05_MICRO_BATCH=1 \
PI05_GRAD_ACCUMULATION=8 \
PI05_SAVE_FREQ=5000 \
./scripts/train_pi05_barbell_expert_v2.sh
```

这里的 `PI05_STEPS` 是优化器更新次数，不是 micro-batch 次数。单卡样本消费量
等于 `PI05_STEPS × PI05_MICRO_BATCH × PI05_GRAD_ACCUMULATION`。输出目录已存在
时脚本会拒绝覆盖；新实验必须更换 `PI05_RUN_NAME`。

旧版 2K/22K 实验由 `train_pi05_barbell_v1.sh` 生成，仅用于历史复现。它没有
全链路 Quantile 限制，也没有梯度累积，不应继续作为新实验起点。

如需先做短冒烟测试，可使用：

```bash
cd /home/user/ljn/barbell_task

CUDA_VISIBLE_DEVICES=2 \
PI05_RUN_NAME=pi05_quantile_accum2_smoke_new \
PI05_STEPS=1 \
PI05_MICRO_BATCH=1 \
PI05_GRAD_ACCUMULATION=2 \
PI05_SAVE_FREQ=1 \
PI05_NUM_WORKERS=0 \
./scripts/train_pi05_barbell_expert_v2.sh
```

π0.5 checkpoint 每份约 9 GB；设置保存频率时需要预留足够磁盘空间。

首次测试 π0.5 真实执行时使用 3 秒、每周期最多 2° 的保守配置：

```bash
cd /home/user/ljn/Evo-RL

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=2 \
.venv/bin/python /home/user/ljn/barbell_task/scripts/run_smolvla_rollout.py \
  --checkpoint /home/user/ljn/Evo-RL/outputs/pi05_barbell_bc_2k_v1/checkpoints/002000/pretrained_model \
  --output /home/user/ljn/barbell_task/reports/pi05_barbell_bc_2k_v1/first_3s_rollout.json \
  --duration-s 3 \
  --fps 30 \
  --replan-every-steps 10 \
  --max-step-deg 2 \
  --tracking-failure-timeout-s 2 \
  --execute
```

加载完成并显示 `READY` 后才可按住空格；松开立即暂停，`Esc` 退出并关闭扭矩。
当前 22K checkpoint 使用历史双相机数据训练，运行时不要添加
`--use-right-wrist-camera`；该参数只供未来三相机 checkpoint 使用。
推理器默认使用 10 步滚动规划：模型仍预测 50 步，但只执行前 10 步，随后
丢弃剩余动作，并根据最新相机图像和关节状态重新预测。可用
`--replan-every-steps` 调整，取值不能超过 checkpoint 的动作块长度。

### 4.3 真实推理

当前正式 checkpoint：

```text
outputs/smolvla_barbell_10k_v1/checkpoints/010000/pretrained_model
```

运行：

```bash
cd /home/user/ljn/lerobot

CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
/home/user/miniconda3/envs/lerobot/bin/python \
  /home/user/ljn/barbell_task/scripts/run_smolvla_rollout.py \
  --checkpoint /home/user/ljn/barbell_task/outputs/smolvla_barbell_10k_v1/checkpoints/010000/pretrained_model \
  --output /home/user/ljn/barbell_task/reports/latest_rollout.json \
  --fps 30 \
  --replan-every-steps 10 \
  --max-step-deg 5 \
  --tracking-failure-timeout-s 2 \
  --execute
```

- 按住空格：执行；
- 松开空格：暂停；
- `Esc`：退出并关闭扭矩；
- 不提供 `--duration-s` 时不设时间上限。

## 5. 强化学习 / AWR 流水线

### 5.0 π0.5 人工介入采集

运行 `./scripts/record_pi05_human_inloop.sh`。模型和设备就绪后，每轮必须按 Enter
才启动策略。策略运行期间按 `s` 保存自主成功并进入不记录的人工 RESET；按 `f`
标记自主策略失败并强制进入人工接管，人工完成后再按 `s` 保存整条成功纠正轨迹，
随后进入人工 RESET。人工恢复机械臂原位并重置场景后，按 Enter 直接启动下一轮。
Space 用于全局暂停/继续，暂停期间 `s`、`f` 和 Enter 均不生效；按 `Esc` 停止。
Episode 和 RESET 均无自动超时，也不会自动驱动机械臂回位。逐帧保存
`policy_action`、`is_intervention` 和
`collector_policy_id`，Episode 元数据同时区分自主成功与人工纠正成功。数据固定写入
`data/rl_rollouts/pi05_human_inloop/` 下的独立时间戳目录。
执行入口会在 quantile 反归一化前把 π0.5 动作限制到 `[-1,1]`，并启用
20° 策略原始目标突变检查、校准软限位、2 秒关节卡滞检测（超过 1.5° 且无运动
进展才触发）、目标寄存器回读以及与评测一致的 5° 单步限制。人工遥操作不应用
20° 策略跳变拒绝，而是使用软限位裁剪和 5° 单步限制。策略新动作块越界时不会发送该动作，
而是清空缓存并用下一帧最新观测重新推理；连续 3 个新动作块仍异常时暂停策略
并自动进入人工接管。数据集 `action` 保存安全层实际发送值，模型未裁剪提议
单独保存在 `complementary_info.policy_action`。

这里暂不使用 PPO。第一轮采用人工介入感知的分段加权行为克隆。代码沿用
LeHome 的 advantage 权重接口，但固定人工分数并不是由 critic 学出的真实
advantage：

```text
BC checkpoint
  → 真实策略 rollout
  → 失败时按 f 切换人工纠正
  → 标注成功/失败及人工介入起始帧
  → 合并 BC 与 rollout
  → 策略片段与人工纠正片段分别设置实验性 advantage 分数
  → weight = exp(advantage / beta)
  → 加权微调
```

### 5.1 采集 5 条验证 rollout

```bash
cd /home/user/ljn/barbell_task

BARBELL_ROLLOUT_EPISODES=5 \
./scripts/record_barbell_policy_rollouts.sh
```

数据集名称必须以 `eval_` 开头，这是 LeRobot 对策略录制数据的要求。
所有 rollout 固定保存在：

```text
/home/user/ljn/barbell_task/data/rl_rollouts/
```

每轮使用独立时间戳子目录，避免覆盖 LeRobot 元数据和视频。

- Recording 阶段 `→`：保存成功；
- Recording 阶段 `←`：保存失败；
- Recording 阶段 `Space`：释放主臂扭矩并永久切换到人工纠正；
- Reset 阶段 `→`：开始下一条；
- Episode 超过 120 秒：自动保存为失败；
- Esc：丢弃当前未完成轨迹并退出。

标签保存在 rollout 数据集的：

```text
meta/rl_episode_labels.json
```

### 5.2 合并 BC 与 rollout

把下面的 rollout 路径替换为实际目录：

```bash
cd /home/user/ljn/barbell_task

/home/user/miniconda3/envs/lerobot/bin/python \
  scripts/prepare_barbell_rl_dataset.py \
  --rollout-root data/rl_rollouts/eval_so101_barbell_policy_rollouts_YYYYMMDD_HHMMSS \
  --output-root data/barbell/so101_barbell_rl_round1 \
  --output-repo-id barbell/so101_barbell_rl_round1
```

脚本会：

- 保留 100 条 BC Episode，并给它们固定 `advantage=0.3`；
- 把 rollout Episode 编号后移 100；
- 保留策略成功/失败标签；
- 保留 `human_intervention` 和 `intervention_start_frame`；
- 拼接数据 Parquet 和两路视频；
- 不修改原始数据集。

### 5.3 AWR 微调

```bash
cd /home/user/ljn/barbell_task

BARBELL_RL_REPO_ID=barbell/so101_barbell_rl_round1 \
BARBELL_RL_DATASET_ROOT=/home/user/ljn/barbell_task/data/barbell/so101_barbell_rl_round1 \
BARBELL_RL_RUN_NAME=smolvla_barbell_rl_round1 \
BARBELL_RL_STEPS=3000 \
./scripts/train_smolvla_barbell_rl_v1.sh
```

对于“人工接管后最终成功”的 Episode，默认分段权重为：

```text
早期自主片段：保留该 Episode 的基础 advantage
接管前 90 帧风险窗口：advantage = -0.5，loss 权重约 exp(-0.5) = 0.61
接管后人工纠正：advantage = +1.0，loss 权重约 exp(+1.0) = 2.72
```

人工接管后仍失败的 Episode 不会获得人工纠正正奖励。可通过
`BARBELL_INTERVENTION_POLICY_ADVANTAGE` 和
`BARBELL_INTERVENTION_HUMAN_ADVANTAGE` 覆盖默认值；通过
`BARBELL_INTERVENTION_RISK_WINDOW_FRAMES` 修改风险窗口长度。

训练脚本首先生成：

```text
segment_advantages.parquet
```

随后使用 `exp(advantage / beta)` 对每条训练样本的 loss 加权。

当前默认只降低接管前 90 帧（30 fps 下约 3 秒）的风险窗口，保留更早的
正确自主动作。依据和实验顺序见
[`docs/HUMAN_INTERVENTION_RESEARCH.md`](docs/HUMAN_INTERVENTION_RESEARCH.md)。

## 6. 安全说明

带有“是否控制机械臂=是”的脚本运行前必须：

1. 清空机械臂工作区；
2. 检查线缆不会拉扯；
3. 操作者留在机械臂旁；
4. 手边保持物理急停；
5. 首次使用新 checkpoint 时先运行无动作探针；
6. 不要通过删除安全参数来解决跟踪失败或限位报警。

程序退出时会尝试关闭从臂扭矩，但物理急停仍是最终安全措施。

## 7. 项目维护规则

以后增加或修改代码统一遵守：

1. 先查找并复用已有函数；两个入口出现相同实现时提取为共用模块；
2. `scripts/` 中的稳定入口保持简短，按采集、训练、推理、数据处理和诊断命名；
3. 新代码写清楚中文注释，重点解释设计原因、数据格式和安全边界；
4. 数据集与 checkpoint 使用新名称，禁止静默覆盖；
5. 所有功能、安全参数和数据格式修改写入 `CHANGELOG.md`；
6. 命令、默认路径或按键变化时同步更新 README；
7. Python、Shell、数据转换和真实机械臂分别按风险等级进行验证。
# PI0.5 HIL safety

The validated SO101 PI0.5 HIL limits and USB stability requirements are recorded
in [docs/PI05_HIL_SAFETY.md](docs/PI05_HIL_SAFETY.md). In particular, real-robot
policy execution uses a constant 5-degree step limit and a 2-degree soft-limit
proposal tolerance.
