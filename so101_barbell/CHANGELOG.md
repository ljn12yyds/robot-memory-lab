# 变更记录

本文件记录会影响数据格式、训练结果、操作流程或机器人安全的修改。
日期使用北京时间。新修改写在最上方；每条记录说明“改了什么、为什么改、
如何验证”，以便在没有 Git 提交记录时仍能追溯实验。

## 2026-08-14

### Track B Experience Memory v1

- 新增 `scripts/build_hil_experience_memory.py`，从已校验 HIL training view 非破坏性
  抽取 State–Action–Effect–Outcome–Correction Experience。
- 新增可检索 Memory Bank；查询键只包含前置状态和坏动作，避免结果与人工答案
  泄漏，并为外部 WAM `z`/`delta_z` 预留稳定 Parquet 接口。
- 新增 `docs/TRACK_B_WAM_MEMORY.md`，明确 WAM 学习 `a+z(+Value/Q)`，Effect、
  Advantage 和 Memory 分别派生或在模型外构造。

## 2026-08-10

### PI0.5 HIL 自主失败→人工纠正→成功状态机

- HIL 中 `s` 表示当前任务成功并结束 Episode；`f` 不再立即保存失败，而是记录
  `policy_failure_detected=true`、强制进入人工接管，并继续录制失败后的纠正轨迹。
  人工完成任务后再按 `s`，整条策略失败＋人工纠正轨迹以成功 Episode 保存；
  纠正期间 Space 切换被锁定，避免误触后重新放权给已判失败的策略。
- 显式按 `f` 后暂停原 45 秒 Episode 截止计时，人工纠正可持续到 `s` 或 `Esc`；
  未按 `f` 且 45 秒内没有结果的 Episode 仍按超时失败保存。连续三个不安全策略
  动作块触发的自动接管也按策略失败＋人工纠正处理。
- Episode 元数据新增 `policy_failure_detected`、`required_intervention`、
  `autonomous_policy_success` 和 `intervention_recovery_success`，可同时用于纯自主
  成功率统计和纠正数据筛选；逐帧 `is_intervention`、执行动作和策略提议保持不变。
- RESET 确认键由 Space 改为 Enter；Space 仅在 RECORDING 阶段手动切换策略/
  人工控制，避免一个按键承担两种阶段语义。Enter 只在 RESET 中生效，进入
  RESET 时会清除旧确认事件，防止自动跳过。
- 已通过相关 Python/Shell 语法检查及 `control_utils`/录制标注共 15 项 pytest：
  验证 `f` 不结束、`s` 最终成功退出、Enter 设置独立 RESET 确认事件。真机
  流程仍需现场验证。

### PI0.5 HIL 跨阶段跟踪状态与 RESET 确认修复

- 修复 HIL Episode 结束后，安全层把上一条策略/自动回位目标带入人工 RESET，
  从而把有效 RESET 动作误判为 `target tracking failure` 的问题。RECORDING、
  自动回位和 RESET 现在分别使用干净的动作跟踪状态；软限位、原始目标检查、
  单步限速和目标寄存器回读保持启用。
- RESET 阶段新按一次 Space 会确认场景复位完成并立即开始下一 Episode；配置的
  `reset_time_s` 只作为操作者未确认时的最大兜底时间。进入 RESET 前会清除上一
  阶段残留的 Space toggle，避免自动跳过 RESET。
- 2026-08-10 10:24 的首次 30K HIL 采集在 Episode 0 超时后、保存前触发旧问题；
  对应目录元信息为 0 Episode / 0 frame，不能用于训练，应使用新时间戳重新采集。
- 10:37 的复测确认跨阶段状态已清空，但 RESET 中的人工主从臂动作仍被通用
  `send_action` 钩子按策略目标执行 2 秒卡滞检测，最终以肘关节 1.98° 误差中止；
  该目录同样为 0 Episode / 0 frame。现已按动作来源启用卡滞检测：仅自主策略
  动作检查 2 秒无进展，人工介入、自动回位和 RESET 不做策略卡滞判定，但仍
  保留有限值、20° 原始跳变、软限位、5° 相对目标和寄存器回读保护。
- 修改后已对四个 Python 入口执行字节码编译，并检查 HIL 启动提示；真机阶段
  切换仍需现场验证。

### PI0.5 多 Episode 成功率评测与人工 RESET 状态机

- `run_smolvla_rollout.py` 新增 `--num-episodes`：每轮执行中按 `s` 立即标记
  `success`；达到 `--duration-s` 而未按 `s` 时自动标记 `failure`。
- 新增显式 `--auto-execute-episodes`，复用强化学习 rollout 的阶段语义：确认
  启动后策略自动执行，不再要求持续按住 Space；任务完成时立即按 `s`，机械臂
  是否回到初始姿态不属于本轮成功条件。Space 只在 RESET 阶段确认人工复位
  完成，并立即开始下一轮。未传该参数的历史单轮命令仍保留死手开关行为。
- 每轮结束后关闭从臂扭矩进入 `RESET`。操作者完成机械臂与场景复位后按一次
  空格确认；`Esc` 在执行或 RESET 阶段都可终止整个会话。
- 每轮切换时清空 PI0.5 动作队列、跟踪检测状态和随机种子，下一轮强制从新的
  相机/关节观测重新规划；推理期间若松开空格或按下 `s/Esc`，待发送动作也会
  被取消。
- JSON 报告新增逐轮 outcome、标签来源、命令数、重规划数，以及会话级成功数、
  失败数和成功率；保留顶层 `records`，并为每条记录增加 `episode_index`。
- 已通过 Python 字节码编译、CLI 帮助及多轮缺少时限的参数门禁测试；真机 RESET
  的扭矩开关与按键流程仍需操作者现场验证。

### 真机推理首次总线握手重试

- `run_smolvla_rollout.py` 连接从臂后的首次 `Torque_Enable` 只读检查由 1 次
  调整为最多 6 次，以容忍 USB/Feetech 总线刚连接时偶发的首包丢失；所有
  尝试失败时仍禁止进入 `READY`，不会发送动作，动作阶段安全检查保持不变。
- 修改前一次短测在发送 0 条命令时遇到全电机 `There is no status packet`；
  随后的只读诊断 200/200 次成功，平均 0.99 ms，确认设备路径、供电和总线正常。
- 新 30K checkpoint 的双相机、20 步重规划、5° 单步上限真机短测正常运行
  3.03 秒并以 `duration_limit` 结束，共发送 53 条命令；最大原始/发送变化均为
  4.48°，目标寄存器最大回读误差 0.09°，没有软限位、跟踪或通信异常。
- 同一模型进一步完成 10 秒和 30 秒闭环测试。30 秒测试发送 581 条命令、
  重规划 29 次并正常以 `duration_limit` 结束；现场观察已接近完成任务。夹爪由
  2.11° 增至 26.61°，肩抬、肘屈等关节形成明显阶段运动；最大原始目标变化
  11.54°，由既有单步安全层限制为 5°，全程没有软限位或跟踪失败。
- 45 秒授权测试首次端到端完成“抓取蓝色哑铃并放入橙色托盘”；操作者在成功后
  按 `Esc` 停止。报告记录 768 条命令、38 次重规划、39.65 秒 active execution，
  最大原始变化 14.08°、发送变化限制为 5°、目标寄存器最大回读误差 0.09°。
  该结果记为一次现场确认成功，尚不能替代多 Episode 成功率评测。

## 2026-08-08

### PI0.5 Quantile、有效 batch 与训练量修正

- π0.5 预处理器新增专用 Quantile 边界步骤：归一化后的训练状态、训练动作、
  推理状态及推理动作统一限制到 `[-1,1]`，然后才进行 q01～q99 反归一化；
  该步骤会随 checkpoint 的 pre/postprocessor 配置保存，不影响其他策略。
- Evo-RL 离线训练新增 `gradient_accumulation_steps`。优化器、学习率调度、
  checkpoint、日志和全局 step 只在完成一次梯度累积后推进，因此 `steps`
  现在明确表示真实优化器更新次数。
- 新增 `train_pi05_barbell_expert_v2.sh`：从干净的 π0.5 Base 开始，保持
  expert-only 和冻结 PaliGemma，默认 micro-batch 1、累积 8、有效 batch 8、
  30000 次更新，即 240000 次样本暴露，约为当前 79508 帧数据的 3.02 个 epoch。
- 单 GPU 冒烟测试以 micro-batch 1、累积 2 完成 1 次更新；日志确认有效 batch
  为 2，动作头参数发生变化，并且保存的 checkpoint 同时包含 Quantile 限制器
  和 `gradient_accumulation_steps=2`。
- 正式实验 `pi05_barbell_expert_quantileclip_effb8_30k_v2` 已于 05:16 完成：
  30000 次优化器更新、240000 个样本、3.02 epoch，最终 loss 约 0.034～0.039；
  005000～030000 六个 checkpoint 均完整保存。
- 使用相同 25 个历史代表帧、每帧 2 次采样对比旧 22K 与新 30K：有界动作
  MAE 从 5.59° 降至 5.25°，原始首动作软限位越界从 9/50 降至 0/50。
  报告保存于 `reports/pi05_barbell_expert_quantileclip_effb8_30k_v2/offline_vs_22k.json`；
  精度改善有限，因此真机仍必须从 3 秒死手开关短测开始。

### PI0.5 checkpoint 分阶段离线拟合评测

- 新增 `evaluate_pi05_checkpoints_offline.py`，在相同的 5 条历史 Episode、每条
  5 个相对时间阶段、每帧 2 个随机种子上，对比累计约 2k/12k/17k/22k 更新；
  该评测只衡量训练集代表帧拟合，不宣称是未见数据泛化。
- 安全裁剪后的全关节 MAE 从 2k 的 6.26° 缓慢下降到 22k 的 5.59°；22k
  原始首动作仍有 9/50 越过软限位，裁剪后为 0/50。
- 22k 的 50 个动作块共有 3466/15000（23.1%）归一化动作值超出
  `[-1,1]`；同批帧“保持当前状态”基线 MAE 仅 1.11°，说明当前问题不仅是
  Episode 时长，模型尚未可靠拟合单步示范动作。
- 完整报告保存于 `reports/pi05_checkpoint_progression/offline_imitation_fit.json`；
  下一步应先做少量 Episode 过拟合冒烟实验验证数据/动作对齐，再决定全量续训。

### PI0.5 HIL 分层安全与异常动作块重采样

- π0.5 的归一化动作在 quantile 反归一化前限制到 `[-1, 1]`，使物理动作
  保持在训练数据的 q01～q99 区间；该修改只作用于 HIL 安全入口。
- 策略动作在同步驱动主从臂前先检查软限位和 20° 原始突变；异常动作块不再
  结束 Episode，而是清空策略缓存、读取下一帧最新观测并重新采样。
- 连续 3 个新动作块异常时暂停策略并自动进入人工接管；没有遥操作设备时才
  安全停止。被拒绝的动作不会发送，也不会写成已执行训练帧。
- 合法策略动作继续保留 5° 单步上限，并按距软限位动态降为 2°/1°；人工动作
  不经过模型裁剪，仍由校准绝对边界及 follower 内置单步限制保护。
- HIL 数据集的主 `action` 字段改为保存安全层实际接受并发送的动作；模型原始
  提议继续保存在 `complementary_info.policy_action`，避免训练执行不到的目标。
- 新增纯离线 `evaluate_pi05_action_bounds_offline.py`。在 3 个历史帧上各随机
  采样 5 次：原始 15 个动作块中 6 个越过软限位（40%），裁剪后为 0 个；
  报告保存到 `reports/pi05_barbell_bc_continue_20k_v1/offline_action_bounds_eval.json`。
- 首次 30 秒 HIL 短测在约 3 秒因 `elbow_flex` 持位误差 1.00° 被旧 0.75°
  阈值误判为卡滞；跟踪容差调整为 1.5°，仍要求误差连续超过容差且 2 秒无
  运动进展才停机，Goal_Position 回读检查保持不变。

### 真实推理滚动重规划

- `run_smolvla_rollout.py` 新增 `--replan-every-steps`，默认每执行 10 步就
  丢弃 50 步动作块中尚未执行的尾部，并使用最新相机图像与关节状态重新推理。
- JSON 诊断记录新增逐周期重规划标记、重规划间隔和累计次数；原有软限位、
  单步限速、跟踪失败检测及死手开关保持不变。
- 新增 `record_pi05_human_inloop.sh`，复用 Evo-RL 原生人工介入状态机并固定
  单左臂 SO101、双相机、π0.5 checkpoint 和数据目录；策略阶段主臂跟随从臂，
  Space 切换人工接管，逐帧保存策略/人工动作来源与介入标签。
- Evo-RL 图形键盘监听器支持把空格配置为介入切换键，与现有采集操作保持一致。
- 修复 Draccus 会裁剪单空格 CLI 参数的问题：配置使用明确的 `space` 标记，
  图形键盘与 TTY 监听器再将实际空格键映射为介入事件。
- 人工介入入口显式复用旧 LeRobot 已验证的 `so101_leader/SO101_LEADER_ID.json`
  主臂校准目录，避免 Evo-RL 因默认类型目录改名而误触发重新校准。
- 新增 `safe_evo_hil_cli.py`，为原生人工介入采集补充原始目标突变、校准软限位、
  关节卡滞和目标寄存器回读安全门；采集脚本统一经该入口启动。
- 主从臂策略同步改为串行：从臂动作必须先通过安全门并成功发送，之后才向
  主臂发送反馈，避免并行模式下从臂拒绝动作但主臂已经开始执行。

## 2026-08-06

### π0.5 / Evo-RL 训练链路

- 新增 `probe_pi05_live_no_motion.py`，使用顶部、左腕、右腕三路真实相机和
  左从臂状态完成只读推理验证；测试过程不发送机械臂动作。
- 修复 Evo-RL 加载 ModelScope 版 π0.5 checkpoint 中旧版
  `relative_actions_processor` 配置的兼容问题，并保留训练数据的相机改名映射。
- π0.5 在 100 条 SO101 哑铃数据上完成 1 步训练冒烟测试：loss 0.156，
  反向传播、优化器更新和 checkpoint 保存均成功。
- 新增 `train_pi05_barbell_v1.sh`，提供带中文说明、强制离线、输出防覆盖的
  正式训练入口；默认 2000 步，仅训练动作专家并冻结视觉编码器。
- 修正 `probe_pi05_live_no_motion.py`：推理时从训练 checkpoint 加载已保存的
  quantile 归一化/反归一化统计，防止把归一化动作误当作 SO101 关节角度。
- 扩展 `run_smolvla_rollout.py` 为 SmolVLA/PI0.5 共用安全执行器；PI0.5 使用
  顶部、左腕、右腕三相机，并继续保留空格死手开关、软限位、单步限速、
  跟踪失败检测、目标寄存器回读和退出断扭矩。
- 根据首次 PI0.5 真实测试增加“原始预测目标越过软限位”即时停机门；修复安全
  异常报告仍显示 `not_started` 的问题。首次测试在第 191 个候选动作前安全停止，
  未发送该异常动作。
- PI0.5 第二阶段 20000 步 BC 训练已完成；它从第一阶段 `002000` 模型权重
  开始，使用新的优化器和面向 20000 步的学习率计划。`005000`、`010000`、
  `015000`、`020000` checkpoint 均完整保存，累计模型更新量约为 22000 步。
- PI0.5 安全执行器默认恢复为与 22K checkpoint 一致的顶部＋左腕双相机输入；
  新增显式 `--use-right-wrist-camera` 开关，仅供未来三相机数据训练的模型使用。
- 安全执行器对超过 10 秒的运行新增独立 `--allow-long-duration` 授权门，防止
  首次 3 秒测试仅修改 duration 数字后意外变成 180 秒长测。

## 2026-07-31

### 项目维护

- 新增 `CHANGELOG.md`，从现在起所有功能、安全参数、数据格式和训练配置修改
  都必须在这里登记。
- 新增 `docs/ARCHITECTURE.md`，规定稳定入口、功能分层、代码复用和文件命名。
- 新增 `docs/HUMAN_INTERVENTION_RESEARCH.md`，记录人工介入数据的文献依据、
  当前实现的适用范围及下一步实验方案。
- README 增加项目维护规范，并区分 BC 数据、RL rollout 和训练输出。

### 人工介入与 RL 数据

- `record_barbell_policy_rollouts.sh`：RL rollout 固定写入
  `data/rl_rollouts/<时间戳数据集>/`，避免数据散落或覆盖旧轮次。
- `safe_lerobot_cli.py`：策略执行期间主臂跟随从臂；按 `Space` 后释放主臂，
  本 Episode 永久切换到人工遥操作。
- 记录 `human_intervention`、`intervention_start_frame` 和逐帧
  `control_source`，使训练时能够区分策略动作和人工纠正动作。
- `compute_barbell_segment_advantages.py`：支持接管前、接管后分段 advantage。
  当前 `-0.5/+1.0` 是第一轮实验性启发式参数，不代表已经学习出的真实
  advantage；正式结论必须通过消融实验验证。
- 根据 EIL/Sirius 调研将分段逻辑改为三段式：早期自主段保留基础权重，
  默认只降低接管前 90 帧（30 fps 下约 3 秒）风险窗口，成功人工纠正段
  提高权重；可用
  `BARBELL_INTERVENTION_RISK_WINDOW_FRAMES` 调整窗口。
- Episode 人工判断时限改为 120 秒；超时且没有成功/失败按键时保存为失败。

### 运行稳定性

- 策略 rollout 强制 Hugging Face 离线模式，避免本地已有模型时仍等待网络。
- 策略录制数据集名称增加 `eval_` 前缀，满足 LeRobot 的策略录制校验。
- Reset 阶段保持无动作并等待人工摆放场景；过滤重复的无策略提示日志。
- 保留全部正式 checkpoint；整理项目时没有删除训练模型。

### 验证

- Python 文件通过语法编译检查。
- Shell 文件通过 `bash -n`。
- 使用合成 Episode 验证成功介入、失败介入和无介入轨迹的逐帧 advantage。

## 2026-07-30

- 完成 SO-101 双相机 BC 数据采集、SmolVLA 训练和真实闭环推理流水线。
- 推理加入每步最大 5°、软限位、跟踪失败超时、空格死手开关和退出断扭矩。
- 修复 50 步动作块只执行首步的问题，支持连续执行完整动作块并周期性重规划。
- 增加策略 rollout 的成功/失败人工标注和数据审计。
