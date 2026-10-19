# Track B：HIL 驱动的 WAM、Value 与 Experience Memory

WAM 只承担动作 `a`、未来潜变量 `z` 和可选 Value/Q 的学习。Effect 是
`phi(z_t, z_t+k)` 的派生量；Advantage 由 Q/V 计算；Memory 从完整 HIL
Experience 抽取，不增加没有监督真值的 memory head。

```text
HIL frames ──→ WAM post-training ──→ a + z ──→ Value/Q ──→ RL
     │                                │
     └──→ Experience extraction ←──── Δz
                    │
                    └──→ Memory Bank ──→ retrieval/correction prior
```

## Experience v1

每次人工接管形成一条：

```text
(pre_state, bad_action, effect, outcome, correction_action, corrected_effect)
```

- `pre_state`：接管前最后十帧状态均值；
- `bad_action`：接管前风险窗口内的策略提议均值；
- `state_effect`：纠正窗口末状态减 pre-state；
- `correction_action`：人工纠正窗口实际动作均值；
- `delta_z`：可选 WAM latent 的 `z_end-z_start`；
- `value_target`：第一版使用 Episode 成败弱监督，不把它称为已学出的 Q；
- `preference_valid`：只有最终成功的人工纠正才形成正偏好和可复用 Memory。

检索 embedding 只使用 `(pre_state, bad_action)`，在全库逐维标准化后计算余弦距离，
并严格排除 outcome 和 correction，防止答案泄漏。当前低维表示是接入 WAM 前的可运行基线；以后可在不改变 schema
的情况下替换成 WAM state/action representation。

## 构建和检索

```bash
cd /home/user/ljn/barbell_task

/home/user/ljn/Evo-RL/.venv/bin/python scripts/build_hil_experience_memory.py build \
  --training-view data/rl_rollouts/pi05_hil_training_v1_128ep_20260814 \
  --output-root data/rl_rollouts/pi05_hil_experience_memory_v1

/home/user/ljn/Evo-RL/.venv/bin/python scripts/build_hil_experience_memory.py query \
  --memory-root data/rl_rollouts/pi05_hil_experience_memory_v1 \
  --experience-id '<episode_uuid>:<intervention_frame>' --top-k 5
```

可选 WAM latent Parquet 必须包含
`source_root, source_episode_index, source_frame_index, z`。提供
`--latent-parquet` 后会同时生成 `delta_z`。

## WAM/RL 接口约束

1. Action–Future consistency 使用 action-conditioned `delta_z`，不是最大化无条件
   `corr(a,z)`。
2. Value/Q 应消费完整 `value_frames.parquet`、outcome 和连续段边界；不能只用成功
   Memory 训练 value。
3. Preference 只使用 `preference_valid=true` 的风险动作与成功人工动作；最终失败的
   人工纠正不能作为 preferred action。
4. 在线检索第一阶段只提供 correction prior，不能越过策略安全层直接控制机械臂。
