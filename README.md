# Robot Memory Lab

Research code for SO-101 manipulation, policy improvement from human intervention, and task-level memory experiments. This repository contains our experiment scripts and patches against pinned upstream projects; it does not redistribute their full source trees, robot datasets, calibration files, or model weights.

## Contents

| Path | Purpose |
| --- | --- |
| `so101_barbell/` | Real-robot data collection, SmolVLA behavior cloning and advantage-weighted fine-tuning, pi0.5 training and evaluation, HIL preparation, and safety notes. |
| `patches/lerobot-smolvla-awr.patch` | LeRobot changes for direct advantage weights and the RABC temperature setting. |
| `patches/evo-rl-pi05-hil.patch` | Evo-RL pi0.5, value-learning, recording-loop, and HIL integration changes. |
| `overlays/evo_rl/` | New HIL dataset implementation and its tests, not present in the pinned upstream commit. |
| `patches/robotwin-memory-data.patch` | Separate RoboTwin simulator/data-collection modifications for memory experiments. |
| `task_memory_agent/` | Task-level planning, JSON memory, subtask annotations, simulator adapters, and unit tests. |
| `libero_eval/` | Qwen upper-layer service and paired LIBERO/pi0.5 evaluation scripts. |
| `UPSTREAMS.md` | Exact upstream commits and patch application notes. |

## Research stages

1. Collect SO-101 demonstrations and fine-tune SmolVLA for the blue-barbell placement task.
2. Record policy rollouts and human recovery; convert outcomes into segment-level training signals for SmolVLA or pi0.5.
3. Extend Evo-RL's pi0.5 recording path with explicit intervention events, executed-action telemetry, safety events, and episode commit checks. See [`so101_barbell/docs/HIL_DATASET_V1.md`](so101_barbell/docs/HIL_DATASET_V1.md).
4. Explore task-level memory and subtask control in simulation. The RoboTwin patch and LIBERO evaluation scripts are experimental and separate from the SO-101 real-robot pipeline.

The work is **research code**, not a validated general-purpose robot controller. Successful training runs do not establish reliable autonomous performance. Real-robot commands must be reviewed for device mapping, workspace limits, emergency-stop access, and calibration before use.

## Reproduction boundary

Install the pinned upstream source trees in [`UPSTREAMS.md`](UPSTREAMS.md), apply the relevant patches, then adapt the paths and hardware identifiers in `so101_barbell/` to your own machine. The original project README contains historical machine-specific commands; treat them as records, not portable defaults. Hardware IDs and serial numbers in this public snapshot are placeholders (`SO101_FOLLOWER_ID`, `SO101_LEADER_ID`, `FOLLOWER_SERIAL`, `LEADER_SERIAL`); configure them before using any robot-control script. Data, pretrained checkpoints, and calibration are not included.

Start with the dataset and no-motion checks before any real-robot rollout. See [`so101_barbell/README.md`](so101_barbell/README.md) for the original experiment sequence and [`so101_barbell/docs/PI05_HIL_SAFETY.md`](so101_barbell/docs/PI05_HIL_SAFETY.md) for HIL safety constraints.

For the task-level unit tests, set `PYTHONPATH=task_memory_agent` and run `python -m unittest discover -s task_memory_agent/tests`. For the Qwen HTTP handler test, also add `libero_eval` to `PYTHONPATH` and run `python -m unittest discover -s libero_eval -p 'test_*.py'`. The LIBERO scripts additionally require OpenPI, LIBERO, Qwen3-VL model weights, and their simulator dependencies; the code alone does not reproduce the reported physical success rates.

Snapshot checks on 2026-10-19: 7 task-level tests, 1 Qwen route test, and 32 targeted Evo-RL/HIL tests passed; the SO-101 shell entry points passed `bash -n`. These checks do not replace a real-robot safety review or a fresh end-to-end training run.

