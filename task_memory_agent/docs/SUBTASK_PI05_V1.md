# Subtask-conditioned PI0.5 V1

## Goal

Test whether an action policy benefits when the upper layer sends only the
currently active language subtask. The lower policy contract is:

```text
multi-view observation + proprioception + current subtask -> action chunk
```

The first version does not ask PI0.5 to generate language. Its only supervised
target is the existing continuous action chunk. This isolates whether subtask
conditioning improves control before adding another loss or decoder path.

## What is borrowed from Show-Harness

- Keep the model-facing interface compact, semantic, and stable.
- Associate every demonstrated control decision with the semantic intent active
  at that moment.
- Keep execution feedback and subtask completion in a closed loop.

We do not reuse its discrete action vocabulary or embodiment interpreters.
PI0.5 remains the learned continuous-action executor.

## First task

Use the existing SO101 task: lift the blue barbell with the left arm and place
it in the orange tray. It already has demonstrations, PI0.5 checkpoints, HIL
rollouts, and evaluation scripts.

Candidate stages are `approach`, `grasp`, `transport`, `place`, and `retreat`.
Frame boundaries must be reviewed from synchronized front and wrist video. The
example JSON is a format example, not a valid training annotation.

## Minimum experiment

Train from the same PI0.5 base checkpoint with the same episodes and optimizer
budget:

1. Global-task baseline: every frame uses the original full-task instruction.
2. Subtask-conditioned SFT: every frame uses its active subtask instruction.
3. Wrong-subtask control: evaluate the trained model with a plausible but
   incorrect stage instruction.

Report full-task success, per-stage success, interventions, stage duration, and
failure category. Also compare output actions under correct and wrong subtasks
from identical recorded observations before any real-robot run.

## Data protocol

- Use half-open frame segments `[start_frame, end_frame)`.
- Cover every episode without gaps or overlaps.
- Keep all frames from one episode in one train/validation/test split.
- Keep transition boundaries visually meaningful; do not infer them from fixed
  percentages alone.
- Record annotation provenance and never modify the source dataset.

## Commands

```bash
python scripts/build_pi05_subtask_view.py \
  --source /home/user/ljn/barbell_task/data/barbell/so101_barbell_100_merged_v1 \
  --output /home/user/ljn/barbell_task/data/barbell/so101_barbell_subtask_v1 \
  --annotations configs/barbell_subtasks_v1.json \
  --validate-only

bash scripts/train_pi05_subtask_barbell_v1.sh
```

