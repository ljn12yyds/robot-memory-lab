# Agent ICLR Robot - Upper Layer V1

This directory contains the first, deliberately small upper-layer prototype.
It owns task planning and task-stage memory. The lower layer receives only the
current subtask string and returns a task-level report.

## Boundary

```text
TaskAgent -> TaskMemory -> "current subtask" -> LowerAdapter
     ^                                      <- SubtaskReport
     +---------------------------------------------------+
```

The lower adapter cannot access the plan or task memory. The upper layer cannot
access actions, policy state, demonstrations, or skill memory.

## Run the deterministic smoke test

```bash
python3 -m unittest discover -s tests -v
python3 -m runtime.runner \
  --task "take the cup out, place it on the table, and close the cabinet" \
  --memory runs/demo.json
```

The second command creates an inspectable JSON memory file and executes a mock
lower layer. Re-running the command with the same memory file resumes unfinished
work instead of starting over.

## Qwen3-VL integration

`upper/qwen_agent.py` contains a backend-independent structured-output adapter.
Pass it a callable that accepts the system prompt, user prompt, and optional
image path and returns model text. This keeps model serving choices outside the
task runtime.

The local Transformers backend is available in `upper/qwen_transformers.py`.
Run it on one GPU with:

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m runtime.qwen_runner \
  --task "take the cup out, place it on the table, and close the cabinet" \
  --image /path/to/current_rgb.jpg
```

## Subtask-conditioned PI0.5 training

`scripts/build_pi05_subtask_view.py` converts an existing LeRobot dataset into
a metadata-light training view where every frame is paired with the active
language subtask. Videos remain symlinked to the immutable source dataset.

The annotation file uses half-open frame ranges and must cover every frame of
every episode without gaps or overlaps. Validate annotations before building:

```bash
python3 scripts/build_pi05_subtask_view.py \
  --source /path/to/source_dataset \
  --output /path/to/subtask_dataset \
  --annotations configs/barbell_subtasks_v1.json \
  --validate-only
```

The resulting dataset contains both `task_index` and `subtask_index`. For the
first experiment they intentionally point to the same current-subtask string,
so the unmodified PI0.5 processor receives only the lower-layer contract:

```text
current images + proprioception + current subtask -> continuous action chunk
```

The example annotation boundaries are illustrative and must not be used for
training until the corresponding episode videos have been reviewed.
