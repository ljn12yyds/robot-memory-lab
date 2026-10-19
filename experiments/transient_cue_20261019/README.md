# One-time visual cue memory pilot (2026-10-19)

## Question

After a cue disappears, can a task-level agent choose the correct next subtask from persisted memory when the current simulator image is identical? This tests task-level decision memory, not learned long-horizon manipulation skill.

## Setup

- LIBERO object scene: first place ketchup in the basket and release it. A one-time red or green band is shown to Qwen3-VL-8B-Instruct before this subtask. Red means stop; green means continue by placing BBQ sauce in the basket.
- Lower policy: official `pi05_libero` checkpoint (`gs://openpi-assets/checkpoints/pi05_libero`) served by OpenPI. The policy receives its normal object-placement language instructions; it is not fine-tuned for this cue experiment.
- Upper policy: Qwen3-VL-8B-Instruct reads the band, stores `initial_cue_color` in JSON task memory, and later chooses `STOP` or `CONTINUE`. The ablation deletes only this memory field before the same decision prompt. Both conditions receive the same stage-end image.
- Five LIBERO initial-state indices (0-4), each tested with both cue colors. Index 0 uses an existing successful stage-one action trace from 2026-10-16 for both colors. Indices 1-4 run stage one online with pi0.5 for green, then replay that exact trace for red. Only a grounded ketchup placement and opened gripper permit a decision.
- If either condition chooses `CONTINUE`, a separate deterministic replay of the shared first-stage actions returns the arm home and runs the BBQ subtask. This shared continuation checks that a selected next subtask is executable; it is **not** an independent rollout for each memory condition. In red trials, this continuation is the *ablated* branch's diagnostic behavior, not the memory branch's behavior.

Implementation: [`libero_eval/run_libero_transient_cue_memory.py`](../../libero_eval/run_libero_transient_cue_memory.py) and [`libero_eval/serve_qwen_libero_upper.py`](../../libero_eval/serve_qwen_libero_upper.py). [`trials.csv`](trials.csv) is the public aggregate table. Raw per-trial JSON, action traces, images, memory snapshots, and videos remain on the research server; they are intentionally excluded from this public repository.

Run `python experiments/transient_cue_20261019/verify_results.py` from the repository root to check the published table and recompute its counts. Holders of the private artifacts can also pass `--raw-dir` and `--media-dir` to cross-check raw summaries and image hashes.

## Results

| Metric | With cue memory | Cue memory removed |
| --- | ---: | ---: |
| Correct next-subtask decision, 10 trials | 10/10 | 5/10 |
| Task objective under the shared-continuation protocol, 10 trials | 10/10 | 5/10 |

All 10 cue reads and first-stage placement/release checks passed. All five green continuations placed BBQ sauce successfully after returning the arm home. Diagnostic continuation also succeeded in every red run, but was *incorrect* for the red task because the robot was supposed to stop. The ablated agent chose `CONTINUE` in every trial. In each red/green pair, the stage-end JPEGs are byte-identical (SHA-256 values in `trials.csv`), and simulator `qpos` replay error recorded by the runner was zero. The only decision input changed within a pair was the stored cue color.

No raw media are published. The image hashes and replay errors in `trials.csv` were audited against the private artifacts before publication but cannot be independently reconstructed from the public table alone.

## Reproduction

With the pinned OpenPI/LIBERO environment and the two services running, invoke the runner once per color and initial-state index. Use `--stage1-actions` for the red replay of a successful green run; do not reuse a trace from a different initial state. Example from the OpenPI root:

```bash
MUJOCO_GL=egl PYTHONPATH=/path/to/agent_iclr_robot:/path/to/openpi/third_party/libero \
  examples/libero/.venv/bin/python examples/libero/run_libero_transient_cue_memory.py \
  --cue GREEN --init-index 1 --qwen-endpoint http://127.0.0.1:8012 \
  --output runs/transient_cue_20261019/green_init1_live.json

MUJOCO_GL=egl PYTHONPATH=/path/to/agent_iclr_robot:/path/to/openpi/third_party/libero \
  examples/libero/.venv/bin/python examples/libero/run_libero_transient_cue_memory.py \
  --cue RED --init-index 1 --qwen-endpoint http://127.0.0.1:8012 \
  --stage1-actions runs/transient_cue_20261019/green_init1_live.actions.json \
  --output runs/transient_cue_20261019/red_init1_replay.json
```

The default lower-policy WebSocket endpoint is port 8000. This pilot used an isolated Qwen endpoint on port 8012 because the previous port-8011 process exited before the new runs. One failed preflight created an incomplete `red_pair1/` media directory; it was excluded from the 10 trials and has no result JSON.

## Interpretation and limits

This is a positive controlled demonstration that the task-memory field can change the upper decision under identical current images and that the lower policy can execute the selected known subtask. It does **not** show a novel learned memory algorithm: the red/green rule is explicit in the VLM prompt, and the cue is a synthetic overlay rather than a natural scene event. It also does not test arbitrary subtask language, unseen task families, real robots, or robust long-horizon success. Five initial states and one model/prompt are too few for a general success-rate claim. The next meaningful test should use a natural, history-dependent task with a memory-free but otherwise matched baseline and multiple seeds.
