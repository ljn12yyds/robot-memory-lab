"""Test one-time visual task memory at a real LIBERO stage boundary."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from upper.schemas import ReportStatus, Subtask, SubtaskReport, TaskPlan
from upper.task_memory import JsonTaskMemoryStore, TaskMemory

from probe_pi05_libero_counterfactual import WAIT_ACTION
from probe_pi05_libero_two_stage import in_basket, record_frame, return_home, run_stage


TASK = "Follow the one-time visual cue after placing the ketchup in the basket."
CUE_COLORS = {"RED": (220, 35, 35), "GREEN": (30, 195, 55)}


def ask_qwen(endpoint, route, payload):
    request = Request(
        endpoint.rstrip("/") + route,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def frame_from_obs(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def make_env(task_file, initial_state):
    env = OffScreenRenderEnv(bddl_file_name=str(task_file), camera_heights=256, camera_widths=256)
    env.seed(7)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(10):
        obs, _, _, _ = env.step(WAIT_ACTION)
    return env, obs


def add_bbq_step(memory, prompt):
    memory.plan.subtasks.append(
        Subtask(
            id=1,
            name=prompt,
            success_condition="bbq_sauce_1 in basket",
            preconditions={"ketchup_1": "in_basket"},
            expected_effects={"bbq_sauce_1": "in_basket"},
        )
    )
    memory.begin_current()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cue", choices=CUE_COLORS, required=True)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--qwen-endpoint", default="http://127.0.0.1:8011")
    parser.add_argument("--max-actions-per-stage", type=int, default=400)
    parser.add_argument("--stage1-actions", type=Path)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    memory_path = output.with_suffix(".memory.json")
    ablated_path = output.with_suffix(".ablated_memory.json")
    trace_path = output.with_suffix(".actions.json")
    media_dir = output.with_suffix("")
    for path in (output, memory_path, ablated_path, trace_path, media_dir):
        if path.exists():
            raise FileExistsError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    tasks = [suite.get_task(index) for index in range(suite.n_tasks)]
    scene_task = next(task for task in tasks if "ketchup" in task.language.lower())
    bbq_task = next(task for task in tasks if "bbq sauce" in task.language.lower())
    task_file = Path(get_libero_path("bddl_files")) / scene_task.problem_folder / scene_task.bddl_file
    initial_state = suite.get_task_init_states(tasks.index(scene_task))[args.init_index]
    client = WebsocketClientPolicy(args.host, args.port)

    env, obs = make_env(task_file, initial_state)
    writer = imageio.get_writer(str(media_dir / "common_ketchup.mp4"), fps=10)
    try:
        home_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
        home_quat = np.asarray(obs["robot0_eef_quat"], dtype=float)
        initial_frame = frame_from_obs(obs)
        cue_frame = initial_frame.copy()
        cue_frame[:48, :] = CUE_COLORS[args.cue]
        cue_image = media_dir / "one_time_cue.jpg"
        imageio.imwrite(str(cue_image), cue_frame)
        cue_response = ask_qwen(
            args.qwen_endpoint, "/read_cue", {"image_path": str(cue_image)}
        )
        parsed_cue = str(cue_response.get("cue", "")).upper()

        memory = TaskMemory(
            plan=TaskPlan(
                task=TASK,
                subtasks=[
                    Subtask(
                        id=0,
                        name=scene_task.language,
                        success_condition="ketchup_1 in basket",
                        expected_effects={"ketchup_1": "in_basket"},
                    )
                ],
            )
        )
        if parsed_cue in CUE_COLORS:
            memory.world_state["initial_cue_color"] = parsed_cue
        store = JsonTaskMemoryStore(memory_path)
        memory.begin_current()
        store.save(memory)

        record_frame(writer, obs, "KETCHUP")
        if args.stage1_actions:
            stage1_actions = json.loads(args.stage1_actions.read_text(encoding="utf-8"))
            for action in stage1_actions:
                obs, _, _, _ = env.step(action)
                record_frame(writer, obs, "REPLAY KETCHUP")
            replay_success = in_basket(env, "ketchup_1") and abs(float(obs["robot0_gripper_qpos"][0])) >= 0.025
            stage1 = {
                "target": "ketchup_1",
                "prompt": scene_task.language,
                "success_step": len(stage1_actions) if replay_success else None,
                "durable_success": in_basket(env, "ketchup_1"),
                "action_trace": None,
                "source_actions": str(args.stage1_actions),
            }
        else:
            obs, stage1 = run_stage(
                env,
                client,
                obs,
                "ketchup_1",
                scene_task.language,
                args.max_actions_per_stage,
                5,
                10,
                writer=writer,
                label="KETCHUP",
                trace_actions=True,
                require_release=True,
            )
            stage1_actions = [item["action"] for item in stage1["action_trace"]]
            if stage1["success_step"] is not None:
                stage1_actions.extend([WAIT_ACTION] * 10)
        trace_path.write_text(json.dumps(stage1_actions), encoding="utf-8")
        stage1["action_trace"] = None
        stage1_ok = (
            stage1["durable_success"]
            and abs(float(obs["robot0_gripper_qpos"][0])) >= 0.025
        )
        memory.apply_report(
            SubtaskReport(
                status=ReportStatus.SUCCESS if stage1_ok else ReportStatus.BLOCKED,
                state_updates={"ketchup_1": "in_basket"} if stage1_ok else {},
                message="grounded placement and release" if stage1_ok else "ketchup stage failed",
            )
        )
        store.save(memory)
        stage1_qpos = env.sim.data.qpos.copy()
        current_image = media_dir / "after_ketchup.jpg"
        imageio.imwrite(str(current_image), frame_from_obs(obs))
    finally:
        writer.close()
        env.close()

    summary = {
        "init_index": args.init_index,
        "cue": args.cue,
        "cue_response": cue_response,
        "cue_read_correct": parsed_cue == args.cue,
        "stage1": stage1,
        "stage1_success": stage1_ok,
        "shared_action_trace": str(trace_path),
        "video_dir": str(media_dir),
    }
    if not stage1_ok:
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)
        return

    persisted = store.load()
    ablated = TaskMemory.from_dict(persisted.to_dict())
    ablated.world_state.pop("initial_cue_color", None)
    JsonTaskMemoryStore(ablated_path).save(ablated)

    decisions = {}
    for condition, state in (("memory", persisted), ("ablated", ablated)):
        response = ask_qwen(
            args.qwen_endpoint,
            "/next_from_memory",
            {
                "task_memory": {
                    "current_step": state.current_step,
                    "world_state": state.world_state,
                },
                "image_path": str(current_image),
            },
        )
        choice = str(response.get("next", "")).upper()
        decisions[condition] = {"response": response, "choice": choice}
    expected = "CONTINUE" if args.cue == "GREEN" else "STOP"
    summary["expected_next"] = expected

    continuation = {"attempted": False, "success": False}
    if any(item["choice"] == "CONTINUE" for item in decisions.values()):
        replay_env, replay_obs = make_env(task_file, initial_state)
        replay_writer = imageio.get_writer(str(media_dir / "shared_continuation.mp4"), fps=10)
        try:
            for action in stage1_actions:
                replay_obs, _, _, _ = replay_env.step(action)
            replay_error = float(np.max(np.abs(replay_env.sim.data.qpos - stage1_qpos)))
            continuation = {"attempted": True, "replay_qpos_error": replay_error, "success": False}
            if replay_error <= 0.001 and in_basket(replay_env, "ketchup_1"):
                record_frame(replay_writer, replay_obs, "REPLAYED KETCHUP")
                replay_obs, home = return_home(
                    replay_env, replay_obs, home_pos, home_quat, replay_writer, tight=True
                )
                continuation["return_home"] = home
                if home["reached_home"] and in_basket(replay_env, "ketchup_1"):
                    replay_obs, stage2 = run_stage(
                        replay_env,
                        client,
                        replay_obs,
                        "bbq_sauce_1",
                        bbq_task.language,
                        args.max_actions_per_stage,
                        5,
                        10,
                        writer=replay_writer,
                        label="BBQ SAUCE",
                        require_release=True,
                    )
                    continuation["stage2"] = stage2
                    continuation["success"] = bool(
                        stage2["durable_success"]
                        and in_basket(replay_env, "ketchup_1")
                        and abs(float(replay_obs["robot0_gripper_qpos"][0])) >= 0.025
                    )
        finally:
            replay_writer.close()
            replay_env.close()

    for condition, state, path in (
        ("memory", persisted, memory_path),
        ("ablated", ablated, ablated_path),
    ):
        choice = decisions[condition]["choice"]
        if choice == "CONTINUE":
            add_bbq_step(state, bbq_task.language)
            state.apply_report(
                SubtaskReport(
                    status=ReportStatus.SUCCESS if continuation["success"] else ReportStatus.BLOCKED,
                    state_updates={"bbq_sauce_1": "in_basket"} if continuation["success"] else {},
                    message="shared continuation rollout",
                )
            )
        JsonTaskMemoryStore(path).save(state)
        decisions[condition]["decision_correct"] = choice == expected
        decisions[condition]["task_success"] = bool(
            choice == expected and (choice == "STOP" or continuation["success"])
        )

    summary["decisions"] = decisions
    summary["continuation"] = continuation
    summary["memory_path"] = str(memory_path)
    summary["ablated_memory_path"] = str(ablated_path)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "cue_read_correct": summary["cue_read_correct"],
        "stage1_success": stage1_ok,
        "expected_next": expected,
        "decisions": {name: {"choice": item["choice"], "task_success": item["task_success"]} for name, item in decisions.items()},
        "continuation_success": continuation["success"],
    }), flush=True)


if __name__ == "__main__":
    main()
