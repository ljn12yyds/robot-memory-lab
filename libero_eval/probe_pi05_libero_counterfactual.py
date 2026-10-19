"""Probe instruction steering with one fixed LIBERO-Object initial state.

Run from the root of an openpi checkout with its LIBERO client environment.
"""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy


OBJECTS = (
    "ketchup_1",
    "bbq_sauce_1",
    "salad_dressing_1",
    "alphabet_soup_1",
    "cream_cheese_1",
    "milk_1",
)
WAIT_ACTION = [0.0] * 6 + [-1.0]


def axis_angle(quat):
    scalar = float(np.clip(quat[3], -1.0, 1.0))
    denominator = math.sqrt(max(0.0, 1.0 - scalar * scalar))
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return np.asarray(quat[:3]) * (2.0 * math.acos(scalar) / denominator)


def object_positions(env):
    return {
        name: env.sim.data.body_xpos[env.env.obj_body_id[name]].astype(float).tolist()
        for name in OBJECTS
        if name in env.env.obj_body_id
    }


def model_observation(obs, prompt):
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
    state = np.concatenate(
        (obs["robot0_eef_pos"], axis_angle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--scene-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    tasks = [suite.get_task(index) for index in range(suite.n_tasks)]
    scene_task = next(task for task in tasks if "ketchup" in task.language.lower())
    prompts = {}
    for name in OBJECTS:
        phrase = name[:-2].replace("_", " ")
        prompts[name] = next(
            task.language for task in tasks if phrase in task.language.lower()
        )

    task_file = Path(get_libero_path("bddl_files")) / scene_task.problem_folder / scene_task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(task_file), camera_heights=256, camera_widths=256)
    env.seed(7)
    initial_state = suite.get_task_init_states(tasks.index(scene_task))[args.init_index]
    client = None if args.scene_only else WebsocketClientPolicy(args.host, args.port)
    results = []
    try:
        for target, prompt in prompts.items():
            env.reset()
            obs = env.set_init_state(initial_state)
            for _ in range(10):
                obs, _, _, _ = env.step(WAIT_ACTION)
            initial = object_positions(env)
            first_input = model_observation(obs, prompt)
            scene_hash = hashlib.sha256(first_input["observation/image"].tobytes()).hexdigest()
            if args.scene_only:
                result = {
                    "target": target,
                    "prompt": prompt,
                    "initial_image_sha256": scene_hash,
                    "initial_positions": initial,
                }
                results.append(result)
                print(json.dumps(result, ensure_ascii=True), flush=True)
                continue
            actions = collections.deque()
            first_action = None
            first_chunk = None
            target_success_step = None
            for step in range(args.max_steps):
                if not actions:
                    action_chunk = client.infer(model_observation(obs, prompt))["actions"]
                    if first_chunk is None:
                        first_chunk = np.asarray(action_chunk[: args.replan_steps]).tolist()
                    actions.extend(action_chunk[: args.replan_steps])
                action = np.asarray(actions.popleft(), dtype=float)
                if first_action is None:
                    first_action = action.tolist()
                obs, _, _, _ = env.step(action.tolist())
                if env.env._eval_predicate(("in", target, "basket_1_contain_region")):
                    target_success_step = step + 1
                    break
            final = object_positions(env)
            displacement = {
                name: float(np.linalg.norm(np.asarray(final[name]) - np.asarray(initial[name])))
                for name in OBJECTS
            }
            result = {
                "target": target,
                "prompt": prompt,
                "init_index": args.init_index,
                "initial_image_sha256": scene_hash,
                "first_action": first_action,
                "first_chunk": first_chunk,
                "initial_positions": initial,
                "final_positions": final,
                "displacement_m": displacement,
                "most_moved_object": max(displacement, key=displacement.get),
                "target_success_step": target_success_step,
                "original_bddl_success": bool(env.check_success()),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
    finally:
        env.close()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
