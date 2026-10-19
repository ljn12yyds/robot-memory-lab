"""Run two canonical LIBERO object instructions without resetting between stages."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import cv2
import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from scipy.spatial.transform import Rotation

from probe_pi05_libero_counterfactual import WAIT_ACTION, model_observation, object_positions


TARGETS = ("ketchup_1", "bbq_sauce_1")
SCENE_TARGETS = (
    *TARGETS,
    "salad_dressing_1",
    "alphabet_soup_1",
    "cream_cheese_1",
    "milk_1",
)
ORDERS = (TARGETS, TARGETS[::-1])


def record_frame(writer, obs, label, detail=None):
    if writer is None:
        return
    frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 48 if detail else 29), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if detail:
        cv2.putText(frame, detail, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    writer.append_data(frame)


def in_basket(env, target):
    return bool(env.env._eval_predicate(("in", target, "basket_1_contain_region")))


def arm_joint_indices(env):
    robot = env.env.robots[0]
    qpos = list(robot._ref_joint_pos_indexes) + list(robot._ref_gripper_joint_pos_indexes)
    qvel = list(robot._ref_joint_vel_indexes) + list(robot._ref_gripper_joint_vel_indexes)
    return robot, qpos, qvel


def return_home(env, obs, home_position, home_quat, writer=None, tight=False):
    safe_height = max(0.35, float(home_position[2]) + 0.08)
    current = np.asarray(obs["robot0_eef_pos"], dtype=float)
    waypoints = (
        np.asarray((current[0], current[1], safe_height)),
        np.asarray((home_position[0], home_position[1], safe_height)),
        home_position,
    )
    steps = 0
    for waypoint in waypoints:
        for _ in range(80):
            current = np.asarray(obs["robot0_eef_pos"], dtype=float)
            relative_rotation = Rotation.from_quat(home_quat) * Rotation.from_quat(
                obs["robot0_eef_quat"]
            ).inv()
            rotation_error = relative_rotation.as_rotvec()
            if np.linalg.norm(waypoint - current) < (0.004 if tight else 0.015) and np.linalg.norm(rotation_error) < (0.05 if tight else 0.15):
                break
            action = np.concatenate(
                (
                    np.clip((waypoint - current) / 0.05, -1.0, 1.0),
                    np.clip(rotation_error / 0.5, -1.0, 1.0),
                    [-1.0],
                )
            )
            obs, _, _, _ = env.step(action.tolist())
            record_frame(writer, obs, "RETURN HOME")
            steps += 1

    final_position = np.asarray(obs["robot0_eef_pos"], dtype=float)
    final_rotation_error = (
        Rotation.from_quat(home_quat) * Rotation.from_quat(obs["robot0_eef_quat"]).inv()
    ).magnitude()
    position_error = float(np.linalg.norm(final_position - home_position))
    return obs, {
        "steps": steps,
        "position_error_m": position_error,
        "orientation_error_rad": float(final_rotation_error),
        "reached_home": position_error < (0.008 if tight else 0.025) and final_rotation_error < (0.08 if tight else 0.2),
        "eef_end": final_position.tolist(),
    }


def run_stage(env, client, obs, target, prompt, max_steps, replan_steps, settle_steps, writer=None, label="", trace_actions=False, continue_after_success=False, require_release=False):
    actions = collections.deque()
    start_positions = object_positions(env)
    eef_start = np.asarray(obs["robot0_eef_pos"], dtype=float)
    target_start = np.asarray(start_positions[target], dtype=float)
    min_eef_target_distance = float(np.linalg.norm(eef_start - target_start))
    eef_trace = [{"step": 0, "position": eef_start.tolist()}]
    success_step = None
    first_action = None
    action_trace = [] if trace_actions else None
    for step in range(max_steps):
        if not actions:
            chunk = client.infer(model_observation(obs, prompt))["actions"]
            if len(chunk) < replan_steps:
                raise ValueError("policy action chunk is shorter than replan_steps")
            actions.extend(chunk[:replan_steps])
        action = np.asarray(actions.popleft(), dtype=float)
        if first_action is None:
            first_action = action.tolist()
        obs, _, _, _ = env.step(action.tolist())
        record_frame(
            writer,
            obs,
            label,
            "cmd:%+.2f open:%.3f" % (action[-1], abs(float(obs["robot0_gripper_qpos"][0]))),
        )
        if trace_actions:
            action_trace.append({
                "step": step + 1,
                "action": action.tolist(),
                "gripper_action": float(action[-1]),
                "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=float).tolist(),
                "target_position": object_positions(env)[target],
                "eef_position": np.asarray(obs["robot0_eef_pos"], dtype=float).tolist(),
                "in_basket": in_basket(env, target),
            })
        eef_position = np.asarray(obs["robot0_eef_pos"], dtype=float)
        min_eef_target_distance = min(
            min_eef_target_distance,
            float(np.linalg.norm(eef_position - target_start)),
        )
        if (step + 1) % 20 == 0:
            eef_trace.append({"step": step + 1, "position": eef_position.tolist()})
        if in_basket(env, target) and (not require_release or abs(float(obs["robot0_gripper_qpos"][0])) >= 0.025) and success_step is None:
            success_step = step + 1
            if not continue_after_success:
                break

    for _ in range(settle_steps if success_step is not None and not continue_after_success else 0):
        obs, _, _, _ = env.step(WAIT_ACTION)
        record_frame(writer, obs, label + " - SETTLE")

    end_positions = object_positions(env)
    eef_end = np.asarray(obs["robot0_eef_pos"], dtype=float)
    return obs, {
        "target": target,
        "prompt": prompt,
        "success_step": success_step,
        "durable_success": in_basket(env, target),
        "first_action": first_action,
        "eef_start": eef_start.tolist(),
        "eef_end": eef_end.tolist(),
        "min_eef_target_distance_m": min_eef_target_distance,
        "eef_trace": eef_trace,
        "start_positions": start_positions,
        "end_positions": end_positions,
        "displacement_m": {
            name: float(np.linalg.norm(np.asarray(end_positions[name]) - np.asarray(start_positions[name])))
            for name in start_positions
        },
        "basket_state_after": {
            name: in_basket(env, name) for name in SCENE_TARGETS if name in env.env.obj_body_id
        },
        "action_trace": action_trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--max-steps-per-stage", type=int, default=280)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--settle-steps", type=int, default=10)
    transition = parser.add_mutually_exclusive_group()
    transition.add_argument("--diagnostic-reset-arm", action="store_true")
    transition.add_argument("--physical-return-home", action="store_true")
    parser.add_argument("--tight-return-home", action="store_true")
    order_selection = parser.add_mutually_exclusive_group()
    order_selection.add_argument("--reverse-only", action="store_true")
    order_selection.add_argument("--forward-only", action="store_true")
    parser.add_argument("--single-target", choices=SCENE_TARGETS)
    parser.add_argument("--native-scene", action="store_true")
    parser.add_argument("--preload-bbq", action="store_true")
    parser.add_argument("--video-dir")
    parser.add_argument("--trace-actions", action="store_true")
    parser.add_argument("--diagnostic-release", action="store_true")
    parser.add_argument("--continue-after-success", action="store_true")
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.preload_bbq and args.single_target != "ketchup_1":
        parser.error("--preload-bbq requires --single-target ketchup_1")
    if args.native_scene and not args.single_target:
        parser.error("--native-scene requires --single-target")

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    tasks = [suite.get_task(index) for index in range(suite.n_tasks)]
    scene_target = args.single_target if args.native_scene else "ketchup_1"
    scene_phrase = scene_target[:-2].replace("_", " ")
    scene_task = next(task for task in tasks if scene_phrase in task.language.lower())
    prompts = {
        target: next(
            task.language
            for task in tasks
            if target[:-2].replace("_", " ") in task.language.lower()
        )
        for target in SCENE_TARGETS
    }
    bddl_path = Path(get_libero_path("bddl_files")) / scene_task.problem_folder / scene_task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256)
    env.seed(7)
    initial_state = suite.get_task_init_states(tasks.index(scene_task))[args.init_index]
    client = WebsocketClientPolicy(args.host, args.port)
    results = []
    try:
        orders = ((args.single_target,),) if args.single_target else (ORDERS[1:] if args.reverse_only else ORDERS[:1] if args.forward_only else ORDERS)
        for order in orders:
            env.reset()
            obs = env.set_init_state(initial_state)
            for _ in range(10):
                obs, _, _, _ = env.step(WAIT_ACTION)
            if args.preload_bbq:
                bbq_joint = env.env.objects_dict["bbq_sauce_1"].joints[0]
                bbq_qpos = env.sim.data.get_joint_qpos(bbq_joint).copy()
                bbq_qpos[:3] = [-0.0196, 0.2636, 0.0716]
                env.sim.data.set_joint_qpos(bbq_joint, bbq_qpos)
                env.sim.forward()
                for _ in range(20):
                    obs, _, _, _ = env.step(WAIT_ACTION)
            robot, robot_qpos, robot_qvel = arm_joint_indices(env)
            initial_robot_qpos = env.sim.data.qpos[robot_qpos].copy()
            home_eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
            home_eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=float)
            initial_image = model_observation(obs, prompts[order[0]])["observation/image"]
            result = {
                "init_index": args.init_index,
                "order": list(order),
                "diagnostic_reset_arm": args.diagnostic_reset_arm,
                "physical_return_home": args.physical_return_home,
                "tight_return_home": args.tight_return_home,
                "require_release": args.require_release,
                "native_scene": args.native_scene,
                "preload_bbq": args.preload_bbq,
                "initial_image_sha256": hashlib.sha256(initial_image.tobytes()).hexdigest(),
                "initial_basket_state": {
                    name: in_basket(env, name) for name in SCENE_TARGETS if name in env.env.obj_body_id
                },
                "stages": [],
            }
            writer = None
            if args.video_dir:
                video_dir = Path(args.video_dir)
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / ("init%d_%s.mp4" % (args.init_index, "_then_".join(order)))
                writer = imageio.get_writer(str(video_path), fps=10)
                result["video_path"] = str(video_path)
                record_frame(writer, obs, "START")
            try:
                for stage_index, target in enumerate(order):
                    label = "%d/%d %s" % (stage_index + 1, len(order), target.upper())
                    obs, stage = run_stage(
                        env,
                        client,
                        obs,
                        target,
                        prompts[target],
                        args.max_steps_per_stage,
                        args.replan_steps,
                        args.settle_steps,
                        writer,
                        label,
                        args.trace_actions,
                        args.continue_after_success,
                        args.require_release,
                    )
                    result["stages"].append(stage)
                    print(
                        json.dumps(
                            {
                                "order": order,
                                "target": target,
                                "success_step": stage["success_step"],
                                "durable_success": stage["durable_success"],
                                "basket_state_after": stage["basket_state_after"],
                            }
                        ),
                        flush=True,
                    )
                    if not stage["durable_success"]:
                        break
                    if args.diagnostic_reset_arm and len(order) == 2 and len(result["stages"]) == 1:
                        env.sim.data.qpos[robot_qpos] = initial_robot_qpos
                        env.sim.data.qvel[robot_qvel] = 0
                        env.sim.forward()
                        robot.controller.update(force=True)
                        robot.controller.reset_goal()
                        for _ in range(args.settle_steps):
                            obs, _, _, _ = env.step(WAIT_ACTION)
                            record_frame(writer, obs, "ARM RESET")
                        result["post_arm_reset_basket_state"] = {
                            name: in_basket(env, name) for name in SCENE_TARGETS if name in env.env.obj_body_id
                        }
                        result["post_arm_reset_eef"] = np.asarray(obs["robot0_eef_pos"], dtype=float).tolist()
                        if not in_basket(env, target):
                            break
                    if args.physical_return_home and len(order) == 2 and len(result["stages"]) == 1:
                        obs, home_result = return_home(env, obs, home_eef_pos, home_eef_quat, writer, args.tight_return_home)
                        result["return_home"] = home_result
                        result["post_return_basket_state"] = {
                            name: in_basket(env, name) for name in SCENE_TARGETS if name in env.env.obj_body_id
                        }
                        if not home_result["reached_home"] or not in_basket(env, target):
                            break
            finally:
                if writer is not None:
                    writer.close()
            result["final_basket_state"] = {
                name: in_basket(env, name) for name in SCENE_TARGETS if name in env.env.obj_body_id
            }
            result["two_stage_success"] = len(result["stages"]) == 2 and all(
                in_basket(env, name) for name in order
            )
            result["single_task_success"] = len(order) == 1 and in_basket(env, order[0])
            if args.diagnostic_release and len(result["stages"]) == 2 and not result["two_stage_success"]:
                before = object_positions(env)[order[1]]
                for _ in range(30):
                    obs, _, _, _ = env.step(WAIT_ACTION)
                result["diagnostic_release"] = {
                    "policy_result_unchanged": True,
                    "target_position_before": before,
                    "target_position_after": object_positions(env)[order[1]],
                    "basket_state_after": {
                        name: in_basket(env, name) for name in SCENE_TARGETS if name in env.env.obj_body_id
                    },
                    "gripper_qpos_after": np.asarray(obs["robot0_gripper_qpos"], dtype=float).tolist(),
                }
            results.append(result)
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "two_stage_success": [r["two_stage_success"] for r in results]}))


if __name__ == "__main__":
    main()
