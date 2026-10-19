"""Connect the existing supervised task loop to official PI0.5-LIBERO."""

import argparse
import json
from urllib.request import Request, urlopen
from pathlib import Path

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from lower_adapter.pi05_adapter import ChunkResult
from runtime.supervised_loop import SupervisedAgentLoop
from upper.qwen_supervisor import SupervisorAction, SupervisorDecision
from upper.schemas import Subtask, TaskPlan
from upper.task_memory import JsonTaskMemoryStore

from probe_pi05_libero_counterfactual import WAIT_ACTION, model_observation
from probe_pi05_libero_two_stage import in_basket, record_frame, return_home


TARGETS = ("bbq_sauce_1", "ketchup_1")
TASK = "Place the bbq sauce and the ketchup in the basket."


def qwen_request(endpoint, route, payload):
    request = Request(
        endpoint.rstrip("/") + route,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def canonical_plan(task, prompts, order):
    return TaskPlan(
        task=task,
        subtasks=[
            Subtask(
                id=index,
                name=prompts[target],
                success_condition="{} in basket".format(target),
                preconditions={order[index - 1]: "in_basket"} if index else {},
                expected_effects={target: "in_basket"},
            )
            for index, target in enumerate(order)
        ],
        success_condition="both objects in basket",
    )


class LiberoTwoObjectPlanner:
    def __init__(self, prompts):
        self.prompts = prompts

    def plan(self, task, observation=None, experience=None):
        if task != TASK:
            raise ValueError("unsupported LIBERO task")
        return canonical_plan(task, self.prompts, TARGETS)


class QwenMenuPlanner:
    def __init__(self, prompts, endpoint):
        self.prompts = prompts
        self.endpoint = endpoint
        self.responses = []

    def plan(self, task, observation=None, experience=None):
        response = qwen_request(
            self.endpoint, "/plan", {"task": task, "image_path": observation}
        )
        self.responses.append(response)
        order = response.get("order")
        if not isinstance(order, list) or len(order) != len(TARGETS) or set(order) != set(TARGETS):
            raise ValueError("Qwen plan must use each canonical target exactly once: %r" % response)
        return canonical_plan(task, self.prompts, order)


class LiberoPredicateSupervisor:
    def __init__(self, prompt_targets):
        self.prompt_targets = prompt_targets

    def decide(self, task, command, image_path, observations=None):
        observations = observations or {}
        if observations.get("transition_failed") == "true":
            return SupervisorDecision(
                action=SupervisorAction.BLOCKED,
                message="return-to-ready motion failed or previous goal was lost",
                observations=observations,
            )
        target = self.prompt_targets[command.instruction]
        if observations.get("durable_target_success") == "true":
            return SupervisorDecision(
                action=SupervisorAction.ADVANCE,
                message="LIBERO in-basket predicate remained true after settling",
                observations=observations,
                state_updates={target: "in_basket"},
                evidence=["in({}, basket_1_contain_region)".format(target)],
            )
        if observations.get("timed_out") == "true":
            return SupervisorDecision(
                action=SupervisorAction.BLOCKED,
                message="subtask exceeded action budget",
                observations=observations,
            )
        return SupervisorDecision(action=SupervisorAction.KEEP, observations=observations)


class QwenStageSupervisor:
    def __init__(self, prompt_targets, env, endpoint, grounded_fallback=False):
        self.prompt_targets = prompt_targets
        self.env = env
        self.endpoint = endpoint
        self.grounded_fallback = grounded_fallback
        self.decisions = []

    def decide(self, task, command, image_path, observations=None):
        observations = observations or {}
        if observations.get("transition_failed") == "true":
            return SupervisorDecision(
                action=SupervisorAction.BLOCKED,
                message="return-to-ready motion failed or previous goal was lost",
                observations=observations,
            )
        target = self.prompt_targets[command.instruction]
        if image_path:
            try:
                response = qwen_request(
                    self.endpoint,
                    "/decide",
                    {
                        "task": task,
                        "active_target": target,
                        "world_state": {
                            **command.world_state,
                            "image_layout": (
                                "Left panel is before this subtask; right panel is current. "
                                "Compare the active target across panels, not other bottles."
                            ),
                        },
                        "stage_actions": int(observations.get("stage_actions", "0")),
                        "image_path": image_path,
                    },
                )
            except Exception as exc:
                response = {"decision": "KEEP", "error": str(exc)}
            decision = str(response.get("decision", "KEEP")).upper()
            oracle = in_basket(self.env, target)
            gripper_open = float(observations.get("gripper_open", "0"))
            grounded = oracle and gripper_open >= 0.025
            accepted = grounded and (decision == "ADVANCE" or self.grounded_fallback)
            accepted_by = (
                "qwen_and_guard" if accepted and decision == "ADVANCE"
                else "grounded_fallback" if accepted else None
            )
            self.decisions.append(
                {
                    "target": target,
                    "stage_actions": observations.get("stage_actions"),
                    "qwen_decision": decision,
                    "evidence": response.get("evidence"),
                    "raw_response": response.get("raw_response"),
                    "error": response.get("error"),
                    "oracle_in_basket": oracle,
                    "gripper_open": gripper_open,
                    "accepted": accepted,
                    "accepted_by": accepted_by,
                }
            )
            if accepted:
                return SupervisorDecision(
                    action=SupervisorAction.ADVANCE,
                    message="subtask completion accepted by %s" % accepted_by,
                    observations=observations,
                    state_updates={target: "in_basket"},
                    evidence=(
                        [str(response.get("evidence", "visual completion"))]
                        if accepted_by == "qwen_and_guard"
                        else ["simulator containment and released gripper"]
                    ),
                )
        if observations.get("timed_out") == "true":
            return SupervisorDecision(
                action=SupervisorAction.BLOCKED,
                message="subtask exceeded action budget without confirmed visual completion",
                observations=observations,
            )
        return SupervisorDecision(action=SupervisorAction.KEEP, observations=observations)


class LiberoPI05ChunkAdapter:
    def __init__(self, env, obs, client, prompt_targets, home_pos, home_quat, replan_steps, max_actions, writer=None, image_dir=None, qwen_mode=False):
        self.env = env
        self.obs = obs
        self.client = client
        self.prompt_targets = prompt_targets
        self.home_pos = home_pos
        self.home_quat = home_quat
        self.replan_steps = replan_steps
        self.max_actions = max_actions
        self.writer = writer
        self.image_dir = image_dir
        self.qwen_mode = qwen_mode
        self.active_subtask_id = None
        self.active_target = None
        self.stage_reference = None
        self.stage_actions = 0
        self.transitions = []

    def execute_chunk(self, command):
        target = self.prompt_targets[command.instruction]
        if command.subtask_id != self.active_subtask_id:
            if self.active_subtask_id is not None:
                self.obs, transition = return_home(
                    self.env, self.obs, self.home_pos, self.home_quat, self.writer, tight=True
                )
                previous_target = self.active_target
                transition["previous_goal_retained"] = in_basket(self.env, previous_target)
                self.transitions.append(transition)
                if not transition["reached_home"] or not transition["previous_goal_retained"]:
                    return ChunkResult(
                        image_path=None,
                        observations={"transition_failed": "true"},
                    )
            self.active_subtask_id = command.subtask_id
            self.active_target = target
            self.stage_actions = 0
            self.stage_reference = np.ascontiguousarray(
                self.obs["agentview_image"][::-1, ::-1]
            )

        if not self.qwen_mode and in_basket(self.env, target):
            return ChunkResult(
                image_path=None,
                observations={"durable_target_success": "true", "target": target},
            )

        chunk = self.client.infer(model_observation(self.obs, command.instruction))["actions"]
        if len(chunk) < self.replan_steps:
            raise ValueError("policy action chunk is shorter than replan_steps")
        actions_sent = 0
        reached = False
        for action in chunk[: self.replan_steps]:
            if self.stage_actions >= self.max_actions:
                break
            self.obs, _, _, _ = self.env.step(np.asarray(action, dtype=float).tolist())
            record_frame(
                self.writer,
                self.obs,
                "SUBTASK %d %s" % (command.subtask_id + 1, target.upper()),
                "cmd:%+.2f open:%.3f" % (action[-1], abs(float(self.obs["robot0_gripper_qpos"][0]))),
            )
            self.stage_actions += 1
            actions_sent += 1
            if not self.qwen_mode and in_basket(self.env, target):
                reached = True
                break
        if reached:
            for _ in range(10):
                self.obs, _, _, _ = self.env.step(WAIT_ACTION)
                record_frame(self.writer, self.obs, "SUBTASK %d SETTLE" % (command.subtask_id + 1))
        durable = in_basket(self.env, target)
        image_path = None
        if self.qwen_mode and (self.stage_actions % 25 == 0 or self.stage_actions >= self.max_actions):
            image_path = self.image_dir / ("stage%d_step%03d.jpg" % (command.subtask_id + 1, self.stage_actions))
            current_image = np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1])
            imageio.imwrite(
                str(image_path),
                np.concatenate((self.stage_reference, current_image), axis=1),
            )
        return ChunkResult(
            image_path=str(image_path) if image_path else None,
            actions_sent=actions_sent,
            observations={
                "target": target,
                "stage_actions": str(self.stage_actions),
                "durable_target_success": str(durable).lower() if not self.qwen_mode else "hidden_from_qwen",
                "gripper_open": str(abs(float(self.obs["robot0_gripper_qpos"][0]))),
                "timed_out": str(
                    self.stage_actions >= self.max_actions and (self.qwen_mode or not durable)
                ).lower(),
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-actions-per-stage", type=int, default=400)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video")
    parser.add_argument("--upper-mode", choices=("rule", "qwen", "qwen_hybrid"), default="rule")
    parser.add_argument("--qwen-endpoint", default="http://127.0.0.1:8011")
    parser.add_argument("--image-dir")
    args = parser.parse_args()
    if Path(args.memory).exists():
        raise FileExistsError("memory path already exists; a reset LIBERO scene cannot resume that state")

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    tasks = [suite.get_task(index) for index in range(suite.n_tasks)]
    scene_task = next(task for task in tasks if "ketchup" in task.language.lower())
    prompts = {
        target: next(
            task.language
            for task in tasks
            if target[:-2].replace("_", " ") in task.language.lower()
        )
        for target in TARGETS
    }
    prompt_targets = {prompt: target for target, prompt in prompts.items()}
    bddl_path = Path(get_libero_path("bddl_files")) / scene_task.problem_folder / scene_task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256)
    env.seed(7)
    writer = None
    image_dir = None
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(tasks.index(scene_task))[args.init_index])
        for _ in range(10):
            obs, _, _, _ = env.step(WAIT_ACTION)
        if args.video:
            video_path = Path(args.video)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video_path), fps=10)
            record_frame(writer, obs, "START")
        initial_image = None
        if args.upper_mode != "rule":
            image_dir = Path(args.image_dir) if args.image_dir else Path(args.output).with_suffix("")
            image_dir.mkdir(parents=True, exist_ok=True)
            initial_image = image_dir / "initial.jpg"
            imageio.imwrite(
                str(initial_image),
                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            )
        lower = LiberoPI05ChunkAdapter(
            env=env,
            obs=obs,
            client=WebsocketClientPolicy(args.host, args.port),
            prompt_targets=prompt_targets,
            home_pos=np.asarray(obs["robot0_eef_pos"], dtype=float),
            home_quat=np.asarray(obs["robot0_eef_quat"], dtype=float),
            replan_steps=args.replan_steps,
            max_actions=args.max_actions_per_stage,
            writer=writer,
            image_dir=image_dir,
            qwen_mode=args.upper_mode != "rule",
        )
        planner = (
            QwenMenuPlanner(prompts, args.qwen_endpoint)
            if args.upper_mode != "rule"
            else LiberoTwoObjectPlanner(prompts)
        )
        supervisor = (
            QwenStageSupervisor(
                prompt_targets,
                env,
                args.qwen_endpoint,
                grounded_fallback=args.upper_mode == "qwen_hybrid",
            )
            if args.upper_mode != "rule"
            else LiberoPredicateSupervisor(prompt_targets)
        )
        loop = SupervisedAgentLoop(
            planner=planner,
            lower=lower,
            supervisor=supervisor,
            store=JsonTaskMemoryStore(args.memory),
        )
        memory = loop.run(
            TASK,
            observation=str(initial_image) if initial_image else None,
            max_chunks=2 * args.max_actions_per_stage,
        )
        basket_state = {target: in_basket(env, target) for target in TARGETS}
        summary = {
            "init_index": args.init_index,
            "task": TASK,
            "upper_mode": args.upper_mode,
            "planned_subtasks": [step.name for step in memory.plan.subtasks],
            "finished": memory.finished,
            "blocked": memory.blocked,
            "current_step": memory.current_step,
            "basket_state": basket_state,
            "success": memory.finished and all(basket_state.values()),
            "transitions": lower.transitions,
            "transition_mode": "physical_tight_return_home",
            "video_path": args.video,
            "memory_path": args.memory,
            "image_dir": str(image_dir) if image_dir else None,
            "qwen_plan_responses": planner.responses if args.upper_mode != "rule" else [],
            "qwen_decisions": supervisor.decisions if args.upper_mode != "rule" else [],
            "stage_events": [
                event for event in memory.events if event["event"] in ("subtask_started", "subtask_reported")
            ],
        }
    finally:
        if writer is not None:
            writer.close()
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
