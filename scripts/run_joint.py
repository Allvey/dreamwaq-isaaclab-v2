"""Two isolated IsaacLab samplers and one unchanged DreamWaQ/RSL-RL learner.

Only joint-train calls runner.learn. joint-check discards one native PPO update.
All process transport is CPU NumPy data; CUDA tensors/contexts are never shared.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    used, total = result.stdout.splitlines()[0].split(",")
    return {"used_mib": int(used), "total_mib": int(total)}


def worker(connection, spec):
    """Own one simulator, no policy, value network or optimizer."""
    log = open(spec["log"], "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    app = env = None
    try:
        sys.argv = [__file__]
        from isaaclab.app import AppLauncher
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        app = AppLauncher(parser.parse_args(["--headless", "--device", spec["device"]])).app
        import torch
        import soft_terrain as soft
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml

        torch.set_num_threads(1)
        cfg = soft.Go2MixedEnvCfg()
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = spec["num_envs"], spec["seed"], spec["device"]
        cfg.scene.terrain.training_stage = spec["stage"]
        cfg.scene.terrain.isolated_training = True
        cfg.scene.terrain.physics_substeps = spec["substeps"]
        cfg.scene.terrain.soft_level_ceiling = spec["level"]
        # Default buffers target thousands of robots and are duplicated by
        # separate PhysX scenes. Size storage, never solver/contact parameters.
        def capacity(minimum, per_robot):
            return 1 << (max(minimum, spec["num_envs"] * per_robot) - 1).bit_length()

        cfg.sim.physx.gpu_max_rigid_contact_count = capacity(2**18, 2048)
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = capacity(2**20, 8192)
        # No changes to rewards, commands, randomization, actuators or observations.
        env = RslRlVecEnvWrapper(soft.make_env(
            cfg, isolated_training=spec["formal"] and not spec["from_scratch"], stage_report=spec["report"],
        ))
        raw = env.unwrapped
        log_offset = 0

        def check_physics_log():
            nonlocal log_offset
            log.flush()
            with open(spec["log"]) as recorded:
                recorded.seek(log_offset)
                lines = recorded.read()
                log_offset = recorded.tell()
            for line in lines.splitlines():
                if ("[Error]" in line or re.search(r"overflow|exceeded|out of memory", line, re.I)) and re.search(
                        r"physx|cuda|contact|buffer", line, re.I):
                    raise RuntimeError(f"Physics capacity/device failure; refusing truncated physics: {line}")

        check_physics_log()
        soil = getattr(raw.action_manager.get_term("joint_pos"), "soil", None)
        assert (soil is None) == (spec["stage"] == "hard")
        assert cfg.decimation == spec["substeps"] and abs(raw.step_dt - 0.02) < 1e-12
        assert abs(cfg.sim.dt * cfg.scene.contact_forces.history_length - .015) < 1e-12
        if soil is not None:
            assert cfg.scene.terrain.resolved_soft_tiles == 5
            assert "terrain" not in raw.scene.terrain.meshes
        else:
            assert set(raw.scene.terrain.meshes) == {"terrain"}
            assert "stage_boundary" not in raw.termination_manager.active_terms
        dump_yaml(str(Path(spec["directory"]) / "params" / f"{spec['stage']}_env.yaml"), cfg)
        agent = soft.Go2MixedRunnerCfg().to_dict()
        state = soft.soil_stage_state(spec["source_state"], cfg.scene.terrain)

        def array(value):
            return value.detach().cpu().numpy().copy()

        def observations():
            # Native reset/step already computed observations. Reading this
            # cache must not append an extra frame to CENet's history.
            return {key: array(value) for key, value in raw.obs_buf.items()}

        def packet():
            return {"obs": observations(), "episode_length": array(env.episode_length_buf)}

        connection.send({
            **packet(), "stage": spec["stage"], "num_envs": env.num_envs,
            "num_actions": env.num_actions, "max_episode_length": env.max_episode_length,
            "physics_dt": cfg.sim.dt, "substeps": cfg.decimation,
            "control_period": raw.step_dt, "contact_samples": cfg.scene.contact_forces.history_length,
            "soil_spec": soft.soil_curriculum_spec(cfg.scene.terrain), "stage_state": state,
            "agent_cfg": agent, "pid": os.getpid(), "policy_instances": 0,
            "soft_tiles": cfg.scene.terrain.resolved_soft_tiles,
        })
        sequence, version = 0, 0
        while True:
            request = connection.recv()
            command = request["command"]
            if command == "close":
                connection.send({"closed": True})
                break
            if command == "episode_length":
                env.episode_length_buf = torch.as_tensor(request["value"], device=env.device, dtype=torch.long)
                connection.send({"ok": True})
            elif command == "reset":
                env.reset()
                sequence = 0
                connection.send(packet())
            elif command == "probe":
                if soil is not None:
                    soil.hole_d[:] = .03
                    untouched = soil.hole_d[1:].clone()
                    with torch.inference_mode():
                        raw._reset_idx(torch.tensor([0], device=env.device))
                    assert soil.hole_d[0].count_nonzero() == 0
                    assert (soil.mem[0] == -1).all()
                    torch.testing.assert_close(soil.hole_d[1:], untouched, rtol=0, atol=0)
                env.reset()
                # The next real step must propagate a native timeout through
                # the transport, independently for both groups.
                env.episode_length_buf[0] = env.max_episode_length - 1
                sequence = 0
                connection.send(packet())
            elif command == "step":
                assert request["sequence"] == sequence + 1, "Stale/out-of-order action packet"
                assert request["version"] >= version, "Stale policy version"
                sequence, version = request["sequence"], request["version"]
                action = torch.as_tensor(request["actions"], device=env.device)
                assert action.shape == (env.num_envs, env.num_actions) and torch.isfinite(action).all()
                with torch.inference_mode():
                    _, rewards, dones, extras = env.step(action)
                check_physics_log()
                info = extras.get("log", {})
                keys = list(info)
                values = [torch.as_tensor(info[key], device=env.device).float().mean() for key in keys]
                logs = dict(zip(keys, torch.stack(values).cpu().tolist())) if keys else {}
                timeouts = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
                connection.send({
                    **packet(), "reward": array(rewards), "done": array(dones),
                    "time_outs": array(timeouts), "log": logs,
                    "sequence": sequence, "version": version,
                    "reset_memory_clear": soil is None or bool((soil.hole_d[dones.bool()] == 0).all()),
                })
            else:
                raise ValueError(f"Unknown sampler command: {command}")
    except BaseException:
        failure = traceback.format_exc()
        print(failure, flush=True)
        try:
            connection.send({"error": failure})
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()


class JointEnv:
    """Synchronous vector environment; hard/soft stay on distinct env axes."""

    def __init__(self, specs, device):
        self.device = device
        self.processes, self.connections, self.ready = [], [], []
        self.sequence = self.policy_version = self.steps_in_version = 0
        self.memory_samples = [{"phase": "before_workers", **gpu_memory()}]
        self.policy = None
        self.parameter_versions = None
        context = mp.get_context("spawn")
        try:
            # Sequential startup avoids concurrent Kit extension/cache setup.
            for spec in specs:
                parent, child = context.Pipe()
                process = context.Process(target=worker, args=(child, spec), name=f"joint_{spec['stage']}")
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
                info = self.receive(len(self.connections) - 1, timeout=240)
                self.ready.append(info)
                sample = {"phase": f"{spec['stage']}_ready", **gpu_memory()}
                self.memory_samples.append(sample)
                print(f"[Joint] {spec['stage']} ready: pid={info['pid']}, envs={info['num_envs']}, "
                      f"substeps={info['substeps']}, sampled GPU memory={sample['used_mib']} MiB", flush=True)
            a, b = self.ready
            assert a["num_actions"] == b["num_actions"] == 12
            assert a["max_episode_length"] == b["max_episode_length"]
            assert a["control_period"] == b["control_period"] == .02
            assert a["agent_cfg"] == b["agent_cfg"]
            assert a["soil_spec"] == b["soil_spec"]
            self.hard_envs = a["num_envs"]
            self.num_envs = self.hard_envs + b["num_envs"]
            self.num_actions = 12
            self.max_episode_length = a["max_episode_length"]
            self.accept(self.ready)
        except BaseException:
            self.close()
            raise

    def receive(self, index, timeout=120):
        connection, process = self.connections[index], self.processes[index]
        deadline = time.monotonic() + timeout
        while not connection.poll(.2):
            if not process.is_alive():
                raise RuntimeError(f"Sampler {process.name} exited with {process.exitcode}; inspect its worker log")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Sampler {process.name} response timed out")
        try:
            result = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"Sampler {process.name} closed without a response; inspect its worker log") from error
        if "error" in result:
            raise RuntimeError(f"Sampler {process.name} failed:\n{result['error']}")
        return result

    def accept(self, packets):
        import numpy as np
        import torch
        from tensordict import TensorDict
        keys = set(packets[0]["obs"])
        assert keys == set(packets[1]["obs"])
        self.obs = TensorDict({key: torch.as_tensor(np.concatenate([p["obs"][key] for p in packets]), device=self.device)
                               for key in keys}, batch_size=[self.num_envs])
        for key, width in {"policy": 45, "prop_history": 225, "height_scan": 187, "privileged_vel": 3}.items():
            assert self.obs[key].shape == (self.num_envs, width)
            if not torch.isfinite(self.obs[key]).all():
                raise RuntimeError(f"Non-finite joint observation: {key}")
        self._episode_length = torch.as_tensor(np.concatenate([p["episode_length"] for p in packets]), device=self.device)

    def get_observations(self):
        return self.obs

    @property
    def episode_length_buf(self):
        return self._episode_length

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        array = value.detach().cpu().numpy()
        for connection, part in zip(self.connections, (array[:self.hard_envs], array[self.hard_envs:])):
            connection.send({"command": "episode_length", "value": part})
        for index in range(2):
            self.receive(index)
        self._episode_length = value.clone()

    def reset(self, probe=False):
        for connection in self.connections:
            connection.send({"command": "probe" if probe else "reset"})
        self.accept([self.receive(index) for index in range(2)])
        self.sequence = 0
        self.steps_in_version = 0
        self.parameter_versions = None
        return self.obs, {}

    def step(self, actions):
        import numpy as np
        import torch
        if self.steps_in_version == 0 and self.policy is not None:
            self.parameter_versions = tuple(p._version for p in self.policy.parameters())
        self.sequence += 1
        array = actions.detach().cpu().numpy()
        for connection, part in zip(self.connections, (array[:self.hard_envs], array[self.hard_envs:])):
            connection.send({"command": "step", "actions": part, "sequence": self.sequence, "version": self.policy_version})
        packets = [self.receive(index) for index in range(2)]
        assert all(p["sequence"] == self.sequence and p["version"] == self.policy_version for p in packets)
        self.accept(packets)
        reward = torch.as_tensor(np.concatenate([p["reward"] for p in packets]), device=self.device)
        done = torch.as_tensor(np.concatenate([p["done"] for p in packets]), device=self.device, dtype=torch.long)
        timeout = torch.as_tensor(np.concatenate([p["time_outs"] for p in packets]), device=self.device, dtype=torch.bool)
        assert torch.isfinite(reward).all() and ((done == 0) | (done == 1)).all()
        assert not (timeout & ~done.bool()).any()
        logs = {}
        for name, packet in zip(("hard", "soft"), packets):
            logs.update({f"Joint/{name}/{key}": value for key, value in packet["log"].items()})
            logs[f"Joint/{name}/reward_per_second"] = float(packet["reward"].mean()) / .02
            logs[f"Joint/{name}/termination_fraction"] = float(((packet["done"] != 0) & ~packet["time_outs"]).mean())
        self.last_packets = packets
        self.steps_in_version += 1
        return self.obs, reward, done, {"time_outs": timeout, "log": logs}

    def close(self):
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send({"command": "close"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for connection, process in zip(self.connections, self.processes):
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()  # Only this coordinator's own sampler.
                process.join(timeout=10)
            connection.close()


def make_runner(env, cfg, source, checkpoint, resume, stage, log_dir):
    import torch
    sys.path.insert(0, str(ROOT))
    import go2_dreamwaq  # Registers the unchanged custom classes, without launching Sim.
    from rsl_rl.runners import OnPolicyRunner

    class JointRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            metadata = dict(stage, soft_started=True)
            return super().save(path, dict(infos or {}, dreamwaq_stage=metadata, dreamwaq_joint={
                "protocol": "joint-sampling-v1", "policy_version": env.policy_version,
                "hard_envs": env.hard_envs, "soft_envs": env.num_envs - env.hard_envs,
                "hard_substeps": env.ready[0]["substeps"], "soft_substeps": env.ready[1]["substeps"],
                "control_period_s": .02, "initialization": self.initialization,
            }))

    runner = JointRunner(env, deepcopy(cfg), log_dir=log_dir, device=env.device)
    runner.initialization = "scratch" if checkpoint is None else "resume" if resume else "checkpoint_weights"
    if checkpoint is not None:
        runner.load(str(checkpoint), load_optimizer=resume, map_location=env.device)
        for key, value in runner.alg.policy.state_dict().items():
            torch.testing.assert_close(value, source["model_state_dict"][key].to(env.device), rtol=0, atol=0)
    if not resume:
        runner.current_learning_iteration = 0
        assert not runner.alg.optimizer.state
    print(f"[Joint] initialization={runner.initialization}; iteration={runner.current_learning_iteration}; "
          f"optimizer_states={len(runner.alg.optimizer.state)}", flush=True)
    env.policy = runner.alg.policy
    native_update = runner.alg.update

    def update():
        assert env.steps_in_version == runner.num_steps_per_env, "Incomplete joint rollout"
        assert env.parameter_versions == tuple(p._version for p in env.policy.parameters()), "Policy changed during sampling"
        losses = native_update()
        if not all(torch.isfinite(torch.as_tensor(value)).all() for value in losses.values()):
            raise RuntimeError("Non-finite joint PPO loss")
        env.policy_version += 1
        env.steps_in_version = 0
        return losses

    runner.alg.update = update  # Provenance guard only; native loss/optimizer remain unchanged.
    return runner


def check(runner, env, directory, stage):
    import torch
    if runner.initialization == "scratch":
        assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
    initial = {key: value.detach().clone() for key, value in runner.alg.policy.state_dict().items()}
    obs, _ = env.reset(probe=True)
    for key in obs.keys():
        torch.testing.assert_close(env.get_observations()[key], obs[key], rtol=0, atol=0)
    lengths = env.episode_length_buf.clone()
    lengths[1] = lengths[env.hard_envs + 1] = env.max_episode_length - 1
    env.episode_length_buf = lengths
    recorded, rewards_by_group = [], [0., 0.]
    start = time.perf_counter()
    runner.train_mode()
    with torch.inference_mode():
        for index in range(runner.num_steps_per_env):
            recorded.append({key: value.clone() for key, value in obs.items()})
            obs, reward, done, extras = env.step(runner.alg.act(obs))
            if index == 0:
                assert done[0] and done[env.hard_envs]
                assert extras["time_outs"][0] and extras["time_outs"][env.hard_envs]
                assert extras["time_outs"][1] and extras["time_outs"][env.hard_envs + 1]
                assert all(packet["reset_memory_clear"] for packet in env.last_packets)
            runner.alg.process_env_step(obs, reward, done, extras)
            rewards_by_group[0] += float(reward[:env.hard_envs].mean())
            rewards_by_group[1] += float(reward[env.hard_envs:].mean())
        last_values = runner.alg.policy.evaluate(obs).detach()
        runner.alg.compute_returns(obs)
    collection_seconds = time.perf_counter() - start
    for key, original in initial.items():
        torch.testing.assert_close(runner.alg.policy.state_dict()[key], original, rtol=0, atol=0)
    storage = runner.alg.storage
    # Check actual transported observations, and independently calculate GAE for
    # each group. No return may cross a robot/group boundary or a done marker.
    for key in recorded[0]:
        torch.testing.assert_close(storage.observations[key], torch.stack([row[key] for row in recorded]), rtol=0, atol=0)
    for selected in (slice(None, env.hard_envs), slice(env.hard_envs, None)):
        advantage = torch.zeros_like(last_values[selected])
        for step in reversed(range(runner.num_steps_per_env)):
            next_values = last_values[selected] if step == runner.num_steps_per_env - 1 else storage.values[step + 1, selected]
            live = 1. - storage.dones[step, selected].float()
            delta = storage.rewards[step, selected] + runner.alg.gamma * next_values * live - storage.values[step, selected]
            advantage = delta + runner.alg.gamma * runner.alg.lam * live * advantage
            torch.testing.assert_close(storage.returns[step, selected], advantage + storage.values[step, selected])
    env.memory_samples.append({"phase": "after_rollout", **gpu_memory()})
    started = time.perf_counter()
    losses = runner.alg.update()
    update_seconds = time.perf_counter() - started
    assert any(not torch.equal(initial[key], value) for key, value in runner.alg.policy.state_dict().items())
    assert all(torch.isfinite(value).all() for value in runner.alg.policy.state_dict().values())
    assert env.policy_version == 1
    runner.logger_type = "tensorboard"
    with tempfile.TemporaryDirectory(prefix="dreamwaq_joint_checkpoint_") as temporary:
        path = Path(temporary) / "check.pt"
        runner.save(str(path))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["infos"]["dreamwaq_stage"] == dict(stage, soft_started=True)
        assert saved["infos"]["dreamwaq_joint"]["policy_version"] == 1
        assert saved["infos"]["dreamwaq_joint"]["initialization"] == runner.initialization
        runner.load(str(path), map_location=env.device)
        assert runner.alg.optimizer.state
    env.memory_samples.append({"phase": "after_update", **gpu_memory()})
    summary = {
        "completed": True, "diagnostic_only": True, "formal_training_started": False,
        "initialization": runner.initialization,
        "policy_instances": 1, "optimizer_instances": 1, "policy_frozen_during_collection": True,
        "hard_envs": env.hard_envs, "soft_envs": env.num_envs - env.hard_envs,
        "hard_substeps": env.ready[0]["substeps"], "soft_substeps": env.ready[1]["substeps"],
        "control_period_s": .02, "samples_per_group": [runner.num_steps_per_env * env.hard_envs,
                                                       runner.num_steps_per_env * (env.num_envs - env.hard_envs)],
        "checks": ["random_initialization_fresh_optimizer" if runner.initialization == "scratch" else "exact_checkpoint_weights",
                   "cached_observation_history", "frozen_policy",
                   "sequence_and_version", "episode_length_assignment", "group_timeout_masks", "isolated_footprint_reset",
                   "transported_rollout", "independent_group_GAE", "native_PPO_update", "checkpoint_metadata"],
        "collection_seconds": collection_seconds, "discarded_update_seconds": update_seconds,
        "diagnostic_reward_sums": rewards_by_group, "losses": losses,
        "memory_samples": env.memory_samples,
        "note": "Sampled total GPU memory includes desktop/other processes. One discarded update verifies interfaces, not trained performance.",
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[JOINT CHECK PASS] {json.dumps(summary)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("joint-check", "joint-train"))
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--checkpoint", help="Load network weights; add --resume to restore the optimizer/counter")
    initialization.add_argument("--from-scratch", action="store_true", help="Random network initialization, fresh optimizer, soil level 0")
    parser.add_argument("--stage-report")
    parser.add_argument("--hard-envs", type=int, default=128)
    parser.add_argument("--soft-envs", type=int, default=128)
    parser.add_argument("--soft-substeps", type=int, choices=(16, 64), default=16)
    parser.add_argument("--soft-level", type=int, choices=range(10))
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--run_name", default="joint_seed1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    formal = args.mode == "joint-train"
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    if min(args.hard_envs, args.soft_envs) < 2 or args.max_iterations < 1:
        parser.error("Use >=2 robots per group and positive iterations")
    if args.from_scratch and (args.resume or args.stage_report):
        parser.error("--from-scratch starts a new model; omit --resume and --stage-report")
    if formal and not args.from_scratch and not args.stage_report:
        parser.error("joint-train needs a matching --stage-report from mixed-eval")
    if formal and args.soft_level is not None:
        parser.error("Training level is authorized by the stage report; --soft-level is diagnostic only")
    import torch
    import yaml
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    source = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint is not None else None
    source_state = (source.get("infos") or {}).get("dreamwaq_stage") if source is not None else None
    report = None
    if args.stage_report:
        from run_soft import validate_stage_report
        report = json.loads(Path(args.stage_report).expanduser().read_text())
        validate_stage_report(report, checkpoint)
        if report.get("soft") is None or report["soft"]["physics_dt"] != .02 / args.soft_substeps:
            parser.error("Stage report must include soil at the selected substep count")
    level = (report["decision"]["next_soft_level"] if formal and report is not None else args.soft_level if args.soft_level is not None
             else source_state.get("soft_level_ceiling", 0) if source_state else 0)
    experiment = "go2_dreamwaq_joint" if formal else "go2_dreamwaq_joint_check"
    directory = ROOT / "logs/rsl_rl/isaaclab" / experiment / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + args.run_name)
    (directory / "params").mkdir(parents=True, exist_ok=False)
    settings = dict(vars(args), checkpoint=str(checkpoint) if checkpoint is not None else None, level=level,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint is not None else None,
                    algorithm="unchanged DreamWaQPPO", backend="isaaclab", control_period_s=.02,
                    transport="CPU NumPy/Pipe; synchronous per control step")
    (directory / "params/joint.json").write_text(json.dumps(settings, indent=2) + "\n")
    env = None
    try:
        specs = [dict(stage=name, num_envs=count, substeps=steps, device=args.device, seed=args.seed,
                      level=level, source_state=source_state, formal=formal, report=args.stage_report,
                      from_scratch=args.from_scratch,
                      directory=str(directory), log=str(directory / f"{name}_worker.log"))
                 for name, count, steps in (("hard", args.hard_envs, 4), ("soft", args.soft_envs, args.soft_substeps))]
        print(f"[Joint] {args.mode}; results={directory}", flush=True)
        env = JointEnv(specs, args.device)
        soil_spec = env.ready[1]["soil_spec"]
        if report is not None and report["soft"]["terrain_spec"] != soil_spec:
            raise ValueError("Stage report geometry/material differs from the joint samplers")
        stage = dict(env.ready[1]["stage_state"] or {}, topology=soil_spec, soft_level_ceiling=level)
        if report is not None:
            stage["hard_reference"] = report["hard_reference"]
        stage.setdefault("soft_started", False)
        cfg = deepcopy(env.ready[0]["agent_cfg"])
        cfg.update(seed=args.seed, experiment_name="isaaclab/go2_dreamwaq_joint", run_name=args.run_name,
                   max_iterations=args.max_iterations, resume=args.resume, init_checkpoint=str(checkpoint) if checkpoint is not None else "",
                   stage_report=args.stage_report or "")
        (directory / "params/agent.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        runner = make_runner(env, cfg, source, checkpoint, args.resume, stage, str(directory) if formal else None)
        env.memory_samples.append({"phase": "learner_ready", **gpu_memory()})
        if formal:
            runner.add_git_repo_to_log(__file__)
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        else:
            check(runner, env, directory, stage)
    except BaseException:
        (directory / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
