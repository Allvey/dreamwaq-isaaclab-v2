"""Run the original IsaacLab runner with flat soil or mixed rough/soil environments."""

import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def stage_decision(hard, soil, reference=None, soft_started=False):
    """Performance gates for manual stages; no policy or reward changes."""
    import math
    limits = {"tracking_rmse_xy_max": 0.25, "fall_fraction_max": 0.2, "progress_min": 0.5}
    def passes(metrics):
        return (all(math.isfinite(metrics[key]) for key in ("tracking_rmse_xy", "fall_fraction", "progress"))
                and metrics["tracking_rmse_xy"] <= limits["tracking_rmse_xy_max"]
                and metrics["fall_fraction"] <= limits["fall_fraction_max"]
                and metrics["progress"] >= limits["progress_min"])
    if reference is not None and hard["evaluation_suite"] != reference["evaluation_suite"]:
        raise ValueError("Hard-reference evaluation suite differs; retain seed, environment count and command schedule")
    retained = reference is None or (
        hard["stage_metrics"]["tracking_rmse_xy"] <= max(reference["stage_metrics"]["tracking_rmse_xy"] * 1.2,
                                                        reference["stage_metrics"]["tracking_rmse_xy"] + 0.03)
        and hard["stage_metrics"]["fall_fraction"] <= reference["stage_metrics"]["fall_fraction"] + 0.05
        and hard["stage_metrics"]["progress"] >= reference["stage_metrics"]["progress"] * 0.8)
    hard_ready = passes(hard["stage_metrics"]) and retained
    soil_ready = (soil is not None and len(soil["scenario_metrics"]) == 5
                  and all(passes(metrics) for metrics in soil["scenario_metrics"].values()))
    level = soil["soft_level"] if soil is not None else hard.get("soft_level", 0)
    next_level = level + int(hard_ready and soil_ready) if soft_started else 0
    return {"limits": limits, "hard_ready": hard_ready, "hard_retained": retained,
            "soft_ready": soil_ready, "can_train_soft": hard_ready,
            "next_soft_level": min(hard.get("soft_num_levels", hard.get("soft_num_rows", 10)) - 1, next_level),
            "recommendation": "hard_rehearsal" if not hard_ready else "soft_advance" if soft_started and soil_ready else "soft_hold"}


def validate_stage_report(report, checkpoint):
    import hashlib
    if (report.get("protocol") != "dreamwaq-stages-v1" or not report.get("completed")
            or report["hard"]["evaluation_suite"].get("protocol") != "fixed-stage-v2"
            or report.get("checkpoint_sha256") != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()):
        raise ValueError("Stage evaluation must be completed and match the exact checkpoint being loaded")
    if any(case is not None and (case["checkpoint_sha256"] != report["checkpoint_sha256"]
                                or not case.get("completed") or not case.get("policy_unchanged"))
           for case in (report["hard"], report.get("soft"))):
        raise ValueError("Hard/soft evaluation cases belong to different checkpoints")
    decision = stage_decision(report["hard"], report.get("soft"), report.get("hard_reference"), report["soft_started"])
    if decision != report["decision"] or not decision["can_train_soft"]:
        raise ValueError("Hard performance gate failed; train --stage hard and evaluate again before returning to soil")
    return decision


def stage_evaluate(arguments):
    """Fixed inference evaluation for manual hard/soft stages; never starts training."""
    import argparse
    import hashlib
    import json
    import subprocess
    import torch
    from datetime import datetime
    parser = argparse.ArgumentParser(description=stage_evaluate.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hard-reference", help="Earlier passing stage report; otherwise inherit the checkpoint's reference")
    parser.add_argument("--num_envs", type=int, default=60, help="Hard holdout: at least 60 covers 20 columns at levels 0/2/4")
    parser.add_argument("--soft_envs", type=int, default=10)
    parser.add_argument("--phase_seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--substeps", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--all-stages", action="store_true",
                        help="Evaluate both domains even when the hard gate fails; admission gates remain unchanged")
    args = parser.parse_args(arguments)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file() or args.num_envs < 60 or args.soft_envs < 5 or args.phase_seconds < 3:
        parser.error("Use an existing checkpoint, >=60 hard robots, >=5 soft robots and >=3 seconds per command")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False).get("infos") or {}
    state = state.get("dreamwaq_stage", {})
    reference = state.get("hard_reference")
    if args.hard_reference:
        previous = json.loads(Path(args.hard_reference).read_text())
        if not previous.get("completed") or not previous["decision"]["hard_ready"]:
            parser.error("The hard reference must be a completed passing stage evaluation")
        reference = previous.get("hard_reference") or previous["hard"]
    level = state.get("soft_level_ceiling", 0)
    output = checkpoint.parent / "evaluations" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    cases = {}
    for stage, count in (("hard", args.num_envs), ("soft", args.soft_envs)):
        directory = output / stage
        command = [sys.executable, __file__, "rollout", "--single", "--mixed_stage", stage,
                   "--checkpoint", str(checkpoint), "--num_envs", str(count), "--seed", str(args.seed),
                   "--phase_seconds", str(args.phase_seconds), "--physics_dt", str(0.02 / args.substeps),
                   "--soft_level", str(level), "--device", args.device, "--out", str(directory)]
        if args.headless:
            command.append("--headless")
        with (output / f"{stage}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not (directory / "summary.json").is_file():
            raise RuntimeError(f"Stage evaluation failed; inspect {output / (stage + '.log')}")
        cases[stage] = json.loads((directory / "summary.json").read_text())
        print(f"[Stage eval] {stage}: {cases[stage]['stage_metrics']}", flush=True)
        # A weak hard policy needs hard rehearsal before any soft exposure.
        if stage == "hard" and not args.all_stages and not stage_decision(cases[stage], None, reference)["hard_ready"]:
            break
    report = {"protocol": "dreamwaq-stages-v1", "completed": True,
              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "hard": cases["hard"], "soft": cases.get("soft"),
              "hard_reference": reference or cases["hard"], "soft_started": state.get("soft_started", False)}
    report["decision"] = stage_decision(report["hard"], report["soft"], report["hard_reference"], report["soft_started"])
    (output / "stage_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[STAGE EVALUATION COMPLETE] {report['decision']}; report={output / 'stage_report.json'}", flush=True)


def diagnostic_status(passed):
    # Kit fast shutdown can terminate Python with exit code 0 even after a failed
    # assertion. The lightweight parent checks this marker and the child status.
    status = os.environ.get("DREAMWAQ_DIAGNOSTIC_STATUS")
    if status:
        Path(status).write_text("PASS" if passed else "FAIL")


def register_tasks(*, overview=False, stage=None, substeps=None, isolated_training=False,
                   stage_report=None, soft_level=None):
    # String entry points keep IsaacLab imports after AppLauncher.
    import gymnasium as gym
    for suffix, config in (("", "Go2SoilCurriculumEnvCfg"), ("-Play", "Go2SoilCurriculumPlayEnvCfg")):
        gym.register(
            id=f"Isaac-Velocity-Go2-DreamWaQ-SoilCurriculum{suffix}-v0",
            entry_point="soft_terrain:make_env", disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"soft_terrain:{config}",
                    "rsl_rl_cfg_entry_point": "soft_terrain:Go2SoilCurriculumRunnerCfg"},
        )
    for suffix, config in (("", "Go2SoftEnvCfg"), ("-Play", "Go2SoftPlayEnvCfg")):
        gym.register(
            id=f"Isaac-Velocity-Flat-Go2-DreamWaQ-Soft{suffix}-v0",
            entry_point="soft_terrain:make_env", disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"soft_terrain:{config}",
                    "rsl_rl_cfg_entry_point": "soft_terrain:Go2SoftRunnerCfg"},
        )
    for suffix, config in (("", "Go2MixedEnvCfg"), ("-Play", "Go2MixedPlayEnvCfg")):
        gym.register(
            id=f"Isaac-Velocity-Rough-Go2-DreamWaQ-Mixed{suffix}-v0",
            entry_point="soft_terrain:make_env", disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"soft_terrain:{config}",
                    "rsl_rl_cfg_entry_point": "soft_terrain:Go2MixedRunnerCfg",
                    "overview": overview, "stage": stage, "physics_substeps": substeps,
                    "isolated_training": isolated_training, "stage_report": stage_report, "soft_level": soft_level},
        )


def check(arguments):
    """Short physics/contact/reset check; never performs a learning update."""
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=check.__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rigid", action="store_true")
    parser.add_argument("--stock", action="store_true", help="Original task with matched flat geometry and timing")
    parser.add_argument("--hard_training", action="store_true", help="Check original rough terrain or its isolated hard-training path")
    parser.add_argument("--record", help="Optional diagnostic NPZ path")
    parser.add_argument("--compare", help="Require bitwise equality with a recorded rigid baseline")
    parser.add_argument("--checkpoint", help="Optional original DreamWaQ policy for inference only")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(arguments)
    if args.num_envs < 2 or args.steps < 20:
        parser.error("Use at least 2 environments and 20 steps")
    app = AppLauncher(args).app
    env = None
    try:
        import numpy as np
        import torch
        import gymnasium as gym
        import soft_terrain as soft
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        cfg = (soft.Go2DreamWaQRoughEnvCfg() if args.stock else
               soft.Go2MixedEnvCfg() if args.hard_training else soft.Go2SoftEnvCfg())
        if args.hard_training and not args.stock:
            cfg.scene.terrain.training_stage = "hard"
            cfg.scene.terrain.isolated_training = True
            soft.enable_soil(cfg)
        elif args.stock and not args.hard_training:
            soft.configure_flat(cfg)
            soft.configure_soil_timing(cfg)
        elif not args.stock:
            cfg.scene.terrain.soft_enabled = not args.rigid
            soft.enable_soil(cfg)
        cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
        cfg.sim.device = args.device
        task = ("Isaac-Velocity-Rough-Go2-DreamWaQ-v0" if args.stock else
                "Isaac-Velocity-Rough-Go2-DreamWaQ-Mixed-v0" if args.hard_training else
                "Isaac-Velocity-Flat-Go2-DreamWaQ-Soft-v0")
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
        raw = env.unwrapped
        robot, sensor = raw.scene["robot"], raw.scene["contact_forces"]
        soil = getattr(raw.action_manager.get_term("joint_pos"), "soil", None)
        if args.hard_training:
            from isaaclab.envs import ManagerBasedRLEnv
            from isaaclab.terrains import TerrainImporter
            import omni.usd
            assert type(raw) is ManagerBasedRLEnv
            assert type(raw.scene.terrain) is TerrainImporter
            assert set(raw.scene.terrain.meshes) == {"terrain"}
            assert soil is None and cfg.scene.height_scanner.mesh_prim_paths == ["/World/ground"]
            assert "stage_boundary" not in raw.termination_manager.active_terms
            assert not omni.usd.get_context().get_stage().GetPrimAtPath("/World/soil_surface").IsValid()
            print("[Hard training check] native environment/terrain/scanner/curriculum; no soil or stage boundary", flush=True)
        agent = soft.Go2SoftRunnerCfg().to_dict()
        baseline_agent = soft.Go2DreamWaQRoughPPORunnerCfg().to_dict()
        agent.pop("experiment_name")
        agent.pop("init_checkpoint")
        baseline_agent.pop("experiment_name")
        assert agent == baseline_agent, "Original policy/PPO/runner settings must remain identical"
        if soil:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/soil_surface")):
                assert not prim.HasAPI(UsdPhysics.CollisionAPI), "Nominal surface must not collide"
            for prim in Usd.PrimRange(stage.GetPrimAtPath(cfg.scene.terrain.prim_path)):
                if prim.IsA(UsdGeom.Plane):
                    z = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2]
                    assert abs(z + cfg.scene.terrain.depth) < 1e-6, ("Hard floor height", z)
        policy = None
        if args.checkpoint:
            from omegaconf import OmegaConf
            runner_cfg = soft.Go2SoftRunnerCfg()
            resolved = OmegaConf.create(runner_cfg.to_dict())
            resolved.init_checkpoint = args.checkpoint
            # Exercise Hydra's conversion path, not just a directly edited dict.
            runner_cfg.from_dict(OmegaConf.to_container(resolved, resolve=True))
            assert runner_cfg.init_checkpoint == args.checkpoint
            runner = soft.make_runner(env, runner_cfg.to_dict(), log_dir=None, device=env.device)
            assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
            source = torch.load(args.checkpoint, weights_only=False, map_location=env.device)
            policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
            for name, value in policy_nn.state_dict().items():
                torch.testing.assert_close(value, source["model_state_dict"][name], rtol=0, atol=0)
            print("[Soil check] fine-tune weights match checkpoint exactly; optimizer and iteration state are fresh", flush=True)
            policy = runner.get_inference_policy(device=env.device)
            if args.hard_training and not args.stock:
                import tempfile
                with tempfile.TemporaryDirectory(prefix="dreamwaq_hard_metadata_") as directory:
                    path = str(Path(directory) / "model.pt")
                    runner.save(path)
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                    expected_stage = soft.soil_stage_state((source.get("infos") or {}).get("dreamwaq_stage"), raw.scene.terrain.cfg)
                    if expected_stage is not None:
                        assert saved["infos"]["dreamwaq_stage"] == expected_stage
                    runner.load(path, load_optimizer=False)
                    assert runner.stage_state == saved["infos"]["dreamwaq_stage"]
                print("[Hard training check] native terrain preserves checkpoint curriculum metadata", flush=True)
        records = {key: [] for key in ("root", "joint", "policy", "history", "scan", "reward", "contact", "air")}
        obs = env.get_observations()
        contact_steps, max_force = 0, 0.0
        for step in range(args.steps):
            with torch.inference_mode():
                actions = policy(obs) if policy else torch.zeros((env.num_envs, 12), device=env.device)
                if policy is None:
                    actions[:, step % 12] = 0.08
                obs, reward, done, extra = env.step(actions)
            groups = obs
            for key, dim in (("policy", 45), ("prop_history", 225), ("height_scan", 187), ("privileged_vel", 3)):
                assert groups[key].shape == (args.num_envs, dim), (key, groups[key].shape)
                assert torch.isfinite(groups[key]).all(), key
            assert torch.isfinite(reward).all() and torch.isfinite(robot.data.root_state_w).all()
            values = (robot.data.root_state_w, robot.data.joint_pos, groups["policy"],
                      groups["prop_history"], groups["height_scan"], reward,
                      sensor.data.net_forces_w, sensor.data.current_air_time)
            if args.record or args.compare:
                for key, value in zip(records, values):
                    records[key].append(value.detach().cpu().numpy().copy())
            if soil:
                assert soil.steps == (step + 1) * cfg.decimation
                assert torch.isfinite(soil.wrench).all()
                contact_steps += int((soil.normal_w[:, soil.feet, 2] > 1).any())
                max_force = max(max_force, float(soil.wrench[..., :3].norm(dim=-1).max()))
                view = sensor.contact_physx_view
                physical = view.physical.get_net_contact_forces(dt=cfg.sim.dt)
                expected = physical + soil.normal_w[:, sensor.robot_body_ids].reshape_as(physical)
                torch.testing.assert_close(view.get_net_contact_forces(dt=cfg.sim.dt), expected)
                active = done == 0
                torch.testing.assert_close(
                    sensor.data.net_forces_w[active], expected.view(args.num_envs, -1, 3)[active],
                )
                foot_sensor_ids = [sensor.body_names.index(robot.body_names[i]) for i in soil.feet]
                in_contact = (soil.normal_w[:, soil.feet, 2] > sensor.cfg.force_threshold) & active[:, None]
                assert (sensor.data.current_air_time[:, foot_sensor_ids][in_contact] == 0).all()
            if (step + 1) % 25 == 0:
                print(f"[Soil check] {step + 1}/{args.steps}, virtual_contact_steps={contact_steps}", flush=True)
        if args.record:
            np.savez_compressed(args.record, **{key: np.stack(values) for key, values in records.items()})
        if args.compare:
            with np.load(args.compare) as reference:
                for key, values in records.items():
                    np.testing.assert_array_equal(np.stack(values), reference[key], err_msg=key)
            print(f"[Soil check] rigid control matches original {'rough' if args.hard_training else 'flat'} task bit for bit", flush=True)
        if soil:
            assert contact_steps > 0, "No virtual foot contact detected"
            print(f"[Soil check] final mean foot depth={float(soil.depth.mean()):.4f} m", flush=True)
            soil.hole_d[:] = 0.03
            untouched = {name: getattr(soil, name)[1:].clone() for name in ("hole_d", "mem", "normal_w")}
            with torch.inference_mode():
                raw._reset_idx(torch.tensor([0], device=env.device))
            assert (soil.mem[0] == -1).all() and (soil.hole_d[0] == 0).all()
            assert (soil.normal_w[0] == 0).all() and (soil.wrench[0] == 0).all()
            for name, value in untouched.items():
                torch.testing.assert_close(getattr(soil, name)[1:], value, rtol=0, atol=0)
            assert (sensor._data.current_air_time[0] == 0).all()
            assert (sensor._data.net_forces_w_history[0] == 0).all()
        print(f"[SOFT CHECK PASS] envs={args.num_envs} steps={args.steps} "
              f"soft={soil is not None} contact_steps={contact_steps} max_body_force={max_force:.2f} N", flush=True)
        diagnostic_status(True)
    except Exception:
        diagnostic_status(False)
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def mixed_profile(arguments):
    """Inference-only throughput and CUDA-synchronized component timing."""
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=mixed_profile.__doc__)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--profile_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--soft_fraction", type=float, default=0.5)
    parser.add_argument("--soft_columns", type=int, default=5)
    parser.add_argument("--stage", choices=("mixed", "hard", "soft"), default="mixed")
    parser.add_argument("--substeps", type=int, choices=(4, 16, 32, 64), default=32)
    parser.add_argument("--checkpoint")
    parser.add_argument("--soft_level", type=int, choices=range(10), help="Fixed logical level; otherwise inherit the checkpoint")
    parser.add_argument("--reference", action="store_true", help="Use the original adapter implementation")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(arguments)
    if min(args.num_envs, args.steps, args.warmup, args.profile_steps) < 1:
        parser.error("Environment and step counts must be positive")
    app, env = AppLauncher(args).app, None
    try:
        from datetime import datetime
        import hashlib
        import json
        import time
        import torch
        import gymnasium as gym
        import soft_terrain as soft
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml

        cfg = soft.Go2MixedEnvCfg()
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
        cfg.scene.terrain.soft_fraction = args.soft_fraction
        cfg.scene.terrain.training_stage = args.stage
        cfg.scene.terrain.isolated_training = args.stage != "mixed"
        if args.soft_level is not None:
            cfg.scene.terrain.soft_level_ceiling = args.soft_level
        elif args.checkpoint:
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            cfg.scene.terrain.soft_level_ceiling = ((saved.get("infos") or {}).get("dreamwaq_stage") or {}).get("soft_level_ceiling", 0)
        cfg.scene.terrain.soft_columns = args.soft_columns
        cfg.scene.terrain.physics_substeps = args.substeps
        soft.configure_soil_timing(cfg, substeps=args.substeps)
        cfg.scene.terrain.optimized = not args.reference
        cfg.scene.terrain.terrain_generator.seed = args.seed
        env = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Rough-Go2-DreamWaQ-Mixed-v0", cfg=cfg))
        raw, policy = env.unwrapped, None
        agent = soft.Go2MixedRunnerCfg()
        if args.checkpoint:
            agent.init_checkpoint = args.checkpoint
            runner = soft.make_runner(env, agent.to_dict(), log_dir=None, device=env.device)
            policy = runner.get_inference_policy(device=env.device)
        obs = env.get_observations()
        zero = torch.zeros((env.num_envs, 12), device=env.device)
        def synchronize():
            if str(env.device).startswith("cuda"):
                torch.cuda.synchronize(env.device)
        def advance():
            nonlocal obs
            with torch.inference_mode():
                actions = policy(obs) if policy else zero
                obs, reward, done, extras = env.step(actions)
            return reward, done
        for _ in range(args.warmup):
            advance()
        synchronize()
        started = time.perf_counter()
        for _ in range(args.steps):
            advance()
        synchronize()
        elapsed = time.perf_counter() - started
        print(f"[Mixed profile] clean throughput={env.num_envs * args.steps / elapsed:.1f} steps/s; "
              f"{args.steps} control steps in {elapsed:.3f}s", flush=True)
        # Component timings intentionally serialize CUDA. Keep these separate
        # from the clean throughput above; PhysX may use non-Torch CUDA streams.
        totals, originals = {}, []
        def instrument(obj, name, label):
            native = getattr(obj, name)
            totals[label] = {"calls": 0, "seconds": 0.0}
            def measured(*positional, **keywords):
                synchronize()
                before = time.perf_counter()
                value = native(*positional, **keywords)
                synchronize()
                totals[label]["seconds"] += time.perf_counter() - before
                totals[label]["calls"] += 1
                return value
            originals.append((obj, name, native))
            setattr(obj, name, measured)
        soil = getattr(raw.action_manager.get_term("joint_pos"), "soil", None)
        if soil is not None:
            instrument(soil, "prepare_state", "soil_state_read_and_pack")
            instrument(soil, "evaluate", "soil_force_and_contact_calculation")
            instrument(soil, "apply_wrench", "soil_wrench_submission")
        instrument(raw.scene["contact_forces"], "_update_buffers_impl", "contact_sensor_history_and_timers")
        instrument(raw.scene, "write_data_to_sim", "actuator_and_scene_submission")
        instrument(raw.sim, "step", "physx_step")
        try:
            synchronize()
            started = time.perf_counter()
            for _ in range(args.profile_steps):
                advance()
            synchronize()
            profile_elapsed = time.perf_counter() - started
        finally:
            for obj, name, native in originals:
                setattr(obj, name, native)
        for value in totals.values():
            value["ms_per_call"] = 1000 * value["seconds"] / max(value["calls"], 1)
            value["serialized_wall_fraction"] = value["seconds"] / profile_elapsed
        output = ROOT / "logs/rsl_rl/isaaclab/go2_dreamwaq_mixed_profile" / (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + ("_reference" if args.reference else "_optimized")
        )
        output.mkdir(parents=True, exist_ok=False)
        dump_yaml(str(output / "env.yaml"), cfg)
        dump_yaml(str(output / "agent.yaml"), agent)
        summary = {"completed": True, "inference_only": True, "seed": args.seed,
                   "backend": "isaaclab", "task": "Isaac-Velocity-Rough-Go2-DreamWaQ-Mixed-v0",
                   "optimized": not args.reference, "num_envs": env.num_envs,
                   "training_stage": cfg.scene.terrain.training_stage,
                   "isolated_training": cfg.scene.terrain.isolated_training,
                   "soft_level": cfg.scene.terrain.soft_level_ceiling,
                   "soft_tiles": cfg.scene.terrain.resolved_soft_tiles,
                   "torch_version": torch.__version__,
                   "device_name": torch.cuda.get_device_name(env.device) if str(env.device).startswith("cuda") else "CPU",
                   "warmup_control_steps": args.warmup, "measured_control_steps": args.steps,
                   "physics_dt": cfg.sim.dt, "decimation": cfg.decimation, "control_period_s": raw.step_dt,
                   "rough_mesh_sha256": cfg.scene.terrain.rough_mesh_sha256,
                   "soft_surface_sha256": cfg.scene.terrain.soft_surface_sha256,
                   "soft_scenarios": cfg.scene.terrain.resolved_soft_scenarios,
                   "adapter_sha256": cfg.scene.terrain.adapter_sha256,
                   "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
                   "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
                   "clean_wall_seconds": elapsed, "clean_steps_per_second": env.num_envs * args.steps / elapsed,
                   "serialized_profile_control_steps": args.profile_steps,
                   "serialized_profile_wall_seconds": profile_elapsed, "components": totals,
                   "timing_note": "Clean timing synchronizes only at block boundaries; component timing synchronizes "
                                  "all CUDA work around each call and changes scheduling. Includes policy inference when supplied; "
                                  "no learning, checkpoint save, or per-step CPU diagnostic copies."}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for label, value in sorted(totals.items(), key=lambda pair: -pair[1]["seconds"]):
            print(f"[Mixed profile] {label}: {value['ms_per_call']:.3f} ms/call; "
                  f"{100 * value['serialized_wall_fraction']:.1f}% serialized wall", flush=True)
        print(f"[MIXED PROFILE PASS] {output}", flush=True)
        diagnostic_status(True)
    except Exception:
        diagnostic_status(False)
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def stage_check(arguments):
    """Validate stage confinement, timing and checkpoint/optimizer continuity; no formal training."""
    import argparse
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=stage_check.__doc__)
    parser.add_argument("--stage", choices=("hard", "soft"), required=True)
    parser.add_argument("--substeps", type=int, choices=(4, 16, 32, 64), default=32)
    parser.add_argument("--num_envs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--checkpoint")
    parser.add_argument("--stage-report", help="Also verify real soft admission for this checkpoint")
    parser.add_argument("--isolated_training", action="store_true", help="Check the soil-only training scene")
    parser.add_argument("--soft_level", type=int, choices=range(10), default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(arguments)
    if args.num_envs < 5 or args.steps < 20:
        parser.error("Use at least five environments and 20 steps")
    if args.isolated_training and args.stage != "soft":
        parser.error("Use soft-check --hard_training for isolated hard parity")
    app, env = AppLauncher(args).app, None
    try:
        import torch
        import gymnasium as gym
        import soft_terrain as soft
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        cfg = soft.Go2MixedEnvCfg()
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, 1, args.device
        cfg.scene.terrain.terrain_generator.seed = 1
        cfg.scene.terrain.training_stage = args.stage
        cfg.scene.terrain.physics_substeps = args.substeps
        cfg.scene.terrain.isolated_training = args.isolated_training
        cfg.scene.terrain.soft_level_ceiling = args.soft_level
        admission = ({"isolated_training": True, "stage_report": args.stage_report}
                     if args.isolated_training and args.stage_report else {})
        env = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Rough-Go2-DreamWaQ-Mixed-v0", cfg=cfg, **admission))
        raw, terrain = env.unwrapped, env.unwrapped.scene.terrain
        hard = args.stage == "hard"
        soil = getattr(raw.action_manager.get_term("joint_pos"), "soil", None)
        assert (soil is None) == hard
        assert (terrain.num_rough, terrain.num_soft) == ((args.num_envs, 0) if hard else (0, args.num_envs))
        assert cfg.decimation == (4 if hard else args.substeps)
        assert abs(raw.step_dt - 0.02) < 1e-12
        assert abs(cfg.scene.contact_forces.history_length * cfg.sim.dt - 0.015) < 1e-12
        assert cfg.terminations.stage_boundary.time_out
        if args.isolated_training:
            assert set(terrain.meshes) == {"soil_bottom", "soil_rim"}
            assert terrain.terrain_origins.shape == (1, 5, 3)
            assert terrain.soft_bounds[0, 1] == -20 and terrain.soft_bounds[1, 1] == 20
        else:
            assert {"terrain", "soil_bottom", "soil_rim"} <= set(terrain.meshes)
        assert cfg.scene.terrain.resolved_soft_tiles == 5
        assert len(terrain.soft_bearing_scales) == 1
        assert terrain.surface_heights.shape == (81, 401)
        if not hard:
            assert (terrain.terrain_levels == args.soft_level).all()
            assert terrain.soft_level == args.soft_level
        assert terrain.soft_scenario_names == ["flat", "mound", "basin", "undulating", "mixed"]
        assert not soft.outside_training_region(raw).any()
        obs = env.get_observations()
        for _ in range(args.steps):
            with torch.inference_mode():
                obs, reward, done, extras = env.step(torch.zeros((args.num_envs, 12), device=env.device))
            for key, width in (("policy", 45), ("prop_history", 225), ("height_scan", 187), ("privileged_vel", 3)):
                assert obs[key].shape == (args.num_envs, width) and torch.isfinite(obs[key]).all()
            assert torch.isfinite(reward).all()
            assert not soft.outside_training_region(raw).any()
            for value in extras["log"].values():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all()
        if soil is not None:
            assert soil.steps == args.steps * cfg.decimation
        # Force one robot across each outer/domain boundary. Real autoreset must
        # use timeout semantics and return it to the requested spawn domain.
        native = raw.termination_manager.compute
        captured = {}
        def record_boundary():
            result = native()
            captured["boundary"] = raw.termination_manager.get_term("stage_boundary").clone()
            captured["timeout"] = raw.termination_manager.time_outs.clone()
            return result
        raw.termination_manager.compute = record_boundary
        for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
            state = raw.scene["robot"].data.root_state_w[:1].clone()
            generator = cfg.scene.terrain.terrain_generator
            bound = ((generator.num_rows if axis == 0 else generator.num_cols) * generator.size[axis] / 2 * sign if hard else
                     float(terrain.soft_bounds[0 if sign < 0 else 1, axis]))
            state[0, axis] = bound + sign * 0.2
            state[0, 2] = 1.0
            state[:, 7:] = 0
            with torch.inference_mode():
                raw.scene["robot"].write_root_state_to_sim(state, env_ids=torch.tensor([0], device=env.device))
                obs, _, done, _ = env.step(torch.zeros((args.num_envs, 12), device=env.device))
            assert done[0] and captured["boundary"][0] and captured["timeout"][0]
            assert not soft.outside_training_region(raw).any()
        raw.termination_manager.compute = native
        if not hard:
            torch.testing.assert_close(terrain.env_origins, terrain.soft_tile_origins[terrain.soft_tile_ids])
            torch.testing.assert_close(terrain.soft_tile_ids % cfg.scene.terrain.soft_columns,
                                       torch.arange(args.num_envs, device=env.device) % cfg.scene.terrain.soft_columns)
        agent = soft.Go2MixedRunnerCfg().to_dict()
        original_agent = soft.Go2DreamWaQRoughPPORunnerCfg().to_dict()
        for key in ("policy", "algorithm", "num_steps_per_env"):
            assert agent[key] == original_agent[key]
        runner = soft.make_runner(env, agent, log_dir=None, device=env.device)
        if args.checkpoint:
            source = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
            runner.load(args.checkpoint)
            assert runner.current_learning_iteration == source["iter"]
            assert runner.alg.optimizer.state
            for key, value in runner.alg.policy.state_dict().items():
                torch.testing.assert_close(value, source["model_state_dict"][key], rtol=0, atol=0)
            loaded = runner.alg.optimizer.state_dict()
            assert loaded["param_groups"] == source["optimizer_state_dict"]["param_groups"]
            for key, values in loaded["state"].items():
                for name, value in values.items():
                    expected = source["optimizer_state_dict"]["state"][key][name]
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, expected.to(value.device), rtol=0, atol=0)
                    else:
                        assert value == expected
        # Stage checkpoint metadata must survive stage changes, preserve PPO
        # state, with one mesh-backed global level for all soil robots.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="dreamwaq_stage_state_") as directory:
            checkpoint_path = str(Path(directory) / "state.pt")
            if not hard:
                expected_levels = terrain.terrain_levels.clone()
            runner.save(checkpoint_path)
            stored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = stored["infos"]["dreamwaq_stage"]
            assert state["soft_started"] == (not hard)
            if not hard:
                assert state["soft_level_ceiling"] == args.soft_level
                assert "levels_by_column" not in state
                terrain.terrain_levels.zero_()
            runner.load(checkpoint_path)
            assert runner.stage_state == state
            if not hard:
                assert (terrain.terrain_levels == args.soft_level).all()
                torch.testing.assert_close(terrain.terrain_levels, expected_levels)
                torch.testing.assert_close(terrain.env_origins, terrain.soft_tile_origins[terrain.soft_tile_ids])
                # Saving while there are no active soft samples retains the
                # dormant curriculum, as in a hard rehearsal process.
                count = terrain.num_soft
                terrain.num_soft = 0
                runner.save(checkpoint_path)
                terrain.num_soft = count
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                assert saved["infos"]["dreamwaq_stage"] == state
                # A different saved level must not relabel an existing mesh.
                wrong = dict(stored)
                wrong["infos"] = {"dreamwaq_stage": dict(state, soft_level_ceiling=(args.soft_level + 1) % 10)}
                wrong_path = str(Path(directory) / "wrong_level.pt")
                torch.save(wrong, wrong_path)
                try:
                    runner.load(wrong_path)
                except ValueError as error:
                    assert "already-built mesh" in str(error)
                else:
                    raise AssertionError("Checkpoint changed a live terrain's geometry level")
            for key, value in stored["model_state_dict"].items():
                torch.testing.assert_close(runner.alg.policy.state_dict()[key].cpu(), value, rtol=0, atol=0)
            if args.stage_report and not hard:
                assert args.checkpoint
                admitted = soft.make_runner(env, dict(soft.Go2MixedRunnerCfg().to_dict(), init_checkpoint=args.checkpoint, stage_report=args.stage_report),
                                            log_dir=directory, device=env.device)
                assert admitted.stage_state["hard_reference"]
                assert terrain.cfg.soft_level_ceiling == args.soft_level
                assert (terrain.terrain_levels == args.soft_level).all()
                assert admitted.current_learning_iteration == 0
                assert not admitted.alg.optimizer.state
                del admitted
        print("[Stage check] checkpoint carries curriculum through hard rehearsal; model/optimizer continuity retained", flush=True)
        with torch.inference_mode():
            obs = env.get_observations()
            for _ in range(runner.num_steps_per_env):
                obs, rewards, dones, extras = env.step(runner.alg.act(obs))
                runner.alg.process_env_step(obs, rewards, dones, extras)
            runner.alg.compute_returns(obs)
        losses = runner.alg.update()
        assert all(torch.isfinite(torch.as_tensor(value)) for value in losses.values())
        print(f"[STAGE CHECK PASS] stage={args.stage} dt={cfg.sim.dt} substeps={cfg.decimation}; "
              f"isolated_training={args.isolated_training}; four boundary resets, checkpoint/optimizer and discarded PPO update verified", flush=True)
        diagnostic_status(True)
    except Exception:
        diagnostic_status(False)
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def mixed_check(arguments):
    """Mixed geometry/force/reset and disposable PPO-update acceptance; no saved policy."""
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=mixed_check.__doc__)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--soft_fraction", type=float, default=0.5)
    parser.add_argument("--soft_columns", type=int, default=5)
    parser.add_argument("--play_scene", action="store_true", help="Check the same reduced terrain/config used by mixed-play")
    parser.add_argument("--substeps", type=int, choices=(4, 16, 32, 64), default=32)
    parser.add_argument("--checkpoint", help="Optional weights to verify exact warm start")
    parser.add_argument("--no_adapter", action="store_true", help="Diagnostic rigid-group reference only")
    parser.add_argument("--fixed_soil", action="store_true", help="Disable grades/progression for the previous fixed-scene control")
    parser.add_argument("--soft_level", type=int, choices=range(10), default=0)
    parser.add_argument("--reference", action="store_true", help="Use the original adapter for an exact paired comparison")
    parser.add_argument("--record", help="Optional two-group state/contact archive for paired acceptance")
    parser.add_argument("--compare", help="Require exact equality to a reference archive (legacy rough-only archives supported)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(arguments)
    if args.num_envs < 4 or args.steps < 50:
        parser.error("Use at least 4 environments and 50 steps")
    app, env = AppLauncher(args).app, None
    try:
        import hashlib
        import json
        import numpy as np
        import torch
        import gymnasium as gym
        from omegaconf import OmegaConf
        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd
        import soft_terrain as soft
        from isaaclab.envs.mdp.actions import JointPositionAction
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.warp import raycast_mesh

        cfg = soft.Go2MixedPlayEnvCfg() if args.play_scene else soft.Go2MixedEnvCfg()
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
        cfg.scene.terrain.soft_fraction = args.soft_fraction
        cfg.scene.terrain.soft_columns = args.soft_columns
        cfg.scene.terrain.physics_substeps = args.substeps
        cfg.scene.terrain.soft_curriculum = not args.fixed_soil
        cfg.scene.terrain.soft_level_ceiling = args.soft_level
        soft.configure_soil_timing(cfg, substeps=args.substeps)
        cfg.scene.terrain.optimized = not args.reference
        cfg.scene.terrain.terrain_generator.seed = args.seed
        baseline = soft.Go2DreamWaQRoughEnvCfg_PLAY() if args.play_scene else soft.Go2DreamWaQRoughEnvCfg()
        for name in ("observations", "rewards", "terminations", "events", "commands"):
            assert getattr(cfg, name).to_dict() == getattr(baseline, name).to_dict(), name
        assert cfg.actions.to_dict() == baseline.actions.to_dict()
        assert cfg.scene.robot.to_dict() == baseline.scene.robot.to_dict()
        assert cfg.scene.terrain.terrain_generator.to_dict() == dict(
            baseline.scene.terrain.terrain_generator.to_dict(), seed=args.seed,
        )
        assert abs(cfg.sim.dt * cfg.decimation - 0.02) < 1e-12
        assert abs(cfg.scene.contact_forces.history_length * cfg.sim.dt - 0.015) < 1e-12
        env = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Rough-Go2-DreamWaQ-Mixed-v0", cfg=cfg))
        raw = env.unwrapped
        terrain, robot, sensor = raw.scene.terrain, raw.scene["robot"], raw.scene["contact_forces"]
        soil = raw.action_manager.get_term("joint_pos").soil
        if cfg.scene.terrain.optimized:
            soil.prepare_state()
            prepared = [value.clone() for value in (soil.q, soil.qd, soil.world_com)]
            cfg.scene.terrain.optimized = False
            soil.prepare_state()
            for actual, expected in zip((soil.q, soil.qd, soil.world_com), prepared):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            cfg.scene.terrain.optimized = True
            print("[Mixed check] optimized state packing and world COM match original exactly", flush=True)
        rigid, sand = ~terrain.soft_mask, terrain.soft_mask
        assert rigid.any() and sand.any()
        assert terrain.num_soft == round(args.num_envs * args.soft_fraction)
        coverage = torch.stack(list(terrain.scenario_masks.values())).int().sum(0, dtype=torch.int32)
        torch.testing.assert_close(coverage, sand.int())
        if terrain.num_soft >= args.soft_columns:
            assert set(terrain.scenario_masks) == set(terrain.soft_scenario_names)
        original_origins = terrain.env_origins.clone()
        mesh = terrain.meshes["terrain"]
        assert hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest() == cfg.scene.terrain.rough_mesh_sha256
        # Regenerate the stock mesh with the same seed, preserving global RNGs.
        from isaaclab.terrains import TerrainGenerator
        np_state, torch_state = np.random.get_state(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        np.random.set_state(terrain._rough_numpy_state)
        torch.set_rng_state(terrain._rough_torch_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(terrain._rough_cuda_state)
        reference = TerrainGenerator(cfg=baseline.scene.terrain.terrain_generator.replace(seed=args.seed), device=env.device)
        source_mesh = reference.terrain_mesh
        assert hashlib.sha256(source_mesh.vertices.tobytes() + source_mesh.faces.tobytes()).hexdigest() == cfg.scene.terrain.source_rough_mesh_sha256
        gen = cfg.scene.terrain.terrain_generator
        half = np.array(gen.size) * [gen.num_rows, gen.num_cols] / 2
        active = (np.abs(source_mesh.triangles[:, :, :2]) <= half + 1e-6).all(axis=(1, 2))
        np.testing.assert_array_equal(source_mesh.triangles[active], mesh.triangles)
        assert np.allclose(source_mesh.triangles[~active, :, 2], 0.0, atol=1e-6)
        print("[Mixed check] original active-cell triangles unchanged; only outer border relocated", flush=True)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)
        del reference
        assert (terrain.terrain_types[rigid] == torch.div(
            torch.arange(terrain.num_rough, device=env.device),
            terrain.num_rough / terrain.terrain_origins.shape[1], rounding_mode="floor").long()).all()
        stage = omni.usd.get_context().get_stage()
        for path in ("/World/soil_surface", "/World/mixed_height_scan"):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
                assert not prim.HasAPI(UsdPhysics.CollisionAPI), path
        # All five graded profiles share the original cell lattice and hard frames.
        generator = cfg.scene.terrain.terrain_generator
        sx, sy = generator.size
        border = cfg.scene.terrain.soft_border
        assert float(terrain.soft_bounds[0, 1]) == generator.num_cols * sy / 2
        assert cfg.scene.terrain.resolved_soft_tiles == args.soft_columns <= 5
        tiles = list(stage.GetPrimAtPath("/World/soil_surface").GetChildren())
        assert len(tiles) == cfg.scene.terrain.resolved_soft_tiles
        torch.testing.assert_close(terrain.env_origins[sand], terrain.soft_tile_origins[terrain.soft_tile_ids])
        expected_x = torch.zeros(1, device=env.device)
        torch.testing.assert_close(terrain.soft_tile_origins[:, 0].reshape(-1, args.soft_columns)[:, 0], expected_x)
        expected_y = (generator.num_cols / 2 + torch.arange(args.soft_columns, device=env.device) + 0.5) * sy
        torch.testing.assert_close(terrain.soft_tile_origins[:, 1].reshape(-1, args.soft_columns)[0], expected_y)
        if terrain.soft_curriculum_active:
            assert (terrain.terrain_levels[sand] == args.soft_level).all()
        parameters = [soft.soil_level_parameters(cfg.scene.terrain, row, cfg.scene.terrain.soft_num_levels)
                      for row in range(cfg.scene.terrain.soft_num_levels)]
        if cfg.scene.terrain.soft_curriculum:
            assert parameters[0][0] == cfg.scene.terrain.soft_min_geometry_scale
            assert parameters[0][1] == cfg.scene.terrain.soft_easy_bearing_scale
            assert all(a[0] <= b[0] and a[1] >= b[1] and a[2] >= b[2]
                       for a, b in zip(parameters, parameters[1:]))
        assert parameters[-1] == (1.0, 1.0, cfg.scene.terrain.soft_stripe_width)
        for scenario in terrain.soft_scenario_names:
            target = soft.soil_cell_grid(cfg.scene.terrain, scenario)
            highest = soft.soil_cell_grid(cfg.scene.terrain, scenario, cfg.scene.terrain.soft_num_levels - 1)
            for expected, actual in zip(target, highest):
                np.testing.assert_array_equal(actual, expected)
        scan_mesh = raw.scene["height_scanner"].meshes["/World/mixed_height_scan"]

        def downward(points, mesh, normals=False):
            query = points.clone()
            query[..., 2] = 5
            direction = torch.zeros_like(query)
            direction[..., 2] = -1
            return raycast_mesh(query, direction, mesh, return_normal=normals)

        # Compare the physical sampler with rendered/critic triangles, including
        # off-grid points on slopes; compare the actual collision bottom as well.
        offsets = torch.tensor([[0, 0], [0.637, 0.173], [-0.843, 0.327], [1.217, -0.283]], device=env.device)
        probes = terrain.soft_tile_origins[:, None].repeat(1, len(offsets), 1)
        probes[..., :2] += offsets
        probes = probes.reshape(-1, 3)
        height, normal, active = terrain.sample_surface(probes)
        hits, _, face_normal, _ = downward(probes, scan_mesh, True)
        torch.testing.assert_close(hits[:, 2], height, rtol=0, atol=2e-5)
        torch.testing.assert_close(face_normal, normal, rtol=0, atol=2e-4)
        bottom = downward(probes, terrain.warp_meshes["soil_bottom"])[0]
        torch.testing.assert_close(bottom[active.bool(), 2], height[active.bool()] - cfg.scene.terrain.depth, rtol=0, atol=2e-5)
        assert not torch.isfinite(bottom[~active.bool()]).all(-1).any()
        for key in ("terrain", "soil_rim"):
            hit = downward(probes, terrain.warp_meshes[key])[0]
            assert not torch.isfinite(hit[active.bool()]).all(-1).any(), f"Rigid {key} covers sand"

        offsets = torch.tensor([[sx / 2 - border / 2, 0], [-sx / 2 + border / 2, 0],
                                [0, sy / 2 - border / 2], [0, -sy / 2 + border / 2],
                                [sx / 2, 0], [0, sy / 2]], device=env.device)
        frames = terrain.soft_tile_origins[:, None].repeat(1, len(offsets), 1)
        frames[..., :2] += offsets
        frames = frames.reshape(-1, 3)
        fh, fn, fa = terrain.sample_surface(frames)
        assert fa.count_nonzero() == 0
        torch.testing.assert_close(fh, torch.zeros_like(fh), rtol=0, atol=1e-6)
        for ray_mesh in (scan_mesh, terrain.warp_meshes["soil_rim"]):
            hit = downward(frames, ray_mesh)[0]
            torch.testing.assert_close(hit[:, 2], torch.zeros_like(hit[:, 2]), rtol=0, atol=2e-5)
        # Native rough height scan geometry stays unchanged.
        nominal = downward(terrain.env_origins[rigid], scan_mesh)[0]
        reference_hits = downward(terrain.env_origins[rigid], terrain.warp_meshes["terrain"])[0]
        torch.testing.assert_close(nominal, reference_hits, rtol=0, atol=1e-5)
        print("[Mixed check] all scenario heights/normals match mesh rays; bottoms, hard lanes and collider holes verified", flush=True)

        rough_id = int(rigid.nonzero()[0, 0])
        def fixture(world_point, depth=0.02):
            soil.reset()
            soil.prepare_state()
            soil.q.zero_()
            soil.q[..., 2], soil.q[..., 6] = 2.0, 1.0
            soil.qd.zero_()
            h, n, enabled = terrain.sample_surface(world_point[None])
            assert enabled.item() == 1
            local = world_point - raw.scene.env_origins[rough_id]
            local[2] = h[0] - raw.scene.env_origins[rough_id, 2]
            soil.q[rough_id, :, :3] = local
            soil.q[rough_id, :, 2] += 0.5
            soil.q[rough_id, soil.feet, 2] = local[2] + (soil.proxy.R_FOOT - depth) / n[0, 2]
            soil.evaluate()
            torch.testing.assert_close(soil.depth[rough_id], torch.full_like(soil.depth[rough_id], depth), rtol=0, atol=1e-5)
            assert (soil.normal_w[rough_id, soil.feet, 2] > 1).all()
            return n[0]

        # Force direction and dissipative tangential drag on every scenario,
        # including a rough-spawn robot at a nonzero local origin/height.
        for col, scenario in enumerate(terrain.soft_scenario_names):
            point = terrain.soft_tile_origins[col].clone()
            if scenario in ("mound", "basin", "undulating"):
                point[:2] += torch.tensor([0.637, 0.173], device=env.device)
            n = fixture(point)
            force = soil.wrench[rough_id, soil.calves, :3].clone()
            tangent_force = force - (force * n).sum(-1, keepdim=True) * n
            torch.testing.assert_close(tangent_force, torch.zeros_like(tangent_force), rtol=0, atol=2e-4)
            if scenario in ("mound", "basin", "undulating"):
                assert torch.linalg.vector_norm(n[:2]) > 1e-4
            tangent = torch.linalg.cross(n, torch.tensor([0., 1., 0.], device=env.device))
            tangent = 0.2 * tangent / torch.linalg.vector_norm(tangent)
            soil.qd[rough_id, soil.feet, :3] = tangent
            soil.evaluate()
            assert ((soil.wrench[rough_id, soil.calves, :3] * tangent).sum(-1) < 0).all()
            print(f"[Mixed check] {scenario}: 2 cm normal penetration, normal support and dissipative drag passed", flush=True)

        # A flat mapped patch must reproduce the original flat proxy equations.
        if "flat" in terrain.soft_scenario_names:
            point = terrain.soft_tile_origins[terrain.soft_scenario_names.index("flat")].clone()
            fixture(point)
            actual = [value[rough_id].clone() for value in (soil.wrench, soil.normal_w, soil.mem)]
            mapped, bounds = soil.surface_terrain, soil.local_soft_bounds
            soil.reset()
            soil.surface_terrain, soil.local_soft_bounds = None, None
            soil.evaluate()
            # The legacy rough-group reporter clears virtual contacts; compare
            # the physical wrench/memory directly before restoring the mapped mode.
            torch.testing.assert_close(soil.wrench[rough_id], actual[0], rtol=2e-5, atol=2e-4)
            soil.surface_terrain, soil.local_soft_bounds = mapped, bounds
            print("[Mixed check] flat mapped force law matches the original proxy", flush=True)
            # All five tiles use the material of the actual fixed logical level.
            col = terrain.soft_scenario_names.index("flat")
            fixture(terrain.soft_tile_origins[col].clone())
            expected = min(soil.proxy.K_FLOOR * 0.02, cfg.scene.terrain.bearing_scale * soil.row["bear_a"]
                           * parameters[terrain.soft_level][1] * 0.02 ** soil.row["bear_n"])
            torch.testing.assert_close(soil.wrench[rough_id, soil.calves, 2],
                                       torch.full((4,), expected, device=env.device), rtol=2e-4, atol=2e-3)
            print("[Mixed check] fixed logical level bearing curve passed", flush=True)

        if "mixed" in terrain.soft_scenario_names:
            point = terrain.soft_tile_origins[terrain.soft_scenario_names.index("mixed")].clone()
            fixture(point)
            soil.q[rough_id, soil.feet[2:], 0] += parameters[terrain.soft_level][2]
            soil.evaluate()
            assert (soil.normal_w[rough_id, soil.feet[:2], 2] > 1).all()
            assert soil.normal_w[rough_id, soil.feet[2:]].count_nonzero() == 0
            assert (soil.mem[rough_id, 2:] == -1).all()
            # Belly support is independent of which material the first foot sees.
            soil.q[rough_id, soil.feet[:2], 0] += parameters[0][2]
            soil.q[rough_id, soil.base, 2] = -raw.scene.env_origins[rough_id, 2] + 0.04
            soil.evaluate()
            assert soil.normal_w[rough_id, soil.feet].count_nonzero() == 0
            assert soil.normal_w[rough_id, soil.base, 2] > 1
            print("[Mixed check] straddling hard/soft bands and independent belly contact passed", flush=True)
        # Hard frames must not generate proxy forces or footprints for either group.
        soil.reset()
        soil.q[..., 2] = -0.01
        soil.q[..., 0] = sx / 2 - border / 2
        soil.prepare_state()
        for env_id in range(env.num_envs):
            point = terrain.soft_tile_origins[0].clone()
            point[0] += sx / 2 - border / 2
            point[2] = -0.01
            soil.q[env_id, :, :3] = point - raw.scene.env_origins[env_id]
        soil.evaluate()
        assert soil.wrench.count_nonzero() == 0 and soil.normal_w.count_nonzero() == 0
        assert (soil.mem == -1).all() and soil.hole_n.count_nonzero() == 0
        soil.reset()
        if args.no_adapter:
            # Same scene, native action implementation; soft robots can fall.
            # Only the physically isolated rough group is compared below.
            term = raw.action_manager.get_term("joint_pos")
            term.apply_actions = lambda: JointPositionAction.apply_actions(term)

        records = {name: [] for name in ("root", "joint", "contact", "reward")}
        soil_records = {name: [] for name in ("soil_root", "soil_joint", "soil_contact", "soil_reward",
                                             "soil_wrench", "soil_depth", "soil_mem", "soil_air", "soil_history")}
        max_soft_depth, soft_contact_steps = 0.0, 0
        zero = torch.zeros((env.num_envs, 12), device=env.device)
        for step in range(args.steps):
            with torch.inference_mode():
                obs, reward, done, extras = env.step(zero)
            for name, width in (("policy", 45), ("prop_history", 225), ("height_scan", 187), ("privileged_vel", 3)):
                assert obs[name].shape == (env.num_envs, width) and torch.isfinite(obs[name]).all(), name
            assert torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(reward).all()
            assert abs(float(extras["log"]["Mixed/soft_sample_fraction"]) - terrain.num_soft / args.num_envs) < 1e-6
            for scenario in terrain.scenario_masks:
                assert torch.isfinite(extras["log"][f"SoilScenario/{scenario}_reward_per_second"])
            if not args.no_adapter:
                # Group labels determine sampling, not contact material. These
                # short rough trajectories stay within the original cells.
                assert soil.wrench[rigid].count_nonzero() == 0 and soil.normal_w[rigid].count_nonzero() == 0
                assert torch.isfinite(soil.wrench).all()
                max_soft_depth = max(max_soft_depth, float(soil.depth[sand].max()))
                soft_contact_steps += int((soil.normal_w[sand][:, soil.feet, 2] > 1).any())
                view = sensor.contact_physx_view
                physical = view.physical.get_net_contact_forces(dt=cfg.sim.dt).view(env.num_envs, -1, 3)
                merged = view.get_net_contact_forces(dt=cfg.sim.dt).view(env.num_envs, -1, 3)
                torch.testing.assert_close(merged[rigid], physical[rigid], rtol=0, atol=0)
                torch.testing.assert_close(merged[sand], (physical + soil.normal_w[:, sensor.robot_body_ids])[sand])
            for name, value in zip(records, (robot.data.root_state_w, robot.data.joint_pos, sensor.data.net_forces_w, reward)):
                records[name].append(value[rigid].cpu().numpy().copy())
            if args.record or args.compare:
                values = (robot.data.root_state_w[sand], robot.data.joint_pos[sand], sensor.data.net_forces_w[sand],
                          reward[sand], soil.wrench[sand], soil.depth[sand], soil.mem[sand],
                          sensor.data.current_air_time[sand], sensor.data.net_forces_w_history[sand])
                for name, value in zip(soil_records, values):
                    soil_records[name].append(value.cpu().numpy().copy())
        if args.record:
            np.savez_compressed(args.record, **{name: np.stack(value) for name, value in {**records, **soil_records}.items()},
                                origins=original_origins.cpu().numpy(), rough_mesh_sha256=cfg.scene.terrain.rough_mesh_sha256,
                                soft_surface_sha256=cfg.scene.terrain.soft_surface_sha256,
                                physics_dt=cfg.sim.dt, decimation=cfg.decimation)
        if args.compare:
            with np.load(args.compare) as reference:
                if "decimation" in reference:
                    assert int(reference["decimation"]) == cfg.decimation and float(reference["physics_dt"]) == cfg.sim.dt, "Reference timing differs; select matching --substeps"
                assert str(reference["rough_mesh_sha256"]) == cfg.scene.terrain.rough_mesh_sha256
                if "soft_surface_sha256" in reference:
                    assert str(reference["soft_surface_sha256"]) == cfg.scene.terrain.soft_surface_sha256
                np.testing.assert_array_equal(reference["origins"], original_origins.cpu().numpy())
                for name, value in records.items():
                    np.testing.assert_array_equal(np.stack(value), reference[name], err_msg=name)
            print("[Mixed check] rough trajectories match native-action reference bit for bit", flush=True)
            with np.load(args.compare) as reference:
                if "soil_root" in reference:
                    for name, value in soil_records.items():
                        np.testing.assert_array_equal(np.stack(value), reference[name], err_msg=name)
                    print("[Mixed check] soft trajectories, wrenches, memory, contact history and air time match bit for bit", flush=True)
        if not args.no_adapter:
            assert soft_contact_steps > 0 and max_soft_depth > 0
            if terrain.soft_curriculum_active:
                assert terrain.soft_sample_steps[sand].sum() > 0
                assert terrain.soft_contact_steps[sand].sum() > 0
                assert (terrain.soft_contact_steps <= terrain.soft_sample_steps).all()
            # A curriculum update containing both groups must only move rough IDs.
            levels, origins = terrain.terrain_levels.clone(), terrain.env_origins.clone()
            ids = torch.arange(env.num_envs, device=env.device)
            terrain.update_env_origins(ids, torch.ones_like(ids, dtype=torch.bool),
                                       torch.zeros_like(ids, dtype=torch.bool))
            torch.testing.assert_close(terrain.env_origins[sand], origins[sand], rtol=0, atol=0)
            torch.testing.assert_close(terrain.terrain_levels[sand], levels[sand], rtol=0, atol=0)
            unsaturated = rigid & (levels + 1 < terrain.max_terrain_level)
            assert (terrain.terrain_levels[unsaturated] == levels[unsaturated] + 1).all()
            soil.prepare_state()
            torch.testing.assert_close(soil.local_soft_bounds,
                                       terrain.soft_bounds[None] - raw.scene.env_origins[:, None, :2])
            torch.testing.assert_close(soil.surface_heights, -raw.scene.env_origins[:, 2])
            terrain.terrain_levels.copy_(levels)
            terrain.env_origins.copy_(origins)
            if terrain.soft_curriculum_active:
                # Controlled outcomes through the production callback; preserve
                # PhysX state and unrelated robots while testing reset routing.
                soil_ids = sand.nonzero().flatten()
                test_ids = soil_ids[:2]
                position = robot.data.root_pos_w.clone()
                command = raw.command_manager.get_command("base_velocity")
                saved_command, saved_terminal = command.clone(), raw.termination_manager.terminated.clone()
                saved_tiles = terrain.soft_tile_ids.clone()
                saved_soft_origins = terrain.soft_origins.clone()
                @torch.inference_mode()
                def outcome(distance, contacts, speed=0.4, failed=False):
                    command[test_ids] = 0
                    command[test_ids, 0] = speed
                    robot.data.root_pos_w[test_ids, :2] = terrain.env_origins[test_ids, :2]
                    robot.data.root_pos_w[test_ids, 0] += distance
                    terrain.soft_sample_steps[test_ids] = 100
                    terrain.soft_contact_steps[test_ids] = contacts
                    raw.termination_manager.terminated[test_ids] = failed
                    before = terrain.terrain_levels.clone()
                    soft.mixed_terrain_levels(raw, test_ids)
                    assert (terrain.soft_sample_steps[test_ids] == 0).all()
                    assert (terrain.soft_contact_steps[test_ids] == 0).all()
                    assert torch.equal(terrain.terrain_levels[~torch.isin(ids, test_ids)], before[~torch.isin(ids, test_ids)])
                    torch.testing.assert_close(terrain.env_origins[sand], terrain.soft_tile_origins[terrain.soft_tile_ids])
                    return before[test_ids], terrain.terrain_levels[test_ids].clone()
                before, after = outcome(4.5, 100)
                assert torch.equal(after, before), "A shared mesh was promoted during a run"
                before, after = outcome(4.5, 0)
                assert torch.equal(after, before), "Hard-frame-only walking promoted"
                before, after = outcome(4.5, 100, failed=True)
                assert torch.equal(after, before), "Failed episode promoted"
                before, after = outcome(0, 100)
                assert torch.equal(after, before), "A shared mesh was demoted during a run"
                before, after = outcome(0, 100, speed=0)
                assert torch.equal(after, before), "Standing command demoted"
                assert torch.equal(terrain.soft_tile_ids % args.soft_columns, saved_tiles % args.soft_columns)
                terrain.terrain_levels.copy_(levels)
                terrain.env_origins.copy_(origins)
                terrain.soft_tile_ids.copy_(saved_tiles)
                terrain.soft_origins.copy_(saved_soft_origins)
                with torch.inference_mode():
                    robot.data.root_pos_w.copy_(position)
                    command.copy_(saved_command)
                    raw.termination_manager.terminated.copy_(saved_terminal)
                terrain.soft_contact_steps.zero_()
                terrain.soft_sample_steps.zero_()
                print("[Mixed check] shared mesh level stays fixed after successful/failed/standing episodes; scenario assignment retained", flush=True)
            # Exercise the real reset/curriculum path with one ID from each group.
            soft_id, rigid_id = sand.nonzero()[0, 0], rigid.nonzero()[0, 0]
            untouched_id = sand.nonzero()[-1, 0]
            soil.hole_d[sand] = 0.03
            memory = soil.hole_d[untouched_id].clone()
            reset_level = terrain.terrain_levels[soft_id].clone()
            with torch.inference_mode():
                if terrain.soft_curriculum_active:
                    # A genuine reset retains this run's geometry and leaves
                    # other robots' counters and footprint memory intact.
                    robot.data.root_pos_w[soft_id, :2] = terrain.env_origins[soft_id, :2]
                    robot.data.root_pos_w[soft_id, 0] += 4.5
                    command = raw.command_manager.get_command("base_velocity")
                    command[soft_id] = 0
                    command[soft_id, 0] = 0.4
                    raw.termination_manager.terminated[soft_id] = False
                    terrain.soft_contact_steps[soft_id] = terrain.soft_sample_steps[soft_id] = 100
                    terrain.soft_sample_steps[untouched_id] = 99
                raw._reset_idx(torch.stack((soft_id, rigid_id)))
            torch.testing.assert_close(terrain.env_origins[sand], terrain.soft_tile_origins[terrain.soft_tile_ids], rtol=0, atol=0)
            if terrain.soft_curriculum_active:
                assert terrain.terrain_levels[soft_id] == reset_level
                assert terrain.soft_sample_steps[soft_id] == 0 and terrain.soft_contact_steps[soft_id] == 0
                if soft_id != untouched_id:
                    assert terrain.soft_sample_steps[untouched_id] == 99
                terrain.soft_sample_steps.zero_()
                terrain.soft_contact_steps.zero_()
                zero_flags = torch.zeros(terrain.num_soft, dtype=torch.bool, device=env.device)
                terrain.update_soft_origins(sand.nonzero().flatten(), zero_flags, zero_flags)
            assert soil.hole_d[soft_id].count_nonzero() == 0 and (soil.mem[soft_id] == -1).all()
            if soft_id != untouched_id:
                torch.testing.assert_close(soil.hole_d[untouched_id], memory, rtol=0, atol=0)
            soil.reset()
            if terrain.soft_curriculum_active:
                with torch.inference_mode():
                    raw._reset_idx(sand.nonzero().flatten())

            agent_cfg = soft.Go2MixedRunnerCfg()
            resolved = OmegaConf.create(agent_cfg.to_dict())
            if args.checkpoint:
                resolved.init_checkpoint = args.checkpoint
            agent_cfg.from_dict(OmegaConf.to_container(resolved, resolve=True))
            agent, original = agent_cfg.to_dict(), soft.Go2DreamWaQRoughPPORunnerCfg().to_dict()
            for name in ("experiment_name", "init_checkpoint", "stage_report"):
                agent.pop(name, None)
                original.pop(name, None)
            assert agent == original, "Original network/PPO settings changed"
            runner = soft.make_runner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
            policy = runner.alg.policy
            if args.checkpoint:
                source = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
                for name, value in policy.state_dict().items():
                    torch.testing.assert_close(value, source["model_state_dict"][name], rtol=0, atol=0)
                assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
            # One disposable update through the original algorithm. Tag only the
            # storage dictionary, never actor/critic observations, to audit both groups.
            with torch.inference_mode():
                obs = env.get_observations()
                for _ in range(runner.num_steps_per_env):
                    actions = runner.alg.act(obs)
                    obs, rewards, dones, extras = env.step(actions)
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                runner.alg.compute_returns(obs)
            storage = runner.alg.storage
            tags = terrain.soft_mask.float().view(1, -1, 1).expand(runner.num_steps_per_env, -1, -1).clone()
            storage.observations["_diagnostic_domain"] = tags
            batches, samples = 0, [0, 0]
            native_generator = storage.mini_batch_generator
            def audited_batches(*args, **kwargs):
                nonlocal batches
                for batch in native_generator(*args, **kwargs):
                    marker = batch[0].pop("_diagnostic_domain").flatten()
                    samples[0] += int((marker == 0).sum())
                    samples[1] += int((marker == 1).sum())
                    batches += 1
                    yield batch
            storage.mini_batch_generator = audited_batches
            before = {name: value.clone() for name, value in policy.state_dict().items()}
            losses = runner.alg.update()
            storage.mini_batch_generator = native_generator
            assert min(samples) > 0 and all(np.isfinite(value) for value in losses.values())
            assert all(torch.isfinite(value).all() for value in policy.state_dict().values())
            assert any(not torch.equal(before[name], value) for name, value in policy.state_dict().items())
            expected = runner.num_steps_per_env * runner.alg.num_learning_epochs
            assert samples == [terrain.num_rough * expected, terrain.num_soft * expected], samples
            print(f"[Mixed check] discarded one PPO update: batches={batches}, rough/soft samples={samples}, losses={losses}", flush=True)
            # Discard the update; no checkpoint or training run is written.
            policy.load_state_dict(before)
        print(f"[MIXED CHECK PASS] rough={terrain.num_rough} soft={terrain.num_soft} "
              f"substeps={cfg.decimation} dt={cfg.sim.dt} "
              f"steps={args.steps} soft_contact_steps={soft_contact_steps} max_soft_depth={max_soft_depth:.4f} m", flush=True)
        diagnostic_status(True)
    except Exception:
        diagnostic_status(False)
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def force_law_checks(soil, weight):
    """Controlled intruder/velocity cycles using the very same production kernel."""
    import torch

    results, curve = {}, []
    def run(name, test):
        try:
            results[name] = {"passed": True, "metrics": test()}
            print(f"[Soil physics] {name}: PASS {results[name]['metrics']}", flush=True)
        except Exception as error:
            import traceback
            traceback.print_exc()
            results[name] = {"passed": False, "error": str(error)}

    def fixture(depth):
        soil.reset()
        soil.q.zero_()
        soil.q[..., 2] = 0.5
        soil.q[..., 6] = 1
        xy = torch.tensor([[0.3, 0.2], [0.3, -0.2], [-0.3, 0.2], [-0.3, -0.2]], device=soil.device)
        for ids, z in ((soil.feet, soil.proxy.R_FOOT - depth), (soil.calves, 0.25), (soil.thighs, 0.45)):
            soil.q[:, ids, :2] = xy
            soil.q[:, ids, 2] = z
        soil.qd.zero_()

    def bearing():
        depths = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.95 * soil.cfg.depth]
        for depth in sorted(set(depths)):
            fixture(depth)
            soil.evaluate()
            expected = min(soil.proxy.K_FLOOR * depth,
                           soil.cfg.bearing_scale * soil.row["bear_a"] * depth ** soil.row["bear_n"])
            measured = soil.wrench[:, soil.calves, 2]
            torch.testing.assert_close(measured, torch.full_like(measured, expected), rtol=2e-5, atol=2e-4)
            assert (soil.wrench[..., :2].abs() < 1e-6).all()
            # Moment balance around the world origin, independent of a body's COM offset.
            com = soil.q[:, soil.calves, :3] + soil.com[:, soil.calves]
            moment = soil.wrench[:, soil.calves, 3:] + torch.cross(com, soil.wrench[:, soil.calves, :3], dim=-1)
            expected_moment = torch.cross(soil.q[:, soil.feet, :3], soil.wrench[:, soil.calves, :3], dim=-1)
            torch.testing.assert_close(moment, expected_moment, rtol=2e-5, atol=2e-4)
            curve.append([depth, float(measured.mean())])
        capacity = 4 * curve[-1][1]
        assert capacity > float(weight.max()), "Bearing capacity is insufficient above the hard bottom"
        equilibrium = torch.maximum(
            (weight / (4 * soil.cfg.bearing_scale * soil.row["bear_a"])) ** (1 / soil.row["bear_n"]),
            weight / (4 * soil.proxy.K_FLOOR),
        )
        assert (equilibrium < soil.cfg.depth - 0.01).all()
        return {"predicted_equal_load_depth_m": equilibrium.cpu().tolist(), "capacity_before_bottom_N": capacity}

    def drag():
        powers = []
        for vx, vy in ((0, 0), (0.01, 0), (-0.01, 0), (0.3, 0.4), (-0.3, -0.4), (0, 1), (0, -1)):
            fixture(0.015)  # The foot centre and both rods are above the nominal surface.
            soil.qd[:, soil.feet, 0] = vx
            soil.qd[:, soil.feet, 1] = vy
            soil.evaluate()
            f = soil.wrench[:, soil.calves, :3]
            power = f[..., 0] * vx + f[..., 1] * vy
            assert torch.isfinite(f).all() and (power <= 1e-6).all()
            if (vx or vy) and soil.cfg.drag_scale > 0:
                assert (power < 0).all(), "Horizontal resistance must oppose motion"
            else:
                assert (f[..., :2] == 0).all()
            # Reverse velocity must reverse drag without changing vertical lift.
            first = f.clone()
            soil.qd[:, soil.feet, :2] *= -1
            soil.evaluate()
            torch.testing.assert_close(soil.wrench[:, soil.calves, :2], -first[..., :2])
            torch.testing.assert_close(soil.wrench[:, soil.calves, 2], first[..., 2], atol=2e-4, rtol=2e-5)
            powers.append(float(power.mean()))
        for sign in (-1, 1):
            fixture(0.08)
            soil.q[:, soil.calves, 2] = -0.02
            for ids in (soil.calves, soil.thighs):
                soil.qd[:, ids, 0] = sign * 0.3
                soil.qd[:, ids, 1] = sign * 0.4
            soil.evaluate()
            for ids in (soil.calves, soil.thighs):
                power = (soil.wrench[:, ids, :2] * soil.qd[:, ids, :2]).sum(-1)
                assert (power <= 0).all(), "Buried rod drag injected horizontal energy"
                if soil.cfg.drag_scale > 0:
                    assert (power < 0).all()
        return {"foot_horizontal_power_W": powers, "shin_thigh_drag_dissipative": True}

    def pullout():
        fixture(0.03)
        soil.mem[:] = 0.04  # Unloaded foot: no elastic bearing to mask the pull-out term.
        soil.qd[:, soil.feet, 2] = 0.2
        soil.evaluate()
        force = soil.wrench[:, soil.calves, 2]
        expected = -soil.row["pull_p"] * 0.03
        torch.testing.assert_close(force, torch.full_like(force, expected), rtol=2e-5, atol=2e-4)
        assert (force * 0.2 < 0).all() and (soil.contact == 0).all()
        return {"upward_speed_m_s": 0.2, "pullout_force_per_foot_N": float(force.mean())}

    def memory():
        fixture(0.06)
        soil.evaluate()
        assert (soil.mem > 0.005).all()
        soil.q[:, soil.feet, 2] = soil.proxy.R_FOOT + 0.02
        soil.evaluate()
        assert (soil.mem == -1).all() and (soil.hole_n == 4).all()
        previous = soil.hole_d[1:].clone()
        soil.reset(torch.tensor([0], device=soil.device))
        torch.testing.assert_close(soil.hole_d[1:], previous, atol=0, rtol=0)
        soil.q[:, soil.feet, 2] = soil.proxy.R_FOOT - 0.005
        soil.evaluate()
        fresh = float(soil.wrench[0, soil.calves, 2].mean())
        used = float(soil.wrench[1:, soil.calves, 2].mean())
        assert fresh > 0 and abs(used) < 1e-5, "A reused hole must unload a shallow returning foot"
        assert (soil.mem[1:] > 0.005).all(), "Other environments lost their footprint memory"
        fixture(0.06)
        for _ in range(20):
            soil.q[:, soil.feet, 2] = soil.proxy.R_FOOT - 0.06
            soil.evaluate()
            soil.q[:, soil.feet, 2] = soil.proxy.R_FOOT + 0.02
            soil.evaluate()
            soil.q[:, soil.feet, 0] += 0.1
        assert (soil.hole_n == 80).all()
        assert torch.isfinite(soil.hole_d).all() and ((soil.hole_d > 0) & (soil.hole_d < 0.06)).all()
        soil.reset()
        assert (soil.mem == -1).all() and (soil.hole_n == 0).all() and (soil.hole_d == 0).all()
        return {"fresh_shallow_force_N": fresh, "reused_shallow_force_N": used,
                "footprints_written_per_environment": 80, "ring_capacity": 16, "reset_isolated": True}

    for name, test in (("bearing_and_moment", bearing), ("drag", drag), ("pullout", pullout), ("footprint_memory", memory)):
        run(name, test)
    soil.reset()
    return results, curve


def horizontal_drag_wrench(soil, com):
    """Diagnostic decomposition only; no force/state is fed back into simulation.

    Keep the kernel's horizontal forces, remove normal/lift forces and their
    moments. Reconstruct shin/thigh application points from the upstream rod law.
    com and q positions are relative to the same environment origin.
    """
    import torch

    q, qd = soil.q, soil.qd
    ids = soil.calves + soil.thighs
    lo = torch.cat((q[:, soil.feet, :3], q[:, soil.calves, :3]), dim=1)
    hi = torch.cat((q[:, soil.calves, :3], q[:, soil.thighs, :3]), dim=1)
    dz = hi[..., 2] - lo[..., 2]
    fraction = torch.where((lo[..., 2] < 0) & (dz > 0),
                           (-lo[..., 2] / dz.clamp_min(1e-9)).clamp(0, 1), 0)
    point = lo + (hi - lo) * (0.5 * fraction[..., None])
    velocity = qd[:, ids, :3] + torch.cross(qd[:, ids, 3:], point - com[:, ids], dim=-1)
    magnitude = soil.cfg.drag_scale * soil.row["shin_s"] * fraction.square() * (hi - lo).square().sum(-1)
    epsilon = (magnitude / soil.proxy.C_MAX).clamp_min(0.1)
    rod_xy = -magnitude[..., None] * velocity[..., :2] / torch.sqrt(
        velocity[..., :2].square().sum(-1) + epsilon.square()
    )[..., None]
    out = torch.zeros_like(soil.wrench)
    out[..., :2] = soil.wrench[..., :2]  # All horizontal kernel forces are drag.
    shin = torch.cat((rod_xy[:, :4], torch.zeros_like(rod_xy[:, :4, :1])), dim=-1)
    foot = out[:, soil.calves, :3] - shin
    out[:, soil.calves, 3:] = (
        torch.cross(q[:, soil.feet, :3] - com[:, soil.calves], foot, dim=-1)
        + torch.cross(point[:, :4] - com[:, soil.calves], shin, dim=-1)
    )
    out[:, soil.thighs, 3:] = torch.cross(
        point[:, 4:] - com[:, soil.thighs], out[:, soil.thighs, :3], dim=-1,
    )
    return out


def velocity_audit(arrays, dt, tail):
    """Report interval-aligned velocity consistency and work; never relax acceptance."""
    import numpy as np

    def mean_rms(x):
        return np.sqrt(np.mean(x[tail] ** 2, axis=tuple(range(2, x.ndim)))).mean(0).tolist()

    # Position spectra and velocity spectra have different units: locate their
    # peaks independently instead of comparing unnormalised powers.
    spectra = {}
    for key in ("joint_pos", "joint_vel"):
        values = arrays[key][tail]
        power = np.abs(np.fft.rfft(values - values.mean(0), axis=0)) ** 2
        power = power.sum(-1)
        frequencies = np.fft.rfftfreq(len(values), dt)
        spectra[key + "_peak_frequency_hz"] = frequencies[np.argmax(power[1:], axis=0) + 1].tolist()
    work = arrays["drag_work_configuration"][tail]
    pre_power = arrays["drag_power_before"][tail]
    return {
        "joint_reported_speed_mean_rms_rad_s": mean_rms(arrays["joint_vel"]),
        "joint_position_difference_speed_mean_rms_rad_s": mean_rms(arrays["joint_fd_vel"]),
        "joint_interval_velocity_mismatch_mean_rms_rad_s": mean_rms(arrays["joint_fd_vel"] - arrays["joint_trapezoid_vel"]),
        "foot_reported_speed_mean_rms_m_s": mean_rms(arrays["foot_vel"]),
        "foot_position_difference_speed_mean_rms_m_s": mean_rms(arrays["foot_fd_vel"]),
        "foot_interval_velocity_mismatch_mean_rms_m_s": mean_rms(arrays["foot_fd_vel"] - arrays["foot_trapezoid_vel"]),
        "horizontal_drag_max_power_before_W": pre_power.max(0).tolist(),
        "horizontal_drag_positive_configuration_work_fraction": (work > 1e-6).mean(0).tolist(),
        "horizontal_drag_net_configuration_work_J": work.sum(0).tolist(),
        "horizontal_drag_positive_configuration_work_J": np.maximum(work, 0).sum(0).tolist(),
        "horizontal_drag_net_trapezoid_work_J": arrays["drag_work_trapezoid"][tail].sum(0).tolist(),
        "work_note": "Horizontal drag only; normal bearing, vertical lift and pull-out excluded. "
                     "Configuration work uses COM displacement and a small-rotation wrench estimate; "
                     "trapezoid work uses reported velocities. These are diagnostics, not total-system energy balance.",
        "positive_work_fraction_deadband_J": 1e-6,
        **spectra,
    }


def physics_sweep(args, arguments):
    """Fresh processes for paired rigid/soft experiments, one dt change at a time."""
    from datetime import datetime
    import json
    import subprocess

    output = Path(args.out) if args.out else ROOT / "logs/rsl_rl/isaaclab/go2_dreamwaq_soft_physics" / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + f"_dt_sweep_seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=False)
    forwarded, skip = [], False
    for item in arguments:
        if skip:
            skip = False
        elif item in ("--out", "--physics_dt", "--sweep_dts"):
            skip = True
        elif item == "--sweep_dt" or item.startswith(("--out=", "--physics_dt=", "--sweep_dts=")):
            continue
        else:
            forwarded.append(item)
    cases, reference_mass = [], None
    timesteps = [float(value) for value in args.sweep_dts.split(",")]
    for dt in timesteps:
        for rigid in (False, True):
            name = f"dt_{dt * 1000:g}ms_{'rigid' if rigid else 'soft'}"
            case_dir = output / name
            print(f"[DT sweep] Starting {name}; output={case_dir}", flush=True)
            command = [sys.executable, __file__, "physics", *forwarded,
                       "--physics_dt", str(dt), "--out", str(case_dir)]
            if rigid:
                command.append("--rigid")
            with (output / "progress.log").open("a") as log:
                log.write(f"\n--- {name} ---\n")
                log.flush()
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if not (case_dir / "summary.json").exists():
                raise RuntimeError(f"{name} failed before producing results; inspect {output / 'progress.log'}")
            summary = json.loads((case_dir / "summary.json").read_text())
            if process.returncode != (0 if summary["passed"] else 1):
                raise RuntimeError(f"{name} process status disagrees with saved acceptance result")
            masses = summary["results"]["fixed_target_standing"]["metrics"]["weight_N"]
            if reference_mass is None:
                reference_mass = masses
            assert masses == reference_mass, "Startup randomisation differs between cases"
            cases.append({"name": name, "run_directory": str(case_dir), **summary})
            print(f"[DT sweep] {name}: {'PASS' if summary['passed'] else 'FAIL'}; "
                  f"joint speeds={summary['velocity_audit']['joint_reported_speed_mean_rms_rad_s']}", flush=True)
            (output / "comparison.json").write_text(json.dumps({"completed": len(cases) == 2 * len(timesteps), "cases": cases}, indent=2) + "\n")
    physics_sweep_report(cases, output)


def physics_sweep_report(cases, output):
    """Rebuild a comparison from raw, preserved runs without repeating simulation."""
    import json
    import numpy as np
    import yaml

    # Validate resolved settings rather than assuming that the CLI left them equal.
    class ConfigLoader(yaml.SafeLoader):
        pass
    ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
    ConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice",
                                 lambda loader, node: slice(*loader.construct_sequence(node)))
    configs = [yaml.load((Path(case["run_directory"]) / "env.yaml").read_text(), Loader=ConfigLoader) for case in cases]
    def invariant(config):
        import copy
        config = copy.deepcopy(config)
        config.pop("decimation")
        for key in ("dt", "render_interval"):
            config["sim"].pop(key)
        scene = config["scene"]
        scene["contact_forces"].pop("update_period")
        scene["contact_forces"].pop("class_type")
        scene["height_scanner"].pop("mesh_prim_paths")
        scene["terrain"].pop("soft_enabled")
        config["actions"]["joint_pos"].pop("class_type")
        return config
    for config in configs:
        assert abs(config["sim"]["dt"] * config["decimation"] - 0.02) < 1e-10
        assert invariant(config) == invariant(configs[0]), "Unexpected task/robot/terrain parameter changes in the sweep"
    for dt in {case["dt"] for case in cases}:
        assert {case["rigid_control"] for case in cases if case["dt"] == dt} == {False, True}
    weights = [case["results"]["fixed_target_standing"]["metrics"]["weight_N"] for case in cases]
    assert all(weight == weights[0] for weight in weights)
    for case in cases:
        with np.load(Path(case["run_directory"]) / "measurements.npz") as raw:
            tail = slice(-round(2 / case["dt"]), None)
            for label, difference in (
                ("joint_end_velocity_difference_mean_rms_rad_s", raw["joint_fd_vel"][tail] - raw["joint_vel"][tail]),
                ("foot_end_velocity_difference_mean_rms_m_s", raw["foot_fd_vel"][tail] - raw["foot_vel"][tail]),
            ):
                case["velocity_audit"][label] = np.sqrt((difference ** 2).mean(tuple(range(2, difference.ndim)))).mean(0).tolist()
    (output / "comparison.json").write_text(json.dumps({
        "completed": True, "configuration_invariants_passed": True, "control_period_s": 0.02,
        "all_acceptance_passed": all(case["passed"] for case in cases), "cases": cases,
        "velocity_note": "Endpoint and trapezoid comparisons are distinct. A trapezoid mismatch alone "
                         "does not prove that reported velocities are invalid.",
    }, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for rigid in (False, True):
        selected = [case for case in cases if case["rigid_control"] == rigid]
        x = np.array([case["dt"] * 1000 for case in selected])
        label = "rigid" if rigid else "soft"
        reported = np.array([case["velocity_audit"]["joint_reported_speed_mean_rms_rad_s"] for case in selected])
        axes[0, 0].plot(x, reported.mean(1), "o-", label=label + " reported")
        axes[0, 0].fill_between(x, reported.min(1), reported.max(1), alpha=0.15)
        axes[0, 0].plot(x, [np.mean(case["velocity_audit"]["joint_position_difference_speed_mean_rms_rad_s"]) for case in selected], "x--", label=label + " position difference")
        axes[0, 1].plot(x, [np.mean(case["results"]["fixed_target_standing"]["metrics"]["final_base_height_m"]) for case in selected], "o-", label=label)
        axes[1, 0].plot(x, [np.mean(case["velocity_audit"]["foot_end_velocity_difference_mean_rms_m_s"]) for case in selected], "o-", label=label)
        axes[1, 1].plot(x, [np.mean(case["velocity_audit"]["horizontal_drag_net_configuration_work_J"]) / 2 for case in selected], "o-", label=label + " configuration")
        axes[1, 1].plot(x, [np.mean(case["velocity_audit"]["horizontal_drag_net_trapezoid_work_J"]) / 2 for case in selected], "x--", label=label + " velocity trapezoid")
    axes[0, 0].set_yscale("log")
    axes[0, 0].axhline(0.2, color="black", linestyle=":", label="acceptance limit")
    timesteps_ms = sorted({case["dt"] * 1000 for case in cases})
    for axis, ylabel in zip(axes.flat, ("Mean joint speed (rad/s)", "Final base height (m)", "Foot endpoint velocity / position difference (m/s)", "Horizontal drag work estimates / time (W)")):
        axis.set(xlabel="Physics timestep (ms)", ylabel=ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks(timesteps_ms, [f"{dt:g}" for dt in timesteps_ms])
        axis.legend()
        axis.grid(alpha=0.3)
    fig.suptitle("Paired timestep convergence | 20 ms control period | original soil and actuator settings")
    fig.savefig(output / "dt_convergence.png", dpi=160)
    plt.close(fig)
    print(f"[DT SWEEP COMPLETE] {len(cases)} results saved in {output}; configuration invariants PASS", flush=True)
    raise SystemExit(0 if all(case["passed"] for case in cases) else 1)


def physics(arguments):
    """Physical acceptance only: force cycles, free-flight force balance, fixed-target standing."""
    import argparse
    from datetime import datetime
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=physics.__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--physics_dt", type=float, default=0.0003125, help="Diagnostic timestep; control period remains 20 ms")
    parser.add_argument("--sweep_dt", action="store_true", help="Run paired soft/rigid cases at 5, 2.5, 1.25 ms")
    parser.add_argument("--sweep_dts", default="0.005,0.0025,0.00125", help="Comma-separated diagnostic timesteps for --sweep_dt")
    parser.add_argument("--report_sweep", nargs="+", help="Rebuild a combined report from saved sweep directories; no simulation")
    parser.add_argument("--rigid", action="store_true", help="Matched flat rigid standing control")
    parser.add_argument("--lock_joints", action="store_true", help="Diagnostic only: constrain joints to the default posture")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--material", default="f0.75")
    parser.add_argument("--depth", type=float, default=0.15)
    parser.add_argument("--drag_scale", type=float, default=1.0)
    parser.add_argument("--floor_stiffness", type=float, help="Experimental footprint spring stiffness, N/m")
    parser.add_argument("--normal_damping", type=float, help="Experimental normal damping, N s/m")
    parser.add_argument("--drag_damping_cap", type=float, help="Experimental low-speed drag slope cap, N s/m")
    parser.add_argument("--out", help="New diagnostic run directory (must not exist)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(arguments)
    if args.num_envs < 2 or args.seconds < 4:
        parser.error("Use at least 2 environments and 4 seconds")
    import math
    if not math.isfinite(args.physics_dt) or args.physics_dt <= 0 or args.physics_dt > 0.02:
        parser.error("Physics timestep must be positive and at most 20 ms")
    decimation = round(0.02 / args.physics_dt)
    if not math.isclose(decimation * args.physics_dt, 0.02, abs_tol=1e-10):
        parser.error("Physics timestep must divide the fixed 20 ms control period")
    if args.report_sweep:
        import json
        cases = []
        for source in args.report_sweep:
            cases.extend(json.loads((Path(source) / "comparison.json").read_text())["cases"])
        output = Path(args.out) if args.out else ROOT / "logs/rsl_rl/isaaclab/go2_dreamwaq_soft_physics" / (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_dt_report"
        )
        output.mkdir(parents=True, exist_ok=False)
        physics_sweep_report(cases, output)
        return
    if args.sweep_dt:
        if args.rigid or args.lock_joints:
            parser.error("The sweep requires free joints and automatically includes both soft and rigid cases")
        try:
            timesteps = [float(value) for value in args.sweep_dts.split(",")]
            valid = len(set(timesteps)) == len(timesteps) and all(
                math.isfinite(dt) and 0 < dt <= 0.02 and math.isclose(round(0.02 / dt) * dt, 0.02, abs_tol=1e-10)
                for dt in timesteps
            )
        except (ValueError, OverflowError):
            valid = False
        if not valid:
            parser.error("Sweep timesteps must be unique, positive, finite divisors of 20 ms")
        physics_sweep(args, arguments)
        return
    output = Path(args.out) if args.out else ROOT / "logs/rsl_rl/isaaclab/go2_dreamwaq_soft_physics" / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + f"_{'rigid' if args.rigid else 'soft'}_seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=False)
    print(f"[Soil physics] Output: {output}", flush=True)
    app = AppLauncher(args).app
    env = None
    try:
        import json
        import numpy as np
        import torch
        import gymnasium as gym
        import soft_terrain as soft
        from isaaclab.utils.io import dump_yaml
        from isaaclab.utils.math import quat_box_minus

        cfg = soft.Go2SoftEnvCfg()
        cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = args.physics_dt, decimation, decimation
        cfg.scene.contact_forces.update_period = cfg.sim.dt
        cfg.scene.height_scanner.update_period = 0.02
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
        cfg.scene.terrain.material, cfg.scene.terrain.depth = args.material, args.depth
        cfg.scene.terrain.soft_enabled = not args.rigid
        cfg.scene.terrain.drag_scale = args.drag_scale
        for key in ("floor_stiffness", "normal_damping", "drag_damping_cap"):
            if getattr(args, key) is not None:
                setattr(cfg.scene.terrain, key, getattr(args, key))
        env = gym.make("Isaac-Velocity-Flat-Go2-DreamWaQ-Soft-v0", cfg=cfg)
        raw = env.unwrapped
        robot, sensor = raw.scene["robot"], raw.scene["contact_forces"]
        soil = (soft.SoilModel(raw, robot) if args.rigid else raw.action_manager.get_term("joint_pos").soil)
        contact_view = getattr(sensor.contact_physx_view, "physical", sensor.contact_physx_view)
        masses = robot.root_physx_view.get_masses().to(raw.device)
        mass = masses.sum(-1)
        gravity = torch.tensor(cfg.sim.gravity, device=raw.device)
        weight = -gravity[2] * mass
        results, curve = force_law_checks(soil, weight)
        dump_yaml(str(output / "env.yaml"), cfg)

        def com_velocity():
            return (robot.data.body_lin_vel_w * masses[..., None]).sum(1) / mass[:, None]

        def foot_velocity():
            return robot.data.body_lin_vel_w[:, soil.feet] + torch.cross(
                robot.data.body_ang_vel_w[:, soil.feet],
                robot.data.body_pos_w[:, soil.feet] - robot.data.body_com_pos_w[:, soil.feet], dim=-1,
            )

        def reset_pose(height):
            env.reset()
            state = robot.data.default_root_state.clone()
            state[:, :3] = raw.scene.env_origins
            state[:, 2] += height
            state[:, 3:7] = torch.tensor([1., 0., 0., 0.], device=raw.device)
            state[:, 7:] = 0
            robot.write_root_state_to_sim(state)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
            raw.action_manager.reset()
            soil.reset()
            raw.action_manager.process_action(torch.zeros((args.num_envs, 12), device=raw.device))
            raw.scene.write_data_to_sim()
            raw.sim.forward()
            raw.scene.update(cfg.sim.dt)

        # Conservation of total linear momentum tests actual PhysX application,
        # including the production apply_wrench -> scene.write_data_to_sim order.
        errors = []
        for body, factor, horizontal in ((soil.base, 0., 0.), (soil.base, 1., 0.),
                                         (soil.base, 2., 0.), (soil.calves[0], 1., 0.3)):
            reset_pose(2.0)
            before = com_velocity().clone()
            raw.action_manager.apply_action()  # Soil must exert zero force in the air.
            assert (soil.wrench == 0).all()
            soil.wrench[:, body, 0] = horizontal * weight
            soil.wrench[:, body, 2] = factor * weight
            soil.apply_wrench()
            raw.scene.write_data_to_sim()
            raw.sim.step(render=False)
            raw.scene.update(cfg.sim.dt)
            acceleration = (com_velocity() - before) / cfg.sim.dt
            expected = gravity + soil.wrench[..., :3].sum(1) / mass[:, None]
            error = float((acceleration - expected).abs().max())
            errors.append(error)
            print(f"[Soil physics] free-flight force x{factor}, body={body}: max error {error:.6f} m/s^2", flush=True)
        results["physx_force_application"] = {"passed": max(errors) < 0.1,
            "metrics": {"max_acceleration_error_m_s2": max(errors), "tolerance_m_s2": 0.1}}

        if args.lock_joints:
            q = robot.data.default_joint_pos
            limits = torch.stack((q - 1e-5, q + 1e-5), dim=-1)
            robot.write_joint_limits_to_sim(limits)
        reset_pose(float(robot.data.default_root_state[0, 2]))
        rows = {key: [] for key in ("time", "root_z", "up_z", "com_vz", "soil_support", "floor_support",
                                    "depth", "joint_error", "joint_speed", "joint_vel", "momentum_residual",
                                    "joint_pos", "joint_fd_vel", "joint_trapezoid_vel", "foot_pos", "foot_vel",
                                    "foot_fd_vel", "foot_trapezoid_vel", "soil_wrench", "drag_power_before",
                                    "drag_work_trapezoid", "drag_work_configuration")}
        steps = int(round(args.seconds / cfg.sim.dt))
        print(f"[Soil physics] Fixed default joint targets, locked={args.lock_joints}, "
              "no policy, no automatic resets", flush=True)
        for step in range(steps):
            before = com_velocity().clone()
            joint_before = robot.data.joint_pos.clone()
            joint_velocity_before = robot.data.joint_vel.clone()
            foot_before = robot.data.body_pos_w[:, soil.feet].clone()
            foot_velocity_before = foot_velocity().clone()
            body_com_before = robot.data.body_com_pos_w.clone()
            body_quat_before = robot.data.body_quat_w.clone()
            body_velocity_before = robot.data.body_vel_w.clone()
            raw.action_manager.apply_action()
            if args.rigid:
                soil.wrench.zero_()
                soil.depth[:] = soil.proxy.R_FOOT - robot.data.body_pos_w[:, soil.feet, 2]
            drag = (torch.zeros_like(soil.wrench) if args.rigid else horizontal_drag_wrench(
                soil, body_com_before - raw.scene.env_origins[:, None],
            ))
            if step % 200 == 0:
                # Normal forces and lift cannot contribute yaw moment.
                torch.testing.assert_close(drag[..., 5], soil.wrench[..., 5], rtol=1e-4, atol=2e-4)
            drag_power = (drag * body_velocity_before).sum((1, 2))
            force = soil.wrench[..., 2].sum(-1).clone()
            raw.scene.write_data_to_sim()
            raw.sim.step(render=False)
            raw.scene.update(cfg.sim.dt)
            physical = contact_view.get_net_contact_forces(dt=cfg.sim.dt)
            floor = physical.view(args.num_envs, -1, 3)[..., 2].sum(-1)
            velocity = com_velocity()[:, 2]
            quat = robot.data.root_quat_w
            joint_fd = (robot.data.joint_pos - joint_before) / cfg.sim.dt
            foot_fd = (robot.data.body_pos_w[:, soil.feet] - foot_before) / cfg.sim.dt
            after_quat = robot.data.body_quat_w
            after_quat = torch.where((after_quat * body_quat_before).sum(-1, keepdim=True) < 0, -after_quat, after_quat)
            rotation = quat_box_minus(after_quat.reshape(-1, 4), body_quat_before.reshape(-1, 4)).reshape(args.num_envs, -1, 3)
            drag_work = (drag[..., :3] * (robot.data.body_com_pos_w - body_com_before)
                         + drag[..., 3:] * rotation).sum((1, 2))
            trapezoid_work = (drag * (robot.data.body_vel_w + body_velocity_before) * 0.5).sum((1, 2)) * cfg.sim.dt
            values = ((step + 1) * cfg.sim.dt, robot.data.root_pos_w[:, 2],
                      1 - 2 * (quat[:, 1].square() + quat[:, 2].square()), velocity, force, floor,
                      soil.depth.clone(), (robot.data.joint_pos - robot.data.default_joint_pos).square().mean(-1).sqrt(),
                      robot.data.joint_vel.square().mean(-1).sqrt(),
                      robot.data.joint_vel,
                      (velocity - before[:, 2]) * mass / cfg.sim.dt - (force + floor - weight),
                      robot.data.joint_pos, joint_fd, (robot.data.joint_vel + joint_velocity_before) * 0.5,
                      robot.data.body_pos_w[:, soil.feet], foot_velocity(), foot_fd,
                      (foot_velocity() + foot_velocity_before) * 0.5, soil.wrench,
                      drag_power, trapezoid_work, drag_work)
            for key, value in zip(rows, values):
                rows[key].append(value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else value)
            assert torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(soil.wrench).all(), "Non-finite physics"
            if (step + 1) % round(1 / cfg.sim.dt) == 0:
                print(f"[Soil physics] t={(step + 1) * cfg.sim.dt:.1f}s "
                      f"height={float(robot.data.root_pos_w[:, 2].mean()):.3f}m "
                      f"support/weight={float(((force + floor) / weight).mean()):.3f}", flush=True)
        arrays = {key: np.asarray(value) for key, value in rows.items()}
        weight_np = weight.cpu().numpy()
        tail = slice(-round(2 / cfg.sim.dt), None)
        support_ratio = (arrays["soil_support"][tail] + arrays["floor_support"][tail]).mean(0) / weight_np
        floor_fraction = np.maximum(arrays["floor_support"][tail], 0).mean(0) / weight_np
        vz_rms = np.sqrt((arrays["com_vz"][tail] ** 2).mean(0))
        drift = (arrays["root_z"][-1] - arrays["root_z"][tail][0]) / (arrays["time"][-1] - arrays["time"][tail][0])
        limits = {
            "support_balanced": bool((np.abs(support_ratio - 1) < 0.05).all()),
            "vertical_velocity_settled": bool((vz_rms < 0.05).all()),
            "height_drift_settled": bool((np.abs(drift) < 0.01).all()),
            "upright": bool((arrays["up_z"][tail] > 0.9).all()),
            "physical_floor_support_correct": bool((floor_fraction > 0.95).all() if args.rigid else (floor_fraction < 0.05).all()),
            "no_large_depth_oscillation": bool((arrays["depth"][tail].std(0).max(1) < 0.005).all()),
            "joints_settled": bool((arrays["joint_speed"][tail].mean(0) < 0.2).all()),
        }
        if args.lock_joints:
            # This fixture tests bearing with imposed geometry, not free-joint
            # stability. TGS position corrections need not zero solver velocity.
            limits.pop("joints_settled")
            limits["locked_pose_held"] = bool((arrays["joint_error"][tail] < 0.001).all())
        results["fixed_target_standing"] = {"passed": all(limits.values()), "checks": limits, "metrics": {
            "weight_N": weight_np.tolist(), "support_weight_ratio": support_ratio.tolist(),
            "hard_bottom_weight_fraction": floor_fraction.tolist(), "com_vertical_speed_rms_m_s": vz_rms.tolist(),
            "height_drift_m_s": drift.tolist(), "final_base_height_m": arrays["root_z"][-1].tolist(),
            "final_foot_depth_m": arrays["depth"][-1].tolist(),
            "mean_joint_rms_speed_rad_s": arrays["joint_speed"][tail].mean(0).tolist(),
            "max_joint_rms_pose_error_rad": arrays["joint_error"][tail].max(0).tolist(),
            "late_momentum_residual_N": arrays["momentum_residual"][tail].mean(0).tolist(),
        }}
        np.savez_compressed(output / "measurements.npz", **arrays, bearing_curve=np.asarray(curve), weight_N=weight_np)
        passed = all(case["passed"] for case in results.values())
        summary = {"passed": passed, "seed": args.seed, "seconds": args.seconds, "dt": cfg.sim.dt,
                   "material": args.material, "depth": args.depth, "policy": "none; fixed original default joint targets",
                   "automatic_resets": False, "results": results}
        summary["control_period_s"] = cfg.decimation * cfg.sim.dt
        summary["decimation"] = cfg.decimation
        summary["velocity_audit"] = velocity_audit(arrays, cfg.sim.dt, tail)
        summary["joint_names"] = robot.joint_names
        summary["body_names"] = robot.body_names
        summary["wrench_frame"] = "World frame, body COM; Fx,Fy,Fz,torque_x,torque_y,torque_z"
        summary["foot_names"] = [robot.body_names[i] for i in soil.feet]
        summary["sample_timing"] = (
            "Row time is interval end; soil wrench/depth are sampled at interval start. "
            "Finite differences and trapezoid velocities refer to the same interval."
        )
        summary["rigid_control"] = args.rigid
        summary["diagnostic_joint_lock"] = args.lock_joints
        summary["acceptance_scope"] = "constrained_geometry_bearing" if args.lock_joints else "free_joint_standing"
        summary["thresholds"] = {"window_s": 2, "support_weight_relative_error": 0.05,
                                 "vertical_speed_rms_m_s": 0.05, "height_drift_m_s": 0.01,
                                 "minimum_up_z": 0.9, "hard_bottom_weight_fraction": 0.05,
                                 "foot_depth_std_m": 0.005, "mean_joint_rms_speed_rad_s": 0.2,
                                 "locked_joint_rms_pose_error_rad": 0.001}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        fig.suptitle(f"{'Rigid control' if args.rigid else args.material} | "
                     f"{'Locked-joint fixture' if args.lock_joints else 'Original joint control'} | "
                     f"seed {args.seed} | {'PASS' if passed else 'FAIL'}")
        c = np.asarray(curve)
        axes[0, 0].plot(c[:, 0], c[:, 1] * 4, "o-", label="4-foot bearing capacity")
        axes[0, 0].axhline(weight_np.mean(), color="black", linestyle="--", label="mean robot weight")
        axes[0, 0].set(xlabel="Penetration (m)", ylabel="Force (N)")
        axes[0, 0].legend()
        axes[0, 1].plot(arrays["time"], (arrays["soil_support"] + arrays["floor_support"]) / weight_np)
        axes[0, 1].axhline(1, color="black", linestyle="--")
        axes[0, 1].set(xlabel="Time (s)", ylabel="Total support / weight")
        axes[1, 0].plot(arrays["time"], arrays["root_z"])
        axes[1, 0].set(xlabel="Time (s)", ylabel="Base height (m)")
        axes[1, 1].plot(arrays["time"], arrays["depth"].mean(-1))
        axes[1, 1].axhline(0 if args.rigid else args.depth, color="black", linestyle="--", label="hard bottom")
        axes[1, 1].set(xlabel="Time (s)", ylabel="Mean foot penetration (m)")
        axes[1, 1].legend()
        fig.savefig(output / "physics.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        env_id, joint_id = np.unravel_index(np.argmax(arrays["joint_vel"][tail].std(0)), (args.num_envs, 12))
        last = slice(-round(0.2 / cfg.sim.dt), None)
        times = arrays["time"][last]
        axes[0, 0].plot(times, arrays["joint_pos"][last, env_id, joint_id])
        axes[0, 0].set(xlabel="Time (s)", ylabel="Joint angle (rad)", title=f"env {env_id}: {robot.joint_names[joint_id]}")
        for key, label in (("joint_vel", "reported at interval end"), ("joint_fd_vel", "position difference"),
                           ("joint_trapezoid_vel", "reported interval average")):
            axes[0, 1].plot(times, arrays[key][last, env_id, joint_id], label=label)
        axes[0, 1].set(xlabel="Time (s)", ylabel="Joint velocity (rad/s)")
        axes[0, 1].legend()
        for key, label in (("foot_vel", "reported"), ("foot_fd_vel", "position difference")):
            axes[1, 0].plot(arrays["time"], np.sqrt((arrays[key][:, env_id] ** 2).mean((1, 2))), label=label)
        axes[1, 0].set(xlabel="Time (s)", ylabel="Foot velocity RMS (m/s)", title=f"env {env_id}")
        axes[1, 0].legend()
        axes[1, 1].plot(arrays["time"], arrays["drag_work_configuration"].cumsum(0))
        axes[1, 1].axhline(0, color="black", linestyle="--")
        axes[1, 1].set(xlabel="Time (s)", ylabel="Horizontal drag work estimate (J)")
        fig.suptitle(f"Velocity/work audit | {'rigid' if args.rigid else 'soft'} | "
                     f"dt={cfg.sim.dt * 1000:g} ms | control={cfg.decimation * cfg.sim.dt * 1000:g} ms")
        fig.savefig(output / "velocity_audit.png", dpi=160)
        plt.close(fig)
        print(f"[SOIL PHYSICS {'PASS' if passed else 'FAIL'}] {json.dumps(results['fixed_target_standing'])}", flush=True)
        print(f"[Soil physics] Results: {output}", flush=True)
        diagnostic_status(passed)
        if not passed:
            raise RuntimeError("Physical acceptance failed; inspect summary.json before training")
    except Exception:
        diagnostic_status(False)
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def rollout_report(cases, output):
    """Compare frozen-policy runs, checking configuration and initial-state parity."""
    import json
    import numpy as np
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
    Loader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice",
                           lambda loader, node: slice(*loader.construct_sequence(node)))
    initial_ray_max_difference = 0.0
    for seed in sorted({case["seed"] for case in cases}):
        pair = [case for case in cases if case["seed"] == seed]
        assert len(pair) == 2 and {case["rigid_control"] for case in pair} == {False, True}
        configs = [yaml.load((Path(case["run_directory"]) / "env.yaml").read_text(), Loader=Loader) for case in pair]
        for cfg in configs:
            cfg["scene"]["terrain"].pop("soft_enabled")
            cfg["scene"]["height_scanner"].pop("mesh_prim_paths")
            cfg["scene"]["contact_forces"].pop("class_type")
            cfg["actions"]["joint_pos"].pop("class_type")
            # Populated by the production soft kernel only; verified in its own run metadata.
            for key in ("proxy_sha256", "table_sha256", "adapter_sha256", "warp_version", "proxy_coefficients"):
                cfg["scene"]["terrain"].pop(key)
        assert configs[0] == configs[1], "Unexpected task/physics difference in paired rollouts"
        agents = [yaml.load((Path(case["run_directory"]) / "agent.yaml").read_text(), Loader=Loader) for case in pair]
        assert agents[0] == agents[1], "Agent settings differ"
        for key in ("checkpoint_sha256", "schedule", "physics_dt", "control_period_s", "policy_state_sha256", "policy_noise", "rollout_source_sha256"):
            assert pair[0][key] == pair[1][key], key
        with np.load(Path(pair[0]["run_directory"]) / "measurements.npz") as a, \
                np.load(Path(pair[1]["run_directory"]) / "measurements.npz") as b:
            for key in a.files:
                if key.startswith("initial_"):
                    if key == "initial_obs_height_scan":
                        # The rigid plane and noncolliding visual mesh have different
                        # triangulations; single-precision ray hits differ by a few microns.
                        difference = float(np.abs(a[key] - b[key]).max())
                        initial_ray_max_difference = max(initial_ray_max_difference, difference)
                        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-5, err_msg=key)
                    else:
                        np.testing.assert_array_equal(a[key], b[key], err_msg=f"seed {seed}: {key}")
            np.testing.assert_array_equal(a["command"], b["command"])
            np.testing.assert_array_equal(a["action"][0], b["action"][0])
    summary = {"completed": True, "configuration_and_initial_state_parity_passed": True,
               "initial_critic_height_scan_max_difference_m": initial_ray_max_difference,
               "initial_critic_height_scan_tolerance_m": 1e-5,
               "inference_only": True, "cases": cases,
               "interpretation": "Frozen rigid-trained policy transfer, not a training comparison. "
               "Falls are recorded before autoreset. After a fall the two trajectories and reset RNG consumption diverge. "
               "Joint high-frequency power is a diagnostic flag, not a validated instability threshold."}
    summary["aggregate"] = {}
    for rigid in (True, False):
        selected = [case for case in cases if case["rigid_control"] == rigid]
        aggregate = {
            "environments": sum(case["num_envs"] for case in selected),
            "non_timeout_resets": sum(sum(case["non_timeout_resets_per_env"]) for case in selected),
            "timeout_resets": sum(sum(case["timeout_resets_per_env"]) for case in selected),
            "tracking_rmse_vx_vy_yaw": np.sqrt(np.mean(np.array([case["tracking_rmse_vx_vy_yaw"] for case in selected]) ** 2, axis=0)).tolist(),
            "cenet_rmse_vx_vy_vz": np.sqrt(np.mean(np.array([case["cenet_rmse_vx_vy_vz"] for case in selected]) ** 2, axis=0)).tolist(),
        }
        phases = {}
        before_timeout, after_timeout, floor_samples, sample_count = 0, 0, 0, 0
        for case in selected:
            with np.load(Path(case["run_directory"]) / "measurements.npz") as data:
                done = data["terminated"] | data["timeout"]
                # Keep each environment only until its first reset; no post-reset
                # states or falls can contaminate the uninterrupted transfer metrics.
                uninterrupted = done.cumsum(axis=0) == 0
                timeout_seen = data["timeout"].cumsum(axis=0) > 0
                before_timeout += int((data["terminated"] & ~timeout_seen).sum())
                after_timeout += int((data["terminated"] & timeout_seen).sum())
                floor_samples += int(((data["floor_support_fraction"] > .05) & uninterrupted).sum())
                sample_count += int(uninterrupted.sum())
                lo = 0
                for phase in case["schedule"]:
                    steps = round(phase["seconds"] / case["control_period_s"])
                    steady = slice(lo + round(1 / case["control_period_s"]), lo + steps)
                    keep = uninterrupted[steady]
                    lo += steps
                    if not keep.any():
                        continue
                    phase_data = phases.setdefault(phase["name"], {"error": [], "actual": [], "height": [], "depth": [], "up": []})
                    phase_data["error"].append((data["velocity"][steady] - data["command"][steady])[keep])
                    phase_data["actual"].append(data["velocity"][steady][keep])
                    phase_data["height"].append(data["base_height"][steady][keep])
                    phase_data["depth"].append(data["foot_depth"][steady].clip(0)[keep])
                    phase_data["up"].append(data["up_z"][steady][keep])
        aggregate.update({
            "non_timeout_resets_before_first_timeout": before_timeout,
            "non_timeout_resets_at_or_after_first_timeout": after_timeout,
            "pre_first_reset_floor_support_over_5pct_sample_fraction": floor_samples / max(sample_count, 1),
            "maximum_phase_high_frequency_power_fraction": max(max(phase["joint_speed_top_20pct_frequency_power_fraction"])
                                                                 for case in selected for phase in case["phase_metrics"]),
            "soil_reset_isolation_checks_passed": sum(case["soil_reset_isolation_checks_passed"] for case in selected),
            "phases_before_first_reset": {},
        })
        for name, phase in phases.items():
            values = {key: np.concatenate(value) for key, value in phase.items()}
            aggregate["phases_before_first_reset"][name] = {
                "environment_control_samples": len(values["height"]),
                "tracking_rmse_vx_vy_yaw": np.sqrt((values["error"] ** 2).mean(0)).tolist(),
                "actual_mean_vx_vy_yaw": values["actual"].mean(0).tolist(),
                "mean_base_height_m": float(values["height"].mean()),
                "mean_max_foot_depth_m": float(values["depth"].max(-1).mean()),
                "maximum_foot_depth_m": float(values["depth"].max()),
                "minimum_up_z": float(values["up"].min()),
            }
        summary["aggregate"]["rigid" if rigid else "soft"] = aggregate
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    for case in cases:
        with np.load(Path(case["run_directory"]) / "measurements.npz") as data:
            t = data["time"]
            label = f"{'rigid' if case['rigid_control'] else 'soft'} seed {case['seed']}"
            for j, axis in enumerate(axes.flat[:3]):
                axis.plot(t, data["velocity"][..., j].mean(1), label=label, linewidth=1)
                if case == cases[0]:
                    axis.plot(t, data["command"][:, 0, j], "k--", label="command")
            axes.flat[3].plot(t, data["base_height"].mean(1), label=label)
            axes.flat[4].plot(t, data["foot_depth"].clip(0).max(-1).mean(1), label=label)
            axes.flat[5].plot(t, data["terminated"].sum(1).cumsum(), label=label)
    for axis, ylabel in zip(axes.flat, ("Body vx (m/s)", "Body vy (m/s)", "Yaw rate (rad/s)",
                                       "Base height (m)", "Mean maximum foot depth (m)", "Cumulative non-timeout resets")):
        axis.set(xlabel="Time (s)", ylabel=ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Same frozen DreamWaQ policy | paired rigid / soft flat terrain | pre-reset state")
    fig.savefig(output / "rollout_comparison.png", dpi=160)
    plt.close(fig)
    print(f"[ROLLOUT COMPARISON] configuration / initial state parity PASS: {output}", flush=True)


def rollout(arguments):
    """Inference-only, matched rigid/soft command schedule with native substep recording."""
    import argparse
    from datetime import datetime
    import hashlib
    import json
    import math
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=rollout.__doc__)
    # AppLauncher pre-parses the parser while adding its options, including --help.
    parser.add_argument("--checkpoint")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--phase_seconds", type=float, default=4.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--command_scale", type=float, default=1.0, help="Scale the fixed evaluation velocity commands; 2 gives 0.6 m/s forward, 0.4 m/s lateral and 0.8 rad/s yaw")
    parser.add_argument("--physics_dt", type=float, default=0.0003125)
    parser.add_argument("--seeds", default="1", help="Comma-separated seeds for the paired batch")
    parser.add_argument("--seed", type=int, default=1, help="Single-case seed (batch uses --seeds)")
    parser.add_argument("--single", action="store_true", help="Evaluate one checkpoint/configuration instead of a rigid/soft batch")
    parser.add_argument("--original_rough", action="store_true", help="With --single, use the original rough Play task and its native timing")
    parser.add_argument("--mixed_stage", choices=("hard", "soft"), help="Fixed-level holdout on the complete mixed map")
    parser.add_argument("--soft_level", type=int, default=0)
    parser.add_argument("--rigid", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", help="New output directory; existing directories are never overwritten")
    parser.add_argument("--report", help="Rebuild a report from an existing paired batch without simulation")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted batch with --out; completed cases are reused")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(arguments)
    if args.report:
        source = Path(args.report)
        output = Path(args.out).resolve() if args.out else source.with_name(datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_rollout_report")
        output.mkdir(parents=True, exist_ok=False)
        rollout_report(json.loads((source / "comparison.json").read_text())["cases"], output)
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error("Checkpoint must be an existing local file")
    if args.mixed_stage and (not args.single or args.original_rough or args.rigid or not 0 <= args.soft_level < 10):
        parser.error("Mixed-stage holdouts require --single, level 0..9, and no original_rough/rigid selector")
    if args.original_rough:
        if not args.single or args.rigid or any(item == "--physics_dt" or item.startswith("--physics_dt=") for item in arguments):
            parser.error("--original_rough requires --single and retains native rough-task timing; omit --rigid and --physics_dt")
        args.physics_dt = 0.005
    if (not math.isfinite(args.physics_dt) or not 0 < args.physics_dt <= 0.005
            or not math.isclose(round(0.02 / args.physics_dt) * args.physics_dt, 0.02, abs_tol=1e-10)
            or not math.isclose(round(0.015 / args.physics_dt) * args.physics_dt, 0.015, abs_tol=1e-10)):
        parser.error("Timestep must divide both the 20 ms control period and 15 ms contact history")
    if (args.num_envs < 2 or args.cycles < 1 or not math.isfinite(args.phase_seconds) or args.phase_seconds < 2
            or not math.isclose(round(args.phase_seconds / 0.02) * 0.02, args.phase_seconds, abs_tol=1e-10)):
        parser.error("Use >=2 environments, >=1 cycle and phase_seconds >=2, divisible by 20 ms")
    if not math.isfinite(args.command_scale) or args.command_scale <= 0:
        parser.error("--command_scale must be positive and finite")
    try:
        seeds = [int(seed) for seed in args.seeds.split(",")]
        assert seeds and len(set(seeds)) == len(seeds)
    except (ValueError, AssertionError):
        parser.error("--seeds requires unique comma-separated integer seeds")
    experiment = "go2_dreamwaq_forgetting" if args.original_rough else "go2_dreamwaq_soft_rollout"
    output = Path(args.out).resolve() if args.out else ROOT / "logs/rsl_rl/isaaclab" / experiment / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + (f"_{'rough' if args.original_rough else 'rigid' if args.rigid else 'soft'}_seed{args.seed}" if args.single else "_paired")
    )
    if args.resume and (not args.out or args.single or not output.is_dir()):
        parser.error("--resume requires an existing --out batch directory")
    output.mkdir(parents=True, exist_ok=args.resume)
    if not args.single:
        import subprocess
        cases = []
        saved_cases = (json.loads((output / "comparison.json").read_text()).get("cases", [])
                       if args.resume and (output / "comparison.json").is_file() else [])
        for seed in seeds:
            for rigid in (True, False):
                case_dir = output / f"seed{seed}_{'rigid' if rigid else 'soft'}"
                if args.resume and not (case_dir / "summary.json").is_file():
                    # A previous successful retry may have a noncanonical directory.
                    saved = [case for case in saved_cases if case["seed"] == seed and case["rigid_control"] == rigid
                             and (Path(case["run_directory"]) / "summary.json").is_file()]
                    if saved:
                        case_dir = Path(saved[-1]["run_directory"])
                if args.resume and (case_dir / "summary.json").is_file():
                    summary = json.loads((case_dir / "summary.json").read_text())
                    assert summary["completed"] and summary["policy_unchanged"]
                    assert summary["seed"] == seed and summary["rigid_control"] == rigid
                    assert summary["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    assert summary["num_envs"] == args.num_envs and summary["physics_dt"] == args.physics_dt
                    assert len(summary["schedule"]) == 5 * args.cycles
                    assert all(phase["seconds"] == args.phase_seconds for phase in summary["schedule"])
                    assert summary.get("command_scale", 1.0) == args.command_scale
                    cases.append({"run_directory": str(case_dir), **summary})
                    print(f"[Rollout] Reusing completed {case_dir.name}", flush=True)
                    continue
                if case_dir.exists():
                    # Preserve partial measurements/logs from interrupted processes.
                    case_dir = output / (case_dir.name + "_retry_" + datetime.now().strftime("%H%M%S_%f"))
                command = [sys.executable, __file__, "rollout", "--single", "--checkpoint", str(checkpoint),
                           "--num_envs", str(args.num_envs), "--phase_seconds", str(args.phase_seconds),
                           "--cycles", str(args.cycles), "--physics_dt", str(args.physics_dt),
                           "--command_scale", str(args.command_scale),
                           "--seed", str(seed), "--out", str(case_dir), "--device", args.device]
                if args.headless:
                    command.append("--headless")
                if rigid:
                    command.append("--rigid")
                print(f"[Rollout] Starting {case_dir.name}: {output}", flush=True)
                with (output / f"{case_dir.name}.log").open("w") as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode or not (case_dir / "summary.json").is_file():
                    raise RuntimeError(f"Rollout failed; inspect {output / (case_dir.name + '.log')}")
                summary = json.loads((case_dir / "summary.json").read_text())
                cases.append({"run_directory": str(case_dir), **summary})
                (output / "comparison.json").write_text(json.dumps({"completed": False, "cases": cases}, indent=2) + "\n")
                print(f"[Rollout] {case_dir.name}: resets={summary['non_timeout_resets_per_env']}, "
                      f"tracking RMSE={summary['tracking_rmse_vx_vy_yaw']}", flush=True)
        rollout_report(cases, output)
        return
    rollout_source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    app = AppLauncher(args).app
    env = None
    try:
        import time
        import numpy as np
        import torch
        import gymnasium as gym
        import soft_terrain as soft
        from isaaclab.envs.mdp.commands import UniformVelocityCommand
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        schedule = [("stand", [0., 0., 0.]), ("forward", [0.3, 0., 0.]),
                    ("lateral", [0., 0.2, 0.]), ("turn", [0., 0., 0.4]), ("stand_again", [0., 0., 0.])]
        schedule = [(name, [value * args.command_scale for value in command]) for name, command in schedule]
        phase_steps = round(args.phase_seconds / 0.02)
        control_steps = len(schedule) * phase_steps * args.cycles

        class RolloutVelocityCommand(UniformVelocityCommand):
            # Set commands *before* the stock observation computation, including on autoreset.
            # No extra observation computation is allowed: it would advance CENet's history.
            def _resample_command(self, env_ids):
                self._update_command()

            def _update_command(self):
                phase = min(self._env.common_step_counter // phase_steps, len(schedule) * args.cycles - 1)
                self.vel_command_b[:] = self.vel_command_b.new_tensor(schedule[phase % len(schedule)][1])

        cfg = soft.Go2MixedEnvCfg() if args.mixed_stage else soft.Go2DreamWaQRoughEnvCfg_PLAY() if args.original_rough else soft.Go2SoftPlayEnvCfg()
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
        if args.mixed_stage:
            cfg.scene.terrain.training_stage = args.mixed_stage
            cfg.scene.terrain.physics_substeps = round(0.02 / args.physics_dt)
            cfg.scene.terrain.soft_level_ceiling = args.soft_level
            cfg.scene.terrain.terrain_generator.seed = args.seed
            cfg.curriculum.terrain_levels = None
            cfg.events.base_external_force_torque = cfg.events.push_robot = None
            for group in (cfg.observations.policy, cfg.observations.prop_history, cfg.observations.height_scan):
                group.enable_corruption = False
            task = "Isaac-Velocity-Rough-Go2-DreamWaQ-Mixed-v0"
        elif args.original_rough:
            cfg.scene.terrain.terrain_generator.seed = args.seed
            task = "Isaac-Velocity-Rough-Go2-DreamWaQ-Play-v0"
            assert cfg.sim.dt == .005 and cfg.decimation == 4 and cfg.scene.contact_forces.history_length == 3
        else:
            cfg.sim.dt = args.physics_dt
            cfg.decimation = cfg.sim.render_interval = round(0.02 / cfg.sim.dt)
            cfg.scene.contact_forces.update_period = cfg.sim.dt
            cfg.scene.contact_forces.history_length = round(0.015 / cfg.sim.dt)
            cfg.scene.height_scanner.update_period = 0.02
            cfg.scene.terrain.soft_enabled = not args.rigid
            task = "Isaac-Velocity-Flat-Go2-DreamWaQ-Soft-Play-v0"
        cfg.commands.base_velocity.class_type = RolloutVelocityCommand
        cfg.commands.base_velocity.heading_command = False
        cfg.commands.base_velocity.ranges.heading = None
        cfg.commands.base_velocity.rel_heading_envs = cfg.commands.base_velocity.rel_standing_envs = 0.0
        # Episode timeouts stay at the original duration. They are reported separately from falls.
        native_env = gym.make(task, cfg=cfg)
        if args.mixed_stage:
            terrain = native_env.unwrapped.scene.terrain
            index = torch.arange(args.num_envs, device=args.device)
            if args.mixed_stage == "hard":
                terrain.terrain_types[:] = index % terrain.terrain_origins.shape[1]
                terrain.terrain_levels[:] = index.new_tensor([0, 2, 4])[(index // terrain.terrain_origins.shape[1]) % 3]
                terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            else:
                terrain.terrain_levels[:] = args.soft_level
                terrain.soft_tile_ids[:] = index % terrain.cfg.soft_columns
                terrain.env_origins[:] = terrain.soft_tile_origins[terrain.soft_tile_ids]
                terrain.soft_origins.copy_(terrain.env_origins)
        env = RslRlVecEnvWrapper(native_env)
        raw, robot = env.unwrapped, env.unwrapped.scene["robot"]
        sensor = raw.scene["contact_forces"]
        soil = getattr(raw.action_manager.get_term("joint_pos"), "soil", None)
        if args.original_rough:
            assert soil is None, "Original rough evaluation must use only native rigid contact"
        feet = [robot.body_names.index(f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")]
        masses = robot.root_physx_view.get_masses().to(raw.device)
        weight = masses.sum(-1) * -cfg.sim.gravity[2]
        agent_cfg = soft.Go2SoftRunnerCfg()
        agent_cfg.seed, agent_cfg.device = args.seed, args.device
        baseline_agent = soft.Go2DreamWaQRoughPPORunnerCfg().to_dict()
        candidate_agent = agent_cfg.to_dict()
        candidate_agent.pop("init_checkpoint")
        for key in ("experiment_name", "seed", "device"):
            baseline_agent.pop(key)
            candidate_agent.pop(key)
        assert candidate_agent == baseline_agent, "Policy/PPO settings changed"
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_optimizer=False)
        policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        policy = runner.get_inference_policy(device=env.device)

        def policy_hash():
            digest = hashlib.sha256()
            for key, value in sorted(policy_nn.state_dict().items()):
                digest.update(key.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        original_policy_hash = policy_hash()
        initial = {"initial_root": robot.data.root_state_w.clone(), "initial_joint": robot.data.joint_pos.clone(),
                   "initial_mass": masses.clone(), "initial_com": robot.root_physx_view.get_coms().clone(),
                   "initial_inertia": robot.root_physx_view.get_inertias().clone(),
                   "initial_material": robot.root_physx_view.get_material_properties().clone()}
        terrain_mesh_hash = None
        if args.original_rough:
            terrain = raw.scene.terrain
            initial.update(initial_env_origins=raw.scene.env_origins.clone(),
                           initial_terrain_levels=terrain.terrain_levels.clone(), initial_terrain_types=terrain.terrain_types.clone())
            digest = hashlib.sha256()
            for name, mesh in sorted(terrain.meshes.items()):
                digest.update(name.encode())
                digest.update(np.ascontiguousarray(mesh.vertices).tobytes())
                digest.update(np.ascontiguousarray(mesh.faces).tobytes())
            terrain_mesh_hash = digest.hexdigest()
        # Wrapper reset already computed the initial observation. Reuse it without pushing history again.
        from tensordict import TensorDict
        obs = TensorDict(raw.obs_buf, batch_size=[args.num_envs])
        for key in ("policy", "prop_history", "height_scan", "privileged_vel"):
            initial["initial_obs_" + key] = obs[key].clone()
        dump_yaml(str(output / "env.yaml"), cfg)
        dump_yaml(str(output / "agent.yaml"), agent_cfg)
        substeps = control_steps * cfg.decimation
        # Stage admission uses control-rate locomotion metrics. Reserve full
        # substep traces for the existing numerical/physics acceptance tools.
        record_stride = cfg.decimation if args.mixed_stage else 1
        joint_samples = substeps // record_stride
        joint_dt = cfg.sim.dt * record_stride
        physics_q = torch.empty((joint_samples, args.num_envs, 12), device=raw.device)
        physics_v = torch.empty_like(physics_q)
        native_update = raw.scene.update
        native_compute = raw.termination_manager.compute
        native_reset = raw._reset_idx
        cursor, physics_calls = 0, 0
        reset_checks = 0
        snapshot = {}

        def record_reset(env_ids):
            nonlocal reset_checks
            if soil:
                untouched = torch.ones(args.num_envs, dtype=torch.bool, device=raw.device)
                untouched[env_ids] = False
                before = {name: getattr(soil, name)[untouched].clone() for name in ("hole_d", "mem", "normal_w")}
            native_reset(env_ids)
            if soil:
                assert (soil.mem[env_ids] == -1).all() and (soil.hole_d[env_ids] == 0).all()
                assert (soil.wrench[env_ids] == 0).all() and (soil.normal_w[env_ids] == 0).all()
                for name, value in before.items():
                    torch.testing.assert_close(getattr(soil, name)[untouched], value, rtol=0, atol=0)
                reset_checks += len(env_ids)

        def record_update(dt):
            nonlocal cursor, physics_calls
            native_update(dt)
            physics_calls += 1
            if physics_calls % record_stride == 0 and cursor < joint_samples:
                physics_q[cursor].copy_(robot.data.joint_pos)
                physics_v[cursor].copy_(robot.data.joint_vel)
                cursor += 1

        def record_terminal_state():
            done = native_compute()
            physical_view = getattr(sensor.contact_physx_view, "physical", sensor.contact_physx_view)
            floor = physical_view.get_net_contact_forces(dt=cfg.sim.dt).view(args.num_envs, -1, 3)[..., 2].sum(-1)
            values = {
                "velocity": torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), -1),
                "base_height": robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2],
                "up_z": -robot.data.projected_gravity_b[:, 2],
                "foot_depth": (soil.depth if soil else torch.zeros((args.num_envs, 4), device=raw.device)
                               if args.original_rough or args.mixed_stage == "hard" else (-robot.data.body_pos_w[:, feet, 2]).clamp_min(0)),
                "floor_support_fraction": floor / weight,
                "joint_velocity": robot.data.joint_vel,
                "terminated": raw.termination_manager.terminated,
                "timeout": raw.termination_manager.time_outs,
                "position": robot.data.root_pos_w - raw.scene.env_origins,
                "heading": robot.data.heading_w,
                "episode_length_steps": raw.episode_length_buf,
            }
            for name in raw.termination_manager.active_terms:
                values["termination_" + name] = raw.termination_manager.get_term(name)
            snapshot.update({key: value.clone() for key, value in values.items()})
            return done

        raw.scene.update = record_update
        raw.termination_manager.compute = record_terminal_state
        raw._reset_idx = record_reset
        records = {}
        started = time.perf_counter()
        for step in range(control_steps):
            with torch.inference_mode():
                command = raw.command_manager.get_command("base_velocity").clone()
                expected = command.new_tensor(schedule[(step // phase_steps) % len(schedule)][1]).expand_as(command)
                torch.testing.assert_close(command, expected, rtol=0, atol=0)
                # Validate the command actually visible to the actor, not just the manager buffer.
                torch.testing.assert_close(obs["policy"][:, 6:9], command, rtol=0, atol=0)
                # The original CENet samples a latent even during inference. Keep that
                # behavior, but give paired cases the same draws independently of resets.
                noise_devices = [torch.device(env.device).index or 0] if str(env.device).startswith("cuda") else []
                with torch.random.fork_rng(devices=noise_devices):
                    torch.manual_seed(args.seed * 1_000_003 + step)
                    actions = policy(obs)
                # Reuse the estimator output actually consumed by the actor, with no extra draw.
                estimate = policy_nn.last_cenet_outputs["vel_hat"]
                before_velocity = robot.data.root_lin_vel_b.clone()
                obs, reward, done, _ = env.step(actions)
                policy_nn.reset(done)
                values = {**snapshot, "command": command, "action": actions, "reward": reward,
                          "cenet_estimate": estimate, "cenet_true_velocity": before_velocity}
                assert torch.stack([torch.isfinite(value).all() for value in values.values()]).all(), "Nonfinite rollout data"
                for key, value in values.items():
                    records.setdefault(key, []).append(value.detach().cpu().numpy().copy())
            if (step + 1) % phase_steps == 0:
                print(f"[Rollout] {args.mixed_stage or ('rough' if args.original_rough else 'rigid' if args.rigid else 'soft')} seed={args.seed} "
                      f"t={(step + 1) * 0.02:.1f}s phase={schedule[(step // phase_steps) % 5][0]} "
                      f"base_z={snapshot['base_height'].tolist()}", flush=True)
        wall_seconds = time.perf_counter() - started
        assert cursor == joint_samples and policy_hash() == original_policy_hash
        raw.scene.update, raw.termination_manager.compute, raw._reset_idx = native_update, native_compute, native_reset
        arrays = {key: np.stack(values) for key, values in records.items()}
        arrays.update({key: value.cpu().numpy() for key, value in initial.items()})
        q, v = physics_q.cpu().numpy(), physics_v.cpu().numpy()
        assert np.isfinite(q).all() and np.isfinite(v).all()
        fd_error = np.diff(q, axis=0) / joint_dt - v[1:]
        fd_valid = np.ones(fd_error.shape[:2], dtype=bool)
        # The first interval after each autoreset contains a position discontinuity.
        for control, environment in zip(*np.nonzero(arrays["terminated"] | arrays["timeout"])):
            discontinuity = (control + 1) * cfg.decimation // record_stride - 1
            if discontinuity < len(fd_valid):
                fd_valid[discontinuity, environment] = False
        arrays.update(time=(np.arange(control_steps) + 1) * 0.02, physics_joint_pos=q, physics_joint_vel=v)
        np.savez_compressed(output / "measurements.npz", **arrays)
        # Drop the first second of each phase when assessing tracking; still retain every reset.
        phase_metrics = []
        for phase in range(len(schedule) * args.cycles):
            lo, hi = phase * phase_steps, (phase + 1) * phase_steps
            steady = slice(lo + round(1 / 0.02), hi)
            native_slice = slice((lo + round(1 / 0.02)) * cfg.decimation // record_stride, hi * cfg.decimation // record_stride)
            block = v[native_slice]
            demeaned = block - block.mean(0)
            power = np.abs(np.fft.rfft(demeaned, axis=0)) ** 2
            frequency = np.fft.rfftfreq(len(block), joint_dt)
            hf_ratio = power[frequency >= 0.8 * (0.5 / joint_dt)].sum((0, 2)) / power.sum((0, 2)).clip(1e-30)
            phase_metrics.append({"phase": schedule[phase % 5][0], "cycle": phase // 5 + 1,
                "tracking_rmse_vx_vy_yaw": np.sqrt(((arrays["velocity"][steady] - arrays["command"][steady]) ** 2).mean((0, 1))).tolist(),
                "non_timeout_resets_per_env": arrays["terminated"][lo:hi].sum(0).tolist(),
                "base_height_mean_per_env_m": arrays["base_height"][steady].mean(0).tolist(),
                "minimum_up_z_per_env": arrays["up_z"][lo:hi].min(0).tolist(),
                "max_foot_depth_per_env_m": arrays["foot_depth"][lo:hi].max((0, 2)).tolist(),
                "joint_speed_rms_per_env_rad_s": np.sqrt((block ** 2).mean((0, 2))).tolist(),
                "joint_speed_top_20pct_frequency_power_fraction": hf_ratio.tolist(),
                "joint_speed_peak_per_env_rad_s": np.abs(block).max((0, 2)).tolist()})
        all_done = arrays["terminated"] | arrays["timeout"]
        episode_lengths = arrays["episode_length_steps"][all_done] * raw.step_dt
        summary = {"completed": True, "inference_only": True, "rigid_control": args.rigid or args.original_rough or args.mixed_stage == "hard", "seed": args.seed,
            "task": task, "original_rough": args.original_rough, "terrain_mesh_sha256": terrain_mesh_hash,
            "num_envs": args.num_envs, "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "rollout_source_sha256": rollout_source_hash,
            "policy_state_sha256": original_policy_hash, "policy_unchanged": True,
            "policy_noise": "Original sampled CENet latent; isolated Torch RNG seeded seed*1000003+control_step, shared by paired cases",
            "physics_dt": cfg.sim.dt, "control_period_s": raw.step_dt,
            "command_scale": args.command_scale,
            "contact_history_s": cfg.scene.contact_forces.history_length * cfg.sim.dt,
            "joint_sample_period_s": joint_dt,
            "schedule": [{"name": name, "command": command, "seconds": args.phase_seconds} for name, command in schedule] * args.cycles,
            "simulated_seconds": control_steps * 0.02, "rollout_wall_seconds": wall_seconds,
            "simulated_seconds_per_wall_second": control_steps * 0.02 / wall_seconds,
            "non_timeout_resets_per_env": arrays["terminated"].sum(0).tolist(),
            "timeout_resets_per_env": arrays["timeout"].sum(0).tolist(),
            "total_reward_per_env": arrays["reward"].sum(0).tolist(),
            "mean_reward_per_simulated_second": float(arrays["reward"].sum(0).mean() / (control_steps * raw.step_dt)),
            "mean_completed_episode_seconds": float(episode_lengths.mean()) if len(episode_lengths) else None,
            "first_non_timeout_reset_seconds_per_env": [float((np.flatnonzero(arrays["terminated"][:, i])[0] + 1) * raw.step_dt)
                                                       if arrays["terminated"][:, i].any() else None for i in range(args.num_envs)],
            "soil_reset_isolation_checks_passed": reset_checks,
            "joint_velocity_position_difference_rmse_rad_s": np.sqrt((fd_error[fd_valid] ** 2).mean()).item(),
            "termination_counts": {key: value.sum(0).tolist() for key, value in arrays.items() if key.startswith("termination_")},
            "tracking_rmse_vx_vy_yaw": np.sqrt(((arrays["velocity"] - arrays["command"]) ** 2).mean((0, 1))).tolist(),
            "cenet_rmse_vx_vy_vz": np.sqrt(((arrays["cenet_estimate"] - arrays["cenet_true_velocity"]) ** 2).mean((0, 1))).tolist(),
            "fraction_samples_floor_support_over_5pct_weight": None if args.original_rough else (arrays["floor_support_fraction"] > .05).mean(0).tolist(),
            "phase_metrics": phase_metrics,
            "timing_note": "State/termination recorded before autoreset; CENet before action, tracking at interval end. "
                           "Foot depth is the kernel input from the last physics substep. "
                           "Joint trace sampling period is joint_sample_period_s (20 ms for stage holdouts; every substep otherwise). "
                           "Runtime includes diagnostic recording; it is not training throughput. "
                           "For original rough terrain and hard holdouts, foot_depth is an unused zero placeholder; floor_support_fraction means native terrain support."}
        if args.mixed_stage:
            summary["mixed_stage"] = args.mixed_stage
            # Signed displacement along the commanded world direction. Exclude
            # reset jumps and the first second of each fixed command phase.
            moving = np.linalg.norm(arrays["command"][..., :2], axis=-1) > 0.1
            steady = np.arange(control_steps) % phase_steps >= 50
            valid = moving & steady[:, None]
            error = arrays["velocity"][..., :2] - arrays["command"][..., :2]
            displacement = np.diff(arrays["position"][..., :2], axis=0)
            command = arrays["command"][1:, :, :2]
            heading = arrays["heading"][:-1]
            world = np.stack((np.cos(heading) * command[..., 0] - np.sin(heading) * command[..., 1],
                              np.sin(heading) * command[..., 0] + np.cos(heading) * command[..., 1]), -1)
            speed = np.linalg.norm(world, axis=-1)
            aligned = (displacement * world).sum(-1) / speed.clip(1e-8)
            travel_valid = valid[1:] & ~all_done[:-1] & ~all_done[1:]
            # Admission requires a working forward gait. Lateral commands still
            # contribute to XY tracking, but do not dilute forward displacement.
            travel_valid &= (command[..., 0] > 0.1) & (np.abs(command[..., 1]) < 1e-6)
            def metrics(mask):
                selected = valid[:, mask]
                travel = travel_valid[:, mask]
                return {"tracking_rmse_xy": float(np.sqrt((error[:, mask] ** 2).sum(-1)[selected].mean())),
                        "fall_fraction": float(arrays["terminated"][:, mask].any(0).mean()),
                        "progress": float(aligned[:, mask][travel].sum() / (speed[:, mask][travel].sum() * 0.02)) if travel.any() else 0.0}
            summary["stage_metrics"] = metrics(np.ones(args.num_envs, dtype=bool))
            summary["scenario_metrics"] = ({name: metrics(mask.cpu().numpy()) for name, mask in terrain.scenario_masks.items()}
                                           if args.mixed_stage == "soft" else {})
            summary["soft_level"] = args.soft_level
            summary["soft_num_rows"] = cfg.scene.terrain.soft_num_levels
            summary["soft_num_levels"] = cfg.scene.terrain.soft_num_levels
            summary["soft_tiles"] = cfg.scene.terrain.resolved_soft_tiles
            summary["terrain_spec"] = soft.soil_curriculum_spec(cfg.scene.terrain)
            summary["evaluation_suite"] = {"protocol": "fixed-stage-v2", "seed": args.seed, "num_envs": args.num_envs,
                "hard_levels": [0, 2, 4], "schedule": summary["schedule"], "physics_dt": cfg.sim.dt,
                "rough_mesh": cfg.scene.terrain.rough_mesh_sha256}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[ROLLOUT COMPLETE] {output}", flush=True)
        diagnostic_status(True)
    except Exception:
        diagnostic_status(False)
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def soil_curriculum_check(arguments):
    """Check spatial soil sampling, native curriculum, reset and a discarded PPO update."""
    import argparse
    import json
    import tempfile
    from datetime import datetime
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=soil_curriculum_check.__doc__)
    parser.add_argument("--num_envs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(arguments)
    if args.num_envs < 25:
        parser.error("Use >=25 robots to cover every difficulty/scenario tile")
    app, env = AppLauncher(args).app, None
    directory = ROOT / "logs/rsl_rl/isaaclab/go2_dreamwaq_soft_curriculum_check" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        import torch
        import soft_terrain as soft
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        cfg = soft.Go2SoilCurriculumEnvCfg()
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
        env = RslRlVecEnvWrapper(soft.make_env(cfg))
        raw, terrain = env.unwrapped, env.unwrapped.scene.terrain
        soil = raw.action_manager.get_term("joint_pos").soil
        assert terrain.num_rough == 0 and "terrain" not in terrain.meshes
        assert "stage_boundary" not in raw.termination_manager.active_terms
        assert cfg.scene.terrain.resolved_soft_tiles == 25
        assert cfg.decimation == 64 and abs(cfg.sim.dt - .0003125) < 1e-12 and raw.step_dt == .02
        assert cfg.scene.contact_forces.history_length == 48
        assert terrain.terrain_levels.min() >= 0 and terrain.terrain_levels.max() <= 1
        ids = torch.arange(args.num_envs, device=env.device)
        columns = ids % 5
        zeros = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        with torch.inference_mode():
            # Query every tile's real height/material and its rigid frame.
            centres = terrain.soft_tile_origins.clone()
            heights, normals, materials = terrain.sample_surface(centres)
            torch.testing.assert_close(heights, centres[:, 2], atol=1e-5, rtol=0)
            assert (materials == 1).all()  # Public surface query returns a soft/hard mask.
            material_grid = torch.as_tensor(terrain.surface_material.numpy(), device=env.device)
            x = (torch.arange(25, device=env.device) // 5) * 80 + 40
            y = (torch.arange(25, device=env.device) % 5) * 80 + 40
            torch.testing.assert_close(material_grid[x, y].long(), torch.arange(25, device=env.device) // 5 + 1)
            frame = centres.clone()
            frame[:, 0] += 3.5
            h, _, material = terrain.sample_surface(frame)
            assert not material.any() and h.abs().max() < 1e-6
            assert torch.isfinite(normals).all()
            torch.testing.assert_close(torch.tensor(terrain.soft_bearing_scales, device=env.device),
                                       torch.tensor([4 ** (1 - row / 4) for row in range(5)], device=env.device))
            for row in range(5):
                scale = .1 + .9 * row / 4
                assert abs(float(heights[row * 5 + 1]) - .30 * scale) < 1e-5
                assert abs(float(heights[row * 5 + 2]) + .20 * scale) < 1e-5
            # Exercise actual native distance decisions: upgrade, downgrade,
            # and hold. Termination/contact flags are deliberately irrelevant.
            terrain.terrain_levels[:] = 2
            terrain.update_soft_origins(ids, zeros, zeros)
            selected = ids[:3]
            state = raw.scene["robot"].data.root_state_w[:3].clone()
            state[:, :3] = terrain.env_origins[:3]
            state[:, 0] += torch.tensor([4.1, .1, 1.], device=env.device)
            state[:, 2] += .4
            raw.scene["robot"].write_root_state_to_sim(state, env_ids=selected)
            raw.command_manager.get_command("base_velocity")[:3, :2] = torch.tensor([[.4, 0], [.4, 0], [0, 0]], device=env.device)
            raw.episode_length_buf[:3] = 100
            raw.termination_manager.terminated[:3] = True
            soft.mixed_terrain_levels(raw, selected)
            torch.testing.assert_close(terrain.terrain_levels[:3], torch.tensor([3, 1, 2], device=env.device))
            terrain.terrain_levels[:] = 0
            terrain.update_soft_origins(ids, zeros, ~zeros)
            assert (terrain.terrain_levels == 0).all()
            terrain.terrain_levels[:] = 4
            terrain.update_soft_origins(ids, ~zeros, zeros)
            assert (terrain.terrain_levels >= 0).all() and (terrain.terrain_levels <= 4).all()
            assert torch.equal(terrain.soft_tile_ids % 5, columns)
            terrain.terrain_levels[:] = (ids // 5) % 5
            terrain.update_soft_origins(ids, zeros, zeros)
            torch.testing.assert_close(terrain.env_origins, terrain.soft_tile_origins[terrain.soft_tile_ids])
            raw.episode_length_buf.zero_()
            env.reset()
            soil.hole_d[:] = .02
            untouched = soil.hole_d[1:].clone()
            raw._reset_idx(ids[:1])
            assert not soil.hole_d[0].any() and (soil.mem[0] == -1).all()
            torch.testing.assert_close(soil.hole_d[1:], untouched)
            env.reset()
        dump_yaml(str(directory / "params/env.yaml"), cfg)
        agent = soft.Go2SoilCurriculumRunnerCfg().to_dict()
        original = soft.Go2DreamWaQRoughPPORunnerCfg().to_dict()
        for key in ("policy", "algorithm", "num_steps_per_env"):
            assert agent[key] == original[key]
        dump_yaml(str(directory / "params/agent.yaml"), agent)
        runner = soft.make_runner(env, agent, log_dir=str(directory), device=env.device)
        assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
        with torch.inference_mode():
            obs = env.get_observations()
            for _ in range(runner.num_steps_per_env):
                obs, reward, done, extras = env.step(runner.alg.act(obs))
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
                runner.alg.process_env_step(obs, reward, done, extras)
            runner.alg.compute_returns(obs)
        losses = runner.alg.update()
        assert all(torch.isfinite(torch.as_tensor(value)).all() for value in losses.values())
        with tempfile.TemporaryDirectory(prefix="soil_curriculum_checkpoint_") as temporary:
            path = str(Path(temporary) / "state.pt")
            levels = terrain.terrain_levels.clone()
            runner.save(path)
            terrain.terrain_levels.zero_()
            runner.load(path, map_location=env.device)
            torch.testing.assert_close(terrain.terrain_levels, levels)
            assert runner.alg.optimizer.state
            torch.testing.assert_close(terrain.env_origins, terrain.soft_tile_origins[terrain.soft_tile_ids])
        result = {"completed": True, "diagnostic_only": True, "formal_training_started": False,
                  "rows": 5, "columns": 5, "substeps": 64, "num_envs": args.num_envs,
                  "checks": ["initial_levels_0_1", "all_tile_heights_materials", "rigid_frames", "native_upgrade_downgrade_hold",
                             "min_and_max_levels", "fixed_scenario_columns", "partial_footprint_reset", "unchanged_model_PPO",
                             "discarded_PPO_update", "checkpoint_levels_optimizer"], "losses": losses}
        (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[SOIL CURRICULUM CHECK PASS] {directory}", flush=True)
        diagnostic_status(True)
    except Exception:
        diagnostic_status(False)
        import traceback
        (directory / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def main():
    mode, arguments = sys.argv[1], sys.argv[2:]
    if mode == "mixed-eval":
        stage_evaluate(arguments)
        return
    if mode == "physics" and ("--sweep_dt" in arguments or "--report_sweep" in arguments):
        physics(arguments)
        return
    if mode in ("soil-check", "check", "mixed-check", "mixed-stage-check", "mixed-profile", "physics", "rollout") and (mode != "rollout" or "--single" in arguments) and "DREAMWAQ_DIAGNOSTIC_STATUS" not in os.environ:
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory(prefix="dreamwaq_check_") as directory:
            status = Path(directory) / "status"
            child_env = dict(os.environ, DREAMWAQ_DIAGNOSTIC_STATUS=str(status))
            child = subprocess.run([sys.executable, __file__, mode, *arguments], env=child_env)
            if child.returncode or "--help" in arguments or "-h" in arguments:
                raise SystemExit(child.returncode if child.returncode >= 0 else 128 - child.returncode)
            passed = status.exists() and status.read_text() == "PASS"
            raise SystemExit(0 if passed else 1)
    overview = mode == "mixed-play" and "--overview" in arguments
    if overview:
        arguments.remove("--overview")
    stage, substeps, stage_report, soft_level = None, None, None, None
    if mode in ("mixed-train", "mixed-play"):
        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--stage", choices=("hard", "soft") if mode == "mixed-train" else ("mixed", "hard", "soft"))
        if mode == "mixed-train":
            parser.add_argument("--stage-report")
        if mode == "mixed-play":
            parser.add_argument("--substeps", type=int, choices=(4, 16, 32, 64))
            parser.add_argument("--soft-level", type=int, choices=range(10), help="Logical level of the five soil tiles; otherwise read an explicit checkpoint")
        selection, arguments = parser.parse_known_args(arguments)
        stage = selection.stage
        substeps = getattr(selection, "substeps", None)
        if mode == "mixed-train":
            import json
            if stage is None and not any(arg.startswith("env.scene.terrain.training_stage=") for arg in arguments):
                stage = "hard"
            if selection.stage_report:
                arguments.append("agent.stage_report=" + json.dumps(str(Path(selection.stage_report).expanduser().resolve())))
            for argument in arguments:
                if argument.startswith("agent.stage_report="):
                    value = argument.split("=", 1)[1]
                    try:
                        stage_report = json.loads(value)
                    except json.JSONDecodeError:
                        stage_report = value.strip("\"'")
            if stage in ("soft", "mixed") and not (selection.stage_report or any(arg.startswith("agent.stage_report=") for arg in arguments)):
                parser.error("Run mixed-eval for the checkpoint, then supply --stage-report <stage_report.json> to introduce soil")
        else:
            soft_level = selection.soft_level
            if soft_level is None:
                for index, argument in enumerate(arguments):
                    checkpoint = (arguments[index + 1] if argument == "--checkpoint" and index + 1 < len(arguments)
                                  else argument.split("=", 1)[1] if argument.startswith("--checkpoint=") else None)
                    if checkpoint and Path(checkpoint).expanduser().is_file():
                        import torch
                        saved = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=False)
                        soft_level = ((saved.get("infos") or {}).get("dreamwaq_stage") or {}).get("soft_level_ceiling", 0)
                        break
    register_tasks(overview=overview, stage=stage, substeps=substeps, isolated_training=mode == "mixed-train",
                   stage_report=stage_report, soft_level=soft_level)
    if mode == "soil-check":
        soil_curriculum_check(arguments)
        return
    soil_curriculum = mode in ("soil-train", "soil-play")
    if soil_curriculum:
        mode = mode.removeprefix("soil-")
    if mode == "mixed-stage-check":
        stage_check(arguments)
        return
    if mode == "mixed-profile":
        mixed_profile(arguments)
        return
    if mode == "mixed-check":
        mixed_check(arguments)
        return
    if mode == "check":
        check(arguments)
        return
    if mode == "physics":
        physics(arguments)
        return
    if mode == "rollout":
        rollout(arguments)
        return
    mixed = mode.startswith("mixed-")
    if mixed:
        mode = mode.removeprefix("mixed-")
    task = ("Isaac-Velocity-Go2-DreamWaQ-SoilCurriculum" if soil_curriculum else "Isaac-Velocity-Rough-Go2-DreamWaQ-Mixed" if mixed else
            "Isaac-Velocity-Flat-Go2-DreamWaQ-Soft") + ("-Play" if mode != "train" else "") + "-v0"
    target = ROOT / "scripts/rollout_eval.py" if mode == "eval" else (
        Path(os.environ["ISAACLAB_PATH"]) / f"scripts/reinforcement_learning/rsl_rl/{mode}.py"
    )
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), "--task", task, *arguments]
    if mode == "train":
        # Let the original script launch Sim and define its Hydra entry point,
        # then supply only the runner factory. Its training loop is unchanged.
        import inspect
        namespace = runpy.run_path(str(target), run_name="dreamwaq_soft_train")
        import soft_terrain as soft
        inspect.unwrap(namespace["main"]).__globals__["OnPolicyRunner"] = soft.make_runner
        try:
            namespace["main"]()
        finally:
            namespace["simulation_app"].close()
    else:
        runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
