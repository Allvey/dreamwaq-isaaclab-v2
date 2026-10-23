"""Run the original IsaacLab runner with an independent flat-soil environment."""

import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def diagnostic_status(passed):
    # Kit fast shutdown can terminate Python with exit code 0 even after a failed
    # assertion. The lightweight parent checks this marker and the child status.
    status = os.environ.get("DREAMWAQ_DIAGNOSTIC_STATUS")
    if status:
        Path(status).write_text("PASS" if passed else "FAIL")


def register_tasks():
    # String entry points keep IsaacLab imports after AppLauncher.
    import gymnasium as gym
    for suffix, config in (("", "Go2SoftEnvCfg"), ("-Play", "Go2SoftPlayEnvCfg")):
        gym.register(
            id=f"Isaac-Velocity-Flat-Go2-DreamWaQ-Soft{suffix}-v0",
            entry_point="soft_terrain:make_env", disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"soft_terrain:{config}",
                    "rsl_rl_cfg_entry_point": "soft_terrain:Go2SoftRunnerCfg"},
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

        cfg = soft.Go2DreamWaQRoughEnvCfg() if args.stock else soft.Go2SoftEnvCfg()
        if args.stock:
            soft.configure_flat(cfg)
            soft.configure_soil_timing(cfg)
        else:
            cfg.scene.terrain.soft_enabled = not args.rigid
            soft.enable_soil(cfg)
        cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
        cfg.sim.device = args.device
        task = "Isaac-Velocity-Rough-Go2-DreamWaQ-v0" if args.stock else "Isaac-Velocity-Flat-Go2-DreamWaQ-Soft-v0"
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
        raw = env.unwrapped
        robot, sensor = raw.scene["robot"], raw.scene["contact_forces"]
        soil = getattr(raw.action_manager.get_term("joint_pos"), "soil", None)
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
            print("[Soil check] rigid control matches original flat task bit for bit", flush=True)
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

        cfg = soft.Go2DreamWaQRoughEnvCfg_PLAY() if args.original_rough else soft.Go2SoftPlayEnvCfg()
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
        if args.original_rough:
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
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
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
        physics_q = torch.empty((substeps, args.num_envs, 12), device=raw.device)
        physics_v = torch.empty_like(physics_q)
        native_update = raw.scene.update
        native_compute = raw.termination_manager.compute
        native_reset = raw._reset_idx
        cursor = 0
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
            nonlocal cursor
            native_update(dt)
            if cursor < substeps:
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
                               if args.original_rough else (-robot.data.body_pos_w[:, feet, 2]).clamp_min(0)),
                "floor_support_fraction": floor / weight,
                "joint_velocity": robot.data.joint_vel,
                "terminated": raw.termination_manager.terminated,
                "timeout": raw.termination_manager.time_outs,
                "position": robot.data.root_pos_w - raw.scene.env_origins,
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
                print(f"[Rollout] {'rough' if args.original_rough else 'rigid' if args.rigid else 'soft'} seed={args.seed} "
                      f"t={(step + 1) * 0.02:.1f}s phase={schedule[(step // phase_steps) % 5][0]} "
                      f"base_z={snapshot['base_height'].tolist()}", flush=True)
        wall_seconds = time.perf_counter() - started
        assert cursor == substeps and policy_hash() == original_policy_hash
        raw.scene.update, raw.termination_manager.compute, raw._reset_idx = native_update, native_compute, native_reset
        arrays = {key: np.stack(values) for key, values in records.items()}
        arrays.update({key: value.cpu().numpy() for key, value in initial.items()})
        q, v = physics_q.cpu().numpy(), physics_v.cpu().numpy()
        assert np.isfinite(q).all() and np.isfinite(v).all()
        fd_error = np.diff(q, axis=0) / cfg.sim.dt - v[1:]
        fd_valid = np.ones(fd_error.shape[:2], dtype=bool)
        # The first interval after each autoreset contains a position discontinuity.
        for control, environment in zip(*np.nonzero(arrays["terminated"] | arrays["timeout"])):
            discontinuity = (control + 1) * cfg.decimation - 1
            if discontinuity < len(fd_valid):
                fd_valid[discontinuity, environment] = False
        arrays.update(time=(np.arange(control_steps) + 1) * 0.02, physics_joint_pos=q, physics_joint_vel=v)
        np.savez_compressed(output / "measurements.npz", **arrays)
        # Drop the first second of each phase when assessing tracking; still retain every reset.
        phase_metrics = []
        for phase in range(len(schedule) * args.cycles):
            lo, hi = phase * phase_steps, (phase + 1) * phase_steps
            steady = slice(lo + round(1 / 0.02), hi)
            native_slice = slice((lo + round(1 / 0.02)) * cfg.decimation, hi * cfg.decimation)
            block = v[native_slice]
            demeaned = block - block.mean(0)
            power = np.abs(np.fft.rfft(demeaned, axis=0)) ** 2
            frequency = np.fft.rfftfreq(len(block), cfg.sim.dt)
            hf_ratio = power[frequency >= 0.8 * (0.5 / cfg.sim.dt)].sum((0, 2)) / power.sum((0, 2)).clip(1e-30)
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
        summary = {"completed": True, "inference_only": True, "rigid_control": args.rigid or args.original_rough, "seed": args.seed,
            "task": task, "original_rough": args.original_rough, "terrain_mesh_sha256": terrain_mesh_hash,
            "num_envs": args.num_envs, "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "rollout_source_sha256": rollout_source_hash,
            "policy_state_sha256": original_policy_hash, "policy_unchanged": True,
            "policy_noise": "Original sampled CENet latent; isolated Torch RNG seeded seed*1000003+control_step, shared by paired cases",
            "physics_dt": cfg.sim.dt, "control_period_s": raw.step_dt,
            "command_scale": args.command_scale,
            "contact_history_s": cfg.scene.contact_forces.history_length * cfg.sim.dt,
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
                           "Native velocities are recorded at every substep, not inferred from 50 Hz samples. "
                           "Runtime includes diagnostic recording; it is not training throughput. "
                           "For original rough terrain, foot_depth is an unused zero placeholder and floor_support_fraction means native terrain support."}
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


def main():
    mode, arguments = sys.argv[1], sys.argv[2:]
    if mode == "physics" and ("--sweep_dt" in arguments or "--report_sweep" in arguments):
        physics(arguments)
        return
    if mode in ("check", "physics", "rollout") and (mode != "rollout" or "--single" in arguments) and "DREAMWAQ_DIAGNOSTIC_STATUS" not in os.environ:
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
    register_tasks()
    if mode == "check":
        check(arguments)
        return
    if mode == "physics":
        physics(arguments)
        return
    if mode == "rollout":
        rollout(arguments)
        return
    task = f"Isaac-Velocity-Flat-Go2-DreamWaQ-Soft{'-Play' if mode != 'train' else ''}-v0"
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
