# Roll out a trained DreamWaQ policy and log per-step data for the eval figures:
#   command_tracking.png  - commanded vs actual base velocity (vx, vy, yaw)   [task space]
#   cenet_plot.png  - CENet v_hat vs true base velocity (vx, vy, vz)
#   base_path.png         - top-down base xy trajectory                       [task space]
#   joint_angles.png      - the 12 joint angles vs time                       [configuration space]   <-- NEW
#
# Built on Isaac Lab's scripts/reinforcement_learning/rsl_rl/play.py so it loads
# the env + policy the exact same (hydra) way. Run it the SAME way you run play,
# and from your HOME dir (so "logs/" resolves to ~/logs), e.g.:
#
#   cd ~
#   ~/IsaacLab/isaaclab.sh -p \
#     ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_dreamwaq/rollout_eval.py \
#     --task Isaac-Velocity-Rough-Go2-DreamWaQ-Play-v0 --num_envs 1 --num_steps 500 --out ~/figures
#
# Or point at a specific checkpoint with --checkpoint ~/logs/rsl_rl/go2_dreamwaq_rough/<run>/model_1999.pt

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# cli_args lives next to play.py; add that dir so we can import it from anywhere.
_LAB_ROOT = Path(os.environ["ISAACLAB_PATH"]) if os.environ.get("ISAACLAB_PATH") else Path(
    importlib.util.find_spec("isaaclab").origin
).resolve().parents[3]
_RSL_RL_SCRIPTS = str(_LAB_ROOT / "scripts/reinforcement_learning/rsl_rl")
if os.path.isdir(_RSL_RL_SCRIPTS) and _RSL_RL_SCRIPTS not in sys.path:
    sys.path.insert(0, _RSL_RL_SCRIPTS)
import cli_args  # noqa: E402  (from the rsl_rl scripts dir)

parser = argparse.ArgumentParser(description="Roll out a DreamWaQ policy and plot eval figures.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (1 = one clean trajectory).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_steps", type=int, default=500, help="Rollout length in policy steps (~10 s at 50 Hz).")
parser.add_argument("--out", type=str, default="./figures", help="Output directory for the PNGs.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Headless, no cameras: we read state directly from the sim, no rendering needed.
args_cli.headless = True
args_cli.enable_cameras = False

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  (registers tasks + injects DreamWaQ classes)
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _to_np(t):
    return t.detach().cpu().numpy()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # --- mirror play.py's env + checkpoint setup ---
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy           # rsl_rl >= 2.3
    except AttributeError:
        policy_nn = runner.alg.actor_critic     # older

    has_cenet = hasattr(policy_nn, "cenet")
    if not has_cenet:
        print("[WARN] policy has no CENet (baseline?). Skipping the money plot.")

    robot = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt

    # joint names for labelling the configuration-space plot (e.g. FL_hip_joint, ...)
    try:
        joint_names = list(robot.data.joint_names)
    except Exception:
        joint_names = []

    # --- rollout, logging one clean episode (stop at first done) ---
    obs = env.get_observations()
    ts, cmd_log, act_lin, act_yaw, vhat_log, vtrue_log, xy_log = [], [], [], [], [], [], []
    joint_log = []   # NEW: per-step joint angles (configuration space)

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            # CENet estimate from the current observation (raw history, as in training)
            vel_hat = None
            if has_cenet and ("prop_history" in obs):
                vel_hat = policy_nn.cenet(obs["prop_history"])[0]   # (num_envs, 3)
            actions = policy(obs)

        # state at time t (matches obs/v_hat), read BEFORE stepping; env 0 only
        cmd = env.unwrapped.command_manager.get_command("base_velocity")[0]   # [vx, vy, yaw]
        lin_b = robot.data.root_lin_vel_b[0]      # [vx, vy, vz] body frame (true)
        ang_b = robot.data.root_ang_vel_b[0]      # [wx, wy, wz]
        pos_w = robot.data.root_pos_w[0, :2]      # [x, y] world
        q = robot.data.joint_pos[0]               # NEW: 12 joint angles (rad)

        ts.append(step * dt)
        cmd_log.append(_to_np(cmd))
        act_lin.append(_to_np(lin_b))
        act_yaw.append(float(ang_b[2].item()))
        vtrue_log.append(_to_np(lin_b))
        vhat_log.append(_to_np(vel_hat[0]) if vel_hat is not None else np.full(3, np.nan))
        xy_log.append(_to_np(pos_w))
        joint_log.append(_to_np(q))               # NEW

        with torch.inference_mode():
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        if bool(dones[0].item()):
            print(f"[INFO] Episode ended at step {step} (robot reset). Logged {len(ts)} steps.")
            break

    env.close()

    # --- arrays ---
    ts = np.asarray(ts)
    cmd_log = np.asarray(cmd_log)      # (T, 3) vx, vy, yaw
    act_lin = np.asarray(act_lin)      # (T, 3) vx, vy, vz
    act_yaw = np.asarray(act_yaw)      # (T,)
    vhat_log = np.asarray(vhat_log)    # (T, 3)
    vtrue_log = np.asarray(vtrue_log)  # (T, 3)
    xy_log = np.asarray(xy_log)        # (T, 2)
    joint_log = np.asarray(joint_log)  # NEW: (T, 12)

    os.makedirs(args_cli.out, exist_ok=True)

    # --- Figure: command tracking (commanded vs actual) ---
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(ts, cmd_log[:, 0], color="firebrick", lw=2, label="Commanded")
    ax[0].plot(ts, act_lin[:, 0], color="steelblue", lw=1.5, label="Actual")
    ax[0].set_ylabel("Forward velocity $v_x$ (m/s)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ts, cmd_log[:, 1], color="firebrick", lw=2, label="Commanded")
    ax[1].plot(ts, act_lin[:, 1], color="steelblue", lw=1.5, label="Actual")
    ax[1].set_ylabel("Lateral velocity $v_y$ (m/s)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[2].plot(ts, cmd_log[:, 2], color="firebrick", lw=2, label="Commanded")
    ax[2].plot(ts, act_yaw, color="steelblue", lw=1.5, label="Actual")
    ax[2].set_ylabel("Yaw rate (rad/s)"); ax[2].set_xlabel("Time (s)"); ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.suptitle("Command Tracking: Commanded vs Actual Base Velocity")
    fig.tight_layout()
    p = os.path.join(args_cli.out, "command_tracking.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")

    # --- Figure: CENet money plot (v_hat vs ground truth) ---
    if has_cenet and np.isfinite(vhat_log).any():
        labels = ["$v_x$ (m/s)", "$v_y$ (m/s)", "$v_z$ (m/s)"]
        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for i in range(3):
            ax[i].plot(ts, vtrue_log[:, i], color="black", lw=2, label="True (sim)")
            ax[i].plot(ts, vhat_log[:, i], color="darkorange", lw=1.5, label=r"CENet estimate $\hat{v}$")
            ax[i].set_ylabel(labels[i]); ax[i].legend(); ax[i].grid(alpha=0.3)
        ax[2].set_xlabel("Time (s)")
        fig.suptitle(r"CENet Body-Velocity Estimation: $\hat{v}$ vs Ground Truth")
        fig.tight_layout()
        p = os.path.join(args_cli.out, "cenet_money_plot.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"  wrote {p}")
        # quick numeric summary for the report
        mse = np.nanmean((vhat_log - vtrue_log) ** 2)
        print(f"  [info] CENet rollout velocity MSE = {mse:.5f}")

    # --- Figure: base path (top-down) ---  [task space]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xy_log[:, 0], xy_log[:, 1], color="steelblue", lw=2)
    ax.scatter(xy_log[0, 0], xy_log[0, 1], color="green", s=60, zorder=5, label="start")
    ax.scatter(xy_log[-1, 0], xy_log[-1, 1], color="red", s=60, zorder=5, label="end")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Base Path (top-down)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args_cli.out, "base_path.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")

    # --- Figure: joint angles vs time (configuration space) ---  NEW
    n_dof = joint_log.shape[1]
    # group the 12 Go2 joints by type (hip / thigh / calf) for a readable plot;
    # fall back to one axis with all joints if the names don't match.
    groups = [("Hip", "hip"), ("Thigh", "thigh"), ("Calf", "calf")]
    names = joint_names if len(joint_names) == n_dof else [f"joint {i}" for i in range(n_dof)]
    group_idx = {g: [i for i, nm in enumerate(names) if key in nm.lower()] for g, key in groups}
    grouped = all(len(group_idx[g]) > 0 for g, _ in groups)

    if grouped:
        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for row, (g, _) in enumerate(groups):
            for i in group_idx[g]:
                ax[row].plot(ts, joint_log[:, i], lw=1.2, label=names[i])
            ax[row].set_ylabel(f"{g} angle (rad)")
            ax[row].legend(fontsize=7, ncol=2, loc="upper right")
            ax[row].grid(alpha=0.3)
        ax[2].set_xlabel("Time (s)")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(n_dof):
            ax.plot(ts, joint_log[:, i], lw=1.0, label=names[i])
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Joint angle (rad)")
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)
    fig.suptitle("Joint Angles vs Time (configuration space)")
    fig.tight_layout()
    p = os.path.join(args_cli.out, "joint_angles.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")

    print("Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
