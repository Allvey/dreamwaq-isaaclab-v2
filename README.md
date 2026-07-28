# DreamWaQ in Isaac Lab - Unitree Go2

A reproduction of the two core contributions of **DreamWaQ** (Nahrendra, Yu & Myung, ICRA 2023) in NVIDIA Isaac Lab on a Unitree Go2: an **asymmetric actor-critic** and **CENet**, a context-aided estimator that lets the deployed policy run on proprioception alone.

**Result:** the proprioception-only policy matched a privileged baseline on terrain difficulty and episode length, trailing it by ~3 reward and ~0.1 m/s tracking error. CENet recovered true body velocity to **MSE ≈ 0.006 (m/s)²** from proprioceptive history alone.

https://github.com/user-attachments/assets/daabeb17-dc1e-4bb3-8c0d-82a0601c9b6a

*Trained Go2 robots crossing discrete-obstacle terrain in Isaac Sim, each tracking its own velocity command.*

---

## The problem

In simulation an RL policy can see privileged information - the exact terrain under the robot, its true base velocity. A real quadruped has neither. It has joint encoders and an IMU. So a policy trained on privileged inputs cannot be deployed.

DreamWaQ's answer is to have the robot infer that hidden state from its own recent proprioceptive history, and to train the policy, the value function, and the estimator **together in a single phase** - no teacher-student distillation.

## Method

**Asymmetric actor-critic.** The actor sees only what hardware can measure (45-D proprioception) plus CENet's outputs. The critic, used only during training, additionally sees a 187-D terrain height scan and the true base velocity.

| Group | Dim | Contents | Seen by |
|---|---|---|---|
| `policy` | 45 | ang. velocity, gravity, command, joint pos/vel, last action | actor + critic |
| `height_scan` | 187 | 17×11 terrain height grid (privileged) | critic only |
| `privileged_vel` | 3 | true base linear velocity | critic + CENet target |

Actor input **64** (45 + 3 velocity + 16 latent) · Critic input **235** (45 + 187 + 3)

**CENet.** A shared encoder over the last 5 proprioceptive frames (225-D) splits into two heads: a velocity head regressing 3-D body velocity, and a β-VAE head producing a 16-D latent context. Trained with a hybrid loss:

```
L = MSE(v̂, v) + MSE(ô, o) + β · KL( z ‖ N(0, I) )
```

CENet's outputs are **detached** before reaching the actor, so the policy's gradient cannot reshape the estimator - it learns only from its own prediction error.

![CENet architecture](docs/figures/cenet_architecture.png)

## Results

Both policies: 2000 iterations, seed 1, identical curriculum and rewards. The only differences are observation routing and the CENet loss.

| Metric | Baseline (privileged) | DreamWaQ |
|---|---|---|
| Mean episodic reward | 20.0 | 17.0 |
| Mean episode length | 940 | 915 |
| Linear-velocity error (m/s) | 0.41 | 0.50 |
| Terrain curriculum level | 5.46 | 5.35 |
| CENet velocity MSE | - | 0.012 (train) / 0.0058 (eval) |
| **Actor observation dim** | **235 (privileged)** | **64 (deployable)** |

The last row is the point. Near-equal locomotion from a quarter of the observations, none of them privileged.

### CENet velocity estimation

![CENet velocity estimate vs ground truth](docs/figures/cenet_velocity_estimate.png)

The estimate (orange) tracks true body velocity (black) on all three axes, including the rapid vertical swings as the body bounces during walking - from proprioceptive history alone. This is what makes the proprioception-only policy work.

### Training curves

| Reward | Episode length |
|---|---|
| ![](docs/figures/training_reward.png) | ![](docs/figures/episode_length.png) |

| Velocity tracking error | Terrain curriculum |
|---|---|
| ![](docs/figures/velocity_tracking_error.png) | ![](docs/figures/terrain_curriculum.png) |

The curves track closely for ~400 iterations, after which the baseline holds a steady lead of roughly three reward points - the expected cost of replacing privileged inputs with inferred state. Both finish at essentially the same terrain level (5.46 vs 5.35).

![CENet hybrid-loss components](docs/figures/cenet_losses.png)

Velocity MSE drops from ~0.10 to ~0.01 within the first hundred iterations. The KL term rises and levels off around 3.0–3.1 rather than collapsing to the prior, so the latent stays active and keeps carrying information.

### Evaluation rollout

| Command tracking | Base path (top-down) | Joint angles |
|---|---|---|
| ![](docs/figures/command_tracking.png) | ![](docs/figures/base_path.png) | ![](docs/figures/joint_angles.png) |

## Setup

Requires NVIDIA Isaac Lab 2.0 (Isaac Sim 4.5) and `rsl_rl` 3.1.2. Trained on a single RTX 4060 Laptop (8 GB), 2048 parallel envs at ~11–13k steps/s.

```bash
# copy the task package into Isaac Lab's locomotion config folder
cp -r go2_dreamwaq \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/
```

Task IDs register automatically on import. No framework files are modified.

```bash
# baseline (privileged, symmetric)
isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-v0 --headless --max_iterations 2000 --seed 1

# DreamWaQ (asymmetric + CENet)
DREAMWAQ_CENET_BETA=0.5 isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Go2-DreamWaQ-v0 --headless --max_iterations 2000 --seed 1

# evaluate
isaaclab.sh -p scripts/rollout_eval.py \
  --task Isaac-Velocity-Rough-Go2-DreamWaQ-Play-v0 --num_envs 1 --num_steps 500 --out ./figures
```

## Verification

Each component was gated on a pass/fail check before the next was built:

1. Stock task trains
2. Forked task trains identically - package works
3. Printed dims confirm asymmetry (actor 64, critic 235)
4. Unit tests: output shapes, finite differentiable loss, gradients reach the shared encoder from CENet's loss but **not** the actor's
5. Live run shows CENet losses decreasing

```bash
isaaclab.sh -p tests/test_cenet.py
isaaclab.sh -p tests/test_dreamwaq_actor_critic.py
```

## Repository layout

```
go2_dreamwaq/
  __init__.py                  task ID + custom class registration
  rough_env_cfg.py             env config, observation group split
  rsl_rl_ppo_cfg.py            PPO config, obs_groups routing
  cenet.py                     CENet module (encoder + 2 heads + hybrid loss)
  dreamwaq_actor_critic.py     wraps CENet around the stock policy
  dreamwaq_ppo.py              single-phase update (PPO + CENet loss)
tests/                         offline unit tests
scripts/                       plotting and rollout evaluation
docs/                          full report (PDF) and figures
```

## Full report

The complete write-up - method, implementation decisions, verification strategy, and discussion - is in [`docs/DreamWaQ_Report_Roman_Kalyna.pdf`](docs/DreamWaQ_Report_Roman_Kalyna.pdf).

## Limitations

Stated plainly, since this is a reproduction rather than a replication of the paper's numbers:

- Single seed - indicative, not statistically supported. No error bars.
- 2048 parallel envs vs the paper's 4096 (8 GB VRAM limit).
- Go2 in Isaac Lab rather than A1 in Isaac Gym.
- The AdaBoot adaptive-bootstrapping gate was not implemented.
- The VAE reconstructs the *current* observation rather than the next, to avoid modifying rollout storage.
- β fixed at 0.5 by hand - chosen because β = 1 caused posterior collapse. Not swept.
- The critic receives the height scan and true velocity but not the paper's random disturbance force, so it is slightly less privileged than the original.

## References

- I. M. A. Nahrendra, B. Yu, H. Myung, "DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning," ICRA 2023. [arXiv:2301.10602](https://arxiv.org/abs/2301.10602)
- N. Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," CoRL 2022 - the `rsl_rl` PPO implementation.
- M. Mittal et al., "Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments," RA-L 2023 - predecessor of Isaac Lab.
- [Manaro-Alpha/DreamWaQ](https://github.com/Manaro-Alpha/DreamWaQ) - public reference implementation targeting Isaac Gym, consulted for network and training design.

## License

MIT - see [LICENSE](LICENSE). Depends on NVIDIA Isaac Lab (BSD-3-Clause) and `rsl_rl` (BSD-3-Clause), neither of which is vendored here.

---

**Roman Kalyna** · Mechanical Engineer, Mechatronics & Robotics
[LinkedIn](https://linkedin.com/in/roman-kalyna)
