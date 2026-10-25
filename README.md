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

## IsaacLab fast soil proxy (local installation)

`soft-train`, `soft-play`, `soft-eval` and `soft-check` retain PhysX and the
original DreamWaQ model, PPO settings, observation dimensions, action scaling and
reward functions. The separate task initially uses **flat** soil: normal bearing,
horizontal drag, shin/thigh resistance, pull-out resistance and footprint memory
come from [newton-terrain's sand proxy](https://github.com/prakash-aryan/newton-terrain/blob/main/newton_terrain/sand_proxy.py).
This is an approximate force model, not MPM or a deforming terrain mesh.

The existing sibling checkout `../newton-terrain` supplies
`newton_terrain/sand_proxy.py` and `runs/proxy_table.json`; set
`NEWTON_TERRAIN_PATH` if it is elsewhere. Newton/MuJoCo need not be installed in
the `dreamwaq` environment. The adapter supports Isaac Sim's bundled Warp 1.5.
Source/table hashes and resolved coefficients are saved in each run's environment
configuration.

```bash
# Short physics/contact/reset verification; does not train.
bash scripts/isaac.sh soft-check --headless --num_envs 4 --steps 100

# Physical acceptance: no policy, original joint targets, no automatic resets.
bash scripts/isaac.sh soft-physics --headless --num_envs 4 --seconds 8
# Matched rigid control; separate diagnostic run.
bash scripts/isaac.sh soft-physics --headless --num_envs 4 --seconds 8 --rigid
# Isolate bearing under imposed geometry (diagnostic joint constraints only).
bash scripts/isaac.sh soft-physics --headless --num_envs 4 --seconds 8 --lock_joints
# Paired timestep/velocity audit; 5, 2.5 and 1.25 ms, always 20 ms control.
bash scripts/isaac.sh soft-physics --headless --num_envs 4 --seconds 8 --sweep_dt
# Extend the convergence study without changing soil or actuator parameters.
bash scripts/isaac.sh soft-physics --headless --num_envs 4 --seconds 8 \
  --sweep_dt --sweep_dts 0.000625,0.0003125

# Manual training with the migrated soil task; original learning algorithm/model.
# Material/depth defaults remain f0.75 sand and 0.15 m.
bash scripts/isaac.sh soft-train --headless --num_envs 2048 \
  --max_iterations 2000 --seed 1 --run_name sand_f075_dt03125_seed1

# Fine-tune the trained DreamWaQ rigid-ground model into a NEW soft run.
# Loads actor, critic and CENet weights; optimizer and iteration counter start fresh.
bash scripts/isaac.sh soft-train --headless --num_envs 2048 \
  --max_iterations 2000 --seed 1 --run_name soft_finetune_from_rough_seed1 \
  agent.init_checkpoint=/home/allvey/Projects/robotics/dreamwaq-isaaclab/logs/rsl_rl/isaaclab/go2_dreamwaq_rough/2026-10-18_14-35-23_clean_seed1/model_1999.pt

# Matched flat rigid control, with otherwise identical task/algorithm settings.
bash scripts/isaac.sh soft-train --headless --num_envs 2048 \
  --max_iterations 2000 --seed 1 --run_name rigid_dt03125_seed1 \
  env.scene.terrain.soft_enabled=false \
  agent.experiment_name=isaaclab/go2_dreamwaq_rigid_flat

# View an explicitly selected checkpoint; may also use an existing rigid policy.
bash scripts/isaac.sh soft-play --num_envs 4 --checkpoint /absolute/path/model_1999.pt
```

Soft runs go to `logs/rsl_rl/isaaclab/go2_dreamwaq_soft_flat/<run>/`;
the control command uses `isaaclab/go2_dreamwaq_rigid_flat/<run>/` under the same
root. Point TensorBoard at `logs/rsl_rl/`. Existing `train`, `play`, `eval` commands
still use the original rough task. Compare the soft task with the matched flat
control; historical rough-terrain runs have a different terrain distribution.
`agent.init_checkpoint` initializes a new run from network weights, with the
original PPO settings and a fresh optimizer/iteration counter. The source path is
saved in `params/agent.yaml`. Do not combine it with `--resume True`, which resumes
an existing run's optimizer and iteration state. Formal training is started manually.

The nominal surface is at z=0 with no collider; PhysX's hard bottom is at z=-depth.
Soil normal forces enter the existing contact sensor before history/air-time
updates, including base-contact termination. Height scans see the nominal top,
not footprint depressions. Memory is per environment and is cleared on reset.
Tune physical parameters with Hydra, e.g. `env.scene.terrain.depth=0.10` or
`env.scene.terrain.material=f0.9`, without changing model code.

### Standalone 5 x 5 soil curriculum

`soil-train` generates an isolated soil scene with 25 fixed tiles, without the
original rough terrain. Five columns are flat soil, mound, basin, undulating soil
and alternating hard/soft bands. Five rows are difficulty levels 0--4. Each tile
is 8 x 8 m with a 6 x 6 m interior and a 1 m flat rigid frame on each side.
The five rows' geometry scales are 0.10, 0.325, 0.55, 0.775 and 1.0; bearing
multipliers are 4, 2.828, 2, 1.414 and 1. Lower rows use wider hard/soft bands.
Drag, soil depth and footprint parameters keep their existing values.

Robots start randomly at levels 0--1, distributed evenly over scenario columns.
At episode reset, the native IsaacLab `terrain_levels_vel` function promotes a
robot when its horizontal distance from the tile origin exceeds 4 m. Otherwise
it demotes the robot when that distance is below half the product of CURRENT
command speed and configured maximum episode duration. This uses straight-line
displacement, not accumulated path length or an integral of earlier commands.
There is no added fall/contact gate. Initial/manual resets with zero elapsed
episode length do not advance the curriculum. The minimum level is 0; advancing
past level 4 redistributes the robot randomly over levels 0--4. Its column stays
fixed. All meshes exist before training and never deform/relabel on promotion;
actual foot position selects the local surface height and row-specific material.
Partial reset clears only that robot's footprint memory. Like the original hard
task, this task uses native episode timeouts/base-contact termination and a flat
outer border, without the staged task's extra domain-edge timeout. This lets
robots in outer rows cross the 4 m promotion threshold.

The task defaults to **64 substeps at 0.3125 ms**, 20 ms control and 48 contact
history samples (15 ms). Model, observations, PPO, rewards and actuators retain
the original settings. Starting without checkpoint/resume randomly initializes
all networks and starts a fresh optimizer at iteration 0. No passing hard-ground
report is required for this standalone task.

```bash
# Diagnostic only: geometry/materials, native grade transitions, partial resets,
# checkpoint restoration and one discarded PPO update. Does not call learn.
bash scripts/isaac.sh soil-check --headless --num_envs 25

# Manually start a new 2000-iteration training run.
bash scripts/isaac.sh soil-train --headless --num_envs 512 \
  --max_iterations 2000 --seed 1 --run_name soil_curriculum_64substeps_seed1

# Visualize a selected checkpoint; the checkpoint supplies the learned grades.
bash scripts/isaac.sh soil-play --num_envs 5 --checkpoint /absolute/path/model_1999.pt
```

Outputs stay under `logs/rsl_rl/isaaclab/go2_dreamwaq_soft_curriculum/<run>/`.
Checkpoints carry the spatial layout, timestep and per-robot curriculum levels;
same-count resume restores grades, optimizer and native iteration counter using
`--resume True --load_run <run-folder> --checkpoint model_N.pt`. Play with fewer
robots cycles saved grades within the same scenario column. Different layout,
material or timing metadata is rejected rather than silently reinterpreted.
TensorBoard at `logs/rsl_rl/` includes `Curriculum/terrain_levels`,
`SoilCurriculum/<scenario>_level` and per-level population fractions, plus
`SoilScenario/<scenario>_tracking_rmse_xy`, soft-contact fraction and termination
statistics. Check mode uses `go2_dreamwaq_soft_curriculum_check/<run>/`.

### Mixed original rough terrain and fixed-level soil stages

Training uses separate hard/soft scenes with the original policy and PPO update
(see the staged commands below). In the combined testing scene, half of the
robots start on each domain by default. The soil cells occupy the next
**five adjacent tiles on the +Y side of the same grid**, with one shared
outer border. Every cell is **8 x 8 m**; its **6 x 6 m** interior is surrounded
by **1 m of flat rigid ground**, giving 2 m lanes between neighbouring interiors.
The scene types, in column order, are:

| Column | Scenario | Lowest grade → target grade |
| --- | --- | --- |
| 1 | `flat` | Flat soil; bearing capacity multiplier 4 → 1 |
| 2 | `mound` | Smooth hill, height 0.03 → 0.30 m |
| 3 | `basin` | Smooth depression, depth 0.02 → 0.20 m |
| 4 | `undulating` | Windowed waves, amplitude 0.01 → 0.10 m, fixed wavelength 2 m |
| 5 | `mixed` | Alternating soft/hard bands, width 3 → 1 m, snapped to the grid |

The existing `mixed-*` and `joint-*` routes retain at most five physical soil tiles: one
tile per type, at one shared logical difficulty level. The ten logical levels
(0–9) do not create additional rows or meshes. Geometry amplitudes increase
linearly with level. All five types use the same graded
bearing capacity (4 → 1, geometric interpolation); drag, pull-out, memory, soil
depth, spring/damping coefficients and physics timing remain fixed. The highest
level reproduces the previous target geometry and `f0.75` material. These are initial
curriculum settings, not a claim of calibrated real sand difficulty. The original
rough-terrain curriculum is retained.

Soft robots start at logical level 0 and retain their scenario tile. All five
tiles use the same level, fixed for the entire process; episode resets neither
promote nor demote robots. Material still follows actual foot position, so hard
frames and internal hard bands remain rigid. Partial resets clear only that
robot's footprint memory and contact counters. `SoilCurriculum/mean_level` and
per-scenario `*_level` curves report the shared level. `soft_level_ceiling` now
selects that exact level, and `soft_num_levels=10` defines the logical range.
Only a passing hard-retention and five-scenario evaluation can authorize the
next level. Its five meshes and material table are built BEFORE the next manually
started soft stage; live geometry is never relabeled or deformed during a rollout.

Play uses five tiles at the checkpoint's level when `--checkpoint` is an explicit
file path; `mixed-play --soft-level 0` (through 9) overrides the displayed level.
Without an explicit checkpoint/level it defaults to level 0. Its original hard
map remains the smaller randomly arranged play map. For the
previous fixed-scene control, train with `env.scene.terrain.soft_curriculum=false`
(all five tiles at the target parameters), or check with `mixed-check --fixed_soil`.
`soft_columns` selects one to five tiles; `soft_scenarios` selects
the ordered scenario list. For a flat-only
control use `env.scene.terrain.soft_columns=1` and
`'env.scene.terrain.soft_scenarios=[flat]'` during training.
`soft_fraction` still controls the number of soil-spawn robots independently of
map area. Spawn sampling cycles across types at the shared level; soil memory remains
isolated per robot. `soft_border`, `soft_resolution`, `soft_mound_height`,
`soft_basin_depth`, `soft_wave_height`, `soft_wave_length` and `soft_stripe_width`
are stored in the resolved configuration. Grid spacing defaults to 0.1 m.

The rendered top, hidden critic scan and per-contact physical sampling use the
same triangulated height grid. There is no rigid collider on a soft top: the
collision bottom follows that profile at a vertical offset of -0.15 m. Hard frames
and internal hard bands have real top colliders. The upstream spring, plastic
yield, footprint, drag and pull-out equations are evaluated in a separate local
tangent frame for each leg; forces/torques are rotated back to world coordinates.
Belly contact uses its own surface sample. This is a gently curved surface proxy,
not volumetric MPM or exact rod integration across curved/material boundaries.
Flat-patch force parity, slope support direction and tangential dissipation are
covered by `mixed-check`. The standalone flat-soil task retains its original
horizontal kernel.

Spawn groups label statistics only: robots from either group acquire soil forces
when their feet enter a soft patch. Local-origin changes from the rough curriculum
are accounted for. Original active-cell triangles are retained; only the flat
perimeter is rebuilt around the extended grid. Source/active rough mesh and soil
surface hashes are recorded. Use a new run name for this five-scenario variant.

Soft physics now defaults to **64 substeps**: 0.3125 ms physics, 50 Hz control,
and a 15 ms contact-history window (48 samples). Explicitly select
`env.scene.terrain.physics_substeps=32` for the 0.625 ms speed experiment
(24 contact samples), or `=16` for the 1.25 ms experiment (12 contact samples).
The policy period stays at 20 ms and the contact window at 15 ms in all cases.
This one selector resolves dt, decimation, render interval
and contact sampling together after Hydra overrides. The standalone flat soil
task retains its 64-substep default; historical original rough runs use 5 ms
physics. Both remain separate evaluation references. Earlier 32-substep standing
tests passed basic thresholds but retained high-frequency oscillation; record
this as a distinct numerical experiment rather than an equivalent replacement.
Network/CENet inputs, losses, PPO settings, rewards, commands and actuators are
unchanged. Hard robots keep the native per-robot curriculum; soil grade changes
only at stage boundaries. Per-environment soil memory remains isolated across resets.

For an explicit **4-substep mixed visualization experiment**, use
`mixed-play --substeps 4 --stage mixed`. Both hard and soft
robots then use 5 ms physics, 20 ms control and a three-frame (15 ms) contact
history. The default is 64, the convergence reference. Earlier
5 ms flat-soil fixed-target standing tests showed strong joint oscillation; a
finite-data/PPO interface check does not establish physically stable training.
Keep 4-substep soil runs for diagnostic/historical comparisons. The staged
training gate requires the evaluated soft timestep to match the training timestep;
the normal workflow below evaluates soft physics at 64 substeps.

The short acceptance and sampling benchmark also accept `--substeps 4`:
`bash scripts/isaac.sh mixed-check --headless --num_envs 12 --steps 100 --substeps 4`
and `bash scripts/isaac.sh mixed-profile --headless --stage mixed --num_envs 2048 --substeps 4`.

```bash
# Acceptance only: geometry, force/contact/reset isolation and one discarded
# PPO update. No trained checkpoint is written and runner.learn is not called.
bash scripts/isaac.sh mixed-check --headless --num_envs 12 --steps 100

# View the mixed task, initially following rough env 0 (at least two required).
bash scripts/isaac.sh mixed-play --num_envs 12 --checkpoint /absolute/path/model.pt
# Show the original terrain and all five soil scenario columns from above.
bash scripts/isaac.sh mixed-play --overview --num_envs 12 --checkpoint /absolute/path/model.pt
# One soil robot; match the checkpoint's training physics (4/16/32/64).
bash scripts/isaac.sh mixed-play --stage soft --num_envs 1 --substeps 4 --checkpoint /absolute/path/model.pt

# Evaluate two legacy references (these do not cover all five soil scenarios):
# the SAME checkpoint on the historical native rough task
# and on the current flat soil, with the same fixed commands and seed.
bash scripts/isaac.sh soft-rollout --single --original_rough --headless \
  --num_envs 16 --seed 1 --phase_seconds 4 --cycles 2 \
  --checkpoint /absolute/path/model.pt
bash scripts/isaac.sh soft-rollout --single --headless \
  --num_envs 16 --seed 1 --phase_seconds 4 --cycles 2 \
  --physics_dt 0.000625 \
  --checkpoint /absolute/path/model.pt
```

Mixed checkpoints, TensorBoard events and resolved configurations are stored in
`logs/rsl_rl/isaaclab/go2_dreamwaq_mixed/<run>/`. The configuration records the
rough mesh hash and actual rounded group sizes. TensorBoard's `Mixed/` series
separate rough/soft reward per simulated second, termination/timeout fractions
per step, and the actual soft sample fraction; these are not episode success
rates. `SoilScenario/` adds the same metrics per soil spawn type.
Use `logs/rsl_rl/` as the TensorBoard root. Evaluate multiple seeds before
selecting a checkpoint; combined training reward alone can hide standing drift.
To continue an existing checkpoint with optimizer state, select
`--stage hard` or `--stage soft --stage-report <matching-report>` and use `--resume True
--load_run <existing-run> --checkpoint model_<iteration>.pt` and a new `--run_name`
containing `32substeps`; omit `agent.init_checkpoint`. The native runner writes a
new run directory and treats `--max_iterations` as additional iterations after
the loaded counter. Existing training processes retain their original timestep;
they are never automatically stopped or restarted.

### Separate hard/soft training scenes; combined evaluation scene

`mixed-train --stage hard` loads only the original rough terrain, using the native
IsaacLab environment, terrain importer, height scanner, curriculum and termination
rules. No soil meshes, proxy or stage-boundary timeout are created. Network/PPO
settings remain unchanged, while checkpoint metadata retains the soft curriculum
through hard rehearsal. Matching configurations do not guarantee deterministic
GPU training or identical learning curves.

`mixed-train --stage soft` loads only five tiles at the approved level, centered around
the origin, with their existing hard frames/lanes and lower support surfaces.
It does not generate or import the original rough grid. Soil difficulty, forces,
observations, rewards and performance gates are unchanged. `mixed-train` defaults
to `--stage hard`; simultaneous `--stage mixed` training is no longer supported.
`mixed-play` and `mixed-eval` retain the combined scene for comparison, including
when play limits robots to `--stage hard` or `--stage soft`.

| Stage | Robot spawn region | Physics per 20 ms control step | Soil proxy |
| --- | --- | --- | --- |
| `hard` | Original rough cells only | 4 steps at 5 ms | Disabled; native actions/contact sensor |
| `soft` | Five soil tiles, including their hard lanes | 64 steps at 0.3125 ms by default | Enabled |

Hard-stage contact history retains the original three samples (15 ms); soft-stage
history retains 12/24/48 samples for 16/32/64 substeps (15 ms). Network inputs, PPO, rewards and PD settings
are unchanged. Hard-stage rough curriculum remains active. Soft-stage robots use
the soil curriculum described above. `soft_fraction` applies
only to the simultaneous `mixed` stage; pure stages use 100% of their domain.
In a pure soft run, `Curriculum/terrain_levels=0` is the empty-hard-group
placeholder; inspect `SoilCurriculum/mean_level` and `level_ceiling` instead.

Only the soil training scene and domain-limited combined evaluations use a boundary
timeout within 0.6 m of the domain edge. Hard training keeps the original outer
flat border and termination rules. `Stage/boundary_fraction` reports boundary
events where enabled. There is no automatic stage scheduler or in-process timestep
switch: finish/checkpoint one manual run, then manually start the next one.
Existing running processes and old results are unaffected. Resuming an earlier
combined-scene hard checkpoint uses the new isolated scene for subsequent updates;
use a fresh run to compare from-scratch training fairly.

Inference-only checks (temporary recordings can be placed outside the project):

```bash
bash scripts/isaac.sh soft-check --headless --hard_training --stock --num_envs 20 --steps 50 --record /tmp/original_hard.npz
bash scripts/isaac.sh soft-check --headless --hard_training --num_envs 20 --steps 50 --compare /tmp/original_hard.npz
bash scripts/isaac.sh mixed-stage-check --headless --stage soft --isolated_training --substeps 64 --num_envs 10 --steps 50
bash scripts/isaac.sh mixed-check --headless --num_envs 12 --steps 50 --substeps 64 --soft_level 9
```

Start from random weights on hard ground (no init checkpoint or resume):

```bash
bash scripts/isaac.sh mixed-train --stage hard --headless --num_envs 2048 \
  --max_iterations 2000 --seed 1 --run_name hard_base_seed1
```

Evaluate a saved checkpoint without training. The fixed hard holdout covers all
20 hard columns at levels 0/2/4 (60 robots). If hard performance passes, a second
process evaluates all five soil types at the current approved level (10 robots).
Commands are stand / forward 0.3 m/s / lateral 0.2 m/s / turn 0.4 rad/s / stand,
four seconds each, with the first second excluded from tracking/progress scores.

```bash
bash scripts/isaac.sh mixed-eval --headless --checkpoint /absolute/path/model_N.pt
```

Results stay beside that checkpoint under `evaluations/<timestamp>/`, including
`stage_report.json` and the existing per-case rollout artifacts. Initial gate
thresholds are XY tracking RMSE <=0.25 m/s, <=20% of robots falling during the
holdout, and forward displacement >=50% of the requested distance. They are
engineering starting thresholds, not a universal locomotion benchmark.

The first passing hard evaluation becomes the persistent hard reference. Later
hard scores must also retain >=80% of reference forward progress, increase the
tracking RMSE by no more than max(20%, 0.03 m/s), and increase fall fraction by
no more than 0.05. An explicit `--hard-reference <earlier-stage_report.json>` can
select an earlier passing reference; seed, robot count, schedule and hard mesh
must match. Default evaluation seed/count/duration should be kept fixed.

Resume a short soft block only after `can_train_soft=true` in the matching report:

```bash
bash scripts/isaac.sh mixed-train --stage soft --headless --num_envs 2048 \
  --max_iterations 10 --seed 1 --run_name soft_block1_seed1 \
  --resume True --load_run <hard-run-folder> --checkpoint model_N.pt \
  --stage-report /absolute/path/evaluations/<timestamp>/stage_report.json
```

Admission checks the exact checkpoint SHA256, evaluated soft geometry/material
and physical timestep. The first soft block starts at level 0 even if zero-shot
soft evaluation passes. Subsequent evaluations open one additional level only
when hard retention and ALL five current soil types pass. Otherwise the ceiling
holds; hard regression yields `hard_rehearsal` and blocks the next soft block.
All five tiles and all soil robots stay at the exact approved level for the run.
The report is read before environment creation, so approval changes real geometry
and bearing parameters, rather than only a recorded level. Old multi-row soil
reports must be regenerated with `mixed-eval`.

For a 16-substep soft speed experiment, first evaluate the SAME checkpoint with
`mixed-eval --substeps 16 --checkpoint ... --headless`, then supply its matching
`--stage-report` and `env.scene.terrain.physics_substeps=16` to `mixed-train
--stage soft`. A 64-substep report cannot authorize a 16-substep run. These are
numerical experiments: finite observations and a successful PPO update do not
establish physical convergence. The 64-substep default/reference is retained.

After every short soft block, run `mixed-eval` on its newest checkpoint. If hard
performance declines, resume a hard rehearsal block:

```bash
bash scripts/isaac.sh mixed-train --stage hard --headless --num_envs 2048 \
  --max_iterations 100 --seed 1 --run_name stage_hard_seed1 \
  --resume True --load_run <previous-soft-run-folder> --checkpoint <model_N.pt>
```

Evaluate again, then use `--stage soft`, its matching report and the hard run's
checkpoint for the next soft block. Evaluations inherit the original hard
reference from checkpoint metadata. The shared soft level travels inside existing
`.pt` checkpoints through hard rehearsal. Legacy ten-row checkpoints can migrate
their approved ceiling to the shared level when material/level definitions match;
old per-robot levels are intentionally discarded. Other geometry/material changes
are rejected instead of silently resetting the curriculum. Footprint and
episode state reset normally when a new process starts.
With resume, `max_iterations` is the number of ADDITIONAL iterations. Do not add
`agent.init_checkpoint` when resuming. Every command creates a new timestamped
run under `logs/rsl_rl/isaaclab/go2_dreamwaq_mixed/`, keeping previous results.
Block lengths are examples, not a validated optimal schedule. There is no
automatic training start/stop or mid-block hard monitoring: keep soft blocks short
and perform the fixed evaluation at stage boundaries.

Stage choice also works with play; its CLI is `--stage hard` or `--stage soft`.
Hydra training can instead set `env.scene.terrain.training_stage=hard/soft`.
Use one stage selector consistently. Resolved `params/env.yaml` records the stage,
actual dt/decimation, boundary margin and realized group counts.

### Joint hard/soft sampling with different physics steps

`joint-train` runs two headless simulator processes and one central DreamWaQ
model/optimizer. Hard sampling uses the original isolated rough scene at 4
substeps; soft sampling uses the isolated five-tile scene at 16 or 64 substeps.
Both return one transition per 20 ms. Actions come from the same frozen policy
throughout each 24-step rollout; both groups are then passed to the original PPO
update. Observations, CENet, rewards, PD gains and PPO settings are unchanged.
Timeouts, episode boundaries and GAE remain independent for every robot.

Start with 128 hard + 128 soft robots. Equal counts give equal numbers of samples
from each domain. This is 256 robots in total, so an iteration contains fewer
samples than an earlier 2048-robot run. Two simulator processes duplicate their
base memory; smaller PhysX contact/aggregate storage capacities are recorded in
each environment YAML, and reported capacity/device errors abort sampling.
CPU NumPy transport avoids sharing CUDA contexts. Sharing one GPU does not
guarantee faster sampling than a single simulator.

First run a short interface check. It discards one native PPO update, writes no
trained checkpoint and never invokes the training loop:

```bash
base=logs/rsl_rl/isaaclab/go2_dreamwaq_rough/2026-10-18_14-35-23_clean_seed1
bash scripts/isaac.sh joint-check --checkpoint "$base/model_1999.pt" \
  --hard-envs 128 --soft-envs 128 --soft-substeps 16
```

To manually train a new model for 2000 iterations in both environments:

```bash
bash scripts/isaac.sh joint-train --from-scratch \
  --hard-envs 128 --soft-envs 128 --soft-substeps 16 \
  --max_iterations 2000 --seed 1 --run_name joint_scratch_2000_seed1
```

`--from-scratch` randomly initializes the entire DreamWaQ network, with an empty
optimizer and iteration counter 0. Omit `--checkpoint`, `--resume` and
`--stage-report`. Hard terrain retains its native curriculum. All five soft tiles
start at level 0 and stay at that level throughout this run; there is no automatic
soft-level advancement inside a joint run. `--soft-substeps 64` also works for
scratch training. To check random initialization and both interfaces without
starting formal training, run `joint-check --from-scratch` with the same robot
counts and substeps.

To manually start a five-iteration joint experiment from the good original model,
use its existing five-tile, 16-substep evaluation report:

```bash
base=logs/rsl_rl/isaaclab/go2_dreamwaq_rough/2026-10-18_14-35-23_clean_seed1
bash scripts/isaac.sh joint-train --checkpoint "$base/model_1999.pt" \
  --stage-report "$base/evaluations/2026-10-24_16-45-59_282941/stage_report.json" \
  --hard-envs 128 --soft-envs 128 --soft-substeps 16 \
  --max_iterations 5 --seed 1 --run_name joint_128x128_16substeps_seed1
```

These commands are always headless. With `--checkpoint`, they load all network weights
with a fresh optimizer and iteration counter. Add `--resume` to restore the
checkpoint's optimizer and counter; `--max_iterations` then adds iterations.
Checkpoint-based training requires the exact checkpoint's passing hard evaluation and matching
soil layout/timestep. The approved soil level stays fixed throughout a run.
16 substeps remain a numerical experiment; successful interface checks do not
establish physical convergence or improved locomotion. To fine-tune using 64, first generate
a matching report with `mixed-eval --substeps 64` and select `--soft-substeps 64`.

Runs are saved under `logs/rsl_rl/isaaclab/go2_dreamwaq_joint/<run>/`, with
checkpoints, TensorBoard events, `params/agent.yaml`, both environment YAMLs,
`params/joint.json` and worker logs together. Diagnostic checks use
`go2_dreamwaq_joint_check/<run>/`. TensorBoard at `logs/rsl_rl/` includes both;
`Joint/hard/` and `Joint/soft/` separate domain statistics.

After a short block, evaluate BOTH domains even if hard performance regresses:

```bash
bash scripts/isaac.sh mixed-eval --headless --all-stages --substeps 16 \
  --checkpoint logs/rsl_rl/isaaclab/go2_dreamwaq_joint/<run>/model_4.pt
```

`--all-stages` collects both results without changing curriculum/admission gates.
Existing `mixed-play --stage hard/soft --checkpoint ...` can visualize these
checkpoints in the combined scene. Include `--substeps 16` when visualizing a
16-substep soft checkpoint; hard play keeps its original 4 steps.

```bash
# Diagnostic acceptance only; no runner.learn or model save.
bash scripts/isaac.sh mixed-stage-check --headless --stage hard --checkpoint /absolute/path/model.pt
bash scripts/isaac.sh mixed-stage-check --headless --stage soft --substeps 32 --checkpoint /absolute/path/model.pt
# Matched inference sampling benchmarks; run sequentially on an idle GPU.
bash scripts/isaac.sh mixed-profile --headless --stage hard --num_envs 2048 --checkpoint /absolute/path/model.pt
bash scripts/isaac.sh mixed-profile --headless --stage soft --substeps 32 --num_envs 2048 --checkpoint /absolute/path/model.pt
```

`mixed-profile --stage hard/soft` measures the corresponding isolated training
scene. Soil difficulty defaults to the checkpoint's shared level; `--soft_level N`
selects a diagnostic level explicitly. `--stage mixed` profiles the combined map.

### Sampling performance

Soil state preparation reuses the existing body-state buffer and the original
`quat_apply` expression for world COM positions. COM offsets are refreshed each
control step and after resets, instead of querying PhysX's CPU COM buffer twice
per physics substep. This is enabled by default for flat and mixed soil; use
`env.scene.terrain.optimized=false` with `soft-train` or `mixed-train` to select
the reference path. This computation optimization preserves whichever timing
profile was selected, the soil force kernel, network and PPO settings.

Use the same checkpoint, seed, environment count and timing counts for the pair:

```bash
bash scripts/isaac.sh mixed-profile --headless --reference \
  --num_envs 2048 --substeps 32 --steps 24 --warmup 8 --profile_steps 2 \
  --checkpoint /absolute/path/model.pt
bash scripts/isaac.sh mixed-profile --headless \
  --num_envs 2048 --substeps 32 --steps 24 --warmup 8 --profile_steps 2 \
  --checkpoint /absolute/path/model.pt
```

Run performance measurements without another simulation/training process on the
GPU. These commands run inference only, save no model and do not call the learning
loop. Each writes `summary.json`, `env.yaml` and `agent.yaml` under
`logs/rsl_rl/isaaclab/go2_dreamwaq_mixed_profile/<run>/`. Clean throughput timing
synchronizes CUDA only at block boundaries. A separate component pass synchronizes
all CUDA work around each call to include PhysX's other streams; this intentionally
changes scheduling, so component times are not normal training throughput.
Use `--substeps 64` for the earlier timing reference. When comparing 32 versus
64 substeps, omit `--reference` on both so the adapter implementation matches.

Exact trajectory/contact parity can be checked in fresh processes:

```bash
bash scripts/isaac.sh mixed-check --headless --reference --num_envs 8 --steps 100 \
  --substeps 32 \
  --record /tmp/dreamwaq_mixed_reference.npz
bash scripts/isaac.sh mixed-check --headless --num_envs 12 --steps 100 \
  --substeps 32 \
  --compare /tmp/dreamwaq_mixed_reference.npz
```

The comparison covers both groups' root/joint states, rewards and contacts, plus
soft wrenches, depth, footprint state, contact history and air time. These checks
also discard one diagnostic PPO update; they never write a trained checkpoint.
A passing short check verifies interfaces and finite physics, not locomotion
performance or calibration to real soil.

`soft-physics` checks bearing/moment balance, dissipative foot/shin/thigh drag,
pull-out resistance, footprint reuse, ring-buffer wraparound, isolated resets,
and actual PhysX force application using free-flight momentum balance. It then
measures fixed-target standing and applies the same recorded thresholds to rigid
and soft runs. Failure returns a nonzero exit code. Each new diagnostic run saves
`env.yaml`, `summary.json`, `measurements.npz` and `physics.png` under
`logs/rsl_rl/isaaclab/go2_dreamwaq_soft_physics/<run>/`.

`--physics_dt` changes only a diagnostic run's timestep and synchronizes its
decimation, rendering interval and contact sampling, retaining a 20 ms control
period and height scan interval. The sweep verifies resolved configuration and
startup-weight invariants across matched rigid/soft cases. Each case also saves
joint/foot positions, reported velocities, position-difference velocities, world
wrenches and horizontal-drag work estimates in `measurements.npz`, with a
`velocity_audit.png` figure. Configuration-based and velocity-trapezoid work
estimates are distinct diagnostics; neither alone is total-system energy balance.

Saved sweeps can be combined with `--report_sweep <sweep_dir> [<sweep_dir> ...]`;
this creates a new comparison report without rerunning physics or replacing old
results. The report includes `comparison.json` and `dt_convergence.png`. A sweep
returns nonzero if any case fails acceptance, including its deliberately retained
5 ms baseline.

The original 5 ms physics timestep passes the force-law checks but fails free-joint
standing because of persistent leg oscillation; the rigid control passes.
`--lock_joints` isolates bearing and does **not** certify free-joint stability.
Experimental `--floor_stiffness`, `--normal_damping`, `--drag_damping_cap` and
`--drag_scale` overrides are recorded per run. Trial values have not been adopted
as training defaults. The network, PPO and actuator gains remain unchanged.

The paired convergence audit (4 environments, seed 1, 8 seconds) reduced mean
joint speed from 5.24 rad/s at 5 ms to 1.16 at 2.5 ms and 0.417 at 1.25 ms.
Those cases failed the original standing criteria. At 0.625 ms the mean was
0.133 rad/s and acceptance passed, although a small two-step oscillation remained.
At 0.3125 ms it fell to 0.00348 rad/s and that oscillation disappeared; both
corresponding rigid controls also passed. The resolved timestep was the only
experimental setting varied, with dependent timing fields synchronized to
retain 50 Hz control. This supports a discrete-coupling instability diagnosis.

The migrated soft task and its matched flat rigid control default to 0.3125 ms
and 64 physics substeps per control step. `soft-train`, `soft-play`, `soft-eval`,
`soft-check` and the default `soft-physics` use this validated coupling timestep.
The original rough task keeps its original timing. For RL/play, 48
contact-history samples retain the original 15 ms window
(3 samples at 5 ms); the standalone physics audit bypasses RL rewards and
terminations. Proprioceptive history stays at 5 policy frames and 50 Hz.
Static acceptance does not establish locomotion performance, multiple-seed
robustness, or training throughput.

An optional inference-only paired rollout measures frozen-policy transfer:

```bash
bash scripts/isaac.sh soft-rollout --headless --num_envs 4 \
  --checkpoint logs/rsl_rl/isaaclab/go2_dreamwaq_rough/2026-10-18_14-35-23_clean_seed1/model_1999.pt \
  --phase_seconds 3 --cycles 2 --seeds 1,2,3
```

This runs fresh rigid/soft processes for each seed, with identical 0.3125 ms
physics, 50 Hz control, 48 contact-history samples, the original play settings,
and five command phases: stand, forward at 0.3 m/s, lateral at 0.2 m/s, turn at
0.4 rad/s, and stand again. The example repeats the schedule for 30 seconds per
case. Original episode timeouts and non-timeout termination rules remain active;
states and termination reasons are captured **before** automatic resets. Soft
resets also check footprint-memory clearing and isolation from other environments.
The original CENet samples a latent even in inference; an isolated per-step Torch
RNG gives each pair the same samples without changing the model or consuming the
environment's random stream. Checkpoint and policy hashes verify frozen weights.

Results go to `logs/rsl_rl/isaaclab/go2_dreamwaq_soft_rollout/<batch>/`, including
each case's resolved environment/agent configuration, measurements, summary and
the batch comparison plot/JSON. The report checks paired configuration, initial
poses, masses, inertias, materials, actor observations and first actions. Initial
privileged height scans allow 10 micrometres of ray-intersection rounding between
the rigid plane and visual soil mesh; these scans do not enter the actor. Native
joint positions and velocities are recorded at every physics substep so that
50 Hz sampling cannot hide fast oscillations. Tracking metrics discard the first
second of each phase; overall metrics retain every sample and every reset.
Runtime includes diagnostic recording and is not a training-throughput estimate.
Rebuild a report without repeating simulation with `soft-rollout --report <batch>`.
Reports also summarize each phase before each environment's first reset, and
separate terminations before/after the first timeout. Resume an interrupted batch
using the same command plus `--resume --out <batch>`: completed matching cases
are reused, while partial cases/logs are preserved in place and retried in new
directories. Other output directories must be new. This evaluates transfer of a rigid-trained
policy; poor soft-ground tracking alone does not establish a physics failure.

The 3-seed audit (4 environments per case, 30 seconds, frozen `model_1999.pt`)
passed configuration/initial-state parity and all 20 soft-memory reset checks.
Before the first reset, mean forward speed under a 0.3 m/s command was 0.178 m/s
on rigid ground and -0.013 m/s on soil; mean yaw rate under a 0.4 rad/s command
was 0.390 and -0.068 rad/s respectively. Both lateral responses were weak.
Neither group had a base-contact termination before its first 20-second timeout;
after timeouts there were 12 rigid and 8 soft base-contact terminations. No
pre-reset soft sample exceeded 5% body weight in hard-floor support, and the
largest phase high-frequency joint-velocity power fraction was 0.029% (top 20%
of the available frequency band). Frozen-policy tracking and post-reset recovery
are transfer observations, not pass/fail criteria for migrating the soil model.
Migration acceptance concerns force-law fidelity, PhysX wrench application,
contact feedback, numerical stability and per-environment soil-state resets.
Real-soil calibration and training throughput remain unvalidated.
The saved batch is
`logs/rsl_rl/isaaclab/go2_dreamwaq_soft_rollout/2026-10-22_23-22-45_467893_paired/`.

To measure forgetting on the original rigid rough task, evaluate each checkpoint
with the same seed and command schedule:

```bash
bash scripts/isaac.sh soft-rollout --single --original_rough --headless \
  --num_envs 16 --phase_seconds 4 --cycles 2 --seed 1 \
  --checkpoint /absolute/path/model_1999.pt
```

This selects the original `Isaac-Velocity-Rough-Go2-DreamWaQ-Play-v0` task,
including its rough terrain generator, native 5 ms physics / 50 Hz control,
3-sample contact history and clean Play observations. No soil adapter is used.
Only the evaluation command schedule replaces the random velocity commands;
the terrain mesh is seeded and hashed, and initial states/terrain assignments
are saved for checking parity across checkpoints. Native episode termination,
timeouts and terrain curriculum remain active; after resets the trajectories
and terrain levels can diverge. Results go to
`logs/rsl_rl/isaaclab/go2_dreamwaq_forgetting/<run>/`, including rewards,
completed episode lengths, termination reasons and first-failure times.
Add `--command_scale 2` to test 0.6 m/s forward, 0.4 m/s lateral and 0.8 rad/s
yaw using the same schedule; use identical scales for both checkpoints.

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
