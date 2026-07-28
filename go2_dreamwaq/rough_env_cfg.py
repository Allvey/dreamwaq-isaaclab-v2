# DreamWaQ fork of the Unitree Go2 rough velocity task.
# asymmetric actor-critic (actor = proprioception, critic += height scan).
# add a flattened H=5 proprioception-history group ('prop_history')
# as CENet's input.
# (fidelity fix): base linear velocity is unmeasurable on hardware and
# is the quantity CENet must ESTIMATE, so it is removed from both the
# actor-visible 'policy' group and CENet's 'prop_history' input. It is
# kept only in a clean, CENet-target-only group 'privileged_vel'.
# Single-frame proprioception is therefore 45 (was 48).

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
    UnitreeGo2RoughEnvCfg_PLAY,
)

# Proprioception history length for CENet (paper uses 50; this project uses 5).
PROP_HISTORY_LEN = 5


@configclass
class DreamWaQObservationsCfg:
    """Observation groups:
      - policy:         single-frame proprioception (45)  -> actor (direct) + critic
      - height_scan:    privileged terrain scan (187)      -> critic only
      - prop_history:   flattened H=5 proprioception (225) -> CENet input only
      - privileged_vel: true base linear velocity (3)      -> CENet target only
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Single-frame proprioception. Actor reads this directly; critic reads it too.

        base_lin_vel is intentionally absent: it is unmeasurable on hardware and is
        the quantity CENet estimates (v_hat), which is concatenated onto the actor
        input in DreamWaQActorCritic. Dim = 45.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HeightScanCfg(ObsGroup):
        """Privileged terrain height scan. critic only."""

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PropHistoryCfg(ObsGroup):
        """Flattened proprioception history (H frames) CENet input only.

        Same terms as PolicyCfg (no base_lin_vel). With history_length set on the
        group and flatten_history_dim=True, the group emits (num_envs, 45 * H).
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # Group-level history overrides each term; flatten into a single vector.
            self.history_length = PROP_HISTORY_LEN
            self.flatten_history_dim = True

    @configclass
    class PrivilegedVelCfg(ObsGroup):
        """True base linear velocity (3) — CENet's velocity-head target ONLY.

        Read only inside the PPO update (DreamWaQPPO) as the MSE target for
        v_hat; never seen by the actor or the critic. Clean (no noise): the
        estimator regresses ground-truth body velocity, as in the paper.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    height_scan: HeightScanCfg = HeightScanCfg()
    prop_history: PropHistoryCfg = PropHistoryCfg()
    privileged_vel: PrivilegedVelCfg = PrivilegedVelCfg()


@configclass
class Go2DreamWaQRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    """DreamWaQ env config. Stage 4a: policy + height_scan + prop_history groups."""

    observations: DreamWaQObservationsCfg = DreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class Go2DreamWaQRoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg_PLAY):
    """Play variant for evaluation/recording."""

    observations: DreamWaQObservationsCfg = DreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # disable corruption for clean evaluation
        self.observations.policy.enable_corruption = False
        self.observations.height_scan.enable_corruption = False
        self.observations.prop_history.enable_corruption = False
