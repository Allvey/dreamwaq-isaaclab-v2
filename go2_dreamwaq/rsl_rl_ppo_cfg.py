# DreamWaQ PPO runner config.
# asymmetric obs_groups (critic sees privileged height scan).
# point policy/algorithm at the DreamWaQ custom classes.
# CENet hyperparameters use the subclass defaults
# (latent=16, vel=3, history_group="prop_history", beta=1.0).
# (critic fix): give the critic the privileged true base velocity.
# Path A removed base_lin_vel from the 'policy' group (correct for the
# actor), but the critic reads that group too, so it lost true velocity.
# An asymmetric critic SHOULD be privileged, and the dominant reward is
# velocity tracking, so we add the clean 'privileged_vel' group to the
# critic only. Critic input: 45 + 187 + 3 = 235. Actor stays 64.

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)


@configclass
class Go2DreamWaQRoughPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_dreamwaq_rough"

        # Asymmetric actor-critic (DreamWaQ contribution #1):
        #   actor  -> proprioception only (45) + CENet's v_hat(3) + z(16) = 64
        #   critic -> proprioception (45) + height_scan (187) + TRUE base vel (3) = 235
        self.obs_groups = {
            "policy": ["policy"],
            "critic": ["policy", "height_scan", "privileged_vel"],
        }

        # Use the DreamWaQ custom classes (CENet integration):
        self.policy.class_name = "DreamWaQActorCritic"
        self.algorithm.class_name = "DreamWaQPPO"
