# DreamWaQ task registration + custom-class registration for rsl_rl.

import gymnasium as gym

from . import agents

# ---------------------------------------------------------------------------
# Make our custom rsl_rl classes resolvable by the runner.
# ---------------------------------------------------------------------------
def _register_dreamwaq_classes():
    try:
        import rsl_rl.runners.on_policy_runner as _runner_mod
        from .dreamwaq_actor_critic import DreamWaQActorCritic
        from .dreamwaq_ppo import DreamWaQPPO

        _runner_mod.DreamWaQActorCritic = DreamWaQActorCritic
        _runner_mod.DreamWaQPPO = DreamWaQPPO
        print("[DreamWaQ] Registered DreamWaQActorCritic + DreamWaQPPO with rsl_rl runner.")
    except Exception as e:  
        # Don't hard-fail env import if rsl_rl isn't present (e.g. skrl-only use).
        print(f"[DreamWaQ] WARNING: could not register custom rsl_rl classes: {e}")


_register_dreamwaq_classes()

##
# Register environments.
##

gym.register(
    id="Isaac-Velocity-Rough-Go2-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2DreamWaQRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2DreamWaQRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-DreamWaQ-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2DreamWaQRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2DreamWaQRoughPPORunnerCfg",
    },
)

print("[DreamWaQ] Registered Isaac-Velocity-Rough-Go2-DreamWaQ-v0")
