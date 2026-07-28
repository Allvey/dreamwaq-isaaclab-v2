# DreamWaQActorCritic - rsl_rl ActorCritic that wires in CENet.
# The actor receives [policy_obs, v_hat, z], where (v_hat, z) come from CENet
# applied to the proprioception-history group. Per our design decision
# (faithful to the paper: "synchronously train policy, value, and CENet"),
# the CENet outputs fed to the actor are DETACHED. The policy gradient does
# NOT flow back into CENet. CENet is trained by its own hybrid loss, added in
# the PPO subclass (Stage 4c).
# CENet's raw (non-detached) outputs + KL targets are exposed via
# `last_cenet_outputs` so the PPO update can compute the hybrid loss.

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization

try:
    from .cenet import CENet          # package context (Isaac Lab runtime)
except ImportError:                   # standalone context (offline unit test)
    from cenet import CENet


class DreamWaQActorCritic(ActorCritic):
    """ActorCritic + CENet. Actor input = policy_obs(45) + v_hat(3) + z(16) = 64."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # CENet-specific
        history_group: str = "prop_history",
        cenet_latent_dim: int = 16,
        cenet_vel_dim: int = 3,
        cenet_encoder_hidden: list[int] | None = None,
        cenet_decoder_hidden: list[int] | None = None,
        # passthrough to parent
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs,
    ) -> None:
        # Let the parent build a standard actor/critic first (actor input = policy dim).
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std,
            **kwargs,
        )

        self.history_group = history_group
        self.cenet_latent_dim = cenet_latent_dim
        self.cenet_vel_dim = cenet_vel_dim

        #CENet input width and single-frame obs width from the env observations.
        hist_dim = obs[history_group].shape[-1]          # 225 = 45 * 5
        single_dim = 0
        for g in obs_groups["policy"]:
            single_dim += obs[g].shape[-1]               # 45
        self._obs_dim = single_dim
        # history_len inferred so CENet's decoder reconstructs a single frame (48)
        assert hist_dim % single_dim == 0, (
            f"history dim {hist_dim} not divisible by single-frame dim {single_dim}"
        )
        history_len = hist_dim // single_dim

        # Build CENet
        self.cenet = CENet(
            obs_dim=single_dim,
            history_len=history_len,
            latent_dim=cenet_latent_dim,
            vel_dim=cenet_vel_dim,
            encoder_hidden=cenet_encoder_hidden,
            decoder_hidden=cenet_decoder_hidden,
        )

        # Rebuild the actor so its input includes v_hat + z.
        num_actor_obs = single_dim + cenet_vel_dim + cenet_latent_dim   # 45 + 3 + 16 = 64
        if state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"[DreamWaQ] Rebuilt Actor MLP (input={num_actor_obs}): {self.actor}")
        print(f"[DreamWaQ] CENet input={hist_dim} (history_len={history_len}), "
              f"latent={cenet_latent_dim}, vel={cenet_vel_dim}")

        # Cache for the PPO update to read (set on each get_actor_obs call).
        self.last_cenet_outputs: dict[str, torch.Tensor] | None = None

    def _run_cenet(self, obs: TensorDict):
        """Run CENet on the history group; return (vel_hat, z, recon, mu, logvar)."""
        hist = obs[self.history_group]
        return self.cenet(hist)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        # Single-frame policy obs (same as parent).
        policy_obs = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
        # CENet outputs.
        vel_hat, z, recon, mu, logvar = self._run_cenet(obs)
        # Cache non-detached tensors so the PPO update can compute CENet's loss.
        self.last_cenet_outputs = {
            "vel_hat": vel_hat, "z": z, "recon": recon, "mu": mu, "logvar": logvar,
        }
        # Feed DETACHED estimates to the actor (policy grad does not reach CENet).
        return torch.cat([policy_obs, vel_hat.detach(), z.detach()], dim=-1)
