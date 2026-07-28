# Offline test for DreamWaQActorCritic. Needs
# torch + tensordict (use: ~/IsaacLab/isaaclab.sh -p <thisfile>).
#
# Verifies:
#   1. Actor MLP input grew to 64 (45 policy + 3 vel + 16 z).
#   2. Critic input is 235 (45 policy + 187 height_scan + 3 privileged_vel).
#   3. act()/evaluate() run and produce correct action/value shapes.
#   4. CENet outputs are cached for the PPO loss.
#   5. The DETACH holds: backprop from an actor-side loss does NOT reach CENet.

import torch
from tensordict import TensorDict

from dreamwaq_actor_critic import DreamWaQActorCritic


def main():
    B = 32
    obs = TensorDict(
        {
            "policy": torch.randn(B, 45),
            "height_scan": torch.randn(B, 187),
            "prop_history": torch.randn(B, 225),  # 45 * 5
            "privileged_vel": torch.randn(B, 3),  # CENet target + critic-only privileged
        },
        batch_size=[B],
    )
    # Critic is privileged: it sees true base velocity (privileged_vel); the actor
    # never does. This matches the asymmetric design and the baseline's critic.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "height_scan", "privileged_vel"],
    }

    ac = DreamWaQActorCritic(
        obs=obs, obs_groups=obs_groups, num_actions=12,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # 1. actor input dim
    first_actor_linear = ac.actor[0]
    assert first_actor_linear.in_features == 64, f"actor in={first_actor_linear.in_features}"
    print(f"[OK] actor input = {first_actor_linear.in_features} (45 + 3 + 16)")

    # 2. critic now includes privileged true velocity
    first_critic_linear = ac.critic[0]
    assert first_critic_linear.in_features == 235, f"critic in={first_critic_linear.in_features}"
    print(f"[OK] critic input = {first_critic_linear.in_features} (45 + 187 + 3 privileged_vel)")

    # 3. forward passes
    actions = ac.act(obs)
    values = ac.evaluate(obs)
    assert actions.shape == (B, 12), f"actions {actions.shape}"
    assert values.shape == (B, 1), f"values {values.shape}"
    print(f"[OK] act -> {tuple(actions.shape)}, evaluate -> {tuple(values.shape)}")

    # 4. cenet outputs cached
    assert ac.last_cenet_outputs is not None
    co = ac.last_cenet_outputs
    assert co["vel_hat"].shape == (B, 3) and co["z"].shape == (B, 16) and co["recon"].shape == (B, 45)
    print(f"[OK] cenet outputs cached: vel_hat {tuple(co['vel_hat'].shape)}, "
          f"z {tuple(co['z'].shape)}, recon {tuple(co['recon'].shape)}")

    # 5. detach check: a loss on the actor output must NOT produce CENet gradients.
    ac.zero_grad()
    actor_obs = ac.get_actor_obs(obs)         # fresh forward (re-caches cenet outputs)
    fake_actor_loss = ac.actor(actor_obs).pow(2).mean()
    fake_actor_loss.backward()
    cenet_grads = [p.grad for p in ac.cenet.parameters() if p.grad is not None]
    assert len(cenet_grads) == 0, (
        f"DETACH FAILED: CENet received {len(cenet_grads)} gradient tensors from actor loss"
    )
    print("[OK] detach holds: actor loss did NOT backprop into CENet.")

    # And confirm CENet IS trainable from its own outputs (sanity, opposite direction).
    ac.zero_grad()
    _, _, recon, mu, logvar = ac.cenet(obs["prop_history"])
    cenet_self_loss = recon.pow(2).mean() + mu.pow(2).mean() + logvar.pow(2).mean()
    cenet_self_loss.backward()
    enc_grad = ac.cenet.encoder[0].weight.grad
    assert enc_grad is not None and torch.isfinite(enc_grad).all()
    print("[OK] CENet IS trainable from its own loss (encoder receives gradient).")

    print("\nAll DreamWaQActorCritic checks passed.")


if __name__ == "__main__":
    main()
