# DreamWaQPPO - PPO subclass that adds CENet's hybrid loss to the update.
# rsl_rl's PPO.update() calls loss.backward() + optimizer.step() inside the
# minibatch loop, so there's no clean post-hook to add an auxiliary loss.
# Reproduce the update path (PPO clip + value loss + adaptive-LR
# KL schedule) - the only branches this project uses (no RND, no symmetry, no
# multi-GPU, no recurrent) - and add the CENet hybrid loss before backward().
# A single optimizer step then trains actor, critic, AND CENet together
# (matches the paper's "synchronously train policy, value, and CENet").
# CENet loss = MSE(v_hat, true_vel) + MSE(recon, obs_t) + beta * KL(z||N(0,I)).
# Velocity target comes from the clean 'privileged_vel' group (true base linear
# velocity, never seen by the actor/critic). Reconstruction target is the
# current single-frame 'policy' obs (o_t) from the minibatch.
# TUNING: cenet_loss_coef scales the whole CENet loss when added to the PPO
# loss (the recon term can be ~100x the PPO losses, so it may dominate the
# shared optimizer). beta scales the KL term inside the CENet loss.
# Defaults below are used if the env vars are unset.

from __future__ import annotations

import os

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

# Defaults; override at launch with env vars (see header).
_DEFAULT_CENET_LOSS_COEF = float(os.environ.get("DREAMWAQ_CENET_LOSS_COEF", "1.0"))
_DEFAULT_CENET_BETA = float(os.environ.get("DREAMWAQ_CENET_BETA", "1.0"))


class DreamWaQPPO(PPO):
    """PPO + CENet hybrid loss. Assumes non-recurrent policy, no RND/symmetry."""

    def __init__(self, *args, cenet_beta: float | None = None,
                 cenet_loss_coef: float | None = None, vel_dim: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.cenet_beta = _DEFAULT_CENET_BETA if cenet_beta is None else cenet_beta
        self.cenet_loss_coef = _DEFAULT_CENET_LOSS_COEF if cenet_loss_coef is None else cenet_loss_coef
        self.vel_dim = vel_dim
        print(f"[DreamWaQ] PPO using cenet_loss_coef={self.cenet_loss_coef}, "
              f"cenet_beta={self.cenet_beta}")

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_cenet_vel = 0.0
        mean_cenet_recon = 0.0
        mean_cenet_kl = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # Recompute log-prob, value, distribution stats for this batch.
            # policy.act runs get_actor_obs, which runs CENet and caches outputs.
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # Adaptive LR via KL (same as parent).
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # PPO surrogate loss.
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss (clipped, same as parent).
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # ---- CENet hybrid loss ----
            cenet = self.policy.last_cenet_outputs
            # True base linear velocity from the clean CENet-target-only group.
            # (No longer sliced from policy obs, which no longer contains it.)
            true_vel = obs_batch["privileged_vel"][:, : self.vel_dim].detach()
            recon_target = obs_batch["policy"].detach()
            recon_pred = cenet["recon"]
            if recon_pred.shape[-1] != recon_target.shape[-1]:
                m = min(recon_pred.shape[-1], recon_target.shape[-1])
                recon_pred = recon_pred[:, :m]
                recon_target = recon_target[:, :m]

            mse_vel = nn.functional.mse_loss(cenet["vel_hat"], true_vel)
            mse_recon = nn.functional.mse_loss(recon_pred, recon_target)
            mu, logvar = cenet["mu"], cenet["logvar"]
            kl_cenet = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            cenet_loss = mse_vel + mse_recon + self.cenet_beta * kl_cenet

            loss = loss + self.cenet_loss_coef * cenet_loss
            # ---- end CENet ----

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_cenet_vel += mse_vel.item()
            mean_cenet_recon += mse_recon.item()
            mean_cenet_kl += kl_cenet.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_cenet_vel /= num_updates
        mean_cenet_recon /= num_updates
        mean_cenet_kl /= num_updates

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "cenet_vel_mse": mean_cenet_vel,
            "cenet_recon_mse": mean_cenet_recon,
            "cenet_kl": mean_cenet_kl,
        }
