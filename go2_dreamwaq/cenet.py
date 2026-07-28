# CENet - Context-aware Estimator Network 
# Shared encoder - (a) velocity head [MSE], (b) beta-VAE head [recon + KL].


from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, hidden: list[int], activation=nn.ELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), activation()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class CENet(nn.Module):
    """Context-aware Estimator Network.

    Input:  flattened proprioceptive history, shape (B, obs_dim * history_len).
    Outputs:
        vel_hat: (B, 3)        estimated base linear velocity
        z:       (B, latent)   sampled latent context (reparameterized)
        recon:   (B, obs_dim)  reconstruction of the CURRENT observation o_t
        mu, logvar: (B, latent) for the KL term
    """

    def __init__(
        self,
        obs_dim: int = 45,
        history_len: int = 5,
        latent_dim: int = 16,
        vel_dim: int = 3,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.latent_dim = latent_dim
        self.vel_dim = vel_dim

        in_dim = obs_dim * history_len
        encoder_hidden = encoder_hidden or [256, 128]
        decoder_hidden = decoder_hidden or [64, 128]

        # Shared encoder trunk
        self.encoder = _mlp(in_dim, encoder_hidden[-1], encoder_hidden[:-1])

        # Velocity head: regress true body velocity (MSE target)
        self.vel_head = nn.Linear(encoder_hidden[-1], vel_dim)

        # beta-VAE head: produce mu and logvar for latent z
        self.fc_mu = nn.Linear(encoder_hidden[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_hidden[-1], latent_dim)

        # Decoder: reconstruct CURRENT observation o_t from z
        self.decoder = _mlp(latent_dim, obs_dim, decoder_hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs_history: torch.Tensor):
        feat = self.encoder(obs_history)
        vel_hat = self.vel_head(feat)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return vel_hat, z, recon, mu, logvar

    @staticmethod
    def loss(vel_hat, vel_true, recon, obs_t, mu, logvar, beta: float = 1.0):
        """Hybrid loss = MSE(v) + MSE(recon, o_t) + beta * KL(z || N(0,I))."""
        
        mse_vel = nn.functional.mse_loss(vel_hat, vel_true)
        mse_recon = nn.functional.mse_loss(recon, obs_t)
        # KL for diagonal Gaussian vs standard normal, averaged over batch
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        total = mse_vel + mse_recon + beta * kl
        return total, {"mse_vel": mse_vel, "mse_recon": mse_recon, "kl": kl}
