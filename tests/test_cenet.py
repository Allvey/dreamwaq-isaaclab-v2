# Offline unit test for CENet — no Isaac Sim required.
# Verifies output shapes and that the hybrid loss backpropagates.

import torch
from cenet import CENet


def main():
    B, obs_dim, H, latent = 64, 45, 5, 16
    net = CENet(obs_dim=obs_dim, history_len=H, latent_dim=latent, vel_dim=3)

    obs_hist = torch.randn(B, obs_dim * H)
    vel_true = torch.randn(B, 3)
    obs_t = torch.randn(B, obs_dim)

    vel_hat, z, recon, mu, logvar = net(obs_hist)

    assert vel_hat.shape == (B, 3), f"vel_hat {vel_hat.shape}"
    assert z.shape == (B, latent), f"z {z.shape}"
    assert recon.shape == (B, obs_dim), f"recon {recon.shape}"
    assert mu.shape == (B, latent), f"mu {mu.shape}"
    assert logvar.shape == (B, latent), f"logvar {logvar.shape}"
    print("[OK] output shapes:",
          "vel_hat", tuple(vel_hat.shape),
          "z", tuple(z.shape),
          "recon", tuple(recon.shape))

    total, parts = CENet.loss(vel_hat, vel_true, recon, obs_t, mu, logvar, beta=1.0)
    total.backward()

    g = net.encoder[0].weight.grad
    assert g is not None and torch.isfinite(g).all(), "no/invalid gradient on encoder"
    print(f"[OK] loss={total.item():.4f} "
          f"(mse_vel={parts['mse_vel'].item():.4f}, "
          f"mse_recon={parts['mse_recon'].item():.4f}, "
          f"kl={parts['kl'].item():.4f})")
    print("[OK] gradient flows to shared encoder. CENet is wired correctly.")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"[INFO] CENet parameters: {n_params:,}")


if __name__ == "__main__":
    main()
