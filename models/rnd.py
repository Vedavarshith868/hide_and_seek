# -*- coding: utf-8 -*-
"""
Random Network Distillation (RND) intrinsic motivation module.

Fixed from original:
  - Input is now 128-D encoder hidden state (not raw LiDAR)
  - Running stats use proper EMA (not broken Welford)
  - Predictor update is mini-batched (4 minibatches)
  - Spatial and interaction channels weighted 0.5 each when both active
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RNDNet(nn.Module):
    def __init__(self, in_dim: int = 128, h: int = 256, out: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModule(nn.Module):
    def __init__(self, spatial_dim: int = 128, interaction_dim: int = 128, lr: float = 3e-4):
        super().__init__()
        self.target_spatial    = RNDNet(spatial_dim)
        self.predictor_spatial = RNDNet(spatial_dim)

        self.target_interact    = RNDNet(interaction_dim)
        self.predictor_interact = RNDNet(interaction_dim)

        for p in self.target_spatial.parameters():
            p.requires_grad_(False)
        for p in self.target_interact.parameters():
            p.requires_grad_(False)

        self.optim = optim.Adam(
            list(self.predictor_spatial.parameters()) + list(self.predictor_interact.parameters()),
            lr=lr
        )

        # EMA-based running stats (replaces broken Welford)
        self._ema_alpha = 0.01
        self.register_buffer('ema_mean_s', torch.zeros(1))
        self.register_buffer('ema_var_s', torch.ones(1))
        self.register_buffer('ema_mean_i', torch.zeros(1))
        self.register_buffer('ema_var_i', torch.ones(1))

    @torch.no_grad()
    def intrinsic_reward(self, spatial_ctx: torch.Tensor, interact_ctx: torch.Tensor, use_interaction: bool) -> torch.Tensor:
        # Spatial channel
        n_s = spatial_ctx + torch.randn_like(spatial_ctx) * 0.01
        raw_s = (self.target_spatial(n_s) - self.predictor_spatial(n_s)).pow(2).mean(-1)

        # EMA update for spatial stats
        alpha = self._ema_alpha
        self.ema_mean_s = (1 - alpha) * self.ema_mean_s + alpha * raw_s.mean()
        self.ema_var_s  = (1 - alpha) * self.ema_var_s  + alpha * raw_s.var().clamp(min=1e-8)

        reward_s = torch.clamp((raw_s - self.ema_mean_s) / (self.ema_var_s + 1e-8).sqrt(), min=0.0)

        if not use_interaction:
            return reward_s

        # Interaction channel
        n_i = interact_ctx + torch.randn_like(interact_ctx) * 0.01
        raw_i = (self.target_interact(n_i) - self.predictor_interact(n_i)).pow(2).mean(-1)

        # EMA update for interaction stats
        self.ema_mean_i = (1 - alpha) * self.ema_mean_i + alpha * raw_i.mean()
        self.ema_var_i  = (1 - alpha) * self.ema_var_i  + alpha * raw_i.var().clamp(min=1e-8)

        reward_i = torch.clamp((raw_i - self.ema_mean_i) / (self.ema_var_i + 1e-8).sqrt(), min=0.0)

        # Weight 0.5 each when both channels active (prevents 2x reward spike)
        return 0.5 * reward_s + 0.5 * reward_i

    def update(self, spatial_ctx: torch.Tensor, interact_ctx: torch.Tensor, use_interaction: bool, n_minibatches: int = 4) -> tuple[float, float]:
        """Mini-batched predictor update (prevents giant single-step overshoot)."""
        n = spatial_ctx.shape[0]
        mb_size = max(1, n // n_minibatches)
        indices = torch.randperm(n, device=spatial_ctx.device)

        total_loss_s = 0.0
        total_loss_i = 0.0

        for start in range(0, n, mb_size):
            mb_idx = indices[start:start + mb_size]
            mb_spatial = spatial_ctx[mb_idx]

            loss_s = F.mse_loss(self.predictor_spatial(mb_spatial), self.target_spatial(mb_spatial).detach())

            loss_i = torch.tensor(0.0, device=spatial_ctx.device)
            if use_interaction:
                mb_interact = interact_ctx[mb_idx]
                loss_i = F.mse_loss(self.predictor_interact(mb_interact), self.target_interact(mb_interact).detach())

            total_loss = loss_s + loss_i
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            total_loss_s += loss_s.item()
            total_loss_i += loss_i.item()

        n_batches = max(1, (n + mb_size - 1) // mb_size)
        return total_loss_s / n_batches, total_loss_i / n_batches
