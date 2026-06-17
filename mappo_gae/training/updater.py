# -*- coding: utf-8 -*-
"""
MAPPO updater: GAE computation + PPO update step.

Advantage source is selectable (constants.ADV_MODE):
  - 'gae'  : per-team GAE advantages from the global value network
             (standard MAPPO — low variance, proven; default)
  - 'coma' : counterfactual advantages from the centralized Q critic
             (the COMA lever; Q is trained on the hider team's lambda-return
              and seeker advantages are negated, valid because the reward is
              exactly zero-sum)

The COMA critic is trained in both modes so the mode can be switched
mid-run without a cold critic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from constants import (
    MAX_ENTITIES, FEAT_DIM, LIDAR_DIM, GLOBAL_DIM, HIDDEN_DIM, N_AGENTS,
    ADV_MODE,
)


class MAPPOUpdater:
    def __init__(
        self,
        actor,
        coma_critic,
        global_value_net,
        coma_engine,
        lr: float = 3e-4,
        clip_param: float = 0.2,
        entropy_start: float = 0.02,
        entropy_end: float = 0.005,
        entropy_decay_steps: int = 10000,
        value_coef: float = 0.5,
        adv_mode: str = ADV_MODE,
    ):
        self.actor            = actor
        self.coma_critic      = coma_critic
        self.global_value_net = global_value_net
        self.coma_engine      = coma_engine
        self.clip_param       = clip_param
        self.entropy_start    = entropy_start
        self.entropy_end      = entropy_end
        self.entropy_decay_steps = entropy_decay_steps
        self.value_coef       = value_coef
        self.adv_mode         = adv_mode
        self.total_steps      = 40000
        self.initial_lr       = lr
        self.update_step      = 0
        self.last_entropy_coef = entropy_start
        self.optimizer = optim.Adam(
            list(actor.parameters())
            + list(coma_critic.parameters())
            + list(global_value_net.parameters()),
            lr=lr,
            eps=1e-5,
        )

    def _current_entropy_coef(self) -> float:
        """Linear decay from entropy_start to entropy_end over entropy_decay_steps."""
        t = min(self.update_step / self.entropy_decay_steps, 1.0)
        return self.entropy_start + (self.entropy_end - self.entropy_start) * t

    # ── GAE ──────────────────────────────────────────────────────────────────
    def compute_gae(self, rewards, values, values_next, dones,
                    gamma: float = 0.998, gae_lambda: float = 0.95):
        """
        rewards, values, values_next have shape (episode_len, num_envs, n_agents).
        dones is (episode_len, num_envs) — unsqueezed inside.
        dones[t]=1 means the episode ended at step t (s_{t+1} is a reset state).
        """
        advantages = torch.zeros_like(rewards)
        last_gae   = 0
        for t in reversed(range(rewards.shape[0])):
            mask  = (1.0 - dones[t]).unsqueeze(-1)           # (E, 1)
            delta = rewards[t] + gamma * values_next[t] * mask - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * mask * last_gae
        returns = advantages + values
        return advantages, returns

    # ── main update ──────────────────────────────────────────────────────────
    def update(self, buffer, next_global_state, w_int, seeker_wr: float = 0.5,
               num_epochs: int = 2, mini_batch_size: int = 2048):

        self.update_step += 1
        entropy_coef = self._current_entropy_coef()
        self.last_entropy_coef = entropy_coef

        # ── compute value targets ─────────────────────────────────────────────
        with torch.no_grad():
            v_h, v_s, v_int = self.global_value_net(
                buffer.global_states.view(-1, GLOBAL_DIM)
            )
            T, E = buffer.episode_len, buffer.num_envs
            A    = buffer.n_agents

            v_h   = v_h.view(T, E, 1)
            v_s   = v_s.view(T, E, 1)
            v_int = v_int.view(T, E, 1).expand(-1, -1, A)
            v_ext = torch.cat([v_h, v_h, v_s, v_s], dim=2)   # (T, E, 4)

            next_vh, next_vs, next_vint = self.global_value_net(next_global_state)
            next_vext = torch.cat(
                [next_vh, next_vh, next_vs, next_vs], dim=1
            )                                                  # (E, 4)
            next_vint_exp = next_vint.expand(-1, A)            # (E, 4)

            values_next_ext = torch.cat([v_ext[1:], next_vext.unsqueeze(0)], dim=0)
            values_next_int = torch.cat([v_int[1:], next_vint_exp.unsqueeze(0)], dim=0)

            gae_adv_ext, ret_ext = self.compute_gae(
                buffer.rewards_ext, v_ext,
                values_next_ext,
                buffer.dones, gamma=0.998,
            )
            adv_int, ret_int = self.compute_gae(
                buffer.rewards_int, v_int,
                values_next_int,
                buffer.dones, gamma=0.99,
            )

            w = w_int.unsqueeze(0).unsqueeze(0).expand(T, E, A)

        # ── flatten buffer for PPO minibatches ────────────────────────────────
        flat_entities = buffer.entities.view(-1, A, MAX_ENTITIES, FEAT_DIM)
        flat_masks    = buffer.masks.view(-1, A, MAX_ENTITIES)
        flat_lidar    = buffer.lidar.view(-1, A, LIDAR_DIM)
        flat_gstates  = buffer.global_states.view(-1, GLOBAL_DIM)
        flat_actions  = buffer.actions.view(-1, A, 4)
        flat_joint_oh = buffer.joint_actions_oh.view(-1, 64)
        old_log_probs = buffer.log_probs.view(-1, A)

        flat_ret_ext  = ret_ext.view(-1, A)
        flat_ret_int  = ret_int.view(-1, A)
        flat_gae_adv  = gae_adv_ext.view(-1, A)
        flat_adv_int  = adv_int.view(-1, A)
        flat_w        = w.reshape(-1, A)

        # make hidden-state slices contiguous BEFORE the loop
        flat_h = buffer.hiddens_h.view(-1, A, HIDDEN_DIM).contiguous()
        flat_c = buffer.hiddens_c.view(-1, A, HIDDEN_DIM).contiguous()

        batch_size = flat_entities.shape[0]

        for epoch in range(num_epochs):
            indices = torch.randperm(batch_size, device=buffer.device)

            for start in range(0, batch_size, mini_batch_size):
                mb_idx = indices[start:start + mini_batch_size]

                mb_entities = flat_entities[mb_idx]
                mb_masks    = flat_masks[mb_idx]
                mb_lidar    = flat_lidar[mb_idx]
                mb_global   = flat_gstates[mb_idx]
                mb_actions  = flat_actions[mb_idx]
                mb_joint_oh = flat_joint_oh[mb_idx]
                mb_old_lp   = old_log_probs[mb_idx]

                mb_ret_ext  = flat_ret_ext[mb_idx]
                mb_ret_int  = flat_ret_int[mb_idx]
                mb_gae_adv  = flat_gae_adv[mb_idx]
                mb_adv_int  = flat_adv_int[mb_idx]
                mb_w        = flat_w[mb_idx]

                mb_h = flat_h[mb_idx]   # (mb, A, HIDDEN_DIM)
                mb_c = flat_c[mb_idx]

                # ── batched forward pass (all agents at once) ──────────────────
                B = mb_entities.shape[0]
                flat_ent = mb_entities.view(B * A, MAX_ENTITIES, FEAT_DIM)
                flat_msk = mb_masks.view(B * A, MAX_ENTITIES)
                flat_lid = mb_lidar.view(B * A, LIDAR_DIM)
                flat_h_state = (
                    mb_h.view(B * A, HIDDEN_DIM).unsqueeze(0).contiguous(),
                    mb_c.view(B * A, HIDDEN_DIM).unsqueeze(0).contiguous(),
                )
                flat_logits, _ = self.actor(flat_ent, flat_msk, flat_lid, flat_h_state)
                mb_logits = {
                    k: v.view(B, A, -1) for k, v in flat_logits.items()
                }

                # ── log-probs & entropy ───────────────────────────────────────
                new_log_probs = torch.zeros_like(mb_old_lp)
                entropy = torch.tensor(0.0, device=buffer.device)
                for a in range(A):
                    dm = Categorical(logits=mb_logits['move'][:, a])
                    dr = Categorical(logits=mb_logits['rot'][:, a])
                    dg = Categorical(logits=mb_logits['grab'][:, a])
                    dl = Categorical(logits=mb_logits['lock'][:, a])
                    new_log_probs[:, a] = (
                        dm.log_prob(mb_actions[:, a, 0])
                        + dr.log_prob(mb_actions[:, a, 1])
                        + dg.log_prob(mb_actions[:, a, 2])
                        + dl.log_prob(mb_actions[:, a, 3])
                    )
                    entropy = entropy + (
                        dm.entropy() + dr.entropy()
                        + dg.entropy() + dl.entropy()
                    ).mean()
                # Mean over agents (a sum quadruples the effective coefficient
                # because the actor parameters are shared across agents)
                entropy = entropy / A

                # ── advantages ────────────────────────────────────────────────
                with torch.no_grad():
                    if self.adv_mode == 'coma':
                        mb_adv_ext = self.coma_engine(mb_global, mb_joint_oh, mb_logits).clone()
                        mb_adv_ext[:, 2:] = -mb_adv_ext[:, 2:]   # zero-sum: negate seekers
                    else:
                        mb_adv_ext = mb_gae_adv

                    mb_advantages = mb_adv_ext + mb_w * mb_adv_int
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean(dim=0, keepdim=True)
                    ) / (mb_advantages.std(dim=0, keepdim=True) + 1e-8)

                # ── PPO surrogate loss ────────────────────────────────────────
                ratio  = torch.exp(new_log_probs - mb_old_lp)
                surr1  = ratio * mb_advantages
                surr2  = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── value loss ────────────────────────────────────────────────
                mb_vh, mb_vs, mb_vint = self.global_value_net(mb_global)
                mb_vext = torch.cat([mb_vh, mb_vh, mb_vs, mb_vs], dim=1)
                mb_vint_exp = mb_vint.expand(-1, A)

                value_loss_ext = F.mse_loss(mb_vext, mb_ret_ext)
                value_loss_int = F.mse_loss(mb_vint_exp, mb_ret_int)
                value_loss = value_loss_ext + value_loss_int

                # ── COMA critic loss (trained in both modes) ──────────────────
                q_target = mb_ret_ext[:, :2].mean(dim=-1, keepdim=True).detach()
                q_actual = self.coma_critic(mb_global, mb_joint_oh)
                critic_loss = F.mse_loss(q_actual, q_target)

                # ── total loss & update ───────────────────────────────────────
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.value_coef * critic_loss
                    - entropy_coef * entropy
                )

                # Learning rate decay
                frac = 1.0 - (self.update_step / self.total_steps)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.initial_lr * max(frac, 0.1)

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.coma_critic.parameters())
                    + list(self.global_value_net.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()
