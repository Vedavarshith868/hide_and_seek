# -*- coding: utf-8 -*-
"""
God-view critic networks:
  - actions_to_one_hot helper
  - COMACritic
  - GlobalValueNetwork
  - COMAAdvantageEngine
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


def actions_to_one_hot(actions: torch.Tensor) -> torch.Tensor:
    """Convert integer action tensor (B, A, 4) to flat one-hot (B, A*16)."""
    batch_size = actions.shape[0]
    move_oh = F.one_hot(actions[..., 0], num_classes=9)
    rot_oh  = F.one_hot(actions[..., 1], num_classes=3)
    grab_oh = F.one_hot(actions[..., 2], num_classes=2)
    lock_oh = F.one_hot(actions[..., 3], num_classes=2)
    combined_one_hot = torch.cat([move_oh, rot_oh, grab_oh, lock_oh], dim=-1).float()
    return combined_one_hot.view(batch_size, -1)


class COMACritic(nn.Module):
    def __init__(self, global_state_dim: int = 109, n_agents: int = 4, action_dim: int = 16):
        super().__init__()
        self.input_dim = global_state_dim + (n_agents * action_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, global_state: torch.Tensor, joint_actions_one_hot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([global_state, joint_actions_one_hot], dim=-1)
        return self.net(x)


class GlobalValueNetwork(nn.Module):
    def __init__(self, global_state_dim: int = 109):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.v_ext_hider  = nn.Linear(256, 1)
        self.v_ext_seeker = nn.Linear(256, 1)
        self.v_int_head   = nn.Linear(256, 1)

    def forward(self, global_state: torch.Tensor):
        features  = self.base(global_state)
        v_hider   = self.v_ext_hider(features)
        v_seeker  = self.v_ext_seeker(features)
        v_int     = self.v_int_head(features)
        return v_hider, v_seeker, v_int


class COMAAdvantageEngine(nn.Module):
    def __init__(self, critic: COMACritic, n_agents: int = 4, action_dim: int = 16):
        super().__init__()
        self.critic    = critic
        self.n_agents  = n_agents
        self.action_dim = action_dim
        action_space = [list(range(9)), list(range(3)), list(range(2)), list(range(2))]
        all_combos   = list(itertools.product(*action_space))
        self.register_buffer('all_combos', torch.tensor(all_combos, dtype=torch.long))
        all_combos_oh = torch.cat([
            F.one_hot(self.all_combos[:, 0], num_classes=9),
            F.one_hot(self.all_combos[:, 1], num_classes=3),
            F.one_hot(self.all_combos[:, 2], num_classes=2),
            F.one_hot(self.all_combos[:, 3], num_classes=2),
        ], dim=-1).float()
        self.register_buffer('all_combos_oh', all_combos_oh)

    def forward(self, global_state: torch.Tensor, joint_actions_one_hot: torch.Tensor, logits: dict) -> torch.Tensor:
        batch_size = global_state.shape[0]
        device     = global_state.device
        q_actual   = self.critic(global_state, joint_actions_one_hot)
        advantages = torch.zeros(batch_size, self.n_agents, device=device)

        for idx in range(self.n_agents):
            cf_joint_actions = joint_actions_one_hot.unsqueeze(1).expand(-1, 108, -1).clone()
            start_col = idx * self.action_dim
            end_col   = start_col + self.action_dim
            cf_joint_actions[:, :, start_col:end_col] = self.all_combos_oh.unsqueeze(0)

            flat_cf_actions  = cf_joint_actions.reshape(batch_size * 108, -1)
            flat_global_state = global_state.unsqueeze(1).expand(-1, 108, -1).reshape(batch_size * 108, -1)
            cf_q_values = self.critic(flat_global_state, flat_cf_actions).view(batch_size, 108)

            p_move = F.softmax(logits['move'][:, idx, :], dim=-1)
            p_rot  = F.softmax(logits['rot'][:,  idx, :], dim=-1)
            p_grab = F.softmax(logits['grab'][:, idx, :], dim=-1)
            p_lock = F.softmax(logits['lock'][:, idx, :], dim=-1)

            p_move_cf = p_move[:, self.all_combos[:, 0]]
            p_rot_cf  = p_rot[:,  self.all_combos[:, 1]]
            p_grab_cf = p_grab[:, self.all_combos[:, 2]]
            p_lock_cf = p_lock[:, self.all_combos[:, 3]]

            cf_probs = p_move_cf * p_rot_cf * p_grab_cf * p_lock_cf
            baseline = (cf_probs * cf_q_values).sum(dim=1)
            advantages[:, idx] = q_actual.squeeze(-1) - baseline

        return advantages
