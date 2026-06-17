# -*- coding: utf-8 -*-
"""
Rollout buffer for storing a full episode of trajectory data.
"""
import torch

from constants import MAX_ENTITIES, FEAT_DIM, LIDAR_DIM, GLOBAL_DIM, HIDDEN_DIM


class RolloutBuffer:
    def __init__(self, num_envs: int, episode_len: int, n_agents: int, device):
        self.num_envs    = num_envs
        self.episode_len = episode_len
        self.n_agents    = n_agents
        self.device      = device
        T, E, A, d = episode_len, num_envs, n_agents, device

        self.entities         = torch.zeros(T, E, A, MAX_ENTITIES, FEAT_DIM, device=d)
        self.masks            = torch.zeros(T, E, A, MAX_ENTITIES,           device=d)
        self.lidar            = torch.zeros(T, E, A, LIDAR_DIM,              device=d)
        self.global_states    = torch.zeros(T, E, GLOBAL_DIM,                device=d)
        self.actions          = torch.zeros(T, E, A, 4, dtype=torch.long,    device=d)
        self.joint_actions_oh = torch.zeros(T, E, 64,                        device=d)
        self.log_probs        = torch.zeros(T, E, A,                         device=d)
        self.rewards_ext      = torch.zeros(T, E, A,                         device=d)
        self.rewards_int      = torch.zeros(T, E, A,                         device=d)
        self.dones            = torch.zeros(T, E,                            device=d)
        self.hiddens_h        = torch.zeros(T, E, A, HIDDEN_DIM,             device=d)
        self.hiddens_c        = torch.zeros(T, E, A, HIDDEN_DIM,             device=d)

    def insert(self, t: int, entities, masks, lidar, gstate, actions,
               joint_oh, log_probs, rewards_ext, rewards_int, dones, hidden):
        h, c = hidden
        h = h.squeeze(0).view(self.num_envs, self.n_agents, HIDDEN_DIM)
        c = c.squeeze(0).view(self.num_envs, self.n_agents, HIDDEN_DIM)
        self.entities[t]         = entities
        self.masks[t]            = masks
        self.lidar[t]            = lidar
        self.global_states[t]    = gstate
        self.actions[t]          = actions
        self.joint_actions_oh[t] = joint_oh
        self.log_probs[t]        = log_probs
        self.rewards_ext[t]      = rewards_ext
        self.rewards_int[t]      = rewards_int
        self.dones[t]            = dones
        self.hiddens_h[t]        = h
        self.hiddens_c[t]        = c

    def clear(self):
        pass  # tensors are overwritten in-place each episode
