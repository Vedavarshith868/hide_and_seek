# -*- coding: utf-8 -*-
"""
MAPPO actor: sensory pipeline -> LSTM -> action heads.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical

from constants import LIDAR_DIM
from models.sensory import SensoryPipeline


class MAPPOActor(nn.Module):
    def __init__(
        self,
        sensory_pipeline: SensoryPipeline,
        lidar_dim: int = LIDAR_DIM,
        context_dim: int = 128,
        lstm_hidden_dim: int = 256,
    ):
        super().__init__()
        self.sensory_pipeline = sensory_pipeline
        self.lstm_input_dim   = context_dim + lidar_dim
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.move_head = nn.Linear(lstm_hidden_dim, 9)
        self.rot_head  = nn.Linear(lstm_hidden_dim, 3)
        self.grab_head = nn.Linear(lstm_hidden_dim, 2)
        self.lock_head = nn.Linear(lstm_hidden_dim, 2)

        # Small orthogonal init on policy heads keeps the initial policy
        # near-uniform without needing a large entropy bonus.
        for head in (self.move_head, self.rot_head, self.grab_head, self.lock_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.constant_(head.bias, 0.0)

    def forward(self, entities, mask, lidar, hidden_state):
        context = self.sensory_pipeline(entities, mask, lidar)
        fused_features = torch.cat([context, lidar], dim=-1).unsqueeze(1)
        lstm_out, new_hidden_state = self.lstm(fused_features, hidden_state)
        lstm_out = lstm_out.squeeze(1)
        logits = {
            'move': self.move_head(lstm_out),
            'rot':  self.rot_head(lstm_out),
            'grab': self.grab_head(lstm_out),
            'lock': self.lock_head(lstm_out),
        }
        return logits, new_hidden_state

    def get_action(self, entities, mask, lidar, hidden_state, deterministic: bool = False):
        logits, new_hidden_state = self.forward(entities, mask, lidar, hidden_state)
        dist_m = Categorical(logits=logits['move'])
        dist_r = Categorical(logits=logits['rot'])
        dist_g = Categorical(logits=logits['grab'])
        dist_l = Categorical(logits=logits['lock'])

        if deterministic:
            a_m = logits['move'].argmax(dim=-1)
            a_r = logits['rot'].argmax(dim=-1)
            a_g = logits['grab'].argmax(dim=-1)
            a_l = logits['lock'].argmax(dim=-1)
        else:
            a_m, a_r = dist_m.sample(), dist_r.sample()
            a_g, a_l = dist_g.sample(), dist_l.sample()

        joint_log_prob = (
            dist_m.log_prob(a_m)
            + dist_r.log_prob(a_r)
            + dist_g.log_prob(a_g)
            + dist_l.log_prob(a_l)
        )
        joint_actions = torch.stack([a_m, a_r, a_g, a_l], dim=-1)
        return joint_actions, joint_log_prob, new_hidden_state
