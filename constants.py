# -*- coding: utf-8 -*-
"""
Shared constants for the Hide-and-Seek 2D environment and training pipeline.
"""
from __future__ import annotations
import math
import numpy as np

# ---------------------------------------------------------------------------
# Arena / physics
# ---------------------------------------------------------------------------
ARENA_SIZE        = 10.0
WALL_THICKNESS    = 0.1
DT                = 0.04
SUBSTEPS          = 10
SUBSTEP_DT        = DT / SUBSTEPS

# ---------------------------------------------------------------------------
# Episode  (OpenAI proportions: 40% preparation phase, 60% competition)
# ---------------------------------------------------------------------------
EPISODE_LEN       = 240
PREP_STEPS        = 96
REWARD_SCALE      = 1.0 / EPISODE_LEN

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
N_HIDERS          = 2
N_SEEKERS         = 2
N_AGENTS          = N_HIDERS + N_SEEKERS

AGENT_RADIUS      = 0.25
AGENT_MASS        = 1.0
AGENT_MOMENT      = 0.5 * AGENT_MASS * (AGENT_RADIUS ** 2)
# Velocity-blend movement controller: each step  v <- 0.85*(0.5*v + 0.5*target).
# Steady-state speed = 0.739 * MOVE_TARGET_SPEED  (~3.7 m/s, close to OpenAI's ~3.3).
MOVE_TARGET_SPEED = 5.0
SPEED_CLAMP       = 6.0     # hard per-step clamp on commanded agent speed
ROTATE_STEP       = 0.15    # rad per env step (kinematic rotation)
MAX_SPEED         = 15.0    # physics-validity reset threshold
PENETRATION_LIMIT = 0.08

# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------
BOX_COUNT_MIN     = 4
BOX_COUNT_MAX     = 6
MIN_ELONGATED     = 2
STD_BOX_SIZE      = (0.5, 0.5)
ELONG_BOX_SIZE    = (1.5, 0.5)
BOX_MASS          = 10.0
BOX_FRICTION      = 0.3
GRAB_GAP          = 0.15    # max surface-to-surface gap for grab / lock reach

# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
MAX_ENTITIES      = 12
ENTITY_FEAT_DIM   = 18
GLOBAL_STATE_DIM  = 109
LIDAR_RAYS        = 72
LIDAR_MAX_RANGE   = 6.0
FOV_HALF_ANGLE    = math.radians(67.5)
# The per-agent observation vector is LiDAR + 2 scalar time features:
#   [30 ray distances, prep_time_remaining, episode_time_remaining]
OBS_EXTRA_DIM     = 2

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
MOVE_ACTIONS   = 9
ROTATE_ACTIONS = 3
GRAB_ACTIONS   = 2
LOCK_ACTIONS   = 2

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Spawn distribution (FINAL_PLAN_2D §1.3 / OpenAI quadrant environment):
# 'uniform'  — everything spawns uniformly at random (hard mode, OpenAI
#              randomized env; where COMA's per-agent credit is testable)
# 'quadrant' — hiders + standard boxes spawn inside the first room,
#              seekers outside it (OpenAI quadrant env; phase-1 fallback if
#              uniform joint-hiding stalls).
SPAWN_MODE = 'uniform'

# 'gae'  : per-team GAE advantages (OpenAI's algorithm = baseline)
# 'coma' : counterfactual COMA advantages from the centralized Q critic.
#   COMA-done-right used a short-horizon Q-target (COMA_Q_GAMMA); SNR rose
#   0.019->0.146 but still <1, and it collapsed to a degenerate fixed-action
#   policy (entropy 0.05). COMA's learned-Q approximation of the per-agent
#   counterfactual is too noisy for this sparse team-coupled reward.
# 'dr'   : DIFFERENCE-REWARD advantage. Computes the per-agent counterfactual
#   EXACTLY from the line-of-sight structure (no learned Q -> no noise):
#       D_i = G - G_{-i}   (global reward minus reward with agent i removed)
#   A_i = standardize(GAE_team) + DR_KAPPA * standardize(discounted-sum D_i).
#   Fixes COMA's noise failure while giving the per-agent credit GAE lacks.
ADV_MODE      = 'dr'
COMA_Q_GAMMA  = 0.95   # short-horizon discount for the COMA Q-target
DR_KAPPA      = 1.0    # weight of per-agent difference-reward term (=team term)
DR_LAMBDA     = 0.95   # decay for the discounted difference-return

# ---------------------------------------------------------------------------
# Neural-network / training aliases  (used by models & training modules)
# ---------------------------------------------------------------------------
FEAT_DIM   = ENTITY_FEAT_DIM            # 18
LIDAR_DIM  = LIDAR_RAYS + OBS_EXTRA_DIM # 32 (rays + time features)
GLOBAL_DIM = GLOBAL_STATE_DIM           # 109
HIDDEN_DIM = 256

# ---------------------------------------------------------------------------
# Pre-computed LiDAR ray directions (UNIT vectors — mj_ray/mj_multiRay return
# distance in units of the direction vector's length, so unit vectors make
# the returned distance metric).
# ---------------------------------------------------------------------------
_LIDAR_ANGLES = np.linspace(0, 2 * math.pi, LIDAR_RAYS, endpoint=False)
LIDAR_VECS = np.column_stack([
    np.cos(_LIDAR_ANGLES),
    np.sin(_LIDAR_ANGLES),
    np.zeros(LIDAR_RAYS),
]).astype(np.float64)
LIDAR_VECS_FLAT = np.ascontiguousarray(LIDAR_VECS.reshape(-1))

# ---------------------------------------------------------------------------
# Pre-computed movement directions
# ---------------------------------------------------------------------------
_MOVE_DIRS = []
for _i in range(8):
    _a = _i * (2 * math.pi / 8)
    _MOVE_DIRS.append((math.cos(_a), math.sin(_a)))
_MOVE_DIRS.append((0.0, 0.0))
MOVE_DIRS = np.array(_MOVE_DIRS, dtype=np.float32)
