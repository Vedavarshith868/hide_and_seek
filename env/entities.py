# -*- coding: utf-8 -*-
"""
Dataclass definitions for simulation entities and a geometric helper.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


def _point_segment_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    t = 0.0 if ab2 == 0 else max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


@dataclass
class Agent:
    idx: int
    team: str
    body_name: str
    grabbed_box: Optional[int] = None
    grab_eq_id: Optional[int] = None
    # MuJoCo address caches (filled after model compile)
    body_id: int = -1
    qvel_adr: int = -1      # address of the x slide dof (y is +1, z hinge is +2)
    qpos_z_adr: int = -1    # qpos address of the z hinge joint


@dataclass
class Box:
    idx: int
    body_name: str
    half_size: tuple
    elongated: bool
    mass: float
    locked_by: Optional[str] = None   # team name that locked it, or None
    grabbed_by: Optional[int] = None
    # MuJoCo address caches (filled after model compile)
    body_id: int = -1
    geom_id: int = -1
    qvel_adr: int = -1

    @property
    def locked(self) -> bool:
        return self.locked_by is not None
