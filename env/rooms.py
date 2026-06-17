# -*- coding: utf-8 -*-
"""
Procedural room layout generator (OpenAI hide-and-seek style).

Generates 1-2 corner rooms per episode with doorways. Doorway width (1.2m,
matching FINAL_PLAN_2D §1.5) is chosen so that:
  - agents (0.5m diameter) pass through easily,
  - it is resolvable by the 30-ray lidar from ~5.7m (so seekers can perceive
    and breach it, not just stumble through),
  - a single elongated (1.5m) box, or two standard (0.5m) boxes, seals it.
"""
from __future__ import annotations
import math

DOOR_WIDTH = 1.2


def _generate_openai_rooms(rng, arena_size=10.0):
    walls, rooms = [], []
    n_rooms = rng.choice([1, 2])
    corners = [(0, 0), (arena_size, 0), (arena_size, arena_size), (0, arena_size)]
    rng.shuffle(corners)

    def cut_door(p1, p2, doorway_width):
        x1, y1 = p1; x2, y2 = p2
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= doorway_width + 1.0:
            return [(p1, p2)]
        t = rng.uniform(0.5, length - doorway_width - 0.5)
        dx, dy = (x2 - x1) / length, (y2 - y1) / length
        return [
            (p1, (x1 + dx * t, y1 + dy * t)),
            ((x1 + dx * (t + doorway_width), y1 + dy * (t + doorway_width)), p2),
        ]

    for i in range(n_rooms):
        cx, cy = corners[i]
        w, h = rng.uniform(3.5, 4.5), rng.uniform(3.5, 4.5)
        ix = w if cx == 0 else arena_size - w
        iy = h if cy == 0 else arena_size - h
        v_wall, h_wall = ((ix, cy), (ix, iy)), ((cx, iy), (ix, iy))
        door_config = rng.choice(['v', 'h', 'both'])

        walls.extend(cut_door(*v_wall, DOOR_WIDTH) if door_config in ['v', 'both'] else [v_wall])
        walls.extend(cut_door(*h_wall, DOOR_WIDTH) if door_config in ['h', 'both'] else [h_wall])
        rooms.append((min(cx, ix), min(cy, iy), max(cx, ix), max(cy, iy)))

    return walls, rooms
