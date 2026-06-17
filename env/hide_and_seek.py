# -*- coding: utf-8 -*-
"""
MuJoCo-based Hide-and-Seek 2D environment.

Key conventions:
  - All raycasts go through self._raycast(), which normalizes the direction
    vector before calling mj_ray. MuJoCo's mj_ray returns distance in units
    of ||vec||, NOT meters — passing unit vectors makes the result metric.
  - Boxes are immovable (huge joint frictionloss) unless grabbed. Locking a
    box prevents grabbing; only the team that locked a box can unlock it.
  - Reward is pure zero-sum line-of-sight: if any hider is visible to any
    seeker, seekers get +1/EPISODE_LEN each and hiders -1/EPISODE_LEN
    (and vice versa). No shaped rewards.
"""
from __future__ import annotations
import math
import random
from typing import Optional

import numpy as np
import mujoco

from constants import (
    ARENA_SIZE, WALL_THICKNESS, SUBSTEP_DT, SUBSTEPS,
    EPISODE_LEN, PREP_STEPS, REWARD_SCALE,
    N_HIDERS, N_SEEKERS, N_AGENTS,
    AGENT_RADIUS, AGENT_MASS, MOVE_TARGET_SPEED, SPEED_CLAMP, ROTATE_STEP,
    MAX_SPEED, PENETRATION_LIMIT,
    BOX_COUNT_MIN, BOX_COUNT_MAX, MIN_ELONGATED, STD_BOX_SIZE,
    ELONG_BOX_SIZE, BOX_MASS, BOX_FRICTION, GRAB_GAP,
    MAX_ENTITIES, ENTITY_FEAT_DIM, GLOBAL_STATE_DIM,
    LIDAR_RAYS, LIDAR_MAX_RANGE, FOV_HALF_ANGLE, OBS_EXTRA_DIM,
    LIDAR_VECS_FLAT, MOVE_DIRS, SPAWN_MODE,
)
from env.entities import Agent, Box, _point_segment_dist
from env.rooms import _generate_openai_rooms

# Joint frictionloss values controlling box mobility
_BOX_IMMOVABLE_FRICTION = 1e6
_BOX_GRABBED_FRICTION   = 1.0
_BOX_IMMOVABLE_DAMPING  = 20.0
_BOX_GRABBED_DAMPING    = 2.0

# mj_multiRay gained a `normal` output argument in newer MuJoCo versions.
# Detect the calling convention once.
_MULTIRAY_HAS_NORMAL: Optional[bool] = None


class HideAndSeek2D:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.step_count = 0
        self.episode_count = 0
        self.model = None
        self.data = None

        self.agents: list[Agent] = []
        self.boxes: list[Box] = []
        self.wall_segments: list[tuple] = []

        self._penetration_flag = False
        self._settle_steps = 10

        # Pre-allocate raycast arrays to avoid per-call allocation
        self._ray_pnt = np.zeros(3, dtype=np.float64)
        self._ray_vec = np.zeros(3, dtype=np.float64)
        self._ray_grp = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._ray_gid = np.zeros(1, dtype=np.int32)
        self._lidar_gid  = np.full(LIDAR_RAYS, -1, dtype=np.int32)
        self._lidar_dist = np.zeros(LIDAR_RAYS, dtype=np.float64)

    # ----------------------------------------------------------------- reset
    def reset(self):
        self.step_count = 0
        self.episode_count += 1
        self.agents.clear()
        self.boxes.clear()
        self.wall_segments.clear()
        self._penetration_flag = False

        self._compile_mujoco_scene()

        # Random initial rotations so agents/boxes don't always face right
        for a in self.agents:
            self.data.qpos[a.qpos_z_adr] = self.rng.uniform(0, 2 * math.pi)
        for b in self.boxes:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{b.body_name}_z")
            self.data.qpos[self.model.jnt_qposadr[jnt_id]] = self.rng.uniform(0, 2 * math.pi)

        # Let the solver resolve any residual spawn overlap, then kill all
        # velocities so settle impulses can't trigger the physics filter.
        for _ in range(self._settle_steps):
            for _ in range(SUBSTEPS):
                mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._penetration_flag = False

        return self._build_all_observations(), self._build_global_state()

    # ----------------------------------------------------------------- scene
    def _compile_mujoco_scene(self):
        boundary = [
            ((0, 0), (ARENA_SIZE, 0)),
            ((ARENA_SIZE, 0), (ARENA_SIZE, ARENA_SIZE)),
            ((ARENA_SIZE, ARENA_SIZE), (0, ARENA_SIZE)),
            ((0, ARENA_SIZE), (0, 0)),
        ]
        internal, rooms = _generate_openai_rooms(self.rng, ARENA_SIZE)
        self.wall_segments = boundary + internal
        self.rooms = rooms
        # Quadrant spawn (FINAL_PLAN_2D §1.3): hiders + standard boxes start
        # inside the first room, seekers outside it.
        quadrant = SPAWN_MODE == 'quadrant' and len(rooms) > 0
        room0 = rooms[0] if quadrant else None

        xml = f"""
        <mujoco model="hide_and_seek_2d">
            <option timestep="{SUBSTEP_DT}" gravity="0 0 0" iterations="50" tolerance="1e-4"/>
            <compiler angle="radian" coordinate="local"/>
            <default>
                <geom friction="{BOX_FRICTION} 0.005 0.0001" margin="0.001"/>
                <joint damping="0.1"/>
            </default>
            <worldbody>
                <geom type="plane" size="{ARENA_SIZE} {ARENA_SIZE} 0.1" pos="{ARENA_SIZE/2} {ARENA_SIZE/2} -0.1" rgba="0.9 0.9 0.9 1" group="0"/>
        """

        for idx, (p1, p2) in enumerate(self.wall_segments):
            cx, cy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            xml += f'<geom type="box" size="{length/2} {WALL_THICKNESS/2} 0.5" pos="{cx} {cy} 0.5" euler="0 0 {angle}" group="0"/>\n'

        # Boxes — avoid walls and previously placed objects
        placed_objects = []   # (x, y, radius)
        num_boxes = self.rng.randint(BOX_COUNT_MIN, BOX_COUNT_MAX)
        for i in range(num_boxes):
            w, h = ELONG_BOX_SIZE if i < MIN_ELONGATED else STD_BOX_SIZE
            rad = max(w, h) / 2
            # Quadrant fort material in the room: all standard boxes plus ONE
            # elongated box (index 0) — a single 1.5m box seals the 1.2m door,
            # making the seal a one-action discovery. The other elongated box
            # stays outside (seeker side).
            in_room_box = quadrant and (i >= MIN_ELONGATED or i == 0)
            if in_room_box:
                pos = self._sample_in_rect(room0, margin=rad + 0.15, avoid=placed_objects)
            else:
                pos = self._sample_free_position(margin=rad + 0.2, avoid=placed_objects)
            placed_objects.append((pos[0], pos[1], rad))
            xml += f"""
            <body name="box_{i}" pos="{pos[0]} {pos[1]} 0.25">
                <joint type="slide" axis="1 0 0" name="box_{i}_x" damping="{_BOX_IMMOVABLE_DAMPING}" frictionloss="{_BOX_IMMOVABLE_FRICTION}"/>
                <joint type="slide" axis="0 1 0" name="box_{i}_y" damping="{_BOX_IMMOVABLE_DAMPING}" frictionloss="{_BOX_IMMOVABLE_FRICTION}"/>
                <joint type="hinge" axis="0 0 1" name="box_{i}_z" damping="{_BOX_IMMOVABLE_DAMPING}" frictionloss="{_BOX_IMMOVABLE_FRICTION}"/>
                <geom name="box_{i}_geom" type="box" size="{w/2} {h/2} 0.25" mass="{BOX_MASS}" group="0" rgba="0.7 0.8 0.3 1"/>
            </body>
            """
            self.boxes.append(Box(i, f"box_{i}", (w / 2, h / 2), max(w, h) > 1.0, BOX_MASS))

        placed_agents = []
        for i, team in enumerate(['hider'] * N_HIDERS + ['seeker'] * N_SEEKERS):
            pos = None
            min_sep = 1.5
            for attempt in range(200):
                if attempt == 100:
                    min_sep = 0.8   # relax separation in tight rooms
                if quadrant and team == 'hider':
                    cand = self._sample_in_rect(
                        room0, margin=AGENT_RADIUS + 0.1, avoid=placed_objects)
                else:
                    cand = self._sample_free_position(
                        margin=AGENT_RADIUS + 0.1, avoid=placed_objects)
                    if quadrant and team == 'seeker' and self._in_rect(cand, room0, pad=0.2):
                        continue   # seekers spawn outside the room
                if any(math.hypot(cand[0] - q[0], cand[1] - q[1]) < min_sep for q in placed_agents):
                    continue
                pos = cand
                break
            if pos is None:
                pos = self._sample_free_position(margin=AGENT_RADIUS + 0.1, avoid=placed_objects)
            placed_agents.append(pos)
            xml += f"""
            <body name="agent_{i}" pos="{pos[0]} {pos[1]} {AGENT_RADIUS}">
                <joint type="slide" axis="1 0 0" name="agent_{i}_x"/><joint type="slide" axis="0 1 0" name="agent_{i}_y"/><joint type="hinge" axis="0 0 1" name="agent_{i}_z"/>
                <geom name="agent_{i}_geom" type="cylinder" size="{AGENT_RADIUS} 0.2" mass="{AGENT_MASS}" group="1" rgba="0.8 0.2 0.2 1"/>
            </body>
            """
            self.agents.append(Agent(i, team, f"agent_{i}"))

        xml += "</worldbody>\n<equality>\n"
        for a in range(N_AGENTS):
            for b in range(num_boxes):
                xml += f'<connect name="grab_a{a}_b{b}" body1="agent_{a}" body2="box_{b}" anchor="0 0 0" active="false"/>\n'
        xml += "</equality>\n</mujoco>"

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._build_id_caches()

    def _build_id_caches(self):
        """Cache all name->id lookups once per compile (mj_name2id is slow)."""
        m = self.model
        self._bid = {}
        self._qvel_adr = {}
        for ent in [*self.agents, *self.boxes]:
            name = ent.body_name
            ent.body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            jx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_x")
            ent.qvel_adr = m.jnt_dofadr[jx]
            self._bid[name] = ent.body_id
            self._qvel_adr[name] = ent.qvel_adr
        for a in self.agents:
            jz = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{a.body_name}_z")
            a.qpos_z_adr = m.jnt_qposadr[jz]
        for b in self.boxes:
            b.geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{b.body_name}_geom")
        self._eq_id = {}
        for a in range(N_AGENTS):
            for b in range(len(self.boxes)):
                self._eq_id[(a, b)] = mujoco.mj_name2id(
                    m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grab_a{a}_b{b}")

    @staticmethod
    def _in_rect(p, rect, pad: float = 0.0) -> bool:
        x0, y0, x1, y1 = rect
        return (x0 - pad) <= p[0] <= (x1 + pad) and (y0 - pad) <= p[1] <= (y1 + pad)

    def _sample_in_rect(self, rect, margin: float, avoid=(), tries: int = 200):
        """Uniform sample inside a room rectangle, clear of walls and objects."""
        x0, y0, x1, y1 = rect
        lo_x, hi_x = x0 + margin, x1 - margin
        lo_y, hi_y = y0 + margin, y1 - margin
        if lo_x >= hi_x or lo_y >= hi_y:
            return ((x0 + x1) / 2, (y0 + y1) / 2)
        for _ in range(tries):
            x = self.rng.uniform(lo_x, hi_x)
            y = self.rng.uniform(lo_y, hi_y)
            ok = True
            for (a, b) in self.wall_segments:
                if _point_segment_dist((x, y), a, b) < margin + WALL_THICKNESS / 2:
                    ok = False
                    break
            if ok:
                for (ax, ay, arad) in avoid:
                    if math.hypot(x - ax, y - ay) < margin + arad + 0.1:
                        ok = False
                        break
            if ok:
                return (x, y)
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def _sample_free_position(self, margin: float, avoid=(), tries: int = 200):
        for _ in range(tries):
            x = self.rng.uniform(margin, ARENA_SIZE - margin)
            y = self.rng.uniform(margin, ARENA_SIZE - margin)
            ok = True
            for (a, b) in self.wall_segments:
                if _point_segment_dist((x, y), a, b) < margin + WALL_THICKNESS / 2:
                    ok = False
                    break
            if ok:
                for (ax, ay, arad) in avoid:
                    if math.hypot(x - ax, y - ay) < margin + arad + 0.1:
                        ok = False
                        break
            if ok:
                return (x, y)
        return (ARENA_SIZE / 2, ARENA_SIZE / 2)

    # ----------------------------------------------------------------- state accessors
    def _pos(self, body_name):
        p = self.data.xpos[self._bid[body_name]]
        return np.array([p[0], p[1]])

    def _ang(self, body_name):
        quat = self.data.xquat[self._bid[body_name]]
        return math.atan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
        )

    def _vel(self, body_name):
        idx = self._qvel_adr[body_name]
        return np.array([self.data.qvel[idx], self.data.qvel[idx + 1]])

    def _raycast(self, origin, vector, ignore_agents=True, bodyexclude=-1):
        """Cast a ray; returns (metric_distance, geom_id).

        The direction vector is normalized internally because mj_ray returns
        distance in units of ||vec||. Returns (-1.0, -1) on no hit.
        """
        vx = float(vector[0]); vy = float(vector[1])
        vz = float(vector[2]) if len(vector) == 3 else 0.0
        norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if norm < 1e-9:
            return -1.0, -1

        self._ray_pnt[0], self._ray_pnt[1] = origin[0], origin[1]
        self._ray_pnt[2] = origin[2] if len(origin) == 3 else 0.25
        self._ray_vec[0] = vx / norm
        self._ray_vec[1] = vy / norm
        self._ray_vec[2] = vz / norm

        dist = mujoco.mj_ray(
            self.model, self.data, self._ray_pnt, self._ray_vec,
            self._ray_grp if ignore_agents else None, 1, bodyexclude, self._ray_gid,
        )
        if dist < 0:
            return -1.0, -1
        return float(dist), int(self._ray_gid[0])

    # ----------------------------------------------------------------- step
    def step(self, joint_actions: np.ndarray):
        assert joint_actions.shape == (N_AGENTS, 4)
        self.step_count += 1
        in_prep = self.step_count <= PREP_STEPS

        for i, agent in enumerate(self.agents):
            if in_prep and agent.team == 'seeker':
                continue   # frozen: velocities are zeroed inside the substep loop
            move_a, rot_a, grab_a, lock_a = joint_actions[i]
            self._apply_movement(agent, int(move_a), int(rot_a))
            self._apply_grab(agent, int(grab_a))
            self._apply_lock(agent, int(lock_a))

        for _ in range(SUBSTEPS):
            for a in self.agents:
                idx = a.qvel_adr
                # Kinematic rotation: zero z spin so friction can't roll agents
                self.data.qvel[idx + 2] = 0.0
                if in_prep and a.team == 'seeker':
                    self.data.qvel[idx:idx + 2] = 0.0

            # Non-grabbed boxes are exactly immovable. Frictionloss alone
            # leaves iterative-solver creep (~5 cm/s under sustained push),
            # enough to slowly shove a locked doorway seal out of place.
            for b in self.boxes:
                if b.grabbed_by is None:
                    self.data.qvel[b.qvel_adr:b.qvel_adr + 3] = 0.0

            mujoco.mj_step(self.model, self.data)

            ncon = self.data.ncon
            if ncon:
                dists = self.data.contact.dist[:ncon]
                deep = dists < -PENETRATION_LIMIT
                if deep.any():
                    g1 = self.data.contact.geom1[:ncon][deep]
                    g2 = self.data.contact.geom2[:ncon][deep]
                    if (self.model.geom_group[g1] == 1).any() or \
                       (self.model.geom_group[g2] == 1).any():
                        self._penetration_flag = True

        reset_needed = self._penetration_flag
        max_speed = 0.0
        for a in self.agents:
            speed = float(np.linalg.norm(self._vel(a.body_name)))
            if speed > max_speed:
                max_speed = speed
            if speed > MAX_SPEED:
                reset_needed = True

        if reset_needed:
            crash_info = {'physics_reset': True, 'prep': in_prep, 'crash_speed': max_speed}
            obs, gstate = self.reset()
            return obs, gstate, np.zeros(N_AGENTS, dtype=np.float32), True, crash_info

        rewards, diff_rewards = self._compute_rewards(in_prep)
        done = self.step_count >= EPISODE_LEN
        return (
            self._build_all_observations(),
            self._build_global_state(),
            rewards,
            done,
            {'physics_reset': False, 'prep': in_prep, 'dr': diff_rewards},
        )

    # ----------------------------------------------------------------- actions
    def _apply_movement(self, agent, move_a, rot_a):
        d = MOVE_DIRS[move_a]
        idx = agent.qvel_adr

        if d[0] != 0.0 or d[1] != 0.0:
            tvx = float(d[0]) * MOVE_TARGET_SPEED
            tvy = float(d[1]) * MOVE_TARGET_SPEED
            self.data.qvel[idx]     = self.data.qvel[idx]     * 0.5 + tvx * 0.5
            self.data.qvel[idx + 1] = self.data.qvel[idx + 1] * 0.5 + tvy * 0.5

        if rot_a == 0:
            self.data.qpos[agent.qpos_z_adr] += ROTATE_STEP
        elif rot_a == 2:
            self.data.qpos[agent.qpos_z_adr] -= ROTATE_STEP

        self.data.qvel[idx:idx + 2] *= 0.85
        self.data.qvel[idx + 2] = 0.0
        speed = math.hypot(self.data.qvel[idx], self.data.qvel[idx + 1])
        if speed > SPEED_CLAMP:
            self.data.qvel[idx]     *= SPEED_CLAMP / speed
            self.data.qvel[idx + 1] *= SPEED_CLAMP / speed

    def _object_reachable(self, a_pos, box) -> bool:
        """True if nothing (wall / other box) blocks the line from the agent
        to the box. The target box itself does not count as an obstruction."""
        b_pos = self._pos(box.body_name)
        to_b = b_pos - a_pos
        dist, geom_id = self._raycast([*a_pos, 0.25], [*to_b, 0], ignore_agents=True)
        if dist < 0 or geom_id == box.geom_id:
            return True
        return dist >= float(np.linalg.norm(to_b)) - 0.01

    def _nearest_box_in_reach(self, agent, include_locked: bool):
        best, best_gap = None, GRAB_GAP
        a_pos = self._pos(agent.body_name)
        for b in self.boxes:
            if not include_locked and (b.grabbed_by is not None or b.locked):
                continue
            b_pos = self._pos(b.body_name)
            gap = float(np.linalg.norm(b_pos - a_pos)) - AGENT_RADIUS - max(b.half_size)
            if gap < best_gap and self._object_reachable(a_pos, b):
                best_gap, best = gap, b
        return best

    def _apply_grab(self, agent, grab_a):
        if grab_a == 0:
            return

        # --- DROP (toggle off) ---
        if agent.grabbed_box is not None:
            if agent.grab_eq_id is not None:
                self.data.eq_active[agent.grab_eq_id] = 0
                agent.grab_eq_id = None
            box = self.boxes[agent.grabbed_box]
            self._set_box_mobility(box, grabbed=False)
            box.grabbed_by = None
            agent.grabbed_box = None
            return

        # --- GRAB ---
        best = self._nearest_box_in_reach(agent, include_locked=False)
        if best is None:
            return

        a_pos = self._pos(agent.body_name)
        eq_id = self._eq_id[(agent.idx, best.idx)]
        a_pos_3d = np.array([*a_pos, 0.0])
        b_pos_3d = np.array([*self._pos(best.body_name), 0.0])
        dir_vec = (b_pos_3d - a_pos_3d) / np.linalg.norm(b_pos_3d - a_pos_3d)
        world_anchor = a_pos_3d + dir_vec * (AGENT_RADIUS + 0.02)

        def to_local(w_pt, pos, ang):
            dx, dy = w_pt[0] - pos[0], w_pt[1] - pos[1]
            c, s = math.cos(-ang), math.sin(-ang)
            return [dx * c - dy * s, dx * s + dy * c, 0.0]

        local_a = to_local(world_anchor, a_pos, self._ang(agent.body_name))
        local_b = to_local(world_anchor, self._pos(best.body_name), self._ang(best.body_name))

        self.model.eq_data[eq_id][:6] = [*local_a, *local_b]
        self.data.eq_active[eq_id] = 1
        self._set_box_mobility(best, grabbed=True)

        agent.grab_eq_id = eq_id
        agent.grabbed_box = best.idx
        best.grabbed_by = agent.idx

    def _set_box_mobility(self, box, grabbed: bool):
        idx = box.qvel_adr
        if grabbed:
            self.model.dof_frictionloss[idx:idx + 3] = _BOX_GRABBED_FRICTION
            self.model.dof_damping[idx:idx + 3] = _BOX_GRABBED_DAMPING
        else:
            self.model.dof_frictionloss[idx:idx + 3] = _BOX_IMMOVABLE_FRICTION
            self.model.dof_damping[idx:idx + 3] = _BOX_IMMOVABLE_DAMPING

    def _apply_lock(self, agent, lock_a):
        """Toggle lock on the nearest reachable box.

        OpenAI semantics: both teams can lock, but a locked box can only be
        unlocked by the team that locked it.
        """
        if lock_a == 0:
            return
        target = self._nearest_box_in_reach(agent, include_locked=True)
        if target is None:
            return

        if target.locked_by is None:
            # Lock: rip from any current grabber, freeze in place
            for a in self.agents:
                if a.grabbed_box == target.idx:
                    self.data.eq_active[a.grab_eq_id] = 0
                    a.grabbed_box, a.grab_eq_id = None, None
            target.grabbed_by = None
            self._set_box_mobility(target, grabbed=False)
            self.data.qvel[target.qvel_adr:target.qvel_adr + 3] = 0.0
            target.locked_by = agent.team
        elif target.locked_by == agent.team:
            target.locked_by = None   # box stays immovable until grabbed
        # else: locked by the other team — cannot unlock

    # ----------------------------------------------------------------- LOS / vision
    def _agent_sees(self, seeker: Agent, hider: Agent) -> bool:
        sp, hp = self._pos(seeker.body_name), self._pos(hider.body_name)
        sang = self._ang(seeker.body_name)
        eye = sp + np.array([math.cos(sang), math.sin(sang)]) * 0.2
        to_h = hp - eye

        dist_to_h = float(np.linalg.norm(to_h))
        if dist_to_h < 1e-6:
            return True
        diff = (math.atan2(to_h[1], to_h[0]) - sang + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > FOV_HALF_ANGLE:
            return False

        dist, _ = self._raycast([*eye, 0.25], [*to_h, 0], ignore_agents=True)
        return dist < 0 or dist >= dist_to_h - 0.05

    def _lidar(self, agent: Agent) -> np.ndarray:
        global _MULTIRAY_HAS_NORMAL
        pos = self._pos(agent.body_name)
        self._ray_pnt[0], self._ray_pnt[1], self._ray_pnt[2] = pos[0], pos[1], 0.25
        self._lidar_gid[:] = -1

        if _MULTIRAY_HAS_NORMAL is None:
            try:
                mujoco.mj_multiRay(
                    self.model, self.data, self._ray_pnt, LIDAR_VECS_FLAT,
                    None, 1, agent.body_id, self._lidar_gid, self._lidar_dist,
                    None, LIDAR_RAYS, LIDAR_MAX_RANGE)
                _MULTIRAY_HAS_NORMAL = True
            except TypeError:
                _MULTIRAY_HAS_NORMAL = False

        if _MULTIRAY_HAS_NORMAL:
            mujoco.mj_multiRay(
                self.model, self.data, self._ray_pnt, LIDAR_VECS_FLAT,
                None, 1, agent.body_id, self._lidar_gid, self._lidar_dist,
                None, LIDAR_RAYS, LIDAR_MAX_RANGE)
        else:
            mujoco.mj_multiRay(
                self.model, self.data, self._ray_pnt, LIDAR_VECS_FLAT,
                None, 1, agent.body_id, self._lidar_gid, self._lidar_dist,
                LIDAR_RAYS, LIDAR_MAX_RANGE)

        out = np.where(
            (self._lidar_dist < 0) | (self._lidar_gid < 0), 1.0,
            np.clip(self._lidar_dist / LIDAR_MAX_RANGE, 0.0, 1.0),
        ).astype(np.float32)
        return out

    # ----------------------------------------------------------------- observations
    def _visible_set_from(self, viewer: Agent):
        vis_opp, vis_box = set(), set()
        vpos, vang = self._pos(viewer.body_name), self._ang(viewer.body_name)
        eye = vpos + np.array([math.cos(vang), math.sin(vang)]) * 0.2

        for j, other in enumerate(self.agents):
            if other.team != viewer.team and self._agent_sees(viewer, other):
                vis_opp.add(j)

        for bi, b in enumerate(self.boxes):
            bpos = self._pos(b.body_name)
            to_b = bpos - eye
            dist_to_b = float(np.linalg.norm(to_b))
            if dist_to_b < 1e-6:
                vis_box.add(bi)
                continue
            diff = (math.atan2(to_b[1], to_b[0]) - vang + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > FOV_HALF_ANGLE:
                continue

            dist, geom_id = self._raycast([*eye, 0.25], [*to_b, 0], ignore_agents=True)
            if dist < 0 or dist >= dist_to_b - 0.05 or geom_id == b.geom_id:
                vis_box.add(bi)

        return vis_opp, vis_box

    def _build_entity_row(self, viewer: Agent, ent_type: str, other, self_idx: int, viewer_idx: int) -> np.ndarray:
        row = np.zeros(ENTITY_FEAT_DIM, dtype=np.float32)
        if ent_type == 'self':
            row[0] = 1.0
            # Team identity flags (2D repurposes the unused ramp/z-pos dims)
            row[4] = 1.0 if viewer.team == 'hider' else 0.0
            row[8] = 1.0 if viewer.team == 'seeker' else 0.0
        elif ent_type == 'team': row[1] = 1.0
        elif ent_type == 'opp':  row[2] = 1.0
        elif ent_type == 'box':  row[3] = 1.0

        vpos, vvel = self._pos(viewer.body_name), self._vel(viewer.body_name)

        if ent_type in ('self', 'team', 'opp'):
            rel = self._pos(other.body_name) - vpos
            vel = self._vel(other.body_name) - vvel
            size = AGENT_RADIUS * 2
            ang = self._ang(other.body_name)
            # Facing direction encoded in the (otherwise unused for agents)
            # lock/grab slots: row[12]=sin(theta), row[13]=cos(theta)
            grabbed = math.sin(ang)
            gb_me   = math.cos(ang)
            gb_team = gb_opp = 0.0
            team_lock = 0.0
        else:
            rel = self._pos(other.body_name) - vpos
            vel = self._vel(other.body_name) - vvel if not other.locked else np.array([0.0, 0.0])
            size = max(other.half_size) * 2
            grabbed = 1.0 if other.locked else 0.0
            # row[11] (z-vel slot, unused in 2D): locked by MY team flag —
            # needed because only the locking team can unlock.
            team_lock = 1.0 if other.locked_by == viewer.team else 0.0
            if other.grabbed_by is None:
                gb_me = gb_team = gb_opp = 0.0
            else:
                holder = self.agents[other.grabbed_by]
                gb_me   = 1.0 if other.grabbed_by == viewer_idx else 0.0
                gb_team = 1.0 if (other.grabbed_by != viewer_idx and holder.team == viewer.team) else 0.0
                gb_opp  = 1.0 if holder.team != viewer.team else 0.0

        row[6],  row[7]  = rel[0] / 10.0, rel[1] / 10.0
        row[9],  row[10] = vel[0] / 15.0, vel[1] / 15.0
        if ent_type == 'box':
            row[11] = team_lock
        row[12], row[13], row[14], row[15] = grabbed, gb_me, gb_team, gb_opp
        row[16] = min(size / 2.0, 1.0)
        row[17] = min(math.hypot(rel[0], rel[1]) / 14.142, 1.0)
        return row

    def _build_observation_for(self, viewer_idx: int):
        viewer = self.agents[viewer_idx]
        vis_opp, vis_box = self._visible_set_from(viewer)
        entities = np.zeros((MAX_ENTITIES, ENTITY_FEAT_DIM), dtype=np.float32)
        mask     = np.zeros(MAX_ENTITIES, dtype=np.float32)
        slot = 0

        entities[slot] = self._build_entity_row(viewer, 'self', viewer, viewer_idx, viewer_idx)
        mask[slot] = 1.0; slot += 1

        for j, other in enumerate(self.agents):
            if j == viewer_idx or other.team != viewer.team:
                continue
            if slot >= MAX_ENTITIES:
                break
            entities[slot] = self._build_entity_row(viewer, 'team', other, j, viewer_idx)
            mask[slot] = 1.0; slot += 1

        for j in vis_opp:
            if slot >= MAX_ENTITIES:
                break
            entities[slot] = self._build_entity_row(viewer, 'opp', self.agents[j], j, viewer_idx)
            mask[slot] = 1.0; slot += 1

        for bi in vis_box:
            if slot >= MAX_ENTITIES:
                break
            entities[slot] = self._build_entity_row(viewer, 'box', self.boxes[bi], bi, viewer_idx)
            mask[slot] = 1.0; slot += 1

        for s in range(slot, MAX_ENTITIES):
            entities[s, 5] = 1.0

        # Observation vector = 30 lidar rays + 2 time features.
        # Time awareness is essential: hiders must distinguish the prep phase
        # (build forts) from competition (stay hidden).
        prep_remaining = max(0.0, (PREP_STEPS - self.step_count) / PREP_STEPS)
        time_remaining = 1.0 - self.step_count / EPISODE_LEN
        obs_vec = np.concatenate([
            self._lidar(viewer),
            np.array([prep_remaining, time_remaining], dtype=np.float32),
        ])

        return {'entities': entities, 'mask': mask, 'lidar': obs_vec}

    def _build_all_observations(self):
        return [self._build_observation_for(i) for i in range(N_AGENTS)]

    def _build_global_state(self):
        v = np.zeros(GLOBAL_STATE_DIM, dtype=np.float32)
        off = 0
        for a in self.agents:
            pos, vel, ang = self._pos(a.body_name), self._vel(a.body_name), self._ang(a.body_name)
            v[off + 0], v[off + 1] = pos[0] / 10.0, pos[1] / 10.0
            v[off + 3], v[off + 4] = vel[0] / 15.0, vel[1] / 15.0
            v[off + 6] = math.atan2(math.sin(ang), math.cos(ang)) / math.pi
            off += 7

        for slot in range(6):
            if slot < len(self.boxes):
                b = self.boxes[slot]
                pos, vel = self._pos(b.body_name), self._vel(b.body_name)
                v[off + 0], v[off + 1] = pos[0] / 10.0, pos[1] / 10.0
                v[off + 3], v[off + 4] = vel[0] / 15.0, vel[1] / 15.0
                # +1 locked by hiders, -1 locked by seekers, 0 unlocked
                if b.locked_by == 'hider':
                    v[off + 6] = 1.0
                elif b.locked_by == 'seeker':
                    v[off + 6] = -1.0
                if b.grabbed_by is not None:
                    holder = self.agents[b.grabbed_by]
                    v[off + 7] = 1.0 if holder.team == 'hider' else 0.0
                    v[off + 8] = 1.0 if holder.team == 'seeker' else 0.0
                    v[off + 9] = 1.0
            off += 10

        off += 20  # ramp slots unused in 2D
        v[off] = 1.0 - (self.step_count / EPISODE_LEN)
        return v

    def _compute_rewards(self, in_prep: bool):
        """Zero-sum line-of-sight team reward, plus the EXACT per-agent
        difference reward D_i = G - G_{-i} (used by ADV_MODE='dr').

        Returns (team_reward[N_AGENTS], diff_reward[N_AGENTS]).
        D_i is computed exactly from the seeker x hider visibility matrix:
          - seeker i: +2*scale iff it is the UNIQUE seer (it alone catches a
            hider that no other seeker sees), else 0.
          - hider j:  -2*scale iff it is the UNIQUE visible hider (it alone
            gets the team caught), else 0.
        """
        r = np.zeros(N_AGENTS, dtype=np.float32)
        d = np.zeros(N_AGENTS, dtype=np.float32)
        if in_prep:
            return r, d

        seekers = [i for i, a in enumerate(self.agents) if a.team == 'seeker']
        hiders  = [i for i, a in enumerate(self.agents) if a.team == 'hider']

        # full seeker x hider visibility matrix
        vis = {}
        for si in seekers:
            for hi in hiders:
                vis[(si, hi)] = self._agent_sees(self.agents[si], self.agents[hi])

        any_visible = any(vis.values())

        for i, a in enumerate(self.agents):
            if any_visible:
                r[i] = +REWARD_SCALE if a.team == 'seeker' else -REWARD_SCALE
            else:
                r[i] = +REWARD_SCALE if a.team == 'hider' else -REWARD_SCALE

        # ── exact difference rewards ───────────────────────────────────────
        for si in seekers:
            # team seen by OTHER seekers (seeker si removed)?
            vis_without = any(vis[(sj, hi)] for sj in seekers if sj != si for hi in hiders)
            # G_seek = +scale if any_visible else -scale; D = G - G_{-i}
            d[si] = 2.0 * REWARD_SCALE * (float(any_visible) - float(vis_without))
        for hi in hiders:
            # team seen ignoring hider hi?
            vis_without = any(vis[(si, hj)] for si in seekers for hj in hiders if hj != hi)
            # G_hide = +scale if NOT visible else -scale; D = G - G_{-j}
            #        = scale*(1-2*any_visible) - scale*(1-2*vis_without)
            d[hi] = 2.0 * REWARD_SCALE * (float(vis_without) - float(any_visible))
        return r, d

    # ----------------------------------------------------------------- render
    def render_frame(self, px_per_m: int = 60) -> np.ndarray:
        from PIL import Image, ImageDraw
        W, H = int(ARENA_SIZE * px_per_m), int(ARENA_SIZE * px_per_m)
        W, H = (W + 15) // 16 * 16, (H + 15) // 16 * 16
        img = Image.new('RGB', (W, H), (245, 245, 240))
        draw = ImageDraw.Draw(img)

        def to_px(p):
            return (p[0] * px_per_m, H - p[1] * px_per_m)

        for (a, b) in self.wall_segments:
            draw.line(
                [to_px(a), to_px(b)],
                fill=(40, 40, 40),
                width=max(2, int(WALL_THICKNESS * px_per_m)),
            )

        for bx in self.boxes:
            pos, ang = self._pos(bx.body_name), self._ang(bx.body_name)
            c, s = math.cos(ang), math.sin(ang)
            pts = []
            for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                lx, ly = dx * bx.half_size[0], dy * bx.half_size[1]
                pts.append(to_px((pos[0] + lx * c - ly * s, pos[1] + lx * s + ly * c)))
            if bx.locked_by == 'hider':
                fill, outline = (90, 120, 200), (40, 60, 140)
            elif bx.locked_by == 'seeker':
                fill, outline = (200, 110, 90), (140, 50, 40)
            else:
                fill, outline = (180, 200, 90), (90, 110, 40)
            draw.polygon(pts, fill=fill, outline=outline)

        for ag in self.agents:
            pos, ang = self._pos(ag.body_name), self._ang(ag.body_name)
            cx, cy = to_px(pos)
            r_px = AGENT_RADIUS * px_per_m
            draw.ellipse(
                [cx - r_px, cy - r_px, cx + r_px, cy + r_px],
                fill=(60, 110, 220) if ag.team == 'hider' else (220, 70, 70),
                outline=(20, 20, 20),
            )
            fx = pos[0] + math.cos(ang) * AGENT_RADIUS * 1.6
            fy = pos[1] + math.sin(ang) * AGENT_RADIUS * 1.6
            draw.line([(cx, cy), to_px((fx, fy))], fill=(255, 255, 255), width=2)

        phase = "PREP (seekers frozen)" if self.step_count < PREP_STEPS else "COMPETITION"
        draw.text((8, 8), f"step {self.step_count}/{EPISODE_LEN}  |  {phase}", fill=(20, 20, 20))
        return np.array(img)
