# -*- coding: utf-8 -*-
"""
Vectorised environment wrappers.

- ForkVecEnv: fork-based multiprocess vectorisation (Linux — used in training)
- SyncVecEnv: in-process sequential fallback (Windows / debugging / tests)
- make_vec_env(): picks the right one for the platform
"""
from __future__ import annotations
import numpy as np
import torch
import torch.multiprocessing as mp

from constants import N_AGENTS, MAX_ENTITIES, FEAT_DIM, LIDAR_DIM, GLOBAL_DIM
from env.hide_and_seek import HideAndSeek2D


# ---------------------------------------------------------------------------
# Worker function (runs in forked child process)
# ---------------------------------------------------------------------------
def _worker(rank, obs_bufs, gstate_buf, reward_buf, dr_buf, done_buf,
            cmd_pipe, ready_pipe, seed):
    """
    Runs in a forked process. HideAndSeek2D is already in memory via fork.
    Writes obs directly into shared memory using zero-copy numpy buffers.
    """
    env = HideAndSeek2D(seed=seed)

    E = len(done_buf)
    np_ent   = np.frombuffer(obs_bufs['entities'], dtype=np.float32).reshape(E, N_AGENTS, MAX_ENTITIES, FEAT_DIM)
    np_mask  = np.frombuffer(obs_bufs['masks'],    dtype=np.float32).reshape(E, N_AGENTS, MAX_ENTITIES)
    np_lidar = np.frombuffer(obs_bufs['lidar'],    dtype=np.float32).reshape(E, N_AGENTS, LIDAR_DIM)
    np_gs    = np.frombuffer(gstate_buf,           dtype=np.float32).reshape(E, GLOBAL_DIM)
    np_rew   = np.frombuffer(reward_buf,           dtype=np.float32).reshape(E, N_AGENTS)
    np_dr    = np.frombuffer(dr_buf,               dtype=np.float32).reshape(E, N_AGENTS)
    np_done  = np.frombuffer(done_buf,             dtype=np.float32)

    def write(obs, gstate):
        for a in range(N_AGENTS):
            np_ent[rank, a]   = obs[a]['entities']
            np_mask[rank, a]  = obs[a]['mask']
            np_lidar[rank, a] = obs[a]['lidar']
        np_gs[rank] = gstate

    try:
        obs, gstate = env.reset()
        write(obs, gstate)
        ready_pipe.send('ready')

        while True:
            cmd = cmd_pipe.recv()
            if cmd[0] == 'step':
                actions = cmd[1]
                obs, gstate, reward, done, info = env.step(actions)
                # On a physics crash env.step() has already reset internally
                # and returned the fresh episode's obs — don't reset twice.
                if done and not info.get('physics_reset', False):
                    obs, gstate = env.reset()

                write(obs, gstate)
                np_rew[rank]  = reward
                np_dr[rank]   = info.get('dr', np.zeros(N_AGENTS, dtype=np.float32))
                np_done[rank] = float(done)
                ready_pipe.send('done')

            elif cmd[0] == 'reset':
                obs, gstate = env.reset()
                write(obs, gstate)
                ready_pipe.send('ready')

            elif cmd[0] == 'close':
                break

    except Exception:
        import traceback
        ready_pipe.send(f'error:{traceback.format_exc()}')


# ---------------------------------------------------------------------------
# Vectorised environment (fork-based)
# ---------------------------------------------------------------------------
class ForkVecEnv:
    """
    Fork-based vectorised env. Works on Linux because fork inherits
    all in-memory state without pickling.
    """

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device   = device
        E, A          = num_envs, N_AGENTS

        self._ent_buf  = mp.Array('f', E * A * MAX_ENTITIES * FEAT_DIM, lock=False)
        self._mask_buf = mp.Array('f', E * A * MAX_ENTITIES,            lock=False)
        self._lid_buf  = mp.Array('f', E * A * LIDAR_DIM,               lock=False)
        self._gs_buf   = mp.Array('f', E * GLOBAL_DIM,                  lock=False)
        self._rew_buf  = mp.Array('f', E * A,                           lock=False)
        self._dr_buf   = mp.Array('f', E * A,                           lock=False)
        self._done_buf = mp.Array('f', E,                               lock=False)

        obs_bufs = {
            'entities': self._ent_buf,
            'masks':    self._mask_buf,
            'lidar':    self._lid_buf,
        }

        ctx = mp.get_context('fork')
        self._cmd_pipes   = []
        self._ready_pipes = []
        self._procs       = []

        for rank in range(num_envs):
            cmd_p,   cmd_c   = ctx.Pipe()
            ready_p, ready_c = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(rank, obs_bufs, self._gs_buf,
                      self._rew_buf, self._dr_buf, self._done_buf,
                      cmd_c, ready_c, rank),
                daemon=True,
            )
            p.start()
            cmd_c.close()
            ready_c.close()
            self._cmd_pipes.append(cmd_p)
            self._ready_pipes.append(ready_p)
            self._procs.append(p)

        self._wait_all()

    def _wait_all(self):
        for rank, pipe in enumerate(self._ready_pipes):
            msg = pipe.recv()
            if isinstance(msg, str) and msg.startswith('error:'):
                raise RuntimeError(f"Worker {rank} crashed:\n{msg[6:]}")

    def _read(self):
        E, A = self.num_envs, N_AGENTS
        ent  = torch.frombuffer(self._ent_buf,  dtype=torch.float32).clone() \
                    .view(E, A, MAX_ENTITIES, FEAT_DIM).to(self.device)
        mask = torch.frombuffer(self._mask_buf, dtype=torch.float32).clone() \
                    .view(E, A, MAX_ENTITIES).to(self.device)
        lid  = torch.frombuffer(self._lid_buf,  dtype=torch.float32).clone() \
                    .view(E, A, LIDAR_DIM).to(self.device)
        gs   = torch.frombuffer(self._gs_buf,   dtype=torch.float32).clone() \
                    .view(E, GLOBAL_DIM).to(self.device)
        return {'entities': ent, 'masks': mask, 'lidar': lid}, gs

    def reset(self):
        for p in self._cmd_pipes:
            p.send(('reset',))
        self._wait_all()
        return self._read()

    def step(self, actions_np: np.ndarray):
        for rank, p in enumerate(self._cmd_pipes):
            p.send(('step', actions_np[rank]))
        self._wait_all()
        obs, gs = self._read()
        rew  = torch.frombuffer(self._rew_buf,  dtype=torch.float32).clone() \
                    .view(self.num_envs, N_AGENTS).to(self.device)
        dr   = torch.frombuffer(self._dr_buf,   dtype=torch.float32).clone() \
                    .view(self.num_envs, N_AGENTS).to(self.device)
        done = torch.frombuffer(self._done_buf, dtype=torch.float32).clone() \
                    .to(self.device)
        return obs, gs, rew, dr, done

    def close(self):
        for p in self._cmd_pipes:
            try:
                p.send(('close',))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()


# ---------------------------------------------------------------------------
# Synchronous in-process fallback (Windows / tests)
# ---------------------------------------------------------------------------
class SyncVecEnv:
    """Sequential vectorised env with the exact same interface as ForkVecEnv."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device   = device
        self.envs     = [HideAndSeek2D(seed=i) for i in range(num_envs)]
        E, A = num_envs, N_AGENTS
        self._ent  = np.zeros((E, A, MAX_ENTITIES, FEAT_DIM), dtype=np.float32)
        self._mask = np.zeros((E, A, MAX_ENTITIES),           dtype=np.float32)
        self._lid  = np.zeros((E, A, LIDAR_DIM),              dtype=np.float32)
        self._gs   = np.zeros((E, GLOBAL_DIM),                dtype=np.float32)
        self._rew  = np.zeros((E, A),                         dtype=np.float32)
        self._dr   = np.zeros((E, A),                         dtype=np.float32)
        self._done = np.zeros(E,                              dtype=np.float32)

    def _write(self, rank, obs, gstate):
        for a in range(N_AGENTS):
            self._ent[rank, a]  = obs[a]['entities']
            self._mask[rank, a] = obs[a]['mask']
            self._lid[rank, a]  = obs[a]['lidar']
        self._gs[rank] = gstate

    def _read(self):
        obs = {
            'entities': torch.from_numpy(self._ent.copy()).to(self.device),
            'masks':    torch.from_numpy(self._mask.copy()).to(self.device),
            'lidar':    torch.from_numpy(self._lid.copy()).to(self.device),
        }
        gs = torch.from_numpy(self._gs.copy()).to(self.device)
        return obs, gs

    def reset(self):
        for rank, env in enumerate(self.envs):
            o, g = env.reset()
            self._write(rank, o, g)
        return self._read()

    def step(self, actions_np: np.ndarray):
        for rank, env in enumerate(self.envs):
            o, g, r, d, info = env.step(actions_np[rank])
            self._dr[rank] = info.get('dr', np.zeros(N_AGENTS, dtype=np.float32))
            if d and not info.get('physics_reset', False):
                o, g = env.reset()
            self._write(rank, o, g)
            self._rew[rank]  = r
            self._done[rank] = float(d)
        obs, gs = self._read()
        rew  = torch.from_numpy(self._rew.copy()).to(self.device)
        dr   = torch.from_numpy(self._dr.copy()).to(self.device)
        done = torch.from_numpy(self._done.copy()).to(self.device)
        return obs, gs, rew, dr, done

    def close(self):
        pass


def make_vec_env(num_envs: int, device: torch.device):
    """ForkVecEnv where fork is available (Linux), SyncVecEnv otherwise."""
    try:
        mp.get_context('fork')
        return ForkVecEnv(num_envs, device)
    except ValueError:
        print("[vec_env] fork unavailable — using in-process SyncVecEnv")
        return SyncVecEnv(num_envs, device)
