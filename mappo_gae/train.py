#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — entrypoint for the Hide-and-Seek 2D MAPPO training run.

Usage:
    python train.py [--iterations 3000] [--envs 32] [--device cuda]
"""
import math
import time
import argparse
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from constants import (
    EPISODE_LEN, PREP_STEPS, N_AGENTS, MAX_ENTITIES, FEAT_DIM, LIDAR_DIM,
    HIDDEN_DIM, ADV_MODE,
)
from env.vec_env import make_vec_env
from models.sensory import SensoryPipeline
from models.actor import MAPPOActor
from models.critics import COMACritic, GlobalValueNetwork, COMAAdvantageEngine, actions_to_one_hot
from models.rnd import RNDModule
from training.buffer import RolloutBuffer
from training.updater import MAPPOUpdater
from evaluate import evaluate

# Global-state indices of per-box lock / grab flags (for behavior metrics)
_BOX_LOCK_IDX = [28 + s * 10 + 6 for s in range(6)]
_BOX_GRAB_IDX = [28 + s * 10 + 9 for s in range(6)]


def train(num_iterations: int = 3000, num_envs: int = 32, device: str = 'cuda', resume_path: str = ""):
    torch.set_num_threads(4)

    dev = torch.device(device)
    print(f"Starting | device={device} | envs={num_envs} | adv_mode={ADV_MODE}")

    vec_env = make_vec_env(num_envs=num_envs, device=dev)

    sensory_pipeline = SensoryPipeline().to(dev)
    actor            = MAPPOActor(sensory_pipeline).to(dev)
    coma_critic      = COMACritic().to(dev)
    val_net          = GlobalValueNetwork().to(dev)
    coma_engine      = COMAAdvantageEngine(coma_critic).to(dev)
    rnd_module       = RNDModule(spatial_dim=128, interaction_dim=128).to(dev)
    updater          = MAPPOUpdater(actor, coma_critic, val_net, coma_engine)
    buffer           = RolloutBuffer(num_envs, EPISODE_LEN, N_AGENTS, dev)

    # RND curiosity, gated to the losing team. Enabled at iter ~8200:
    # probes showed single-hider room-hiding stuck at ~30% of steps — the
    # losing team needs more "both hidden" samples for GAE to reinforce.
    w_max, w_floor, w_decay = 5e-4, 1e-4, 50_000_000
    win_hist: deque = deque(maxlen=1000)
    timing:   deque = deque(maxlen=20)

    rnd_spatial_hist = deque(maxlen=200)
    use_interaction = False

    lock_frac_hist = deque(maxlen=100)
    grab_frac_hist = deque(maxlen=100)

    start_update = 0
    if resume_path:
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=dev)
        actor.load_state_dict(ckpt['actor'])
        coma_critic.load_state_dict(ckpt['coma_critic'])
        val_net.load_state_dict(ckpt['val_net'])
        if 'rnd_module' in ckpt:
            try:
                rnd_module.load_state_dict(ckpt['rnd_module'])
            except RuntimeError:
                print("  [WARN] RND architecture changed, reinitializing RND weights")
        updater.optimizer.load_state_dict(ckpt['optimizer'])
        win_hist.extend(ckpt.get('win_hist', []))
        start_update = ckpt.get('update', 0)
        updater.update_step = start_update
        print(f"Successfully resumed at update {start_update}")

    try:
        for update in tqdm(range(start_update, num_iterations), desc="Updates"):
            t0 = time.perf_counter()

            steps  = update * EPISODE_LEN * num_envs
            w_base = w_max * math.exp(-steps / w_decay) + w_floor if w_max > 0 else 0.0
            wr_h   = float(np.mean(win_hist)) if win_hist else 0.5
            wr_s   = 1.0 - wr_h
            # Continuous per-team scaling: curiosity stays high for
            # the losing team, never drops below 30% of w_base.
            hider_need  = max(0.0, 1.0 - 2.0 * wr_h)
            seeker_need = max(0.0, 1.0 - 2.0 * wr_s)
            w_h = max(w_floor, w_base * (0.3 + 0.7 * hider_need)) if w_max > 0 else 0.0
            w_s = max(w_floor, w_base * (0.3 + 0.7 * seeker_need)) if w_max > 0 else 0.0
            w_t = torch.tensor([w_h, w_h, w_s, w_s], dtype=torch.float32, device=dev)

            obs, gstate = vec_env.reset()
            hidden = (
                torch.zeros(1, num_envs * N_AGENTS, HIDDEN_DIM, device=dev).contiguous(),
                torch.zeros(1, num_envs * N_AGENTS, HIDDEN_DIM, device=dev).contiguous(),
            )
            rnd_spatial_ctx = []
            rnd_interact_ctx = []

            # Per-env episode return tracker for win-rate measurement
            episode_ret_h = torch.zeros(num_envs, device=dev)

            for step in range(EPISODE_LEN):
                fe = obs['entities'].view(num_envs * N_AGENTS, MAX_ENTITIES, FEAT_DIM)
                fm = obs['masks']   .view(num_envs * N_AGENTS, MAX_ENTITIES)
                fl = obs['lidar']   .view(num_envs * N_AGENTS, LIDAR_DIM)

                # Hidden state BEFORE acting — this is what the policy used,
                # and what the PPO update must replay from.
                pre_hidden = hidden

                with torch.no_grad():
                    rnd_ctx = sensory_pipeline(fe, fm, fl)  # (num_envs*N_AGENTS, 128)
                    af, lpf, nh = actor.get_action(fe, fm, fl, hidden)

                    if w_max > 0:
                        rint = rnd_module.intrinsic_reward(rnd_ctx, rnd_ctx, use_interaction).view(num_envs, N_AGENTS)
                        rnd_spatial_ctx.append(rnd_ctx.detach())
                        if use_interaction:
                            rnd_interact_ctx.append(rnd_ctx.detach())
                    else:
                        rint = torch.zeros(num_envs, N_AGENTS, device=dev)

                actions  = af.view(num_envs, N_AGENTS, 4)
                log_prob = lpf.view(num_envs, N_AGENTS)
                joint_oh = actions_to_one_hot(actions)

                next_obs, next_gstate, rext, dones = vec_env.step(actions.cpu().numpy())

                # Behavior metrics from the global state (actual box state,
                # not action-sampling rates)
                lock_frac_hist.append(float(gstate[:, _BOX_LOCK_IDX].abs().gt(0).float().mean()))
                grab_frac_hist.append(float(gstate[:, _BOX_GRAB_IDX].gt(0).float().mean()))

                episode_ret_h += rext[:, 0]   # hider-team return (rewards are 0 in prep)
                for env_i in range(num_envs):
                    if dones[env_i]:
                        win_hist.append(float(episode_ret_h[env_i] > 0))
                        episode_ret_h[env_i] = 0.0

                buffer.insert(
                    step,
                    obs['entities'], obs['masks'], obs['lidar'],
                    gstate, actions, joint_oh, log_prob,
                    rext, rint, dones, pre_hidden,
                )

                h, c = nh
                hidden = (h.contiguous(), c.contiguous())
                # Reset LSTM state for envs whose episode ended (physics crash
                # resets happen mid-rollout; normal dones at the last step)
                if bool(dones.any()):
                    keep = (1.0 - dones).repeat_interleave(N_AGENTS).view(1, num_envs * N_AGENTS, 1)
                    hidden = ((hidden[0] * keep).contiguous(), (hidden[1] * keep).contiguous())

                obs, gstate = next_obs, next_gstate

            if w_max > 0 and rnd_spatial_ctx:
                if use_interaction and rnd_interact_ctx:
                    loss_s, loss_i = rnd_module.update(
                        torch.cat(rnd_spatial_ctx, dim=0),
                        torch.cat(rnd_interact_ctx, dim=0),
                        use_interaction
                    )
                else:
                    loss_s, loss_i = rnd_module.update(
                        torch.cat(rnd_spatial_ctx, dim=0),
                        torch.zeros(1, device=dev),
                        use_interaction
                    )

                rnd_spatial_hist.append(loss_s)
                if not use_interaction and len(rnd_spatial_hist) >= 100 and np.mean(rnd_spatial_hist) < 0.1:
                    use_interaction = True
                    print("\n[RND] Spatial novelty exhausted. Activating Interaction RND channel!")
            else:
                loss_s, loss_i = 0.0, 0.0

            updater.update(buffer, gstate, w_t, seeker_wr=wr_s, num_epochs=2)
            buffer.clear()

            timing.append(time.perf_counter() - t0)
            if (update + 1) % 20 == 0:
                print(
                    f"\n[{update+1:4d}] {np.mean(timing):.2f}s/it | "
                    f"wr_h={wr_h:.3f} | rnd_s={loss_s:.4f} rnd_i={loss_i:.4f} | "
                    f"box_grab={np.mean(grab_frac_hist):.3f} box_lock={np.mean(lock_frac_hist):.3f} | "
                    f"b_h={w_h:.4f} b_s={w_s:.4f} | "
                    f"ent={updater.last_entropy_coef:.4f}"
                )
                ckpt_path = f"checkpoint_{update+1}.pth"
                torch.save({
                    'update': update + 1,
                    'actor': actor.state_dict(),
                    'coma_critic': coma_critic.state_dict(),
                    'val_net': val_net.state_dict(),
                    'rnd_module': rnd_module.state_dict(),
                    'optimizer': updater.optimizer.state_dict(),
                    'win_hist': list(win_hist),
                }, ckpt_path)

            # Auto video generation
            if (update + 1) % 20 == 0:
                out_vid = f"eval_update_{update+1}.mp4"
                print(f"\n[EVAL] Generating video: {out_vid}")
                evaluate(ckpt_path, out_vid)

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        vec_env.close()

    return actor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hide-and-Seek 2D MAPPO agent")
    parser.add_argument("--iterations", type=int, default=3000, help="Number of training iterations")
    parser.add_argument("--envs",       type=int, default=32,   help="Number of parallel environments")
    parser.add_argument("--device",     type=str, default="cuda", help="Torch device (cuda / cpu)")
    parser.add_argument("--resume",     type=str, default="", help="Path to checkpoint_X.pth to resume from")
    args = parser.parse_args()

    trained_actor = train(
        num_iterations=args.iterations,
        num_envs=args.envs,
        device=args.device,
        resume_path=args.resume,
    )
