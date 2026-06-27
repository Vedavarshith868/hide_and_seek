#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script to load a trained model and render an episode to an MP4 video.
Can be run on Lightning Studio or Google Colab.
"""
import argparse
import imageio
import torch
import numpy as np
from tqdm import tqdm

from constants import EPISODE_LEN, N_AGENTS, MAX_ENTITIES, FEAT_DIM, LIDAR_DIM, HIDDEN_DIM
from env.hide_and_seek import HideAndSeek2D
from models.sensory import SensoryPipeline
from models.actor import MAPPOActor

def evaluate(checkpoint_path: str, output_video: str = "eval_output.mp4",
             deterministic: bool = True):
    dev = torch.device('cpu')  # Evaluation is perfectly fine on CPU
    print(f"Loading checkpoint: {checkpoint_path}")
    
    sensory_pipeline = SensoryPipeline().to(dev)
    actor = MAPPOActor(sensory_pipeline).to(dev)
    
    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=dev)
    if isinstance(ckpt, dict) and 'actor' in ckpt:
        actor.load_state_dict(ckpt['actor'])
    else:
        actor.load_state_dict(ckpt)
    actor.eval()

    # We use a single direct environment so we can call render_frame() easily
    env = HideAndSeek2D()
    obs_raw, _ = env.reset()
    
    # Initialize LSTM hidden states
    hidden = (
        torch.zeros(1, N_AGENTS, HIDDEN_DIM, device=dev).contiguous(),
        torch.zeros(1, N_AGENTS, HIDDEN_DIM, device=dev).contiguous(),
    )
    
    frames = []
    print("Simulating episode...")
    for _ in tqdm(range(EPISODE_LEN)):
        # Capture current visual frame
        frames.append(env.render_frame())
        
        # Stack individual agent observations into a single batch
        fe = np.stack([o['entities'] for o in obs_raw])
        fm = np.stack([o['mask'] for o in obs_raw])
        fl = np.stack([o['lidar'] for o in obs_raw])
        
        fe_t = torch.tensor(fe, dtype=torch.float32, device=dev).view(N_AGENTS, MAX_ENTITIES, FEAT_DIM)
        fm_t = torch.tensor(fm, dtype=torch.float32, device=dev).view(N_AGENTS, MAX_ENTITIES)
        fl_t = torch.tensor(fl, dtype=torch.float32, device=dev).view(N_AGENTS, LIDAR_DIM)
        
        # Forward pass to get actions (argmax by default so videos show the
        # mode behavior, not exploration noise)
        with torch.no_grad():
            af, lpf, nh = actor.get_action(fe_t, fm_t, fl_t, hidden,
                                           deterministic=deterministic)
            
        h, c = nh
        hidden = (h.contiguous(), c.contiguous())
        
        # Convert actions back to numpy and step env
        actions = af.cpu().numpy()
        obs_raw, _, _, done, _ = env.step(actions)
        
        if done: break
    # Capture the final frame of the episode
    frames.append(env.render_frame())

    print(f"Encoding video to {output_video} (this may take a moment)...")
    imageio.mimsave(output_video, frames, fps=25)
    print(f"Success! Video saved as {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a MAPPO actor and render a video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to actor_X.pth")
    parser.add_argument("--output", type=str, default="eval_output.mp4", help="Output MP4 filename")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.output, deterministic=not args.stochastic)
