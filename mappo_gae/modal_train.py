#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modal_train.py — Run Hide-and-Seek training on Modal Labs.

Setup (one-time):
    pip install modal
    modal setup          # authenticate with your Modal account

Upload checkpoint:
    modal volume create hide-and-seek-data
    modal volume put hide-and-seek-data checkpoint_2700.pth /checkpoints/

Run training:
    modal run modal_train.py

Monitor logs:
    modal app logs

Download results:
    modal volume get hide-and-seek-data /checkpoints/ ./downloaded_checkpoints/
    modal volume get hide-and-seek-data /videos/ ./downloaded_videos/
"""
import modal

# ---------------------------------------------------------------------------
# Modal app and persistent volume
# ---------------------------------------------------------------------------
app = modal.App("hide-and-seek-training")

# Persistent volume for checkpoints and videos (survives container restarts)
vol = modal.Volume.from_name("hide-and-seek-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container image with all dependencies
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libglfw3", "libosmesa6")
    .pip_install(
        "torch",
        "numpy",
        "mujoco>=3.3",
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
        "tqdm",
    )
    .add_local_dir(
        ".", remote_path="/root/hide_and_seek",
        ignore=[
            ".git", "**/__pycache__", "*.pth", "*.mp4", "*.gif",
            "assets", "my_videos", "rl_videos", "logs.txt",
            "downloaded_checkpoints", "downloaded_videos",
        ],
    )
)


def _latest_valid_checkpoint(ckpt_dir: str = "/data/checkpoints") -> str:
    """Newest architecture-compatible checkpoint in the volume, or ''.

    Validates by loading: skips stale old-run files (30-dim lidar embedder)
    and anything corrupt. Critical for container restarts — Modal re-executes
    the function with its ORIGINAL arguments after a crash, which would
    otherwise roll training back to the originally requested checkpoint.
    """
    import os
    import re
    import torch

    if not os.path.isdir(ckpt_dir):
        return ""
    cands = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"checkpoint_(\d+)\.pth", f)
        if m:
            cands.append((int(m.group(1)), f))
    cands.sort(reverse=True)
    for _, f in cands:
        try:
            ck = torch.load(os.path.join(ckpt_dir, f), map_location="cpu")
            w = ck["actor"]["sensory_pipeline.lidar_embedder.net.0.weight"]
            if w.shape[1] == 32 and "optimizer" in ck:
                return f
        except Exception:
            continue
    return ""


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",         # A10G handles the heavy COMA advantage engine calculations
    cpu=40.0,           # 40 physical cores (Modal max) — 1 per MuJoCo env worker
    memory=32768,       # 32 GB RAM
    timeout=86400,      # 24 hour max (Modal's limit)
    volumes={"/data": vol},
)
def train_on_modal(
    iterations: int = 40000,
    num_envs: int = 32,
    resume_checkpoint: str = "auto",
):
    """Run MAPPO training inside a Modal container.

    resume_checkpoint:
      "auto" (default) — resume from the newest valid checkpoint in the
                         volume; fresh start if none exists. Crash-safe.
      "checkpoint_X.pth" — resume from a specific checkpoint.
      "" — force a fresh start.
    """
    import os
    import sys
    import shutil

    # CRITICAL: Lock down all hidden multi-threading to 1 thread per process
    # to prevent context-switching thrashing across the 40 cores.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Add our code to the Python path
    sys.path.insert(0, "/root/hide_and_seek")
    os.chdir("/root/hide_and_seek")

    # Resolve "auto" to the newest valid checkpoint (crash-restart safe)
    if resume_checkpoint == "auto":
        resume_checkpoint = _latest_valid_checkpoint()
        print(f"[AUTO-RESUME] Latest valid checkpoint: {resume_checkpoint or 'none — fresh start'}")

    # If resuming, copy checkpoint from volume into the working directory
    ckpt_path = ""
    if resume_checkpoint:
        src = f"/data/checkpoints/{resume_checkpoint}"
        if os.path.exists(src):
            shutil.copy(src, f"/root/hide_and_seek/{resume_checkpoint}")
            ckpt_path = f"/root/hide_and_seek/{resume_checkpoint}"
            print(f"Copied checkpoint from volume: {src}")
        else:
            print(f"WARNING: Checkpoint {src} not found in volume! Starting fresh.")
            # List what IS in the volume for debugging
            ckpt_dir = "/data/checkpoints"
            if os.path.exists(ckpt_dir):
                print(f"  Available files: {os.listdir(ckpt_dir)}")

    # Monkey-patch train.py to save checkpoints and videos to the volume
    from train import train as original_train
    import train as train_module

    # Override the checkpoint and video save paths
    _original_torch_save = __import__('torch').save

    def _patched_save(obj, f, *args, **kwargs):
        """Save to both local dir AND the persistent volume."""
        _original_torch_save(obj, f, *args, **kwargs)
        # Also copy to volume
        if isinstance(f, str) and f.endswith('.pth'):
            os.makedirs("/data/checkpoints", exist_ok=True)
            dest = f"/data/checkpoints/{os.path.basename(f)}"
            shutil.copy(f, dest)
            vol.commit()  # Persist to volume immediately
            print(f"  [MODAL] Checkpoint saved to volume: {dest}")

    __import__('torch').save = _patched_save

    # Also patch evaluate to save videos to volume
    _original_evaluate = train_module.evaluate

    def _patched_evaluate(ckpt, out_vid):
        _original_evaluate(ckpt, out_vid)
        if os.path.exists(out_vid):
            os.makedirs("/data/videos", exist_ok=True)
            dest = f"/data/videos/{os.path.basename(out_vid)}"
            shutil.copy(out_vid, dest)
            vol.commit()
            print(f"  [MODAL] Video saved to volume: {dest}")

    train_module.evaluate = _patched_evaluate

    print(f"\n{'='*60}")
    print(f"  MODAL TRAINING START")
    print(f"  Iterations: {iterations}")
    print(f"  Environments: {num_envs}")
    print(f"  Resume from: {ckpt_path or 'scratch'}")
    print(f"  GPU: A10G | CPU: 40 cores (32 envs + 8 ML/OS) | RAM: 32 GB")
    print(f"{'='*60}\n")

    # Run the actual training
    original_train(
        num_iterations=iterations,
        num_envs=num_envs,
        device="cuda",
        resume_path=ckpt_path,
    )


# ---------------------------------------------------------------------------
# Entry point — `modal run modal_train.py`
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    iterations: int = 40000,
    envs: int = 32,
    resume: str = "auto",
):
    """
    Usage:
        modal run modal_train.py                                    # fresh start
        modal run modal_train.py --resume checkpoint_2700.pth       # resume
        modal run modal_train.py --iterations 80000 --envs 48       # custom
    """
    train_on_modal.remote(
        iterations=iterations,
        num_envs=envs,
        resume_checkpoint=resume,
    )
