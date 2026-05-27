"""
HPC / SLURM environment helpers.

Handles offload directory management, CUDA device selection, environment
variable setup, and SLURM array job index parsing so runners stay clean.
"""

from __future__ import annotations
import os
from pathlib import Path


def get_offload_dir(base: str = "offload_dir", suffix: str = "") -> Path:
    """
    Return the path to a CPU offload directory, creating it if necessary.

    Args:
        base: base directory name
        suffix: appended to base to allow parallel jobs to use distinct dirs
                (e.g. suffix from SLURM_ARRAY_TASK_ID)
    """
    raise NotImplementedError


def get_slurm_task_id() -> int | None:
    """Return SLURM_ARRAY_TASK_ID as int, or None if not in a SLURM array job."""
    val = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(val) if val is not None else None


def setup_cuda_env(visible_devices: str | None = None):
    """
    Set CUDA_VISIBLE_DEVICES and any other environment variables required
    before importing torch or transformers.
    """
    raise NotImplementedError


def log_gpu_memory(tag: str = ""):
    """Print current GPU memory usage for all visible devices."""
    raise NotImplementedError
