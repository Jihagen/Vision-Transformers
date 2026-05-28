"""
HPC / SLURM environment helpers.

Migrated from: get_activations_base.py (print_cuda_memory, emergency_cleanup)
"""

from __future__ import annotations
import gc
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Set before importing torch so it applies to the allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_offload_dir(base: str = "hpc_infrastructure/offload_dir", suffix: str = "") -> Path:
    """
    Return path to a CPU offload directory, creating it if necessary.

    Args:
        base:   base directory path (relative to project root or absolute)
        suffix: appended to base so parallel jobs use distinct dirs
    """
    path = Path(f"{base}_{suffix}" if suffix else base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_slurm_task_id() -> int | None:
    """Return SLURM_ARRAY_TASK_ID as int, or None if not in a SLURM array job."""
    val = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(val) if val is not None else None


def setup_cuda_env(visible_devices: str | None = None) -> None:
    """Set CUDA_VISIBLE_DEVICES before importing torch."""
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


def print_cuda_memory(prefix: str = "") -> None:
    """Print current GPU memory usage for all visible devices."""
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 2**30
            r = torch.cuda.memory_reserved()  / 2**30
            logger.debug(f"{prefix} CUDA mem → Allocated: {a:.2f} GiB, Reserved: {r:.2f} GiB")
    except ImportError:
        pass


log_gpu_memory = print_cuda_memory   # alias


def emergency_cleanup(preserve_embeddings: bool = False, embeddings_dict: dict | None = None) -> None:
    """
    Free GPU memory and run Python GC.

    Args:
        preserve_embeddings: if False, also clear the embeddings_dict in-place
        embeddings_dict:     the HookState.embeddings dict to clear (if preserve_embeddings=False)
    """
    if not preserve_embeddings and embeddings_dict is not None:
        embeddings_dict.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.warning(f"ipc_collect failed: {e}")
    except ImportError:
        pass
    gc.collect()
