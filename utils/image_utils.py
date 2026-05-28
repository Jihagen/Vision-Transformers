"""
Image loading and preprocessing utilities.

Migrated from: get_activations_base.py (preprocess_image, remove_batch_dimension)
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MAX_SIDE_DEFAULT = 800
DOWNSCALE_LADDER = [800, 640, 512, 384, 256, 128]


def load_image(img_path: str | Path) -> Image.Image:
    """Load image from disk as PIL.Image in RGB mode."""
    return Image.open(img_path).convert("RGB")


def preprocess_image(img_path: str | Path, max_side: int = MAX_SIDE_DEFAULT) -> Image.Image:
    """
    Load and resize image so its longest side is at most max_side pixels.
    """
    image = load_image(img_path)
    w, h = image.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        logger.debug(f"Resized image from {w}x{h} to {int(w*ratio)}x{int(h*ratio)}")
    return image


def downscale_image(image: Image.Image, current_max_side: int) -> tuple:
    """
    Return (resized_image, next_max_side) for the OOM retry ladder.

    Returns (None, None) if current_max_side is already the minimum.
    """
    try:
        idx = DOWNSCALE_LADDER.index(current_max_side)
    except ValueError:
        idx = 0
    if idx >= len(DOWNSCALE_LADDER) - 1:
        return None, None
    next_size = DOWNSCALE_LADDER[idx + 1]
    w, h = image.size
    ratio = next_size / max(w, h)
    resized = image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    return resized, next_size


def remove_batch_dimension(t: torch.Tensor) -> np.ndarray:
    """Convert a batched (1, ...) or (N, ...) tensor to a numpy array without the batch dim."""
    try:
        cpu = t.to(torch.float32).cpu()
        if cpu.dim() > 0:
            unbatched = cpu.squeeze(0) if cpu.shape[0] == 1 else cpu[0]
        else:
            unbatched = cpu
        return unbatched.numpy()
    except Exception as e:
        logger.warning(f"remove_batch_dimension error: {e}")
        return t.to(torch.float32).cpu().numpy()
