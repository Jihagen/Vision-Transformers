"""
Image loading and preprocessing utilities.

Handles PIL loading, max-side resizing, and the OOM-triggered downscale ladder
used during activation collection.

Migrated from: get_activations_base.py (preprocess_image, downscale ladder)
"""

from __future__ import annotations
from pathlib import Path

# from PIL import Image

MAX_SIDE_DEFAULT = 800
DOWNSCALE_LADDER = [800, 640, 512, 384, 256, 128]


def load_image(img_path: str | Path):
    """Load image from disk as PIL.Image in RGB mode."""
    raise NotImplementedError


def preprocess_image(img_path: str | Path, max_side: int = MAX_SIDE_DEFAULT):
    """
    Load and resize image so its longest side is at most max_side pixels.

    Returns:
        PIL.Image
    """
    raise NotImplementedError


def downscale_image(image, current_max_side: int) -> tuple:
    """
    Return (resized_image, next_max_side) for the OOM retry ladder.

    Returns (None, None) if current_max_side is already the minimum.
    """
    raise NotImplementedError
