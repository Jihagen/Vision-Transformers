"""
Generic HuggingFace vision-language model loading and image preprocessing.

No cluster/offline-cache assumptions beyond what `transformers` itself makes;
pass `model_path` as any local directory or HF Hub repo id that
`AutoModelForImageTextToText` / `AutoProcessor` can load.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


def load_vlm(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    offload_dir: str | None = None,
    max_memory: dict | None = None,
) -> tuple:
    """
    Load a HuggingFace vision-language model + processor.

    Args:
        model_path: local directory or HF Hub repo id.
        max_memory: optional per-device memory cap for accelerate's device_map,
            e.g. {0: "38GiB", 1: "38GiB", ...}. Leave headroom for activations —
            accelerate otherwise packs weights up to the full device capacity.

    Returns:
        (model, processor), model in eval mode.
    """
    kwargs: dict = dict(
        device_map=device_map,
        torch_dtype=torch_dtype,
        offload_folder=offload_dir,
        offload_state_dict=offload_dir is not None,
    )
    if max_memory is not None:
        kwargs["max_memory"] = max_memory

    logger.info(f"Loading model from {model_path} (max_memory={max_memory})...")
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    model.eval()

    if hasattr(model, "hf_device_map"):
        for k, v in model.hf_device_map.items():
            logger.debug(f"  {k} -> {v}")

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# ── Image preprocessing ─────────────────────────────────────────────────────

MAX_SIDE_DEFAULT = 800
DOWNSCALE_LADDER = [800, 640, 512, 384, 256, 128]


def load_image(img_path: str | Path) -> Image.Image:
    return Image.open(img_path).convert("RGB")


def preprocess_image(img_path: str | Path, max_side: int = MAX_SIDE_DEFAULT) -> Image.Image:
    """Load and resize so the longest side is at most max_side pixels."""
    image = load_image(img_path)
    w, h = image.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    return image


def downscale_image(image: Image.Image, current_max_side: int) -> tuple:
    """Return (resized_image, next_max_side) for an OOM retry ladder, or (None, None) at the floor."""
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
    cpu = t.to(torch.float32).cpu()
    if cpu.dim() > 0:
        unbatched = cpu.squeeze(0) if cpu.shape[0] == 1 else cpu[0]
    else:
        unbatched = cpu
    return unbatched.numpy()
