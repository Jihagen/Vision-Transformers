"""
Model and processor loading for Llama-4-Scout-17B (or any HF vision-language model).

Migrated from: get_activations_base.py (model/processor init block in run_experiment)
"""

from __future__ import annotations
import os
import sys
import logging

logger = logging.getLogger(__name__)

# ── Local HF/dev path (fallback for custom transformers builds) ───────────────
DEVROOT  = os.environ.get("DEVROOT", "/anvme/workspace/iwi5268h-vision-transformers/hpc_infrastructure/dev")
HF_DEPS  = os.path.join(DEVROOT, "hf_deps")
TF_SRC   = os.path.join(DEVROOT, "transformers", "src")

if os.path.isdir(HF_DEPS):
    sys.path.insert(0, HF_DEPS)
if os.path.isdir(TF_SRC):
    sys.path.insert(0, TF_SRC)

try:
    import huggingface_hub.utils._validators as _hf_val
    _hf_val.validate_repo_id = lambda *a, **k: None
except Exception:
    pass

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    import transformers.utils.hub as _tfh
    _tfh.validate_repo_id = lambda *a, **k: None
except Exception:
    pass

MODEL_PATH_DEFAULT = "/anvme/workspace/iwi5268h-vision-transformers/hpc_infrastructure/hf_cache"


def load_model(
    model_path: str = MODEL_PATH_DEFAULT,
    offload_dir: str | None = None,
    device_map: str = "auto",
    max_memory: dict | None = None,
) -> object:
    """
    Load the vision-language model with optional CPU offloading.

    Args:
        max_memory: Optional per-device memory cap for accelerate's auto device_map,
            e.g. {0: "40GiB", 1: "38GiB", ..., 5: "40GiB"}.  When provided, accelerate
            will not place more than the specified amount of model weights on each device,
            leaving headroom for forward-pass activations.  If None, accelerate uses the
            full GPU capacity (may leave as little as 35 MB free on dense GPUs, causing
            OOM during training forward passes).

    Returns:
        model in eval mode
    """
    kwargs: dict = dict(
        local_files_only=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        offload_folder=offload_dir,
        offload_state_dict=offload_dir is not None,
    )
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
        logger.info(f"Loading model from {model_path} with max_memory={max_memory} ...")
    else:
        logger.info(f"Loading model from {model_path} ...")

    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    model.eval()

    if hasattr(model, "hf_device_map"):
        for k, v in model.hf_device_map.items():
            logger.info(f"  {k} → {v}")

    return model


def load_processor(model_path: str = MODEL_PATH_DEFAULT) -> object:
    """Load the tokenizer/processor for the model."""
    return AutoProcessor.from_pretrained(model_path, local_files_only=True)


def load_model_and_processor(
    model_path: str = MODEL_PATH_DEFAULT,
    offload_dir: str | None = None,
    max_memory: dict | None = None,
) -> tuple:
    """Convenience wrapper returning (model, processor)."""
    model = load_model(model_path, offload_dir=offload_dir, max_memory=max_memory)
    processor = load_processor(model_path)
    return model, processor
