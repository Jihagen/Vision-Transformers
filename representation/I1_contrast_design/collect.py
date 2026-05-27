"""
Activation collection for a single persona condition.

Core logic migrated from get_activations_base.py.  Key design decisions preserved:
  - Two-stage inference: full forward pass (get rating) + teacher-forced one-step
    (capture LLM layer activations at the label token position).
  - OOM fallback: downscale image through DOWNSCALE_LADDER before aborting.
  - JSON retry: expand max_new_tokens if model output cannot be parsed.
  - Checkpoint every N images so long runs survive preemption.

Result format (saved as .npy pickled dict):
  {
    "version": "experiment_v1",
    "persona_key": str,
    "selected_images": [{"filename": str, "img_path": str}, ...],
    "results": [
      {
        "persona_key": str,
        "filename": str,
        "img_path": str,
        "interestingness": str,        # one of VALID_LABELS
        "explanation": str,
        "embeddings": {
          "vision_layer_<N>_cls":        np.ndarray,  # shape (D_vis,)
          "llm_layer_<N>_rating_token":  np.ndarray,  # shape (D_llm,)
        }
      }, ...
    ]
  }
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np

# from utils.model_loader import load_model_and_processor
# from utils.hooks import HookState, register_extract_hooks
# from utils.image_utils import preprocess_image, downscale_image
# from utils.prompt_builder import build_persona_prompt, RATING_ANCHOR, parse_model_response
# from utils.hpc import get_offload_dir


def model_response(
    prompt: str,
    image,           # PIL.Image
    model,
    processor,
    state,           # HookState
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """
    Run two-stage inference for one image × persona pair.

    Stage 1: full generation — obtain interestingness rating and explanation.
    Stage 2: teacher-forced single step — capture LLM layer hidden states at
             the first label token, avoiding label-token representation confound.

    Returns:
        dict with keys: interestingness, explanation, embeddings
    Raises:
        torch.cuda.OutOfMemoryError  — caller handles via downscale
        ValueError                   — JSON parse failure; caller expands tokens
    """
    raise NotImplementedError


def call_model_with_retries(
    prompt: str,
    img_path: str | Path,
    model,
    processor,
    state,           # HookState
) -> dict[str, Any]:
    """
    Wrap model_response with OOM and JSON-parse retry logic.

    Retry ladder:
      - OOM: downscale image (800 → 640 → 512 → 384 → 256 → 128), retry
      - Invalid JSON: expand max_new_tokens (64 → 96 → 128), retry
      - CUDA illegal memory access: abort with error logged

    Returns result dict or None on unrecoverable failure.
    """
    raise NotImplementedError


def run_experiment(
    model_path: str,
    persona_key: str,
    persona_dict: dict | None,
    manifest_path: str | Path,
    output_dir: str | Path,
    offload_dir: str | None = None,
    checkpoint_every: int = 2,
) -> Path:
    """
    Main collection loop: for each image in manifest, call model and store embeddings.

    Args:
        persona_key:     short identifier used in output filename and stored results
        persona_dict:    persona definition dict (None → blank/baseline run)
        manifest_path:   path to selected_uniform_*.pkl image manifest
        output_dir:      directory for results_<persona_key>.npy and checkpoints
        offload_dir:     CPU offload dir for model weights (None = auto)
        checkpoint_every: save checkpoint after every N images

    Returns:
        Path to final saved .npy file
    """
    raise NotImplementedError


def load_checkpoint(checkpoint_path: str | Path) -> dict:
    """Load an in-progress checkpoint and return already-processed results."""
    data = np.load(checkpoint_path, allow_pickle=True).item()
    return data
