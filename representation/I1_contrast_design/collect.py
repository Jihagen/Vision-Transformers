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
        "interestingness": str,         # one of VALID_LABELS
        "explanation": str,
        "embeddings": {
          "vision_layer_<N>_cls":       np.ndarray,  # shape (D_vis,)
          "llm_layer_<N>_rating_token": np.ndarray,  # shape (D_llm,)
        }
      }, ...
    ]
  }
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.hooks import HookState, register_extract_hooks
from utils.image_utils import preprocess_image, remove_batch_dimension, DOWNSCALE_LADDER
from utils.prompt_builder import build_persona_prompt, extract_last_json_block, RATING_ANCHOR
from utils.model_loader import load_model_and_processor
from utils.hpc import print_cuda_memory, emergency_cleanup

logger = logging.getLogger(__name__)


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load the image manifest pickle and validate required columns."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing manifest: {path}")
    df = pd.read_pickle(path)
    required = {"filename", "img_path"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must have {required}, got {set(df.columns)}")
    return df.copy()


def get_missing_rows(manifest_df: pd.DataFrame, results_list: list[dict]) -> pd.DataFrame:
    """Return manifest rows whose filename is not yet in results_list."""
    done = {r.get("filename") for r in results_list if isinstance(r, dict) and "filename" in r}
    return manifest_df[~manifest_df["filename"].isin(done)].reset_index(drop=True)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_existing_results(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        obj = np.load(path, allow_pickle=True).item()
        return obj.get("results", []) if isinstance(obj, dict) else []
    except Exception as e:
        logger.warning(f"Could not load existing results from {path}: {e}")
        return []


def _save_checkpoint(
    ckpt_path: str,
    ckpt_temp: str,
    results_list: list,
    manifest_df: pd.DataFrame,
    persona_key: str,
) -> None:
    payload = {
        "version": "experiment_v1",
        "persona_key": persona_key,
        "selected_images": manifest_df[["filename", "img_path"]].to_dict(orient="records"),
        "results": results_list,
    }
    np.save(ckpt_temp, payload)
    os.replace(ckpt_temp, ckpt_path)
    logger.info(f"Checkpoint saved → {ckpt_path}")


def load_checkpoint(checkpoint_path: str | Path) -> dict:
    """Load an in-progress checkpoint and return already-processed results."""
    return np.load(checkpoint_path, allow_pickle=True).item()


# ── Two-stage model inference ─────────────────────────────────────────────────

def model_response(
    prompt: str,
    image_path: str | Path,
    model,
    processor,
    state: HookState,
    image_max_size: int = 800,
    max_new_tokens: int = 64,
) -> tuple[str, dict]:
    """
    Run two-stage inference for one image × persona pair.

    Stage 1: full generation — obtain interestingness rating and explanation.
             Vision hooks fire during this pass (phase="vision_once").
    Stage 2: teacher-forced single step — capture LLM layer hidden states at
             the first label token (phase="rating_step").

    Returns:
        (decoded_text, parsed_json_dict)
    Raises:
        torch.cuda.OutOfMemoryError — caller handles via downscale
        ValueError                  — JSON parse failure; caller expands tokens
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image = preprocess_image(image_path, max_side=image_max_size)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=chat_input,
        images=image,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        max_length=512,
    )

    target_device = next(model.parameters()).device
    _keep = {"input_ids", "attention_mask", "pixel_values"}
    model_inputs = {
        k: v.to(target_device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
        if k in _keep
    }

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    # Stage 1: full generation — vision hooks fire here
    logger.info(f"Generation (image_max_size={image_max_size}, max_new_tokens={max_new_tokens})...")
    state.phase = "vision_once"
    try:
        print_cuda_memory("Before generate")
        with torch.no_grad():
            gen_ids = model.generate(**model_inputs, **gen_kwargs)
        print_cuda_memory("After generate")
    finally:
        state.phase = "idle"

    decoded = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    try:
        data = extract_last_json_block(decoded)
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}; raw tail: {decoded[-400:]}")
        raise ValueError("Invalid JSON in response")

    # Stage 2: teacher-forced one step — LLM hooks fire here
    full_seq_ids = gen_ids[0].tolist()
    input_len    = model_inputs["input_ids"].shape[1]
    forced_ids   = processor.tokenizer.encode(RATING_ANCHOR, add_special_tokens=False)
    prefix_ids   = full_seq_ids[:input_len] + forced_ids

    local = {
        "input_ids": torch.tensor([prefix_ids], device=model_inputs["input_ids"].device),
        "attention_mask": torch.ones(
            1, len(prefix_ids),
            device=model_inputs["attention_mask"].device,
            dtype=model_inputs["attention_mask"].dtype,
        ),
        "pixel_values": model_inputs["pixel_values"],
    }

    logger.info("Teacher-forced one-step for LLM layer capture...")
    state.phase = "rating_step"
    with torch.no_grad():
        _ = model.generate(
            **local,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    state.phase = "idle"

    del gen_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()

    return decoded, data


def call_model_with_retries(
    prompt: str,
    image_path: str | Path,
    model,
    processor,
    state: HookState,
    force_image_max_size: int | None = None,
) -> tuple[str, dict, int]:
    """
    Wrap model_response with OOM (image-downscale) and JSON-parse retry logic.

    Retry ladder:
      - OOM: downscale image (800 → 640 → 512 → 384 → 256 → 128), retry
      - Invalid JSON: expand max_new_tokens (64 → 96 → 128), retry
      - CUDA illegal memory access: abort with error logged

    Args:
        force_image_max_size: if set, skip the downscale ladder and use this
            fixed size (used by reconcile to enforce consistent sizes).

    Returns:
        (decoded_text, parsed_json_dict, image_max_size_used)
    """
    import gc

    if force_image_max_size is not None:
        state.reset_embeddings()
        decoded, data = model_response(
            prompt, image_path, model, processor, state,
            image_max_size=force_image_max_size, max_new_tokens=64,
        )
        return decoded, data, force_image_max_size

    initial_size = DOWNSCALE_LADDER[0]   # 800

    try:
        state.reset_embeddings()
        decoded, data = model_response(prompt, image_path, model, processor, state,
                                       image_max_size=initial_size, max_new_tokens=64)
        return decoded, data, initial_size

    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        e.__traceback__ = None
        last_exc = e
        for oom_size in DOWNSCALE_LADDER[1:]:
            logger.warning(f"OOM; retrying with image_max_size={oom_size}")
            emergency_cleanup(embeddings_dict=state.embeddings)
            state.reset_embeddings()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            try:
                decoded, data = model_response(
                    prompt, image_path, model, processor, state,
                    image_max_size=oom_size, max_new_tokens=64,
                )
                return decoded, data, oom_size
            except RuntimeError as e2:
                if "illegal memory access" in str(e2).lower():
                    logger.error("CUDA illegal memory access — aborting.")
                    raise
                if "out of memory" in str(e2).lower():
                    e2.__traceback__ = None
                    last_exc = e2
                    continue
                raise
        logger.error("OOM persists at minimum image size; giving up on this image.")
        raise last_exc

    except ValueError as e:
        if "Invalid JSON" not in str(e):
            raise
        for n_tokens in [96, 128]:
            logger.warning(f"Invalid JSON; retrying with max_new_tokens={n_tokens}")
            emergency_cleanup(embeddings_dict=state.embeddings)
            state.reset_embeddings()
            try:
                decoded, data = model_response(
                    prompt, image_path, model, processor, state,
                    image_max_size=initial_size, max_new_tokens=n_tokens,
                )
                return decoded, data, initial_size
            except ValueError as e2:
                if "Invalid JSON" not in str(e2):
                    raise
        raise


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(
    model_path: str,
    persona_key: str,
    persona_dict: dict | None,
    manifest_path: str | Path,
    output_dir: str | Path,
    offload_dir: str | None = None,
    checkpoint_every: int = 2,
    max_retry_rounds: int = 5,
    model_and_processor=None,
) -> Path:
    """
    Main collection loop: for each image in manifest, call model and store embeddings.

    Args:
        model_path:           path to HF model directory (used only if model_and_processor is None)
        persona_key:          short identifier used in output filename and stored results
        persona_dict:         persona definition dict (None → blank/baseline run)
        manifest_path:        path to selected_uniform_*.pkl image manifest
        output_dir:           directory for results_<persona_key>.npy and checkpoints
        offload_dir:          CPU offload dir for model weights (None = no offloading)
        checkpoint_every:     save checkpoint after every N images
        max_retry_rounds:     retry passes over the manifest for skipped images
        model_and_processor:  optional (model, processor) tuple; if provided, skips loading

    Returns:
        Path to final saved .npy file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_and_processor is not None:
        model, processor = model_and_processor
    else:
        model, processor = load_model_and_processor(model_path, offload_dir=offload_dir)

    state = HookState()
    register_extract_hooks(model, state)

    manifest_df = load_manifest(manifest_path)
    logger.info(f"Loaded manifest with {len(manifest_df)} target images")

    ckpt_path = str(output_dir / f"results_{persona_key}.npy")
    ckpt_temp = str(output_dir / f"results_{persona_key}_temp.npy")

    prompt = build_persona_prompt(persona_dict)

    existing     = _load_existing_results(ckpt_path)
    results_list = list(existing)
    logger.info(f"Existing results: {len(results_list)}")

    total_new = 0

    try:
        for retry_round in range(max_retry_rounds + 1):
            missing_df = get_missing_rows(manifest_df, results_list)
            if len(missing_df) == 0:
                logger.info(f"All images processed for {persona_key}")
                break

            logger.info(f"Round {retry_round}/{max_retry_rounds} | missing: {len(missing_df)}")
            processed_this_round = 0

            for idx, row in enumerate(missing_df.itertuples(index=False), start=1):
                img_path_str = str(row.img_path)
                try:
                    logger.info(f"[{retry_round}] {idx}/{len(missing_df)}: {img_path_str}")
                    state.reset_embeddings()

                    _, data, size_used = call_model_with_retries(
                        prompt, img_path_str, model, processor, state
                    )
                    if size_used < DOWNSCALE_LADDER[0]:
                        logger.info(f"Used downscaled size {size_used} for {row.filename}")

                    logger.info(f"Captured {len(state.embeddings)} embedding tensors")
                    embeds = {k: remove_batch_dimension(v) for k, v in state.embeddings.items()}
                    state.reset_embeddings()

                    rec = {
                        "persona_key":     persona_key,
                        "filename":        os.path.basename(img_path_str),
                        "img_path":        img_path_str,
                        "interestingness": data["interestingness"],
                        "explanation":     data["explanation"],
                        "image_max_size":  size_used,
                        "embeddings":      embeds,
                    }
                    results_list.append(rec)
                    total_new += 1
                    processed_this_round += 1

                    if len(results_list) % checkpoint_every == 0:
                        _save_checkpoint(ckpt_path, ckpt_temp, results_list, manifest_df, persona_key)

                except Exception as e:
                    if "illegal memory access" in str(e).lower():
                        logger.error(f"CUDA illegal memory access on {img_path_str} — aborting run.")
                        raise
                    logger.warning(f"Error on {img_path_str}: {e}; skipping")
                    emergency_cleanup(embeddings_dict=state.embeddings)
                    continue

            _save_checkpoint(ckpt_path, ckpt_temp, results_list, manifest_df, persona_key)
            logger.info(f"Round {retry_round} done; processed this round: {processed_this_round}")

            if processed_this_round == 0:
                logger.warning("No progress this round; stopping retries early.")
                break

        final_missing = get_missing_rows(manifest_df, results_list)
        logger.info(
            f"Total new for {persona_key}: {total_new} | "
            f"remaining missing: {len(final_missing)}"
        )

        if len(final_missing) > 0:
            missing_path = str(output_dir / f"missing_{persona_key}.pkl")
            final_missing.to_pickle(missing_path)
            logger.warning(f"Saved missing rows → {missing_path}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user; last checkpoint is on disk.")
        raise
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        state.remove_hooks()

    return Path(ckpt_path)
