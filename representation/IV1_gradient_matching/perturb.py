"""
IV.1 Gradient-matching image perturbation.

TIER IV step 2 (README Sec 9.1): given a validated activation-injection
direction (e.g. v_interest_blank @ language_29, alpha chosen from TIER IV
step 1's saturation sweep), find a bounded pixel-space perturbation of an
INPUT IMAGE such that its rating-token hidden state at that layer moves
toward the same target the activation injection produces -- i.e.
gradient-match the image to the *intervention's effect on activations*
for THIS image, not just the raw direction vector.

Pipeline
--------
1. compute_baseline_hidden -- one no-grad forward pass on the original image,
   teacher-forced to the rating-token position (same position the md_vectors
   and injection hooks operate on). Returns h_baseline and the reusable
   token tensors.
2. gradient_match_perturbation -- PGD loop: repeatedly forward
   (image + delta) with grad enabled, maximise
   cos(h(image+delta), h_baseline + alpha * v_hat), sign-gradient step on
   delta, clip to an L_inf ball (in processor-normalised pixel_values units).
3. evaluate_perturbation -- close the loop: run full generation (NO
   activation injection) on the original vs. perturbed pixel_values and
   compare interestingness ratings.

Status: first-draft scaffold, not yet smoke-tested on GPU (both GPU
allocations were occupied by Branch D / TIER IV step 1 jobs at the time this
was written). Before scaling up:
  - smoke-test on 1 image with n_steps=2-3 to confirm the forward/backward
    pass works under this model's device_map="auto" + bfloat16 setup
  - calibrate epsilon/lr empirically (pixel_values units, not raw [0,255])
  - pick alpha from TIER IV step 1's wide dose-response sweep
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _build_teacher_forced_inputs(model, processor, image, prompt: str):
    """
    Build (full_input_ids, full_attention_mask, pixel_values, prompt_input_ids,
    prompt_attention_mask) for a single forward pass at the rating-token
    position: prompt + RATING_ANCHOR teacher-forced, matching the position
    register_extract_hooks captures during phase="rating_step" (i.e. the
    position the md_vectors / injection vectors were defined at).
    """
    from utils.prompt_builder import RATING_ANCHOR

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=chat_input, images=image, return_tensors="pt",
        add_special_tokens=True, truncation=False, max_length=512,
    )

    target_device = next(model.parameters()).device
    prompt_ids = inputs["input_ids"].to(target_device)
    prompt_mask = inputs["attention_mask"].to(target_device)
    pixel_values = inputs["pixel_values"].to(target_device)

    forced_ids = processor.tokenizer.encode(RATING_ANCHOR, add_special_tokens=False)
    forced = torch.tensor([forced_ids], device=target_device)
    full_input_ids = torch.cat([prompt_ids, forced], dim=1)
    full_attention_mask = torch.cat([prompt_mask, torch.ones_like(forced)], dim=1)

    return full_input_ids, full_attention_mask, pixel_values, prompt_ids, prompt_mask


def _hidden_at_layer(model, layer_key: str, full_input_ids, full_attention_mask,
                      pixel_values, requires_grad: bool) -> torch.Tensor:
    """
    Single forward pass; returns the rating-token (last-token) hidden state
    at layer_key, shape (1, D). If requires_grad, runs under
    torch.enable_grad() -- caller must pass a pixel_values tensor that is
    part of an autograd graph for gradients to flow back to it.
    """
    from utils.hooks import register_grad_extract_hook

    container: dict = {}
    handle = register_grad_extract_hook(model, layer_key, container)
    try:
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                pixel_values=pixel_values.to(model.dtype),
                use_cache=False,
            )
    finally:
        handle.remove()
    return container["h"]


def compute_baseline_hidden(model, processor, image_path: str | Path, prompt: str,
                             layer_key: str) -> dict:
    """
    No-grad forward pass on the ORIGINAL image. Returns a dict with:
        h_baseline:            (1, D) float32 cpu tensor
        pixel_values:          (1, ...) float32 tensor, model device
        full_input_ids / full_attention_mask:   teacher-forced sequence
        prompt_input_ids / prompt_attention_mask: for later generation
    """
    from utils.image_utils import preprocess_image

    image = preprocess_image(image_path, max_side=800)
    full_ids, full_mask, pixel_values, prompt_ids, prompt_mask = \
        _build_teacher_forced_inputs(model, processor, image, prompt)

    h = _hidden_at_layer(model, layer_key, full_ids, full_mask, pixel_values, requires_grad=False)
    return {
        "h_baseline": h.detach().float().cpu(),
        "pixel_values": pixel_values.detach().float(),
        "full_input_ids": full_ids,
        "full_attention_mask": full_mask,
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": prompt_mask,
    }


def gradient_match_perturbation(
    model, processor,
    image_path: str | Path,
    prompt: str,
    layer_key: str,
    direction_vector: np.ndarray,
    alpha: float,
    epsilon: float = 0.05,
    lr: float = 0.01,
    n_steps: int = 20,
) -> dict:
    """
    PGD: find a bounded pixel-space perturbation delta (L_inf <= epsilon, in
    processor-normalised pixel_values units) such that the rating-token
    hidden state at layer_key for (image + delta) moves toward

        target_h = h_baseline + alpha * direction_vector_hat

    i.e. the SAME target an activation injection of alpha * direction_vector
    at layer_key would produce for this image.

    Returns dict with pixel_values_original / pixel_values_perturbed (model
    dtype, ready for model.generate), prompt_input_ids / prompt_attention_mask,
    and per-step loss_history / cos_history diagnostics.
    """
    base = compute_baseline_hidden(model, processor, image_path, prompt, layer_key)

    v = torch.tensor(direction_vector, dtype=torch.float32)
    v = v / v.norm()
    target_h = (base["h_baseline"] + alpha * v).to(base["pixel_values"].device)

    pixel_values = base["pixel_values"]
    full_ids, full_mask = base["full_input_ids"], base["full_attention_mask"]
    delta = torch.zeros_like(pixel_values)

    loss_history, cos_history = [], []
    for step in range(n_steps):
        delta.requires_grad_(True)
        perturbed = pixel_values + delta
        h = _hidden_at_layer(model, layer_key, full_ids, full_mask, perturbed, requires_grad=True)
        h = h.float()
        cos = torch.nn.functional.cosine_similarity(h, target_h, dim=-1).mean()
        loss = 1.0 - cos
        loss.backward()

        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta = (delta - lr * grad_sign).clamp(-epsilon, epsilon)
        delta = delta.detach()

        loss_history.append(loss.item())
        cos_history.append(cos.item())
        logger.info(f"  step {step + 1}/{n_steps}: loss={loss.item():.4f} cos={cos.item():.4f}")

    return {
        "pixel_values_original": pixel_values.to(model.dtype),
        "pixel_values_perturbed": (pixel_values + delta).to(model.dtype),
        "prompt_input_ids": base["prompt_input_ids"],
        "prompt_attention_mask": base["prompt_attention_mask"],
        "loss_history": loss_history,
        "cos_history": cos_history,
    }


def evaluate_perturbation(model, processor, result: dict, max_new_tokens: int = 64) -> dict:
    """
    Step 3 of the loop: run full generation (NO activation injection) on the
    ORIGINAL vs PERTURBED pixel_values, continuing from the prompt-only
    token sequence (same as model_response stage 1).

    Returns {"original": (rating, explanation), "perturbed": (rating, explanation)}.
    """
    from utils.prompt_builder import extract_last_json_block

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens, use_cache=True, do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    out = {}
    for key in ("original", "perturbed"):
        pixel_values = result[f"pixel_values_{key}"]
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=result["prompt_input_ids"],
                attention_mask=result["prompt_attention_mask"],
                pixel_values=pixel_values,
                **gen_kwargs,
            )
        decoded = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        try:
            data = extract_last_json_block(decoded)
            out[key] = (data.get("interestingness", "?"), data.get("explanation", ""))
        except Exception as e:
            logger.warning(f"[{key}] JSON parse failed: {e}")
            out[key] = ("?", "")
    return out
