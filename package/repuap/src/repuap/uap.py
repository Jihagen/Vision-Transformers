"""
Stage IV.1 — Universal Adversarial Perturbation.

Per-image gradient matching (gradient_match_perturbation) and the universal
(single-delta-for-all-images) variant (gradient_match_universal): find a
bounded pixel-space delta such that h(image + delta) @ layer_key aligns with
a validated direction vector from discover.py.

Notes
-----
- Pixel range assumption: the processor's normalised pixel_values have a
  valid range of [-1, 1] (image_mean=image_std=0.5, as for Llama-4/SigLIP-style
  vision towers). Both training and evaluation project the perturbed image
  into that range after adding delta. If your processor uses different
  normalisation, adjust the clamp bounds in gradient_match_universal.
- device_map="auto": h from a mid/late layer can land on a different GPU than
  GPU-0; target tensors are moved to h.device lazily inside the loop.
- max_side is fixed across all training/eval images so every image resolves
  to the SAME pixel_values shape — required for one shared delta.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch

from .hooks import register_grad_extract_hook
from .model_io import preprocess_image
from .tasks import JudgmentTask, parse_response

logger = logging.getLogger(__name__)


class _EarlyStop(Exception):
    """Raised by the layer-(idx+1) pre-hook to abort the forward pass after layer idx fires."""


def _build_teacher_forced_inputs(model, processor, image, prompt: str, task: JudgmentTask):
    """Build (full_input_ids, full_attention_mask, pixel_values, prompt_input_ids,
    prompt_attention_mask): prompt + task.anchor() teacher-forced, matching the
    position a discover.py mean-difference/probe direction is defined at."""
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

    forced_ids = processor.tokenizer.encode(task.anchor(), add_special_tokens=False)
    forced = torch.tensor([forced_ids], device=target_device)
    full_input_ids = torch.cat([prompt_ids, forced], dim=1)
    full_attention_mask = torch.cat([prompt_mask, torch.ones_like(forced)], dim=1)

    return full_input_ids, full_attention_mask, pixel_values, prompt_ids, prompt_mask


def _hidden_at_layer(model, layer_key: str, full_input_ids, full_attention_mask,
                      pixel_values, requires_grad: bool, early_stop: bool = False) -> torch.Tensor:
    """Single forward pass; returns the label-token (last-token) hidden state
    at layer_key, shape (1, D). early_stop=True aborts the forward pass right
    after layer_key fires (avoids OOM on later-layer GPUs under device_map='auto')."""
    import re
    from .hooks import HookState

    container: dict = {}
    handle = register_grad_extract_hook(model, layer_key, container)

    stop_handle = None
    if early_stop:
        m = re.search(r"language_(\d+)", layer_key)
        if m:
            idx = int(m.group(1))
            try:
                next_layer = model.language_model.model.layers[idx + 1]

                def _stop_pre_hook(module, inp):
                    raise _EarlyStop()

                stop_handle = next_layer.register_forward_pre_hook(_stop_pre_hook)
            except (AttributeError, IndexError):
                pass

    try:
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            try:
                model(input_ids=full_input_ids, attention_mask=full_attention_mask,
                      pixel_values=pixel_values.to(model.dtype), use_cache=False)
            except _EarlyStop:
                pass
    finally:
        handle.remove()
        if stop_handle is not None:
            stop_handle.remove()

    if "h" not in container:
        raise RuntimeError(f"Hook never fired for {layer_key} -- layer was not reached.")
    return container["h"]


def compute_baseline_hidden(model, processor, image_path: str | Path, prompt: str,
                             layer_key: str, task: JudgmentTask) -> dict:
    """No-grad forward pass on the ORIGINAL image; returns h_baseline + reusable inputs."""
    image = preprocess_image(image_path, max_side=800)
    full_ids, full_mask, pixel_values, prompt_ids, prompt_mask = \
        _build_teacher_forced_inputs(model, processor, image, prompt, task)
    h = _hidden_at_layer(model, layer_key, full_ids, full_mask, pixel_values, requires_grad=False)
    return {
        "h_baseline": h.detach().float().cpu(),
        "pixel_values": pixel_values.detach().float(),
        "full_input_ids": full_ids, "full_attention_mask": full_mask,
        "prompt_input_ids": prompt_ids, "prompt_attention_mask": prompt_mask,
    }


def gradient_match_perturbation(
    model, processor, image_path: str | Path, prompt: str, layer_key: str,
    task: JudgmentTask, direction_vector: np.ndarray, alpha: float,
    epsilon: float = 0.05, lr: float = 0.01, n_steps: int = 20,
) -> dict:
    """
    Per-image PGD: find delta (L_inf <= epsilon) such that h(image+delta) @
    layer_key moves toward h_baseline + alpha * direction_vector_hat -- the
    same target an activation injection of alpha*direction_vector at
    layer_key would produce for this image.
    """
    base = compute_baseline_hidden(model, processor, image_path, prompt, layer_key, task)
    v = torch.tensor(direction_vector, dtype=torch.float32)
    v = v / v.norm()
    target_h_cpu = base["h_baseline"] + alpha * v

    pixel_values = base["pixel_values"]
    full_ids, full_mask = base["full_input_ids"], base["full_attention_mask"]
    delta = torch.zeros_like(pixel_values)

    model.requires_grad_(False)
    loss_history, cos_history = [], []
    for step in range(n_steps):
        delta.requires_grad_(True)
        perturbed = pixel_values + delta
        h = _hidden_at_layer(model, layer_key, full_ids, full_mask, perturbed, requires_grad=True).float()
        cos = torch.nn.functional.cosine_similarity(h, target_h_cpu.to(h.device), dim=-1).mean()
        loss = 1.0 - cos
        loss.backward()
        with torch.no_grad():
            delta = (delta - lr * delta.grad.sign()).clamp(-epsilon, epsilon)
        delta = delta.detach()
        loss_history.append(loss.item())
        cos_history.append(cos.item())

    return {
        "pixel_values_original": pixel_values.to(model.dtype),
        "pixel_values_perturbed": (pixel_values + delta).to(model.dtype),
        "prompt_input_ids": base["prompt_input_ids"],
        "prompt_attention_mask": base["prompt_attention_mask"],
        "loss_history": loss_history, "cos_history": cos_history,
    }


def evaluate_perturbation(model, processor, task: JudgmentTask, result: dict, max_new_tokens: int = 64) -> dict:
    """Full generation (no activation injection) on original vs perturbed
    pixel_values. Returns {"original": ParsedResponse, "perturbed": ParsedResponse}."""
    gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True, do_sample=False,
                       pad_token_id=processor.tokenizer.eos_token_id)
    out = {}
    for key in ("original", "perturbed"):
        pixel_values = result[f"pixel_values_{key}"]
        with torch.no_grad():
            gen_ids = model.generate(input_ids=result["prompt_input_ids"],
                                      attention_mask=result["prompt_attention_mask"],
                                      pixel_values=pixel_values, **gen_kwargs)
        decoded = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        out[key] = parse_response(decoded, task)
    return out


# ── Universal Adversarial Perturbation ───────────────────────────────────────

def _load_image_inputs(model, processor, image_path: str | Path, prompt: str,
                        task: JudgmentTask, max_side: int = 336):
    """max_side is fixed so every image resolves to a single processor tile ->
    consistent pixel_values shape, required for a delta shared across images."""
    image = preprocess_image(image_path, max_side=max_side)
    full_ids, full_mask, pixel_values, _, _ = _build_teacher_forced_inputs(
        model, processor, image, prompt, task
    )
    return full_ids, full_mask, pixel_values.float().cpu()


def gradient_match_universal(
    model, processor, task: JudgmentTask,
    train_image_paths: list, eval_image_paths: list, prompt: str, layer_key: str,
    direction_vector: np.ndarray, epsilon: float = 0.5, lr: float = 0.005,
    n_epochs: int = 5, max_side: int = 336, out_dir: Path | None = None,
) -> dict:
    """
    Find a SINGLE pixel-space delta such that adding it to any image
    maximises cos(h(image+delta) @ layer_key, v_hat).

    Loss: L = mean_i [ 1 - cos(h_i(image_i + delta), v_hat) ]
    Update (FGSM-style): delta <- clip(delta - lr * sign(dL/ddelta), -eps, eps)

    Returns dict with delta (np.ndarray), ref_shape, epsilon, cos_history,
    eval_results (cosine-alignment only -- run evaluate_perturbation
    separately for behavioural/label-level eval, which needs the full
    lm_head and so may not fit under a tight training-time memory cap).
    """
    import torch.nn.functional as F

    v = torch.tensor(direction_vector, dtype=torch.float32)
    v = v / v.norm()
    target_device = next(model.parameters()).device
    model.requires_grad_(False)

    _vision_gc, _llm_gc = False, False
    try:
        vm = model.vision_model.model
        if hasattr(vm, "gradient_checkpointing_enable"):
            vm.gradient_checkpointing_enable()
            _vision_gc = True
    except AttributeError:
        logger.warning("Could not enable gradient checkpointing on vision model -- may OOM")
    try:
        lm = model.language_model.model
        if hasattr(lm, "gradient_checkpointing_enable"):
            try:
                lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                lm.gradient_checkpointing_enable()
            _llm_gc = True
    except AttributeError:
        logger.warning("Could not enable gradient checkpointing on LLM -- may OOM")

    logger.info(f"Pre-loading {len(train_image_paths)} training images (max_side={max_side})...")
    train_data, ref_shape, skipped = [], None, 0
    for path in train_image_paths:
        try:
            fids, fmask, pv_cpu = _load_image_inputs(model, processor, path, prompt, task, max_side)
        except Exception as e:
            logger.warning(f"  skip {path}: {e}")
            skipped += 1
            continue
        if ref_shape is None:
            ref_shape = pv_cpu.shape
        elif pv_cpu.shape != ref_shape:
            logger.warning(f"  skip {path}: shape {pv_cpu.shape} != {ref_shape}")
            skipped += 1
            continue
        train_data.append({"full_ids": fids, "full_mask": fmask, "pv_cpu": pv_cpu})

    logger.info(f"  loaded {len(train_data)}/{len(train_image_paths)} images ({skipped} skipped). shape={ref_shape}")
    if not train_data:
        raise RuntimeError("No training images loaded -- check image paths / max_side.")

    delta = torch.zeros(ref_shape, dtype=torch.float32, device=target_device, requires_grad=True)
    cos_history = []

    for epoch in range(n_epochs):
        if delta.grad is not None:
            delta.grad.zero_()
        total_cos, n = 0.0, len(train_data)
        for img_data in train_data:
            pv_i = img_data["pv_cpu"].to(target_device)
            perturbed = (pv_i + delta).clamp(-1.0, 1.0)
            h = _hidden_at_layer(model, layer_key, img_data["full_ids"], img_data["full_mask"],
                                  perturbed, requires_grad=True, early_stop=True).float()
            cos = F.cosine_similarity(h, v.to(h.device), dim=-1).mean()
            loss = (1.0 - cos) / n
            loss.backward()
            total_cos += cos.item()

        mean_cos = total_cos / n
        cos_history.append(mean_cos)
        with torch.no_grad():
            delta_new = (delta - lr * delta.grad.sign()).clamp(-epsilon, epsilon)
        delta = delta_new.detach().requires_grad_(True)
        logger.info(f"  [epoch {epoch+1}/{n_epochs}] mean cos={mean_cos:.4f}  |delta|_inf={delta.abs().max().item():.4f}")

        if out_dir:
            np.save(Path(out_dir) / f"delta_eps{epsilon:.2f}_epoch{epoch+1:02d}.npy", delta.detach().cpu().numpy())

    delta_np = delta.detach().cpu().numpy()

    if _vision_gc:
        try:
            model.vision_model.model.gradient_checkpointing_disable()
        except Exception:
            pass
    if _llm_gc:
        try:
            model.language_model.model.gradient_checkpointing_disable()
        except Exception:
            pass

    logger.info(f"Evaluating on {len(eval_image_paths)} held-out images (cosine only)...")
    eval_results = []
    for path in eval_image_paths:
        try:
            fids, fmask, pv_cpu = _load_image_inputs(model, processor, path, prompt, task, max_side)
        except Exception as e:
            logger.warning(f"  skip eval {path}: {e}")
            continue
        if pv_cpu.shape != ref_shape:
            continue
        pv = pv_cpu.to(target_device)
        pv_pert = (pv + delta.detach()).clamp(-1.0, 1.0)
        with torch.no_grad():
            h_orig = _hidden_at_layer(model, layer_key, fids, fmask, pv, requires_grad=False, early_stop=True).float()
            h_pert = _hidden_at_layer(model, layer_key, fids, fmask, pv_pert, requires_grad=False, early_stop=True).float()
        cos_orig = float(F.cosine_similarity(h_orig, v.to(h_orig.device), dim=-1).mean())
        cos_pert = float(F.cosine_similarity(h_pert, v.to(h_pert.device), dim=-1).mean())
        eval_results.append({"filename": Path(path).name, "cos_original": cos_orig,
                              "cos_perturbed": cos_pert, "cos_delta": cos_pert - cos_orig})

    return {"delta": delta_np, "ref_shape": ref_shape, "epsilon": epsilon,
            "cos_history": cos_history, "eval_results": eval_results}
