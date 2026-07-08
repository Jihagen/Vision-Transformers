"""
IV.1 Gradient-matching image perturbation — per-image and universal.

Two modes
---------
Per-image PGD  (gradient_match_perturbation)
    For a single image, find delta such that h(image+delta) ≈ h_baseline +
    alpha * v_interest — reproducing the activation-injection target via pixel
    space alone.  Useful as a sanity check: "can pixel perturbation reach the
    same latent state injection produces?"

Universal Adversarial Perturbation  (gradient_match_universal)
    Find a SINGLE delta that, when added to ANY image, maximises alignment of
    h(image+delta) with v_interest at the target layer.  Direction-only loss
    (no per-image baseline): loss = 1 - cos(h(image+delta), v_hat).
    Update: delta -= lr * sign(mean_grad_over_images), clip to L_inf ball.
    This is the core "make images very interesting" experiment: one delta,
    swept over epsilon ∈ {0.1, 0.5, 1.0, 2.0}, trained on 400 images,
    evaluated on 100 held-out images.

Notes
-----
- pixel_values units are processor-normalised (roughly [-2, 2] for ImageNet
  stats), so epsilon=0.05 ≈ imperceptible, epsilon=2.0 ≈ visible pattern.
- device_map="auto": h from layer 29 lands on a different GPU than GPU-0.
  target_h / v are always cast to h.device inside the loop (cheap: 5120-d).
- Images are loaded with max_side=560 in UAP mode to ensure a single
  processor tile and thus a consistent pixel_values shape across all images.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class _EarlyStop(Exception):
    """Raised by the layer-(idx+1) pre-hook to abort the forward pass after layer idx fires."""


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
                      pixel_values, requires_grad: bool,
                      early_stop: bool = False) -> torch.Tensor:
    """
    Single forward pass; returns the rating-token (last-token) hidden state
    at layer_key, shape (1, D).

    early_stop=True registers a forward pre-hook on the layer immediately
    after layer_key that raises _EarlyStop, aborting the forward pass once
    the target layer has fired.  Layers beyond the target never execute,
    which avoids OOM on the GPUs that hold those layers under device_map="auto".
    The gradient graph for layers 0..idx is fully built before the exception
    is raised, so loss.backward() works normally.
    """
    import re
    from utils.hooks import register_grad_extract_hook

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
                pass  # no next layer (idx is last) — forward completes normally

    try:
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            try:
                model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    pixel_values=pixel_values.to(model.dtype),
                    use_cache=False,
                )
            except _EarlyStop:
                pass  # intentional: layers idx+1 .. N never executed
    finally:
        handle.remove()
        if stop_handle is not None:
            stop_handle.remove()

    if "h" not in container:
        raise RuntimeError(
            f"Hook never fired for {layer_key} — layer was not reached. "
            "Check that layer_key is valid and the forward pass reaches it."
        )
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
    # Keep target_h on CPU — h comes from a hook deep in the network and
    # may land on a different GPU under device_map="auto". We move target_h
    # to h's device lazily inside the loop (tiny 5120-d vector, negligible).
    target_h_cpu = base["h_baseline"] + alpha * v  # float32, cpu

    pixel_values = base["pixel_values"]
    full_ids, full_mask = base["full_input_ids"], base["full_attention_mask"]
    delta = torch.zeros_like(pixel_values)

    model.requires_grad_(False)
    loss_history, cos_history = [], []
    for step in range(n_steps):
        delta.requires_grad_(True)
        perturbed = pixel_values + delta
        h = _hidden_at_layer(model, layer_key, full_ids, full_mask, perturbed, requires_grad=True)
        h = h.float()
        cos = torch.nn.functional.cosine_similarity(h, target_h_cpu.to(h.device), dim=-1).mean()
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


# ── Universal Adversarial Perturbation ───────────────────────────────────────

def _load_image_inputs(model, processor, image_path: str | Path, prompt: str,
                       max_side: int = 560):
    """
    Pre-load teacher-forced token sequences and pixel_values for one image.
    max_side=560 forces a single processor tile → consistent pixel_values shape
    across all images, which is required for the shared UAP delta.
    Returns (full_input_ids, full_attention_mask, pixel_values_float32_cpu).
    """
    from utils.image_utils import preprocess_image
    image = preprocess_image(image_path, max_side=max_side)
    full_ids, full_mask, pixel_values, _, _ = _build_teacher_forced_inputs(
        model, processor, image, prompt
    )
    return full_ids, full_mask, pixel_values.float().cpu()


def gradient_match_universal(
    model,
    processor,
    train_image_paths: list,
    eval_image_paths: list,
    prompt: str,
    layer_key: str,
    direction_vector: np.ndarray,
    epsilon: float = 0.5,
    lr: float = 0.005,
    n_epochs: int = 5,
    max_side: int = 336,
    out_dir: Path | None = None,
) -> dict:
    """
    Universal Adversarial Perturbation (UAP) for interestingness.

    Finds a SINGLE pixel-space delta such that adding it to any image
    maximises cos(h(image+delta) @ layer_key, v_interest).

    Loss (direction-only, no per-image baseline):
        L = mean_i [ 1 - cos(h_i(image_i + delta), v_hat) ]

    Update rule (FGSM-style, standard for UAP):
        delta <- clip( delta - lr * sign(∂L/∂delta), -epsilon, epsilon )

    Gradient accumulation: backward() is called per image; delta.grad
    accumulates across all images in an epoch, then a single update is made.

    Args:
        train_image_paths: images used to optimise delta.
        eval_image_paths:  held-out images for behavioural evaluation.
        epsilon:           L_inf bound in processor-normalised pixel_values units.
                           0.05≈imperceptible, 0.5≈slight pattern, 2.0≈visible.
        lr:                FGSM step size (default 0.005).
        n_epochs:          passes over train_image_paths.
        max_side:          resize images to at most this size before processing,
                           ensuring a single processor tile and consistent shape.

    Returns dict with:
        delta:             (shape) float32 numpy array — the universal perturbation.
        cos_history:       mean cos(h+delta, v) per epoch on training images.
        eval_results:      list of per-image dicts on eval set (rating, cos, etc.)
    """
    import torch.nn.functional as F
    from utils.prompt_builder import extract_last_json_block

    v = torch.tensor(direction_vector, dtype=torch.float32)
    v = v / v.norm()

    target_device = next(model.parameters()).device

    # Freeze model parameters: we only need gradient w.r.t. delta, not model weights.
    # Without this, loss.backward() allocates ~2.5 GiB gradient buffers for the
    # shared-expert weight matrices on each GPU, causing OOM even with GC enabled.
    model.requires_grad_(False)

    # ── Pre-load training images ──────────────────────────────────────────────
    # Enable gradient checkpointing on both vision encoder and LLM.
    # GC recomputes per-layer activations during backward instead of storing them.
    # This is safe with early-stop: HF's per-layer GC wraps each decoder layer
    # individually via checkpoint(layer.__call__, ...).  When backward recomputes
    # layer 29 it only re-runs layer29.__call__ — layer 30 is never called during
    # recompute, so the _EarlyStop pre-hook on layer 30 never triggers.
    _vision_gc_enabled = False
    try:
        vm = model.vision_model.model  # Llama4VisionModel
        if hasattr(vm, "gradient_checkpointing_enable"):
            vm.gradient_checkpointing_enable()
            _vision_gc_enabled = True
            logger.info("Gradient checkpointing enabled on vision_model.model")
    except AttributeError:
        logger.warning("Could not enable gradient checkpointing on vision model — may OOM on GPU 0")

    _llm_gc_enabled = False
    try:
        lm = model.language_model.model  # Llama4TextModel (MoE, GPUs 1-5)
        if hasattr(lm, "gradient_checkpointing_enable"):
            try:
                lm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                lm.gradient_checkpointing_enable()
            _llm_gc_enabled = True
            logger.info("Gradient checkpointing enabled on language_model.model")
    except AttributeError:
        logger.warning("Could not enable gradient checkpointing on LLM — may OOM on GPU 3 backward")

    logger.info(f"Pre-loading {len(train_image_paths)} training images (max_side={max_side})…")
    train_data, ref_shape = [], None
    skipped = 0
    for path in train_image_paths:
        try:
            fids, fmask, pv_cpu = _load_image_inputs(model, processor, path, prompt, max_side)
        except Exception as e:
            logger.warning(f"  skip {path}: {e}")
            skipped += 1
            continue
        if ref_shape is None:
            ref_shape = pv_cpu.shape
            logger.info(f"  pixel_values shape: {ref_shape}")
        elif pv_cpu.shape != ref_shape:
            logger.warning(f"  skip {path}: shape {pv_cpu.shape} != {ref_shape}")
            skipped += 1
            continue
        train_data.append({"full_ids": fids, "full_mask": fmask, "pv_cpu": pv_cpu})

    logger.info(f"  loaded {len(train_data)} / {len(train_image_paths)} images "
                f"({skipped} skipped).  shape={ref_shape}")
    if not train_data:
        raise RuntimeError("No training images loaded — check image paths / max_side.")

    # ── Initialise universal delta ────────────────────────────────────────────
    delta = torch.zeros(ref_shape, dtype=torch.float32, device=target_device,
                        requires_grad=True)

    cos_history = []

    # ── Optimisation loop ─────────────────────────────────────────────────────
    for epoch in range(n_epochs):
        if delta.grad is not None:
            delta.grad.zero_()

        total_cos = 0.0
        n = len(train_data)

        for img_data in train_data:
            pv_i = img_data["pv_cpu"].to(target_device)        # float32, no grad
            fids  = img_data["full_ids"]
            fmask = img_data["full_mask"]

            perturbed = pv_i + delta                           # grad flows through delta
            h = _hidden_at_layer(model, layer_key, fids, fmask, perturbed,
                                 requires_grad=True, early_stop=True)
            h_f = h.float()
            cos = F.cosine_similarity(h_f, v.to(h_f.device), dim=-1).mean()
            loss = (1.0 - cos) / n                             # normalise before accumulating
            loss.backward()
            total_cos += cos.item()

        mean_cos = total_cos / n
        cos_history.append(mean_cos)

        with torch.no_grad():
            delta_new = (delta - lr * delta.grad.sign()).clamp(-epsilon, epsilon)
        delta = delta_new.detach().requires_grad_(True)

        logger.info(f"  [epoch {epoch+1}/{n_epochs}] mean cos={mean_cos:.4f}  "
                    f"|delta|_inf={delta.abs().max().item():.4f}")

        if out_dir:
            np.save(Path(out_dir) / f"delta_eps{epsilon:.2f}_epoch{epoch+1:02d}.npy",
                    delta.detach().cpu().numpy())

    # Final delta (save regardless of out_dir)
    delta_np = delta.detach().cpu().numpy()

    # Disable GC before eval (no backward needed)
    if _vision_gc_enabled:
        try:
            model.vision_model.model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled on vision_model.model")
        except Exception:
            pass
    if _llm_gc_enabled:
        try:
            model.language_model.model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled on language_model.model")
        except Exception:
            pass

    # ── Evaluate on held-out images (cosine alignment only) ──────────────────
    # Behavioural eval (model.generate) is skipped here: lm_head sits beyond
    # layer 47 which may be disk-offloaded under the tight 6-GPU memory config.
    # Loading it would OOM GPU 0.  Run a separate rating script post-hoc once
    # the delta is saved.
    logger.info(f"Evaluating on {len(eval_image_paths)} held-out images (cosine only)…")
    eval_results = []
    for path in eval_image_paths:
        try:
            fids, fmask, pv_cpu = _load_image_inputs(model, processor, path, prompt, max_side)
        except Exception as e:
            logger.warning(f"  skip eval {path}: {e}")
            continue
        if pv_cpu.shape != ref_shape:
            logger.warning(f"  skip eval {path}: shape mismatch")
            continue

        pv = pv_cpu.to(target_device)
        pv_pert = (pv + delta.detach()).clamp(-10, 10)

        with torch.no_grad():
            h_orig = _hidden_at_layer(model, layer_key, fids, fmask, pv,
                                      requires_grad=False, early_stop=True).float()
            h_pert = _hidden_at_layer(model, layer_key, fids, fmask, pv_pert,
                                      requires_grad=False, early_stop=True).float()
        cos_orig = float(F.cosine_similarity(h_orig, v.to(h_orig.device), dim=-1).mean())
        cos_pert = float(F.cosine_similarity(h_pert, v.to(h_pert.device), dim=-1).mean())

        eval_results.append({
            "filename": Path(path).name,
            "cos_original": cos_orig,
            "cos_perturbed": cos_pert,
            "cos_delta": cos_pert - cos_orig,
        })
        logger.info(f"  {Path(path).name}: cos {cos_orig:.4f}→{cos_pert:.4f}  "
                    f"Δcos={cos_pert - cos_orig:+.4f}")

    return {
        "delta": delta_np,
        "ref_shape": ref_shape,
        "epsilon": epsilon,
        "cos_history": cos_history,
        "eval_results": eval_results,
    }
