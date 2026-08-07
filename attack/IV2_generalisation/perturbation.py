"""
Shared UAP-application + model-inference layer for Generalisation Level 1.

Reuses EXACTLY the perturbation mechanics already validated in
attack/tasks/interestingness_eval.py and attack/tasks/relevance_eval.py:
processor at max_side=336, delta added in processor-normalised pixel_values
space, complete adversarial image clamped to Llama 4's valid [-1, 1] range,
cast to model dtype immediately before generate(). Nothing here reimplements
that logic independently — it is copied verbatim from those two files.

Do not train or scale UAPs here. This module only loads already-saved
delta.npy files and applies them.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Where the currently-active (post-projection) UAP deltas live. "projected"
# is the current/superseding UAP set (see results/attack_eval_projected/) —
# this is what the two already-run generalisation-precursor evals
# (interestingness, relevance) used, so Level 1 must match.
DEFAULT_UAP_ROOTS: dict[str, str] = {
    "interest": "results/universal_perturbation_projected/interest",
    "excited_vs_angry": "results/universal_perturbation_projected/excited_vs_angry",
    "workload": "results/universal_perturbation_projected/workload",
}
DEFAULT_EPSILONS = [0.1, 0.5, 1.0, 2.0]


def discover_uap_conditions(
    roots: dict[str, str] | None = None,
    epsilons: list[float] | None = None,
) -> list[tuple[str, float, Path]]:
    """
    Return [(attack_name, epsilon, delta_path), ...] for every delta.npy that
    actually exists on disk. Attacks/epsilons whose training hasn't finished
    yet (e.g. workload/stress, per the task brief) are silently omitted —
    call this again later once the files land instead of hard-coding a list.
    """
    roots = roots or DEFAULT_UAP_ROOTS
    epsilons = epsilons or DEFAULT_EPSILONS
    found = []
    for attack, root in roots.items():
        for eps in epsilons:
            delta_path = Path(root) / f"eps{eps:.2f}" / "delta.npy"
            if delta_path.exists():
                found.append((attack, eps, delta_path))
            else:
                logger.info(f"  [not yet available] {attack} eps={eps}: {delta_path}")
    return found


def load_delta(delta_path: str | Path) -> np.ndarray:
    return np.load(delta_path).astype(np.float32)


def build_model_inputs(processor, image, prompt: str, device, delta_np: np.ndarray | None, delta_t_cache: dict):
    """
    Tokenise prompt+image and (optionally) add a UAP delta to pixel_values.

    delta_t_cache: caller-owned dict used to cache the delta tensor on-device
    across calls (avoids re-uploading the same ~1.3MB delta every image).
    Keyed by id(delta_np).
    """
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=chat_input, images=image, return_tensors="pt",
        add_special_tokens=True, truncation=False, max_length=512,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].float()

    if delta_np is not None:
        ref_shape = tuple(delta_np.shape)
        if tuple(pixel_values.shape) != ref_shape:
            raise ValueError(
                f"pixel_values shape {tuple(pixel_values.shape)} != delta shape {ref_shape} "
                "(image did not resolve to a single processor tile at this max_side)"
            )
        key = id(delta_np)
        if key not in delta_t_cache:
            delta_t_cache[key] = torch.tensor(delta_np, device=device)
        pixel_values = (pixel_values.to(device) + delta_t_cache[key]).clamp(-1.0, 1.0)
    else:
        pixel_values = pixel_values.to(device)

    return input_ids, attention_mask, pixel_values


def generate_response(
    model,
    processor,
    image,
    prompt: str,
    delta_np: np.ndarray | None,
    delta_t_cache: dict,
    max_side: int = 336,
    max_new_tokens: int = 128,
) -> str:
    """
    Run one model.generate() call on `image` (already preprocess_image'd by
    the caller at `max_side`) with `prompt`, optionally perturbed by
    `delta_np`. Returns the full decoded text (prompt + generation), matching
    the existing convention where the JSON parser pulls the LAST JSON block
    out of the decode.
    """
    device = next(model.parameters()).device
    input_ids, attention_mask, pixel_values = build_model_inputs(
        processor, image, prompt, device, delta_np, delta_t_cache
    )
    pixel_values = pixel_values.to(model.dtype)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    return processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
