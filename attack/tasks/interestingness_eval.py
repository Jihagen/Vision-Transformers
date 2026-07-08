"""
Behavioral evaluation: interestingness rating under UAP.

Applies a saved delta.npy (or no delta for the clean baseline) to images
and collects model.generate() ratings using the EXACT same prompt and label
set used throughout the original study (build_blank_prompt(), VALID_LABELS).

This is the primary behavioral success metric for comparing the interest
attack vs the emotion attack: does pushing the representation toward a
direction actually shift what the model SAYS about the image?
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

LABEL_TO_SCORE = {
    "Not Interesting": 1,
    "Slightly Interesting": 2,
    "Moderately Interesting": 3,
    "Very Interesting": 4,
    "Extremely Interesting": 5,
}


def rate_images(
    model,
    processor,
    image_paths: list[str | Path],
    prompt: str,
    delta_np: np.ndarray | None = None,
    max_side: int = 336,
    max_new_tokens: int = 64,
) -> list[dict]:
    """
    Rate each image with model.generate() using the original interestingness prompt.

    delta_np: float32 numpy array matching pixel_values shape from the processor
              at max_side=336 (the same max_side used during UAP training).
              Pass None for the clean baseline (no perturbation applied).

    Returns list of dicts per image:
        filename, label (str), score (int 1-5), parse_ok (bool)

    Images whose pixel_values shape doesn't match delta_np.shape are skipped
    with a warning (same as during UAP training).
    """
    from utils.image_utils import preprocess_image
    from utils.prompt_builder import parse_model_response

    delta_t = None
    ref_shape = None
    if delta_np is not None:
        ref_shape = tuple(delta_np.shape)

    results = []
    n = len(image_paths)

    for i, path in enumerate(image_paths):
        path = Path(path)
        try:
            image = preprocess_image(path, max_side=max_side)
        except Exception as e:
            logger.warning(f"  [{i+1}/{n}] skip {path.name}: load failed — {e}")
            continue

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
        input_ids = inputs["input_ids"].to(target_device)
        attention_mask = inputs["attention_mask"].to(target_device)
        pixel_values = inputs["pixel_values"].float()

        if delta_np is not None:
            if tuple(pixel_values.shape) != ref_shape:
                logger.warning(
                    f"  [{i+1}/{n}] skip {path.name}: "
                    f"pixel_values shape {pixel_values.shape} != delta shape {ref_shape}"
                )
                continue
            if delta_t is None:
                delta_t = torch.tensor(delta_np, device=target_device)
            pixel_values = (pixel_values.to(target_device) + delta_t).clamp(-10, 10)
        else:
            pixel_values = pixel_values.to(target_device)

        pixel_values = pixel_values.to(model.dtype)

        try:
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
            decoded = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"  [{i+1}/{n}] {path.name}: generate failed — {e}")
            results.append({"filename": path.name, "label": None, "score": None, "parse_ok": False})
            continue

        try:
            data = parse_model_response(decoded)
            label = data["interestingness"]
            score = LABEL_TO_SCORE[label]
            parse_ok = True
        except Exception:
            label, score, parse_ok = None, None, False
            logger.warning(f"  [{i+1}/{n}] {path.name}: parse failed — raw: {decoded[:120]!r}")

        results.append({"filename": path.name, "label": label, "score": score, "parse_ok": parse_ok})
        logger.info(
            f"  [{i+1}/{n}] {path.name}: {label!r} (score={score})"
            + (f"  [PARSE FAIL]" if not parse_ok else "")
        )

    return results
