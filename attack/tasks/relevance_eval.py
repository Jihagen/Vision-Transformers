"""
Behavioral evaluation: binary relevance rating under UAP.

Prompt: "Is this image relevant to you?" — binary yes/no.
Removes the "interesting" token entirely; tests whether the attack is
token-level (exploiting the word "interesting") or semantic (shifting
the model's engagement state regardless of label vocabulary).

Expected pattern if attacks are semantic:
    interest attack  → shifts relevance toward yes (interest-adjacent)
    emotion attack   → also shifts toward yes, possibly by similar margin
    neutral task     → neither attack shifts meaningfully
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

VALID_RELEVANT = {"yes", "no"}
RELEVANT_TO_SCORE = {"yes": 1, "no": 0}

_RELEVANCE_PROMPT = (
    "You are rating images.\n\n"
    "Look at this image and answer: Is this image relevant to you?\n\n"
    "Return ONLY valid JSON.\n"
    "Do not use markdown.\n"
    "Do not use code fences.\n"
    "Do not add any text before or after the JSON.\n\n"
    "Keys:\n"
    '- "relevant": "yes" or "no"\n'
    '- "explanation": exactly 1 short sentence'
)


def build_relevance_prompt() -> str:
    return _RELEVANCE_PROMPT


def rate_relevance(
    model,
    processor,
    image_paths: list[str | Path],
    delta_np: np.ndarray | None = None,
    max_side: int = 336,
    max_new_tokens: int = 64,
) -> list[dict]:
    """
    Rate each image with model.generate() using the binary relevance prompt.

    delta_np: float32 numpy array matching pixel_values shape (max_side=336).
              Pass None for the clean baseline.

    Returns list of dicts per image:
        filename, relevant ("yes"/"no"/None), score (1/0/None), parse_ok
    """
    from utils.image_utils import preprocess_image
    from utils.prompt_builder import extract_last_json_block

    prompt = build_relevance_prompt()
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
            text=chat_input, images=image, return_tensors="pt",
            add_special_tokens=True, truncation=False, max_length=512,
        )

        target_device = next(model.parameters()).device
        input_ids     = inputs["input_ids"].to(target_device)
        attention_mask = inputs["attention_mask"].to(target_device)
        pixel_values  = inputs["pixel_values"].float()

        if delta_np is not None:
            if tuple(pixel_values.shape) != ref_shape:
                logger.warning(f"  [{i+1}/{n}] skip {path.name}: shape mismatch")
                continue
            if delta_t is None:
                delta_t = torch.tensor(delta_np, device=target_device)
            pixel_values = (pixel_values.to(target_device) + delta_t).clamp(-1.0, 1.0)
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
            results.append({"filename": path.name, "relevant": None, "score": None, "parse_ok": False})
            continue

        try:
            data = extract_last_json_block(decoded)
            relevant = str(data.get("relevant", "")).strip().lower()
            if relevant not in VALID_RELEVANT:
                raise ValueError(f"unexpected value: {relevant!r}")
            score = RELEVANT_TO_SCORE[relevant]
            parse_ok = True
            explanation = str(data.get("explanation", ""))
        except Exception:
            relevant, score, parse_ok, explanation = None, None, False, None
            logger.warning(f"  [{i+1}/{n}] {path.name}: parse failed — {decoded[:120]!r}")

        results.append({
            "filename": path.name, "relevant": relevant, "score": score, "parse_ok": parse_ok,
            "explanation": explanation, "raw_response": decoded,
        })
        logger.info(f"  [{i+1}/{n}] {path.name}: relevant={relevant!r}  explanation={explanation!r}")

    return results
