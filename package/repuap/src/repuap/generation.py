"""Shared single-pass generation helpers used by causal.py and generalise.py."""

from __future__ import annotations
import torch


def build_inputs(processor, image, prompt: str, device) -> dict:
    """Tokenise one image+prompt pair. pixel_values stays float32 and un-cast
    to model dtype so callers can perturb it before the final `.to(model.dtype)`."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=chat_input, images=image, return_tensors="pt",
        add_special_tokens=True, truncation=False, max_length=512,
    )
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "pixel_values": inputs["pixel_values"].float(),
    }


def generate_text(model, processor, input_ids, attention_mask, pixel_values, max_new_tokens: int = 64) -> str:
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values.to(model.dtype),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    return processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
