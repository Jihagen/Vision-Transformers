"""
Cross-model transfer of the saved UAPs: show a second vision-language model the
same pixels Llama 4 saw, clean and perturbed, and read its label.

The deltas live in Llama 4's processor-normalised pixel_values space (one
336x336 tile, mean = std = 0.5, valid range [-1, 1]). To hand them to another
model, the Llama 4 image processor is run on the image, the delta is added and
clamped exactly as in perturbation.build_model_inputs, and the result is mapped
back to an 8-bit RGB image. Clean and perturbed conditions both go through this
round trip, so the only difference between them is the delta. The second model
then applies its own processor to that 336x336 image.

The 8-bit round trip moves each value by at most half a grey level (0.004 in
the [-1, 1] units of the budgets), which is small against the smallest budget
(eps = 0.1). No model is trained or adapted here.
"""
from __future__ import annotations
import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def llama_view(llama_image_processor, image: Image.Image, delta_np: np.ndarray | None) -> Image.Image:
    """The 336x336 image Llama 4 receives for `image`, with `delta_np` added if given."""
    pixel_values = llama_image_processor(images=image, return_tensors="pt")["pixel_values"].float()
    if delta_np is not None:
        if tuple(pixel_values.shape) != tuple(delta_np.shape):
            raise ValueError(
                f"pixel_values shape {tuple(pixel_values.shape)} != delta shape {tuple(delta_np.shape)} "
                "(image did not resolve to a single processor tile at this max_side)"
            )
        pixel_values = (pixel_values + torch.from_numpy(delta_np)).clamp(-1.0, 1.0)
    rgb = ((pixel_values[0] + 1.0) * 127.5).round().clamp(0, 255).byte()
    return Image.fromarray(rgb.permute(1, 2, 0).numpy(), mode="RGB")


def make_generate_fn(llama_image_processor):
    """
    Build a generate function with the signature of perturbation.generate_response.
    Unlike that function it decodes only the newly generated tokens, so the parser
    can never pick up the example JSON block that is part of the prompt.
    """
    logged = {"done": False}

    def generate(model, processor, image, prompt, delta_np, delta_t_cache,
                 max_side: int = 336, max_new_tokens: int = 128) -> str:
        view = llama_view(llama_image_processor, image, delta_np)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=chat_input, images=view, return_tensors="pt").to(model.device)
        if not logged["done"]:
            shapes = {k: tuple(v.shape) for k, v in inputs.items() if k != "input_ids" and k != "attention_mask"}
            grid = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None
            logger.info(f"  second-model vision input for a {view.size} view: {shapes} grid={grid}")
            logged["done"] = True
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        return processor.tokenizer.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate
