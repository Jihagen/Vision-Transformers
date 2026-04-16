#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Non-intrusive execution order tracer for LLaMA-ViT (Scout) style models.

Implements:
1) Module tree snapshot (candidate stacks: vision encoder, projector, LLM decoder)
2) Non-mutating entry/exit hooks with only metadata logging (no tensor copying)
3) Prefill-only forward to reveal stage order: ViT -> projector -> LLM (prefill)
4) One-token generate to reveal the autoregressive decoder loop: LLM layers repeat

Usage (example):
  python trace_order.py \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --image ./demo.jpg \
    --prompt "Describe this image briefly." \
    --local-files-only

Notes:
- Hooks only read .shape metadata. No .cpu(), no dtype changes, no copies.
- Run under eval() and no_grad() for determinism and zero gradient overhead.
- For clarity, we hook at "layer granularity" (vision-layer-i, projector, language-layer-i).
"""

import os
import re
import sys
import argparse
import logging
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Dict

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trace")

# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
DEFAULT_IMAGE = "/anvme/workspace/iwi5268h-vision_transformers/data/imagesDemographics/1000.jpg"
DEFAULT_PROMPT = "describe the image and its interestingness."

# ─── CLI ────────────────────────────────────────────────────────────────────────
def build_argparser():
    p = argparse.ArgumentParser(description="Non-intrusive execution order tracer")
    p.add_argument("--model", "--model_path", default=DEFAULT_MODEL,
                   help="HF repo id or local path (default: %(default)s)")
    p.add_argument("--image", default=DEFAULT_IMAGE,
                   help="Path to an RGB image (default: %(default)s)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Short prompt (default: %(default)s)")
    p.add_argument("--max_image_side", type=int, default=1024, help="Max side for image resize")
    p.add_argument("--device-map", default="auto", help="transformers device_map (default: auto)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"], help="Torch dtype")
    p.add_argument("--local-files-only", action="store_true", help="Force local files only")
    p.add_argument("--verbose", action="store_true", help="More logs")
    return p

# ─── Helpers ────────────────────────────────────────────────────────────────────
def str_dtype(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]

def load_image(path: str, max_side: int = 1024) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        r = max_side / float(m)
        img = img.resize((int(w * r), int(h * r)), Image.Resampling.LANCZOS)
    return img

def first_tensor_shape(x: Any) -> Optional[Tuple[int, ...]]:
    """Return a representative tensor shape if output is tensor/tuple/list; else None."""
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (tuple, list)):
        for y in x:
            if isinstance(y, torch.Tensor):
                return tuple(y.shape)
    return None

@dataclass
class Event:
    idx: int
    phase: str               # 'prefill' or 'decode'
    kind: str                # 'enter' or 'exit'
    path: str
    cls: str
    shape: Optional[Tuple[int, ...]]

# ─── Hook selection ─────────────────────────────────────────────────────────────
# We want "layer granularity" + projector block(s).
# We'll include modules whose *name path* suggests:
# - vision encoder layers
# - projector / projection / multi_modal
# - language model decoder layers
CANDIDATE_PATTERNS = [
    re.compile(r"vision.*layers\.\d+"),                 # e.g., vision_model.model.layers.0
    re.compile(r"vision_model\..*encoder\.layers\.\d+"),
    re.compile(r"(project(or|ion)|multi_modal|mm_?proj)", re.IGNORECASE),
    re.compile(r"language_model\.model\.layers\.\d+"),  # e.g., language_model.model.layers.0
]

def is_candidate_module(name: str, module: torch.nn.Module) -> bool:
    for pat in CANDIDATE_PATTERNS:
        if pat.search(name):
            return True
    return False

# ─── Order logger (non-mutating) ────────────────────────────────────────────────
class OrderLogger:
    def __init__(self):
        self.events: List[Event] = []
        self.counter = 0
        self._handles = []
        self._registered: List[Tuple[str, torch.nn.Module]] = []

    def _pre(self, phase: str, name: str, module: torch.nn.Module):
        def _fn(mod, inputs):
            self.counter += 1
            self.events.append(
                Event(self.counter, phase, "enter", name, mod.__class__.__name__, None)
            )
        return _fn

    def _post(self, phase: str, name: str, module: torch.nn.Module):
        def _fn(mod, inputs, output):
            self.counter += 1
            shp = first_tensor_shape(output)
            self.events.append(
                Event(self.counter, phase, "exit", name, mod.__class__.__name__, shp)
            )
        return _fn

    def register(self, model: torch.nn.Module, phase: str = "prefill"):
        """Attach hooks to candidate modules."""
        for name, module in model.named_modules():
            if is_candidate_module(name, module):
                self._registered.append((name, module))
                self._handles.append(module.register_forward_pre_hook(self._pre(phase, name, module)))
                self._handles.append(module.register_forward_hook(self._post(phase, name, module)))

        log.info(f"Registered hooks on {len(self._handles)//2} modules for phase={phase}")

    def clear_phase(self, phase: str):
        """Change future events' phase tag (for a new run) without re-registering hooks."""
        # Easiest: remove & re-register with new phase.
        self.remove()
        self.events.clear()
        self.counter = 0

    def reregister_for_phase(self, model: torch.nn.Module, phase: str):
        self.remove()
        self._handles.clear()
        for name, module in model.named_modules():
            if is_candidate_module(name, module):
                self._handles.append(module.register_forward_pre_hook(self._pre(phase, name, module)))
                self._handles.append(module.register_forward_hook(self._post(phase, name, module)))
        log.info(f"Re-registered hooks (phase={phase}) on {len(self._handles)//2} modules")

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def print_timeline(self, title: str):
        print("\n" + "="*88)
        print(f"{title}")
        print("="*88)
        for ev in self.events:
            shape_str = f" shape={list(ev.shape)}" if ev.shape is not None else ""
            print(f"[{ev.idx:04d}] {ev.phase:<6} {ev.kind.upper():5}  {ev.path}  ({ev.cls}){shape_str}")
        print("="*88 + "\n")

# ─── Module tree snapshot ───────────────────────────────────────────────────────
def snapshot_module_tree(model: torch.nn.Module) -> List[Tuple[str, str]]:
    items = []
    for name, module in model.named_modules():
        items.append((name, module.__class__.__name__))
    return items

def print_candidate_summary(tree: List[Tuple[str, str]]):
    print("\n" + "-"*88)
    print("Candidate module summary (vision layers, projector, language layers)")
    print("-"*88)
    kept = []
    for name, cls in tree:
        if is_candidate_module(name, None):
            kept.append((name, cls))
    # Try to group for readability
    def group_key(n):
        if "vision" in n:
            return "vision"
        if re.search(r"(project(or|ion)|multi_modal|mm_?proj)", n, re.IGNORECASE):
            return "projector"
        if "language_model.model.layers" in n:
            return "language"
        return "other"
    kept.sort(key=lambda x: (group_key(x[0]), x[0]))
    for name, cls in kept:
        print(f"{name:80s}  [{cls}]")
    print("-"*88 + "\n")

# ─── Prefill-only forward ───────────────────────────────────────────────────────
def run_prefill_forward(model, processor, image: Image.Image, prompt: str):
    # Build chat template (scout models usually expect the multimodal chat format)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=chat_input,
        images=image,
        return_tensors="pt",
        add_special_tokens=True,
    )
    # Whitelist to what the model expects
    whitelist = {"input_ids", "attention_mask", "pixel_values"}
    model_inputs = {k: v for k, v in inputs.items() if k in whitelist}

    # Move to the model's primary device (first param device)
    target_device = next(model.parameters()).device
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(target_device)

    with torch.no_grad():
        # This will execute: ViT -> Projector -> LLM (prefill)
        _ = model(**model_inputs)
    return True

# ─── One-token generate ─────────────────────────────────────────────────────────
def run_one_token_generate(model, processor, image: Image.Image, prompt: str):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat_input = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=chat_input,
        images=image,
        return_tensors="pt",
        add_special_tokens=True,
    )
    whitelist = {"input_ids", "attention_mask", "pixel_values"}
    model_inputs = {k: v for k, v in inputs.items() if k in whitelist}
    target_device = next(model.parameters()).device
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(target_device)

    gen_kwargs = dict(
        max_new_tokens=1,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    with torch.no_grad():
        _ = model.generate(**model_inputs, **gen_kwargs)
    return True

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = build_argparser().parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)

    dtype = str_dtype(args.dtype)
    log.info(f"Loading model: {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        offload_state_dict=True,
    )
    model.eval()
    # We want a clean trace; disable grad checkpointing if present.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )

    image = load_image(args.image, max_side=args.max_image_side)

    # 1) Module tree snapshot
    tree = snapshot_module_tree(model)
    print_candidate_summary(tree)

    # 2) Register non-mutating hooks for PREFILL
    tracer = OrderLogger()
    tracer.register(model, phase="prefill")

    # 3) Prefill-only forward
    log.info("Running PREFILL-ONLY forward...")
    run_prefill_forward(model, processor, image, args.prompt)
    tracer.print_timeline("PREFILL EXECUTION TIMELINE (expected: ViT -> projector -> LLM once)")

    # 4) One-token generate (decoder loop)
    log.info("Re-registering hooks for DECODE phase...")
    tracer.reregister_for_phase(model, phase="decode")
    log.info("Running ONE-TOKEN GENERATE...")
    run_one_token_generate(model, processor, image, args.prompt)
    tracer.print_timeline("DECODE EXECUTION TIMELINE (expected: LLM layers repeat once; no ViT/projector)")

    # Cleanup hooks
    tracer.remove()
    log.info("Done.")

if __name__ == "__main__":
    main()
