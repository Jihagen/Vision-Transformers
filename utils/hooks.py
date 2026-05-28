"""
Forward hook registration for activation extraction and activation injection.

Two distinct hook modes:
  EXTRACT  — read-only capture of hidden states into a HookState store (used in I.1)
  INJECT   — modify hidden states in-place during the forward pass (used in III.*)

Phase convention:
  "vision_once"  — fires during prefill on vision encoder layers (capture CLS token)
  "rating_step"  — fires during teacher-forced one-step on LLM layers (capture last token)
  "idle"         — hooks registered but not capturing / injecting

Migrated from: get_activations_base.py
"""

from __future__ import annotations
import logging
from typing import Callable, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ── Shared state ──────────────────────────────────────────────────────────────

class HookState:
    """
    Mutable container replacing module-level globals from the original codebase.

    Attributes:
        embeddings:        {layer_key: cpu tensor} captured during extraction
        injection_vectors: {layer_key: (vector, alpha)} for injection mode
        phase:             current capture gate ("vision_once"|"rating_step"|"idle")
    """

    def __init__(self):
        self.embeddings: dict = {}
        self.injection_vectors: dict = {}
        self.phase: str = "idle"
        self._handles: list = []

    def reset_embeddings(self):
        self.embeddings.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        logger.info("Removed all hooks")


# ── Token selectors ───────────────────────────────────────────────────────────

def select_cls(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0, :]


def select_last_token(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1, :]


# ── Hook factory ──────────────────────────────────────────────────────────────

def _make_extract_hook(
    name: str,
    state: HookState,
    selector: Optional[Callable],
    phase_needed: Optional[str],
    dtype: torch.dtype = torch.float16,
) -> Callable:
    def hook(module, _, output):
        if phase_needed and state.phase != phase_needed:
            return
        try:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if out is None or not hasattr(out, "detach"):
                return
            x = out.detach()
            if selector is not None:
                x = selector(x)
            state.embeddings[name] = x.to(dtype).cpu()
        except Exception as e:
            logger.warning(f"Hook failed on {name}: {e}")
    return hook


def _make_inject_hook(
    name: str,
    state: HookState,
    dtype: torch.dtype = torch.float16,
) -> Callable:
    """Return a hook that adds state.injection_vectors[name] * alpha to h."""
    def hook(module, _, output):
        if name not in state.injection_vectors:
            return
        vector, alpha = state.injection_vectors[name]
        try:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if out is None or not hasattr(out, "detach"):
                return
            v = torch.tensor(vector, dtype=out.dtype, device=out.device)
            modified = out + alpha * v
            if isinstance(output, (tuple, list)):
                return (modified,) + output[1:]
            return modified
        except Exception as e:
            logger.warning(f"Inject hook failed on {name}: {e}")
    return hook


def _make_projection_removal_hook(
    name: str,
    state: HookState,
) -> Callable:
    """Return a hook that removes the projection of h onto state.injection_vectors[name][0]."""
    def hook(module, _, output):
        if name not in state.injection_vectors:
            return
        vector, _ = state.injection_vectors[name]
        try:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if out is None or not hasattr(out, "detach"):
                return
            v = torch.tensor(vector, dtype=out.dtype, device=out.device)
            v_hat = v / (v.norm() + 1e-12)
            proj = (out * v_hat).sum(dim=-1, keepdim=True) * v_hat
            modified = out - proj
            if isinstance(output, (tuple, list)):
                return (modified,) + output[1:]
            return modified
        except Exception as e:
            logger.warning(f"Projection-removal hook failed on {name}: {e}")
    return hook


# ── Model structure inspection ────────────────────────────────────────────────

def _inspect_model_structure(model) -> tuple:
    """Return (llm_layers, vision_layers, vision_path_used)."""
    logger.info("Inspecting model structure...")
    llm_layers = None
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        if hasattr(model.language_model.model, "layers"):
            llm_layers = model.language_model.model.layers
            logger.info(f"Found {len(llm_layers)} language layers")

    vision_paths = [
        ("vision_model.model.layers",
         lambda: model.vision_model.model.layers
         if hasattr(model, "vision_model") and hasattr(model.vision_model, "model")
            and hasattr(model.vision_model.model, "layers") else None),
        ("vision_model.encoder.layers",
         lambda: model.vision_model.encoder.layers
         if hasattr(model, "vision_model") and hasattr(model.vision_model, "encoder")
            and hasattr(model.vision_model.encoder, "layers") else None),
        ("vision_model.vision_model.encoder.layers",
         lambda: model.vision_model.vision_model.encoder.layers
         if hasattr(model, "vision_model") and hasattr(model.vision_model, "vision_model")
            and hasattr(model.vision_model.vision_model, "encoder")
            and hasattr(model.vision_model.vision_model.encoder, "layers") else None),
    ]
    vision_layers = None
    vision_path_used = None
    for path_name, path_fn in vision_paths:
        try:
            layers = path_fn()
            if layers is not None:
                logger.info(f"Found {len(layers)} vision layers at {path_name}")
                vision_layers = layers
                vision_path_used = path_name
                break
        except Exception as e:
            logger.debug(f"Path {path_name} failed: {e}")

    if vision_layers is None and hasattr(model, "vision_model"):
        logger.warning("Could not find vision layers")

    return llm_layers, vision_layers, vision_path_used


# ── Public API ────────────────────────────────────────────────────────────────

def register_extract_hooks(model, state: HookState) -> HookState:
    """
    Register read-only forward hooks on all vision and LLM layers.

    Vision layers: capture CLS token (index 0) during phase="vision_once".
    LLM layers: capture last token during phase="rating_step".
    """
    llm_layers, vision_layers, vision_path = _inspect_model_structure(model)
    handles = []

    if llm_layers is not None:
        for i, layer in enumerate(llm_layers):
            h = layer.register_forward_hook(_make_extract_hook(
                name=f"llm_layer_{i}_rating_token",
                state=state,
                selector=select_last_token,
                phase_needed="rating_step",
            ))
            handles.append(h)
        logger.info(f"Registered {len(llm_layers)} LLM hooks (phase=rating_step)")

    if vision_layers is not None:
        for i, layer in enumerate(vision_layers):
            h = layer.register_forward_hook(_make_extract_hook(
                name=f"vision_layer_{i}_cls",
                state=state,
                selector=select_cls,
                phase_needed="vision_once",
            ))
            handles.append(h)
        logger.info(f"Registered {len(vision_layers)} vision hooks via {vision_path}")
    else:
        logger.warning("No vision layers found for hooking")

    state._handles.extend(handles)
    logger.info(f"Total hooks registered: {len(handles)}")
    return state


def register_inject_hooks(
    model,
    state: HookState,
    injection_vectors: dict[str, tuple],
) -> HookState:
    """
    Register hooks that add alpha * vector to the hidden state at specified layers.

    injection_vectors: {layer_key: (np.ndarray vector, float alpha)}
    alpha > 0 → addition; alpha < 0 → subtraction.
    """
    state.injection_vectors = dict(injection_vectors)
    llm_layers, vision_layers, _ = _inspect_model_structure(model)

    all_layers = {}
    if llm_layers is not None:
        for i, layer in enumerate(llm_layers):
            all_layers[f"llm_layer_{i}_rating_token"] = layer
    if vision_layers is not None:
        for i, layer in enumerate(vision_layers):
            all_layers[f"vision_layer_{i}_cls"] = layer

    handles = []
    for name, layer in all_layers.items():
        if name in injection_vectors:
            h = layer.register_forward_hook(_make_inject_hook(name, state))
            handles.append(h)

    state._handles.extend(handles)
    logger.info(f"Registered {len(handles)} injection hooks")
    return state


def register_projection_removal_hooks(
    model,
    state: HookState,
    removal_vectors: dict[str, np.ndarray],
) -> HookState:
    """
    Register hooks that remove the projection of h onto each direction vector.

    h' = h - (h · v̂) * v̂
    """
    state.injection_vectors = {k: (v, 1.0) for k, v in removal_vectors.items()}
    llm_layers, vision_layers, _ = _inspect_model_structure(model)

    all_layers = {}
    if llm_layers is not None:
        for i, layer in enumerate(llm_layers):
            all_layers[f"llm_layer_{i}_rating_token"] = layer
    if vision_layers is not None:
        for i, layer in enumerate(vision_layers):
            all_layers[f"vision_layer_{i}_cls"] = layer

    handles = []
    for name, layer in all_layers.items():
        if name in removal_vectors:
            h = layer.register_forward_hook(_make_projection_removal_hook(name, state))
            handles.append(h)

    state._handles.extend(handles)
    logger.info(f"Registered {len(handles)} projection-removal hooks")
    return state
