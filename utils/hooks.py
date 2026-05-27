"""
Forward hook registration for activation extraction and activation injection.

Two distinct hook modes:
  - EXTRACT  : read-only capture of hidden states into a global store (used in I.1)
  - INJECT   : modify hidden states in-place during the forward pass (used in III.*)

Hook targeting follows the two-phase convention from the original codebase:
  phase="vision_once"  — fires during the prefill on vision encoder layers (CLS token)
  phase="rating_step"  — fires during the teacher-forced one-step on LLM layers (last token)

Migrated from: get_activations_base.py (register_hooks, select_cls, select_last_token,
               layer_embeddings global store)
"""

from __future__ import annotations
from typing import Callable

# import torch
# import torch.nn as nn


# ── Shared state ──────────────────────────────────────────────────────────────

class HookState:
    """
    Mutable container passed around instead of module-level globals.

    Attributes:
        embeddings: dict mapping layer key → captured tensor (extract mode)
        injection_vectors: dict mapping layer key → (vector, alpha) to add (inject mode)
        phase: current capture phase ("vision_once" | "rating_step" | None)
    """

    def __init__(self):
        self.embeddings: dict = {}
        self.injection_vectors: dict = {}
        self.phase: str | None = None
        self._handles: list = []

    def reset_embeddings(self):
        self.embeddings = {}

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ── Extract hooks ─────────────────────────────────────────────────────────────

def register_extract_hooks(model, state: HookState) -> HookState:
    """
    Register read-only forward hooks on all vision and LLM layers.

    Vision layers: capture CLS token (index 0) during phase="vision_once".
    LLM layers: capture last token during phase="rating_step".

    Args:
        model: loaded HuggingFace model
        state: HookState instance that receives captured embeddings

    Returns:
        state (same object, hooks registered in-place)
    """
    raise NotImplementedError


def _make_vision_hook(layer_key: str, state: HookState) -> Callable:
    """Return a hook function that captures the CLS token from a vision layer output."""
    raise NotImplementedError


def _make_llm_hook(layer_key: str, state: HookState) -> Callable:
    """Return a hook function that captures the last token from an LLM layer output."""
    raise NotImplementedError


# ── Inject hooks ──────────────────────────────────────────────────────────────

def register_inject_hooks(
    model,
    state: HookState,
    injection_vectors: dict[str, tuple],  # layer_key → (vector: np.ndarray, alpha: float)
) -> HookState:
    """
    Register hooks that add alpha * vector to the hidden state of the specified layers.

    Used by control/III1_vector_addition and control/III3_vector_subtraction.

    Args:
        model: loaded HuggingFace model
        state: HookState instance
        injection_vectors: mapping from layer key to (direction_vector, scalar_alpha).
                           alpha > 0 → addition; alpha < 0 → subtraction.

    Returns:
        state with injection hooks registered
    """
    raise NotImplementedError


def register_projection_removal_hooks(
    model,
    state: HookState,
    removal_vectors: dict[str, object],  # layer_key → direction_vector to project out
) -> HookState:
    """
    Register hooks that remove the projection of the hidden state onto the given vector.

    h' = h - (h · v̂) * v̂   where v̂ is the unit direction vector.

    Used by control/III3_vector_subtraction (stronger ablation variant).
    """
    raise NotImplementedError
