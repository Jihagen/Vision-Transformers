"""
Logit-lens projection of a residual-stream direction.

For a unit direction v at some layer, projects v through the model's final
RMSNorm gain + LM head to get a (vocab_size,) vector of "logit deltas":
the change in each output token's logit that would result from adding v to
the residual stream immediately before the final norm + unembedding, holding
everything else fixed.

This is a ONE-STEP LINEAR APPROXIMATION. It ignores whatever transformation v
would undergo passing through the transformer layers between the injection
point and the output (e.g. for language_29_D5120, 19 of 48 layers remain).
Treat results as a hypothesis about injection effects — not a confirmed one.

CPU-only: loads just the final-norm and lm_head tensors directly from the
safetensors shards (~2GB), without loading the full model.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

_NORM_KEY = "language_model.model.norm.weight"
_HEAD_KEY = "language_model.lm_head.weight"


def load_unembedding(model_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the final RMSNorm weight (D,) and LM head weight (V, D) from the
    safetensors shards listed in model.safetensors.index.json.
    """
    from safetensors import safe_open

    model_path = Path(model_path)
    weight_map = json.load(open(model_path / "model.safetensors.index.json"))["weight_map"]

    with safe_open(model_path / weight_map[_NORM_KEY], framework="pt") as f:
        norm_w = f.get_tensor(_NORM_KEY).float()
    with safe_open(model_path / weight_map[_HEAD_KEY], framework="pt") as f:
        lm_head = f.get_tensor(_HEAD_KEY).float()

    return norm_w, lm_head


def project_to_vocab(
    vector: np.ndarray,
    norm_weight: torch.Tensor,
    lm_head: torch.Tensor,
) -> torch.Tensor:
    """
    Project a direction vector (will be unit-normalised) through the final
    RMSNorm gain + LM head. Returns a (vocab_size,) tensor of logit deltas.
    """
    v = torch.tensor(vector, dtype=torch.float32)
    v = v / v.norm()
    return lm_head @ (v * norm_weight)


def top_tokens(
    logit_deltas: torch.Tensor,
    tokenizer,
    k: int = 30,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Return (top_positive, top_negative): lists of (decoded_token, score),
    sorted by |score| descending within each direction.
    """
    top_pos = torch.topk(logit_deltas, k)
    top_neg = torch.topk(-logit_deltas, k)
    pos = [(tokenizer.decode([i.item()]), v.item()) for v, i in zip(top_pos.values, top_pos.indices)]
    neg = [(tokenizer.decode([i.item()]), -v.item()) for v, i in zip(top_neg.values, top_neg.indices)]
    return pos, neg


if __name__ == "__main__":
    from representation.III1_vector_control.control import load_averaged_gender_vector
    from transformers import AutoTokenizer

    MODEL_PATH = (
        "hpc_infrastructure/hf_cache/"
        "models--meta-llama--Llama-4-Scout-17B-16E-Instruct/local-repo"
    )
    LAYER = "language_29_D5120"

    print("Loading gender vector...")
    v = load_averaged_gender_vector("results/representation_discovery/base/md_vectors", layer_key=LAYER)

    print("Loading final norm + lm_head (~2GB, CPU)...")
    norm_w, lm_head = load_unembedding(MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    deltas = project_to_vocab(v, norm_w, lm_head)
    pos, neg = top_tokens(deltas, tok, k=20)

    print(f"\n=== {LAYER} gender vector (female - male): top +v tokens ===")
    for tok_str, score in pos:
        print(f"{score:+8.3f}  {tok_str!r}")
    print(f"\n=== {LAYER} gender vector: top -v tokens ===")
    for tok_str, score in neg:
        print(f"{score:+8.3f}  {tok_str!r}")
