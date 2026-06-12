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
import re
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


_WORD_RE = re.compile(r"^[A-Za-z]+$")


def derive_lexicon(
    logit_deltas: torch.Tensor,
    tokenizer,
    k: int = 200,
    min_len: int = 3,
    max_words: int = 30,
) -> tuple[list[str], list[str]]:
    """
    Derive a word-level lexicon from logit-lens deltas, for use as a
    dose-response validation probe (see count_lexicon_occurrences).

    Generalises the hand-curated pronoun list in
    representation/III1_vector_control/blank_validation.py::count_pronouns
    (itself derived by eyeballing the gender vector's logit-lens output) into
    an automatic procedure applicable to any direction vector.

    Scans the top-k tokens in each direction, keeps only whole-word,
    ASCII-alphabetic tokens (>= min_len chars after stripping whitespace/BPE
    markers), lowercases and deduplicates while preserving rank order, and
    caps each list at max_words.

    Returns (positive_words, negative_words).
    """
    top_pos = torch.topk(logit_deltas, k)
    top_neg = torch.topk(-logit_deltas, k)

    def _clean(indices) -> list[str]:
        seen, words = set(), []
        for i in indices:
            tok = tokenizer.decode([i.item()]).strip()
            if not _WORD_RE.match(tok) or len(tok) < min_len:
                continue
            w = tok.lower()
            if w in seen:
                continue
            seen.add(w)
            words.append(w)
            if len(words) >= max_words:
                break
        return words

    return _clean(top_pos.indices), _clean(top_neg.indices)


def count_lexicon_occurrences(
    text: str,
    pos_words: list[str],
    neg_words: list[str],
) -> tuple[int, int]:
    """
    Count whole-word, case-insensitive occurrences of pos_words / neg_words
    in text. Returns (n_positive, n_negative).

    Generalises count_pronouns() to an arbitrary logit-lens-derived lexicon
    for any direction vector (gender, interest, persona, ...). Intended use:
    for a dose-response sweep (alpha = -2..2), counting (n_pos - n_neg) per
    generation should trend monotonically with alpha if the vector causally
    drives generation toward/away from the concept it was derived from.
    """
    def _count(words: list[str]) -> int:
        if not words:
            return 0
        pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
        return len(pattern.findall(text))

    return _count(pos_words), _count(neg_words)


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
