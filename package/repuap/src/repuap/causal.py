"""
Stage III — Causal Control.

Tests causal sufficiency: does a candidate direction (from discover.py)
actually drive the model's judgment when injected into the residual stream,
or is it merely correlated with it? Sweep injection strength alpha and
measure how the judgment shifts — a real dose-response curve (one fixed
direction, continuously scaled), unlike the UAP "budget-response" in uap.py
(independently-trained deltas at different L_inf budgets).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .hooks import HookState, register_inject_hooks
from .model_io import preprocess_image
from .generation import build_inputs, generate_text
from .tasks import JudgmentTask, parse_response

logger = logging.getLogger(__name__)


@dataclass
class ControlResult:
    filename: str
    alpha: float
    layer_key: str
    label: object
    score: float | None
    explanation: str | None
    parse_ok: bool


def _comp_key_to_hook_name(layer_key: str) -> str:
    """Map a comp_key like 'language_29_D5120' to the hook name registered
    by register_inject_hooks ('llm_layer_29_rating_token'), or 'vision_N_...'
    to 'vision_layer_N_cls'."""
    import re
    m = re.search(r"language_(\d+)", layer_key)
    if m:
        return f"llm_layer_{m.group(1)}_rating_token"
    m = re.search(r"vision_(\d+)", layer_key)
    if m:
        return f"vision_layer_{m.group(1)}_cls"
    raise ValueError(f"Cannot map layer_key {layer_key!r} to a hook name.")


def run_dose_response(
    model,
    processor,
    task: JudgmentTask,
    images: list[dict],       # [{"filename", "img_path"}, ...]
    prompt: str,
    vector: np.ndarray,       # unit-normalised concept direction, shape (D,)
    layer_key: str,           # e.g. "language_29_D5120"
    alphas: list[float],
    max_new_tokens: int = 64,
    max_side: int = 800,
    checkpoint_every: int | None = None,
    out_dir: str | Path | None = None,
) -> list[ControlResult]:
    """
    For each alpha, inject alpha*vector at layer_key and record the model's
    judgment. alpha=0 should reproduce the uninjected baseline.
    """
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    hook_name = _comp_key_to_hook_name(layer_key)
    device = next(model.parameters()).device
    results: list[ControlResult] = []

    for alpha in alphas:
        logger.info(f"alpha={alpha:+.2f} -- {len(images)} images")
        state = HookState()
        if abs(alpha) > 1e-9:
            register_inject_hooks(model, state, {hook_name: (vector, float(alpha))})

        for idx, img in enumerate(images, 1):
            img_path, filename = img["img_path"], img.get("filename", Path(img["img_path"]).name)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                image = preprocess_image(img_path, max_side=max_side)
                inputs = build_inputs(processor, image, prompt, device)
                decoded = generate_text(model, processor, inputs["input_ids"],
                                         inputs["attention_mask"], inputs["pixel_values"],
                                         max_new_tokens=max_new_tokens)
                parsed = parse_response(decoded, task)
                score = task.score(parsed.label) if parsed.parse_ok else None
                results.append(ControlResult(filename, alpha, layer_key, parsed.label,
                                              score, parsed.explanation, parsed.parse_ok))
            except Exception as e:
                logger.warning(f"  [{alpha:+.2f}] {filename}: {e} -- skipping")

            if out_dir and checkpoint_every and idx % checkpoint_every == 0:
                _save_results(results, out_dir / "control_results_partial.csv")

        state.remove_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if out_dir:
        _save_results(results, out_dir / "control_results.csv")
    return results


def _save_results(results: list[ControlResult], path: Path) -> None:
    pd.DataFrame([r.__dict__ for r in results]).to_csv(path, index=False)


def summarise_dose_response(results: list[ControlResult]) -> pd.DataFrame:
    """Per-alpha mean/std score + n, for plotting or a monotonicity check."""
    df = pd.DataFrame([r.__dict__ for r in results])
    df = df[df["parse_ok"]]
    return (df.groupby("alpha")["score"]
            .agg(mean_score="mean", std_score="std", n="count")
            .reset_index().round(4))


def plot_dose_response(summary: pd.DataFrame, layer_key: str, save_path: str | Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(summary["alpha"], summary["mean_score"], yerr=summary["std_score"],
                fmt="o-", color="steelblue", lw=2, capsize=4, ms=7)
    ax.axvline(0, color="grey", ls="--", lw=1, alpha=0.6, label="baseline (alpha=0)")
    ax.set_xlabel("injection alpha")
    ax.set_ylabel("mean judgment score")
    ax.set_title(f"Dose-response @ {layer_key}")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
