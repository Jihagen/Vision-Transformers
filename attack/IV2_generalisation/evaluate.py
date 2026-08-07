"""
Core evaluation loop for Generalisation Level 1: for one task, run every
sample in the frozen manifest through clean + every available UAP condition,
using the exact same model.generate() / perturbation mechanics as the
existing interestingness/relevance evals (see perturbation.py).

No training, no scaling, no persona injection — see attack/IV2_generalisation
module docstring and the implementation brief for the constraints this must
respect.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from attack.IV2_generalisation.perturbation import generate_response, load_delta
from attack.IV2_generalisation.tasks import TaskSpec, parse_task_response
from utils.image_utils import preprocess_image

logger = logging.getLogger(__name__)


def build_conditions(uap_conditions: list[tuple[str, float, Path]]) -> list[tuple[str, float | None, np.ndarray | None]]:
    """[('clean', None, None), ('interest', 0.1, delta_array), ...]"""
    conditions = [("clean", None, None)]
    for attack, eps, path in uap_conditions:
        conditions.append((attack, eps, load_delta(path)))
    return conditions


def evaluate_task(
    model,
    processor,
    task: TaskSpec,
    manifest: pd.DataFrame,
    conditions: list[tuple[str, float | None, np.ndarray | None]],
    max_side: int = 336,
    max_new_tokens: int = 128,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Returns one row per (sample, condition) with the common schema from spec
    section 5, plus task.extra_manifest_cols carried through from the
    manifest. 'clean' condition rows have attack='clean', epsilon=NaN,
    delta=0, model_label==clean_model_label.
    """
    if manifest.empty:
        raise ValueError("Empty manifest — did prepare_datasets.py run for this task?")
    rows_df = manifest.head(limit) if limit else manifest
    delta_t_cache: dict = {}

    records = []
    n = len(rows_df)
    for i, (_, row) in enumerate(rows_df.iterrows()):
        try:
            image = preprocess_image(row["img_path"], max_side=max_side)
        except Exception as e:
            logger.warning(f"  [{i+1}/{n}] skip {row['sample_id']}: image load failed — {e}")
            continue

        prompt = task.prompt_builder(row)
        clean_label = None
        for attack, eps, delta_np in conditions:
            raw = generate_response(
                model, processor, image, prompt,
                delta_np=delta_np, delta_t_cache=delta_t_cache,
                max_side=max_side, max_new_tokens=max_new_tokens,
            )
            parsed = parse_task_response(raw, task.valid_labels)

            if attack == "clean":
                clean_label = parsed["label"]

            delta_val = (parsed["label"] - clean_label
                         if (parsed["label"] is not None and clean_label is not None) else None)

            record = {
                "dataset": task.dataset,
                "task": task.name,
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "attack": attack,
                "epsilon": eps,
                "condition": attack if attack == "clean" else f"{attack}_eps{eps:.2f}",
                "model_label": parsed["label"],
                "clean_model_label": clean_label,
                "delta": delta_val,
                "explanation": parsed["explanation"],
                "raw_output": parsed["raw_output"],
                "parse_ok": parsed["parse_ok"],
                "used_fallback_parser": parsed["used_fallback"],
            }
            for col in task.extra_manifest_cols:
                record[col] = row.get(col)
            records.append(record)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            logger.info(f"  [{i+1}/{n}] {row['sample_id']}: clean_label={clean_label}")

    return pd.DataFrame(records)
