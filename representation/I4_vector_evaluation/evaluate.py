"""
Vector evaluation: decide which candidate directions are worth keeping.

Each evaluation function is stateless — pass in the vectors and activations,
get back a score or boolean decision.  The summary function collects all checks
and returns a single VectorReport per (contrast, layer).
"""

from __future__ import annotations
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from representation.I1_contrast_design.load import (
    build_layer_matrix,
    get_results_list,
    sort_layer_keys,
)

logger = logging.getLogger(__name__)


@dataclass
class VectorReport:
    contrast: str
    layer_key: str
    projection_auc: float           # AUROC when separating pos/neg via scalar projection
    md_cav_cosine: float            # cosine similarity between mean-diff and CAV vectors
    probe_accuracy_cv: float        # from I.3
    probe_above_chance: bool        # probe_accuracy_cv > chance_threshold
    separation_significant: bool    # projection_auc > auc_threshold
    keep: bool                      # overall recommendation
    notes: str = ""
    extra: dict = field(default_factory=dict)


def projection_separation(
    H_pos: np.ndarray,    # (N_pos, D)
    H_neg: np.ndarray,    # (N_neg, D)
    vector: np.ndarray,   # (D,) candidate direction
) -> float:
    """
    Project all activations onto vector and measure how well the scalar
    projection separates positive from negative examples.

    Returns:
        AUROC score (0.5 = chance, 1.0 = perfect separation)
    """
    v_unit = vector / (np.linalg.norm(vector) + 1e-12)
    proj_pos = H_pos @ v_unit
    proj_neg = H_neg @ v_unit
    scores = np.concatenate([proj_pos, proj_neg])
    labels = np.concatenate([np.ones(len(H_pos)), np.zeros(len(H_neg))])
    try:
        auc = roc_auc_score(labels, scores)
        # Report the higher end (direction may be flipped)
        return float(max(auc, 1.0 - auc))
    except Exception:
        return 0.5


def alignment_score(
    mean_diff_vector: np.ndarray,    # (D,) from I.2
    cav_vector: np.ndarray,          # (D,) from I.3
) -> float:
    """
    Cosine similarity between the mean-difference and CAV vectors.

    High alignment (> ~0.7) indicates a clean, approximately linear concept.
    Low alignment suggests the concept may be compound or the probe is unstable.
    """
    a = mean_diff_vector / (np.linalg.norm(mean_diff_vector) + 1e-12)
    b = cav_vector / (np.linalg.norm(cav_vector) + 1e-12)
    return float(np.dot(a, b))


def layer_profile(
    vectors_by_layer: dict[str, np.ndarray],  # {layer_key: vector}
    H_pos_by_layer: dict[str, np.ndarray],
    H_neg_by_layer: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Compute projection_separation for each layer.

    Returns:
        {layer_key: auroc} — useful for locating where signal is strongest.
    """
    profile: dict[str, float] = {}
    for lk in sort_layer_keys(list(vectors_by_layer.keys())):
        if lk not in H_pos_by_layer or lk not in H_neg_by_layer:
            continue
        profile[lk] = projection_separation(
            H_pos_by_layer[lk],
            H_neg_by_layer[lk],
            vectors_by_layer[lk],
        )
    return profile


def evaluate_vector(
    contrast: str,
    layer_key: str,
    H_pos: np.ndarray,
    H_neg: np.ndarray,
    mean_diff_vector: np.ndarray,
    cav_vector: np.ndarray,
    probe_accuracy_cv: float,
    chance_threshold: float = 0.60,
    auc_threshold: float = 0.65,
) -> VectorReport:
    """
    Run all checks and return a VectorReport with a keep/drop recommendation.
    """
    auc    = projection_separation(H_pos, H_neg, mean_diff_vector)
    cos    = alignment_score(mean_diff_vector, cav_vector)
    above  = probe_accuracy_cv > chance_threshold
    sep    = auc > auc_threshold
    keep   = above and sep

    notes_parts = []
    if not above:
        notes_parts.append(
            f"probe acc {probe_accuracy_cv:.3f} below threshold {chance_threshold}"
        )
    if not sep:
        notes_parts.append(
            f"AUC {auc:.3f} below threshold {auc_threshold}"
        )

    return VectorReport(
        contrast=contrast,
        layer_key=layer_key,
        projection_auc=auc,
        md_cav_cosine=cos,
        probe_accuracy_cv=probe_accuracy_cv,
        probe_above_chance=above,
        separation_significant=sep,
        keep=keep,
        notes="; ".join(notes_parts),
    )


def evaluate_all_vectors(
    loaded_data: dict[str, dict],
    md_vectors: dict[str, dict[str, np.ndarray]],    # {contrast: {layer: vector}}
    cav_vectors: dict[str, dict[str, np.ndarray]],   # {contrast: {layer: vector}}
    probe_results: dict[str, list],
    save_dir=None,
    chance_threshold: float = 0.60,
    auc_threshold: float = 0.65,
) -> dict[str, list[VectorReport]]:
    """
    Evaluate all contrasts × layers and return {contrast: [VectorReport]}.

    Parses contrast names as "{pos_key}_vs_{neg_key}" to retrieve activations.
    Optionally saves a summary CSV to save_dir.
    """
    from representation.I3_linear_probe.probe import ProbeResult

    output: dict[str, list[VectorReport]] = {}

    for contrast_name, md_by_layer in md_vectors.items():
        parts = contrast_name.split("_vs_", 1)
        if len(parts) != 2:
            logger.warning(f"[Eval] Cannot parse contrast name '{contrast_name}' — skipping")
            continue
        pos_key, neg_key = parts
        if pos_key not in loaded_data or neg_key not in loaded_data:
            logger.warning(f"[Eval] Missing keys for '{contrast_name}' — skipping")
            continue

        matrix_pos = build_layer_matrix(get_results_list(loaded_data[pos_key]))
        matrix_neg = build_layer_matrix(get_results_list(loaded_data[neg_key]))

        # Build probe_accuracy lookup for this contrast
        probe_acc: dict[str, float] = {}
        for r in probe_results.get(contrast_name, []):
            if isinstance(r, ProbeResult):
                probe_acc[r.layer_key] = r.accuracy_cv

        cav_by_layer = cav_vectors.get(contrast_name, {})
        common_layers = set(md_by_layer.keys()) & set(matrix_pos.keys()) & set(matrix_neg.keys())

        reports: list[VectorReport] = []
        for lk in sort_layer_keys(list(common_layers)):
            H_pos = matrix_pos[lk]["X"]
            H_neg = matrix_neg[lk]["X"]
            md_v  = md_by_layer[lk]
            cav_v = cav_by_layer.get(lk, md_v)  # fall back to MD vector if no CAV
            acc   = probe_acc.get(lk, float("nan"))

            try:
                report = evaluate_vector(
                    contrast=contrast_name,
                    layer_key=lk,
                    H_pos=H_pos,
                    H_neg=H_neg,
                    mean_diff_vector=md_v,
                    cav_vector=cav_v,
                    probe_accuracy_cv=acc,
                    chance_threshold=chance_threshold,
                    auc_threshold=auc_threshold,
                )
                reports.append(report)
                logger.debug(
                    f"[Eval] {contrast_name} | {lk}: "
                    f"auc={report.projection_auc:.3f} "
                    f"cos={report.md_cav_cosine:.3f} "
                    f"keep={report.keep}"
                )
            except Exception as e:
                logger.warning(f"[Eval] {contrast_name} | {lk} failed: {e}")

        output[contrast_name] = reports
        kept = sum(1 for r in reports if r.keep)
        logger.info(
            f"[Eval] {contrast_name}: {kept}/{len(reports)} layers recommended"
        )

    if save_dir is not None:
        _save_reports(output, Path(save_dir))

    return output


def _save_reports(
    all_reports: dict[str, list[VectorReport]],
    save_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "vector_evaluation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "contrast", "layer_key",
            "projection_auc", "md_cav_cosine", "probe_accuracy_cv",
            "probe_above_chance", "separation_significant", "keep", "notes",
        ])
        for contrast_name, reports in all_reports.items():
            for r in reports:
                w.writerow([
                    r.contrast, r.layer_key,
                    r.projection_auc, r.md_cav_cosine, r.probe_accuracy_cv,
                    r.probe_above_chance, r.separation_significant, r.keep, r.notes,
                ])
    logger.info(f"[Eval] Saved evaluation CSV → {csv_path}")
