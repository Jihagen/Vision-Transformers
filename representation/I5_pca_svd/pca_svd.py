"""
Subspace check: does the concept require more than one direction?

Approach:
  1. Stack H_pos − H_neg residuals (or contrast differences) for a given layer.
  2. Run SVD / PCA on the difference matrix.
  3. Inspect the explained variance ratio: if the first singular value dominates
     (e.g. > 80 %), a single vector is likely sufficient.
  4. If the variance is distributed across k > 1 components, report the leading
     k directions as a candidate concept subspace.

The output (a set of k basis vectors) can be used for projection-based ablation
in control/III3_vector_subtraction instead of a single direction.
"""

from __future__ import annotations
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from representation.I1_contrast_design.load import (
    build_layer_matrix,
    get_results_list,
    sort_layer_keys,
)

logger = logging.getLogger(__name__)


@dataclass
class SubspaceResult:
    layer_key: str
    contrast: str
    singular_values: np.ndarray          # shape (k,)
    explained_variance_ratio: np.ndarray # cumulative EVR, shape (k,)
    n_components_90pct: int              # components needed to explain 90 % variance
    basis_vectors: np.ndarray            # shape (k, D) — leading k right singular vectors
    one_vector_sufficient: bool          # True if first component explains >= threshold


def fit_pca_svd(
    H_pos: np.ndarray,    # (N_pos, D)
    H_neg: np.ndarray,    # (N_neg, D)
    n_components: int | None = None,
    one_vector_threshold: float = 0.80,
    layer_key: str = "",
    contrast: str = "",
) -> SubspaceResult:
    """
    Fit PCA/SVD on the positive–negative difference matrix and report subspace structure.

    The difference matrix is constructed by centering each class separately and
    concatenating, then globally centering:
        D = [H_pos − mean(H_pos); H_neg − mean(H_neg)] − global_mean

    Args:
        n_components: number of components to keep (None = all)
        one_vector_threshold: EVR fraction above which one vector is deemed sufficient
        layer_key: stored in SubspaceResult for reference
        contrast: stored in SubspaceResult for reference

    Returns:
        SubspaceResult
    """
    # Center each class, then stack
    H_pos_c = H_pos - H_pos.mean(axis=0)
    H_neg_c = H_neg - H_neg.mean(axis=0)
    D_mat   = np.vstack([H_pos_c, H_neg_c]).astype(np.float64)

    # Global centering
    D_mat = D_mat - D_mat.mean(axis=0)

    max_k = min(D_mat.shape[0], D_mat.shape[1])
    k = int(n_components) if n_components is not None else max_k
    k = min(k, max_k)

    # Full SVD — use scipy for large matrices if available
    try:
        from scipy.sparse.linalg import svds as sparse_svds
        if k < max_k and k <= min(D_mat.shape) // 2:
            U, s, Vt = sparse_svds(D_mat, k=k)
            # svds returns in ascending order — reverse
            idx = np.argsort(s)[::-1]
            s, Vt = s[idx], Vt[idx]
        else:
            raise ValueError("use full SVD")
    except Exception:
        _, s_full, Vt_full = np.linalg.svd(D_mat, full_matrices=False)
        s   = s_full[:k]
        Vt  = Vt_full[:k]

    # Explained variance ratio (cumulative)
    total_var = float((s ** 2).sum()) + 1e-30
    evr = np.cumsum(s ** 2) / total_var

    # Number of components to reach 90 % explained variance
    n_comp_90 = int(np.searchsorted(evr, 0.90)) + 1
    n_comp_90 = min(n_comp_90, len(evr))

    one_vec_ok = bool(len(evr) > 0 and evr[0] >= one_vector_threshold)

    return SubspaceResult(
        layer_key=layer_key,
        contrast=contrast,
        singular_values=s.astype(np.float32),
        explained_variance_ratio=evr.astype(np.float32),
        n_components_90pct=n_comp_90,
        basis_vectors=Vt.astype(np.float32),
        one_vector_sufficient=one_vec_ok,
    )


def run_subspace_check(
    loaded_data: dict[str, dict],
    contrasts: list[tuple[str, str]],
    layer_keys: list[str] | None = None,
    n_components: int | None = None,
    one_vector_threshold: float = 0.80,
    save_dir=None,
) -> dict[str, list[SubspaceResult]]:
    """
    Run subspace check for all contrasts × specified layers.

    Returns:
        {contrast_name: [SubspaceResult per layer]}
    """
    output: dict[str, list[SubspaceResult]] = {}

    for pos_key, neg_key in contrasts:
        name = f"{pos_key}_vs_{neg_key}"
        if pos_key not in loaded_data or neg_key not in loaded_data:
            logger.warning(f"[SVD] Missing keys for '{name}' — skipping")
            continue

        matrix_pos = build_layer_matrix(get_results_list(loaded_data[pos_key]))
        matrix_neg = build_layer_matrix(get_results_list(loaded_data[neg_key]))

        common = set(matrix_pos.keys()) & set(matrix_neg.keys())
        if layer_keys is not None:
            common = common & set(layer_keys)

        results: list[SubspaceResult] = []
        for lk in sort_layer_keys(list(common)):
            H_pos = matrix_pos[lk]["X"]
            H_neg = matrix_neg[lk]["X"]
            try:
                sr = fit_pca_svd(
                    H_pos, H_neg,
                    n_components=n_components,
                    one_vector_threshold=one_vector_threshold,
                    layer_key=lk,
                    contrast=name,
                )
                results.append(sr)
                logger.debug(
                    f"[SVD] {name} | {lk}: "
                    f"EVR[0]={sr.explained_variance_ratio[0]:.3f} "
                    f"one_vec={sr.one_vector_sufficient} "
                    f"90pct_in_{sr.n_components_90pct}_components"
                )
            except Exception as e:
                logger.warning(f"[SVD] {name} | {lk} failed: {e}")

        output[name] = results
        n_single = sum(1 for r in results if r.one_vector_sufficient)
        logger.info(
            f"[SVD] {name}: {n_single}/{len(results)} layers single-vector sufficient"
        )

    if save_dir is not None:
        _save_subspace_results(output, Path(save_dir))

    return output


def _save_subspace_results(
    all_results: dict[str, list[SubspaceResult]],
    save_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "subspace_check.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "contrast", "layer_key",
            "evr_first", "n_components_90pct", "one_vector_sufficient",
            "n_singular_values",
        ])
        for contrast_name, results in all_results.items():
            for r in results:
                evr0 = float(r.explained_variance_ratio[0]) if len(r.explained_variance_ratio) else float("nan")
                w.writerow([
                    r.contrast, r.layer_key,
                    evr0, r.n_components_90pct, r.one_vector_sufficient,
                    len(r.singular_values),
                ])
    logger.info(f"[SVD] Saved subspace CSV → {csv_path}")

    # Save basis vectors per contrast
    for contrast_name, results in all_results.items():
        bases = {r.layer_key: r.basis_vectors for r in results}
        np.save(save_dir / f"basis_{contrast_name}.npy", bases)
