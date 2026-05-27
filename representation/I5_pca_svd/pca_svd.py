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
from dataclasses import dataclass

import numpy as np


@dataclass
class SubspaceResult:
    layer_key: str
    contrast: str
    singular_values: np.ndarray          # shape (min(N,D),)
    explained_variance_ratio: np.ndarray # cumulative EVR
    n_components_90pct: int              # components needed to explain 90 % variance
    basis_vectors: np.ndarray            # shape (k, D) — leading k right singular vectors
    one_vector_sufficient: bool          # True if first component explains >= threshold


def fit_pca_svd(
    H_pos: np.ndarray,    # (N_pos, D)
    H_neg: np.ndarray,    # (N_neg, D)
    n_components: int | None = None,
    one_vector_threshold: float = 0.80,
) -> SubspaceResult:
    """
    Fit PCA/SVD on the positive–negative difference matrix and report subspace structure.

    The difference matrix is constructed as:
        D = H_pos - mean(H_pos)   stacked with   H_neg - mean(H_neg)
    then centred across both sets before SVD.

    Args:
        n_components: number of components to keep (None = all)
        one_vector_threshold: EVR fraction above which one vector is deemed sufficient

    Returns:
        SubspaceResult
    """
    raise NotImplementedError


def run_subspace_check(
    loaded_data: dict[str, dict],
    contrasts: list[tuple[str, str]],
    layer_keys: list[str] | None = None,
    save_dir=None,
) -> dict[str, list[SubspaceResult]]:
    """
    Run subspace check for all contrasts × specified layers.

    Returns:
        {contrast_name: [SubspaceResult per layer]}
    """
    raise NotImplementedError
