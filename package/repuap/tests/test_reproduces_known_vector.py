"""
Light validation against real study data: rebuild the exact
blank_interest_high_vs_blank_interest_low contrast (same split rule as
runners/run_representation_discovery.py) from the raw activation file
already on disk, and confirm repuap.discover reproduces the already-saved
mean-difference vector -- proving the ported math matches what actually
produced this study's real results, with no model load required.

Skips (does not fail) if the study's own data files aren't present, e.g. in
a fresh clone of just the package.
"""
from pathlib import Path

import numpy as np
import pytest

from repuap.discover import build_layer_matrix, compute_mean_difference_vector

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ACTIVATIONS = REPO_ROOT / "data" / "results_blank_activations.npy"
SAVED_VECTOR = (REPO_ROOT / "results" / "representation_discovery" / "interestingness" /
                 "md_vectors" / "blank_interest_high_vs_blank_interest_low.npy")

_INTEREST_HIGH_LABELS = {"Very Interesting", "Extremely Interesting"}
_INTEREST_LOW_LABELS = {"Not Interesting", "Slightly Interesting"}
_LAYER_KEY = "language_29_D5120"

pytestmark = pytest.mark.skipif(
    not (RAW_ACTIVATIONS.exists() and SAVED_VECTOR.exists()),
    reason="study data files not present (expected outside the full study repo checkout)",
)


def _as_package_records(raw_results: list[dict]) -> list[dict]:
    """Adapt the study's raw result-dict schema to what build_layer_matrix expects."""
    records = []
    for r in raw_results:
        label = r.get("interestingness") or r.get("interestingness_label")
        records.append({"label": label, "parse_ok": True, "embeddings": r["embeddings"]})
    return records


def test_mean_difference_vector_matches_saved_study_result():
    obj = np.load(RAW_ACTIVATIONS, allow_pickle=True).item()
    results = obj["results"]

    high = _as_package_records([r for r in results
                                 if (r.get("interestingness") or r.get("interestingness_label")) in _INTEREST_HIGH_LABELS])
    low = _as_package_records([r for r in results
                                if (r.get("interestingness") or r.get("interestingness_label")) in _INTEREST_LOW_LABELS])
    assert len(high) > 2 and len(low) > 2

    matrix_pos = build_layer_matrix(high)
    matrix_neg = build_layer_matrix(low)
    assert _LAYER_KEY in matrix_pos and _LAYER_KEY in matrix_neg

    v_repuap = compute_mean_difference_vector(matrix_pos[_LAYER_KEY], matrix_neg[_LAYER_KEY])

    saved = np.load(SAVED_VECTOR, allow_pickle=True).item()
    v_saved = saved[_LAYER_KEY]

    np.testing.assert_allclose(v_repuap, v_saved, rtol=1e-5, atol=1e-6)
