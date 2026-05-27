"""
Runner: Full three-tier pipeline orchestration

Runs the complete pipeline for one or more persona contrast conditions:
  I.1  → collect activations
  I.2  → compute mean-difference vectors
  I.3  → train linear probes / CAV vectors
  I.4  → evaluate candidate vectors
  II.1 → persona analytics (GDV, PCA, UMAP, t-SNE, layer heatmaps)
  II.2 → interestingness analytics
  III.* → control experiments (requires --run_control flag; loads model again)

Intended for end-to-end runs on HPC after activations are already collected.
For activation collection use run_collect_activations.py (GPU job).
The representation→analytics portion is CPU-only and can run on a login/compute
node without the model loaded.

Usage:
    python runners/run_pipeline.py \
        --conditions female_anger female_excitement male_anger \
        --data_dir data/experiments/gender_emotion \
        --results_dir results/experiments/gender_emotion \
        [--run_control]
"""

from __future__ import annotations
import argparse

# from representation.I1_contrast_design.load import load_activation_results
# from representation.I2_mean_difference.mean_difference import compute_all_contrasts
# from representation.I3_linear_probe.probe import train_all_probes
# from representation.I4_vector_evaluation.evaluate import evaluate_all_vectors
# from analytics.II1_persona_analytics.gdv import run_gdv_pipeline
# from analytics.II1_persona_analytics.umap_embed import run_umap_pipeline
# from analytics.II1_persona_analytics.tsne_embed import run_tsne_pipeline
# from analytics.II2_interestingness_analytics.global_direction import find_global_interest_direction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--run_control", action="store_true",
                   help="Also run control experiments (requires GPU + model)")
    return p.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
