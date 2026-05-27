"""
Runner: I.1 Activation Collection

Entry point for running a single persona condition through the full activation
collection pipeline (representation/I1_contrast_design/collect.py).

Usage:
    python runners/run_collect_activations.py --persona_key female_anger \
        --manifest data/selected_uniform_500_manifest.pkl \
        --output_dir data/experiments/gender_emotion

This script is intentionally thin: all logic lives in representation/I1_contrast_design/.
Adjust --persona_key to run different contrast conditions.
"""

from __future__ import annotations
import argparse
from pathlib import Path

# from representation.I1_contrast_design.collect import run_experiment
# from utils.model_loader import load_model_and_processor
# from utils.hpc import get_offload_dir, setup_cuda_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--persona_key", required=True,
                   help="Identifier for the persona condition, e.g. 'female_anger'")
    p.add_argument("--manifest", required=True,
                   help="Path to selected_uniform_*.pkl image manifest")
    p.add_argument("--output_dir", required=True,
                   help="Directory where results_<persona_key>.npy is saved")
    p.add_argument("--model_path",
                   default="/anvme/workspace/iwi5268h-vision-transformers/hf_cache")
    p.add_argument("--offload_suffix", default="",
                   help="Suffix for CPU offload dir (set per parallel job)")
    p.add_argument("--checkpoint_every", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("Wire up run_experiment() from representation/I1_contrast_design/collect.py")


if __name__ == "__main__":
    main()
