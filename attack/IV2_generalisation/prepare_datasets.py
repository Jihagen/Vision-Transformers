"""
LOGIN-NODE ONLY: downloads dataset shards from HuggingFace Hub and freezes a
stratified sample_manifest.csv + extracted local images per task, per spec
section 4 ("Store the selected sample IDs to disk so every UAP condition
uses exactly the same frozen evaluation subset").

Run this on a login node (internet reachable); attack/IV2_generalisation/
evaluate.py then runs entirely offline against the local files this writes.

Usage
-----
    python attack/IV2_generalisation/prepare_datasets.py --task shopping_relevance --n_samples 300
    python attack/IV2_generalisation/prepare_datasets.py --task damage_severity --n_samples 300
    python attack/IV2_generalisation/prepare_datasets.py --task moral_evaluation --n_samples 300 \\
        --hf_token <token with approved access to AIML-TUDA/smid>
    python attack/IV2_generalisation/prepare_datasets.py --task all --n_samples 300
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download + freeze evaluation samples for Generalisation Level 1.")
    p.add_argument("--task", choices=["shopping_relevance", "damage_severity", "moral_evaluation", "all"],
                    required=True)
    p.add_argument("--n_samples", type=int, default=300,
                    help="Total sample size (default 300 main-subset; use e.g. 20 for a smoke-test manifest).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_shards", type=int, default=5,
                    help="Max parquet shards to download per dataset (shopping/medic only).")
    p.add_argument("--hf_token", default=None, help="HF token with approved access to AIML-TUDA/smid.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from attack.IV2_generalisation import datasets as ds

    tasks = ["shopping_relevance", "damage_severity", "moral_evaluation"] if args.task == "all" else [args.task]

    for task in tasks:
        logger.info(f"\n{'='*60}\n  Preparing: {task}  (n={args.n_samples}, seed={args.seed})\n{'='*60}")
        if task == "shopping_relevance":
            ds.build_shopping_manifest(n_samples=args.n_samples, seed=args.seed, max_shards=args.max_shards)
        elif task == "damage_severity":
            ds.build_medic_manifest(n_samples=args.n_samples, seed=args.seed, max_shards=args.max_shards)
        elif task == "moral_evaluation":
            ds.build_smid_manifest(n_samples=args.n_samples, seed=args.seed, hf_token=args.hf_token)

    print("\n✅ Dataset preparation complete.")


if __name__ == "__main__":
    main()
