"""
Runner: TIER IV Step 2 -- gradient-matching image perturbation.

For each image, find a bounded pixel-space perturbation whose rating-token
hidden state at `--layer` moves toward h_baseline + alpha * direction_vector
(the same target an activation injection of alpha*direction_vector would
produce), then evaluate whether full generation (no injection) on the
perturbed image shifts the interestingness rating in the predicted direction.

See attack/IV1_gradient_matching/perturb.py for the pipeline and its
"first-draft scaffold, not yet smoke-tested" caveats.

Usage
-----
Dry-run (check paths/vector/imports, no model load):
    python runners/run_gradient_matching.py --dry_run

Smoke-test (1 image, few PGD steps, prints diagnostics, no large artifacts):
    python runners/run_gradient_matching.py --smoke_test --n_images 1 --n_steps 3

Full run:
    python runners/run_gradient_matching.py --n_images 20
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = (
    "/anvme/workspace/iwi5268h-vision-transformers/hpc_infrastructure/hf_cache/"
    "models--meta-llama--Llama-4-Scout-17B-16E-Instruct/local-repo"
)
_DEFAULT_VECTOR = (
    "results/representation_discovery/interestingness/md_vectors/"
    "blank_interest_high_vs_blank_interest_low.npy"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TIER IV Step 2: gradient-matching image perturbation "
                     "toward an activation-injection target."
    )
    p.add_argument("--vector_path", default=_DEFAULT_VECTOR,
                    help="Path to the .npy mean-difference vector dict (default: v_interest_blank).")
    p.add_argument("--layer", default="language_29_D5120",
                    help="Target layer (default: language_29_D5120).")
    p.add_argument("--alpha", type=float, default=2.0,
                    help="Injection strength to match (default: 2.0, Branch B's tested value; "
                         "refine using TIER IV step 1's saturation sweep).")
    p.add_argument("--epsilon", type=float, default=0.05,
                    help="L_inf perturbation bound in pixel_values units (default: 0.05).")
    p.add_argument("--lr", type=float, default=0.01,
                    help="PGD step size (default: 0.01).")
    p.add_argument("--n_steps", type=int, default=20,
                    help="PGD iterations per image (default: 20).")
    p.add_argument("--n_images", type=int, default=5,
                    help="Number of images to process (default: 5).")
    p.add_argument("--manifest", default=None,
                    help="Path to image manifest .pkl (defaults to data/selected_uniform_500_manifest.pkl).")
    p.add_argument("--model_path", default=_DEFAULT_MODEL_PATH,
                    help="Path to HuggingFace model directory.")
    p.add_argument("--offload_suffix", default="gradient_matching",
                    help="Suffix for CPU offload directory (default: gradient_matching).")
    p.add_argument("--output_dir", default=None,
                    help="Directory for results (default: results/gradient_matching/<layer>).")
    p.add_argument("--smoke_test", action="store_true",
                    help="Run on a small number of images/steps and print diagnostics only.")
    p.add_argument("--dry_run", action="store_true",
                    help="Print config and validate paths/vector/imports without loading the model.")
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    layer_slug = args.layer.replace("_D5120", "").replace("_D1408", "")
    return Path("results/gradient_matching") / layer_slug


def main() -> None:
    args = parse_args()
    out_dir = _resolve_output_dir(args)

    import numpy as np
    import pandas as pd

    if args.dry_run:
        print(f"Vector path:  {args.vector_path}")
        print(f"Layer:        {args.layer}")
        print(f"alpha:        {args.alpha}")
        print(f"epsilon/lr/n_steps: {args.epsilon} / {args.lr} / {args.n_steps}")
        print(f"n_images:     {args.n_images}  -> {out_dir}")

        vec = np.load(args.vector_path, allow_pickle=True).item()
        v = vec[args.layer]
        print(f"Vector @ {args.layer}: shape={v.shape}, norm={float(np.linalg.norm(v)):.4f}")

        manifest_path = Path(args.manifest) if args.manifest else \
                        Path("data/selected_uniform_500_manifest.pkl")
        manifest_df = pd.read_pickle(manifest_path)
        print(f"Manifest: {len(manifest_df)} images @ {manifest_path}")
        first = manifest_df.iloc[0]
        print(f"First image: {first['filename']} -> exists={Path(first['img_path']).exists()}")

        from attack.IV1_gradient_matching import perturb  # noqa: F401
        print("attack.IV1_gradient_matching.perturb imports OK")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    from attack.IV1_gradient_matching.perturb import (
        gradient_match_perturbation, evaluate_perturbation,
    )
    from utils.prompt_builder import build_blank_prompt

    logger.info(f"Loading direction vector @ {args.layer}")
    vec = np.load(args.vector_path, allow_pickle=True).item()
    vector = vec[args.layer].astype(np.float32)
    vector = vector / np.linalg.norm(vector)

    manifest_path = Path(args.manifest) if args.manifest else \
                    Path("data/selected_uniform_500_manifest.pkl")
    manifest_df = pd.read_pickle(manifest_path)
    n = 1 if args.smoke_test else args.n_images
    images = manifest_df[["filename", "img_path"]].head(n).to_dict(orient="records")
    logger.info(f"Images: {len(images)} (from {manifest_path})")

    from utils.model_loader import load_model_and_processor
    from utils.hpc import get_offload_dir, setup_cuda_env
    setup_cuda_env()
    offload = get_offload_dir(
        base=Path(args.model_path).parent / "offload_dir",
        suffix=args.offload_suffix,
    )
    logger.info("Loading model...")
    model, processor = load_model_and_processor(
        args.model_path, offload_dir=str(offload) if offload else None
    )

    prompt = build_blank_prompt()
    n_steps = 3 if args.smoke_test else args.n_steps

    rows = []
    for img in images:
        logger.info(f"=== {img['filename']} ===")
        result = gradient_match_perturbation(
            model=model, processor=processor,
            image_path=img["img_path"], prompt=prompt,
            layer_key=args.layer, direction_vector=vector, alpha=args.alpha,
            epsilon=args.epsilon, lr=args.lr, n_steps=n_steps,
        )
        ratings = evaluate_perturbation(model, processor, result)
        logger.info(f"  original:  {ratings['original']}")
        logger.info(f"  perturbed: {ratings['perturbed']}")
        rows.append({
            "filename": img["filename"],
            "cos_before": result["cos_history"][0],
            "cos_after": result["cos_history"][-1],
            "rating_original": ratings["original"][0],
            "rating_perturbed": ratings["perturbed"][0],
        })

    df = pd.DataFrame(rows)
    print("\nGradient-matching results:")
    print(df.to_string(index=False))
    if not args.smoke_test:
        df.to_csv(out_dir / "gradient_matching_results.csv", index=False)
        print(f"\n✅ TIER IV Step 2 (gradient matching) complete -> {out_dir}")


if __name__ == "__main__":
    main()
