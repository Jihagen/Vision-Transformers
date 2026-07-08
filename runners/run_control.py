"""
Runner: III.1 Vector Addition Control Experiment

Injects a concept direction vector (h' = h + alpha * v) at a target layer
and measures the shift in interestingness ratings.

The primary use case is the gender vector: inject the female-direction vector
at language_29 into a male-persona prompt and measure whether ratings shift
toward the female-persona distribution.

Usage
-----
Dry-run (check paths, no model load):
    python runners/run_control.py --dry_run

Run with default settings (gender vector, male persona, language_29):
    python runners/run_control.py \\
        --md_vectors_dir results/representation_discovery/base/md_vectors \\
        --layer language_29_D5120 \\
        --persona_key male_contentment \\
        --alphas -2 -1 -0.5 0 0.5 1 2

Inject emotion-averaged vector (use discovery output from emotion mode):
    python runners/run_control.py \\
        --md_vectors_dir results/representation_discovery/base_emotion_anchor_contentment/md_vectors \\
        --layer language_29_D5120 \\
        --persona_key female_anger \\
        --alphas -1 0 1
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
_DEFAULT_ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="III.1 Vector addition control: inject concept vector and measure output shift."
    )
    p.add_argument(
        "--md_vectors_dir", required=True,
        help="Directory with I.2 mean-difference .npy files (e.g. results/.../md_vectors).",
    )
    p.add_argument(
        "--layer", default="language_29_D5120",
        help="Target layer for injection (default: language_29_D5120).",
    )
    p.add_argument(
        "--persona_key", default="male_contentment",
        help="Persona condition to use as the base prompt (default: male_contentment).",
    )
    p.add_argument(
        "--alphas", nargs="+", type=float, default=_DEFAULT_ALPHAS,
        help="Dose-response alpha values (default: -2 -1 -0.5 0 0.5 1 2).",
    )
    p.add_argument(
        "--n_images", type=int, default=100,
        help="Number of test images to use (default: 100; use a subset for speed).",
    )
    p.add_argument(
        "--manifest", default=None,
        help="Path to image manifest .pkl (defaults to data/selected_uniform_500_manifest.pkl).",
    )
    p.add_argument(
        "--model_path", default=_DEFAULT_MODEL_PATH,
        help="Path to HuggingFace model directory.",
    )
    p.add_argument(
        "--offload_suffix", default="control",
        help="Suffix for CPU offload directory (default: control).",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Directory for results (default: results/control/<layer>/<persona_key>).",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print config without loading model or running inference.",
    )
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    layer_slug = args.layer.replace("_D5120", "").replace("_D1408", "")
    return Path("results/control") / layer_slug / args.persona_key


def main() -> None:
    args = parse_args()
    out_dir = _resolve_output_dir(args)

    if args.dry_run:
        print(f"MD vectors dir:  {args.md_vectors_dir}")
        print(f"Layer:           {args.layer}")
        print(f"Persona:         {args.persona_key}")
        print(f"Alphas:          {args.alphas}")
        print(f"N images:        {args.n_images}")
        print(f"Output:          {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load vector ───────────────────────────────────────────────────────────
    from control.III1_vector_control.control import (
        load_averaged_gender_vector, run_control_experiment, summarise_and_plot,
    )
    logger.info(f"Loading gender vector from {args.md_vectors_dir} @ {args.layer}")
    vector = load_averaged_gender_vector(args.md_vectors_dir, layer_key=args.layer)

    # ── Load persona prompt ───────────────────────────────────────────────────
    from runners.experiment_definitions import get_all_conditions
    from utils.prompt_builder import build_persona_prompt
    conditions = get_all_conditions("base")
    if args.persona_key not in conditions:
        logger.error(f"Unknown persona_key '{args.persona_key}'. Valid: {sorted(conditions)}")
        sys.exit(1)
    prompt = build_persona_prompt(conditions[args.persona_key])

    # ── Load test images ──────────────────────────────────────────────────────
    import pandas as pd
    manifest_path = Path(args.manifest) if args.manifest else \
                    Path("data/selected_uniform_500_manifest.pkl")
    manifest_df = pd.read_pickle(manifest_path)
    images = manifest_df[["filename", "img_path"]].head(args.n_images).to_dict(orient="records")
    logger.info(f"Using {len(images)} test images from {manifest_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    from utils.model_loader import load_model_and_processor
    from utils.hpc import get_offload_dir, setup_cuda_env
    setup_cuda_env()
    offload = get_offload_dir(
        base=Path(args.model_path).parent / "offload_dir",
        suffix=args.offload_suffix,
    )
    logger.info("Loading model…")
    model, processor = load_model_and_processor(
        args.model_path, offload_dir=str(offload) if offload else None
    )

    # ── Run experiment ────────────────────────────────────────────────────────
    logger.info(f"Running control: alphas={args.alphas}, {len(images)} images, {len(args.alphas)} doses")
    results = run_control_experiment(
        model=model,
        processor=processor,
        images=images,
        prompt=prompt,
        vector=vector,
        layer_key=args.layer,
        alphas=sorted(args.alphas),
        out_dir=out_dir,
    )

    # ── Summarise ─────────────────────────────────────────────────────────────
    logger.info("Summarising results…")
    summary = summarise_and_plot(results, baseline_alpha=0.0, out_dir=out_dir)
    logger.info(f"Control experiment complete → {out_dir}")
    print(f"\n✅ Results saved to {out_dir}")


if __name__ == "__main__":
    main()
