"""
Runner: Branch A blank-prompt gender-vector validation (primary + secondary).

PRIMARY (confound-free): blank (no-persona) prompt, 4 conditions at a target
layer (default language_29_D5120, gender direction):
    no_inject     — read from existing data/results_blank_activations.npy
    inject_pos2   — h' = h + 2*v  ("toward female")
    inject_neg2   — h' = h - 2*v  ("toward male")
    ablate        — h' = h - (h.v_hat) v_hat

Primary measure: pronoun counts (she/her/herself vs he/him/his/himself) — an
a priori prediction from the logit-lens projection of this vector (see
control/III1_vector_control/logit_lens.py). Secondary measures:
rating-distribution shift (causal potency) and a TF-IDF gender-coding probe
score (exploratory; trained on existing persona-labeled explanations).

SECONDARY (ecological "finding", not validation): male_contentment persona
prompt, alpha sweep over the same vector/layer. Rating-distribution shift
only — explicitly NOT treated as evidence about the vector's identity, since
the persona prompt's explicit "Gender: Male" field is a known confound.

Usage
-----
Dry-run (check paths, no model load):
    python runners/run_blank_validation.py --dry_run

Full run (defaults: language_29_D5120, n=100 each, male_contentment alphas -2..2):
    python runners/run_blank_validation.py \\
        --md_vectors_dir results/representation_discovery/base/md_vectors
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
_SECONDARY_ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Branch A: blank-prompt gender-vector validation (primary) "
                     "+ male_contentment alpha sweep (secondary)."
    )
    p.add_argument(
        "--md_vectors_dir", required=True,
        help="Directory with I.2 mean-difference .npy files (e.g. results/.../md_vectors).",
    )
    p.add_argument(
        "--layer", default="language_29_D5120",
        help="Target layer for injection/ablation (default: language_29_D5120).",
    )
    p.add_argument(
        "--n_images_primary", type=int, default=100,
        help="Number of images for the blank-prompt primary experiment (default: 100).",
    )
    p.add_argument(
        "--n_images_secondary", type=int, default=100,
        help="Number of images for the male_contentment secondary sweep (default: 100).",
    )
    p.add_argument(
        "--secondary_persona", default="male_contentment",
        help="Persona condition for the secondary alpha sweep (default: male_contentment).",
    )
    p.add_argument(
        "--secondary_alphas", nargs="+", type=float, default=_SECONDARY_ALPHAS,
        help="Alpha values for the secondary sweep (default: -2 -1 0 1 2).",
    )
    p.add_argument(
        "--manifest", default=None,
        help="Path to image manifest .pkl (defaults to data/selected_uniform_500_manifest.pkl).",
    )
    p.add_argument(
        "--blank_activations", default="data/results_blank_activations.npy",
        help="Path to existing blank-condition activations (no_inject baseline).",
    )
    p.add_argument(
        "--model_path", default=_DEFAULT_MODEL_PATH,
        help="Path to HuggingFace model directory.",
    )
    p.add_argument(
        "--offload_suffix", default="blank_validation",
        help="Suffix for CPU offload directory (default: blank_validation).",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Directory for results (default: results/blank_validation/<layer>).",
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
    return Path("results/blank_validation") / layer_slug


def main() -> None:
    args = parse_args()
    out_dir = _resolve_output_dir(args)
    primary_dir = out_dir / "primary"
    secondary_dir = out_dir / "secondary" / args.secondary_persona

    if args.dry_run:
        print(f"MD vectors dir:      {args.md_vectors_dir}")
        print(f"Layer:               {args.layer}")
        print(f"Primary n_images:    {args.n_images_primary}  -> {primary_dir}")
        print(f"Secondary persona:   {args.secondary_persona}")
        print(f"Secondary alphas:    {args.secondary_alphas}")
        print(f"Secondary n_images:  {args.n_images_secondary}  -> {secondary_dir}")
        print(f"Blank activations:   {args.blank_activations}")
        return

    primary_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)

    # ── Load vector ───────────────────────────────────────────────────────────
    from control.III1_vector_control.control import (
        load_averaged_gender_vector, run_control_experiment, summarise_and_plot,
    )
    from control.III1_vector_control.blank_validation import (
        run_blank_conditions, load_blank_baseline, summarise_blank_results,
    )
    from control.III1_vector_control.gender_probe import (
        load_gender_corpus, train_gender_probe, score_explanations,
    )

    logger.info(f"Loading gender vector from {args.md_vectors_dir} @ {args.layer}")
    vector = load_averaged_gender_vector(args.md_vectors_dir, layer_key=args.layer)

    # ── Load images ───────────────────────────────────────────────────────────
    import pandas as pd
    manifest_path = Path(args.manifest) if args.manifest else \
                    Path("data/selected_uniform_500_manifest.pkl")
    manifest_df = pd.read_pickle(manifest_path)
    images_primary = manifest_df[["filename", "img_path"]].head(args.n_images_primary).to_dict(orient="records")
    images_secondary = manifest_df[["filename", "img_path"]].head(args.n_images_secondary).to_dict(orient="records")
    logger.info(f"Primary images: {len(images_primary)} | Secondary images: {len(images_secondary)} (from {manifest_path})")

    # ── Train gender-coding probe (CPU, fast) ────────────────────────────────
    logger.info("Training TF-IDF gender-coding probe on existing persona explanations...")
    texts, labels = load_gender_corpus()
    gender_probe, cv_acc = train_gender_probe(texts, labels)
    logger.info(f"Gender probe 5-fold CV accuracy on labeled persona corpus: {cv_acc:.4f}")

    # ── Load model ────────────────────────────────────────────────────────────
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

    # ── PRIMARY: blank-prompt injection/ablation ────────────────────────────────
    logger.info(f"=== PRIMARY: blank prompt, {args.layer}, n={len(images_primary)} ===")
    primary_results = run_blank_conditions(
        model=model, processor=processor, images=images_primary,
        vector=vector, layer_key=args.layer, out_dir=primary_dir,
    )
    baseline_results = load_blank_baseline(images_primary, blank_path=args.blank_activations)
    all_primary = baseline_results + primary_results

    logger.info("Summarising primary (blank-prompt) results...")
    summarise_blank_results(all_primary, gender_probe=gender_probe, out_dir=primary_dir)

    # ── SECONDARY: male_contentment alpha sweep ─────────────────────────────────
    logger.info(f"=== SECONDARY: {args.secondary_persona}, {args.layer}, "
                f"alphas={args.secondary_alphas}, n={len(images_secondary)} ===")
    from runners.experiment_definitions import get_all_conditions
    from utils.prompt_builder import build_persona_prompt
    conditions = get_all_conditions("base")
    if args.secondary_persona not in conditions:
        logger.error(f"Unknown persona '{args.secondary_persona}'. Valid: {sorted(conditions)}")
        sys.exit(1)
    secondary_prompt = build_persona_prompt(conditions[args.secondary_persona])

    secondary_results = run_control_experiment(
        model=model, processor=processor, images=images_secondary,
        prompt=secondary_prompt, vector=vector, layer_key=args.layer,
        alphas=sorted(args.secondary_alphas), out_dir=secondary_dir,
    )

    logger.info("Summarising secondary (persona-prompt) results [FINDING, not validation]...")
    summarise_and_plot(secondary_results, baseline_alpha=0.0, out_dir=secondary_dir)

    # Exploratory: TF-IDF gender-probe score per alpha, to document the size
    # of the prompt-echo confound the blank-prompt design avoids.
    import pandas as pd
    sec_df = pd.DataFrame([
        {"alpha": r.alpha, "explanation": r.explanation} for r in secondary_results
    ])
    sec_df["tfidf_gender_score"] = score_explanations(gender_probe, sec_df["explanation"].tolist())
    tfidf_by_alpha = sec_df.groupby("alpha")["tfidf_gender_score"].agg(["mean", "std", "count"]).round(4)
    print("\nSecondary (persona-prompt) TF-IDF gender-probe score by alpha "
          "[documents prompt-echo confound magnitude]:")
    print(tfidf_by_alpha.to_string())
    tfidf_by_alpha.to_csv(secondary_dir / "tfidf_gender_score_by_alpha.csv")

    print(f"\n✅ Branch A blank validation complete -> {out_dir}")


if __name__ == "__main__":
    main()
