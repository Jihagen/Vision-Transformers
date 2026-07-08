"""
Runner: Branch C Nigeria country-vector causal validation (persona track).

Mirrors Branch B's structure (run_interest_validation.py), but validates the
NIGERIA country direction @ language_23_D5120 (averaged "extended_nigeria
minus base" vector over 16 gender x emotion conditions; see
results/extra_checks/logit_lens/country_nigeria_language_23.json), using the
clean nigeria/nigerian/african/africa/... lexicon as the measure -- a much
stronger expected signal than Branch A's pronoun-count measure.

PRIMARY (confound-free): blank (no-persona) prompt, 4 conditions at
language_23_D5120:
    no_inject     -- read from existing data/results_blank_activations.npy
    inject_pos2   -- h' = h + 2*v  ("toward Nigeria/Africa framing")
    inject_neg2   -- h' = h - 2*v  ("away from Nigeria/Africa framing")
    ablate        -- h' = h - (h.v_hat) v_hat

Primary measure: lexicon-count shift (Nigeria/African country-name words) in
the generated explanation. A blank prompt has zero reason to mention any
country, so any inject_pos2 > no_inject shift in country-name mentions is
strong, surprising causal evidence -- the "better signal" version of Branch A.

SECONDARY (generalisation, persona prompt without a country field): does
injecting the Nigeria direction into a BASE persona (no "Country:" field --
default male_excitement) cause it to spontaneously adopt Nigeria/Africa
framing? alpha sweep -2..2, rating distribution + lexicon-count shift.

Usage
-----
Dry-run (check paths, no model load):
    python runners/run_country_validation.py --dry_run

Full run (defaults: language_23_D5120, n=100 each, male_excitement alphas -2..2):
    python runners/run_country_validation.py
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
_DEFAULT_COUNTRY_MD_VECTORS_DIR = (
    "results/representation_discovery/extended_nigeria_country/md_vectors"
)
_DEFAULT_LEXICON_JSON = "results/extra_checks/logit_lens/country_nigeria_language_23.json"
_SECONDARY_ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Branch C: Nigeria country-vector causal validation "
                     "(blank-prompt lexicon-count primary, base-persona alpha sweep secondary)."
    )
    p.add_argument(
        "--country_md_vectors_dir", default=_DEFAULT_COUNTRY_MD_VECTORS_DIR,
        help="Directory of '<cond>_extended_nigeria_vs_<cond>.npy' contrast vectors "
             "(default: extended_nigeria_country/md_vectors).",
    )
    p.add_argument(
        "--layer", default="language_23_D5120",
        help="Target layer for injection/ablation (default: language_23_D5120).",
    )
    p.add_argument(
        "--lexicon_json", default=_DEFAULT_LEXICON_JSON,
        help="Path to logit-lens evidence JSON with a 'lexicon' field "
             "(default: country_nigeria_language_23.json).",
    )
    p.add_argument(
        "--n_images_primary", type=int, default=100,
        help="Number of images for the blank-prompt primary experiment (default: 100).",
    )
    p.add_argument(
        "--n_images_secondary", type=int, default=100,
        help="Number of images for the persona-prompt secondary sweep (default: 100).",
    )
    p.add_argument(
        "--secondary_persona", default="male_excitement",
        help="Persona condition (from get_all_conditions('base'), no Country field) "
             "for the secondary alpha sweep (default: male_excitement).",
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
        "--offload_suffix", default="country_validation",
        help="Suffix for CPU offload directory (default: country_validation).",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Directory for results (default: results/country_validation/<layer>).",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print config and validate paths/vector/lexicon without loading the model.",
    )
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    layer_slug = args.layer.replace("_D5120", "").replace("_D1408", "")
    return Path("results/country_validation") / layer_slug


def main() -> None:
    args = parse_args()
    out_dir = _resolve_output_dir(args)
    primary_dir = out_dir / "primary"
    secondary_dir = out_dir / "secondary" / args.secondary_persona

    from control.III1_vector_control.control import load_averaged_country_vector
    from control.III1_vector_control.lexicon_validation import load_lexicon

    if args.dry_run:
        print(f"Country md_vectors dir: {args.country_md_vectors_dir}")
        print(f"Layer:               {args.layer}")
        print(f"Lexicon JSON:        {args.lexicon_json}")
        print(f"Primary n_images:    {args.n_images_primary}  -> {primary_dir}")
        print(f"Secondary persona:   {args.secondary_persona}")
        print(f"Secondary alphas:    {args.secondary_alphas}")
        print(f"Secondary n_images:  {args.n_images_secondary}  -> {secondary_dir}")
        print(f"Blank activations:   {args.blank_activations}")

        import numpy as np
        v = load_averaged_country_vector(args.country_md_vectors_dir, args.layer)
        print(f"Vector @ {args.layer}: shape={v.shape}, norm={float(np.linalg.norm(v)):.4f}")

        pos_words, neg_words = load_lexicon(args.lexicon_json)
        print(f"Lexicon: {len(pos_words)} positive words, {len(neg_words)} negative words")
        print(f"  positive: {pos_words}")
        print(f"  negative: {neg_words}")

        import pandas as pd
        manifest_path = Path(args.manifest) if args.manifest else \
                        Path("data/selected_uniform_500_manifest.pkl")
        manifest_df = pd.read_pickle(manifest_path)
        print(f"Manifest: {len(manifest_df)} images @ {manifest_path}")

        baseline = np.load(args.blank_activations, allow_pickle=True).item()
        print(f"Blank baseline: {len(baseline['results'])} results @ {args.blank_activations}")

        from runners.experiment_definitions import get_all_conditions
        conditions = get_all_conditions("base")
        if args.secondary_persona not in conditions:
            raise KeyError(
                f"Unknown persona {args.secondary_persona!r}. "
                f"Sample valid keys: {sorted(conditions)[:5]}..."
            )
        print(f"Secondary persona '{args.secondary_persona}' OK "
              f"({len(conditions)} conditions in base, no Country field)")
        return

    primary_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)

    # ── Load vector + lexicon ────────────────────────────────────────────────
    import numpy as np
    from control.III1_vector_control.lexicon_validation import (
        summarise_lexicon_results, lexicon_count_by_alpha,
    )
    from control.III1_vector_control.blank_validation import (
        run_blank_conditions, load_blank_baseline,
    )
    from control.III1_vector_control.control import (
        run_control_experiment, summarise_and_plot,
    )

    logger.info(f"Loading averaged Nigeria country vector @ {args.layer}")
    vector = load_averaged_country_vector(args.country_md_vectors_dir, args.layer)

    lexicon = load_lexicon(args.lexicon_json)
    logger.info(f"Lexicon: {len(lexicon[0])} positive / {len(lexicon[1])} negative words")

    # ── Load images ───────────────────────────────────────────────────────────
    import pandas as pd
    manifest_path = Path(args.manifest) if args.manifest else \
                    Path("data/selected_uniform_500_manifest.pkl")
    manifest_df = pd.read_pickle(manifest_path)
    images_primary = manifest_df[["filename", "img_path"]].head(args.n_images_primary).to_dict(orient="records")
    images_secondary = manifest_df[["filename", "img_path"]].head(args.n_images_secondary).to_dict(orient="records")
    logger.info(f"Primary images: {len(images_primary)} | Secondary images: {len(images_secondary)} (from {manifest_path})")

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
    summarise_lexicon_results(
        all_primary, lexicon=lexicon, out_dir=primary_dir,
        title="Blank-prompt Nigeria-vector condition summary "
              "(predicted: inject_pos2 has the most Nigeria/Africa mentions)",
    )

    # ── SECONDARY: country-less persona alpha sweep (generalisation) ───────────
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

    logger.info("Summarising secondary (persona-prompt) results [generalisation test]...")
    summarise_and_plot(secondary_results, baseline_alpha=0.0, out_dir=secondary_dir)

    # Exploratory: lexicon-count shift by alpha in the country-less persona
    # [generalisation: does the Nigeria direction inject Nigeria/Africa framing
    #  into a persona whose prompt never mentions a country?]
    lexicon_count_by_alpha(secondary_results, lexicon, out_path=secondary_dir / "lexicon_diff_by_alpha.csv")

    print(f"\n✅ Branch C country validation complete -> {out_dir}")


if __name__ == "__main__":
    main()
