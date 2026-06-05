"""
Runner: II.1 Persona Vector Analytics

Loads collected activation data and produces:
  - Condition geometry: cosine similarity matrix + UMAP/PCA scatter of mean
    activation vectors per condition at the target layer
  - (with --additivity) Additivity check: how well do individual feature vectors
    sum to reproduce compound persona activations

Usage
-----
Geometry for base variant at language_29:
    python runners/run_analytics.py --variant base --layer language_29_D5120

Geometry for a country variant:
    python runners/run_analytics.py --variant extended_germany --layer language_29_D5120

Additivity check (requires base + one country variant to be collected):
    python runners/run_analytics.py --additivity --country Germany \\
        --layer language_29_D5120

Multiple layers (e.g. include a vision layer for comparison):
    python runners/run_analytics.py --variant base \\
        --layer language_29_D5120 language_4_D5120 vision_0_D1408
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

_DEFAULT_LAYERS = ["language_29_D5120"]
_RESULTS_ROOT   = Path("results/analytics")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="II.1 Persona vector geometry analytics.")
    p.add_argument(
        "--variant", default="base",
        help="Variant to analyse, e.g. 'base', 'extended_germany'. Default: base.",
    )
    p.add_argument(
        "--layer", nargs="+", default=_DEFAULT_LAYERS, metavar="LAYER_KEY",
        help="One or more comp_keys to analyse (default: language_29_D5120).",
    )
    p.add_argument(
        "--additivity", action="store_true",
        help="Run additivity check. Requires --country and base variant to be collected.",
    )
    p.add_argument(
        "--country", default=None,
        help="Country for additivity check, e.g. 'Germany'. Registers the country variant.",
    )
    p.add_argument(
        "--md_vectors_dir", default=None,
        help="Path to I.2 md_vectors output directory (used for additivity). "
             "Defaults to results/representation_discovery/base/md_vectors.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print what would run without doing any computation.",
    )
    return p.parse_args()


def _out_dir(variant: str, layer_key: str) -> Path:
    layer_slug = layer_key.replace("_D5120", "").replace("_D1408", "")
    d = _RESULTS_ROOT / variant / layer_slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_variant(variant: str) -> dict[str, dict]:
    import numpy as np
    from runners.experiment_definitions import get_all_result_paths, register_country_variant

    # Auto-register country variants from key
    if variant.startswith("extended_"):
        country = variant.replace("extended_", "").capitalize()
        register_country_variant(country)

    paths = get_all_result_paths(variant)
    existing = {pk: p for pk, p in paths.items() if p.exists()}
    if not existing:
        raise FileNotFoundError(
            f"No result files found for variant '{variant}'. Run collection first."
        )
    loaded: dict[str, dict] = {}
    for pk, path in existing.items():
        obj = np.load(path, allow_pickle=True).item()
        if len(obj.get("results", [])) == 0:
            logger.warning(f"  {pk}: 0 results — skipping")
            continue
        loaded[pk] = obj
    logger.info(f"Loaded {len(loaded)} conditions for variant '{variant}'")
    return loaded


def _run_geometry(loaded_data: dict, layer_key: str, variant: str) -> dict:
    from analytics.II1_persona_analytics.condition_geometry import run_geometry_analysis
    out_dir = _out_dir(variant, layer_key)
    logger.info(f"Geometry analysis: {len(loaded_data)} conditions @ {layer_key}")
    return run_geometry_analysis(loaded_data, layer_key, out_dir, variant_label=variant)


def _run_additivity(
    base_loaded: dict,
    country_loaded: dict,
    layer_key: str,
    md_vectors_dir: Path,
    country: str,
) -> None:
    import numpy as np
    from analytics.II1_persona_analytics.additivity import (
        derive_feature_vectors, check_additivity,
    )
    from analytics.II1_persona_analytics.condition_geometry import get_condition_mean_vectors

    out_dir = _RESULTS_ROOT / "additivity" / country.lower()

    # Extract mean vectors per condition at target layer
    base_means    = get_condition_mean_vectors(base_loaded,    layer_key)
    country_means = get_condition_mean_vectors(country_loaded, layer_key)

    if not base_means or not country_means:
        logger.error("Cannot run additivity: mean vectors missing for one variant.")
        return

    # Load gender MD vectors from I.2 output
    gender_md: dict[str, np.ndarray] = {}
    if md_vectors_dir.exists():
        for f in md_vectors_dir.glob("female_*_vs_male_*.npy"):
            vecs = np.load(f, allow_pickle=True).item()
            if layer_key in vecs:
                gender_md[f.stem] = vecs[layer_key]
    else:
        logger.warning(f"md_vectors_dir not found: {md_vectors_dir}")

    # Country MD vectors (country_variant_X vs base_X)
    country_md: dict[str, np.ndarray] = {}
    country_md_dir = _RESULTS_ROOT.parent / "representation_discovery" / \
                     f"extended_{country.lower()}" / "md_vectors"
    if country_md_dir.exists():
        for f in country_md_dir.glob("*.npy"):
            vecs = np.load(f, allow_pickle=True).item()
            if layer_key in vecs:
                country_md[f.stem] = vecs[layer_key]

    feature_vecs = derive_feature_vectors(
        base_means, country_means, gender_md, country_md, layer_key
    )
    logger.info(f"Derived {len(feature_vecs)} feature vectors: {list(feature_vecs)}")

    df = check_additivity(base_means, country_means, feature_vecs, out_dir)
    if not df.empty:
        logger.info(f"Additivity results saved → {out_dir}")
        print("\nAdditivity summary:")
        print(df.groupby(["gender","country"])["additivity_score"]
              .agg(["mean","min","max"]).round(3).to_string())


def main() -> None:
    args = parse_args()

    if args.country:
        from runners.experiment_definitions import register_country_variant
        register_country_variant(args.country)

    if args.dry_run:
        print(f"Variant:    {args.variant}")
        print(f"Layers:     {args.layer}")
        print(f"Additivity: {args.additivity}")
        if args.additivity:
            print(f"Country:    {args.country}")
        print(f"Output:     {_RESULTS_ROOT / args.variant}")
        return

    loaded = _load_variant(args.variant)

    for layer_key in args.layer:
        logger.info(f"=== Layer: {layer_key} ===")
        result = _run_geometry(loaded, layer_key, args.variant)
        if result:
            sim = result["sim_matrix"]
            labels = result["labels"]
            n_f = sum(1 for l in labels if "female" in l)
            n_m = len(labels) - n_f
            if n_f and n_m:
                idx_f = [i for i, l in enumerate(labels) if "female" in l]
                idx_m = [i for i, l in enumerate(labels) if "female" not in l]
                wf = sim[np.ix_(idx_f, idx_f)]
                wm = sim[np.ix_(idx_m, idx_m)]
                cr = sim[np.ix_(idx_f, idx_m)]
                wf_off = wf[~np.eye(n_f, dtype=bool)]
                wm_off = wm[~np.eye(n_m, dtype=bool)]
                print(f"\n{layer_key} — cosine summary:")
                print(f"  within-female: {wf_off.mean():.3f}" if len(wf_off) > 0 else "  within-female: n/a")
                print(f"  within-male:   {wm_off.mean():.3f}" if len(wm_off) > 0 else "  within-male:   n/a (only 1 condition)")
                print(f"  cross-gender:  {cr.mean():.3f}")

    if args.additivity:
        if not args.country:
            logger.error("--additivity requires --country <Country>")
            sys.exit(1)
        country_variant = f"extended_{args.country.lower()}"
        try:
            country_loaded = _load_variant(country_variant)
        except FileNotFoundError:
            logger.error(
                f"Country variant '{country_variant}' not collected yet. "
                f"Run collection with --variant {country_variant} --country {args.country} first."
            )
            sys.exit(1)

        md_dir = Path(args.md_vectors_dir) if args.md_vectors_dir else \
                 Path("results/representation_discovery/base/md_vectors")

        for layer_key in args.layer:
            logger.info(f"=== Additivity @ {layer_key} ===")
            _run_additivity(loaded, country_loaded, layer_key, md_dir, args.country)


import numpy as np  # needed for inline numpy in main

if __name__ == "__main__":
    main()
