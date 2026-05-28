"""
Runner: I.1 Activation Collection — multi-condition batch runner.

Runs activation collection for one or all conditions of an experimental variant.
All logic lives in representation/I1_contrast_design/collect.py; this script
is just the CLI entry point.

Modes
-----
Single condition (useful for SLURM job arrays):
    python runners/run_collect_activations.py \\
        --persona_key female_anger --variant base

All missing conditions for a variant (sequential, for local/interactive use):
    python runners/run_collect_activations.py --variant base --all

Full run — one model load: fix mismatches first, then collect all missing:
    python runners/run_collect_activations.py --variant base --full_run

Reconcile size mismatches only (dry-run: no model needed):
    python runners/run_collect_activations.py --variant base --reconcile --dry_run

Reconcile size mismatches only (loads model, applies fixes):
    python runners/run_collect_activations.py --variant base --reconcile

List what is still missing:
    python runners/run_collect_activations.py --variant base --list_missing

Extended variant with a specific country:
    python runners/run_collect_activations.py \\
        --variant extended --country Germany --full_run
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Make project root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hpc import get_offload_dir, setup_cuda_env
from runners.experiment_definitions import (
    get_all_conditions,
    get_all_result_paths,
    get_missing_conditions,
    get_result_path,
    get_manifest_path,
    set_extended_country,
    _VARIANT_DIRS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = (
    "/anvme/workspace/iwi5268h-vision-transformers/hpc_infrastructure/hf_cache"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect activations for gender × emotion persona conditions."
    )
    p.add_argument(
        "--variant", choices=["base", "extended"], default="base",
        help="Experimental variant to run (default: base).",
    )
    p.add_argument(
        "--persona_key", default=None,
        help="Run a single condition by key (e.g. 'female_anger'). "
             "Mutually exclusive with --all.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Run all missing conditions for the variant sequentially.",
    )
    p.add_argument(
        "--full_run", action="store_true",
        help=(
            "Load the model once, fix any vision mismatches (reconcile), "
            "then collect all missing conditions.  Recommended for HPC runs."
        ),
    )
    p.add_argument(
        "--reconcile", action="store_true",
        help="Detect and fix cross-condition image-size mismatches (then exit).",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="With --reconcile or --full_run: report mismatches without re-running images.",
    )
    p.add_argument(
        "--list_missing", action="store_true",
        help="Print conditions not yet collected and exit.",
    )
    p.add_argument(
        "--country", default=None,
        help="Country for the extended variant (e.g. 'Germany'). "
             "Required when --variant extended.",
    )
    p.add_argument(
        "--manifest", default=None,
        help="Path to image manifest .pkl. Defaults to data/selected_uniform_500_manifest.pkl.",
    )
    p.add_argument(
        "--model_path", default=_DEFAULT_MODEL_PATH,
        help="Path to HuggingFace model directory.",
    )
    p.add_argument(
        "--offload_suffix", default="",
        help="Suffix appended to the CPU offload directory name (for parallel jobs).",
    )
    p.add_argument("--checkpoint_every", type=int, default=2)
    p.add_argument("--max_retry_rounds", type=int, default=5)
    return p.parse_args()


def _maybe_set_country(args: argparse.Namespace) -> None:
    if args.variant == "extended":
        if not args.country:
            logger.error(
                "Extended variant requires --country (e.g. --country Germany)."
            )
            sys.exit(1)
        set_extended_country(args.country)


def _collect_one(
    persona_key: str,
    persona_dict: dict | None,
    variant: str,
    args: argparse.Namespace,
    model_and_processor=None,
) -> None:
    """Run collection for a single condition, skipping if already complete."""
    from representation.I1_contrast_design.collect import run_experiment

    output_dir  = _VARIANT_DIRS[variant]
    result_path = get_result_path(persona_key, variant)

    if result_path.exists():
        import numpy as np
        existing = np.load(result_path, allow_pickle=True).item()
        n_done   = len(existing.get("results", []))
        logger.info(f"[skip] {persona_key}: already has {n_done} results at {result_path}")
        return

    manifest = Path(args.manifest) if args.manifest else get_manifest_path()
    offload  = get_offload_dir(
        base=Path(args.model_path).parent / "offload_dir",
        suffix=args.offload_suffix or persona_key,
    )

    logger.info(f"Starting collection: {persona_key} → {output_dir}")

    run_experiment(
        model_path=args.model_path,
        persona_key=persona_key,
        persona_dict=persona_dict,
        manifest_path=manifest,
        output_dir=output_dir,
        offload_dir=str(offload) if offload else None,
        checkpoint_every=args.checkpoint_every,
        max_retry_rounds=args.max_retry_rounds,
        model_and_processor=model_and_processor,
    )


def _run_reconcile(variant: str, args: argparse.Namespace) -> None:
    """Detect and fix cross-condition image-size mismatches for a variant."""
    from representation.I1_contrast_design.reconcile import (
        analyze_size_mismatches, print_mismatch_report,
        analyze_vision_mismatches, print_vision_mismatch_report,
        reconcile_by_vision, reconcile_conditions,
    )

    all_paths      = get_all_result_paths(variant)
    existing_paths = {pk: p for pk, p in all_paths.items() if p.exists()}

    if len(existing_paths) < 2:
        logger.info("Need at least 2 collected conditions to reconcile. Nothing to do.")
        return

    conditions    = get_all_conditions(variant)
    persona_dicts = {pk: conditions.get(pk) for pk in existing_paths}

    # Detect whether records have image_max_size (new format) or not (old format)
    import numpy as np
    sample_pk   = next(iter(existing_paths))
    sample_recs = np.load(existing_paths[sample_pk], allow_pickle=True).item().get("results", [])
    has_size_field = any("image_max_size" in r for r in sample_recs[:5])

    if has_size_field:
        # New-format records: use stored sizes
        if args.dry_run:
            mismatches = analyze_size_mismatches(existing_paths)
            print_mismatch_report(mismatches)
            return
        reconcile_fn = reconcile_conditions
    else:
        # Old-format records: compare vision activations directly
        logger.info(
            "Records predate image_max_size tracking — "
            "using vision-activation comparison to detect mismatches."
        )
        if args.dry_run:
            mismatches = analyze_vision_mismatches(existing_paths)
            print_vision_mismatch_report(mismatches)
            return
        reconcile_fn = reconcile_by_vision

    from utils.model_loader import load_model_and_processor
    setup_cuda_env()
    offload = get_offload_dir(
        base=Path(args.model_path).parent / "offload_dir",
        suffix=args.offload_suffix or "reconcile",
    )
    logger.info("Loading model for reconciliation…")
    model, processor = load_model_and_processor(
        args.model_path, offload_dir=str(offload) if offload else None
    )

    counts = reconcile_fn(
        result_paths=existing_paths,
        persona_dicts=persona_dicts,
        model=model,
        processor=processor,
        dry_run=False,
    )
    logger.info(f"Reconciliation counts: {counts}")


def _load_model(args: argparse.Namespace):
    from utils.model_loader import load_model_and_processor
    setup_cuda_env()
    offload = get_offload_dir(
        base=Path(args.model_path).parent / "offload_dir",
        suffix=args.offload_suffix or "main",
    )
    logger.info("Loading model…")
    return load_model_and_processor(
        args.model_path, offload_dir=str(offload) if offload else None
    )


def _run_full(variant: str, args: argparse.Namespace) -> None:
    """
    Single-model-load session:
      1. Detect and fix vision mismatches in already-collected conditions.
      2. Collect all missing conditions.
    """
    import numpy as np
    from representation.I1_contrast_design.reconcile import (
        analyze_vision_mismatches, print_vision_mismatch_report,
        analyze_size_mismatches, print_mismatch_report,
        reconcile_by_vision, reconcile_conditions,
    )

    all_paths      = get_all_result_paths(variant)
    existing_paths = {pk: p for pk, p in all_paths.items() if p.exists()}
    missing        = get_missing_conditions(variant)
    conditions     = get_all_conditions(variant)

    logger.info(
        f"Full run — variant='{variant}': "
        f"{len(existing_paths)} collected, {len(missing)} missing"
    )

    # --- Step 1: dry-run mismatch analysis (no model needed) ---
    if existing_paths:
        sample_recs = np.load(
            next(iter(existing_paths.values())), allow_pickle=True
        ).item().get("results", [])
        has_size_field = any("image_max_size" in r for r in sample_recs[:5])

        if has_size_field:
            mm = analyze_size_mismatches(existing_paths)
            print_mismatch_report(mm)
        else:
            mm = analyze_vision_mismatches(existing_paths)
            print_vision_mismatch_report(mm)
    else:
        mm = {}

    need_reconcile = bool(mm)
    need_collect   = bool(missing)

    if not need_reconcile and not need_collect:
        logger.info("Nothing to do — all conditions collected and consistent.")
        return

    if args.dry_run:
        if need_collect:
            logger.info(f"Would collect {len(missing)} missing conditions: {missing}")
        return

    # --- Load model once ---
    model, processor = _load_model(args)
    mp = (model, processor)

    # --- Step 2: reconcile mismatches ---
    if need_reconcile and existing_paths:
        persona_dicts = {pk: conditions.get(pk) for pk in existing_paths}
        if has_size_field:
            reconcile_conditions(
                result_paths=existing_paths,
                persona_dicts=persona_dicts,
                model=model, processor=processor,
            )
        else:
            reconcile_by_vision(
                result_paths=existing_paths,
                persona_dicts=persona_dicts,
                model=model, processor=processor,
            )

    # --- Step 3: collect missing conditions ---
    if need_collect:
        logger.info(f"Collecting {len(missing)} missing conditions: {missing}")
        for pk in missing:
            _collect_one(pk, conditions[pk], variant, args, model_and_processor=mp)


def main() -> None:
    args = parse_args()
    _maybe_set_country(args)

    if args.list_missing:
        missing = get_missing_conditions(args.variant)
        if missing:
            print(f"Missing conditions for variant '{args.variant}' ({len(missing)}):")
            for pk in missing:
                print(f"  {pk}")
        else:
            print(f"All conditions for variant '{args.variant}' are collected.")
        return

    if args.full_run:
        _run_full(args.variant, args)
        return

    if args.reconcile:
        _run_reconcile(args.variant, args)
        return

    conditions = get_all_conditions(args.variant)

    if args.persona_key and args.all:
        logger.error("--persona_key and --all are mutually exclusive.")
        sys.exit(1)

    if args.persona_key:
        if args.persona_key not in conditions:
            logger.error(
                f"Unknown persona_key '{args.persona_key}' for variant '{args.variant}'. "
                f"Valid keys: {sorted(conditions)}"
            )
            sys.exit(1)
        setup_cuda_env()
        _collect_one(args.persona_key, conditions[args.persona_key], args.variant, args)

    elif args.all:
        missing = get_missing_conditions(args.variant)
        if not missing:
            logger.info(f"All conditions for variant '{args.variant}' already collected.")
            return
        logger.info(
            f"Running {len(missing)} missing conditions for variant '{args.variant}': {missing}"
        )
        setup_cuda_env()
        model, processor = _load_model(args)
        for pk in missing:
            _collect_one(pk, conditions[pk], args.variant, args, model_and_processor=(model, processor))

    else:
        logger.error(
            "Specify one of: --persona_key <key> | --all | --full_run | "
            "--reconcile | --list_missing"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
