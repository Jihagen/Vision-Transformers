"""
Runner: Generalisation Level 1 — evaluate the already-trained UAPs
(interest, excited_vs_angry, and workload/stress once available) on three
unseen out-of-domain tasks: shopping relevance (Marqo-GS-10M), moral
evaluation (SMID), and damage severity (QCRI/MEDIC).

No UAP training or scaling happens here — see attack/IV2_generalisation/
perturbation.py, which loads already-saved delta.npy files and applies them
with the exact same clamp/pixel-space mechanics as
attack/tasks/interestingness_eval.py and attack/tasks/relevance_eval.py.

Sample manifests must already exist (run attack/IV2_generalisation/
prepare_datasets.py on a login node first — it needs internet).

Usage
-----
Smoke test (10 samples/task, one UAP condition, all available tasks):
    python attack/runners/run_generalisation_eval.py --smoke_test

Full run (all tasks with a frozen manifest, all available UAP conditions):
    python attack/runners/run_generalisation_eval.py \\
        --tasks shopping_relevance damage_severity moral_evaluation \\
        --output_dir results/generalisation
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

_DEFAULT_MODEL_PATH = (
    "/anvme/workspace/iwi5268h-vision-transformers/hpc_infrastructure/hf_cache/"
    "models--meta-llama--Llama-4-Scout-17B-16E-Instruct/local-repo"
)
_ALL_TASKS = ["shopping_relevance", "damage_severity", "moral_evaluation"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generalisation Level 1 evaluation.")
    p.add_argument("--tasks", nargs="+", choices=_ALL_TASKS, default=_ALL_TASKS)
    p.add_argument("--output_dir", default="results/generalisation")
    p.add_argument("--max_side", type=int, default=336,
                   help="Must match max_side used during UAP training (default 336).")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--model_path", default=_DEFAULT_MODEL_PATH)
    p.add_argument("--offload_suffix", default="genlvl1")
    p.add_argument("--smoke_test", action="store_true",
                   help="10 samples/task, only the eps=2.00 interest condition + clean.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap samples per task (overridden to 10 by --smoke_test).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)

    from attack.IV2_generalisation import datasets as ds
    from attack.IV2_generalisation import evaluate as ev
    from attack.IV2_generalisation import analyse as an
    from attack.IV2_generalisation.perturbation import discover_uap_conditions
    from attack.IV2_generalisation.tasks import TASKS
    from attack.IV2_generalisation.metrics import budget_response

    limit = 10 if args.smoke_test else args.limit
    uap_conditions = discover_uap_conditions()
    if args.smoke_test:
        uap_conditions = [c for c in uap_conditions if c[0] == "interest" and c[1] == 2.0]
    logger.info(f"UAP conditions available: {[(a, e) for a, e, _ in uap_conditions]}")
    if not uap_conditions:
        raise RuntimeError("No UAP delta.npy files found under results/universal_perturbation_projected/*")

    # Validate manifests exist before loading the (expensive) model.
    tasks_to_run = []
    for task_name in args.tasks:
        try:
            manifest = ds.load_manifest(task_name)
            tasks_to_run.append((task_name, manifest))
            logger.info(f"  {task_name}: manifest with {len(manifest)} samples")
        except FileNotFoundError as e:
            logger.warning(f"  {task_name}: SKIPPED — {e}")
    if not tasks_to_run:
        raise RuntimeError("No task has a frozen sample_manifest.csv. Run prepare_datasets.py first.")

    # ── Load model ────────────────────────────────────────────────────────
    from utils.model_loader import load_model_and_processor
    from utils.hpc import get_offload_dir, setup_cuda_env
    import torch

    setup_cuda_env()
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "38GiB" for i in range(n_gpus)} if n_gpus >= 6 else None

    offload = get_offload_dir(base=Path(args.model_path).parent / "offload_dir", suffix=args.offload_suffix)
    logger.info(f"Loading model (max_memory={max_memory})…")
    model, processor = load_model_and_processor(
        args.model_path, offload_dir=str(offload) if offload else None, max_memory=max_memory,
    )
    model.requires_grad_(False)
    model.eval()

    conditions = ev.build_conditions(uap_conditions)

    summaries = {}
    for task_name, manifest in tasks_to_run:
        task = TASKS[task_name]
        out_dir = out_root / task_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}\n  Task: {task_name}  ({len(manifest) if not limit else min(limit, len(manifest))} samples x {len(conditions)} conditions)\n{'='*60}")
        predictions = ev.evaluate_task(
            model, processor, task, manifest, conditions,
            max_side=args.max_side, max_new_tokens=args.max_new_tokens, limit=limit,
        )
        predictions.to_csv(out_dir / "predictions.csv", index=False)
        logger.info(f"  Wrote {len(predictions)} prediction rows -> {out_dir / 'predictions.csv'}")

        # ── task-specific enrichment + summary ──────────────────────────
        plots_dir = out_dir / "plots"
        if task_name == "shopping_relevance":
            predictions = an.add_shopping_reference_metrics(predictions)
            predictions.to_csv(out_dir / "predictions.csv", index=False)
        elif task_name == "damage_severity":
            predictions = an.add_medic_gt_metrics(predictions)
            predictions.to_csv(out_dir / "predictions.csv", index=False)
            an.medic_gt_summary(predictions).to_csv(out_dir / "gt_disagreement_summary.csv", index=False)
        elif task_name == "moral_evaluation":
            predictions = an.add_smid_bias_metrics(predictions)
            predictions.to_csv(out_dir / "predictions.csv", index=False)
            arousal_result = an.smid_arousal_analysis(predictions)
            if isinstance(arousal_result, tuple):
                corr_df, reg_df = arousal_result
                corr_df.to_csv(out_dir / "arousal_analysis.csv", index=False)
                reg_df.to_csv(out_dir / "arousal_regression.csv", index=False)
            else:
                arousal_result.to_csv(out_dir / "arousal_analysis.csv", index=False)
            an.plot_arousal_stratified(predictions, plots_dir)

        summary = an.build_summary(predictions, task)
        summary.to_csv(out_dir / "summary.csv", index=False)
        summaries[task_name] = summary

        budget_response(summary).to_csv(out_dir / "budget_response.csv", index=False)
        an.plot_budget_response(summary, plots_dir, title=task_name)

        logger.info(f"\n{summary.to_string(index=False) if not summary.empty else '(empty summary)'}")

    cross = an.cross_task_matrix(summaries)
    cross.to_csv(out_root / "cross_task_normalized_effect.csv", index=False)

    print("\n" + "=" * 60)
    print("Generalisation Level 1 evaluation complete:")
    for task_name in summaries:
        print(f"  {task_name}: results/generalisation/{task_name}/")
    print(f"\n✅ Results -> {out_root}")


if __name__ == "__main__":
    main()
