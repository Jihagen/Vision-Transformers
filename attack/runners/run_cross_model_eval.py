"""
Runner: cross-model transfer — evaluate UAPs that were trained on Llama 4 Scout
on a different vision-language model, on the Generalisation Level 1 tasks.

Same frozen sample manifests, prompts, parser and summary tables as
run_generalisation_eval.py. Only the model changes; see
attack/IV2_generalisation/cross_model.py for how the perturbation is handed over.

Usage
-----
    python attack/runners/run_cross_model_eval.py \\
        --model_path hpc_infrastructure/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/local-repo \\
        --model_name qwen2.5-vl-7b --tasks damage_severity
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

from utils.paths import LOCAL_MODEL_REPO as _LLAMA_MODEL_PATH
_ALL_TASKS = ["shopping_relevance", "damage_severity", "moral_evaluation"]
# The workload UAP plus its two magnitude-matched controls.
_DEFAULT_ROOTS = {
    "workload": "results/universal_perturbation_projected/workload",
    "shuffled_workload": "results/universal_perturbation_controls/shuffled_workload",
    "noise_seed0": "results/universal_perturbation_controls/noise_seed0",
    "noise_seed1": "results/universal_perturbation_controls/noise_seed1",
    "noise_seed2": "results/universal_perturbation_controls/noise_seed2",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-model UAP transfer evaluation.")
    p.add_argument("--model_path", required=True, help="Local directory of the second model.")
    p.add_argument("--model_name", required=True, help="Short name used for the output folder.")
    p.add_argument("--tasks", nargs="+", choices=_ALL_TASKS, default=["damage_severity", "moral_evaluation"])
    p.add_argument("--output_dir", default="results/generalisation_cross_model")
    p.add_argument("--llama_processor_path", default=_LLAMA_MODEL_PATH,
                   help="Llama 4 Scout directory; only its image processor is loaded.")
    p.add_argument("--uap_roots", nargs="+", default=None, metavar="NAME=DIR")
    p.add_argument("--epsilons", nargs="+", type=float, default=None,
                   help="Budgets to evaluate (default: 0.1 0.5 1.0 2.0).")
    p.add_argument("--max_side", type=int, default=336,
                   help="Must match max_side used during UAP training (default 336).")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=None, help="Cap samples per task.")
    p.add_argument("--clean_only", action="store_true",
                   help="Competence check: unperturbed images only, writes predictions.csv per task.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir) / args.model_name

    from attack.IV2_generalisation import datasets as ds
    from attack.IV2_generalisation import evaluate as ev
    from attack.IV2_generalisation import analyse as an
    from attack.IV2_generalisation.cross_model import make_generate_fn
    from attack.IV2_generalisation.perturbation import discover_uap_conditions
    from attack.IV2_generalisation.tasks import TASKS
    from attack.IV2_generalisation.metrics import budget_response

    roots = dict(r.split("=", 1) for r in args.uap_roots) if args.uap_roots else _DEFAULT_ROOTS
    uap_conditions = [] if args.clean_only else discover_uap_conditions(roots, args.epsilons)
    logger.info(f"Perturbation conditions: {[(a, e) for a, e, _ in uap_conditions]}")
    if not uap_conditions and not args.clean_only:
        raise RuntimeError("No delta.npy files found for the requested perturbation sets and budgets")
    manifests = {t: ds.load_manifest(t) for t in args.tasks}

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoProcessor

    llama_image_processor = AutoImageProcessor.from_pretrained(args.llama_processor_path, local_files_only=True)
    logger.info(f"Loading {args.model_name} from {args.model_path} …")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model.requires_grad_(False)
    model.eval()

    conditions = ev.build_conditions(uap_conditions)
    generate_fn = make_generate_fn(llama_image_processor)

    summaries = {}
    for task_name, manifest in manifests.items():
        task = TASKS[task_name]
        out_dir = out_root / task_name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Task {task_name}: {len(manifest)} samples x {len(conditions)} conditions")
        predictions = ev.evaluate_task(
            model, processor, task, manifest, conditions, max_side=args.max_side,
            max_new_tokens=args.max_new_tokens, limit=args.limit, generate_fn=generate_fn,
        )
        if args.clean_only:
            predictions.insert(0, "model", args.model_name)
            predictions.to_csv(out_dir / "predictions.csv", index=False)
            logger.info(f"clean label distribution:\n{predictions.model_label.value_counts().sort_index()}")
            continue
        if task_name == "shopping_relevance":
            predictions = an.add_shopping_reference_metrics(predictions)
        elif task_name == "damage_severity":
            predictions = an.add_medic_gt_metrics(predictions)
            an.medic_gt_summary(predictions).to_csv(out_dir / "gt_disagreement_summary.csv", index=False)
        elif task_name == "moral_evaluation":
            predictions = an.add_smid_bias_metrics(predictions)
        predictions.insert(0, "model", args.model_name)
        predictions.to_csv(out_dir / "predictions.csv", index=False)

        summary = an.build_summary(predictions, task)
        summary.to_csv(out_dir / "summary.csv", index=False)
        budget_response(summary).to_csv(out_dir / "budget_response.csv", index=False)
        summaries[task_name] = summary
        logger.info(f"\n{summary.to_string(index=False)}")

    if args.clean_only:
        print(f"\n✅ Clean predictions -> {out_root}")
        return
    # One file per job: per-task jobs for the same model run in parallel.
    an.cross_task_matrix(summaries).to_csv(
        out_root / f"normalized_effect_{'_'.join(sorted(summaries))}.csv", index=False)
    print(f"\n✅ Results -> {out_root}")


if __name__ == "__main__":
    main()
