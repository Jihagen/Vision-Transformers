"""
Runner: behavioral evaluation of UAP deltas on the interestingness rating task.

Uses the 100 held-out eval-split images (every 5th position in the manifest,
UAP eval set, never used for training the delta) together with their existing
clean interestingness labels from the original study.

Only model.generate() calls for perturbed conditions are made — the clean
baseline comes from data/selected_uniform_total_500.pkl and is free.

Per condition output (in --output_dir):
    <condition>_labels.csv    — per-image: filename, clean_label, clean_score,
                                perturbed_label, perturbed_score, score_delta
    summary.csv               — mean_clean_score, mean_perturbed_score,
                                mean_score_delta per condition

Usage
-----
Smoke test (10 images, one delta):
    python attack/runners/run_behavioral_eval.py --smoke_test \\
        --delta_dirs results/universal_perturbation/interest/eps2.00

Full run (100 eval images, all epsilons for both attacks):
    python attack/runners/run_behavioral_eval.py \\
        --delta_dirs \\
            results/universal_perturbation/interest/eps0.10 \\
            results/universal_perturbation/interest/eps0.50 \\
            results/universal_perturbation/interest/eps1.00 \\
            results/universal_perturbation/interest/eps2.00 \\
            results/universal_perturbation/excited_vs_angry/eps0.10 \\
            results/universal_perturbation/excited_vs_angry/eps0.50 \\
            results/universal_perturbation/excited_vs_angry/eps1.00 \\
            results/universal_perturbation/excited_vs_angry/eps2.00 \\
        --output_dir results/attack_eval/interestingness
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
_MANIFEST_PATH  = Path("data/selected_uniform_500_manifest.pkl")
_LABELS_PATH    = Path("data/selected_uniform_total_500.pkl")


def _load_eval_images() -> "pd.DataFrame":
    """
    Return the 100 eval-split images with their existing clean labels.
    Eval split: every 5th manifest position (indices 4, 9, 14, ...) —
    same mask used during UAP training, so these images were never seen
    by the delta optimizer.
    """
    import pandas as pd
    manifest = pd.read_pickle(_MANIFEST_PATH)
    labels   = pd.read_pickle(_LABELS_PATH)[["filename", "interestingness_label"]]
    eval_mask = [i % 5 == 4 for i in range(len(manifest))]
    eval_df   = manifest[eval_mask].reset_index(drop=True)
    eval_df   = eval_df.merge(labels, on="filename", how="left")
    assert eval_df["interestingness_label"].notna().all(), \
        "Some eval images are missing clean labels — check selected_uniform_total_500.pkl"
    return eval_df


def _condition_name(delta_dir: Path) -> str:
    """
    results/universal_perturbation/interest/eps2.00      → interest_eps2.00
    results/universal_perturbation/excited_vs_angry/eps1 → excited_vs_angry_eps1.00
    """
    parts = delta_dir.parts
    return f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else delta_dir.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Behavioral eval of UAP deltas on the original interestingness task."
    )
    p.add_argument(
        "--delta_dirs", nargs="+", default=[], metavar="DIR",
        help="Directories each containing a delta.npy to evaluate.",
    )
    p.add_argument("--output_dir", default="results/attack_eval/interestingness")
    p.add_argument("--max_side", type=int, default=336,
                   help="Must match max_side used during UAP training (default 336).")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--model_path", default=_DEFAULT_MODEL_PATH)
    p.add_argument("--offload_suffix", default="beval")
    p.add_argument("--smoke_test", action="store_true",
                   help="Use only 10 eval images for a quick sanity check.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pandas as pd
    from attack.tasks.interestingness_eval import rate_images, LABEL_TO_SCORE

    # ── Load eval images + existing clean labels ──────────────────────────────
    eval_df = _load_eval_images()
    if args.smoke_test:
        eval_df = eval_df.head(10)
    logger.info(f"Eval images: {len(eval_df)} (clean labels loaded from study data)")

    image_paths = eval_df["img_path"].tolist()
    clean_labels = dict(zip(eval_df["filename"], eval_df["interestingness_label"]))

    if not args.delta_dirs:
        raise ValueError("Provide at least one --delta_dirs path.")

    # Validate and load all deltas before touching the model
    delta_conditions: list[tuple[str, np.ndarray]] = []
    for d in args.delta_dirs:
        delta_path = Path(d) / "delta.npy"
        if not delta_path.exists():
            raise FileNotFoundError(f"delta.npy not found: {delta_path}")
        name = _condition_name(Path(d))
        delta_np = np.load(delta_path).astype(np.float32)
        delta_conditions.append((name, delta_np))
        logger.info(f"  Loaded '{name}': shape={delta_np.shape}, "
                    f"|delta|_inf={np.abs(delta_np).max():.4f}")

    # ── Load model (inference only — no tight memory caps, lm_head on GPU) ───
    from utils.model_loader import load_model_and_processor
    from utils.hpc import get_offload_dir, setup_cuda_env
    from utils.prompt_builder import build_blank_prompt
    import torch

    setup_cuda_env()
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "38GiB" for i in range(n_gpus)} if n_gpus >= 6 else None

    offload = get_offload_dir(
        base=Path(args.model_path).parent / "offload_dir",
        suffix=args.offload_suffix,
    )
    logger.info(f"Loading model (max_memory={max_memory})…")
    model, processor = load_model_and_processor(
        args.model_path,
        offload_dir=str(offload) if offload else None,
        max_memory=max_memory,
    )
    model.requires_grad_(False)
    model.eval()

    prompt = build_blank_prompt()

    # ── Evaluate each delta condition ─────────────────────────────────────────
    all_label_values = list(LABEL_TO_SCORE.keys())
    summary_rows = []

    for condition, delta_np in delta_conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Condition: {condition}  ({len(image_paths)} images)")
        logger.info(f"{'='*60}")

        perturbed_results = rate_images(
            model=model,
            processor=processor,
            image_paths=image_paths,
            prompt=prompt,
            delta_np=delta_np,
            max_side=args.max_side,
            max_new_tokens=args.max_new_tokens,
        )

        rows = []
        for r in perturbed_results:
            fname = r["filename"]
            clean_lbl = clean_labels.get(fname)
            clean_score = LABEL_TO_SCORE.get(clean_lbl) if clean_lbl else None
            pert_score = r["score"]
            score_delta = (pert_score - clean_score) if (pert_score is not None and clean_score is not None) else None
            rows.append({
                "filename":        fname,
                "clean_label":     clean_lbl,
                "clean_score":     clean_score,
                "perturbed_label": r["label"],
                "perturbed_score": pert_score,
                "score_delta":     score_delta,
                "parse_ok":        r["parse_ok"],
            })

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{condition}_labels.csv", index=False)

        parsed = df[df["parse_ok"] == True]
        mean_clean    = parsed["clean_score"].mean()
        mean_perturbed = parsed["perturbed_score"].mean()
        mean_delta    = parsed["score_delta"].mean()
        dist = {lbl: int((parsed["perturbed_label"] == lbl).sum()) for lbl in all_label_values}

        row = {
            "condition":          condition,
            "n_parsed":           len(parsed),
            "mean_clean_score":   round(mean_clean, 4),
            "mean_perturbed_score": round(mean_perturbed, 4),
            "mean_score_delta":   round(mean_delta, 4),
        }
        row.update({f"n_{lbl.replace(' ', '_')}": v for lbl, v in dist.items()})
        summary_rows.append(row)

        logger.info(f"  clean={mean_clean:.3f}  perturbed={mean_perturbed:.3f}  "
                    f"Δscore={mean_delta:+.3f}  parsed={len(parsed)}/{len(rows)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print("\n" + "=" * 60)
    print("Behavioral eval complete:")
    cols = ["condition", "mean_clean_score", "mean_perturbed_score", "mean_score_delta", "n_parsed"]
    print(summary_df[cols].to_string(index=False))
    print(f"\n✅ Results → {out_dir}")


if __name__ == "__main__":
    main()
