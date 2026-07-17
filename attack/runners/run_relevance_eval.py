"""
Runner: behavioral evaluation of UAP deltas on the binary relevance task.

"Is this image relevant to you?" — removes the "interestingness" token entirely.
Tests whether the attack is token-level or semantic: a purely token-level attack
should fail to transfer here; a semantic attack on the model's engagement state
should still shift the yes/no distribution.

Unlike the interestingness runner, no pre-existing clean labels exist for this
task, so the clean baseline is always run first (one model.generate() pass).

Output (in --output_dir):
    clean_labels.csv              — per-image: filename, relevant, score, parse_ok
    <condition>_labels.csv        — same schema, one file per delta condition
    summary.csv                   — p_yes and mean_score_delta per condition
                                    (delta relative to clean baseline on same images)

Usage
-----
Smoke test:
    python attack/runners/run_relevance_eval.py --smoke_test \\
        --delta_dirs results/universal_perturbation/interest/eps2.00

Full run:
    python attack/runners/run_relevance_eval.py \\
        --delta_dirs \\
            results/universal_perturbation/interest/eps0.10 \\
            results/universal_perturbation/interest/eps0.50 \\
            results/universal_perturbation/interest/eps1.00 \\
            results/universal_perturbation/interest/eps2.00 \\
            results/universal_perturbation/excited_vs_angry/eps0.10 \\
            results/universal_perturbation/excited_vs_angry/eps0.50 \\
            results/universal_perturbation/excited_vs_angry/eps1.00 \\
            results/universal_perturbation/excited_vs_angry/eps2.00 \\
        --output_dir results/attack_eval/relevance
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
_MANIFEST_PATH = Path("data/selected_uniform_500_manifest.pkl")


def _load_eval_images() -> "pd.DataFrame":
    """100 eval-split images — every 5th manifest position."""
    import pandas as pd
    manifest = pd.read_pickle(_MANIFEST_PATH)
    eval_mask = [i % 5 == 4 for i in range(len(manifest))]
    return manifest[eval_mask].reset_index(drop=True)


def _condition_name(delta_dir: Path) -> str:
    parts = delta_dir.parts
    return f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else delta_dir.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Behavioral eval of UAP deltas on the binary relevance task."
    )
    p.add_argument("--delta_dirs", nargs="+", default=[], metavar="DIR")
    p.add_argument("--output_dir", default="results/attack_eval/relevance")
    p.add_argument("--max_side", type=int, default=336)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--model_path", default=_DEFAULT_MODEL_PATH)
    p.add_argument("--offload_suffix", default="reval")
    p.add_argument("--smoke_test", action="store_true",
                   help="Use only 10 eval images.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pandas as pd
    from attack.tasks.relevance_eval import rate_relevance, RELEVANT_TO_SCORE

    eval_df = _load_eval_images()
    if args.smoke_test:
        eval_df = eval_df.head(10)
    image_paths = eval_df["img_path"].tolist()
    logger.info(f"Eval images: {len(image_paths)}")

    if not args.delta_dirs:
        raise ValueError("Provide at least one --delta_dirs path.")

    delta_conditions: list[tuple[str, np.ndarray]] = []
    for d in args.delta_dirs:
        delta_path = Path(d) / "delta.npy"
        if not delta_path.exists():
            raise FileNotFoundError(f"delta.npy not found: {delta_path}")
        name = _condition_name(Path(d))
        delta_np = np.load(delta_path).astype(np.float32)
        delta_conditions.append((name, delta_np))
        logger.info(f"  Loaded '{name}': shape={delta_np.shape}")

    # ── Load model ────────────────────────────────────────────────────────────
    from utils.model_loader import load_model_and_processor
    from utils.hpc import get_offload_dir, setup_cuda_env
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

    # ── Clean baseline (no delta) ─────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  Condition: clean  ({len(image_paths)} images)")
    logger.info(f"{'='*60}")
    clean_results = rate_relevance(
        model=model, processor=processor, image_paths=image_paths,
        delta_np=None, max_side=args.max_side, max_new_tokens=args.max_new_tokens,
    )
    clean_df = pd.DataFrame(clean_results)
    clean_df.to_csv(out_dir / "clean_labels.csv", index=False)
    clean_scores = {r["filename"]: r["score"] for r in clean_results}
    clean_p_yes = clean_df[clean_df["parse_ok"] == True]["score"].mean()
    logger.info(f"  clean p(yes)={clean_p_yes:.3f}  parsed={clean_df['parse_ok'].sum()}/{len(clean_df)}")

    summary_rows = [{"condition": "clean", "p_yes": round(clean_p_yes, 4),
                     "mean_score_delta": 0.0,
                     "n_parsed": int(clean_df["parse_ok"].sum())}]

    # ── Perturbed conditions ──────────────────────────────────────────────────
    for condition, delta_np in delta_conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Condition: {condition}  ({len(image_paths)} images)")
        logger.info(f"{'='*60}")

        pert_results = rate_relevance(
            model=model, processor=processor, image_paths=image_paths,
            delta_np=delta_np, max_side=args.max_side, max_new_tokens=args.max_new_tokens,
        )

        rows = []
        for r in pert_results:
            cs = clean_scores.get(r["filename"])
            ps = r["score"]
            rows.append({
                "filename":      r["filename"],
                "clean_score":   cs,
                "perturbed_relevant": r["relevant"],
                "perturbed_score":    ps,
                "score_delta":   (ps - cs) if (ps is not None and cs is not None) else None,
                "parse_ok":      r["parse_ok"],
                "explanation":   r.get("explanation"),
                "raw_response":  r.get("raw_response"),
            })

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{condition}_labels.csv", index=False)

        parsed = df[df["parse_ok"] == True]
        p_yes = parsed["perturbed_score"].mean()
        mean_delta = parsed["score_delta"].mean()
        summary_rows.append({
            "condition": condition,
            "p_yes": round(p_yes, 4),
            "mean_score_delta": round(mean_delta, 4),
            "n_parsed": len(parsed),
        })
        logger.info(f"  p(yes)={p_yes:.3f}  Δp(yes)={mean_delta:+.3f}  parsed={len(parsed)}/{len(rows)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print("\n" + "=" * 60)
    print("Relevance eval complete:")
    print(summary_df[["condition", "p_yes", "mean_score_delta", "n_parsed"]].to_string(index=False))
    print(f"\n✅ Results → {out_dir}")


if __name__ == "__main__":
    main()
