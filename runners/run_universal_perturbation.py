"""
Runner: TIER IV — Universal Adversarial Perturbation (UAP) for interestingness.

Finds a single pixel-space delta that, when added to any image, maximises
cos(h(image+delta) @ language_29, v_interest_blank) — i.e. pushes every image
toward the "very interesting" pole of the interestingness axis.

Unlike per-image gradient matching, the delta here is SHARED across all
training images.  We sweep epsilon ∈ {0.1, 0.5, 1.0, 2.0} (processor-
normalised units) to explore the trade-off between perturbation size and
rating shift.

Results per epsilon (in results/universal_perturbation/<layer>/eps<epsilon>/):
    delta.npy                 — the universal perturbation (ref_shape float32)
    uap_eval.csv              — per held-out-image cos and rating shift
    uap_summary.csv           — aggregate: mean cos_delta, mean rating shift
    delta_eps*_epoch*.npy     — intermediate deltas per epoch (if --save_epochs)

Usage
-----
Dry-run:
    python runners/run_universal_perturbation.py --dry_run

Smoke-test (5 train + 2 eval images, 2 epochs, one epsilon):
    python runners/run_universal_perturbation.py --smoke_test

Full run (defaults: 400 train, 100 eval, 5 epochs, epsilons 0.1 0.5 1.0 2.0):
    python runners/run_universal_perturbation.py
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
_DEFAULT_EPSILONS = [0.1, 0.5, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TIER IV: Universal Adversarial Perturbation for interestingness."
    )
    p.add_argument("--vector_path", default=_DEFAULT_VECTOR)
    p.add_argument("--layer", default="language_29_D5120")
    p.add_argument("--epsilons", nargs="+", type=float, default=_DEFAULT_EPSILONS,
                   help="L_inf epsilon values to sweep (processor-normalised units). "
                        "Default: 0.1 0.5 1.0 2.0")
    p.add_argument("--lr", type=float, default=None,
                   help="FGSM step size per epoch. Default: epsilon/n_epochs, so the "
                        "delta exactly fills its L_inf budget after n_epochs steps.")
    p.add_argument("--n_epochs", type=int, default=20,
                   help="Passes over training images per epsilon (default 20).")
    p.add_argument("--n_train", type=int, default=400,
                   help="Number of training images (default 400).")
    p.add_argument("--n_eval", type=int, default=100,
                   help="Number of eval images (default 100).")
    p.add_argument("--max_side", type=int, default=336,
                   help="Max image side before processor (default 336 → 1-2 tiles, "
                        "consistent pixel_values shape; lower = less GPU 0 memory).")
    p.add_argument("--manifest", default=None,
                   help="Path to manifest .pkl (default: data/selected_uniform_500_manifest.pkl).")
    p.add_argument("--model_path", default=_DEFAULT_MODEL_PATH)
    p.add_argument("--offload_suffix", default="uap")
    p.add_argument("--output_dir", default=None,
                   help="Output root (default: results/universal_perturbation/<layer>).")
    p.add_argument("--save_epochs", action="store_true",
                   help="Save delta after every epoch (not just final).")
    p.add_argument("--smoke_test", action="store_true",
                   help="5 train + 2 eval images, 2 epochs, epsilon=[0.5] only.")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate paths/vector/imports without loading model.")
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    layer_slug = args.layer.replace("_D5120", "").replace("_D1408", "")
    return Path("results/universal_perturbation") / layer_slug


def main() -> None:
    args = parse_args()
    root_out = _resolve_output_dir(args)

    import numpy as np
    import pandas as pd

    # ── Dry-run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"Vector path:  {args.vector_path}")
        print(f"Layer:        {args.layer}")
        print(f"Epsilons:     {args.epsilons}")
        print(f"lr/n_epochs:  {args.lr} / {args.n_epochs}")
        print(f"n_train/eval: {args.n_train} / {args.n_eval}")
        print(f"max_side:     {args.max_side}")
        print(f"Output root:  {root_out}")

        vec = np.load(args.vector_path, allow_pickle=True).item()
        v = vec[args.layer]
        print(f"Vector @ {args.layer}: shape={v.shape}, norm={float(np.linalg.norm(v)):.4f}")

        manifest_path = (Path(args.manifest) if args.manifest
                         else Path("data/selected_uniform_500_manifest.pkl"))
        manifest_df = pd.read_pickle(manifest_path)
        print(f"Manifest: {len(manifest_df)} images")
        print(f"Train images: {args.n_train}  Eval images: {args.n_eval}")
        assert args.n_train + args.n_eval <= len(manifest_df), \
            f"n_train + n_eval ({args.n_train + args.n_eval}) > manifest size ({len(manifest_df)})"

        from representation.IV1_gradient_matching import perturb  # noqa
        print("representation.IV1_gradient_matching.perturb imports OK")
        return

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest_path = (Path(args.manifest) if args.manifest
                     else Path("data/selected_uniform_500_manifest.pkl"))
    manifest_df = pd.read_pickle(manifest_path)

    if args.smoke_test:
        n_train, n_eval = 5, 2
        epsilons = [0.5]
        n_epochs = 2
    else:
        n_train = args.n_train
        n_eval  = args.n_eval
        epsilons = args.epsilons
        n_epochs = args.n_epochs

    # Stratified split: manifest is ordered by human-perceived interest.
    # Every 5th position (indices 4,9,14,...) → eval; rest → train.
    # Both splits therefore span the full interest distribution uniformly.
    eval_mask   = pd.Series([i % 5 == 4 for i in range(len(manifest_df))],
                            index=manifest_df.index)
    train_paths = manifest_df.loc[~eval_mask, "img_path"].iloc[:n_train].tolist()
    eval_paths  = manifest_df.loc[eval_mask,  "img_path"].iloc[:n_eval].tolist()
    logger.info(f"Train: {len(train_paths)} images  |  Eval: {len(eval_paths)} images")

    # ── Load vector ───────────────────────────────────────────────────────────
    vec = np.load(args.vector_path, allow_pickle=True).item()
    vector = vec[args.layer].astype(np.float32)
    vector = vector / np.linalg.norm(vector)
    logger.info(f"Loaded v_interest @ {args.layer}, shape={vector.shape}")

    # ── Load model ────────────────────────────────────────────────────────────
    from utils.model_loader import load_model_and_processor
    from utils.hpc import get_offload_dir, setup_cuda_env
    from utils.prompt_builder import build_blank_prompt
    setup_cuda_env()
    offload = get_offload_dir(
        base=Path(args.model_path).parent / "offload_dir",
        suffix=args.offload_suffix,
    )

    # Cap all GPUs so accelerate leaves activation headroom for the requires_grad
    # forward pass.  Default auto map packs GPUs 1-2 to ~39.5 GiB (only 35 MiB
    # free), which is not enough for the MoE routing tensor (~40-80 MiB).
    # GPU 0 actual ~38 GiB (vision+projector+embed+7 LLM layers) → cap 39 GiB.
    # GPUs 1-4 capped at 35 GiB → 8 layers each, ~3-4 GiB free per GPU.
    # GPU 5 capped at 38 GiB → holds layers 39-47 + norm + lm_head without disk.
    import torch
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 6:
        # Cap GPUs 1-5 at 35 GiB so each gets 8 layers (~36.5 GiB actual, ~3 GiB free).
        # Without a cap GPUs 1-2 are packed to ~39.5 GiB (35 MiB free) — not enough for
        # the MoE routing tensor (40 MiB) under requires_grad.
        # GPU 0 capped at 39 GiB: actual ~38 GiB (vision+projector+embed+7 LLM layers).
        # Layer 47 + lm_head overflow to disk under this config, but the early-stop
        # mechanism aborts the forward pass after layer 29 — GPUs 4-5 never execute
        # during training or cosine eval, so disk-offloaded modules are never loaded.
        _uap_max_memory = {i: "35GiB" for i in range(n_gpus)}
        _uap_max_memory[0] = "39GiB"
    else:
        _uap_max_memory = None
    logger.info(f"Loading model (max_memory={_uap_max_memory})…")
    model, processor = load_model_and_processor(
        args.model_path,
        offload_dir=str(offload) if offload else None,
        max_memory=_uap_max_memory,
    )
    prompt = build_blank_prompt()

    from representation.IV1_gradient_matching.perturb import gradient_match_universal

    # ── Sweep epsilons ────────────────────────────────────────────────────────
    summary_rows = []

    for eps in epsilons:
        # lr scales with epsilon so the delta exactly fills its budget in n_epochs steps.
        # Each epoch applies one sign step of size lr; after n_epochs the L_inf norm
        # equals epsilon (assuming consistent gradient direction across epochs).
        lr = args.lr if args.lr is not None else eps / n_epochs
        logger.info(f"\n{'='*60}")
        logger.info(f"  EPSILON = {eps}  lr = {lr:.5f}")
        logger.info(f"{'='*60}")

        eps_dir = root_out / f"eps{eps:.2f}"
        eps_dir.mkdir(parents=True, exist_ok=True)

        result = gradient_match_universal(
            model=model, processor=processor,
            train_image_paths=train_paths,
            eval_image_paths=eval_paths,
            prompt=prompt,
            layer_key=args.layer,
            direction_vector=vector,
            epsilon=eps,
            lr=lr,
            n_epochs=n_epochs,
            max_side=args.max_side,
            out_dir=eps_dir if args.save_epochs else None,
        )

        # Save delta
        np.save(eps_dir / "delta.npy", result["delta"])
        logger.info(f"  Saved delta → {eps_dir / 'delta.npy'}")

        # Save per-image eval CSV (cosine only — no behavioural ratings)
        if result["eval_results"]:
            eval_df = pd.DataFrame(result["eval_results"])
            eval_df.to_csv(eps_dir / "uap_eval.csv", index=False)
            mean_cos_orig = eval_df["cos_original"].mean()
            mean_cos_pert = eval_df["cos_perturbed"].mean()
        else:
            mean_cos_orig = mean_cos_pert = float("nan")

        # Save cos_history
        pd.DataFrame({"epoch": range(1, len(result["cos_history"]) + 1),
                      "mean_cos": result["cos_history"]}).to_csv(
            eps_dir / "cos_history.csv", index=False)

        row = {
            "epsilon": eps,
            "mean_cos_original": round(mean_cos_orig, 4),
            "mean_cos_perturbed": round(mean_cos_pert, 4),
            "mean_cos_delta": round(mean_cos_pert - mean_cos_orig, 4),
            "final_train_cos": round(result["cos_history"][-1], 4),
        }
        summary_rows.append(row)
        logger.info(f"  eps={eps}: Δcos={row['mean_cos_delta']:+.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(root_out / "uap_summary.csv", index=False)

    print("\n" + "="*60)
    print("UAP sweep complete:")
    print(summary_df.to_string(index=False))
    print(f"\n✅ Results → {root_out}")


if __name__ == "__main__":
    main()
