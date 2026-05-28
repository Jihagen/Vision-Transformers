"""
Runner: Representation Discovery Pipeline (I.2 → I.3 → I.4 → I.5)

For a given experimental variant (base or extended), loads all collected
activation result files, then runs the full representation discovery pipeline:

  I.2  Mean-difference vectors per layer per contrast
  I.3  Linear probes (CAV vectors) per layer per contrast
  I.4  Vector evaluation — AUROC, MD-CAV alignment, keep/drop per layer
  I.5  Subspace check — is one direction sufficient per layer?

All outputs are saved to results/representation_discovery/<variant>/ in both
machine-readable (.npy / .csv) and human-readable (summary JSON + notebook)
formats.

Usage
-----
Dry-run (print what would run, no computation):
    python runners/run_representation_discovery.py --variant base --dry_run

Full pipeline:
    python runners/run_representation_discovery.py --variant base

Specific contrasts only (e.g. just anger):
    python runners/run_representation_discovery.py --variant base \\
        --contrasts female_anger:male_anger female_awe:male_awe

Extended variant (country must already be set in experiment_definitions.py):
    python runners/run_representation_discovery.py --variant extended --country Germany
"""

from __future__ import annotations
import argparse
import json
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


# ── Paths ─────────────────────────────────────────────────────────────────────

_RESULTS_ROOT = Path("results/representation_discovery")


def _out_dir(variant: str) -> Path:
    d = _RESULTS_ROOT / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run representation discovery (I.2–I.5) for a variant."
    )
    p.add_argument("--variant", choices=["base", "extended"], default="base")
    p.add_argument(
        "--contrasts", nargs="*", default=None,
        metavar="POS:NEG",
        help=(
            "Subset of contrasts to run, as 'pos_key:neg_key' pairs. "
            "Default: all gender contrasts for the variant."
        ),
    )
    p.add_argument(
        "--country", default=None,
        help="Country for extended variant (sets it in experiment_definitions).",
    )
    p.add_argument(
        "--cv_folds", type=int, default=5,
        help="Cross-validation folds for linear probes (default: 5).",
    )
    p.add_argument(
        "--chance_threshold", type=float, default=0.60,
        help="Probe accuracy threshold for 'above chance' (default: 0.60).",
    )
    p.add_argument(
        "--auc_threshold", type=float, default=0.65,
        help="AUROC threshold for 'significant separation' (default: 0.65).",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print what would run without doing any computation.",
    )
    p.add_argument(
        "--skip_notebook", action="store_true",
        help="Skip notebook generation (useful on headless servers without nbformat).",
    )
    return p.parse_args()


# ── Pipeline steps ────────────────────────────────────────────────────────────

def _load_data(
    variant: str,
    country: str | None,
) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """
    Load all existing result files for a variant and return
    (loaded_data, gender_contrasts).
    """
    import numpy as np
    from runners.experiment_definitions import (
        get_all_result_paths, get_gender_contrasts, set_extended_country,
    )

    if variant == "extended" and country:
        set_extended_country(country)

    all_paths = get_all_result_paths(variant)
    existing  = {pk: p for pk, p in all_paths.items() if p.exists()}
    if not existing:
        raise FileNotFoundError(
            f"No result files found for variant '{variant}'. "
            f"Run collection first."
        )

    loaded_data: dict[str, dict] = {}
    for pk, path in existing.items():
        loaded_data[pk] = np.load(path, allow_pickle=True).item()
        n = len(loaded_data[pk].get("results", []))
        logger.info(f"  Loaded {pk}: {n} results")

    all_contrasts = get_gender_contrasts(variant)
    # Keep only contrasts where BOTH conditions are loaded
    contrasts = [
        (pos, neg) for pos, neg in all_contrasts
        if pos in loaded_data and neg in loaded_data
    ]
    return loaded_data, contrasts


def _run_i2(loaded_data, contrasts, out_dir: Path) -> dict:
    from representation.I2_mean_difference.mean_difference import compute_all_contrasts
    md_dir = out_dir / "md_vectors"
    md_dir.mkdir(exist_ok=True)
    logger.info("I.2 — Computing mean-difference vectors…")
    md_vectors = compute_all_contrasts(
        loaded_data=loaded_data,
        contrasts=contrasts,
        normalize=True,
        save_dir=md_dir,
    )
    logger.info(f"  Done: {len(md_vectors)} contrasts")
    return md_vectors


def _run_i3(loaded_data, contrasts, out_dir: Path, cv_folds: int) -> dict:
    from representation.I3_linear_probe.probe import train_all_probes
    probe_dir = out_dir / "probes"
    probe_dir.mkdir(exist_ok=True)
    logger.info(f"I.3 — Training linear probes (cv_folds={cv_folds})…")
    probe_results = train_all_probes(
        loaded_data=loaded_data,
        contrasts=contrasts,
        cv_folds=cv_folds,
        save_dir=probe_dir,
    )
    logger.info(f"  Done: {sum(len(v) for v in probe_results.values())} probe×layer results")
    return probe_results


def _run_i4(loaded_data, md_vectors, cav_vectors, probe_results, out_dir: Path,
            chance_threshold: float, auc_threshold: float) -> dict:
    from representation.I4_vector_evaluation.evaluate import evaluate_all_vectors
    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    logger.info("I.4 — Evaluating vectors…")
    reports = evaluate_all_vectors(
        loaded_data=loaded_data,
        md_vectors=md_vectors,
        cav_vectors=cav_vectors,
        probe_results=probe_results,
        save_dir=eval_dir,
        chance_threshold=chance_threshold,
        auc_threshold=auc_threshold,
    )
    n_keep = sum(1 for rlist in reports.values() for r in rlist if r.keep)
    n_total = sum(len(v) for v in reports.values())
    logger.info(f"  Done: {n_keep}/{n_total} (contrast×layer) marked keep")
    return reports


def _run_i5(loaded_data, contrasts, out_dir: Path) -> dict:
    from representation.I5_pca_svd.pca_svd import run_subspace_check
    svd_dir = out_dir / "subspace"
    svd_dir.mkdir(exist_ok=True)
    logger.info("I.5 — Subspace check…")
    subspace = run_subspace_check(
        loaded_data=loaded_data,
        contrasts=contrasts,
        save_dir=svd_dir,
    )
    n_single = sum(1 for rlist in subspace.values() for r in rlist if r.one_vector_sufficient)
    n_total  = sum(len(v) for v in subspace.values())
    logger.info(f"  Done: {n_single}/{n_total} (contrast×layer) one-vector sufficient")
    return subspace


def _save_summary(
    variant: str,
    contrasts: list[tuple[str, str]],
    md_vectors: dict,
    probe_results: dict,
    reports: dict,
    subspace: dict,
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Write a human-readable summary JSON."""
    import numpy as np

    summary: dict = {
        "variant":    variant,
        "contrasts":  [f"{p}_vs_{n}" for p, n in contrasts],
        "n_contrasts": len(contrasts),
        "settings": {
            "cv_folds":          args.cv_folds,
            "chance_threshold":  args.chance_threshold,
            "auc_threshold":     args.auc_threshold,
        },
        "per_contrast": {},
    }

    for contrast_name, rlist in reports.items():
        if not rlist:
            continue
        keep_layers = [r.layer_key for r in rlist if r.keep]
        best_layers = sorted(rlist, key=lambda r: r.projection_auc, reverse=True)[:5]
        probe_accs  = [r.probe_accuracy_cv for r in rlist if not (r.probe_accuracy_cv != r.probe_accuracy_cv)]

        svd_list = subspace.get(contrast_name, [])
        single_vec_layers = [r.layer_key for r in svd_list if r.one_vector_sufficient]

        summary["per_contrast"][contrast_name] = {
            "n_layers_evaluated": len(rlist),
            "n_layers_keep":      len(keep_layers),
            "keep_layers":        keep_layers,
            "best_5_by_auc": [
                {
                    "layer":      r.layer_key,
                    "auc":        round(r.projection_auc, 4),
                    "probe_acc":  round(r.probe_accuracy_cv, 4),
                    "md_cav_cos": round(r.md_cav_cosine, 4),
                    "keep":       r.keep,
                }
                for r in best_layers
            ],
            "mean_probe_accuracy": round(float(np.mean(probe_accs)), 4) if probe_accs else None,
            "one_vector_sufficient_layers": single_vec_layers,
        }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved → {summary_path}")
    return summary_path


# ── Notebook generation ───────────────────────────────────────────────────────

def _generate_notebook(
    variant: str,
    contrasts: list[tuple[str, str]],
    out_dir: Path,
) -> Path | None:
    try:
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    except ImportError:
        logger.warning("nbformat not installed — skipping notebook generation.")
        return None

    contrast_names = [f"{p}_vs_{n}" for p, n in contrasts]
    out_dir_str    = str(out_dir.resolve())

    cells = []

    # ── Title ──
    cells.append(new_markdown_cell(f"""\
# Representation Discovery Report — variant: `{variant}`

This notebook imports pre-computed results from the representation discovery pipeline
(I.2 Mean-Difference → I.3 Linear Probe → I.4 Vector Evaluation → I.5 Subspace Check)
and presents them with explanatory text.

**Contrasts analysed:** {", ".join(f"`{c}`" for c in contrast_names)}

Results directory: `{out_dir_str}`
"""))

    # ── Imports + load ──
    cells.append(new_markdown_cell("## Setup — load results"))
    cells.append(new_code_cell(f"""\
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path("{out_dir_str}").parent.parent.parent))
OUT_DIR  = Path("{out_dir_str}")
VARIANT  = "{variant}"

# Load summary JSON
with open(OUT_DIR / "summary.json") as f:
    summary = json.load(f)

# Load evaluation CSV
eval_csv  = OUT_DIR / "evaluation" / "vector_evaluation.csv"
probe_csv = OUT_DIR / "probes"     / "probe_accuracy.csv"
svd_csv   = OUT_DIR / "subspace"   / "subspace_check.csv"

eval_df  = pd.read_csv(eval_csv)  if eval_csv.exists()  else pd.DataFrame()
probe_df = pd.read_csv(probe_csv) if probe_csv.exists() else pd.DataFrame()
svd_df   = pd.read_csv(svd_csv)   if svd_csv.exists()   else pd.DataFrame()

CONTRASTS = {repr(contrast_names)}
print(f"Loaded {{len(eval_df)}} evaluation rows across {{len(CONTRASTS)}} contrasts")
"""))

    # ── Summary table ──
    cells.append(new_markdown_cell("""\
## Overall summary

Each row is one contrast (e.g. female_anger vs male_anger).
- **keep layers**: layers where both the mean-difference AUROC and probe accuracy exceed
  their respective thresholds — these are candidate layers for concept vector extraction.
- **best AUC layer**: the single layer where the mean-difference vector most cleanly
  separates the two conditions.
"""))
    cells.append(new_code_cell("""\
rows = []
for cname, info in summary["per_contrast"].items():
    best = info["best_5_by_auc"][0] if info["best_5_by_auc"] else {}
    rows.append({
        "Contrast":           cname,
        "Layers evaluated":   info["n_layers_evaluated"],
        "Layers keep":        info["n_layers_keep"],
        "Mean probe acc":     info["mean_probe_accuracy"],
        "Best layer (AUC)":   best.get("layer", "—"),
        "Best AUC":           best.get("auc", float("nan")),
        "Best probe acc":     best.get("probe_acc", float("nan")),
        "MD-CAV cosine":      best.get("md_cav_cos", float("nan")),
        "1-vec layers":       len(info.get("one_vector_sufficient_layers", [])),
    })
pd.DataFrame(rows).set_index("Contrast").round(4)
"""))

    # ── Layer profile plots ──
    cells.append(new_markdown_cell("""\
## Layer-depth profiles

For each contrast, plot AUROC and probe accuracy across all LLM layers.
Strong representation discovery = high and stable AUROC in the middle-to-late
LLM layers, with good alignment (high MD-CAV cosine) at the peak.
"""))
    cells.append(new_code_cell(f"""\
if eval_df.empty:
    print("No evaluation data found.")
else:
    llm_df = eval_df[eval_df["layer_key"].str.startswith("language_")].copy()
    llm_df["layer_num"] = llm_df["layer_key"].str.extract(r"language_(\\d+)").astype(int)

    fig, axes = plt.subplots(
        len(CONTRASTS), 1,
        figsize=(14, 4 * len(CONTRASTS)),
        sharex=False,
    )
    if len(CONTRASTS) == 1:
        axes = [axes]

    for ax, cname in zip(axes, CONTRASTS):
        sub = llm_df[llm_df["contrast"] == cname].sort_values("layer_num")
        if sub.empty:
            ax.set_title(f"{{cname}} — no data")
            continue
        ax.plot(sub["layer_num"], sub["projection_auc"],  label="MD-AUC",      lw=2)
        ax.plot(sub["layer_num"], sub["probe_accuracy_cv"], label="Probe acc",  lw=2, ls="--")
        ax.plot(sub["layer_num"], sub["md_cav_cosine"],   label="MD-CAV cos",  lw=1.5, ls=":")
        ax.axhline(0.65, color="gray", lw=1, ls="-.", alpha=0.5, label="AUC threshold")
        ax.axhline(0.60, color="gray", lw=1, ls="-.",  alpha=0.3)
        ax.set_title(cname, fontsize=11)
        ax.set_xlabel("LLM layer index")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Layer-depth profiles — {{VARIANT}}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "layer_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()
"""))

    # ── Best candidate layers per contrast ──
    cells.append(new_markdown_cell("""\
## Best candidate layers

Layers marked **keep** satisfy both:
- Mean-difference AUROC > threshold (default 0.65)
- Probe accuracy > threshold (default 0.60)

A high MD-CAV cosine (> 0.85) additionally indicates that the mean-difference
vector and the probe's learned CAV agree on the concept direction — the strongest
signal for a clean, linear concept.
"""))
    cells.append(new_code_cell("""\
for cname in CONTRASTS:
    info = summary["per_contrast"].get(cname, {})
    print(f"\\n{'='*60}")
    print(f"Contrast: {cname}")
    print(f"  Kept {info.get('n_layers_keep', 0)} / {info.get('n_layers_evaluated', 0)} layers")
    print(f"  Top 5 by AUC:")
    for b in info.get("best_5_by_auc", []):
        mark = "✓ KEEP" if b["keep"] else "✗ drop"
        print(f"    {b['layer']:<25}  AUC={b['auc']:.3f}  probe={b['probe_acc']:.3f}  "
              f"cos={b['md_cav_cos']:.3f}  {mark}")
"""))

    # ── Subspace check ──
    cells.append(new_markdown_cell("""\
## Subspace check (I.5)

For each contrast × layer, the subspace check asks: is the concept direction
effectively 1-dimensional, or does it require multiple basis vectors?

`one_vector_sufficient = True` means the first principal component of the
within-class difference matrix explains ≥ 80 % of variance — a single direction
vector is enough for steering.

`n_components_90pct` tells you how many components are needed to explain 90 %
of the within-class variation.  A low number suggests a clean, concentrated concept.
"""))
    cells.append(new_code_cell("""\
if svd_df.empty:
    print("No subspace data found.")
else:
    llm_svd = svd_df[svd_df["layer_key"].str.startswith("language_")].copy()
    llm_svd["layer_num"] = llm_svd["layer_key"].str.extract(r"language_(\\d+)").astype(int)

    for cname in CONTRASTS:
        sub = llm_svd[llm_svd["contrast"] == cname].sort_values("layer_num")
        if sub.empty:
            continue
        n_single = sub["one_vector_sufficient"].sum()
        print(f"\\n{cname}: {n_single}/{len(sub)} LLM layers are one-vector sufficient")
        best_evr = sub.nlargest(3, "evr_first")[["layer_key", "evr_first", "n_components_90pct", "one_vector_sufficient"]]
        print(best_evr.to_string(index=False))
"""))

    # ── Cross-contrast comparison ──
    cells.append(new_markdown_cell("""\
## Cross-contrast consistency

If the same layers show the strongest gender signal across all emotions, that
is evidence of a stable, emotion-independent gender representation in the LLM.
Conversely, if peak layers vary substantially across emotions, the concept
direction is emotion-entangled.
"""))
    cells.append(new_code_cell("""\
if not eval_df.empty:
    llm_df = eval_df[eval_df["layer_key"].str.startswith("language_")].copy()
    llm_df["layer_num"] = llm_df["layer_key"].str.extract(r"language_(\\d+)").astype(int)

    pivot = llm_df.pivot_table(
        index="layer_num", columns="contrast",
        values="projection_auc", aggfunc="mean"
    )
    ax = pivot.plot(figsize=(14, 4), title="MD-AUC by layer across all contrasts")
    ax.set_xlabel("LLM layer index"); ax.set_ylabel("MD-AUC")
    ax.axhline(0.65, color="gray", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cross_contrast_auc.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Mean AUC across contrasts per layer — most consistent layers
    pivot["mean_across_contrasts"] = pivot.mean(axis=1)
    top = pivot["mean_across_contrasts"].nlargest(10)
    print("Top 10 LLM layers by mean AUC across all contrasts:")
    print(top.to_string())
"""))

    nb = new_notebook(cells=cells)
    nb_path = out_dir / "discovery_report.ipynb"
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)
    logger.info(f"Notebook saved → {nb_path}")
    return nb_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    from runners.experiment_definitions import get_gender_contrasts, set_extended_country
    if args.variant == "extended" and args.country:
        set_extended_country(args.country)

    # Parse user-specified contrasts if given
    if args.contrasts:
        contrasts: list[tuple[str, str]] = []
        for c in args.contrasts:
            if ":" not in c:
                logger.error(f"Contrast '{c}' must be in 'pos_key:neg_key' format.")
                sys.exit(1)
            pos, neg = c.split(":", 1)
            contrasts.append((pos.strip(), neg.strip()))
    else:
        contrasts = get_gender_contrasts(args.variant)

    out_dir = _out_dir(args.variant)

    if args.dry_run:
        print(f"Variant:   {args.variant}")
        print(f"Contrasts: {[f'{p}_vs_{n}' for p, n in contrasts]}")
        print(f"Output:    {out_dir}")
        print(f"Steps:     I.2 mean-diff  |  I.3 probes (cv={args.cv_folds})  |  I.4 eval  |  I.5 subspace")
        return

    logger.info(f"Representation discovery — variant={args.variant}, {len(contrasts)} contrasts")
    logger.info(f"Output directory: {out_dir}")

    # Load data
    logger.info("Loading activation result files…")
    loaded_data, available_contrasts = _load_data(args.variant, args.country)

    # Filter to available contrasts
    available_set = {(p, n) for p, n in available_contrasts}
    contrasts = [(p, n) for p, n in contrasts if (p, n) in available_set]
    if not contrasts:
        logger.error("No requested contrasts have both conditions collected. Exiting.")
        sys.exit(1)
    logger.info(f"Running {len(contrasts)} contrasts: {[f'{p}_vs_{n}' for p,n in contrasts]}")

    # I.2
    md_vectors = _run_i2(loaded_data, contrasts, out_dir)

    # I.3
    probe_results = _run_i3(loaded_data, contrasts, out_dir, cv_folds=args.cv_folds)

    # Derive CAV vectors from probe results
    from representation.I3_linear_probe.probe import get_cav_vectors
    cav_vectors = {
        cname: get_cav_vectors(rlist)
        for cname, rlist in probe_results.items()
    }

    # I.4
    reports = _run_i4(
        loaded_data, md_vectors, cav_vectors, probe_results, out_dir,
        chance_threshold=args.chance_threshold,
        auc_threshold=args.auc_threshold,
    )

    # I.5
    subspace = _run_i5(loaded_data, contrasts, out_dir)

    # Human-readable summary JSON
    summary_path = _save_summary(
        args.variant, contrasts, md_vectors, probe_results, reports, subspace,
        out_dir, args,
    )

    # Notebook
    if not args.skip_notebook:
        nb_path = _generate_notebook(args.variant, contrasts, out_dir)
        if nb_path:
            logger.info(f"Open the report: jupyter notebook {nb_path}")

    logger.info("Representation discovery complete.")
    logger.info(f"Results: {out_dir}")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
