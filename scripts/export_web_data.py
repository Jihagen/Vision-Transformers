"""
Export compact, browser-friendly data for the project website.

Every number written by this script is read from a saved experiment output
under results/ (or computed from one by a transformation stated in the
export's metadata). Nothing is typed in by hand: the tables below only
attach labels, units and caveats to source files.

Usage (from the repository root):

    python scripts/export_web_data.py                 # write web_export/data + manifest
    python scripts/export_web_data.py --check         # report available sources, write nothing
    python scripts/export_web_data.py --results_dir /path/to/results

Requires numpy and pandas only. No model, GPU or raw image is needed.
An export whose source files are missing is skipped and listed as such in
web_export/release_manifest.json rather than filled with placeholders.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "0.1"

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
MODEL_CONFIG_REL = (
    "hpc_infrastructure/hf_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/"
    "local-repo/config.json"
)

PRELIMINARY = "preliminary"

INTEREST_LABELS = [
    "Not Interesting", "Slightly Interesting", "Moderately Interesting",
    "Very Interesting", "Extremely Interesting",
]

# ── Labels for saved experiment folders (labels only, never values) ──────────

# results/representation_discovery/<variant>/
DISCOVERY_VARIANTS = {
    "base": ("gender", "Gender (female vs male, emotion-matched persona pairs)"),
    "extended_germany": ("gender", "Gender, personas extended with Country = Germany"),
    "extended_nigeria": ("gender", "Gender, personas extended with Country = Nigeria"),
    "extended_germany_country": ("country", "Country (Germany-extended vs base persona)"),
    "extended_nigeria_country": ("country", "Country (Nigeria-extended vs base persona)"),
    "base_emotion": ("emotion_one_vs_rest", "Emotion, one vs rest (gender-averaged)"),
    "base_emotion_anchor_contentment": (
        "emotion_vs_contentment", "Emotion vs contentment anchor (gender-averaged)"),
    "mental_workload": ("mental_workload", "Mental workload (overwhelming vs minimal)"),
    "interestingness": ("interestingness", "Interestingness (high- vs low-rated images)"),
}

# results/<dir>/<layer>/.../dose_response_blank/control_results.csv
DOSE_RESPONSE_RUNS = [
    ("interest", "Interestingness (blank-prompt high vs low)",
     "interest_validation/language_29/dose_response_blank"),
    ("excited_vs_angry", "Excitement vs anger",
     "emotion_validation/language_29/excited_vs_angry/dose_response_blank"),
    ("workload", "Mental workload (overwhelming vs minimal)",
     "workload_validation/language_24/dose_response_blank"),
    ("workload", "Mental workload (overwhelming vs minimal)",
     "workload_validation/language_29/dose_response_blank"),
    ("country_nigeria", "Country (Nigeria-extended vs base, averaged over personas)",
     "country_validation/language_23/dose_response_blank"),
]

# results/<dir>/<layer>/primary/*_summary.csv  (no_inject / inject ±2 / ablate)
SINGLE_DOSE_RUNS = [
    ("gender", "Gender (female minus male, averaged over emotions)",
     "blank_validation/language_29/primary/blank_validation_summary.csv"),
    ("country_nigeria", "Country (Nigeria-extended vs base, averaged over personas)",
     "country_validation/language_23/primary/lexicon_validation_summary.csv"),
    ("interest", "Interestingness (blank-prompt high vs low)",
     "interest_validation/language_29/primary/lexicon_validation_summary.csv"),
]
EMOTION_SINGLE_DOSE = "emotion_validation/language_29/all_emotions_summary.csv"

# results/universal_perturbation_projected/<target>/
UAP_TARGETS = {
    "interest": ("Interestingness", "language_29", True),
    "excited_vs_angry": ("Excitement vs anger", "language_29", True),
    "workload": ("Mental workload", "language_29", True),
    # trained, but no behavioural evaluation of this variant is saved
    "workload_language24": ("Mental workload", "language_24", False),
}

# results/generalisation/<task>/
GENERALISATION_TASKS = {
    "shopping_relevance": dict(
        dataset="Marqo/marqo-GS-10M", label="Shopping relevance",
        judgement="relevance of a product image to a shopping query", scale_min=1, scale_max=5),
    "moral_evaluation": dict(
        dataset="AIML-TUDA/smid (SMID)", label="Moral evaluation",
        judgement="morality of a photographed scene", scale_min=1, scale_max=5),
    "damage_severity": dict(
        dataset="QCRI/MEDIC", label="Damage severity",
        judgement="visible disaster-damage severity", scale_min=0, scale_max=2),
}

# The L_inf budget is expressed in processor-normalised pixel units, whose valid
# range is [-1, 1] (attack/IV1_gradient_matching/perturb.py). Used only to add a
# derived "fraction of the full pixel range" column next to epsilon.
PIXEL_RANGE_WIDTH = 2.0

LAYER_RE = re.compile(r"^(vision|language)_(\d+)_D(\d+)$")


# ── Bookkeeping ──────────────────────────────────────────────────────────────

class Exporter:
    def __init__(self, results_dir: Path, out_dir: Path, check_only: bool):
        self.results = results_dir
        self.out = out_dir
        self.data_dir = out_dir / "data"
        self.check_only = check_only
        self.sources: dict[str, dict] = {}
        self.exports: list[dict] = []
        self.skipped: list[dict] = []

    def rel(self, path: Path) -> str:
        """Path as written in metadata: results/... regardless of --results_dir."""
        return "results/" + path.resolve().relative_to(self.results.resolve()).as_posix()

    def src(self, rel_path: str) -> Path:
        """Resolve a source under results/, registering it for the manifest."""
        path = self.results / rel_path
        if not path.is_file():
            raise FileNotFoundError(rel_path)
        key = self.rel(path)
        if key not in self.sources:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            st = path.stat()
            self.sources[key] = {
                "path": key,
                "sha256": h.hexdigest(),
                "bytes": st.st_size,
                "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        return path

    def _record(self, name: str, meta: dict, n_rows: int | None) -> None:
        self.exports.append({
            "file": f"data/{name}",
            "tier": meta["tier"],
            "title": meta["title"],
            "status": meta["status"],
            "rows": n_rows,
            "sources": meta["sources"],
        })
        print(f"  wrote  data/{name}" + (f"  ({n_rows} rows)" if n_rows is not None else ""))

    def write_table(self, name: str, df: pd.DataFrame, meta: dict) -> None:
        """CSV table plus a <name>.meta.json sidecar describing it."""
        meta = {**meta, "columns": list(df.columns), "rows": len(df)}
        if self.check_only:
            print(f"  ok     data/{name}  ({len(df)} rows)")
            return
        path = self.data_dir / name
        atomic_write_text(path, df.to_csv(index=False, float_format="%.6g", lineterminator="\n"))
        atomic_write_text(path.with_suffix(".meta.json"), json.dumps(meta, indent=2) + "\n")
        self._record(name, meta, len(df))

    def write_json(self, name: str, payload: dict, meta: dict) -> None:
        """Nested data with the metadata block embedded under 'meta'."""
        if self.check_only:
            print(f"  ok     data/{name}")
            return
        atomic_write_text(self.data_dir / name,
                          json.dumps({"meta": meta, **payload}, indent=1) + "\n")
        self._record(name, meta, None)

    def run(self, fn) -> None:
        try:
            fn(self)
        except FileNotFoundError as e:
            self.skipped.append({"export": fn.__name__.removeprefix("export_"),
                                 "missing_source": f"results/{e}"})
            print(f"  SKIP   {fn.__name__}: missing results/{e}")


def atomic_write_text(path: Path, text: str) -> None:
    """
    Write text to path without ever exposing a partial file.

    The content goes to a temporary file in the destination directory, is
    flushed and synced, and is renamed over the target only once that has
    succeeded. If anything fails, the temporary file is removed and a
    previously written export at path is left exactly as it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def base_meta(tier: str, title: str, sources: list[str], **extra) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "tier": tier,
        "title": title,
        "status": PRELIMINARY,
        "model": MODEL_ID,
        "sources": sources,
        **extra,
    }


def parse_layer(layer_key: str) -> tuple[str, int, int]:
    m = LAYER_RE.match(layer_key)
    if not m:
        raise ValueError(f"Unrecognised layer key: {layer_key!r}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def layer_from_path(rel_path: str) -> str:
    return re.search(r"language_\d+", rel_path).group(0)


# ── Tier I: representation discovery ─────────────────────────────────────────

def export_discovery(ex: Exporter) -> None:
    frames, sources, settings = [], [], {}
    for variant, (family, family_label) in DISCOVERY_VARIANTS.items():
        base = f"representation_discovery/{variant}"
        try:
            ev = pd.read_csv(ex.src(f"{base}/evaluation/vector_evaluation.csv"))
            pr = pd.read_csv(ex.src(f"{base}/probes/probe_accuracy.csv"))
            summary = json.loads(ex.src(f"{base}/summary.json").read_text())
        except FileNotFoundError as e:
            ex.skipped.append({"export": f"discovery/{variant}", "missing_source": f"results/{e}"})
            continue
        sources += [f"results/{base}/evaluation/vector_evaluation.csv",
                    f"results/{base}/probes/probe_accuracy.csv",
                    f"results/{base}/summary.json"]
        settings[variant] = summary.get("settings", {})
        df = ev.merge(pr[["contrast", "layer_key", "accuracy_std", "n_pos", "n_neg"]],
                      on=["contrast", "layer_key"], how="left", validate="one_to_one")
        parsed = df["layer_key"].map(parse_layer)
        df.insert(0, "variant", variant)
        df.insert(1, "family", family)
        df["stream"] = parsed.map(lambda t: t[0])
        df["layer_index"] = parsed.map(lambda t: t[1])
        df["hidden_dim"] = parsed.map(lambda t: t[2])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("representation_discovery/*/evaluation/vector_evaluation.csv")

    full = pd.concat(frames, ignore_index=True).rename(columns={
        "probe_accuracy_cv": "probe_accuracy_cv_mean", "accuracy_std": "probe_accuracy_cv_std",
        "keep": "passes_thresholds"})
    full = full[["variant", "family", "contrast", "stream", "layer_index", "layer_key",
                 "hidden_dim", "projection_auc", "md_cav_cosine", "probe_accuracy_cv_mean",
                 "probe_accuracy_cv_std", "n_pos", "n_neg", "passes_thresholds"]]

    caveats = [
        "projection_auc is computed on the same activations the mean-difference vector was "
        "estimated from (no held-out split) and is reported as max(AUC, 1-AUC), so its floor "
        "is 0.5 and values near 1.0 are optimistic, especially for small-n contrasts.",
        "Persona contrasts differ only in the text prompt; vision-tower activations are "
        "therefore near-identical across the two classes of a contrast, which is why vision "
        "layers sit at AUC of about 0.5 and can show below-chance cross-validated probe accuracy.",
        "Cross-validation folds are stratified by class, not grouped by image.",
        "Single model, single prompt template, one activation collection run per condition.",
    ]
    common = dict(
        model_config="residual-stream activations at every vision and language layer; "
                     "thresholds per variant under 'settings'",
        settings=settings,
        feature_labels={v: {"family": f, "label": l} for v, (f, l) in DISCOVERY_VARIANTS.items()},
        units={
            "projection_auc": "AUROC of the scalar projection onto the mean-difference vector, 0.5-1",
            "md_cav_cosine": "cosine similarity between mean-difference vector and linear-probe "
                             "(CAV) direction, -1 to 1",
            "probe_accuracy_cv_mean": "logistic-regression accuracy, mean over stratified CV folds, 0-1",
            "n_pos / n_neg": "number of activation samples per class",
        },
        caveats=caveats,
    )
    ex.write_table(
        "discovery/layerwise_separability.csv", full,
        base_meta("I", "Layer-wise separability of persona, affect and interestingness contrasts",
                  sources, aggregation="none: one row per contrast and layer, as saved", **common))

    grp = full.groupby(["family", "stream", "layer_index"], sort=True)
    fam = grp.agg(
        n_contrasts=("contrast", "nunique"),
        projection_auc_mean=("projection_auc", "mean"),
        projection_auc_min=("projection_auc", "min"),
        projection_auc_max=("projection_auc", "max"),
        md_cav_cosine_mean=("md_cav_cosine", "mean"),
        probe_accuracy_cv_mean=("probe_accuracy_cv_mean", "mean"),
        share_passing_thresholds=("passes_thresholds", "mean"),
    ).reset_index()
    ex.write_table(
        "discovery/layerwise_family_summary.csv", fam,
        base_meta("I", "Layer-wise separability, summarised per contrast family",
                  sources,
                  aggregation="mean / min / max over all contrasts of a family at one layer "
                              "(families pool the variants listed under feature_labels)",
                  **common))


# ── Tier II: geometry ────────────────────────────────────────────────────────

def _cosine(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.where(norms < 1e-12, 1.0, norms)
    return unit @ unit.T


def export_condition_similarity(ex: Exporter) -> None:
    files = sorted((ex.results / "analytics").glob("*/language_*/mean_vectors_language_*.npy"))
    if not files:
        raise FileNotFoundError("analytics/*/language_*/mean_vectors_language_*.npy")
    sets, sources = [], []
    for f in files:
        rel = f.relative_to(ex.results).as_posix()
        means = np.load(ex.src(rel), allow_pickle=True).item()
        labels = list(means.keys())
        vecs = np.stack([means[l] for l in labels]).astype(np.float64)
        sources.append(f"results/{rel}")
        sets.append({
            "variant": f.parts[-3],
            "layer": f.parts[-2],
            "source": f"results/{rel}",
            "labels": labels,
            "hidden_dim": int(vecs.shape[1]),
            "cosine_raw": np.round(_cosine(vecs), 4).tolist(),
            "cosine_centered": np.round(_cosine(vecs - vecs.mean(axis=0, keepdims=True)), 4).tolist(),
        })
    ex.write_json(
        "geometry/condition_cosine_similarity.json", {"sets": sets},
        base_meta(
            "II", "Cosine similarity between persona-condition mean activations", sources,
            model_config="mean residual-stream activation per persona condition, language stream",
            feature_labels="labels are persona keys <gender>_<emotion>[_extended_<country>]",
            units={"cosine_raw": "cosine similarity between saved condition means, -1 to 1",
                   "cosine_centered": "cosine similarity after subtracting the mean over the "
                                      "conditions of that set, -1 to 1"},
            aggregation="condition means are the saved per-condition averages over images; "
                        "both matrices are computed by this script from those saved vectors "
                        "(cosine_raw reproduces analytics/II1_persona_analytics/"
                        "cosine_similarity.py; cosine_centered is an export-time transformation)",
            caveats=["Saved for a small number of layers only.",
                     "Raw cosines between mean activations are dominated by the shared mean "
                     "component; the centred matrix removes it but depends on which conditions "
                     "are in the set."]))


def export_nn_grouping(ex: Exporter) -> None:
    rel = "analytics/early_layer_sweep/early_layer_sweep.json"
    sweep = json.loads(ex.src(rel).read_text())
    rows = pd.DataFrame(sweep["table"]).rename(columns={"layer": "layer_index"})
    rows.insert(0, "stream", "language")
    ex.write_table(
        "geometry/nn_grouping_by_layer.csv", rows,
        base_meta(
            "II", "What nearest neighbours share, by layer (country vs emotion and gender)",
            [f"results/{rel}"],
            model_config="48 persona-condition mean activations (16 base + 16 Germany + 16 "
                         "Nigeria), language stream, embedded with UMAP (cosine, "
                         "n_neighbors=10, random_state=42)",
            feature_labels={"country_nn_share": "nearest neighbour has the same country variant",
                            "emotion_gender_nn_share": "nearest neighbour has the same emotion "
                                                       "and the same gender"},
            units={"country_nn_share": "share of condition means, 0-1",
                   "emotion_gender_nn_share": "share of condition means, 0-1"},
            aggregation="share over the 48 condition means of nearest neighbours in the 2-D "
                        "UMAP embedding, as saved by "
                        "analytics/II1_persona_analytics/early_layer_sweep.py",
            selection_rule=sweep.get("rule"),
            candidate_layers=sweep.get("candidate_layers"),
            selected_early_layer=sweep.get("early_layer"),
            caveats=["Neighbours are taken in a 2-D UMAP embedding, not in activation space; "
                     "UMAP distorts distances and a single seed was used.",
                     "Computed on 48 condition means, not on individual samples.",
                     "Only the listed layers were evaluated."]))


def export_interest_alignment(ex: Exporter) -> None:
    rel = "representation_discovery/interestingness/analytics/concept_alignment_summary.csv"
    df = pd.read_csv(ex.src(rel))
    parsed = df["layer"].map(parse_layer)
    df.insert(0, "stream", parsed.map(lambda t: t[0]))
    df.insert(1, "layer_index", parsed.map(lambda t: t[1]))
    df = df.rename(columns={"layer": "layer_key"})
    ex.write_table(
        "geometry/interest_concept_alignment.csv", df,
        base_meta(
            "II", "Alignment of interestingness directions with persona-concept directions",
            [f"results/{rel}"],
            model_config="mean-difference directions, language layers 20-47",
            feature_labels={
                "*_matched": "persona-specific interestingness direction vs that persona's own "
                             "gender / emotion / country direction",
                "*_mismatched": "same, against other personas' directions of that concept type",
                "*_gap": "matched minus mismatched",
                "blank_cos_*": "blank-prompt interestingness direction vs averaged concept direction",
            },
            units="mean absolute cosine similarity, 0-1 (as saved, rounded to 3 decimals)",
            aggregation="mean of |cosine| over personas, as saved by "
                        "analytics/II2_interestingness_analytics/concept_alignment.py",
            caveats=["Absolute cosines: sign information is not in this table.",
                     "Persona-specific interestingness contrasts use between 33 and 166 samples "
                     "per class (see discovery/layerwise_separability.csv)."]))


# ── Tier III: causal intervention ────────────────────────────────────────────

def export_dose_response(ex: Exporter) -> None:
    rows, sources = [], []
    for direction, label, rel_dir in DOSE_RESPONSE_RUNS:
        try:
            df = pd.read_csv(ex.src(f"{rel_dir}/control_results.csv"))
        except FileNotFoundError as e:
            ex.skipped.append({"export": f"causal/dose_response/{direction}",
                               "missing_source": f"results/{e}"})
            continue
        sources.append(f"results/{rel_dir}/control_results.csv")
        fail = {}
        try:
            pf = pd.read_csv(ex.src(f"{rel_dir}/parse_fail_rate_by_alpha.csv"))
            fail = dict(zip(pf["alpha"], pf["parse_fail_rate"]))
            sources.append(f"results/{rel_dir}/parse_fail_rate_by_alpha.csv")
        except FileNotFoundError:
            pass
        layer_keys = df["layer_key"].unique()
        assert len(layer_keys) == 1, f"{rel_dir}: expected a single layer, got {layer_keys}"
        base = df[df["alpha"] == 0.0].set_index("filename")["score"]
        for alpha, g in df.groupby("alpha", sort=True):
            paired = (g.set_index("filename")["score"] - base).dropna()
            counts = g["rating"].value_counts()
            row = {
                "direction": direction, "direction_label": label,
                "layer": layer_from_path(rel_dir), "layer_key": layer_keys[0],
                "alpha": alpha, "n_images": len(g),
                "mean_score": g["score"].mean(), "std_score": g["score"].std(ddof=1),
                "sem_score": g["score"].std(ddof=1) / np.sqrt(len(g)),
                "mean_paired_delta_vs_alpha0": paired.mean(), "n_paired": len(paired),
                "parse_fail_rate": fail.get(alpha, np.nan),
            }
            for lab in INTEREST_LABELS:
                row["n_" + lab.lower().replace(" ", "_")] = int(counts.get(lab, 0))
            rows.append(row)
    if not rows:
        raise FileNotFoundError("*/dose_response_blank/control_results.csv")
    ex.write_table(
        "causal/dose_response.csv", pd.DataFrame(rows),
        base_meta(
            "III", "Dose-response of interestingness ratings to residual-stream injection",
            sources,
            model_config="h' = h + alpha * unit direction added to the residual stream at one "
                         "language layer; blank (persona-free) prompt; interestingness task",
            feature_labels={d: l for d, l, _ in DOSE_RESPONSE_RUNS},
            units={"alpha": "multiplier on the unit-normalised direction (residual-stream units)",
                   "mean_score": "interestingness rating, 1 (Not Interesting) to 5 (Extremely Interesting)",
                   "n_<label>": "number of images given that rating",
                   "parse_fail_rate": "share of model outputs that could not be parsed, 0-1"},
            aggregation="mean, sample std and standard error over images at each alpha; "
                        "mean_paired_delta_vs_alpha0 is the per-image difference to the same "
                        "image at alpha = 0, averaged",
            caveats=["One run per direction and layer on the same image subset; no repeats "
                     "or seeds, so the standard error reflects image variance only.",
                     "A gender direction was only tested at single doses "
                     "(see causal/single_dose_validation.csv), not as a blank-prompt sweep."]))


def export_single_dose(ex: Exporter) -> None:
    rows, sources = [], []
    for direction, label, rel in SINGLE_DOSE_RUNS:
        try:
            df = pd.read_csv(ex.src(rel))
        except FileNotFoundError as e:
            ex.skipped.append({"export": f"causal/single_dose/{direction}",
                               "missing_source": f"results/{e}"})
            continue
        sources.append(f"results/{rel}")
        for r in df.itertuples():
            rows.append({"direction": direction, "direction_label": label,
                         "layer": layer_from_path(rel), "condition": r.condition,
                         "mean_score": r.score_mean, "std_score": r.score_std,
                         "n_images": int(r.score_count)})
    try:
        emo = pd.read_csv(ex.src(EMOTION_SINGLE_DOSE))
        sources.append(f"results/{EMOTION_SINGLE_DOSE}")
        for r in emo.itertuples():
            for cond in ["no_inject", "inject_neg2", "inject_pos2", "ablate"]:
                rows.append({"direction": f"emotion_{r.emotion}",
                             "direction_label": f"Emotion: {r.emotion} vs rest (valence {r.valence})",
                             "layer": layer_from_path(EMOTION_SINGLE_DOSE), "condition": cond,
                             "mean_score": getattr(r, cond), "std_score": np.nan,
                             "n_images": pd.NA})
    except FileNotFoundError as e:
        ex.skipped.append({"export": "causal/single_dose/emotions", "missing_source": f"results/{e}"})
    if not rows:
        raise FileNotFoundError("*/primary/*_summary.csv")
    ex.write_table(
        "causal/single_dose_validation.csv", pd.DataFrame(rows),
        base_meta(
            "III", "Single-dose injection and ablation checks per direction", sources,
            model_config="blank prompt, interestingness task; conditions: no_inject, "
                         "inject_neg2 (alpha = -2), inject_pos2 (alpha = +2), ablate "
                         "(direction projected out)",
            feature_labels={d: l for d, l, _ in SINGLE_DOSE_RUNS},
            units={"mean_score": "interestingness rating, 1-5"},
            aggregation="mean and std over images, as saved; per-emotion rows come from a "
                        "summary table that stores means only (std and n not saved there)",
            caveats=["The no_inject baseline covers more images than the injected conditions "
                     "(see n_images), so differences to it are not paired.",
                     "Differences between conditions are small relative to the score std."]))


# ── Tier IV: universal adversarial perturbation ──────────────────────────────

def _eps_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["epsilon_fraction_of_pixel_range"] = df["epsilon"] / PIXEL_RANGE_WIDTH
    return df


EPS_UNITS = ("L_inf bound in processor-normalised pixel units (valid range [-1, 1]); "
             "epsilon_fraction_of_pixel_range = epsilon / 2")
EPS_CAVEAT = ("The budgets are large: epsilon = 1.0 allows a change of half the pixel range and "
              "epsilon = 2.0 spans the whole range, so epsilon = 2.0 is not an imperceptible "
              "perturbation and should be read as a saturation regime.")


def export_uap_alignment(ex: Exporter) -> None:
    summ, hist, sources = [], [], []
    for target, (label, layer, _) in UAP_TARGETS.items():
        base = f"universal_perturbation_projected/{target}"
        try:
            s = pd.read_csv(ex.src(f"{base}/uap_summary.csv"))
        except FileNotFoundError as e:
            ex.skipped.append({"export": f"attack/uap/{target}", "missing_source": f"results/{e}"})
            continue
        sources.append(f"results/{base}/uap_summary.csv")
        n_eval = []
        for eps in s["epsilon"]:
            d = f"{base}/eps{eps:.2f}"
            try:
                n_eval.append(len(pd.read_csv(ex.src(f"{d}/uap_eval.csv"))))
                sources.append(f"results/{d}/uap_eval.csv")
            except FileNotFoundError:
                n_eval.append(pd.NA)
            try:
                h = pd.read_csv(ex.src(f"{d}/cos_history.csv"))
                sources.append(f"results/{d}/cos_history.csv")
                h.insert(0, "target", target)
                h.insert(1, "layer", layer)
                h.insert(2, "epsilon", eps)
                hist.append(h.rename(columns={"mean_cos": "mean_cos_train"}))
            except FileNotFoundError:
                pass
        s.insert(0, "target", target)
        s.insert(1, "target_label", label)
        s.insert(2, "layer", layer)
        s["n_eval_images"] = n_eval
        summ.append(s)
    if not summ:
        raise FileNotFoundError("universal_perturbation_projected/*/uap_summary.csv")
    common = dict(
        model_config="one fixed pixel-space perturbation per target and epsilon, trained by "
                     "gradient matching against an internal direction at the stated language "
                     "layer (runs under results/universal_perturbation_projected)",
        feature_labels={t: f"{l} direction at {ly}" for t, (l, ly, _) in UAP_TARGETS.items()},
        caveats=[EPS_CAVEAT,
                 "One training run per target and epsilon; no seeds.",
                 "workload_language24 was trained but has no saved behavioural evaluation.",
                 "The earlier, superseded runs under results/universal_perturbation are not exported."])
    ex.write_table(
        "attack/uap_alignment.csv", _eps_columns(pd.concat(summ, ignore_index=True)),
        base_meta("IV", "UAP: shift of hidden states toward the target direction (held-out images)",
                  sources,
                  units={"epsilon": EPS_UNITS,
                         "mean_cos_*": "cosine similarity between the hidden state at the target "
                                       "layer and the target direction, -1 to 1",
                         "mean_cos_delta": "perturbed minus original"},
                  aggregation="mean over held-out evaluation images, as saved in uap_summary.csv; "
                              "final_train_cos is the last training-epoch mean",
                  **common))
    if hist:
        ex.write_table(
            "attack/uap_training_history.csv", _eps_columns(pd.concat(hist, ignore_index=True)),
            base_meta("IV", "UAP: training-set alignment per epoch", sources,
                      units={"epsilon": EPS_UNITS,
                             "mean_cos_train": "mean cosine similarity to the target direction "
                                               "over training images in that epoch, -1 to 1"},
                      aggregation="mean over training images per epoch, as saved in cos_history.csv",
                      **common))


def _split_condition(cond: str) -> tuple[str, float]:
    m = re.match(r"^(.*)_eps([\d.]+)$", cond)
    return (m.group(1), float(m.group(2))) if m else (cond, 0.0)


def export_uap_behaviour(ex: Exporter) -> None:
    rows, sources = [], []
    rel_i = "attack_eval_projected/interestingness/summary.csv"
    rel_r = "attack_eval_projected/relevance/summary.csv"
    found = False
    try:
        df = pd.read_csv(ex.src(rel_i))
        sources.append(f"results/{rel_i}")
        found = True
        for r in df.to_dict("records"):
            target, eps = _split_condition(r["condition"])
            rows.append({"task": "interestingness", "target": target, "epsilon": eps,
                         "n_images": r["n_parsed"], "metric": "mean_score",
                         "clean_value": r["mean_clean_score"],
                         "perturbed_value": r["mean_perturbed_score"],
                         "delta": r["mean_score_delta"],
                         **{k: r[k] for k in r if k.startswith("n_") and k != "n_parsed"}})
    except FileNotFoundError as e:
        ex.skipped.append({"export": "attack/uap_behaviour/interestingness",
                           "missing_source": f"results/{e}"})
    try:
        df = pd.read_csv(ex.src(rel_r))
        sources.append(f"results/{rel_r}")
        found = True
        clean = df.loc[df["condition"] == "clean", "p_yes"].iloc[0]
        for r in df[df["condition"] != "clean"].to_dict("records"):
            target, eps = _split_condition(r["condition"])
            rows.append({"task": "relevance", "target": target, "epsilon": eps,
                         "n_images": r["n_parsed"], "metric": "p_yes",
                         "clean_value": clean, "perturbed_value": r["p_yes"],
                         "delta": r["mean_score_delta"]})
    except FileNotFoundError as e:
        ex.skipped.append({"export": "attack/uap_behaviour/relevance",
                           "missing_source": f"results/{e}"})
    if not found:
        raise FileNotFoundError(rel_i)
    out = _eps_columns(pd.DataFrame(rows))
    targets = sorted(out["target"].unique())
    ex.write_table(
        "attack/uap_behaviour_original_tasks.csv", out,
        base_meta(
            "IV", "UAP: behavioural effect on the original evaluation tasks", sources,
            model_config="held-out images from the main image set; clean vs perturbed image, "
                         "same prompt; UAPs from results/universal_perturbation_projected",
            feature_labels={"interestingness": "five-level interestingness rating",
                            "relevance": "yes/no 'is this image relevant to you'"},
            targets_evaluated=targets,
            units={"epsilon": EPS_UNITS,
                   "mean_score": "interestingness rating, 1-5",
                   "p_yes": "share of images judged relevant, 0-1",
                   "n_<label>": "number of perturbed images given that rating"},
            aggregation="mean over images, as saved in summary.csv",
            caveats=[EPS_CAVEAT,
                     f"Only these targets have saved results on these tasks: {', '.join(targets)}.",
                     "Single evaluation run; no confidence intervals saved."]))


def export_generalisation(ex: Exporter) -> None:
    summ, budget, sources = [], [], []
    for task, info in GENERALISATION_TASKS.items():
        base = f"generalisation/{task}"
        try:
            s = pd.read_csv(ex.src(f"{base}/summary.csv"))
        except FileNotFoundError as e:
            ex.skipped.append({"export": f"attack/generalisation/{task}",
                               "missing_source": f"results/{e}"})
            continue
        sources.append(f"results/{base}/summary.csv")
        dist = [c for c in s.columns if re.match(r"^(clean|perturbed)_n_\d+$", c)]
        s["clean_label_counts"] = s.apply(
            lambda r: json.dumps({c.split("_")[-1]: int(r[c]) for c in dist if c.startswith("clean")}), axis=1)
        s["perturbed_label_counts"] = s.apply(
            lambda r: json.dumps({c.split("_")[-1]: int(r[c]) for c in dist if c.startswith("perturbed")}), axis=1)
        s = s.drop(columns=dist)
        s.insert(0, "task", task)
        s.insert(1, "dataset", info["dataset"])
        s.insert(2, "scale_min", info["scale_min"])
        s.insert(3, "scale_max", info["scale_max"])
        summ.append(s)
        try:
            b = pd.read_csv(ex.src(f"{base}/budget_response.csv"))
            sources.append(f"results/{base}/budget_response.csv")
            b.insert(0, "task", task)
            # A p-value from a rank correlation over four budget levels is not
            # informative and is deliberately not exported.
            budget.append(b.drop(columns=["spearman_p"], errors="ignore"))
        except FileNotFoundError:
            pass
    if not summ:
        raise FileNotFoundError("generalisation/*/summary.csv")
    common = dict(
        model_config="UAPs trained on the main image set (target layer language_29) applied "
                     "unchanged to new datasets and judgement tasks; each perturbed image is "
                     "compared with its own clean version",
        feature_labels={t: f"{i['label']}: {i['judgement']} ({i['scale_min']}-{i['scale_max']}), "
                           f"{i['dataset']}" for t, i in GENERALISATION_TASKS.items()},
    )
    ex.write_table(
        "attack/generalisation_summary.csv", _eps_columns(pd.concat(summ, ignore_index=True)),
        base_meta(
            "IV", "UAP transfer to unseen datasets and judgement tasks", sources,
            units={"epsilon": EPS_UNITS,
                   "clean_mean / perturbed_mean": "mean model label on the task scale",
                   "mean_signed_shift": "perturbed minus clean label, task-scale units",
                   "normalized_signed_shift": "mean_signed_shift divided by (scale_max - scale_min)",
                   "pct_up / pct_unchanged / pct_down": "share of samples, 0-1",
                   "wasserstein": "1-Wasserstein distance between clean and perturbed label "
                                  "distributions, task-scale units",
                   "*_label_counts": "JSON object, label -> number of samples"},
            aggregation="per attack and epsilon over the frozen stratified sample, as saved",
            caveats=[EPS_CAVEAT,
                     "One frozen sample per task, one evaluation run, no seeds or confidence intervals.",
                     "At epsilon = 2.0 outputs are dominated by attack-specific content rather "
                     "than the image; treat as a separate regime, not a stronger graded effect.",
                     "Source images are not redistributed; only aggregate model outputs are exported."],
            **common))
    if budget:
        ex.write_table(
            "attack/generalisation_budget_response.csv", pd.concat(budget, ignore_index=True),
            base_meta(
                "IV", "UAP transfer: monotonicity of the shift across perturbation budgets",
                [s for s in sources if s.endswith("budget_response.csv")],
                units={"spearman_rho_eps_vs_meandelta": "Spearman rank correlation between "
                                                        "epsilon and mean signed shift, -1 to 1",
                       "eps2_mean_signed_shift": "mean signed shift at epsilon = 2.0, task-scale units"},
                aggregation="rank correlation over n_epsilons budget levels, as saved",
                caveats=["The correlation is over four budget levels only: it describes whether "
                         "the four means are ordered, and carries no statistical weight. The "
                         "saved p-values are omitted for that reason."],
                **common))


def export_arousal(ex: Exporter) -> None:
    rel_a = "generalisation/moral_evaluation/arousal_analysis.csv"
    rel_r = "generalisation/moral_evaluation/arousal_regression.csv"
    a = _eps_columns(pd.read_csv(ex.src(rel_a)))
    r = pd.read_csv(ex.src(rel_r))
    info = GENERALISATION_TASKS["moral_evaluation"]
    common = dict(
        model_config="moral evaluation task (SMID), UAPs with target layer language_29",
        feature_labels={"human_arousal": "normative human arousal rating of the image (SMID)",
                        "delta_morality": "perturbed minus clean model morality label"},
        caveats=["Exploratory, post-hoc analysis on one frozen sample; not corrected for "
                 "multiple comparisons.",
                 "The regression pools the four epsilon levels of the same images, so "
                 "observations are not independent and the p-values are optimistic.",
                 f"Human ratings come from {info['dataset']} and are not redistributed here."])
    ex.write_table(
        "attack/moral_arousal_correlation.csv", a,
        base_meta("IV", "Moral evaluation: image arousal vs attack-induced shift",
                  [f"results/{rel_a}"],
                  units={"corr_*": "Pearson correlation as saved, -1 to 1", "epsilon": EPS_UNITS},
                  aggregation="correlation over images per attack and epsilon, as saved",
                  **common))
    ex.write_table(
        "attack/moral_arousal_regression.csv", r,
        base_meta("IV", "Moral evaluation: arousal x epsilon regression coefficients",
                  [f"results/{rel_r}"],
                  units={"coef": "regression coefficient, morality-scale units per unit of the term",
                         "r_squared": "0-1"},
                  aggregation="one pooled regression per attack "
                              "(delta_morality ~ arousal * epsilon + clean_morality), as saved",
                  **common))


EXPORTS = [
    export_discovery,
    export_condition_similarity, export_nn_grouping, export_interest_alignment,
    export_dose_response, export_single_dose,
    export_uap_alignment, export_uap_behaviour, export_generalisation, export_arousal,
]


# ── Manifest ─────────────────────────────────────────────────────────────────

def _git(*args: str) -> str | None:
    # Only the trailing newline is removed. `git status --porcelain` lines start
    # with a two-column status that may be a space (" M path"), so stripping
    # leading whitespace from the output would shift the first line by one.
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).rstrip("\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def porcelain_paths(status: str) -> list[str]:
    """Paths from `git status --porcelain` output: two status columns, one
    space, then the path. Lines must be passed with their leading columns."""
    return [line[3:] for line in status.splitlines() if line]


def _self_test() -> None:
    path = "analytics/II1_persona_analytics/hyperplane_geometry.py"
    cases = {
        f" M {path}": [path],                                   # unstaged change, first line
        f"M  {path}": [path],                                   # staged change
        f"MM {path}": [path],
        f" M {path}\n M web_export/README.md": [path, "web_export/README.md"],
        f" M {path}\n": [path],
        "": [],
    }
    for status, expected in cases.items():
        got = porcelain_paths(status)
        assert got == expected, f"porcelain_paths({status!r}) = {got!r}, expected {expected!r}"
    print(f"  self-test passed ({len(cases)} cases)")


def model_block() -> dict:
    block = {"id": MODEL_ID,
             "revision": "not recorded: experiments used a local snapshot of the checkpoint",
             "config_source": None}
    cfg_path = REPO_ROOT / MODEL_CONFIG_REL
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        block["config_source"] = MODEL_CONFIG_REL
        block["config"] = {
            "torch_dtype": cfg.get("torch_dtype"),
            "transformers_version_in_checkpoint": cfg.get("transformers_version"),
            "language_layers": (cfg.get("text_config") or {}).get("num_hidden_layers"),
            "vision_layers": (cfg.get("vision_config") or {}).get("num_hidden_layers"),
        }
    return block


def write_manifest(ex: Exporter) -> None:
    status = _git("status", "--porcelain", "--untracked-files=no")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/export_web_data.py",
        "code": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_tracked_changes": (
                porcelain_paths(status) if status is not None else None),
        },
        "note": "results/ is not version-controlled. Source files are identified by path, "
                "SHA-256 and modification time instead of a commit.",
        "model": model_block(),
        "release_status": PRELIMINARY,
        "exports": ex.exports,
        "preliminary_outputs": [e["file"] for e in ex.exports if e["status"] == PRELIMINARY],
        "skipped": ex.skipped,
        "source_files": sorted(ex.sources.values(), key=lambda s: s["path"]),
    }
    atomic_write_text(ex.out / "release_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote  release_manifest.json  ({len(ex.exports)} exports, "
          f"{len(ex.sources)} source files, {len(ex.skipped)} skipped)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results_dir", type=Path, default=REPO_ROOT / "results",
                   help="Directory holding saved experiment outputs (default: <repo>/results).")
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / "web_export",
                   help="Output directory (default: <repo>/web_export).")
    p.add_argument("--check", action="store_true",
                   help="Read and validate sources, but write nothing.")
    p.add_argument("--self_test", action="store_true",
                   help="Run the script's internal parser checks and exit.")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.results_dir.is_dir():
        raise SystemExit(f"No results directory at {args.results_dir}. Saved experiment "
                         f"outputs are required; see web_export/README.md.")
    ex = Exporter(args.results_dir, args.out_dir, args.check)
    for fn in EXPORTS:
        ex.run(fn)
    if not args.check:
        write_manifest(ex)
    elif ex.skipped:
        print(f"  {len(ex.skipped)} export(s) would be skipped.")


if __name__ == "__main__":
    main()
