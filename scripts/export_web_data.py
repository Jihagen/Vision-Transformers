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

Requires numpy and pandas only, except the UAP example gallery, which also needs
Pillow, the model's image processor (transformers) and the selected source
images. No model weights or GPU are needed.
scripts/validate_web_export.py checks the result.
An export whose source files are missing is skipped and listed as such in
web_export/release_manifest.json rather than filled with placeholders.
"""

from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "0.1"

sys.path.insert(0, str(REPO_ROOT))
from analytics.II1_persona_analytics.additivity import (  # noqa: E402  (numpy/pandas only)
    COMPOSITION_COEFFICIENTS, COMPOSITION_FEATURES, composition_components,
    derive_raw_feature_vectors, shift_cosine,
)

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

# Shared reference layer of the two interactive geometry figures.
GEOMETRY_LAYER = "language_29_D5120"
DISPLAY_DECIMALS = 6

PERSONA_EMOTIONS = ["anger", "amusement", "awe", "contentment", "disgust", "excitement", "fear", "sad"]
PERSONA_GENDERS = ("female", "male")
PERSONA_COUNTRIES = ("Germany", "Nigeria")
EMOTION_LABELS = {e: e for e in PERSONA_EMOTIONS} | {"sad": "sadness"}

# Direction set of the "vector alignment" figure, in the order used by
# results/EX1_T1_workload_analytics.ipynb. Paths are relative to
# results/representation_discovery/; "glob" entries are averaged over files.
_ONE_VS_REST = ("gender-averaged {e} personas (female and male merged) minus the pooled "
                "gender-averaged personas of the other seven emotions, base persona set")
VECTOR_FIELD_DIRECTIONS = [
    dict(id="workload_overwhelming_vs_minimal", label="Workload / stress: overwhelming − minimal",
         family="workload",
         file="mental_workload/md_vectors/workload_overwhelming_vs_workload_minimal.npy",
         definition="blank prompt plus the single attribute 'Mental workload: overwhelming' minus "
                    "the same prompt with 'Mental workload: minimal'"),
    dict(id="emotion_excited_vs_angry", label="Excitement − anger", family="emotion_attack",
         file="base_emotion/md_vectors/avg_excitement_vs_avg_anger.npy",
         definition="gender-averaged excitement personas minus gender-averaged anger personas, "
                    "base persona set; the direction targeted by the excited-vs-angry UAP"),
    dict(id="gender_female_vs_male", label="Gender: female − male (averaged over emotions)",
         family="gender", glob=("base/md_vectors", "female_*_vs_male_*.npy"),
         definition="mean of the eight emotion-matched female-minus-male directions, base persona set"),
    dict(id="country_germany", label="Country: Germany", family="country",
         glob=("extended_germany_country/md_vectors", "*.npy"),
         definition="mean over the 16 gender x emotion personas of (persona extended with "
                    "Country = Germany) minus (same persona without a country)"),
    dict(id="country_nigeria", label="Country: Nigeria", family="country",
         glob=("extended_nigeria_country/md_vectors", "*.npy"),
         definition="mean over the 16 gender x emotion personas of (persona extended with "
                    "Country = Nigeria) minus (same persona without a country)"),
    dict(id="interest_blank_high_vs_low", label="Blank-prompt interestingness: high − low",
         family="interestingness",
         file="interestingness/md_vectors/blank_interest_high_vs_blank_interest_low.npy",
         definition="blank (persona-free) prompt: images rated Very or Extremely Interesting minus "
                    "images rated Not or Slightly Interesting; Moderately Interesting dropped"),
    dict(id="interest_global_high_vs_low", label="Global interestingness: high − low",
         family="interestingness",
         file="interestingness/md_vectors/global_interest_high_vs_global_interest_low.npy",
         definition="same high-vs-low split pooled over the blank prompt and all 48 persona "
                    "conditions, classes balanced by subsampling"),
] + [
    dict(id=f"emotion_{e}", label=f"Emotion: {EMOTION_LABELS[e]}", family="emotion",
         file=f"base_emotion/md_vectors/avg_{e}_vs_avg_rest_{e}.npy",
         definition=_ONE_VS_REST.format(e=EMOTION_LABELS[e]))
    for e in PERSONA_EMOTIONS
]

LAYER_RE = re.compile(r"^(vision|language)_(\d+)_D(\d+)$")


# ── Bookkeeping ──────────────────────────────────────────────────────────────

class Exporter:
    def __init__(self, results_dir: Path, out_dir: Path, check_only: bool):
        self.results = results_dir
        self.out = out_dir
        self.data_dir = out_dir / "data"
        self.check_only = check_only
        self.sources: dict[str, dict] = {}
        self.assets: list[dict] = []
        self.exports: list[dict] = []
        self.skipped: list[dict] = []

    def rel(self, path: Path) -> str:
        """Path as written in metadata: results/... regardless of --results_dir."""
        return "results/" + path.resolve().relative_to(self.results.resolve()).as_posix()

    def src(self, rel_path: str, root: str = "results") -> Path:
        """Resolve a source under results/ (or the repository's data/), registering it for the manifest."""
        path = (self.results if root == "results" else REPO_ROOT / root) / rel_path
        if not path.is_file():
            raise FileNotFoundError(rel_path)
        key = self.rel(path) if root == "results" else f"{root}/{rel_path}"
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

    def write_json(self, name: str, payload: dict, meta: dict, inline_arrays: bool = False) -> None:
        """Nested data with the metadata block embedded under 'meta'."""
        if self.check_only:
            print(f"  ok     data/{name}")
            return
        doc = {"meta": meta, **payload}
        text = dumps_inline_arrays(doc) if inline_arrays else json.dumps(doc, indent=1)
        atomic_write_text(self.data_dir / name, text + "\n")
        self._record(name, meta, None)

    def write_asset(self, name: str, content: bytes, **info) -> None:
        """Binary file under web_export/assets/, listed with its hash in the manifest."""
        if self.check_only:
            return
        path = self.out / "assets" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(content)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
        self.assets.append({"file": f"assets/{name}", "sha256": hashlib.sha256(content).hexdigest(),
                            "bytes": len(content), **info})

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


def dumps_inline_arrays(obj, level: int = 0) -> str:
    """json.dumps(indent=1), except that arrays of scalars stay on one line."""
    pad, end = " " * (level + 1), " " * level
    if isinstance(obj, dict) and obj:
        return "{\n" + ",\n".join(f"{pad}{json.dumps(k)}: {dumps_inline_arrays(v, level + 1)}"
                                  for k, v in obj.items()) + f"\n{end}}}"
    if isinstance(obj, list) and any(isinstance(v, (dict, list)) for v in obj):
        return "[\n" + ",\n".join(pad + dumps_inline_arrays(v, level + 1) for v in obj) + f"\n{end}]"
    if isinstance(obj, list):
        return "[" + ", ".join(json.dumps(v, allow_nan=False) for v in obj) + "]"
    return json.dumps(obj, allow_nan=False)


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


# ── Tier II: interactive geometry figures at one reference layer ─────────────

def _rounded(x, decimals: int = DISPLAY_DECIMALS):
    """Finite float64 rounded for export; -0.0 becomes 0.0."""
    a = np.round(np.asarray(x, dtype=np.float64), decimals) + 0.0
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite value in geometry export")
    return a.tolist()


def _top3_basis(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Top-3 right singular vectors of rows (exact LAPACK SVD, no random state)
    and all singular values. Each component is oriented so that the fitted row
    with the largest absolute score on it has a positive score.
    """
    _, S, Vt = np.linalg.svd(rows, full_matrices=False)
    basis = Vt[:3]
    scores = rows @ basis.T
    signs = np.sign(scores[np.argmax(np.abs(scores), axis=0), np.arange(3)])
    return basis * signs[:, None], S


def _projection_hash(basis: np.ndarray, offset: np.ndarray | None = None) -> str:
    """SHA-256 identifying a display basis (and centring offset) without publishing it."""
    h = hashlib.sha256()
    for a in ([basis] if offset is None else [basis, offset]):
        h.update((np.round(a, DISPLAY_DECIMALS) + 0.0).astype("<f8").tobytes())
    return h.hexdigest()


def _source_hash(ex: Exporter, sources: list[str]) -> str:
    """SHA-256 over the (path, sha256) pairs of the source files of one export."""
    lines = [f"{s}:{ex.sources[s]['sha256']}" for s in sorted(set(sources))]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _notebook_stdout(path: Path) -> str:
    """All saved stream output of a notebook, used to compare against reported values."""
    cells = json.loads(path.read_text())["cells"]
    return "".join("".join(o.get("text", [])) for c in cells for o in c.get("outputs", []))


def _reported(text: str, pattern: str, what: str) -> list[float]:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"reference value not found in saved notebook output: {what}")
    return [float(g) for g in m.groups()]


def _must_match(what: str, computed: float, reported: float, tol: float) -> None:
    if abs(computed - reported) > tol:
        raise ValueError(f"{what}: computed {computed:.6f} does not reproduce the saved "
                         f"value {reported} (tolerance {tol})")


def _unit_directions(ex: Exporter) -> tuple[list[dict], np.ndarray, list[float], list[str]]:
    """The VECTOR_FIELD_DIRECTIONS at GEOMETRY_LAYER: entries, unit rows, raw norms, sources."""
    root = "representation_discovery"
    rows, raw_norms, entries, sources = [], [], [], []
    for d in VECTOR_FIELD_DIRECTIONS:
        if "file" in d:
            rels = [f"{root}/{d['file']}"]
            source = f"results/{rels[0]} @ {GEOMETRY_LAYER}"
            norm_kind = "norm of the saved mean-difference direction (saved unit-normalised)"
        else:
            folder, pattern = d["glob"]
            rels = sorted(f.relative_to(ex.results).as_posix()
                          for f in (ex.results / root / folder).glob(pattern))
            if not rels:
                raise FileNotFoundError(f"{root}/{folder}/{pattern}")
            source = f"results/{root}/{folder}/{pattern} ({len(rels)} files, averaged) @ {GEOMETRY_LAYER}"
            norm_kind = (f"norm of the mean of {len(rels)} saved unit-normalised mean-difference "
                         f"directions, before re-normalisation")
        # float32 mean as in control.III1_vector_control.control.load_averaged_*_vector
        vecs = [np.load(ex.src(r), allow_pickle=True).item()[GEOMETRY_LAYER] for r in rels]
        v = np.mean(np.stack(vecs), axis=0).astype(np.float32).astype(np.float64)
        sources += [f"results/{r}" for r in rels]
        raw_norms.append(float(np.linalg.norm(v)))
        rows.append(v / raw_norms[-1])
        entries.append({"id": d["id"], "label": d["label"], "family": d["family"],
                        "layer": GEOMETRY_LAYER, "source": source,
                        "condition_definition": d["definition"],
                        "raw_norm_definition": norm_kind})
    return entries, np.stack(rows), raw_norms, sources


def export_vector_field_3d(ex: Exporter) -> None:
    entries, U, raw_norms, sources = _unit_directions(ex)
    basis, S = _top3_basis(U)
    coords = U @ basis.T
    evr = S ** 2 / (S ** 2).sum()

    nb_rel = "EX1_T1_workload_analytics.ipynb"
    nb_out = _notebook_stdout(ex.src(nb_rel))
    sources.append(f"results/{nb_rel}")
    (captured_reported,) = _reported(
        nb_out, r"top-3 \(uncentered\) components: ([\d.]+)%", "top-3 variance captured")
    _must_match("top-3 variance captured (%)", 100 * evr[:3].sum(), captured_reported, 0.051)

    directions = []
    for e, xyz, raw, u in zip(entries, coords, raw_norms, U):
        directions.append({
            "id": e["id"], "label": e["label"], "family": e["family"], "layer": e["layer"],
            "display_endpoint": _rounded(xyz),
            "display_norm": _rounded(np.linalg.norm(xyz)),
            "raw_norm": _rounded(raw),
            "raw_norm_definition": e["raw_norm_definition"],
            "full_space_unit_norm": _rounded(np.linalg.norm(u)),
            "source": e["source"],
            "condition_definition": e["condition_definition"],
        })
    payload = {
        "layer": GEOMETRY_LAYER,
        "origin": {"id": "origin", "label": "Neutral origin (zero vector)", "display": [0.0, 0.0, 0.0]},
        "projection": {
            "method": "top-3 uncentered SVD of the unit-normalised directions listed under "
                      "'directions' (one basis for all of them)",
            "centering": "none: the direction set is not mean-centred, so the zero vector maps "
                         "to the display origin",
            "solver": "numpy.linalg.svd (LAPACK, exact), full_matrices=False",
            "random_state": None,
            "sign_convention": "each component is oriented so that the direction with the "
                               "largest absolute coordinate on it has a positive coordinate",
            "dimensions": 3,
            "n_directions_fit": len(directions),
            "full_space_dimensions": int(U.shape[1]),
            "components": "not published (three 5120-dimensional vectors); identified by projection_hash",
            "singular_values": _rounded(S[:3]),
            "explained_variance_ratio": _rounded(evr[:3]),
            "explained_variance_ratio_total": _rounded(evr[:3].sum()),
            "explained_variance_definition": "share of the total squared norm of the unit "
                                             "directions captured by each component (uncentered)",
            "reproduces": {"source": f"results/{nb_rel}",
                           "top3_variance_captured_percent_reported": captured_reported,
                           "top3_variance_captured_percent_computed": _rounded(100 * evr[:3].sum(), 4)},
            "source_hash": _source_hash(ex, sources),
            "projection_hash": _projection_hash(basis),
        },
        "directions": directions,
        "cosine_matrix": {
            "space": f"full {U.shape[1]}-dimensional activation space at {GEOMETRY_LAYER}",
            "ids": [d["id"] for d in directions],
            "values": _rounded(U @ U.T),
        },
    }
    ex.write_json(
        "geometry/vector_field_3d.json", payload,
        base_meta(
            "II", "Concept directions at one layer: 3D display projection and exact cosines",
            sources,
            model_config=f"mean-difference directions in the residual stream at {GEOMETRY_LAYER}, "
                         "all drawn from a common origin",
            feature_labels={d["id"]: d["label"] for d in directions},
            units={"display_endpoint": "coordinates of the unit-normalised direction in the shared "
                                       "3D display basis; its length (display_norm, 0-1) is the "
                                       "share of the direction that the display retains",
                   "raw_norm": "norm of the analysis input before the figure's unit normalisation "
                               "(see raw_norm_definition); not an activation-scale magnitude",
                   "full_space_unit_norm": "norm of the direction as used for the projection and "
                                           "the cosines (1 by construction)",
                   "cosine_matrix.values": "signed cosine similarity in the full activation "
                                           "space, -1 to 1; rows and columns follow cosine_matrix.ids"},
            aggregation="directions are the saved mean-difference vectors; gender and country are "
                        "averages over per-emotion / per-persona directions, as in "
                        "control/III1_vector_control/control.py; projection and cosines are "
                        "computed by this script and reproduce results/EX1_T1_workload_analytics.ipynb",
            caveats=[
                "This is a 3D display projection of 5120-dimensional directions. Angles and "
                "lengths in the display are approximate; exact alignment is given by cosine_matrix.",
                "Geometric alignment between directions does not establish an intervention "
                "effect: no causal claim follows from this figure.",
                "Directions with a small display_norm lie mostly outside the displayed subspace "
                "and their displayed angles are the least reliable.",
                "The saved mean-difference directions are unit-normalised; their original "
                "activation-scale magnitudes were not saved, so raw_norm is 1 for single "
                "contrasts and below 1 for averaged directions (it then measures how well the "
                "averaged directions agree).",
                "Directions are estimated from different contrasts and sample sizes on a single "
                "model, prompt template and image sample.",
            ]),
        inline_arrays=True)


def export_persona_composition(ex: Exporter) -> None:
    layer_dir = GEOMETRY_LAYER.rsplit("_D", 1)[0]
    sets = {"base": "base", "Germany": "extended_germany", "Nigeria": "extended_nigeria"}
    rels = {k: f"analytics/{v}/{layer_dir}/mean_vectors_{layer_dir}.npy" for k, v in sets.items()}
    means = {k: {pk: np.asarray(v, dtype=np.float64)
                 for pk, v in np.load(ex.src(r), allow_pickle=True).item().items()}
             for k, r in rels.items()}
    nb_rel = "EX1_T2_additivity.ipynb"
    nb_out = _notebook_stdout(ex.src(nb_rel))
    sources = [f"results/{r}" for r in rels.values()] + [f"results/{nb_rel}"]

    base_means = means["base"]
    country_means = {c: means[c] for c in PERSONA_COUNTRIES}
    fv = derive_raw_feature_vectors(base_means, country_means, PERSONA_EMOTIONS, PERSONA_GENDERS)
    grand_mean = fv["grand_mean"]

    # One display basis, fitted on the real observed condition means only.
    fit_labels = [pk for k in sets for pk in means[k]]
    X = np.stack([means[k][pk] for k in sets for pk in means[k]])
    offset = X.mean(axis=0)
    basis, S = _top3_basis(X - offset)
    evr = S ** 2 / (S ** 2).sum()
    point = lambda v: (v - offset) @ basis.T        # positions
    delta = lambda v: v @ basis.T                   # displacements

    orders = [o for r in (1, 2, 3) for o in itertools.permutations(COMPOSITION_FEATURES, r)]
    conditions, per_set = [], {}
    for country in PERSONA_COUNTRIES:
        for gender in PERSONA_GENDERS:
            for emotion in PERSONA_EMOTIONS:
                cid = f"{gender}_{emotion}_extended_{country.lower()}"
                observed = country_means[country][cid]
                comp = composition_components(fv, gender, emotion, country)

                # One point per feature set, summed in the project's order
                # (gender, emotion, country); every visiting order must reach it.
                reached = {}
                for r in (1, 2, 3):
                    for subset in itertools.combinations(COMPOSITION_FEATURES, r):
                        p = grand_mean + sum(comp[f] for f in subset)
                        reached[frozenset(subset)] = {
                            "point": p,
                            "display": _rounded(point(p)),
                            "cos": _rounded(shift_cosine(p, observed, grand_mean)),
                            "dist": _rounded(np.linalg.norm(p - observed)),
                            "journey": _rounded(np.linalg.norm(p - grand_mean)),
                        }
                        per_set.setdefault("+".join(subset), []).append(
                            (country, reached[frozenset(subset)]))

                by_order = {}
                for order in orders:
                    p, stages = grand_mean, []
                    for i, f in enumerate(order, start=1):
                        p = p + comp[f]
                        hit = reached[frozenset(order[:i])]
                        if not np.allclose(p, hit["point"], rtol=0, atol=1e-9):
                            raise ValueError(f"{cid}: order {order} does not reach the same point")
                        stages.append({"included": list(order[:i]), "display": hit["display"],
                                       "full_space_cosine_to_observed": hit["cos"],
                                       "full_space_distance_to_observed": hit["dist"]})
                    by_order[">".join(order)] = {
                        "stages": stages,
                        "final_full_space_cosine_to_observed": hit["cos"],
                        "final_full_space_distance_to_observed": hit["dist"],
                        "total_full_space_journey_from_start": hit["journey"],
                    }
                full = reached[frozenset(COMPOSITION_FEATURES)]
                conditions.append({
                    "id": cid,
                    "labels": {"gender": gender, "emotion": EMOTION_LABELS[emotion], "country": country},
                    "grand_mean_display": _rounded(point(grand_mean)),
                    "observed_display": _rounded(point(observed)),
                    "observed_full_space_distance_from_start": _rounded(np.linalg.norm(observed - grand_mean)),
                    "components": {
                        "gender": {"vector": "v_gender", "sign": 1 if gender == "female" else -1,
                                   "coefficient": COMPOSITION_COEFFICIENTS["gender"],
                                   "display_delta": _rounded(delta(comp["gender"])),
                                   "full_space_norm": _rounded(np.linalg.norm(comp["gender"]))},
                        "emotion": {"vector": f"v_emotion[{emotion}]",
                                    "coefficient": COMPOSITION_COEFFICIENTS["emotion"],
                                    "display_delta": _rounded(delta(comp["emotion"])),
                                    "full_space_norm": _rounded(np.linalg.norm(comp["emotion"]))},
                        "country": {"vector": f"v_country[{country}]",
                                    "coefficient": COMPOSITION_COEFFICIENTS["country"],
                                    "display_delta": _rounded(delta(comp["country"])),
                                    "full_space_norm": _rounded(np.linalg.norm(comp["country"]))},
                    },
                    "predicted_display": full["display"],
                    "final_full_space_cosine_to_observed": full["cos"],
                    "final_full_space_distance_to_observed": full["dist"],
                    "predictions_by_order": by_order,
                })

    def summarise(items):
        return {"n_conditions": len(items),
                "mean_full_space_cosine_to_observed": _rounded(np.mean([h["cos"] for _, h in items])),
                "mean_full_space_distance_to_observed": _rounded(np.mean([h["dist"] for _, h in items]))}

    by_set = {}
    for key, items in per_set.items():
        by_set[key] = summarise(items) | {
            "by_country": {c: summarise([i for i in items if i[0] == c]) for c in PERSONA_COUNTRIES}}
    full_key = "+".join(COMPOSITION_FEATURES)

    # The saved notebook output is the pre-existing result this export must reproduce.
    cos_rep, resid_rep = _reported(
        nb_out, r"Using ±v_gender/2\s+\(Stage-2-validated scaling\):\s+mean cos = ([\d.]+)\s+"
                r"mean residual norm = ([\d.]+)", "Stage 3 mean cosine and residual norm")
    pc1_rep, pc2_rep = _reported(nb_out, r"PC1 ([\d.]+)% · PC2 ([\d.]+)%", "PCA explained variance")
    (gender_norm_rep,) = _reported(nb_out, r"\|\|v_gender\|\|\s+=\s+([\d.]+)", "norm of v_gender")
    _must_match("full-composition mean cosine",
                by_set[full_key]["mean_full_space_cosine_to_observed"], cos_rep, 5.1e-5)
    _must_match("full-composition mean residual norm",
                by_set[full_key]["mean_full_space_distance_to_observed"], resid_rep, 5.1e-4)
    _must_match("PC1 explained variance (%)", 100 * evr[0], pc1_rep, 0.051)
    _must_match("PC2 explained variance (%)", 100 * evr[1], pc2_rep, 0.051)
    _must_match("norm of v_gender", float(np.linalg.norm(fv["gender"])), gender_norm_rep, 5.1e-3)

    def vector_entry(v, definition):
        return {"definition": definition, "full_space_norm": _rounded(np.linalg.norm(v)),
                "display_delta": _rounded(delta(v))}

    payload = {
        "layer": GEOMETRY_LAYER,
        "projection": {
            "method": "PCA fit on real observed condition means only",
            "fit_on": f"{len(fit_labels)} observed persona-condition means (16 base, 16 Germany, "
                      "16 Nigeria); the grand mean, components and predictions are transformed "
                      "afterwards and never enter the fit",
            "fit_conditions": fit_labels,
            "centering": "mean of the fitted condition means is subtracted (standard PCA)",
            "solver": "numpy.linalg.svd of the centred matrix (LAPACK, exact), full_matrices=False",
            "random_state": None,
            "sign_convention": "each component is oriented so that the fitted condition with the "
                               "largest absolute score on it has a positive score",
            "dimensions": 3,
            "full_space_dimensions": int(X.shape[1]),
            "components": "not published (three 5120-dimensional vectors); identified by projection_hash",
            "singular_values": _rounded(S[:3]),
            "explained_variance_ratio": _rounded(evr[:3]),
            "explained_variance_ratio_total": _rounded(evr[:3].sum()),
            "transform": "positions: (h - fit mean) . components; displacements (display_delta): "
                         "v . components. The map is affine, so a displayed path is the exact "
                         "shadow of the full-space path",
            "source_hash": _source_hash(ex, sources),
            "projection_hash": _projection_hash(basis, offset),
        },
        "composition_convention": {
            "path": "grand mean -> +-1/2 gender -> emotion -> country -> predicted compound persona",
            "start": "grand_mean",
            "gender": "signed half-gender vector: +1/2 v_gender for female, -1/2 v_gender for male",
            "emotion": "condition-matched emotion component v_emotion[emotion], coefficient 1",
            "country": "condition-matched country component v_country[country], coefficient 1",
            "coefficients": dict(COMPOSITION_COEFFICIENTS),
            "calculation_space": "raw activation space",
            "scaling": "raw naturally scaled mean-difference vectors; no component is "
                       "unit-normalised for the composition",
            "order_invariance": "vector addition commutes: the visiting order changes the "
                                "intermediate stages, not the final point of a given feature set",
            "cosine_definition": "cos(prediction - grand_mean, observed - grand_mean)",
            "distance_definition": "Euclidean norm of (prediction - observed)",
            "journey_definition": "Euclidean norm of (final prediction - grand_mean)",
        },
        "feature_vectors": {
            "provenance": "derived from the saved condition means by "
                          "analytics/II1_persona_analytics/additivity.py::derive_raw_feature_vectors, "
                          "the definition used in results/EX1_T2_additivity.ipynb",
            "grand_mean": {"definition": "mean of the 16 base persona-condition means",
                           "full_space_norm": _rounded(np.linalg.norm(grand_mean)),
                           "display": _rounded(point(grand_mean))},
            "gender": vector_entry(fv["gender"], "mean of the 8 female base condition means minus "
                                                 "mean of the 8 male base condition means"),
            "emotion": {e: vector_entry(fv["emotion"][e],
                                        f"mean of the base condition means with emotion {e} "
                                        "(female and male) minus the grand mean")
                        for e in PERSONA_EMOTIONS},
            "country": {c: vector_entry(fv["country"][c],
                                        f"mean over the 16 gender x emotion personas of ({c}-extended "
                                        "condition mean minus matching base condition mean)")
                        for c in PERSONA_COUNTRIES},
        },
        "feature_orders": [">".join(o) for o in orders],
        "aggregate": {
            "full_composition": by_set[full_key],
            "by_feature_set": by_set,
            "reproduces": {
                "source": f"results/{nb_rel} (Stage 3, +-1/2 v_gender scaling)",
                "mean_cosine_reported": cos_rep,
                "mean_residual_norm_reported": resid_rep,
                "pca_explained_variance_percent_reported": {"PC1": pc1_rep, "PC2": pc2_rep},
            },
        },
        "n_conditions_expected": len(PERSONA_COUNTRIES) * len(PERSONA_GENDERS) * len(PERSONA_EMOTIONS),
        "conditions": conditions,
    }
    ex.write_json(
        "geometry/persona_composition_paths.json", payload,
        base_meta(
            "II", "Additive composition of persona feature vectors: staged paths for every "
                  "feature order", sources,
            model_config=f"mean residual-stream activation per persona condition at {GEOMETRY_LAYER}; "
                         "compound personas are gender x emotion x country",
            feature_labels={"id": "persona key <gender>_<emotion>_extended_<country>, as saved; "
                                  "the emotion key 'sad' is labelled 'sadness'",
                            "features": list(COMPOSITION_FEATURES)},
            units={"*_display, display, display_delta": "coordinates in the shared 3D PCA basis, "
                                                        "in activation units",
                   "full_space_norm, *_distance_*, *_journey_*": "Euclidean norm in the full "
                                                                 "activation space, activation units",
                   "*_cosine_to_observed": "cosine in the full activation space, -1 to 1 "
                                           "(see composition_convention.cosine_definition)",
                   "coefficient": "multiplier applied to the raw feature vector"},
            aggregation="one entry per compound condition and per ordered non-empty subset of "
                        "(gender, emotion, country); 'aggregate' holds means over the 32 "
                        "conditions and reproduces the Stage 3 result of "
                        "results/EX1_T2_additivity.ipynb",
            caveats=[
                "This export visualises representational geometry and approximate additivity "
                "only. It is not a causal-control result: nothing here was obtained by "
                "intervening on the model.",
                "Display coordinates are a 3D projection; cosines, distances and norms are "
                "computed in the full 5120-dimensional space and are the values to quote.",
                "The feature vectors are averages over the same conditions they are used to "
                "predict (in-sample), so agreement is an upper bound on transfer to unseen conditions.",
                "Partial feature sets are compared with the fully specified observed persona, so "
                "their cosines and distances describe how much of the gap that subset closes, "
                "not a prediction of a matching partial persona.",
                "Condition means come from a single activation collection run on one model, "
                "prompt template and image sample.",
            ]),
        inline_arrays=True)


# UAP targets (attack name -> direction id in VECTOR_FIELD_DIRECTIONS), from the
# --vector_path arguments of the UAP training jobs.
UAP_TARGET_DIRECTIONS = {
    "interest": "interest_blank_high_vs_low",
    "excited_vs_angry": "emotion_excited_vs_angry",
    "workload": "workload_overwhelming_vs_minimal",
}
COSINE_BASELINE_DIRECTIONS = ["interest_global_high_vs_low", "gender_female_vs_male",
                              "country_germany", "country_nigeria"]


def export_uap_target_cosines(ex: Exporter) -> None:
    entries, U, _, sources = _unit_directions(ex)
    ids = [e["id"] for e in entries]
    labels = {e["id"]: e["label"] for e in entries}
    C = U @ U.T
    rows = []
    for d in list(UAP_TARGET_DIRECTIONS.values()) + COSINE_BASELINE_DIRECTIONS:
        row = {"direction": d, "direction_label": labels[d],
               "role": "uap_target" if d in UAP_TARGET_DIRECTIONS.values() else "baseline"}
        for attack, target in UAP_TARGET_DIRECTIONS.items():
            row[f"cos_to_{attack}_target"] = C[ids.index(d), ids.index(target)]
        rows.append(row)
    dim = U.shape[1]
    ex.write_table(
        "geometry/uap_target_direction_cosines.csv", pd.DataFrame(rows),
        base_meta(
            "II", "Cosine similarity between the three UAP target directions, with baselines",
            sources,
            model_config=f"mean-difference directions in the residual stream at {GEOMETRY_LAYER}",
            feature_labels={"uap_targets": UAP_TARGET_DIRECTIONS,
                            **{d: labels[d] for d in COSINE_BASELINE_DIRECTIONS}},
            units="signed cosine similarity in the full activation space, -1 to 1",
            aggregation="computed by this script from the saved directions (same values as "
                        "geometry/vector_field_3d.json)",
            random_direction_null={
                "method": f"analytic: the cosine between a fixed direction and a uniformly random "
                          f"direction in {dim} dimensions has standard deviation 1/sqrt({dim})",
                "sd": round(1 / np.sqrt(dim), 6),
                "abs_cosine_95_percent_bound": round(1.96 / np.sqrt(dim), 6)},
            caveats=["The interest UAP targets the blank-prompt interestingness direction, not "
                     "the pooled (global) one; the global direction is listed as a baseline only.",
                     "A random direction is a weak null: learned directions share variance, so "
                     "the gender and country rows are the more informative baselines.",
                     "Cosine similarity between directions does not establish that the "
                     "perturbations act on a shared state."]))


# ── Tier IV: statistics on the transfer evaluation ───────────────────────────

BOOTSTRAP_REPS = 4000
# Explanations that describe the image itself as degraded: the model may then be
# rating the perturbation, not the scene.
DEGRADATION_RE = (r"degrad|distort|overlay|illegible|pixelat|noise|noisy|corrupt|glitch|blurr|"
                  r"artifact|poor quality|low quality|discern|profan|chaotic")


def _mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean with a 95% percentile bootstrap interval over rows (images)."""
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(values), (BOOTSTRAP_REPS, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def _transfer_predictions(ex: Exporter, task: str) -> tuple[pd.DataFrame, list[str]]:
    """Perturbed-condition rows of one task; noise-control rows are appended when saved."""
    cols = ["sample_id", "attack", "epsilon", "delta", "explanation"]
    rel = f"generalisation/{task}/predictions.csv"
    frames, sources = [pd.read_csv(ex.src(rel), usecols=cols)], [f"results/{rel}"]
    try:
        rel_n = f"generalisation_noise/{task}/predictions.csv"
        frames.append(pd.read_csv(ex.src(rel_n), usecols=cols))
        sources.append(f"results/{rel_n}")
    except FileNotFoundError:
        pass
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["attack"] != "clean") & df["delta"].notna()].copy()
    df["mentions_degradation"] = df["explanation"].fillna("").str.contains(DEGRADATION_RE, case=False)
    return df, sources


def export_transfer_statistics(ex: Exporter) -> None:
    rng = np.random.default_rng(20261008)
    rows, contrasts, sources = [], [], []
    for task, info in GENERALISATION_TASKS.items():
        try:
            df, src = _transfer_predictions(ex, task)
        except FileNotFoundError as e:
            ex.skipped.append({"export": f"attack/transfer_statistics/{task}",
                               "missing_source": f"results/{e}"})
            continue
        sources += src
        width = info["scale_max"] - info["scale_min"]
        df["shift"] = df["delta"] / width
        for (attack, eps), g in df.groupby(["attack", "epsilon"], sort=True):
            d = g["shift"].to_numpy()
            up, down = float((d > 0).mean()), float((d < 0).mean())
            mean, lo, hi = _mean_ci(d, rng)
            clear = g.loc[~g["mentions_degradation"], "shift"].to_numpy()
            c_mean, c_lo, c_hi = _mean_ci(clear, rng)
            rows.append({
                "task": task, "attack": attack, "epsilon": eps, "n": len(d),
                "signed_shift": mean, "signed_shift_ci_low": lo, "signed_shift_ci_high": hi,
                "signed_shift_nonzero": bool(lo > 0 or hi < 0),
                "mean_abs_shift": float(np.abs(d).mean()),
                "share_up": up, "share_down": down, "share_changed": up + down,
                "dominant_direction": "up" if up >= down else "down",
                "dominant_direction_share_of_changes": max(up, down) / (up + down) if up + down else np.nan,
                "share_mentioning_degradation": float(g["mentions_degradation"].mean()),
                "n_not_mentioning_degradation": len(clear),
                "signed_shift_not_mentioning": c_mean,
                "signed_shift_not_mentioning_ci_low": c_lo,
                "signed_shift_not_mentioning_ci_high": c_hi,
            })
        # Same images under two UAPs: difference in absolute shift against the interest UAP.
        wide = df[df["attack"].isin(UAP_TARGET_DIRECTIONS)].pivot_table(
            index="sample_id", columns=["attack", "epsilon"], values="shift").abs().dropna()
        for eps in sorted({e for _, e in wide.columns}):
            for attack in ("excited_vs_angry", "workload"):
                diff = (wide[(attack, eps)] - wide[("interest", eps)]).to_numpy()
                mean, lo, hi = _mean_ci(diff, rng)
                contrasts.append({"task": task, "epsilon": eps, "contrast": f"{attack} minus interest",
                                  "n_paired": len(diff), "diff_mean_abs_shift": mean,
                                  "ci_low": lo, "ci_high": hi, "nonzero": bool(lo > 0 or hi < 0)})
    if not rows:
        raise FileNotFoundError("generalisation/*/predictions.csv")
    common = dict(
        model_config="UAPs trained on the main image set (target layer language_29) applied "
                     "unchanged to three unseen datasets and judgement tasks; every perturbed "
                     "image is compared with its own clean version",
        feature_labels={t: f"{i['label']} ({i['scale_min']}-{i['scale_max']}), {i['dataset']}"
                        for t, i in GENERALISATION_TASKS.items()},
        bootstrap={"replicates": BOOTSTRAP_REPS, "resampled_unit": "image", "interval": "95% percentile",
                   "seed": 20261008},
    )
    ex.write_table(
        "attack/transfer_shift_statistics.csv", pd.DataFrame(rows),
        base_meta(
            "IV", "UAP transfer: label shift per task, attack and budget, with bootstrap intervals",
            sources,
            units={"signed_shift": "mean of (perturbed - clean label) / (scale_max - scale_min)",
                   "mean_abs_shift": "mean of the absolute value of the same quantity",
                   "share_up / share_down / share_changed": "share of images, 0-1",
                   "dominant_direction_share_of_changes": "max(share_up, share_down) / share_changed; "
                                                          "0.5 means changes go both ways equally",
                   "share_mentioning_degradation": "share of explanations matching degradation_pattern",
                   "signed_shift_not_mentioning": "signed_shift over the images whose explanation "
                                                  "does not match degradation_pattern"},
            aggregation="per task, attack and epsilon over the frozen sample; intervals by "
                        "bootstrap over images; no averaging across tasks",
            degradation_pattern=DEGRADATION_RE,
            caveats=[EPS_CAVEAT,
                     "One training run per UAP and one evaluation run; the interval reflects "
                     "image sampling only.",
                     "Budgets are 0.1, 0.5, 1.0 and 2.0 only: any statement about the budget at "
                     "which an effect sets in rests on a single step.",
                     "At epsilon = 0.5 the workload UAP also moved hidden states further toward "
                     "its target than the other two (attack/uap_alignment.csv), so an earlier "
                     "onset may reflect how reachable the direction is.",
                     "The degradation split is a proxy from free text: an image can be perceived "
                     "as degraded without the explanation saying so.",
                     "Rows with attack = noise_seed* are magnitude-matched noise controls and are "
                     "present only once those runs are saved; without them a shift cannot be "
                     "separated from generic image corruption.",
                     "Source images are not redistributed; only aggregate model outputs are exported."],
            **common))
    ex.write_table(
        "attack/transfer_paired_contrasts.csv", pd.DataFrame(contrasts),
        base_meta(
            "IV", "UAP transfer: absolute label shift of each UAP against the interest UAP, same images",
            sources,
            units={"diff_mean_abs_shift": "mean over images of |shift under the named UAP| minus "
                                          "|shift under the interest UAP|, as a fraction of the scale"},
            aggregation="paired over images within a task and epsilon; bootstrap over images",
            caveats=["One training run per UAP; see attack/transfer_shift_statistics.csv."],
            **common))


def _ols(columns: list[np.ndarray], y: np.ndarray) -> tuple[float, np.ndarray]:
    X = np.column_stack([np.ones(len(y))] + columns)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum(), beta


def export_arousal_valence(ex: Exporter) -> None:
    rel = "generalisation/moral_evaluation/predictions.csv"
    p = pd.read_csv(ex.src(rel), usecols=["sample_id", "attack", "epsilon", "delta",
                                          "clean_model_label", "human_arousal", "human_valence"])
    rng = np.random.default_rng(20261008)
    reps = 2000
    rows = []
    for attack in UAP_TARGET_DIRECTIONS:
        x = p[p["attack"] == attack]
        wide = x.pivot(index="sample_id", columns="epsilon", values="delta").dropna()
        img = x.drop_duplicates("sample_id").set_index("sample_id").loc[wide.index]
        eps = wide.columns.to_numpy(dtype=float)
        Y = wide.to_numpy(dtype=float)

        def fits(i):
            k = len(eps)
            ar, va = np.repeat(img["human_arousal"].to_numpy()[i], k), np.repeat(img["human_valence"].to_numpy()[i], k)
            cm, e, y = np.repeat(img["clean_model_label"].to_numpy()[i], k), np.tile(eps, len(i)), Y[i].ravel()
            base, _ = _ols([e, cm], y)
            r_ar, b_ar = _ols([ar, e, ar * e, cm], y)
            r_va, _ = _ols([va, e, va * e, cm], y)
            r_both, b_both = _ols([ar, e, ar * e, va, va * e, cm], y)
            return base, r_ar, r_va, r_both, b_ar[3], b_both[3], b_both[5]

        base, r_ar, r_va, r_both, c_ar, c_ar_joint, c_va_joint = fits(np.arange(len(img)))
        boot = np.array([fits(rng.integers(0, len(img), len(img)))[4:] for _ in range(reps)])
        (a_lo, j_lo, v_lo), (a_hi, j_hi, v_hi) = np.percentile(boot, [2.5, 97.5], axis=0)
        rows.append({
            "attack": attack, "n_images": len(img), "n_rows": int(Y.size),
            "r2_epsilon_and_clean_label": base,
            "r2_with_arousal_terms": r_ar, "r2_added_by_arousal": r_ar - base,
            "r2_with_valence_terms": r_va, "r2_added_by_valence": r_va - base,
            "r2_with_both": r_both,
            "arousal_x_epsilon": c_ar, "arousal_x_epsilon_ci_low": a_lo, "arousal_x_epsilon_ci_high": a_hi,
            "arousal_x_epsilon_given_valence": c_ar_joint,
            "arousal_x_epsilon_given_valence_ci_low": j_lo, "arousal_x_epsilon_given_valence_ci_high": j_hi,
            "valence_x_epsilon_given_arousal": c_va_joint,
            "valence_x_epsilon_given_arousal_ci_low": v_lo, "valence_x_epsilon_given_arousal_ci_high": v_hi,
            "corr_arousal_valence": float(img["human_arousal"].corr(img["human_valence"])),
        })
    info = GENERALISATION_TASKS["moral_evaluation"]
    ex.write_table(
        "attack/moral_arousal_valence_regression.csv", pd.DataFrame(rows),
        base_meta(
            "IV", "Moral evaluation: does image arousal or image valence moderate the UAP shift?",
            [f"results/{rel}"],
            model_config="moral evaluation task (SMID), UAPs with target layer language_29, "
                         "four budgets per image",
            feature_labels={"human_arousal / human_valence": "normative human ratings of the image (SMID)",
                            "delta": "perturbed minus clean model morality label"},
            units={"r2_*": "share of variance of delta explained, 0-1",
                   "*_x_epsilon": "interaction coefficient, morality-scale units per rating unit "
                                  "and unit of epsilon"},
            aggregation="ordinary least squares on delta. Base model: epsilon + clean label. "
                        "Arousal model adds arousal and arousal x epsilon; valence model adds "
                        "valence and valence x epsilon; the joint model adds all four. Intervals "
                        f"by bootstrap over images ({reps} replicates, all four budgets of an "
                        "image resampled together, seed 20261008)",
            caveats=["Exploratory and post hoc; not corrected for multiple comparisons.",
                     "Most of the explained variance comes from epsilon and the clean label, "
                     "not from arousal or valence.",
                     "Arousal and valence are correlated in this sample, so their separate "
                     "contributions are only partly identifiable.",
                     "No noise baseline: emotionally intense images may simply be more fragile "
                     "under any perturbation.",
                     f"Human ratings come from {info['dataset']} and are not redistributed here."]))


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


# results/extra_checks/logit_lens/<file>.json, one per released causal series
LOGIT_LENS_SERIES = [
    ("interest", "language_29", "interest_blank_language_29"),
    ("excited_vs_angry", "language_29", "excited_vs_angry_language_29"),
    ("workload", "language_24", "workload_language_24"),
    ("workload", "language_29", "workload_language_29"),
    ("country_nigeria", "language_23", "country_nigeria_language_23"),
]


def _display_token(token: str) -> str:
    """Token text with whitespace and control characters made visible."""
    out = token.replace(" ", "\u2423").replace("\n", "\u23ce").replace("\t", "\u21e5")
    return "".join(c if c.isprintable() else f"\\x{ord(c):02x}" for c in out)


def export_logit_lens(ex: Exporter) -> None:
    labels = {d: l for d, l, _ in DOSE_RESPONSE_RUNS}
    series, sources = [], []
    for direction, layer, name in LOGIT_LENS_SERIES:
        rel = f"extra_checks/logit_lens/{name}.json"
        try:
            lens = json.loads(ex.src(rel).read_text())
        except FileNotFoundError as e:
            ex.skipped.append({"export": f"causal/logit_lens/{name}", "missing_source": f"results/{e}"})
            continue
        sources.append(f"results/{rel}")

        def tokens(items):
            return [{"rank": i, "token": t["token"], "display": _display_token(t["token"]),
                     "incomplete_utf8": "\ufffd" in t["token"],
                     "delta_logit_per_unit": round(t["delta_logit"], 6)}
                    for i, t in enumerate(items, start=1)]

        series.append({
            "direction": direction, "direction_label": labels.get(direction), "layer": layer,
            "direction_source": lens["source"], "description": lens["description"],
            "direction_norm": lens["vector_norm"],
            "pushed_up": tokens(lens["top_positive_tokens"]),
            "pushed_down": tokens(lens["top_negative_tokens"]),
        })
    if not series:
        raise FileNotFoundError("extra_checks/logit_lens/*.json")
    alphas = sorted({a for _, _, d in DOSE_RESPONSE_RUNS
                     for a in pd.read_csv(ex.results / d / "control_results.csv", usecols=["alpha"])["alpha"].unique()})
    ex.write_json(
        "causal/logit_lens_tokens.json",
        {"released_alphas": alphas, "tokenizer": MODEL_ID, "series": series},
        base_meta(
            "III", "Vocabulary tokens each intervention direction pushes up or down (logit lens)",
            sources,
            model_config="logit lens: the unit-normalised direction is multiplied by the weight of "
                         "the final RMSNorm and by the unembedding matrix, "
                         "delta_logit = lm_head @ (norm_weight * direction). No image, prompt or "
                         "forward pass is involved",
            feature_labels={f"{d} @ {l}": labels.get(d) for d, l, _ in LOGIT_LENS_SERIES},
            units={"delta_logit_per_unit": "change in a token's logit per unit of the direction "
                                           "under the logit-lens approximation; not a probability",
                   "rank": "1 = largest absolute change in that direction; 30 tokens per side, as saved",
                   "display": "token with space shown as U+2423, newline as U+23CE and "
                              "non-printable characters as \\xNN"},
            aggregation="token lists as saved by runners/run_logit_lens_evidence.py; nothing is "
                        "recomputed here",
            caveats=[
                "These are not next-token probabilities and not measured under intervention. The "
                "model's actual next-token distribution at each alpha was not saved.",
                "The logit lens applies the unembedding directly to a direction read at an "
                "intermediate layer and ignores every layer in between; it is a qualitative "
                "reading of what the direction resembles in vocabulary space.",
                "Multiplying delta_logit_per_unit by alpha is a linear extrapolation, not a result.",
                "Token lists are unfiltered model vocabulary: they contain word fragments, "
                "tokens from many languages and incomplete byte sequences (incomplete_utf8), and "
                "have not been reviewed for content.",
            ]),
        inline_arrays=True)


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
        # the workload UAP was evaluated on this task in a later run, into its own folder
        rel_w = "attack_eval_projected/interestingness_workload/summary.csv"
        if (ex.results / rel_w).is_file():
            df = pd.concat([df, pd.read_csv(ex.src(rel_w))], ignore_index=True)
            sources.append(f"results/{rel_w}")
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


# ── Tier IV: example gallery ─────────────────────────────────────────────────

GALLERY_EPSILONS = ["0.10", "0.50", "1.00", "2.00"]
GALLERY_RANK_EPSILON = "1.00"
GALLERY_N_DOMINANT, GALLERY_N_OPPOSITE = 4, 1
GALLERY_MAX_SIDE = 336          # as in attack/runners/run_behavioral_eval.py
GALLERY_EVAL_DIR = "attack_eval_projected/interestingness"
# Targets evaluated in a later run keep their outputs in their own folder.
GALLERY_EVAL_DIRS = {"workload": "attack_eval_projected/interestingness_workload"}
# Evaluation text of perturbed images, saved by later runs of the same runner.
GALLERY_TEXT_DIRS = ["attack_eval_projected/interestingness_gallery_text",
                     "attack_eval_projected/interestingness_workload"]


def _select_gallery(rank: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    From a ranking (columns rank, delta; best first): the highest-ranked shifts
    in the dominant direction (sign of the mean shift over all ranked items)
    plus the highest-ranked shift the other way. If one side has too few, the
    next-ranked items fill the set.
    """
    dominant = "increase" if rank["delta"].mean() >= 0 else "decrease"
    up, down = rank[rank["delta"] > 0], rank[rank["delta"] < 0]
    main, other = (up, down) if dominant == "increase" else (down, up)
    chosen = pd.concat([main.head(GALLERY_N_DOMINANT), other.head(GALLERY_N_OPPOSITE)])
    n = GALLERY_N_DOMINANT + GALLERY_N_OPPOSITE
    if len(chosen) < n:
        chosen = pd.concat([chosen, rank[~rank["rank"].isin(chosen["rank"])].head(n - len(chosen))])
    return chosen.sort_values("rank"), dominant


def _webp_lossless(pixel_values: np.ndarray) -> bytes:
    """Model-input tensor (3, H, W) in [-1, 1] as a lossless 8-bit WebP."""
    import io
    from PIL import Image
    rgb = np.clip(np.rint((pixel_values * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb.transpose(1, 2, 0), "RGB").save(buf, "WEBP", lossless=True, quality=50, method=4)
    return buf.getvalue()


def export_uap_gallery(ex: Exporter) -> None:
    try:
        from transformers import AutoImageProcessor
        from utils.image_utils import preprocess_image
        from utils.paths import LOCAL_MODEL_REPO
    except ImportError as e:
        ex.skipped.append({"export": "uap_gallery", "missing_source": f"python package: {e.name}"})
        return
    clean_rel = "selected_uniform_total_500.pkl"
    clean = pd.read_pickle(ex.src(clean_rel, root="data")).set_index("filename")
    sources = [f"data/{clean_rel}"]
    score_of = {lab: i for i, lab in enumerate(INTEREST_LABELS, start=1)}

    targets = []
    for target, (label, layer, _) in UAP_TARGETS.items():
        frames = {}
        try:
            for eps in GALLERY_EPSILONS:
                rel = f"{GALLERY_EVAL_DIRS.get(target, GALLERY_EVAL_DIR)}/{target}_eps{eps}_labels.csv"
                frames[eps] = pd.read_csv(ex.src(rel)).set_index("filename")
                sources.append(f"results/{rel}")
        except FileNotFoundError:
            continue        # this UAP has no saved evaluation on the interestingness task yet

        # Perturbed evaluation text, where a later run saved it; used only if that
        # run reproduced the released rating.
        text = {}
        for d in GALLERY_TEXT_DIRS:
            for eps in GALLERY_EPSILONS:
                rel = f"{d}/{target}_eps{eps}_labels.csv"
                if (ex.results / rel).is_file():
                    t = pd.read_csv(ex.src(rel))
                    if "perturbed_explanation" in t.columns:
                        sources.append(f"results/{rel}")
                        for r in t.itertuples():
                            text[(r.filename, eps)] = (r.perturbed_label, r.perturbed_explanation)

        # Ranking over all valid held-out images at the ranking budget.
        ok = pd.concat([f["parse_ok"] for f in frames.values()], axis=1).all(axis=1)
        at = frames[GALLERY_RANK_EPSILON][ok]
        rank = pd.DataFrame({"sample_id": [f.rsplit(".", 1)[0] for f in at.index],
                             "filename": at.index, "clean_score": at["clean_score"].to_numpy(),
                             "score": at["perturbed_score"].to_numpy(),
                             "delta": at["score_delta"].to_numpy()})
        rank["abs_delta"] = rank["delta"].abs()
        rank["crosses_category"] = rank["delta"] != 0
        rank = rank.sort_values(["abs_delta", "sample_id"], ascending=[False, True]).reset_index(drop=True)
        rank.insert(0, "rank", rank.index + 1)
        chosen, dominant = _select_gallery(rank)

        processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_REPO)
        deltas = {}
        for eps in GALLERY_EPSILONS:
            rel = f"universal_perturbation_projected/{target}/eps{eps}/delta.npy"
            deltas[eps] = np.load(ex.src(rel)).astype(np.float32)[0]
            sources.append(f"results/{rel}")

        cards = []
        for row in chosen.itertuples():
            img_rel = f"imagesDemographics/{row.filename}"
            image = preprocess_image(ex.src(img_rel, root="data"), max_side=GALLERY_MAX_SIDE)
            sources.append(f"data/{img_rel}")
            pv = processor(images=image, return_tensors="pt")["pixel_values"][0].float().numpy()
            base = f"uap_gallery/{target}/{row.sample_id}"
            info = dict(width=int(pv.shape[2]), height=int(pv.shape[1]), target=target, sample_id=row.sample_id)
            ex.write_asset(f"{base}/clean.webp", _webp_lossless(pv), epsilon=None, **info)
            c = clean.loc[row.filename]
            steps = []
            for eps in GALLERY_EPSILONS:
                ex.write_asset(f"{base}/eps_{eps}.webp",
                               _webp_lossless(np.clip(pv + deltas[eps], -1.0, 1.0)), epsilon=float(eps), **info)
                r = frames[eps].loc[row.filename]
                lab, expl = text.get((row.filename, eps), (None, None))
                has_text = lab == r["perturbed_label"] and isinstance(expl, str)
                steps.append({
                    "epsilon": float(eps), "epsilon_fraction_of_pixel_range": float(eps) / PIXEL_RANGE_WIDTH,
                    "image": f"assets/{base}/eps_{eps}.webp",
                    "label": r["perturbed_label"], "score": int(r["perturbed_score"]),
                    "parse_ok": bool(r["parse_ok"]),
                    "delta_from_clean": int(r["score_delta"]),
                    "text": expl if has_text else None,
                    "text_status": "available" if has_text else "pending",
                })
            cards.append({
                "sample_id": row.sample_id, "rank": int(row.rank),
                "selected_as": "increase" if row.delta > 0 else "decrease",
                "clean": {"image": f"assets/{base}/clean.webp", "label": c["interestingness_label"],
                          "score": score_of[c["interestingness_label"]], "parse_ok": True,
                          "text": c["explanation"], "text_status": "available"},
                "steps": steps,
            })
        sel, all_d = chosen["delta"].to_numpy(), rank["delta"].to_numpy()
        stats = lambda d: {"n": int(len(d)), "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
                           "mean_abs_delta": float(np.abs(d).mean()), "median_abs_delta": float(np.median(np.abs(d)))}
        targets.append({
            "target": target, "target_label": label, "layer": layer,
            "target_direction": UAP_TARGET_DIRECTIONS.get(target),
            "evaluation_source": f"results/{GALLERY_EVAL_DIRS.get(target, GALLERY_EVAL_DIR)}",
            "n_valid_images": int(len(rank)), "n_selected": int(len(cards)),
            "dominant_direction": dominant,
            "selected_shift_at_ranking_epsilon": stats(sel),
            "all_images_shift_at_ranking_epsilon": stats(all_d),
            "selection_includes_every_larger_shift": bool(
                np.abs(sel).min() >= np.abs(rank.loc[~rank["sample_id"].isin(chosen["sample_id"]), "delta"]).max()),
            "text_pending": sum(s["text_status"] == "pending" for c in cards for s in c["steps"]),
            "cards": cards,
            "ranking": rank.drop(columns="filename").to_dict("records"),
        })
    if not targets:
        raise FileNotFoundError(f"{GALLERY_EVAL_DIR}/*_labels.csv")
    ex.write_json(
        "attack/uap_gallery.json",
        {"task": "interestingness", "epsilons": [float(e) for e in GALLERY_EPSILONS],
         "ranking_epsilon": float(GALLERY_RANK_EPSILON),
         "panel_label": "Largest observed rating shifts in the dominant direction and the largest opposite shift at "
                        f"ε = {GALLERY_RANK_EPSILON}; not representative examples",
         "selection_rule": {
             "pool": "the 100 held-out evaluation images whose output parsed at every budget",
             "metric": f"absolute change in numeric rating at epsilon = {GALLERY_RANK_EPSILON}; a "
                       "non-zero change always crosses a rating category on this five-label scale",
             "tie_break": "image id, ascending",
             "selection": f"the {GALLERY_N_DOMINANT} highest-ranked shifts in the dominant "
                          "direction (sign of the mean shift over all valid images) and the "
                          f"{GALLERY_N_OPPOSITE} highest-ranked shift in the opposite direction",
             "text_criterion": "not applied: perturbed evaluation text was not saved by the "
                               "released run for all images",
             "same_images_at_every_epsilon": True},
         "image_encoding": {"format": "WebP, lossless", "size": [GALLERY_MAX_SIDE, GALLERY_MAX_SIDE],
                            "content": "the model-input tensor (image processor output, plus the "
                                       "perturbation, clamped to [-1, 1]) mapped to 8-bit RGB"},
         "targets": targets},
        base_meta(
            "IV", "UAP example gallery: selected held-out images across perturbation budgets",
            sorted(set(sources)),
            model_config="original interestingness task, blank prompt; each image is shown as "
                         "the model received it, clean and with the UAP of each budget added",
            feature_labels={t["target"]: f"{t['target_label']} direction at {t['layer']}" for t in targets},
            units={"score": "interestingness rating, 1 (Not Interesting) to 5 (Extremely Interesting)",
                   "delta_from_clean": "perturbed minus clean rating",
                   "epsilon": EPS_UNITS},
            aggregation="per-image model outputs as saved; ratings from "
                        f"results/{GALLERY_EVAL_DIR}, clean label and text from the study labels; "
                        "perturbed text from a later run of the same runner, used only where that "
                        "run reproduced the released rating",
            caveats=[
                "The examples are chosen for the size of their shift. They are not "
                "representative: see all_images_shift_at_ranking_epsilon for the whole sample.",
                "For a target where selection_includes_every_larger_shift is false, some "
                "unselected images shift more than the selected opposite-direction example.",
                EPS_CAVEAT,
                "Perturbed evaluation text marked 'pending' has not been generated yet.",
                "The clean rating and text come from the study's label collection, not from the "
                "UAP evaluation run.",
                "The images are 8-bit renderings of the model input; subtracting the clean image "
                "recovers the perturbation wherever it is not clipped.",
                "Source photographs are from Open Images V7 (listed there as CC BY 2.0); "
                "per-image attribution is not yet included.",
            ]),
        inline_arrays=True)


# Transfer-task gallery: what may be shown differs per dataset.
TRANSFER_IMAGE_POLICY = {
    "shopping_relevance": {
        "status": "published",
        "source": "Marqo-GS-10M (Marqo/marqo-GS-10M on the Hugging Face Hub)",
        "source_url": "https://huggingface.co/datasets/Marqo/marqo-GS-10M",
        "licence": "Apache-2.0", "licence_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "attribution": "Product image from Marqo-GS-10M (Zhu, Jung and Clark, 2024, "
                       "arXiv:2404.08535), Apache-2.0. Perturbed versions are derived from it.",
        "content_warning": None,
    },
    "damage_severity": {
        "status": "unavailable",
        "source": "MEDIC (QCRI/MEDIC on the Hugging Face Hub)",
        "source_url": "https://huggingface.co/datasets/QCRI/MEDIC",
        "licence": "CC BY-NC-SA 4.0", "licence_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "attribution": "Alam et al., MEDIC: A Multi-Task Learning Dataset for Disaster Image "
                       "Classification, Neural Computing and Applications 35(3), 2023.",
        "reason": "Images are not shown. The publisher's terms of use "
                  "(https://crisisnlp.qcri.org/terms-of-use.html) restrict use to research on "
                  "humanitarian computing and require the contents to be kept confidential; "
                  "only identifiers may be shared.",
        "content_warning": "The source images show disaster damage and may show injured people.",
    },
    "moral_evaluation": {
        "status": "unavailable",
        "source": "Socio-Moral Image Database (AIML-TUDA/smid on the Hugging Face Hub)",
        "source_url": "https://huggingface.co/datasets/AIML-TUDA/smid",
        "licence": "access-controlled; see the dataset card", "licence_url": None,
        "attribution": "Crone et al., The Socio-Moral Image Database (SMID), PLoS ONE 13(1), 2018.",
        "reason": "Images are not shown: the source is access-controlled and is not redistributed.",
        "content_warning": "The source images include morally charged and distressing scenes.",
    },
}
STRONG_LANGUAGE_RE = re.compile(
    r"\b(f+u+c+k\w*|sh[i1]t\w*|bitch\w*|cunt\w*|asshole\w*|motherf\w*|whore\w*|slut\w*|nigg\w*|fagg?ot\w*)\b",
    re.IGNORECASE)


def _masked(text: str) -> tuple[str, bool]:
    """Text with strong language reduced to its first letter, and whether any was found."""
    out = STRONG_LANGUAGE_RE.sub(lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), text)
    return out, out != text


def export_transfer_gallery(ex: Exporter) -> None:
    try:
        from transformers import AutoImageProcessor
        from attack.IV2_generalisation.tasks import TASKS
        from utils.image_utils import preprocess_image
        from utils.paths import LOCAL_MODEL_REPO
    except ImportError as e:
        ex.skipped.append({"export": "transfer_gallery", "missing_source": f"python package: {e.name}"})
        return
    processor, deltas, sources = None, {}, []
    eps_values = [float(e) for e in GALLERY_EPSILONS]
    item_fields = {"shopping_relevance": ["query", "title", "product_id"],
                   "damage_severity": ["image_id"], "moral_evaluation": ["image_id"]}
    image_folder = {"shopping_relevance": "shopping"}
    tasks_out = []
    for task, info in GENERALISATION_TASKS.items():
        rel = f"generalisation/{task}/predictions.csv"
        try:
            pred = pd.read_csv(ex.src(rel))
            manifest = pd.read_csv(ex.src(f"generalisation/{task}/sample_manifest.csv")).set_index("sample_id")
        except FileNotFoundError as e:
            ex.skipped.append({"export": f"transfer_gallery/{task}", "missing_source": f"results/{e}"})
            continue
        sources += [f"results/{rel}", f"results/generalisation/{task}/sample_manifest.csv"]
        spec, policy = TASKS[task], TRANSFER_IMAGE_POLICY[task]
        template = spec.prompt_builder({"query": "{query}"})
        meanings = {int(k): v.strip() for k, v in re.findall(r"^(\d) = (.+)$", template, flags=re.M)}
        clean = pred[pred["attack"] == "clean"].set_index("sample_id")

        targets_out = []
        for target in UAP_TARGET_DIRECTIONS:
            rows = pred[pred["attack"] == target]
            wide = rows.pivot(index="sample_id", columns="epsilon", values="delta").dropna()
            ok = rows.groupby("sample_id")["parse_ok"].all()
            wide = wide[ok.reindex(wide.index).to_numpy() & clean["parse_ok"].reindex(wide.index).to_numpy()]
            rank = pd.DataFrame({"sample_id": wide.index, "delta": wide[float(GALLERY_RANK_EPSILON)].to_numpy()})
            rank["abs_delta"] = rank["delta"].abs()
            rank = rank.sort_values(["abs_delta", "sample_id"], ascending=[False, True]).reset_index(drop=True)
            rank["rank"] = rank.index + 1
            chosen, dominant = _select_gallery(rank)

            cards = []
            for row in chosen.itertuples():
                sid = row.sample_id
                m = manifest.loc[sid]
                pv, base = None, f"transfer_gallery/{task}/{target}/{sid}"
                if policy["status"] == "published":
                    if processor is None:
                        processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_REPO)
                    img_rel = f"generalisation/{image_folder[task]}/images/{sid}.jpg"
                    image = preprocess_image(ex.src(img_rel, root="data"), max_side=GALLERY_MAX_SIDE)
                    sources.append(f"data/{img_rel}")
                    pv = processor(images=image, return_tensors="pt")["pixel_values"][0].float().numpy()
                    meta_img = dict(width=int(pv.shape[2]), height=int(pv.shape[1]), task=task,
                                    target=target, sample_id=sid, licence=policy["licence"])
                    ex.write_asset(f"{base}/clean.webp", _webp_lossless(pv), epsilon=None, **meta_img)

                def output(r, image_name, eps=None):
                    text, strong = _masked(str(r["explanation"]))
                    out = {"image": f"assets/{base}/{image_name}" if pv is not None else None,
                           "label": int(r["model_label"]), "label_meaning": meanings[int(r["model_label"])],
                           "parse_ok": bool(r["parse_ok"]), "text": text,
                           "contains_strong_language": strong}
                    if strong:
                        out["text_unmasked"] = str(r["explanation"])
                    if eps is not None:
                        out = {"epsilon": eps, "epsilon_fraction_of_pixel_range": eps / PIXEL_RANGE_WIDTH,
                               **out, "delta_from_clean": int(r["delta"])}
                    return out

                steps = []
                for eps_s, eps in zip(GALLERY_EPSILONS, eps_values):
                    if pv is not None:
                        if (target, eps_s) not in deltas:
                            d_rel = f"universal_perturbation_projected/{target}/eps{eps_s}/delta.npy"
                            deltas[(target, eps_s)] = np.load(ex.src(d_rel)).astype(np.float32)[0]
                            sources.append(f"results/{d_rel}")
                        ex.write_asset(f"{base}/eps_{eps_s}.webp",
                                       _webp_lossless(np.clip(pv + deltas[(target, eps_s)], -1.0, 1.0)),
                                       epsilon=eps, **meta_img)
                    r = rows[(rows["sample_id"] == sid) & np.isclose(rows["epsilon"], eps)].iloc[0]
                    steps.append(output(r, f"eps_{eps_s}.webp", eps))
                cards.append({
                    "sample_id": sid, "rank": int(row.rank),
                    "selected_as": "increase" if row.delta > 0 else "decrease" if row.delta < 0 else "unchanged",
                    "item": {k: (m[k].item() if hasattr(m[k], "item") else m[k]) for k in item_fields[task]},
                    "prompt": spec.prompt_builder(m),
                    "clean": output(clean.loc[sid], "clean.webp"),
                    "steps": steps,
                })
            stats = lambda d: {"n": int(len(d)), "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
                               "mean_abs_delta": float(np.abs(d).mean())}
            targets_out.append({
                "target": target, "target_label": UAP_TARGETS[target][0], "layer": UAP_TARGETS[target][1],
                "n_valid_samples": int(len(rank)), "n_selected": len(cards),
                "dominant_direction": dominant,
                "selected_shift_at_ranking_epsilon": stats(chosen["delta"].to_numpy()),
                "all_samples_shift_at_ranking_epsilon": stats(rank["delta"].to_numpy()),
                "selected_ids": [c["sample_id"] for c in cards],
                "cards": cards,
            })
        tasks_out.append({
            "task": task, "task_label": info["label"], "dataset": info["dataset"],
            "judgement": info["judgement"], "scale_min": info["scale_min"], "scale_max": info["scale_max"],
            "score_meanings": {str(k): v for k, v in sorted(meanings.items())},
            "prompt_template": template,
            "image_policy": policy,
            "targets": targets_out,
        })
    if not tasks_out:
        raise FileNotFoundError("generalisation/*/predictions.csv")
    ex.write_json(
        "attack/transfer_gallery.json",
        {"epsilons": eps_values, "ranking_epsilon": float(GALLERY_RANK_EPSILON),
         "panel_label": "Largest observed label shifts in the dominant direction and the largest opposite shift at "
                        f"ε = {GALLERY_RANK_EPSILON}; not representative examples",
         "selection_rule": {
             "pool": "all samples of the frozen task sample whose output parsed in every condition",
             "metric": f"absolute change in the label at epsilon = {GALLERY_RANK_EPSILON}",
             "tie_break": "sample id, ascending",
             "selection": f"per task and UAP, the {GALLERY_N_DOMINANT} highest-ranked shifts in "
                          "the dominant direction (sign of the mean shift over all valid samples) "
                          f"and the {GALLERY_N_OPPOSITE} highest-ranked shift in the opposite "
                          "direction; if one side has too few, the next-ranked samples fill the set",
             "same_samples_at_every_epsilon": True},
         "strong_language": {"handling": "'text' has strong language reduced to its first letter; "
                                         "'text_unmasked' is present only where "
                                         "contains_strong_language is true and is meant for a "
                                         "reveal control",
                             "pattern": STRONG_LANGUAGE_RE.pattern},
         "image_encoding": {"format": "WebP, lossless", "size": [GALLERY_MAX_SIDE, GALLERY_MAX_SIDE],
                            "content": "the model-input tensor (image processor output, plus the "
                                       "perturbation, clamped to [-1, 1]) mapped to 8-bit RGB"},
         "tasks": tasks_out},
        base_meta(
            "IV", "UAP transfer gallery: selected samples of three unseen tasks across budgets",
            sorted(set(sources)),
            model_config="UAPs trained on the main image set (target layer language_29) applied "
                         "unchanged; clean and perturbed outputs of the same sample",
            feature_labels={t: f"{i['label']}: {i['judgement']} ({i['scale_min']}-{i['scale_max']}), "
                               f"{i['dataset']}" for t, i in GENERALISATION_TASKS.items()},
            units={"label": "model label on the task scale; see score_meanings",
                   "delta_from_clean": "perturbed minus clean label", "epsilon": EPS_UNITS},
            aggregation="per-sample model outputs as saved in results/generalisation/<task>/predictions.csv",
            caveats=[
                "The examples are chosen for the size of their shift and are not representative: "
                "all_samples_shift_at_ranking_epsilon gives the whole sample.",
                EPS_CAVEAT,
                "Images are published for the shopping task only. For damage severity and moral "
                "evaluation the source terms do not allow redistribution; see image_policy.",
                "Published perturbed images are derived from the source image and carry its "
                "licence and attribution. Subtracting the clean image recovers the perturbation "
                "wherever it is not clipped.",
                "At large budgets the model often describes content that is not in the image, "
                "including offensive text; such explanations are model output, not a description "
                "of the source image.",
                "No noise baseline: part of any shift may be generic image corruption.",
            ]),
        inline_arrays=True)


EXPORTS = [
    export_discovery,
    export_condition_similarity, export_nn_grouping, export_interest_alignment,
    export_vector_field_3d, export_persona_composition, export_uap_target_cosines,
    export_dose_response, export_single_dose, export_logit_lens,
    export_uap_alignment, export_uap_behaviour, export_generalisation, export_arousal,
    export_transfer_statistics, export_arousal_valence, export_uap_gallery,
    export_transfer_gallery,
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
        "assets": sorted(ex.assets, key=lambda a: a["file"]),
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
