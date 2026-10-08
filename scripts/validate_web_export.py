"""
Validate the website export, in particular the two interactive geometry files
(geometry/vector_field_3d.json and geometry/persona_composition_paths.json).

Usage (from the repository root):

    python scripts/validate_web_export.py                 # all checks
    python scripts/validate_web_export.py --export_only   # checks that need web_export/ only

Four groups of checks:

  structure    the exported files alone: ids, labels, finite coordinates, valid
               cosine matrix, one shared basis per figure, every ordered feature
               subset, same end point for every order of a feature set
  sources      recomputes both figures from the saved results with code that is
               independent of the exporter, and compares with the values saved by
               the analysis notebooks (needs results/)
  determinism  builds the export twice and compares the outputs byte for byte,
               and with the files in web_export/ (needs results/)
  gallery      every card has its five images and outputs, deterministic selection,
               ratings equal the saved evaluation outputs
  hygiene      no raw data, absolute paths, user names or tokens in web_export/

Requires numpy and pandas only. Exits with status 1 if any check fails.
"""

from __future__ import annotations
import argparse
import getpass
import itertools
import json
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LAYER = "language_29_D5120"
TOL = 2e-6                      # exports are rounded to 6 decimals

EMOTIONS = ["anger", "amusement", "awe", "contentment", "disgust", "excitement", "fear", "sad"]
GENDERS = ["female", "male"]
COUNTRIES = ["Germany", "Nigeria"]
FEATURES = ["gender", "emotion", "country"]
ORDERS = [">".join(o) for r in (1, 2, 3) for o in itertools.permutations(FEATURES, r)]

EXPECTED_DIRECTIONS = {
    "workload_overwhelming_vs_minimal": ("Workload / stress: overwhelming − minimal", "workload"),
    "emotion_excited_vs_angry": ("Excitement − anger", "emotion_attack"),
    "gender_female_vs_male": ("Gender: female − male (averaged over emotions)", "gender"),
    "country_germany": ("Country: Germany", "country"),
    "country_nigeria": ("Country: Nigeria", "country"),
    "interest_blank_high_vs_low": ("Blank-prompt interestingness: high − low", "interestingness"),
    "interest_global_high_vs_low": ("Global interestingness: high − low", "interestingness"),
    "emotion_anger": ("Emotion: anger", "emotion"),
    "emotion_amusement": ("Emotion: amusement", "emotion"),
    "emotion_awe": ("Emotion: awe", "emotion"),
    "emotion_contentment": ("Emotion: contentment", "emotion"),
    "emotion_disgust": ("Emotion: disgust", "emotion"),
    "emotion_excitement": ("Emotion: excitement", "emotion"),
    "emotion_fear": ("Emotion: fear", "emotion"),
    "emotion_sad": ("Emotion: sadness", "emotion"),
}

NEW_EXPORTS = ["data/geometry/vector_field_3d.json", "data/geometry/persona_composition_paths.json"]
ALLOWED_SUFFIXES = {".csv", ".json", ".md"}
MAX_NUMERIC_ARRAY = 64          # longest legitimate numeric array is one cosine-matrix row
SECRET_PATTERNS = {
    "absolute path": r"(?<![\w.])/(?:home|anvme|scratch|lustre|gpfs|nfs|mnt|Users|tmp)/",
    "Hugging Face token": r"hf_[A-Za-z0-9]{30,}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}",
    "API key": r"sk-[A-Za-z0-9_-]{20,}",
    "bearer token": r"(?i)bearer\s+[a-z0-9._-]{20,}",
    "private key": r"BEGIN [A-Z ]*PRIVATE KEY",
    "e-mail address": r"[\w.+-]+@[\w-]+\.[a-z]{2,}",
}


class Report:
    def __init__(self):
        self.failed = 0
        self.passed = 0

    def check(self, ok: bool, what: str, detail: str = "") -> bool:
        ok = bool(ok)
        self.passed += ok
        self.failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"  [{detail}]" if detail else ""))
        return ok

    def skip(self, what: str, why: str) -> None:
        print(f"  SKIP  {what}  [{why}]")


def load(out_dir: Path, name: str) -> dict:
    return json.loads((out_dir / name).read_text())


def finite(x) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x, dtype=np.float64))))


# ── structure: exported files only ───────────────────────────────────────────

def check_vector_field(rep: Report, vf: dict) -> None:
    dirs = vf["directions"]
    ids = [d["id"] for d in dirs]
    rep.check(ids == list(EXPECTED_DIRECTIONS), "vector field: expected direction ids, in order",
              f"{len(ids)} directions")
    rep.check(all((d["label"], d["family"]) == EXPECTED_DIRECTIONS.get(d["id"]) for d in dirs),
              "vector field: expected labels and families")
    rep.check(vf["layer"] == LAYER and all(d["layer"] == LAYER for d in dirs),
              f"vector field: every direction at {LAYER}")
    rep.check(all(d["source"] and d["condition_definition"] for d in dirs),
              "vector field: source and condition definition for every direction")
    E = np.array([d["display_endpoint"] for d in dirs])
    rep.check(E.shape == (len(dirs), 3) and finite(E), "vector field: finite 3D endpoints")
    rep.check(vf["origin"]["display"] == [0.0, 0.0, 0.0], "vector field: neutral origin at (0, 0, 0)")
    rep.check(all(abs(d["full_space_unit_norm"] - 1) < TOL for d in dirs)
              and all(0 < d["raw_norm"] <= 1 + TOL for d in dirs),
              "vector field: unit full-space norm, raw norm kept as metadata")

    cm = vf["cosine_matrix"]
    C = np.array(cm["values"])
    rep.check(cm["ids"] == ids and C.shape == (len(ids), len(ids)), "cosine matrix: one row and column per direction")
    rep.check(finite(C) and np.abs(C).max() <= 1 + TOL, "cosine matrix: finite, within [-1, 1]")
    rep.check(np.abs(C - C.T).max() <= TOL and np.abs(np.diag(C) - 1).max() <= TOL,
              "cosine matrix: symmetric with unit diagonal")
    w, V = np.linalg.eigh(C)
    rep.check(w.min() > -1e-5, "cosine matrix: positive semi-definite", f"min eigenvalue {w.min():.2e}")

    # One shared uncentered top-3 basis: the endpoints' Gram matrix must then be
    # the best rank-3 approximation of the full-space cosine matrix.
    top = np.argsort(w)[::-1][:3]
    rank3 = (V[:, top] * w[top]) @ V[:, top].T
    gap = np.abs(E @ E.T - rank3).max()
    rep.check(gap < 2e-5, "vector field: endpoints are one shared top-3 uncentered SVD projection", f"max gap {gap:.1e}")
    pr = vf["projection"]
    rep.check(np.abs(np.array(pr["explained_variance_ratio"]) - w[top] / w.sum()).max() < 2e-5,
              "vector field: explained variance matches the cosine matrix",
              f"total {pr['explained_variance_ratio_total']:.4f}")
    rep.check(pr["dimensions"] == 3 and pr["random_state"] is None
              and all(re.fullmatch(r"[0-9a-f]{64}", pr[k]) for k in ("source_hash", "projection_hash")),
              "vector field: projection method, solver and hashes recorded")
    rep.check(np.abs(np.linalg.norm(E, axis=1) - [d["display_norm"] for d in dirs]).max() <= TOL
              and np.linalg.norm(E, axis=1).max() <= 1 + TOL,
              "vector field: display norms at most 1 (projections of unit directions)")
    caveats = " ".join(vf["meta"]["caveats"]).lower()
    rep.check("display projection" in caveats and "intervention" in caveats,
              "vector field: projection and no-intervention caveats present")
    rep.check(vf["meta"]["status"] == "preliminary", "vector field: marked preliminary")


def check_composition(rep: Report, pc: dict) -> None:
    conds = pc["conditions"]
    expected = [f"{g}_{e}_extended_{c.lower()}" for c in COUNTRIES for g in GENDERS for e in EMOTIONS]
    rep.check([c["id"] for c in conds] == expected == expected[:pc["n_conditions_expected"]],
              "composition: all compound persona conditions present", f"{len(conds)} of {len(expected)}")
    rep.check(pc["layer"] == LAYER, f"composition: layer {LAYER}")
    rep.check(all(c["id"].startswith(c["labels"]["gender"] + "_")
                  and c["id"].endswith(c["labels"]["country"].lower()) for c in conds),
              "composition: labels agree with condition ids")
    rep.check(all(list(c["predictions_by_order"]) == ORDERS for c in conds) and pc["feature_orders"] == ORDERS,
              "composition: every ordered non-empty feature subset", f"{len(ORDERS)} orders per condition")

    gm = np.array(pc["feature_vectors"]["grand_mean"]["display"])
    ok_finite = ok_start = ok_prefix = ok_affine = ok_same = ok_final = ok_coef = True
    worst_affine = 0.0
    for c in conds:
        ok_start &= c["grand_mean_display"] == gm.tolist()
        comp = c["components"]
        ok_coef &= (comp["gender"]["coefficient"] == 0.5 and comp["emotion"]["coefficient"] == 1.0
                    and comp["country"]["coefficient"] == 1.0
                    and comp["gender"]["sign"] == (1 if c["labels"]["gender"] == "female" else -1))
        ok_finite &= finite(c["observed_display"]) and all(finite(comp[f]["display_delta"]) for f in FEATURES)
        by_set = {}
        for key, entry in c["predictions_by_order"].items():
            order = key.split(">")
            stages = entry["stages"]
            ok_prefix &= [s["included"] for s in stages] == [order[:i + 1] for i in range(len(order))]
            for s in stages:
                ok_finite &= finite(s["display"]) and finite(
                    [s["full_space_cosine_to_observed"], s["full_space_distance_to_observed"]])
                # stage = start + displacements of the included features, in one basis
                want = gm + sum(np.array(comp[f]["display_delta"]) for f in s["included"])
                worst_affine = max(worst_affine, np.abs(want - s["display"]).max())
                by_set.setdefault(frozenset(s["included"]), []).append(
                    (s["display"], s["full_space_cosine_to_observed"], s["full_space_distance_to_observed"]))
            last = stages[-1]
            ok_final &= (entry["final_full_space_cosine_to_observed"] == last["full_space_cosine_to_observed"]
                         and entry["final_full_space_distance_to_observed"] == last["full_space_distance_to_observed"]
                         and finite(entry["total_full_space_journey_from_start"]))
        ok_same &= all(all(v == vals[0] for v in vals) for vals in by_set.values())
        ok_same &= len({e["total_full_space_journey_from_start"]
                        for k, e in c["predictions_by_order"].items() if k.count(">") == 2}) == 1
        full = c["predictions_by_order"][">".join(FEATURES)]
        ok_final &= (full["stages"][-1]["display"] == c["predicted_display"]
                     and full["final_full_space_cosine_to_observed"] == c["final_full_space_cosine_to_observed"])
    ok_affine = worst_affine <= 4 * TOL
    rep.check(ok_finite, "composition: finite display coordinates and metrics")
    rep.check(ok_start, "composition: every path starts at the grand mean")
    rep.check(ok_coef, "composition: signed half-gender, unit emotion and country coefficients")
    rep.check(ok_prefix, "composition: stages follow the selected feature order")
    rep.check(ok_same, "composition: same point for every order of the same feature set")
    rep.check(ok_final, "composition: final values agree with the last stage and the full prediction")
    rep.check(ok_affine, "composition: one shared basis (stage = start + sum of display deltas)",
              f"max gap {worst_affine:.1e}")

    agg = pc["aggregate"]
    cos = np.mean([c["final_full_space_cosine_to_observed"] for c in conds])
    dist = np.mean([c["final_full_space_distance_to_observed"] for c in conds])
    full = agg["full_composition"]
    rep.check(abs(cos - full["mean_full_space_cosine_to_observed"]) <= TOL
              and abs(dist - full["mean_full_space_distance_to_observed"]) <= TOL,
              "composition: aggregate is the mean of the per-condition values")
    r = agg["reproduces"]
    rep.check(abs(cos - r["mean_cosine_reported"]) <= 5.1e-5 and abs(r["mean_cosine_reported"] - 0.9684) < 1e-9,
              "composition: reproduces the existing full-composition mean cosine",
              f"{cos:.6f} vs reported {r['mean_cosine_reported']}")
    rep.check(abs(dist - r["mean_residual_norm_reported"]) <= 5.1e-4,
              "composition: reproduces the existing mean distance to observed",
              f"{dist:.6f} vs reported {r['mean_residual_norm_reported']}")
    pr, cv = pc["projection"], pc["composition_convention"]
    rep.check(pr["dimensions"] == 3 and len(pr["fit_conditions"]) == 48 and pr["random_state"] is None
              and not any("pred" in f for f in pr["fit_conditions"])
              and all(re.fullmatch(r"[0-9a-f]{64}", pr[k]) for k in ("source_hash", "projection_hash")),
              "composition: basis fitted on the 48 observed condition means, hashes recorded")
    rep.check(cv["calculation_space"] == "raw activation space" and cv["start"] == "grand_mean",
              "composition: convention recorded (raw activation space, grand-mean start)")
    caveats = " ".join(pc["meta"]["caveats"]).lower()
    rep.check("not a causal-control result" in caveats and pc["meta"]["status"] == "preliminary",
              "composition: marked preliminary, with the no-causal-claim caveat")


# ── sources: independent recomputation from results/ ─────────────────────────

def _unit(v):
    return v / np.linalg.norm(v)


def _oriented_top3(rows):
    _, _, Vt = np.linalg.svd(rows, full_matrices=False)
    scores = rows @ Vt[:3].T
    return Vt[:3] * np.sign(scores[np.abs(scores).argmax(axis=0), range(3)])[:, None]


def _notebook_text(path: Path) -> str:
    cells = json.loads(path.read_text())["cells"]
    return "".join("".join(o.get("text", [])) for c in cells for o in c.get("outputs", []))


def check_vector_field_sources(rep: Report, vf: dict, results: Path) -> None:
    disc = results / "representation_discovery"
    one = lambda rel: np.load(disc / rel, allow_pickle=True).item()[LAYER].astype(np.float64)
    avg = lambda folder, pat: np.mean(np.stack(
        [np.load(f, allow_pickle=True).item()[LAYER] for f in sorted((disc / folder).glob(pat))]), axis=0)
    vecs = {
        "workload_overwhelming_vs_minimal": one("mental_workload/md_vectors/workload_overwhelming_vs_workload_minimal.npy"),
        "emotion_excited_vs_angry": one("base_emotion/md_vectors/avg_excitement_vs_avg_anger.npy"),
        "gender_female_vs_male": avg("base/md_vectors", "female_*_vs_male_*.npy"),
        "country_germany": avg("extended_germany_country/md_vectors", "*.npy"),
        "country_nigeria": avg("extended_nigeria_country/md_vectors", "*.npy"),
        "interest_blank_high_vs_low": one("interestingness/md_vectors/blank_interest_high_vs_blank_interest_low.npy"),
        "interest_global_high_vs_low": one("interestingness/md_vectors/global_interest_high_vs_global_interest_low.npy"),
    } | {f"emotion_{e}": one(f"base_emotion/md_vectors/avg_{e}_vs_avg_rest_{e}.npy") for e in EMOTIONS}
    ids = vf["cosine_matrix"]["ids"]
    U = np.stack([_unit(vecs[i].astype(np.float64)) for i in ids])
    C = np.array(vf["cosine_matrix"]["values"])
    rep.check(np.abs(U @ U.T - C).max() <= TOL, "vector field: cosine matrix recomputed from the saved directions",
              f"max gap {np.abs(U @ U.T - C).max():.1e}")
    E = np.array([d["display_endpoint"] for d in vf["directions"]])
    gap = np.abs(U @ _oriented_top3(U).T - E).max()
    rep.check(gap <= TOL, "vector field: endpoints recomputed with a single basis", f"max gap {gap:.1e}")

    nb = results / "EX1_T1_workload_analytics.ipynb"
    if nb.is_file():
        text = _notebook_text(nb)
        pairs = {"excitement vs interest (global)": ("emotion_excitement", "interest_global_high_vs_low"),
                 "anger vs interest (global)": ("emotion_anger", "interest_global_high_vs_low"),
                 "emotion attack (exc−ang) vs interest (global)": ("emotion_excited_vs_angry", "interest_global_high_vs_low")}
        got = {}
        for label, (a, b) in pairs.items():
            m = re.search(re.escape(label) + r":\s+([+-][\d.]+)", text)
            got[label] = (C[ids.index(a), ids.index(b)], float(m.group(1)) if m else np.nan)
        rep.check(all(abs(c - r) <= 5.1e-5 for c, r in got.values()),
                  "vector field: reproduces the cosines printed by the analysis notebook",
                  ", ".join(f"{r:+.4f}" for _, r in got.values()))
        m = re.search(r"top-3 \(uncentered\) components: ([\d.]+)%", text)
        total = 100 * vf["projection"]["explained_variance_ratio_total"]
        rep.check(m and abs(total - float(m.group(1))) <= 0.051,
                  "vector field: reproduces the notebook's top-3 variance captured",
                  f"{total:.2f}% vs reported {m.group(1) if m else '?'}%")
    else:
        rep.skip("vector field: notebook reference values", "notebook not found")

    fig = results / "analytics" / "fig_workload_alignment_3d.html"
    if fig.is_file():
        num = r"(-?[\d.]+(?:e-?\d+)?)"
        old = np.array(re.findall(rf'"x":\[0,{num}\],"y":\[0,{num}\],"z":\[0,{num}\]', fig.read_text()), dtype=float)
        same = old.shape == E.shape and np.abs(old - E).max() <= 1e-5
        rep.check(same, "vector field: endpoints equal those of the existing 3D figure",
                  f"max gap {np.abs(old - E).max():.1e}" if old.shape == E.shape else f"shape {old.shape}")
    else:
        rep.skip("vector field: comparison with the existing 3D figure", "figure not found")


def check_composition_sources(rep: Report, pc: dict, results: Path) -> None:
    # Written out as in results/EX1_T2_additivity.ipynb, without the exporter's helpers.
    def means(slug):
        return {k: v.astype(np.float64) for k, v in np.load(
            results / "analytics" / slug / "language_29" / "mean_vectors_language_29.npy",
            allow_pickle=True).item().items()}
    base = means("base")
    ctry = {"Germany": means("extended_germany"), "Nigeria": means("extended_nigeria")}
    grand = np.mean(np.stack(list(base.values())), axis=0)
    v_gender = (np.mean(np.stack([v for k, v in base.items() if k.startswith("female_")]), axis=0)
                - np.mean(np.stack([v for k, v in base.items() if k.startswith("male_")]), axis=0))
    v_country = {c: np.mean(np.stack([ctry[c][f"{g}_{e}_extended_{c.lower()}"] - base[f"{g}_{e}"]
                                      for g in GENDERS for e in EMOTIONS]), axis=0) for c in COUNTRIES}
    v_emotion = {e: np.mean(np.stack([base[f"female_{e}"], base[f"male_{e}"]]), axis=0) - grand for e in EMOTIONS}

    real = np.stack(list(base.values()) + list(ctry["Germany"].values()) + list(ctry["Nigeria"].values()))
    centre = real.mean(axis=0)
    basis = _oriented_top3(real - centre)
    proj = lambda v: (v - centre) @ basis.T

    worst = {"display": 0.0, "cosine": 0.0, "distance": 0.0, "journey": 0.0, "order": 0.0, "norm": 0.0}
    for c in pc["conditions"]:
        g, country = c["labels"]["gender"], c["labels"]["country"]
        e = c["id"].split("_")[1]
        observed = ctry[country][c["id"]]
        step = {"gender": (1 if g == "female" else -1) * v_gender / 2.0,
                "emotion": v_emotion[e], "country": v_country[country]}
        worst["display"] = max(worst["display"], np.abs(proj(observed) - c["observed_display"]).max(),
                               np.abs(proj(grand) - c["grand_mean_display"]).max())
        for f in FEATURES:
            worst["norm"] = max(worst["norm"], abs(np.linalg.norm(step[f]) - c["components"][f]["full_space_norm"]))
        ends = {}
        for key, entry in c["predictions_by_order"].items():
            p = grand
            for f, stage in zip(key.split(">"), entry["stages"]):
                p = p + step[f]                              # added in the visitor's order
                a, b = p - grand, observed - grand
                worst["display"] = max(worst["display"], np.abs(proj(p) - stage["display"]).max())
                worst["cosine"] = max(worst["cosine"], abs(
                    a @ b / np.linalg.norm(a) / np.linalg.norm(b) - stage["full_space_cosine_to_observed"]))
                worst["distance"] = max(worst["distance"], abs(
                    np.linalg.norm(p - observed) - stage["full_space_distance_to_observed"]))
            worst["journey"] = max(worst["journey"], abs(
                np.linalg.norm(p - grand) - entry["total_full_space_journey_from_start"]))
            ends.setdefault(frozenset(key.split(">")), []).append(p)
        for pts in ends.values():
            worst["order"] = max(worst["order"], max(np.abs(q - pts[0]).max() for q in pts))
    rep.check(worst["display"] <= TOL, "composition: display coordinates recomputed from the saved condition means",
              f"max gap {worst['display']:.1e}")
    rep.check(max(worst["cosine"], worst["distance"], worst["journey"], worst["norm"]) <= TOL,
              "composition: per-condition cosines, distances, journeys and norms recomputed",
              f"max gap {max(worst['cosine'], worst['distance'], worst['journey'], worst['norm']):.1e}")
    rep.check(worst["order"] <= 1e-9, "composition: full-space end point independent of feature order",
              f"max gap {worst['order']:.1e}")

    nb = results / "EX1_T2_additivity.ipynb"
    if not nb.is_file():
        rep.skip("composition: notebook reference values", "notebook not found")
        return
    text = _notebook_text(nb)
    m = re.search(r"Using ±v_gender/2\s+\(Stage-2-validated scaling\):\s+mean cos = ([\d.]+)\s+"
                  r"mean residual norm = ([\d.]+)", text)
    full = pc["aggregate"]["full_composition"]
    rep.check(m and abs(full["mean_full_space_cosine_to_observed"] - float(m.group(1))) <= 5.1e-5
              and abs(full["mean_full_space_distance_to_observed"] - float(m.group(2))) <= 5.1e-4,
              "composition: reproduces the notebook's Stage 3 result",
              f"cos {full['mean_full_space_cosine_to_observed']:.4f} / residual "
              f"{full['mean_full_space_distance_to_observed']:.3f} vs reported "
              f"{m.group(1) if m else '?'} / {m.group(2) if m else '?'}")
    norms = dict(re.findall(r"\|\|v_emotion\[(\w+)\s*\]\|\|\s+=\s+([\d.]+)", text))
    norms |= {c: n for c, n in re.findall(r"\|\|v_country\[(\w+)\s*\]\|\|\s+=\s+([\d.]+)", text)}
    fv = pc["feature_vectors"]
    got = {**{e: fv["emotion"][e]["full_space_norm"] for e in EMOTIONS},
           **{c: fv["country"][c]["full_space_norm"] for c in COUNTRIES}}
    rep.check(len(norms) == 10 and all(abs(got[k] - float(v)) <= 5.1e-3 for k, v in norms.items()),
              "composition: reproduces the notebook's feature-vector norms", f"{len(norms)} vectors")
    by_c = {c: re.search(rf"{c}: mean ‖predicted − actual‖ = ([\d.]+)", text) for c in COUNTRIES}
    rep.check(all(by_c[c] and abs(full["by_country"][c]["mean_full_space_distance_to_observed"]
                                  - float(by_c[c].group(1))) <= 5.1e-3 for c in COUNTRIES),
              "composition: reproduces the notebook's per-country distance to observed",
              ", ".join(f"{c} {by_c[c].group(1)}" for c in COUNTRIES if by_c[c]))


# ── determinism ──────────────────────────────────────────────────────────────

def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def check_determinism(rep: Report, results: Path, out_dir: Path) -> None:
    builds = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("a", "b"):
            run = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "export_web_data.py"),
                 "--results_dir", str(results), "--out_dir", str(Path(tmp) / name)],
                capture_output=True, text=True)
            if not rep.check(run.returncode == 0, f"build {name}: exporter ran", run.stderr.strip()[-300:]):
                return
            builds.append(_files(Path(tmp) / name))
    a, b = builds
    built = lambda files: {k: v for k, v in files.items() if k.startswith(("data/", "assets/"))}
    data_a = built(a)
    rep.check(data_a == built(b),
              "two consecutive builds: data and asset files byte-identical", f"{len(data_a)} files")
    strip = lambda raw: {k: v for k, v in json.loads(raw).items() if k != "generated_utc"}
    rep.check(strip(a["release_manifest.json"]) == strip(b["release_manifest.json"]),
              "two consecutive builds: manifest identical apart from generated_utc")
    current = built(_files(out_dir))
    changed = sorted(k for k in data_a.keys() | current.keys() if data_a.get(k) != current.get(k))
    rep.check(not changed, f"files in {out_dir.name}/data and assets are byte-identical to a fresh build", ", ".join(changed[:5]))


# ── hygiene ──────────────────────────────────────────────────────────────────

def _long_numeric_arrays(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _long_numeric_arrays(v, f"{path}/{k}")
    elif isinstance(node, list):
        if len(node) > MAX_NUMERIC_ARRAY and all(isinstance(v, (int, float)) for v in node):
            yield f"{path} ({len(node)} numbers)"
        for i, v in enumerate(node):
            if isinstance(v, (dict, list)):
                yield from _long_numeric_arrays(v, f"{path}[{i}]")


def check_hygiene(rep: Report, out_dir: Path) -> None:
    files = _files(out_dir)
    odd = sorted(k for k in files if Path(k).suffix not in ALLOWED_SUFFIXES
                 and not (k.startswith(("assets/uap_gallery/", "assets/transfer_gallery/shopping_relevance/"))
                          and k.endswith(".webp")))
    rep.check(not odd, "hygiene: only .csv, .json and .md files, plus the gallery's .webp images "
                       "(no .npy or archives)", ", ".join(odd[:5]))

    private = {"user name": re.escape(getpass.getuser()), "home directory": re.escape(str(Path.home())),
               "host name": re.escape(socket.gethostname())}
    hits = []
    for name, raw in files.items():
        if name.endswith(".webp"):
            continue
        text = raw.decode("utf-8")
        for label, pattern in {**SECRET_PATTERNS, **private}.items():
            if re.search(pattern, text):
                hits.append(f"{label} in {name}")
    rep.check(not hits, "hygiene: no absolute paths, user or host names, e-mail addresses or tokens", "; ".join(hits[:5]))

    long = [f"{n}{p}" for n in files if n.endswith(".json") for p in _long_numeric_arrays(json.loads(files[n]))]
    rep.check(not long, f"hygiene: no numeric array longer than {MAX_NUMERIC_ARRAY} (no activation or direction vectors)",
              "; ".join(long[:3]))

    manifest = json.loads(files["release_manifest.json"])
    listed = {e["file"] for e in manifest["exports"]}
    on_disk = {k for k in files if k.startswith("data/") and not k.endswith(".meta.json")}
    rep.check(listed == on_disk, "manifest: lists exactly the data files on disk", f"{len(listed)} exports")
    rep.check(all(n in listed and n in manifest["preliminary_outputs"] for n in NEW_EXPORTS),
              "manifest: both geometry figures listed and marked preliminary")
    rep.check(all(s["path"].startswith(("results/", "data/")) and re.fullmatch(r"[0-9a-f]{64}", s["sha256"])
                  for s in manifest["source_files"]),
              "manifest: sources given as results/- or data/-relative paths with SHA-256", f"{len(manifest['source_files'])} sources")
    used = {s for e in manifest["exports"] if e["file"] in NEW_EXPORTS for s in e["sources"]}
    rep.check(used and used <= {s["path"] for s in manifest["source_files"]},
              "manifest: every source of the geometry figures is hashed", f"{len(used)} sources")
    lens = json.loads(files.get("data/causal/logit_lens_tokens.json", b"{}"))
    series = lens.get("series", [])
    rep.check(len(series) == 5 and all(
                  [t["rank"] for t in s[side]] == list(range(1, len(s[side]) + 1)) and len(s[side]) == 30
                  and all(t["display"].isprintable() and " " not in t["display"] for t in s[side])
                  for s in series for side in ("pushed_up", "pushed_down")),
              "logit lens: five series, 30 ranked display-safe tokens per side")
    rep.check("not next-token probabilities" in " ".join(lens.get("meta", {}).get("caveats", [])),
              "logit lens: stated as not being next-token probabilities")
    sizes = {n: len(files[n]) for n in NEW_EXPORTS if n in files}
    rep.check(sizes and max(sizes.values()) < 1_000_000, "hygiene: geometry figure files stay compact",
              ", ".join(f"{Path(n).name} {s / 1024:.0f} kB" for n, s in sizes.items()))


def check_gallery(rep: Report, out_dir: Path, results: Path | None) -> None:
    import hashlib
    g = load(out_dir, "data/attack/uap_gallery.json")
    manifest = load(out_dir, "release_manifest.json")
    assets = {a["file"]: a for a in manifest.get("assets", [])}
    eps = g["epsilons"]
    rep.check(eps == [0.1, 0.5, 1.0, 2.0] and g["ranking_epsilon"] == 1.0 and g["targets"]
              and g["meta"]["status"] == "preliminary" and "not representative" in g["panel_label"],
              "gallery: schema, budgets and honest panel label", f"{len(g['targets'])} targets")
    ok_cards = ok_files = ok_rank = ok_out = ok_sel = True
    n_images = 0
    for t in g["targets"]:
        rank = t["ranking"]
        ok_rank &= rank == sorted(rank, key=lambda r: (-abs(r["delta"]), r["sample_id"])) \
            and [r["rank"] for r in rank] == list(range(1, len(rank) + 1))
        up_first = sum(r["delta"] for r in rank) >= 0
        ups = [r["sample_id"] for r in rank if r["delta"] > 0][:4 if up_first else 1]
        downs = [r["sample_id"] for r in rank if r["delta"] < 0][:1 if up_first else 4]
        ok_sel &= t["dominant_direction"] == ("increase" if up_first else "decrease")
        ok_sel &= sorted(c["sample_id"] for c in t["cards"]) == sorted(ups + downs) \
            and [c["rank"] for c in t["cards"]] == sorted(c["rank"] for c in t["cards"])
        ok_cards &= len(t["cards"]) == t["n_selected"] == 5
        by_id = {r["sample_id"]: r for r in rank}
        for c in t["cards"]:
            ok_cards &= [s["epsilon"] for s in c["steps"]] == eps
            base = f"assets/uap_gallery/{t['target']}/{c['sample_id']}"
            want = [f"{base}/clean.webp"] + [f"{base}/eps_{e:.2f}.webp" for e in eps]
            ok_cards &= [c["clean"]["image"]] + [s["image"] for s in c["steps"]] == want
            for f in want:
                path = out_dir / f
                a = assets.get(f, {})
                ok_files &= path.is_file() and a.get("width") == a.get("height") == 336 \
                    and a.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest() \
                    and path.read_bytes()[:4] == b"RIFF" and path.read_bytes()[8:15] == b"WEBPVP8"
                n_images += 1
            ok_out &= bool(c["clean"]["label"]) and bool(c["clean"]["text"]) and 1 <= c["clean"]["score"] <= 5
            for s in c["steps"]:
                ok_out &= bool(s["label"]) and 1 <= s["score"] <= 5 and s["parse_ok"] is True \
                    and s["delta_from_clean"] == s["score"] - c["clean"]["score"] \
                    and s["epsilon_fraction_of_pixel_range"] == s["epsilon"] / 2 \
                    and ((s["text_status"] == "available" and bool(s["text"]))
                         or (s["text_status"] == "pending" and s["text"] is None))
            at = next(s for s in c["steps"] if s["epsilon"] == 1.0)
            ok_out &= at["delta_from_clean"] == by_id[c["sample_id"]]["delta"]
    rep.check(ok_cards, "gallery: five cards per target, each with clean plus all four budgets")
    rep.check(ok_files, "gallery: every image exists, is a 336 x 336 WebP and matches its manifest hash",
              f"{n_images} images")
    rep.check(set(assets) == {p.relative_to(out_dir).as_posix() for p in (out_dir / "assets").rglob("*") if p.is_file()},
              "gallery: manifest lists exactly the asset files on disk", f"{len(assets)} assets")
    rep.check(ok_rank and ok_sel, "gallery: deterministic ranking and selection (|change| at 1.00, then image id)")
    rep.check(ok_out, "gallery: rating, score and change for every card; text available or marked pending")
    pending = sum(t["text_pending"] for t in g["targets"])
    print(f"  NOTE  gallery: {pending} perturbed evaluation texts still pending")
    if results is None:
        rep.skip("gallery: comparison with the saved evaluation outputs", "results/ not used")
        return
    import pandas as pd
    same = True
    for t in g["targets"]:
        for e in eps:
            src = pd.read_csv(results.parent / t["evaluation_source"]
                              / f"{t['target']}_eps{e:.2f}_labels.csv").set_index("filename")
            for c in t["cards"]:
                r = src.loc[c["sample_id"] + ".jpg"]
                s = next(s for s in c["steps"] if s["epsilon"] == e)
                same &= (s["label"], s["score"], s["delta_from_clean"]) == (
                    r["perturbed_label"], r["perturbed_score"], r["score_delta"]) \
                    and c["clean"]["label"] == r["clean_label"]
    rep.check(same, "gallery: ratings equal the saved evaluation outputs")


def check_transfer_gallery(rep: Report, out_dir: Path, results: Path | None) -> None:
    g = load(out_dir, "data/attack/transfer_gallery.json")
    assets = {a["file"] for a in load(out_dir, "release_manifest.json").get("assets", [])}
    eps = g["epsilons"]
    strong = re.compile(g["strong_language"]["pattern"], re.IGNORECASE)
    rep.check([t["task"] for t in g["tasks"]] == ["shopping_relevance", "moral_evaluation", "damage_severity"]
              and all([x["target"] for x in t["targets"]] == ["interest", "excited_vs_angry", "workload"]
                      for t in g["tasks"]) and g["meta"]["status"] == "preliminary",
              "transfer gallery: three tasks, three UAPs each, marked preliminary")
    ok_cards = ok_out = ok_img = ok_mask = ok_policy = True
    n_cards = n_img = n_strong = 0
    for t in g["tasks"]:
        pol = t["image_policy"]
        published = pol["status"] == "published"
        ok_policy &= published == (t["task"] == "shopping_relevance") and bool(pol["attribution"]) \
            and bool(pol["licence"]) and (published or bool(pol.get("reason")))
        ok_policy &= set(t["score_meanings"]) == {str(i) for i in range(t["scale_min"], t["scale_max"] + 1)} \
            and "label" in t["prompt_template"]
        for x in t["targets"]:
            ok_cards &= len(x["cards"]) == 5 == len(set(x["selected_ids"])) \
                and x["selected_ids"] == [c["sample_id"] for c in x["cards"]] \
                and [c["rank"] for c in x["cards"]] == sorted(c["rank"] for c in x["cards"])
            for c in x["cards"]:
                n_cards += 1
                ok_cards &= [s["epsilon"] for s in c["steps"]] == eps and bool(c["prompt"])
                base = f"assets/transfer_gallery/{t['task']}/{x['target']}/{c['sample_id']}"
                want = [f"{base}/clean.webp"] + [f"{base}/eps_{e:.2f}.webp" for e in eps]
                for o, w in zip([c["clean"]] + c["steps"], want):
                    ok_out &= bool(o["text"]) and o["parse_ok"] is True \
                        and o["label_meaning"] == t["score_meanings"][str(o["label"])]
                    ok_img &= o["image"] == (w if published else None) \
                        and (not published or ((out_dir / w).is_file() and w in assets))
                    n_img += published
                    ok_mask &= not strong.search(o["text"]) \
                        and o["contains_strong_language"] == ("text_unmasked" in o)
                    n_strong += o["contains_strong_language"]
                ok_out &= all(s["delta_from_clean"] == s["label"] - c["clean"]["label"] for s in c["steps"])
    rep.check(ok_cards, "transfer gallery: five distinct samples per task and UAP, all four budgets, in rank order",
              f"{n_cards} cards")
    rep.check(ok_out, "transfer gallery: label, score meaning, text and change for clean and every budget")
    rep.check(ok_policy, "transfer gallery: prompt, score meanings, licence and attribution per task; "
                         "images published for the shopping task only")
    rep.check(ok_img and not any(a.startswith(("assets/transfer_gallery/moral", "assets/transfer_gallery/damage"))
                                 for a in assets),
              "transfer gallery: shopping images present and listed; no SMID or MEDIC image published",
              f"{n_img} images")
    rep.check(ok_mask, "transfer gallery: strong language masked in 'text', raw only in 'text_unmasked'",
              f"{n_strong} flagged texts")
    if results is None:
        rep.skip("transfer gallery: comparison with the saved predictions", "results/ not used")
        return
    import pandas as pd
    same = True
    for t in g["tasks"]:
        pred = pd.read_csv(results / "generalisation" / t["task"] / "predictions.csv")
        for x in t["targets"]:
            rows = pred[pred["attack"] == x["target"]]
            at = rows[np.isclose(rows["epsilon"], 1.0)].assign(a=lambda d: d["delta"].abs())
            at = at.sort_values(["a", "sample_id"], ascending=[False, True])
            down_first = at["delta"].mean() < 0
            main = at[at["delta"] < 0] if down_first else at[at["delta"] > 0]
            other = at[at["delta"] > 0] if down_first else at[at["delta"] < 0]
            same &= sorted(x["selected_ids"]) == sorted(list(main["sample_id"][:4]) + list(other["sample_id"][:1]))
            for c in x["cards"]:
                for s in c["steps"]:
                    r = rows[(rows["sample_id"] == c["sample_id"]) & np.isclose(rows["epsilon"], s["epsilon"])].iloc[0]
                    same &= s["label"] == r["model_label"] and s["delta_from_clean"] == r["delta"] \
                        and s.get("text_unmasked", s["text"]) == r["explanation"]
    rep.check(same, "transfer gallery: selection and outputs equal the saved predictions")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results_dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / "web_export")
    p.add_argument("--export_only", action="store_true",
                   help="Skip the checks that need results/ (recomputation and rebuild).")
    args = p.parse_args()

    rep = Report()
    vf = load(args.out_dir, NEW_EXPORTS[0])
    pc = load(args.out_dir, NEW_EXPORTS[1])
    print("structure")
    check_vector_field(rep, vf)
    check_composition(rep, pc)
    has_results = args.results_dir.is_dir() and not args.export_only
    print("sources")
    if has_results:
        check_vector_field_sources(rep, vf, args.results_dir)
        check_composition_sources(rep, pc, args.results_dir)
    else:
        rep.skip("recomputation from saved results", "results/ not used")
    print("determinism")
    if has_results:
        check_determinism(rep, args.results_dir, args.out_dir)
    else:
        rep.skip("two consecutive builds", "results/ not used")
    print("gallery")
    check_gallery(rep, args.out_dir, args.results_dir if has_results else None)
    check_transfer_gallery(rep, args.out_dir, args.results_dir if has_results else None)
    print("hygiene")
    check_hygiene(rep, args.out_dir)
    print(f"\n{rep.passed} passed, {rep.failed} failed")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
