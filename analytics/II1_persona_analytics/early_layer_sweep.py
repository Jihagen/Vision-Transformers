"""
Early-layer sweep for the Chapter 8 presentation figure.

For each candidate language layer before the country-dominated layer (23),
embed the same 48 conditions (16 base + 16 Germany + 16 Nigeria) with the
exact UMAP settings used in the notebook (cosine, n_neighbors=10,
random_state=42), and measure what each point's nearest neighbour shares:
country vs emotion+gender.

Selection rule (stated up front, not chosen by eye): EARLY_LAYER is the
candidate layer with the LOWEST country nearest-neighbour share — i.e. the
latest layer before country takes over — with ties broken by the HIGHEST
emotion+gender nearest-neighbour share. This picks a layer where persona
content, not country, organises the geometry, so the presentation sequence
reads early -> country-dominated (23) -> persona-content (29).

Run as a batch CPU job (loads ~32 GB of raw activations):
    sbatch hpc_infrastructure/run_early_layer_sweep.slurm
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import umap
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "data" / "experiments"
OUT = ROOT / "results" / "analytics" / "early_layer_sweep"

EMOTIONS = ["anger", "amusement", "awe", "contentment", "disgust", "excitement", "fear", "sad"]
CANDIDATE_LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
REF_LAYERS = [23, 29]

VARIANTS = {
    "Base": (EXP / "gender_emotion", "{g}_{e}"),
    "Germany": (EXP / "gender_emotion_germany", "{g}_{e}_extended_germany"),
    "Nigeria": (EXP / "gender_emotion_nigeria", "{g}_{e}_extended_nigeria"),
}


def load_condition_means(path: Path, layers: list[int]) -> dict[int, np.ndarray]:
    obj = np.load(path, allow_pickle=True).item()
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for r in obj["results"]:
        for L in layers:
            key = f"llm_layer_{L}_rating_token"
            if key not in r["embeddings"]:
                continue
            v = np.asarray(r["embeddings"][key], dtype=np.float64).ravel()
            sums[L] = sums.get(L, 0) + v
            counts[L] = counts.get(L, 0) + 1
    return {L: sums[L] / counts[L] for L in sums}


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def main() -> None:
    all_layers = sorted(set(CANDIDATE_LAYERS + REF_LAYERS))
    labels, countries, genders, emotions = [], [], [], []
    means: dict[str, dict[int, np.ndarray]] = {}

    for country, (folder, pattern) in VARIANTS.items():
        for g in ["female", "male"]:
            for e in EMOTIONS:
                label = pattern.format(g=g, e=e)
                suffix = "" if country == "Base" else f"_extended_{country.lower()}"
                path = folder / f"results_{g}_{e}{suffix}.npy"
                if not path.exists():
                    raise FileNotFoundError(path)
                means[label] = load_condition_means(path, all_layers)
                labels.append(label)
                countries.append(country)
                genders.append(g)
                emotions.append(e)
        print(f"loaded {country}", flush=True)

    rows = []
    for L in all_layers:
        mat = np.stack([unit(means[l][L]) for l in labels])
        emb = umap.UMAP(n_neighbors=10, metric="cosine", random_state=42).fit_transform(mat)
        D = cdist(emb, emb)
        np.fill_diagonal(D, np.inf)
        nn = D.argmin(axis=1)
        n = len(labels)
        country_share = sum(countries[i] == countries[nn[i]] for i in range(n)) / n
        content_share = sum(emotions[i] == emotions[nn[i]] and genders[i] == genders[nn[i]]
                            for i in range(n)) / n
        rows.append({"layer": L, "country_nn_share": country_share, "emotion_gender_nn_share": content_share})
        print(f"language_{L}: country {country_share:.2f}  emotion+gender {content_share:.2f}", flush=True)

    candidates = [r for r in rows if r["layer"] in CANDIDATE_LAYERS]
    early = min(candidates, key=lambda r: (r["country_nn_share"], -r["emotion_gender_nn_share"]))
    result = {
        "rule": "min country NN share among candidates; tie-break max emotion+gender NN share",
        "candidate_layers": CANDIDATE_LAYERS,
        "early_layer": early["layer"],
        "table": rows,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "early_layer_sweep.json").write_text(json.dumps(result, indent=2))

    L = early["layer"]
    for slug, variant in [("base", "Base"), ("extended_germany", "Germany"), ("extended_nigeria", "Nigeria")]:
        d = {l: means[l][L].astype(np.float32) for l in labels if countries[labels.index(l)] == variant}
        p = ROOT / "results" / "analytics" / slug / f"language_{L}" / f"mean_vectors_language_{L}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, d, allow_pickle=True)
    print(f"EARLY_LAYER = {L}  (mean vectors saved to results/analytics/<variant>/language_{L}/)", flush=True)


if __name__ == "__main__":
    main()
