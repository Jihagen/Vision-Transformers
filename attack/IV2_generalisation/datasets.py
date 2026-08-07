"""
Dataset loaders + stratified sampling for Generalisation Level 1.

Downloading requires internet, which on this cluster is only reachable from
login nodes (see attack/IV2_generalisation/prepare_datasets.py, run there).
Evaluation (evaluate.py) then runs offline on a GPU node against the frozen
local sample_manifest.csv + extracted image files this module writes.

Marqo-GS-10M (21GB for the zero_shot split alone) and QCRI/MEDIC (2.6GB for
the test split alone) are both far larger than we need for a stratified
sample of a few hundred images, so shards are fetched ONE AT A TIME (in
repo file order) and accumulated only until the requested per-stratum quota
is met (capped by max_shards) — not downloaded in full.
"""
from __future__ import annotations
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "generalisation"
RESULTS_ROOT = REPO_ROOT / "results" / "generalisation"

SHOPPING_REPO = "Marqo/marqo-GS-10M"
MEDIC_REPO = "QCRI/MEDIC"
SMID_REPO = "AIML-TUDA/smid"   # gated — requires an approved HF token


# ── shared helpers ──────────────────────────────────────────────────────

def _list_shard_files(repo_id: str, split_prefix: str) -> list[str]:
    """All parquet files for one split, in stable sorted order (shard 0 first)."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith(split_prefix) and f.endswith(".parquet"))
    if not shards:
        raise RuntimeError(f"No parquet shards found for {repo_id} split prefix {split_prefix!r}")
    return shards


def _download_shard(repo_id: str, filename: str, local_dir: Path, token: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename,
                            local_dir=str(local_dir), token=token)
    return Path(path)


def _save_image(img_field, out_path: Path) -> None:
    """img_field: HF Image-feature dict {'bytes': ..., 'path': ...} as read via pandas.read_parquet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
    im.save(out_path, format="JPEG", quality=95)


# ── Task 1: shopping relevance (Marqo-GS-10M) ────────────────────────────

def build_shopping_manifest(
    n_samples: int,
    seed: int = 42,
    split: str = "zero_shot",
    max_shards: int = 5,
    n_score_bins: int = 5,
    max_per_query: int | None = None,
) -> pd.DataFrame:
    """
    Stratified sample across the native score_linear range (1-100), capped
    per-query to avoid repetition. Downloads shards incrementally until the
    pool comfortably covers every score bin, up to max_shards.
    """
    rng = np.random.default_rng(seed)
    raw_dir = DATA_ROOT / "shopping" / "raw"
    img_dir = DATA_ROOT / "shopping" / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)

    max_per_query = max_per_query or max(2, n_samples // 20)
    shards = _list_shard_files(SHOPPING_REPO, f"data/{split}-")

    cols = ["image", "query", "product_id", "position", "title", "pair_id",
            "score_linear", "score_reciprocal", "query_id"]
    pool = []
    for shard in shards[:max_shards]:
        path = _download_shard(SHOPPING_REPO, shard, raw_dir)
        df = pd.read_parquet(path, columns=cols)
        pool.append(df)
        logger.info(f"  downloaded {shard}: +{len(df)} rows (pool={sum(len(p) for p in pool)})")
        pooled = pd.concat(pool, ignore_index=True)
        bin_counts = pd.cut(pooled["score_linear"], bins=n_score_bins).value_counts()
        if (bin_counts >= n_samples // n_score_bins).all() and len(pooled) >= n_samples * 10:
            break
    pooled = pd.concat(pool, ignore_index=True)

    pooled["score_bin"] = pd.cut(pooled["score_linear"], bins=n_score_bins, labels=False)
    per_bin = n_samples // n_score_bins
    remainder = n_samples - per_bin * n_score_bins

    selected_idx = []
    query_counts: dict[str, int] = {}
    for b in range(n_score_bins):
        bin_df = pooled[pooled["score_bin"] == b]
        target = per_bin + (1 if b < remainder else 0)
        bin_df = bin_df.sample(frac=1.0, random_state=rng.integers(1e9))
        for idx, row in bin_df.iterrows():
            if len([i for i in selected_idx if pooled.loc[i, "score_bin"] == b]) >= target:
                break
            q = row["query"]
            if query_counts.get(q, 0) >= max_per_query:
                continue
            selected_idx.append(idx)
            query_counts[q] = query_counts.get(q, 0) + 1

    sample = pooled.loc[selected_idx].reset_index(drop=True)
    logger.info(f"Shopping: selected {len(sample)}/{n_samples} requested "
                f"across {sample['query'].nunique()} unique queries")

    rows = []
    for i, row in sample.iterrows():
        sample_id = f"shopping_{i:05d}"
        img_path = img_dir / f"{sample_id}.jpg"
        _save_image(row["image"], img_path)
        reference_relevance_5 = 1 + 4 * (row["score_linear"] - 1) / 99
        rows.append({
            "sample_id": sample_id,
            "dataset": "marqo_gs10m",
            "task": "shopping_relevance",
            "image_id": str(row["product_id"]),
            "img_path": str(img_path),
            "query": row["query"],
            "title": row["title"],
            "product_id": str(row["product_id"]),
            "position": int(row["position"]),
            "score_linear": float(row["score_linear"]),
            "score_reciprocal": float(row["score_reciprocal"]),
            "reference_relevance_5": float(reference_relevance_5),
        })
    manifest = pd.DataFrame(rows)
    _write_manifest(manifest, "shopping_relevance")
    return manifest


# ── Task 3: damage severity (QCRI/MEDIC) ──────────────────────────────────

_MEDIC_SEVERITY_NAMES = ["little_or_none", "mild", "severe"]
_MEDIC_INFORMATIVE_NAMES = ["informative", "not_informative"]
_MEDIC_HUMANITARIAN_NAMES = ["affected_injured_or_dead_people", "infrastructure_and_utility_damage",
                             "not_humanitarian", "rescue_volunteering_or_donation_effort"]
_MEDIC_DISASTER_NAMES = ["earthquake", "flood", "hurricane", "fire", "landslide",
                          "not_disaster", "other_disaster"]


def build_medic_manifest(
    n_samples: int,
    seed: int = 42,
    split: str = "test",
    max_shards: int = 10,
) -> pd.DataFrame:
    """Stratified sample by the three damage_severity classes (0/1/2)."""
    rng = np.random.default_rng(seed)
    raw_dir = DATA_ROOT / "medic" / "raw"
    img_dir = DATA_ROOT / "medic" / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)

    n_classes = 3
    per_class = n_samples // n_classes
    remainder = n_samples - per_class * n_classes
    targets = {c: per_class + (1 if c < remainder else 0) for c in range(n_classes)}

    shards = _list_shard_files(MEDIC_REPO, f"data/{split}-")
    cols = ["image", "image_id", "event_name", "image_path", "damage_severity",
            "informative", "humanitarian", "disaster_types"]
    pool = []
    for shard in shards[:max_shards]:
        path = _download_shard(MEDIC_REPO, shard, raw_dir)
        df = pd.read_parquet(path, columns=cols)
        pool.append(df)
        pooled = pd.concat(pool, ignore_index=True)
        counts = pooled["damage_severity"].value_counts()
        logger.info(f"  downloaded {shard}: +{len(df)} rows, class counts={counts.to_dict()}")
        if all(counts.get(c, 0) >= targets[c] for c in range(n_classes)):
            break
    pooled = pd.concat(pool, ignore_index=True)

    selected_idx = []
    for c in range(n_classes):
        class_df = pooled[pooled["damage_severity"] == c]
        take = min(targets[c], len(class_df))
        if take < targets[c]:
            logger.warning(f"  class {_MEDIC_SEVERITY_NAMES[c]}: only {take}/{targets[c]} available "
                            f"in downloaded shards")
        chosen = class_df.sample(n=take, random_state=rng.integers(1e9))
        selected_idx.extend(chosen.index.tolist())

    sample = pooled.loc[selected_idx].reset_index(drop=True)
    logger.info(f"MEDIC: selected {len(sample)}/{n_samples} requested, "
                f"class dist={sample['damage_severity'].value_counts().to_dict()}")

    rows = []
    for i, row in sample.iterrows():
        sample_id = f"medic_{i:05d}"
        img_path = img_dir / f"{sample_id}.jpg"
        _save_image(row["image"], img_path)
        rows.append({
            "sample_id": sample_id,
            "dataset": "medic",
            "task": "damage_severity",
            "image_id": row["image_path"],
            "img_path": str(img_path),
            "gt_damage_severity": int(row["damage_severity"]),
            "gt_damage_severity_name": _MEDIC_SEVERITY_NAMES[int(row["damage_severity"])],
            "informative": _MEDIC_INFORMATIVE_NAMES[int(row["informative"])],
            "humanitarian": _MEDIC_HUMANITARIAN_NAMES[int(row["humanitarian"])],
            "disaster_type": _MEDIC_DISASTER_NAMES[int(row["disaster_types"])],
            "event_name": row["event_name"],
        })
    manifest = pd.DataFrame(rows)
    _write_manifest(manifest, "damage_severity")
    return manifest


# ── Task 2: moral evaluation (SMID) ────────────────────────────────────────

def build_smid_manifest(
    n_samples: int,
    seed: int = 42,
    n_bins_per_axis: int = 3,
    hf_token: str | None = None,
) -> pd.DataFrame:
    """
    Stratified 2D sample across (morality, arousal) so the sample isn't only
    morally-extreme images (spec section 4). AIML-TUDA/smid is a gated HF
    dataset — pass hf_token or export HUGGINGFACE_HUB_TOKEN, having first
    accepted the dataset's access terms on huggingface.co.

    Column names in metadata.csv are resolved fuzzily (case-insensitive
    substring match on "moral"/"arous"/"valence") since we haven't been able
    to inspect the file ourselves (access pending) — if the real file uses
    different names this will raise with the actual column list so the fix
    is a one-line lookup, not a silent wrong-column bug.
    """
    from huggingface_hub import hf_hub_download

    rng = np.random.default_rng(seed)
    raw_dir = DATA_ROOT / "smid" / "raw"
    img_dir = DATA_ROOT / "smid" / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)

    meta_path = hf_hub_download(repo_id=SMID_REPO, repo_type="dataset",
                                 filename="metadata.csv", local_dir=str(raw_dir), token=hf_token)
    meta = pd.read_csv(meta_path)

    # A handful of metadata rows (~1.4%) have no corresponding file under
    # images/ in the repo (withheld, presumably for content-licensing
    # reasons) — filter those out before sampling so every selected row is
    # guaranteed downloadable.
    from huggingface_hub import HfApi
    repo_files = HfApi(token=hf_token).list_repo_files(SMID_REPO, repo_type="dataset")
    available_images = {f.split("/", 1)[1] for f in repo_files if f.startswith("images/")}
    id_col = meta.columns[0]  # "file_name", e.g. "b1_p1_1.jpg"
    n_before = len(meta)
    meta = meta[meta[id_col].isin(available_images)].reset_index(drop=True)
    if len(meta) < n_before:
        logger.info(f"  {n_before - len(meta)}/{n_before} metadata rows have no matching image file — dropped")

    def _find_col(substr: str) -> str:
        matches = [c for c in meta.columns if substr.lower() in c.lower()]
        if not matches:
            raise RuntimeError(f"No column containing {substr!r} in metadata.csv. "
                                f"Available columns: {meta.columns.tolist()}")
        # Prefer an explicit "_mean" column when multiple matches exist (sd/n variants).
        mean_matches = [c for c in matches if "mean" in c.lower()]
        return mean_matches[0] if mean_matches else matches[0]

    moral_col = _find_col("moral")
    arousal_col = _find_col("arous")
    valence_col = _find_col("valence")

    meta = meta.dropna(subset=[moral_col, arousal_col, valence_col]).reset_index(drop=True)
    meta["moral_bin"] = pd.cut(meta[moral_col], bins=n_bins_per_axis, labels=False)
    meta["arousal_bin"] = pd.cut(meta[arousal_col], bins=n_bins_per_axis, labels=False)
    n_cells = n_bins_per_axis * n_bins_per_axis
    per_cell = n_samples // n_cells
    remainder = n_samples - per_cell * n_cells

    selected_idx = []
    cell_i = 0
    for mb in range(n_bins_per_axis):
        for ab in range(n_bins_per_axis):
            cell_df = meta[(meta["moral_bin"] == mb) & (meta["arousal_bin"] == ab)]
            target = per_cell + (1 if cell_i < remainder else 0)
            take = min(target, len(cell_df))
            if take > 0:
                chosen = cell_df.sample(n=take, random_state=rng.integers(1e9))
                selected_idx.extend(chosen.index.tolist())
            cell_i += 1

    sample = meta.loc[selected_idx].reset_index(drop=True)
    logger.info(f"SMID: selected {len(sample)}/{n_samples} requested across "
                f"{n_bins_per_axis}x{n_bins_per_axis} morality x arousal grid")

    rows = []
    for i, row in sample.iterrows():
        sample_id = f"smid_{i:05d}"
        img_name = str(row[id_col])
        remote_path = f"images/{img_name}" if not img_name.startswith("images/") else img_name
        local_raw = hf_hub_download(repo_id=SMID_REPO, repo_type="dataset",
                                     filename=remote_path, local_dir=str(raw_dir), token=hf_token)
        img_path = img_dir / f"{sample_id}{Path(local_raw).suffix}"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(local_raw).convert("RGB").save(img_path, format="JPEG", quality=95)
        rows.append({
            "sample_id": sample_id,
            "dataset": "smid",
            "task": "moral_evaluation",
            "image_id": img_name,
            "img_path": str(img_path),
            "human_morality": float(row[moral_col]),
            "human_arousal": float(row[arousal_col]),
            "human_valence": float(row[valence_col]),
        })
    manifest = pd.DataFrame(rows)
    _write_manifest(manifest, "moral_evaluation")
    return manifest


# ── manifest persistence (spec section 6 layout) ──────────────────────────

def _write_manifest(manifest: pd.DataFrame, task_name: str) -> Path:
    out_dir = RESULTS_ROOT / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_manifest.csv"
    manifest.to_csv(out_path, index=False)
    logger.info(f"Wrote manifest ({len(manifest)} rows) -> {out_path}")
    return out_path


def load_manifest(task_name: str) -> pd.DataFrame:
    path = RESULTS_ROOT / task_name / "sample_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No frozen sample manifest at {path} — run prepare_datasets.py for {task_name!r} first."
        )
    return pd.read_csv(path)
