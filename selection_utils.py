# selection_utils.py
import os
import numpy as np
import pandas as pd

LABELS = [
    "Not Interesting",
    "Slightly Interesting",
    "Moderately Interesting",
    "Very Interesting",
    "Extremely Interesting",
]

SEED = 42
N_PER_LABEL = 5
SELECTED_25_PKL = "data/selected_uniform_25.pkl"
SELECTED_TOTAL_PKL = "data/selected_uniform_total.pkl"  # output for the concatenated set
DEFAULT_TOTAL_N = 500

os.makedirs("data", exist_ok=True)

def canonicalize_label(val):
    """Map raw values to one of the five canonical LABELS."""
    if pd.isna(val):
        return None
    try:
        as_int = int(val)
        if 0 <= as_int <= 4:
            return LABELS[as_int]
    except Exception:
        pass

    s = str(val).strip().lower()
    mapping = {
        "not interesting": LABELS[0],
        "slightly interesting": LABELS[1],
        "moderately interesting": LABELS[2],
        "very interesting": LABELS[3],
        "extremely interesting": LABELS[4],
    }
    for k, v in mapping.items():
        if k in s:
            return v
    for v in LABELS:
        if v.lower() == s:
            return v
    return None

def build_uniform_selection(df_images, save_path=SELECTED_25_PKL, seed=SEED, n_per_label=N_PER_LABEL):
    """Sample n_per_label per label (oversample within-label if needed) and persist the selection."""
    if os.path.exists(save_path):
        return pd.read_pickle(save_path)

    df = df_images.copy()
    df["interestingness_label"] = df["interestingness"].apply(canonicalize_label)
    df = df[df["interestingness_label"].isin(LABELS)].copy()

    rng = np.random.default_rng(seed)
    parts = []
    for lab in LABELS:
        pool = df[df["interestingness_label"] == lab]
        if len(pool) == 0:
            raise ValueError(f"No images for label: {lab}")
        replace = len(pool) < n_per_label
        if replace:
            print(f"[warn] Label '{lab}' has only {len(pool)} items; oversampling to reach {n_per_label}.")
        sample_idx = rng.choice(pool.index, size=n_per_label, replace=replace)
        parts.append(pool.loc[sample_idx, ["filename", "interestingness_label"]])

    df_sel = pd.concat(parts, ignore_index=True)
    df_sel.to_pickle(save_path)
    return df_sel

def build_concat_total(
    df_images,
    total_n=DEFAULT_TOTAL_N,
    include_path=SELECTED_25_PKL,
    save_path=SELECTED_TOTAL_PKL,
    seed=SEED,
):
    """
    Load/create the base-25, then simply append UNIQUE rows (no replacement)
    from df_images until reaching total_n. If insufficient unique rows exist,
    return as many as possible and warn.
    """
    # Canonicalize labels to keep consistent columns
    df = df_images.copy()
    df["interestingness_label"] = df["interestingness"].apply(canonicalize_label)
    df = df[df["interestingness_label"].isin(LABELS)].copy()

    # Load or create the base 25
    if os.path.exists(include_path):
        df_base = pd.read_pickle(include_path)[["filename", "interestingness_label"]].copy()
    else:
        df_base = build_uniform_selection(df_images, save_path=include_path, seed=seed)

    base_n = len(df_base)
    if total_n <= base_n:
        print(f"[info] total_n={total_n} ≤ base_n={base_n}; returning the base selection.")
        df_base.to_pickle(save_path)
        return df_base

    # Exclude already chosen filenames
    already = set(df_base["filename"])
    pool = df[~df["filename"].isin(already)][["filename", "interestingness_label"]]

    need = total_n - base_n
    take = min(need, len(pool))

    rng = np.random.default_rng(seed)
    if take > 0:
        idx = rng.choice(pool.index, size=take, replace=False)
        df_more = pool.loc[idx]
        df_final = pd.concat([df_base, df_more], ignore_index=True)
    else:
        df_final = df_base.copy()

    if len(df_final) < total_n:
        print(f"[warn] Only {len(df_final)} unique items available (short by {total_n - len(df_final)}).")

    df_final.to_pickle(save_path)
    return df_final

if __name__ == "__main__":
    # Load the CSV produced by rate.py
    df_csv = pd.read_csv("data/interestingness_results.csv")  # ← correct file

    # 1) Ensure or create the base 25
   # df_25 = build_uniform_selection(df_csv, save_path=SELECTED_25_PKL, seed=SEED)
   # print("Saved base 25 ->", SELECTED_25_PKL)
   # print(df_25.groupby("interestingness_label").size())

    # 2) Append unique rows up to DEFAULT_TOTAL_N
    df_total = build_concat_total(
        df_csv,
        total_n=DEFAULT_TOTAL_N,
        include_path=SELECTED_25_PKL,
        save_path=SELECTED_TOTAL_PKL,
        seed=SEED,
    )
    print(f"Saved total (≤{DEFAULT_TOTAL_N}) ->", SELECTED_TOTAL_PKL)
    print(df_total.groupby("interestingness_label").size())
