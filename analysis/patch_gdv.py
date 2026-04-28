#!/usr/bin/env python3
"""
Patch all gdv.pkl and gdv_values.csv files in-place with the corrected GDV formula.

The bug: a K-weighted formula was used for K>2 which scaled GDV by ~1/K.
The fix: gdv = (intra - inter) / sqrt(D) for all K>=2.

Intra/inter distances stored in the PKLs are correct — only the scalar GDV is wrong.
This script avoids re-running expensive UMAP/t-SNE computation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import csv
import pickle
import numpy as np

METRICS_DIR = 'results/experiments/metrics'
EXPERIMENTS = [
    'gender_female_anger', 'gender_female_fear', 'gender_female_disgust',
    'gender_female_sad', 'gender_female_amusement', 'gender_female_awe',
    'gender_female_contentment', 'gender_female_excitement', 'gender_male_anger',
    'synonym_5504', 'synonym_7319',
]


def patch_experiment(exp_key: str):
    exp_dir = os.path.join(METRICS_DIR, exp_key)
    pkl_path = os.path.join(exp_dir, 'gdv.pkl')
    csv_path = os.path.join(exp_dir, 'gdv_values.csv')

    if not os.path.exists(pkl_path):
        print(f"  SKIP {exp_key} — gdv.pkl not found")
        return

    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)

    n_updated = 0
    for layer_key, info in d['layer_data'].items():
        intra = info['intra_euclidean']
        inter = info['inter_euclidean']
        D     = info['width_D']

        intra_cos = info.get('intra_cosine', 0.0)
        inter_cos = info.get('inter_cosine', 0.0)

        new_gdv_euc = (intra - inter) / np.sqrt(D)
        new_gdv_cos = (intra_cos - inter_cos) / np.sqrt(D)

        old_gdv = info.get('gdv_euclidean', float('nan'))
        info['gdv_euclidean'] = float(new_gdv_euc)
        info['gdv_cosine']    = float(new_gdv_cos)
        d['gdv_per_layer'][layer_key]        = float(new_gdv_euc)
        d['gdv_per_layer_cosine'][layer_key] = float(new_gdv_cos)
        n_updated += 1

    # Update meta — best = most negative (min)
    all_gdv = d['gdv_per_layer']
    d['meta']['best_gdv_layer'] = min(all_gdv, key=all_gdv.get)

    with open(pkl_path, 'wb') as f:
        pickle.dump(d, f)

    # Rewrite CSV
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            lk    = row['LayerKey']
            info  = d['layer_data'].get(lk, {})
            row['GDV_Euclidean']   = info.get('gdv_euclidean', row['GDV_Euclidean'])
            row['GDV_Cosine']      = info.get('gdv_cosine',    row.get('GDV_Cosine', 0))
            rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = min(all_gdv, key=all_gdv.get)
    print(f"  {exp_key:<35}  {n_updated} layers patched  "
          f"best GDV={all_gdv[best]:.4f}  ({best})")


if __name__ == '__main__':
    print("Patching GDV values (corrected formula: (intra-inter)/sqrt(D) for all K)...\n")
    for exp in EXPERIMENTS:
        patch_experiment(exp)
    print("\nDone. All PKLs and CSVs updated in-place.")
