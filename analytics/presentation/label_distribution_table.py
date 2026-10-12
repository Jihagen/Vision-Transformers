"""Label distribution (share of 5 ordinal ratings) for all 16 base persona conditions + the blank condition.
Reads raw activation result files; writes results/presentation/label_distribution_16.csv."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'data' / 'experiments' / 'gender_emotion'
OUT = ROOT / 'results' / 'presentation' / 'label_distribution_16.csv'
LABELS = ['Not Interesting', 'Slightly Interesting', 'Moderately Interesting', 'Very Interesting', 'Extremely Interesting']
EMOTIONS = ['anger', 'amusement', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sad']

def shares(records):
    labs = [r.get('interestingness') or r.get('interestingness_label') for r in records]
    labs = [l for l in labs if l in LABELS]
    n = len(labs)
    return {l: 100.0 * labs.count(l) / n for l in LABELS} | {'n': n}

rows = {}
rows['blank (no persona)'] = shares(np.load(ROOT / 'data' / 'results_blank_activations.npy', allow_pickle=True).item()['results'])
for g in ['female', 'male']:
    for e in EMOTIONS:
        d = np.load(EXP / f'results_{g}_{e}.npy', allow_pickle=True).item()
        rows[f'{g}_{e}'] = shares(d['results'])
        print(g, e, rows[f'{g}_{e}']['n'], flush=True)
df = pd.DataFrame(rows).T
df['mean_score'] = sum(df[l] * (i + 1) for i, l in enumerate(LABELS)) / 100.0
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT)
print('wrote', OUT)
