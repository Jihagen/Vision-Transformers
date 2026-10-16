"""
Bootstrap confidence intervals for the cross-model agreement figures (17/18).

Resamples images (the same resamples for every perturbation of one model × task) and
recomputes direction consistency and the per-image Spearman ρ with Llama 4 Scout, plus
the paired difference from the matched control at the same ε: random noise where it was
run, otherwise the shuffled workload UAP. Reads predictions.csv only; writes
results/presentation/cross_model_agreement_ci.csv, which figures 17/18 pick up.

Repeated resampling: run it inside a SLURM job (hpc_infrastructure/cross_model_bootstrap.slurm),
not on the login node.
    python analytics/presentation/cross_model_bootstrap.py [--n_boot 2000]
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_presentation_figures import OUT, XM_MODELS, _model_predictions

CONTROLS = ('noise', 'shuffled_workload')


def _canon(attack: str) -> str:
    return 'noise' if attack.startswith('noise_seed') else attack


def _consistency(x: np.ndarray) -> np.ndarray:
    """x: B × n × seeds -> (n up − n down) / n changed, seeds pooled."""
    up, down = (x > 0).sum((1, 2)), (x < 0).sum((1, 2))
    moved = up + down
    return np.where(moved > 0, (up - down) / np.maximum(moved, 1), np.nan)


def _spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x, y: B × n × seeds -> Spearman ρ over images, averaged over seeds."""
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # constant resample -> NaN, mean of all-NaN
        return np.nanmean((rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1)), axis=1)


def _ci(b: np.ndarray) -> tuple[float, float]:
    b = b[~np.isnan(b)]
    return (np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b) else (np.nan, np.nan)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    wide = lambda df: df.pivot_table(index='sample_id', columns=['attack', 'epsilon'], values='delta')
    preds = {m: p for m, _ in XM_MODELS if (p := _model_predictions(m))}
    llama = {t: wide(d) for t, d in preds.get('llama4-scout', {}).items()}
    rows = []
    for model, per_task in preds.items():
        for task, df in per_task.items():
            w = wide(df)
            ref = llama.get(task) if model != 'llama4-scout' else None
            if ref is not None:
                ids = w.index.intersection(ref.index)
                w, ref = w.loc[ids], ref.loc[ids]
            idx = rng.integers(0, len(w), (args.n_boot, len(w)))
            conds: dict[tuple, list] = {}
            for a, e in w.columns:
                conds.setdefault((_canon(a), e), []).append((a, e))
            stats = {}
            for key, cols in conds.items():
                x = w[cols].to_numpy()
                stats[key] = {'consistency': (_consistency(x[None])[0], _consistency(x[idx]))}
                if ref is not None and all(c in ref.columns for c in cols):
                    y = ref[cols].to_numpy()
                    stats[key]['rho_llama'] = (_spearman(x[None], y[None])[0], _spearman(x[idx], y[idx]))
            for (a, e), per_metric in stats.items():
                ctrl = None if a in CONTROLS else next((c for c in [('noise', e), ('shuffled_workload', e)]
                                                        if c in stats), None)
                for metric, (est, boot) in per_metric.items():
                    lo, hi = _ci(boot)
                    r = dict(model=model, task=task, attack=a, epsilon=e, metric=metric, n_images=len(w),
                             estimate=est, ci_lo=lo, ci_hi=hi, control=None, diff=np.nan, diff_lo=np.nan, diff_hi=np.nan)
                    if ctrl and metric in stats[ctrl]:
                        c_est, c_boot = stats[ctrl][metric]
                        d_lo, d_hi = _ci(boot - c_boot)
                        r.update(control=ctrl[0], diff=est - c_est, diff_lo=d_lo, diff_hi=d_hi)
                    rows.append(r)
            print(f'{model} / {task}: {len(conds)} conditions, {len(w)} images', flush=True)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'cross_model_agreement_ci.csv', index=False)
    print('wrote', OUT / 'cross_model_agreement_ci.csv', f'({len(out)} rows, {args.n_boot} resamples)')


if __name__ == '__main__':
    main()
