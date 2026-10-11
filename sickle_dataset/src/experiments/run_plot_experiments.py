"""Run the plot-level downstream experiments (variants x folds x seeds).

Examples
  python -m src.experiments.run_plot_experiments --exp harvest_ltae --model ltae --variants A B C D E F
  python -m src.experiments.run_plot_experiments --exp harvest_gridxgb --model gridxgb --variants A B C D E F
  python -m src.experiments.run_plot_experiments --exp reduced_ltae --model ltae --variants B C --fracs 0.1 0.25 0.5
  python -m src.experiments.run_plot_experiments --exp yield_ltae --model ltae --target YIELD --variants A B C

Output: results/predictions/<exp>.csv with one row per (sample, variant, fold, seed, frac).
Resumable: finished (variant, fold, seed, frac) jobs are skipped.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import p  # noqa
from src.experiments import data as D  # noqa
from src.experiments import models as M  # noqa

_S = {}


def setup(target):
    st, ps = D.load_tables()
    folds = D.make_folds(st)
    y = st[target].values.astype(np.float64)
    _S.update(st=st, ps=ps, folds=folds, y=y)


def job(args):
    model, variant, fold, seed, frac = args
    st, ps, folds, y = _S["st"], _S["ps"].copy(), _S["folds"], _S["y"]
    if variant == "E":
        ps["BS_PROB_SHUF"] = D.shuffled_prob(ps, seed)
    seqs = D.sequences(st, ps, D.VARIANTS[variant])
    te = np.where(folds == fold)[0]
    tr_all = np.where(folds != fold)[0]
    tr_all = D.subsample_plots(tr_all, st, frac, seed)
    t0 = time.time()
    if model == "ltae":
        tr, va = D.inner_split(tr_all, st, seed + 100 * fold)
        pred, info = M.train_ltae(seqs, y, tr, va, te, seed=seed + 1000 * fold)
    else:
        G = M.grid_features(seqs)
        pred = M.train_gridxgb(G, y, tr_all, te, seed=seed + 1000 * fold)
        info = {}
    return pd.DataFrame({
        "UNIQUE_ID": st.UNIQUE_ID.values[te], "PLOT_ID": st.PLOT_ID.values[te],
        "SPLIT": st.SPLIT.values[te], "STANDARD_SEASON": st.STANDARD_SEASON.values[te],
        "YEAR": st.YEAR.values[te], "fold": fold, "seed": seed, "variant": variant, "model": model,
        "frac": frac, "n_train_samples": len(tr_all), "y_true": y[te], "y_pred": pred,
        "best_epoch": info.get("best_epoch", np.nan), "secs": time.time() - t0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", default="ltae", choices=["ltae", "gridxgb"])
    ap.add_argument("--target", default="HARVESTING_DAY")
    ap.add_argument("--variants", nargs="+", default=list("ABCDE"))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--fracs", nargs="+", type=float, default=[1.0])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = p("results", "predictions", f"{a.exp}.csv")
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev.variant, prev.fold, prev.seed, prev.frac))
    jobs = [(a.model, v, f, s, fr) for fr in a.fracs for s in a.seeds for v in a.variants for f in a.folds
            if (v, f, s, fr) not in done]
    print(f"{a.exp}: {len(jobs)} jobs to run ({len(done)} done)", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=setup, initargs=(a.target,)) as pool:
        for k, res in enumerate(pool.imap_unordered(job, jobs), 1):
            res.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            r = res.iloc[0]
            mae = np.abs(res.y_true - res.y_pred).mean()
            print(f"[{k}/{len(jobs)} {time.time() - t0:.0f}s] {r.variant} fold{r.fold} seed{r.seed} "
                  f"frac{r.frac} n_test={len(res)} MAE={mae:.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
