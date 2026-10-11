"""Data assembly for the plot-level downstream experiments: folds, variants A-F, sequences."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import CFG, S2_BANDS, p  # noqa

PROC = p(CFG["paths"]["processed_dir"])

VARIANTS = {
    "A": S2_BANDS,
    "B": S2_BANDS + ["NDVI", "BSI"],
    "C": S2_BANDS + ["NDVI", "BSI", "BS_PROB"],
    "D": S2_BANDS + ["NDVI", "BSI", "BS_BIN"],
    "E": S2_BANDS + ["NDVI", "BSI", "BS_PROB_SHUF"],
    "F": S2_BANDS + ["NDVI", "BSI", "SCL_BARE"],
}
VARIANT_DESC = {
    "A": "S2 bands only",
    "B": "S2 + NDVI + BSI",
    "C": "S2 + NDVI + BSI + bare-soil probability (frozen model)",
    "D": "S2 + NDVI + BSI + binary bare-soil mask (p>=0.5, plot fraction)",
    "E": "C with bare-soil probability shuffled across dates",
    "F": "S2 + NDVI + BSI + Sen2Cor SCL bare-soil fraction (exploratory)",
}


def load_tables(label="HARVESTING_DAY"):
    st = pd.read_csv(os.path.join(PROC, "sample_table.csv"))
    st = st[st.YIELD > 0].copy()  # official filter for date / yield tasks
    ps = pd.read_csv(os.path.join(PROC, "plot_series.csv"))
    ps = ps[ps.UNIQUE_ID.isin(st.UNIQUE_ID) & ps.usable].copy()
    ps = ps.sort_values(["UNIQUE_ID", "day"]).reset_index(drop=True)
    st = st.set_index("UNIQUE_ID").loc[sorted(ps.UNIQUE_ID.unique())].reset_index()
    return st, ps


def make_folds(st, seed=2024):
    """Fold 0 = official split (test = official val). Folds 1-4 = official-train plots split
    into 4 plot-disjoint groups. Returns dict uid -> fold."""
    rng = np.random.default_rng(seed)
    train_plots = np.array(sorted(st[st.SPLIT == "train"].PLOT_ID.unique()))
    rng.shuffle(train_plots)
    groups = np.array_split(train_plots, 4)
    plot_fold = {pl: 0 for pl in st[st.SPLIT == "val"].PLOT_ID.unique()}
    for k, g in enumerate(groups, start=1):
        for pl in g:
            plot_fold[pl] = k
    return st.PLOT_ID.map(plot_fold).astype(int).values


def shuffled_prob(ps, seed):
    """Variant E: permute BS_PROB across the usable dates of each sample (seeded by uid & seed)."""
    out = ps.BS_PROB.values.copy()
    for uid, idx in ps.groupby("UNIQUE_ID").indices.items():
        rng = np.random.default_rng(int(uid) * 1000 + seed)
        out[idx] = out[idx][rng.permutation(len(idx))]
    return out


def sequences(st, ps, feats):
    """List of (days[T], X[T,F]) per sample, ordered like st."""
    g = ps.groupby("UNIQUE_ID")
    seqs = []
    for uid in st.UNIQUE_ID:
        b = g.get_group(uid)
        seqs.append((b.day.values.astype(np.float32), b[feats].values.astype(np.float32)))
    return seqs


def inner_split(train_idx, st, seed, frac=0.15):
    """Hold out ~15 % of training PLOTS for early stopping."""
    rng = np.random.default_rng(10_000 + seed)
    plots = np.array(sorted(st.PLOT_ID.values[train_idx]))
    plots = np.unique(plots)
    rng.shuffle(plots)
    n_val = max(1, int(round(frac * len(plots))))
    vplots = set(plots[:n_val])
    is_val = np.array([st.PLOT_ID.values[i] in vplots for i in train_idx])
    return train_idx[~is_val], train_idx[is_val]


def subsample_plots(train_idx, st, frac, seed):
    """Reduced-data experiment: keep `frac` of the training plots (seeded, test-independent)."""
    if frac >= 1:
        return train_idx
    rng = np.random.default_rng(20_000 + seed + int(frac * 1000))
    plots = np.unique(st.PLOT_ID.values[train_idx])
    rng.shuffle(plots)
    keep = set(plots[: max(2, int(round(frac * len(plots))))])
    return np.array([i for i in train_idx if st.PLOT_ID.values[i] in keep])
