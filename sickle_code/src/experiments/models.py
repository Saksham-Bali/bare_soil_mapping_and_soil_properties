"""Downstream models.

PlotLTAE  : per-date MLP embedding -> official SICKLE L-TAE temporal encoder (third_party/
            SICKLE_official/models/ltae.py, unmodified) -> MLP regression head.
            The L-TAE is applied with H=W=1 (one "pixel" = the plot-mean time series).
GridXGB   : XGBoost regressor on features linearly interpolated to a fixed 10-day grid.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "SICKLE_official"))
from models.ltae import LTAE2d  # noqa: E402  (official implementation)

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))


class PlotLTAE(nn.Module):
    def __init__(self, n_in, d_emb=128, n_head=16, d_k=4, d_model=256, mlp=(256, 128), dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, d_emb))
        self.ltae = LTAE2d(in_channels=d_emb, n_head=n_head, d_k=d_k, mlp=list(mlp), dropout=dropout,
                           d_model=d_model, T=1000, return_att=True, positional_encoding=True)
        self.head = nn.Sequential(nn.Linear(mlp[-1], 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, days, pad_mask):
        # x: B x T x F ; days: B x T ; pad_mask: B x T (True = padding)
        e = self.embed(x)                                    # B x T x d
        e = e.masked_fill(pad_mask[..., None], 0.0)
        out, att = self.ltae(e[..., None, None], batch_positions=days, pad_mask=pad_mask)
        return self.head(out[:, :, 0, 0]).squeeze(-1), att


def collate(seqs, idx, mu, sd, drop_p=0.0, rng=None):
    sel = [seqs[i] for i in idx]
    if drop_p > 0:
        out = []
        for d, x in sel:
            keep = rng.random(len(d)) >= drop_p
            if keep.sum() < 2:
                keep[:] = True
            out.append((d[keep], x[keep]))
        sel = out
    T = max(len(d) for d, _ in sel)
    F = sel[0][1].shape[1]
    X = np.zeros((len(sel), T, F), np.float32)
    D = np.zeros((len(sel), T), np.float32)
    M = np.ones((len(sel), T), bool)
    for k, (d, x) in enumerate(sel):
        n = len(d)
        X[k, :n] = (x - mu) / sd
        D[k, :n] = d
        M[k, :n] = False
    X = np.nan_to_num(X)
    return torch.from_numpy(X), torch.from_numpy(D), torch.from_numpy(M)


def train_ltae(seqs, y, tr, va, te, seed, epochs=200, patience=40, bs=16, lr=1e-3, wd=1e-4, drop_p=0.2):
    """Train on tr, early-stop on va (inner validation), predict te. Returns (pred_te, info)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    allx = np.concatenate([seqs[i][1] for i in tr])
    mu = np.nanmean(allx, 0)
    sd = np.nanstd(allx, 0) + 1e-6
    ymu, ysd = y[tr].mean(), y[tr].std() + 1e-6
    model = PlotLTAE(n_in=allx.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    yt = torch.tensor((y - ymu) / ysd, dtype=torch.float32)
    Xv, Dv, Mv = collate(seqs, va, mu, sd)
    best, best_state, best_ep, bad = np.inf, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), bs):
            idx = perm[b:b + bs]
            if len(idx) < 2:
                continue
            X, D, M = collate(seqs, idx, mu, sd, drop_p, rng)
            pred, _ = model(X, D, M)
            loss = ((pred - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv, _ = model(Xv, Dv, Mv)
        mae = np.abs(pv.numpy() * ysd + ymu - y[va]).mean()
        if mae < best - 1e-6:
            best, best_ep, bad = mae, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    Xt, Dt, Mt = collate(seqs, te, mu, sd)
    with torch.no_grad():
        pt, att = model(Xt, Dt, Mt)
    return pt.numpy() * ysd + ymu, {"best_epoch": best_ep, "inner_val_mae": float(best), "epochs_run": ep + 1}


GRID = np.arange(1, 182, 10, dtype=float)  # days 1, 11, ..., 181


def grid_features(seqs):
    out = []
    for d, x in seqs:
        cols = []
        for f in range(x.shape[1]):
            ok = ~np.isnan(x[:, f])
            cols.append(np.interp(GRID, d[ok], x[ok, f]) if ok.sum() else np.full(len(GRID), np.nan))
        out.append(np.concatenate(cols))
    return np.stack(out)


def train_gridxgb(G, y, tr, te, seed):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.5, random_state=seed, n_jobs=1)
    m.fit(G[tr], y[tr])
    return m.predict(G[te])
