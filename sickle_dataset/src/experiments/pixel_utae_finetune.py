"""Secondary experiment with the OFFICIAL SICKLE architecture (pixel-level U-TAE, official split).

Training U-TAE from scratch is ~6.5 h per run on this CPU-only setup (measured: 0.84 s per
sample fwd+bwd), so for A-E we FINE-TUNE the authors' released S2 U-TAE harvest checkpoint
(weights.zip: runs/wacv_2024/harvesting_date/[S2]_utae) for a fixed small number of epochs.

  * data: official SICKLE preprocessing (all S2 dates in the 184-day window, raw DN, 32x32),
          official masks / ignore index / "remove plots not in split", official RMSE loss.
  * extra per-date channels (pixel level), scaled x10000 to the DN range of the bands:
        A: none | B: NDVI, BSI | C: B + bare-soil prob | D: B + binary mask (p>=0.5)
        E: C with the bare-soil prob time axis permuted per sample (seeded)
    New input channels get zero-initialised weights in the first conv, so every variant starts
    from exactly the same function as the released model.
  * train = official train minus 15 % of its plots (inner validation, logged only; a FIXED number of
    epochs is used because the released checkpoint already saw those plots); BatchNorm stats frozen;
    test = official val (the official test labels are withheld).
  * no cloud masking (official pipeline has none) - identical for all variants.
Caveat: the released checkpoint's best epoch was chosen by the authors on official val, so val is
not a pristine test set for this model family; the bias is shared by all variants.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "SICKLE_official"))
from utils import model_utils  # noqa: E402
from utils.utae_utils import pad_collate  # noqa: E402
from src.common import BS_FEATURES, BS_THRESHOLD, S2_BANDS, bsi, ndvi  # noqa: E402
from src.sickle.eval_official_checkpoint import IGN, official_mask  # noqa: E402
from src.experiments.data import inner_split  # noqa: E402

PROC = os.path.join(ROOT, "data_processed")
EXTRA = {"A": [], "B": ["NDVI", "BSI"], "C": ["NDVI", "BSI", "PROB"], "D": ["NDVI", "BSI", "BIN"],
         "E": ["NDVI", "BSI", "PROB_SHUF"]}


def load_samples(model_pkl):
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    bs = pickle.load(open(model_pkl, "rb"))
    df = pd.read_csv(os.path.join(PROC, "sickle_dataset_tabular.csv"))
    df = df[df.YIELD > 0].reset_index(drop=True)
    S = {}
    for split in ["train", "val"]:
        sub = df[df.SPLIT == split]
        pids = set(sub.PLOT_ID)
        for _, r in sub.iterrows():
            uid = int(r.UNIQUE_ID)
            d = np.load(os.path.join(PROC, "s2_32", f"{uid}.npz"))
            x = d["bands"].astype(np.float32)                     # T,12,32,32
            T = x.shape[0]
            X = x.transpose(0, 2, 3, 1).reshape(-1, 12)[:, [S2_BANDS.index(f) for f in BS_FEATURES]]
            prob = bs.predict_proba(pd.DataFrame(X, columns=BS_FEATURES))[:, 1].reshape(T, 32, 32)
            b = {k: x[:, S2_BANDS.index(k)] for k in ["B2", "B4", "B8", "B11"]}
            ex = {"NDVI": np.nan_to_num(ndvi(b["B4"], b["B8"])), "BSI": np.nan_to_num(bsi(b["B2"], b["B4"], b["B8"], b["B11"])),
                  "PROB": prob, "BIN": (prob >= BS_THRESHOLD).astype(np.float32)}
            m = official_mask(d["mask"].astype(np.float32), pids)
            S[uid] = dict(x=x, ex=ex, dates=d["dates"].astype(np.int64), y=m[4], own=(d["mask"][0] == r.PLOT_ID),
                          plot=int(r.PLOT_ID), split=split, label=float(r.HARVESTING_DAY))
    assert max(len(s["dates"]) for s in S.values()) <= T_FIX
    return df, S


def build_input(s, variant, seed, uid):
    chans = [s["x"]]
    for k in EXTRA[variant]:
        if k == "PROB_SHUF":
            rng = np.random.default_rng(uid * 1000 + seed)
            a = s["ex"]["PROB"][rng.permutation(len(s["dates"]))]
        else:
            a = s["ex"][k]
        chans.append((a * 10000.0)[:, None].astype(np.float32))
    return torch.tensor(np.concatenate(chans, axis=1))


def make_model(n_extra):
    cdir = os.path.join(PROC, "official_checkpoints", "harvesting_date", "[S2]_utae")
    conf = json.load(open(os.path.join(cdir, "conf.json")))
    conf["device"] = "cpu"
    m = model_utils.Fusion_model(SimpleNamespace(**conf))
    m.load_state_dict(torch.load(os.path.join(cdir, "checkpoint_best.pth.tar"), map_location="cpu",
                                 weights_only=False)["model"])
    if n_extra:
        conv = m.models["S2"].in_conv.conv.conv[0]
        w = conv.weight.data
        new = torch.nn.Conv2d(w.shape[1] + n_extra, w.shape[0], conv.kernel_size, conv.stride, conv.padding,
                              bias=conv.bias is not None, padding_mode=conv.padding_mode)
        new.weight.data.zero_()
        new.weight.data[:, : w.shape[1]] = w
        if conv.bias is not None:
            new.bias.data = conv.bias.data.clone()
        m.models["S2"].in_conv.conv.conv[0] = new
    return m, conf


T_FIX = 74  # max number of S2 dates of any harvest sample (checked in load_samples)


def run_batches(model, S, uids, variant, seed, bs, train=False, opt=None, rng=None):
    order = list(uids) if not train else list(rng.permutation(uids))
    preds = {}
    for b in range(0, len(order), bs):
        ids = order[b:b + bs]
        # Every sequence is zero-padded to the SAME length T_FIX (max over the dataset). The official
        # U-TAE output depends on the amount of padding (GroupNorm over time incl. padded steps): with
        # the released checkpoint, val MAE is 12.3 d at batch 32 but 21.2 d at batch 4. Fixed padding
        # removes this batch dependence and is identical for all variants, train and test.
        items = []
        for u in ids:
            xi = build_input(S[u], variant, seed, u)
            di = torch.tensor(S[u]["dates"])
            padn = T_FIX - xi.shape[0]
            xi = torch.cat([xi, torch.zeros((padn,) + tuple(xi.shape[1:]))])
            di = torch.cat([di, torch.zeros(padn, dtype=di.dtype)])
            items.append({"S2": (xi, di)})
        data = pad_collate(items, pad_value=0)
        y = torch.tensor(np.stack([S[u]["y"] for u in ids]))
        if train:
            yp = model(data).squeeze(1)
            sel = y != IGN
            loss = torch.sqrt(((yp[sel] - y[sel]) ** 2).mean() + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                yp = model(data).squeeze(1)
        for k, u in enumerate(ids):
            preds[u] = yp[k].detach().numpy()
    return preds


def pixel_mae(S, preds):
    e = np.concatenate([np.abs(preds[u] - S[u]["y"])[S[u]["y"] != IGN] for u in preds])
    return float(e.mean()), int(e.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out_dir = os.path.join(ROOT, "results", "predictions", "pixel_utae")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    df, S = load_samples(os.path.join(PROC, "universal_bare_soil_xgboost.pkl"))
    st = pd.DataFrame([{"UNIQUE_ID": u, "PLOT_ID": s["plot"], "split": s["split"]} for u, s in S.items()])
    tr_idx = np.where(st.split == "train")[0]
    tr, va = inner_split(tr_idx, st, a.seed)
    tr_u, va_u = st.UNIQUE_ID.values[tr], st.UNIQUE_ID.values[va]
    te_u = st.UNIQUE_ID.values[st.split == "val"]
    model, conf = make_model(len(EXTRA[a.variant]))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    model.eval()
    p0 = run_batches(model, S, va_u, a.variant, a.seed, a.bs)
    best, best_ep, best_state = pixel_mae(S, p0)[0], 0, {k: v.clone() for k, v in model.state_dict().items()}
    log = [{"epoch": 0, "inner_val_pixel_mae": best}]
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        for mod in model.modules():  # freeze BatchNorm statistics (small CPU batches)
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()
        run_batches(model, S, tr_u, a.variant, a.seed, a.bs, train=True, opt=opt, rng=rng)
        model.eval()
        mae = pixel_mae(S, run_batches(model, S, va_u, a.variant, a.seed, a.bs))[0]
        log.append({"epoch": ep, "inner_val_pixel_mae": mae, "secs": time.time() - t0})
        print(tag, log[-1], flush=True)
        # NOTE: no checkpoint selection. The released checkpoint was trained on ALL official-train
        # plots (incl. our inner-val plots), so inner-val MAE cannot select an epoch fairly; every
        # variant is evaluated after the same fixed number of epochs. Inner-val MAE is only logged.
    best_ep = a.epochs
    model.eval()
    pt = run_batches(model, S, te_u, a.variant, a.seed, a.bs)
    pmae, npx = pixel_mae(S, pt)
    rows = [{"UNIQUE_ID": u, "PLOT_ID": S[u]["plot"], "variant": a.variant, "seed": a.seed,
             "y_true": S[u]["label"], "y_pred_plot_mean": float(pt[u][S[u]["own"]].mean() + 1.0)} for u in te_u]
    # +1: official target is mask-1 (0-based day); plot-level labels use the CSV convention (1-based)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{tag}.csv"), index=False)
    json.dump({"variant": a.variant, "seed": a.seed, "best_epoch": best_ep, "val_pixel_MAE": pmae, "n_pixels": npx,
               "log": log}, open(os.path.join(out_dir, f"{tag}.json"), "w"), indent=1)
    print(tag, "VAL pixel MAE", pmae, "best epoch", best_ep, flush=True)


if __name__ == "__main__":
    main()
