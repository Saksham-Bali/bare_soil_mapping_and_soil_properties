"""Evaluation + statistics for all plot-level experiments (pre-registered analysis, PREREGISTRATION.md).

Writes CSV/JSON tables to results/tables/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import p  # noqa

PRED = p("results", "predictions")
TAB = p("results", "tables")
os.makedirs(TAB, exist_ok=True)
N_BOOT = 10_000


def metrics(err):
    err = np.asarray(err, float)
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err ** 2)), "bias": np.mean(err),
            "MedAE": np.median(np.abs(err))}


def per_seed_table(d):
    """MAE/RMSE/bias/MedAE computed per seed over all OOF predictions, then mean +- sd over seeds."""
    rows = []
    for v, g in d.groupby("variant"):
        ms = pd.DataFrame([metrics(s.y_pred - s.y_true) for _, s in g.groupby("seed")])
        r = {"variant": v, "n_samples": g.UNIQUE_ID.nunique(), "n_plots": g.PLOT_ID.nunique(),
             "n_seeds": g.seed.nunique()}
        for c in ms:
            r[f"{c}_mean"] = ms[c].mean()
            r[f"{c}_sd"] = ms[c].std(ddof=1) if len(ms) > 1 else np.nan
        # seed-ensemble (mean prediction over seeds), reported for information
        ens = g.groupby("UNIQUE_ID").agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        r["MAE_seed_ensemble"] = metrics(ens.y_pred - ens.y_true)["MAE"]
        rows.append(r)
    return pd.DataFrame(rows)


def sample_abs_err(d):
    """Per-sample absolute error averaged over seeds -> DataFrame [UNIQUE_ID, PLOT_ID, variant, ae]."""
    d = d.assign(ae=(d.y_pred - d.y_true).abs(), se=(d.y_pred - d.y_true))
    return d.groupby(["variant", "UNIQUE_ID", "PLOT_ID"], as_index=False).agg(ae=("ae", "mean"), se=("se", "mean"))


def compare(ae, base, new, rng_seed=0):
    """Improvement = MAE_base - MAE_new (positive = `new` better). Plot-level paired tests."""
    a = ae[ae.variant == base].set_index("UNIQUE_ID")
    b = ae[ae.variant == new].set_index("UNIQUE_ID")
    j = a[["PLOT_ID", "ae"]].join(b[["ae"]], rsuffix="_new").dropna()
    j["diff"] = j.ae - j.ae_new
    plot = j.groupby("PLOT_ID")["diff"].mean()
    imp = j["diff"].mean()
    w = stats.wilcoxon(plot.values, alternative="two-sided", zero_method="wilcox")
    t = stats.ttest_1samp(plot.values, 0.0)
    # plot-cluster bootstrap of the sample-level mean difference
    rng = np.random.default_rng(rng_seed)
    groups = {k: v.values for k, v in j.groupby("PLOT_ID")["diff"]}
    keys = np.array(list(groups))
    sums = np.array([groups[k].sum() for k in keys])
    cnts = np.array([len(groups[k]) for k in keys])
    bs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(keys), len(keys))
        bs.append(sums[idx].sum() / cnts[idx].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"comparison": f"{new} vs {base}", "MAE_base": j.ae.mean(), "MAE_new": j.ae_new.mean(),
            "improvement_days": imp, "ci95_lo": lo, "ci95_hi": hi, "n_samples": len(j), "n_plots": len(plot),
            "plots_new_better": int((plot > 0).sum()), "plots_new_worse": int((plot < 0).sum()),
            "p_wilcoxon_plot": w.pvalue, "p_ttest_plot": t.pvalue}


def holm(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    m = len(pvals)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def analyse_harvest(name, primary=False):
    d = pd.read_csv(os.path.join(PRED, f"{name}.csv"))
    d = d[d.frac == 1.0]
    out = {}
    tab = per_seed_table(d)
    tab.to_csv(os.path.join(TAB, f"{name}_metrics_allfolds.csv"), index=False)
    tab0 = per_seed_table(d[d.fold == 0])
    tab0.to_csv(os.path.join(TAB, f"{name}_metrics_fold0_official_val.csv"), index=False)
    ae = sample_abs_err(d)
    ae0 = sample_abs_err(d[d.fold == 0])
    # Comparison families (PREREGISTRATION.md §7):
    #   primary:                 C vs B (not Holm-adjusted; tested alone against the success criterion)
    #   pre-registered secondary: A-B, D-C, E-C, F-B, R1-C  -> Holm-adjusted within this family only
    #   exploratory (post hoc):  B-D, B-E, A-C              -> unadjusted, reported separately
    # compare(base, new): improvement = MAE_base - MAE_new (positive = `new` better).
    rule = pd.read_csv(os.path.join(PRED, "rule_baseline.csv"))
    r1 = pd.DataFrame({"variant": "R1", "UNIQUE_ID": rule.UNIQUE_ID, "PLOT_ID": rule.PLOT_ID,
                       "ae": (rule.R1_pred - rule.y_true).abs(), "se": rule.R1_pred - rule.y_true, "fold": rule.fold})
    families = [("primary", "C vs B", "B", "C"),
                ("pre-registered secondary", "A vs B", "A", "B"),
                ("pre-registered secondary", "D vs C", "C", "D"),
                ("pre-registered secondary", "E vs C", "C", "E"),
                ("pre-registered secondary", "F vs B", "B", "F"),
                ("pre-registered secondary", "R1 vs C", "R1", "C"),
                ("exploratory (post hoc)", "D vs B", "B", "D"),
                ("exploratory (post hoc)", "E vs B", "B", "E"),
                ("exploratory (post hoc)", "C vs A", "A", "C")]

    def run_family(ae_, r1_):
        ae_ = pd.concat([ae_, r1_[["variant", "UNIQUE_ID", "PLOT_ID", "ae", "se"]]], ignore_index=True)
        rr = []
        for fam, lab, b, n in families:
            if {b, n} <= set(ae_.variant):
                r = compare(ae_, b, n)
                rr.append({"family": fam, "prereg_label": lab, **r})
        pre = [i for i, r in enumerate(rr) if r["family"] == "pre-registered secondary"]
        adj = holm([rr[i]["p_wilcoxon_plot"] for i in pre])
        for i, r in enumerate(rr):
            r["p_wilcoxon_holm_prereg_family"] = np.nan
        for i, a_ in zip(pre, adj):
            rr[i]["p_wilcoxon_holm_prereg_family"] = a_
        return rr

    rows = run_family(ae, r1)
    rows0 = run_family(ae0, r1[r1.fold == 0])
    pd.DataFrame(rows).to_csv(os.path.join(TAB, f"{name}_comparisons_allfolds.csv"), index=False)
    pd.DataFrame(rows0).to_csv(os.path.join(TAB, f"{name}_comparisons_fold0_official_val.csv"), index=False)
    # groups
    g = d.assign(ae=(d.y_pred - d.y_true).abs())
    grp = []
    for col in ["STANDARD_SEASON", "YEAR"]:
        x = g.groupby([col, "variant"]).agg(MAE=("ae", "mean"), n_samples=("UNIQUE_ID", "nunique")).reset_index()
        x = x.rename(columns={col: "group_value"})
        x.insert(0, "group", col)
        grp.append(x)
    pd.concat(grp).to_csv(os.path.join(TAB, f"{name}_by_group.csv"), index=False)
    out["metrics"] = tab
    out["comparisons"] = pd.DataFrame(rows)
    out["comparisons_fold0"] = pd.DataFrame(rows0)
    if primary:
        cb = [r for r in rows if r["comparison"] == "C vs B"][0]
        cb0 = [r for r in rows0 if r["comparison"] == "C vs B"][0]
        crit = {"criterion": "improvement >= 1.0 day AND two-sided Wilcoxon (plot level) p < 0.05",
                "primary_model": name,
                "improvement_days": cb["improvement_days"], "ci95": [cb["ci95_lo"], cb["ci95_hi"]],
                "p_wilcoxon_plot": cb["p_wilcoxon_plot"], "p_ttest_plot": cb["p_ttest_plot"],
                "MAE_B": cb["MAE_base"], "MAE_C": cb["MAE_new"], "n_samples": cb["n_samples"],
                "n_plots": cb["n_plots"],
                "success": bool(cb["improvement_days"] >= 1.0 and cb["p_wilcoxon_plot"] < 0.05),
                "official_val_only": {k: cb0[k] for k in ["MAE_base", "MAE_new", "improvement_days", "p_wilcoxon_plot",
                                                          "n_samples", "n_plots"]}}
        json.dump(crit, open(os.path.join(TAB, "PRIMARY_RESULT.json"), "w"), indent=2, default=float)
        out["criterion"] = crit
    return out


def context_baselines():
    """Label-only reference baselines (NOT pre-registered; added for interpretation).
    Uses the same folds; fitted on training folds only."""
    from src.experiments import data as D
    st, _ = D.load_tables()
    st["fold"] = D.make_folds(st)
    rows = []
    for name in ["training-fold mean harvest day", "season-median calendar baseline"]:
        errs, errs0 = [], []
        for k in range(5):
            tr, te = st[st.fold != k], st[st.fold == k]
            if name.startswith("training"):
                pred = np.full(len(te), tr.HARVESTING_DAY.mean())
            else:
                med = tr.groupby("STANDARD_SEASON").HARVESTING_DAY.median()
                pred = te.STANDARD_SEASON.map(med).fillna(tr.HARVESTING_DAY.median()).values
            e = pred - te.HARVESTING_DAY.values
            errs += list(e)
            if k == 0:
                errs0 = list(e)
        rows.append({"baseline": name, **metrics(errs), "MAE_fold0_official_val": metrics(errs0)["MAE"], "n": len(errs)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB, "context_baselines.csv"), index=False)
    return t


def analyse_rule():
    r = pd.read_csv(os.path.join(PRED, "rule_baseline.csv"))
    rows = []
    for c, label in [("R1_pred", "R1 (first bare-soil p>=0.5 after NDVI peak)"),
                     ("R2_pred", "R2 (midpoint variant)"),
                     ("R1_SCL_pred", "R1 with Sen2Cor SCL bare fraction (exploratory)")]:
        m = metrics(r[c] - r.y_true)
        trig = r["SCL_triggered" if "SCL" in c else "triggered"]
        mt = metrics((r[c] - r.y_true)[trig])
        m0 = metrics((r[c] - r.y_true)[r.fold == 0])
        rows.append({"rule": label, **{f"{k}_all": v for k, v in m.items()}, "trigger_rate": trig.mean(),
                     "MAE_triggered_only": mt["MAE"], "MAE_fold0_official_val": m0["MAE"], "n": len(r)})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "rule_baseline_metrics.csv"), index=False)
    return pd.DataFrame(rows)


def analyse_reduced():
    full = pd.read_csv(os.path.join(PRED, "harvest_ltae.csv"))
    full = full[full.variant.isin(["B", "C"])]
    red = pd.read_csv(os.path.join(PRED, "reduced_ltae.csv"))
    d = pd.concat([full, red])
    rows = []
    for fr, g in d.groupby("frac"):
        ae = sample_abs_err(g)
        c = compare(ae, "B", "C")
        mt = per_seed_table(g).set_index("variant")
        rows.append({"train_fraction": fr, "mean_n_train_samples": g.n_train_samples.mean(),
                     "MAE_B_mean": mt.loc["B", "MAE_mean"], "MAE_B_sd": mt.loc["B", "MAE_sd"],
                     "MAE_C_mean": mt.loc["C", "MAE_mean"], "MAE_C_sd": mt.loc["C", "MAE_sd"],
                     "improvement_C_over_B": c["improvement_days"], "ci95_lo": c["ci95_lo"], "ci95_hi": c["ci95_hi"],
                     "p_wilcoxon_plot": c["p_wilcoxon_plot"]})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB, "reduced_data_B_vs_C.csv"), index=False)
    return t


def analyse_yield():
    d = pd.read_csv(os.path.join(PRED, "yield_ltae.csv"))
    rows = []
    for v, g in d.groupby("variant"):
        per = []
        for s, gs in g.groupby("seed"):
            e = gs.y_pred - gs.y_true
            ss_res = (e ** 2).sum()
            ss_tot = ((gs.y_true - gs.y_true.mean()) ** 2).sum()
            per.append({"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "R2": 1 - ss_res / ss_tot,
                        "MAPE_pct": (e.abs() / gs.y_true).mean() * 100})
        per = pd.DataFrame(per)
        rows.append({"variant": v, **{f"{k}_mean": per[k].mean() for k in per}, **{f"{k}_sd": per[k].std(ddof=1) for k in per}})
    t = pd.DataFrame(rows)
    ae = sample_abs_err(d)
    comps = pd.DataFrame([compare(ae, b, n) for b, n in [("B", "C"), ("A", "B"), ("C", "E"), ("C", "D")]
                          if {b, n} <= set(d.variant)])
    comps = comps.rename(columns={"improvement_days": "improvement_yield_units"})
    # context: predict the training-fold mean yield (same folds as the models)
    from src.experiments import data as D
    st, _ = D.load_tables()
    st["fold"] = D.make_folds(st)
    e = np.concatenate([st[st.fold == k].YIELD.values - st[st.fold != k].YIELD.mean() for k in range(5)])
    yt = np.concatenate([st[st.fold == k].YIELD.values for k in range(5)])
    t = pd.concat([t, pd.DataFrame([{"variant": "null: training-fold mean", "MAE_mean": np.abs(e).mean(),
                                     "RMSE_mean": np.sqrt((e ** 2).mean()),
                                     "R2_mean": 1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum(),
                                     "MAPE_pct_mean": (np.abs(e) / yt).mean() * 100}])])
    t.to_csv(os.path.join(TAB, "yield_metrics.csv"), index=False)
    comps.to_csv(os.path.join(TAB, "yield_comparisons.csv"), index=False)
    return t, comps


def analyse_pixel():
    import glob
    rows, preds = [], []
    for f in sorted(glob.glob(os.path.join(PRED, "pixel_utae", "*.json"))):
        j = json.load(open(f))
        pr = pd.read_csv(f[:-5] + ".csv")
        preds.append(pr)
        e = pr.y_pred_plot_mean - pr.y_true
        rows.append({"variant": j["variant"], "seed": j["seed"], "epochs": j["best_epoch"],
                     "val_pixel_MAE_official_metric": j["val_pixel_MAE"], "n_pixels": j["n_pixels"],
                     "inner_val_pixel_MAE_epoch0": j["log"][0]["inner_val_pixel_mae"],
                     "inner_val_pixel_MAE_final": j["log"][-1]["inner_val_pixel_mae"],
                     "val_plot_level_MAE": e.abs().mean(), "val_plot_level_RMSE": np.sqrt((e ** 2).mean()),
                     "n_samples": len(pr)})
    if not rows:
        return None, None
    t = pd.DataFrame(rows).sort_values("variant")
    t.to_csv(os.path.join(TAB, "pixel_utae_official_split.csv"), index=False)
    d = pd.concat(preds).rename(columns={"y_pred_plot_mean": "y_pred"})
    ae = sample_abs_err(d)
    comps = pd.DataFrame([compare(ae, b, n) for b, n in [("B", "C"), ("A", "B"), ("C", "D"), ("C", "E")]
                          if {b, n} <= set(d.variant)])
    comps.to_csv(os.path.join(TAB, "pixel_utae_comparisons_plot_level.csv"), index=False)
    return t, comps


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for name, prim in [("harvest_ltae", True), ("harvest_gridxgb", False)]:
        if os.path.exists(os.path.join(PRED, f"{name}.csv")):
            o = analyse_harvest(name, primary=prim)
            print(f"\n=== {name}\n", o["metrics"].round(2).to_string(index=False))
            print(o["comparisons"].round(4).to_string(index=False))
            if prim:
                print(json.dumps(o["criterion"], indent=1, default=float))
    print(analyse_rule().round(2).to_string(index=False))
    print(context_baselines().round(2).to_string(index=False))
    if os.path.exists(os.path.join(PRED, "reduced_ltae.csv")):
        print(analyse_reduced().round(3).to_string(index=False))
    if os.path.exists(os.path.join(PRED, "yield_ltae.csv")):
        t, c = analyse_yield()
        print(t.round(3).to_string(index=False))
        print(c.round(4).to_string(index=False))
    t, c = analyse_pixel()
    if t is not None:
        print(t.round(3).to_string(index=False))
        print(c.round(4).to_string(index=False))
