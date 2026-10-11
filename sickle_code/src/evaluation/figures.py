"""Research figures (PNG 300 dpi + PDF) -> results/figures/"""
import os
import pickle
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import BS_FEATURES, BS_THRESHOLD, INVALID_SCL, S2_BANDS, ndvi, p  # noqa: E402
from src.experiments import data as D  # noqa: E402
from src.evaluation.stats import sample_abs_err  # noqa: E402

warnings.filterwarnings("ignore")
FIG = p("results", "figures")
PRED = p("results", "predictions")
os.makedirs(FIG, exist_ok=True)

# reference categorical palette (fixed order, light mode) + neutral inks
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4", "F": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 110})
LABEL = {"A": "A  S2 bands", "B": "B  + NDVI, BSI", "C": "C  + bare-soil prob.", "D": "D  + binary mask",
         "E": "E  C, prob. shuffled in time", "F": "F  + SCL bare (expl.)"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_variant_comparison():
    t = pd.read_csv(p("results", "tables", "harvest_ltae_metrics_allfolds.csv"))
    g = pd.read_csv(p("results", "tables", "harvest_gridxgb_metrics_allfolds.csv"))
    r = pd.read_csv(p("results", "tables", "rule_baseline_metrics.csv"))
    ctx = pd.read_csv(p("results", "tables", "context_baselines.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    vs = list("ABCDEF")
    y = np.arange(len(vs))[::-1]
    for k, (tab, mk, off, name) in enumerate([(t, "o", 0.14, "L-TAE (primary)"), (g, "s", -0.14, "grid-XGBoost (secondary)")]):
        tt = tab.set_index("variant").loc[vs]
        for yi, v in zip(y, vs):
            ax.errorbar(tt.loc[v, "MAE_mean"], yi + off, xerr=tt.loc[v, "MAE_sd"], fmt=mk, ms=7,
                        color=COL[v], mfc=COL[v] if k == 0 else "white", mew=1.6, capsize=2.5, lw=1.2)
            ax.text(tt.loc[v, "MAE_mean"] + tt.loc[v, "MAE_sd"] + 0.15, yi + off, f"{tt.loc[v, 'MAE_mean']:.1f}",
                    va="center", fontsize=7.5, color=INK2)
    for val, lab, ls in [(ctx.loc[ctx.baseline.str.startswith("season"), "MAE"].iloc[0], "season-median calendar baseline", "--"),
                         (r.loc[0, "MAE_all"], "no-training rule R1", ":")]:
        ax.axvline(val, color=INK2, ls=ls, lw=1)
        ax.text(val, len(vs) - 0.35, f" {lab}\n {val:.1f} d", fontsize=7.5, color=INK2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[v] for v in vs])
    ax.set_xlabel("Harvest-date MAE (days), 5-fold plot-grouped CV, mean ± SD over 5 seeds")
    ax.set_ylim(-0.7, len(vs) + 0.5)
    ax.plot([], [], "o", color=INK2, label="L-TAE (primary, filled)")
    ax.plot([], [], "s", color=INK2, mfc="white", label="grid-XGBoost (secondary, open)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("Harvest-date error by input variant (n = 319 plot-seasons, 131 plots)", loc="left", fontsize=10, color=INK)
    save(fig, "fig3_variant_comparison")


def fig_error_distribution():
    d = pd.read_csv(os.path.join(PRED, "harvest_ltae.csv"))
    ae = sample_abs_err(d)
    vs = list("ABCDE")
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    for v in vs:
        x = np.sort(ae[ae.variant == v].ae.values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COL[v], lw=2 if v in "BC" else 1.2, label=LABEL[v])
    ax.set_xlabel("absolute harvest-date error (days, seed-averaged)")
    ax.set_ylabel("cumulative share of samples")
    ax.legend(fontsize=7)
    ax.set_title("Error distribution (L-TAE)", loc="left", fontsize=10)
    ax = axs[1]
    a = ae[ae.variant == "B"].set_index("UNIQUE_ID")
    c = ae[ae.variant == "C"].set_index("UNIQUE_ID")
    diff = (a.ae - c.ae).groupby(a.PLOT_ID).mean()
    ax.hist(diff, bins=np.arange(-12, 12.5, 1), color=COL["C"], edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(diff.mean(), color=INK2, ls="--", lw=1)
    ax.set_xlabel("|err B| − |err C| per plot (days)\n> 0 means bare-soil prob. helped")
    ax.set_ylabel("plots")
    ax.set_title("Paired C vs B (131 plots)", loc="left", fontsize=10)
    save(fig, "fig4_error_distribution")


def fig_actual_vs_pred():
    d = pd.read_csv(os.path.join(PRED, "harvest_ltae.csv"))
    ens = d.groupby(["variant", "UNIQUE_ID"]).agg(y=("y_true", "first"), yp=("y_pred", "mean"), fold=("fold", "first")).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True, sharey=True)
    for ax, v in zip(axs, "BC"):
        e = ens[ens.variant == v]
        off = e[e.fold != 0]
        on = e[e.fold == 0]
        ax.scatter(off.y, off.yp, s=12, color=COL[v], alpha=0.6, lw=0, label="CV folds 1–4")
        ax.scatter(on.y, on.yp, s=16, facecolor="white", edgecolor=COL[v], lw=1.2, label="fold 0 = official val")
        lim = [95, 190]
        ax.plot(lim, lim, color=INK2, lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        mae = np.abs(e.yp - e.y).mean()
        ax.set_title(f"{LABEL[v]}   (seed-ensemble MAE {mae:.1f} d)", loc="left", fontsize=9)
        ax.set_xlabel("actual harvest day (days since season start)")
    axs[0].set_ylabel("predicted harvest day")
    axs[0].legend(fontsize=7, loc="upper left")
    save(fig, "fig5_actual_vs_predicted")


def fig_event_aligned():
    """Mean plot-level signals aligned on the true harvest date (descriptive; all 319 samples)."""
    st, ps = D.load_tables()
    m = ps.merge(st[["UNIQUE_ID", "HARVESTING_DAY"]], on="UNIQUE_ID")
    m["rel"] = m.day - m.HARVESTING_DAY
    sh = m.copy()
    sh["BS_PROB"] = D.shuffled_prob(ps, 0)
    bins = np.arange(-120, 41, 10)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
    ax = axs[0]
    for col, lab, c in [("NDVI", "NDVI", COL["A"]), ("BSI", "BSI", COL["B"]), ("BS_PROB", "bare-soil prob. (C)", COL["C"]),
                        ("SCL_BARE", "SCL bare fraction (F)", COL["F"])]:
        g = m.assign(b=pd.cut(m.rel, bins, labels=bins[:-1] + 5)).groupby("b", observed=True)[col]
        mu, se, n = g.mean(), g.std() / np.sqrt(g.count()), g.count()
        x = mu.index.astype(float)
        ax.plot(x, mu, color=c, lw=2, marker="o", ms=3, label=lab)
        ax.fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.15, lw=0)
    ax.axvline(0, color=INK, lw=1)
    ax.text(1, ax.get_ylim()[1] * 0.95, "harvest", fontsize=7.5, color=INK)
    ax.set_xlabel("days relative to true harvest date")
    ax.set_ylabel("plot mean (cloud-free dates)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Signals aligned on harvest (319 samples)", loc="left", fontsize=10)
    ax = axs[1]
    for frame, lab, c, ls in [(m, "C: real timing", COL["C"], "-"), (sh, "E: shuffled across dates", COL["E"], "--")]:
        g = frame.assign(b=pd.cut(frame.rel, bins, labels=bins[:-1] + 5)).groupby("b", observed=True).BS_PROB
        mu, se = g.mean(), g.std() / np.sqrt(g.count())
        x = mu.index.astype(float)
        ax.plot(x, mu, color=c, lw=2, ls=ls, label=lab)
        ax.fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.15, lw=0)
    ax.axvline(0, color=INK, lw=1)
    ax.axhline(BS_THRESHOLD, color=INK2, lw=0.8, ls=":")
    ax.text(38, BS_THRESHOLD + 0.01, "p = 0.5 threshold", fontsize=7, color=INK2, ha="right")
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("days relative to true harvest date")
    ax.set_ylabel("mean bare-soil probability")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.8))
    ax.set_title("Why E differs from C", loc="left", fontsize=10)
    save(fig, "fig6_event_aligned_signals")


def fig_example_sequences():
    st, ps = D.load_tables()
    rng = np.random.default_rng(3)
    # 6 samples with >= 3 cloud-free dates after harvest (chosen by a fixed random draw)
    cand = [u for u in st.UNIQUE_ID if ((ps.UNIQUE_ID == u) & (ps.day > st.set_index("UNIQUE_ID").loc[u, "HARVESTING_DAY"])).sum() >= 3]
    pick = sorted(rng.choice(cand, 6, replace=False))
    fig, axs = plt.subplots(2, 3, figsize=(8.2, 4.6), sharex=True, sharey=True)
    for ax, u in zip(axs.ravel(), pick):
        b = ps[ps.UNIQUE_ID == u]
        r = st.set_index("UNIQUE_ID").loc[u]
        ax.plot(b.day, b.NDVI, color=COL["A"], lw=1.5, marker="o", ms=3.5, label="NDVI")
        ax.plot(b.day, b.BS_PROB, color=COL["C"], lw=1.5, marker="s", ms=3.5, label="bare-soil prob.")
        ax.axvline(r.HARVESTING_DAY, color=INK, lw=1.2)
        ax.axhline(BS_THRESHOLD, color=INK2, lw=0.6, ls=":")
        ax.set_title(f"uid {u}  plot {int(r.PLOT_ID)}  {r.STANDARD_SEASON} {int(r.YEAR)}", fontsize=8, loc="left")
        ax.set_ylim(-0.1, 1.0)
    axs[0, 0].legend(fontsize=7, loc="upper left")
    for ax in axs[1]:
        ax.set_xlabel("day of season")
    for ax in axs[:, 0]:
        ax.set_ylabel("value")
    fig.suptitle("Example plot-level series (cloud-free dates only); black line = recorded harvest date", x=0.01, ha="left", fontsize=10)
    save(fig, "fig2_example_sequences")


def fig_example_plot(uid=None):
    model = pickle.load(open(p("data_processed", "universal_bare_soil_xgboost.pkl"), "rb"))
    st, ps = D.load_tables()
    m = ps.merge(st[["UNIQUE_ID", "HARVESTING_DAY", "n_plot_px"]])
    if uid is None:
        # deterministic choice: larger plot (>= 30 px) with >= 2 cloud-free post-harvest dates; lowest uid
        c = m[(m.n_plot_px >= 30)].assign(post=lambda x: x.day > x.HARVESTING_DAY).groupby("UNIQUE_ID").post.sum()
        uid = int(c[c >= 2].index[0])
    r = st.set_index("UNIQUE_ID").loc[uid]
    d = np.load(p("data_processed", "s2_32", f"{uid}.npz"))
    x = d["bands"].astype(np.float32)
    b = ps[ps.UNIQUE_ID == uid]
    H = r.HARVESTING_DAY
    peak = b.day.values[np.argmax(b.NDVI.values)]
    pre = b[(b.day < H) & (b.day > peak)].day.values
    post = b[b.day > H].day.values
    days = [peak, pre[len(pre) // 2] if len(pre) else peak, pre[-1] if len(pre) else peak, post[0]]
    own = d["mask"][0] == r.PLOT_ID
    fig, axs = plt.subplots(4, 4, figsize=(7.6, 7.9))
    for j, dy in enumerate(days):
        k = int(np.where(d["dates"] == dy)[0][0])
        img = x[k]
        rgb = np.stack([img[S2_BANDS.index(c)] for c in ["B4", "B3", "B2"]], -1) / 2500.0
        nd = ndvi(img[S2_BANDS.index("B4")], img[S2_BANDS.index("B8")])
        X = img.transpose(1, 2, 0).reshape(-1, 12)[:, [S2_BANDS.index(f) for f in BS_FEATURES]]
        pr = model.predict_proba(pd.DataFrame(X, columns=BS_FEATURES))[:, 1].reshape(32, 32)
        layers = [(np.clip(rgb, 0, 1), None, "RGB (B4,B3,B2)"), (nd, "Greens", "NDVI"),
                  (pr, "YlOrBr", "bare-soil probability"), ((pr >= BS_THRESHOLD).astype(float), "YlOrBr", "bare mask (p ≥ 0.5)")]
        for i, (a, cmap, name) in enumerate(layers):
            ax = axs[i, j]
            im = ax.imshow(a) if cmap is None else ax.imshow(a, cmap=cmap, vmin=0, vmax=1)
            ax.contour(own, levels=[0.5], colors=["#2a78d6"], linewidths=1.4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                rel = int(dy - H)
                tag = ["NDVI peak", "ripening", "last clear before harvest", "first clear after harvest"][j]
                ax.set_title(f"day {int(dy)} ({rel:+d} d)\n{tag}", fontsize=8)
            if j == 0:
                ax.set_ylabel(name, fontsize=8.5)
            if j == 3 and cmap is not None:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=7)
                if i == 3:
                    cb.set_ticks([0, 1])
                    cb.set_ticklabels(["not bare", "bare"])
    fig.suptitle(f"SICKLE sample {uid} (plot {int(r.PLOT_ID)}, {r.STANDARD_SEASON} {int(r.YEAR)}), recorded harvest day {int(H)}. "
                 f"Blue outline = the plot; 320 m x 320 m patch", x=0.01, ha="left", fontsize=8.5)
    save(fig, "fig1_example_plot_maps")
    return uid


def fig_reduced():
    t = pd.read_csv(p("results", "tables", "reduced_data_B_vs_C.csv")).sort_values("train_fraction")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axs[0]
    for v in "BC":
        ax.errorbar(t.train_fraction * 100, t[f"MAE_{v}_mean"], yerr=t[f"MAE_{v}_sd"], color=COL[v], marker="o",
                    lw=2, capsize=2.5, label=LABEL[v])
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_xlabel("share of training plots used")
    ax.set_ylabel("harvest MAE (days)")
    ax.legend(fontsize=7)
    ax = axs[1]
    ax.errorbar(t.train_fraction * 100, t.improvement_C_over_B,
                yerr=[t.improvement_C_over_B - t.ci95_lo, t.ci95_hi - t.improvement_C_over_B],
                color=COL["C"], marker="o", lw=2, capsize=2.5)
    ax.axhline(0, color=INK, lw=1)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.text(10, 1.05, "pre-specified 1-day target", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_xlabel("share of training plots used")
    ax.set_ylabel("MAE(B) − MAE(C), days (95% CI)")
    fig.subplots_adjust(wspace=0.4)
    fig.suptitle("Reduced training data: does bare-soil information help more when data are scarce?", x=0.01, ha="left", fontsize=10)
    save(fig, "fig7_reduced_data")


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    fns = {"variants": fig_variant_comparison, "errors": fig_error_distribution, "scatter": fig_actual_vs_pred,
           "aligned": fig_event_aligned, "sequences": fig_example_sequences, "maps": fig_example_plot,
           "reduced": fig_reduced}
    for k, f in fns.items():
        if "all" in which or k in which:
            try:
                f()
                print("ok", k)
            except FileNotFoundError as e:
                print("skip", k, e)
