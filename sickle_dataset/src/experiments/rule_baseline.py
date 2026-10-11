"""No-training rule baseline (pre-registered, nothing fitted on SICKLE).

R1: on usable (cloud-free) dates, find the day of maximum plot NDVI; predicted harvest = first usable
    date strictly after that peak with plot-mean bare-soil probability >= 0.5 (model default threshold).
    If never triggered: predict 183 (end of the official 184-day window).
R2: as R1 but predict the midpoint between the detection date and the previous usable date
    (harvest happened somewhere in that interval).
Also reported (exploratory): R1 using the Sen2Cor SCL bare-soil fraction >= 0.5 instead of our model.

Output: results/predictions/rule_baseline.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import BS_THRESHOLD, p  # noqa
from src.experiments import data as D  # noqa

FALLBACK = 183


def rule(days, ndvi, signal, thr=BS_THRESHOLD):
    k_peak = int(np.nanargmax(ndvi))
    for k in range(k_peak + 1, len(days)):
        if signal[k] >= thr:
            return days[k], (days[k] + days[k - 1]) / 2.0, True, days[k_peak]
    return FALLBACK, FALLBACK, False, days[k_peak]


def main():
    st, ps = D.load_tables()
    folds = D.make_folds(st)
    g = ps.groupby("UNIQUE_ID")
    rows = []
    for i, r in st.iterrows():
        b = g.get_group(r.UNIQUE_ID)
        d, nd = b.day.values.astype(float), b.NDVI.values
        r1, r2, trig, peak = rule(d, nd, b.BS_PROB.values)
        s1, s2, strig, _ = rule(d, nd, b.SCL_BARE.values)
        rows.append({"UNIQUE_ID": r.UNIQUE_ID, "PLOT_ID": r.PLOT_ID, "SPLIT": r.SPLIT, "fold": folds[i],
                     "STANDARD_SEASON": r.STANDARD_SEASON, "YEAR": r.YEAR,
                     "y_true": r.HARVESTING_DAY, "ndvi_peak_day": peak,
                     "R1_pred": r1, "R2_pred": r2, "triggered": trig,
                     "R1_SCL_pred": s1, "R2_SCL_pred": s2, "SCL_triggered": strig,
                     "last_usable_day": d[-1]})
    out = pd.DataFrame(rows)
    out.to_csv(p("results", "predictions", "rule_baseline.csv"), index=False)
    for c in ["R1_pred", "R2_pred", "R1_SCL_pred"]:
        e = out[c] - out.y_true
        print(c, "MAE %.2f  bias %.2f" % (e.abs().mean(), e.mean()))
    print("trigger rate", out.triggered.mean().round(3), "SCL trigger", out.SCL_triggered.mean().round(3))


if __name__ == "__main__":
    main()
