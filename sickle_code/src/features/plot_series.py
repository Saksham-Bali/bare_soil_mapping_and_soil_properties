"""Step 2: collect per-date plot-level aggregates (written by src/sickle/preprocess.py) into
one long table:  data_processed/plot_series.csv  (one row per sample x observation date)

Columns: UNIQUE_ID, PLOT_ID, SPLIT, day (official index), date, 12 band means, NDVI, BSI, BS_PROB,
BS_BIN, SCL_BARE, BS_PROB_ALLPX, clear_frac, n_px, usable (clear_frac >= 0.5).
All means are over the sample's own plot pixels that are cloud-free according to SCL.
Also writes data_processed/sample_table.csv (one row per sample with labels + QC counts).
"""
import glob
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import CFG, MIN_CLEAR, S2_BANDS, p, season_start  # noqa

OUT = p(CFG["paths"]["processed_dir"])
COLS = S2_BANDS + ["NDVI", "BSI", "BS_PROB", "BS_BIN", "SCL_BARE", "BS_PROB_ALLPX", "clear_frac", "n_px"]


def main():
    df = pd.read_csv(os.path.join(OUT, "sickle_dataset_tabular.csv"))
    rows, samples = [], []
    for f in sorted(glob.glob(os.path.join(OUT, "plot_stats", "*.npz"))):
        uid = int(os.path.basename(f)[:-4])
        r = df[df.UNIQUE_ID == uid].iloc[0]
        d = np.load(f)
        log = json.load(open(os.path.join(OUT, "logs", f"{uid}.json")))
        start = season_start(r.STANDARD_SEASON, r.YEAR)
        s = {"UNIQUE_ID": uid, "PLOT_ID": int(r.PLOT_ID), "SPLIT": r.SPLIT, "CROP": r.CROP,
             "STANDARD_SEASON": r.STANDARD_SEASON, "YEAR": int(r.YEAR),
             "HARVESTING_DAY": r.HARVESTING_DAY, "SOWING_DAY": r.SOWING_DAY, "YIELD": r.YIELD,
             "n_dates_official": log["n_dates"], "n_plot_px": log.get("n_plot_px", 0)}
        if "NDVI" not in d.files:
            s["n_usable"] = 0
            samples.append(s)
            continue
        s["mask_harvest_day"] = float(d["mask_harvest_day"][0])
        s["mask_crop"] = float(d["mask_crop"][0])
        s["mask_yield_sum"] = float(d["mask_yield_sum"][0])
        T = len(d["dates"])
        block = pd.DataFrame({c: d[c] for c in COLS})
        block.insert(0, "day", d["dates"].astype(int))
        block.insert(0, "UNIQUE_ID", uid)
        block["date"] = [str(start + timedelta(days=int(x) - 1)) for x in d["dates"]]
        block["usable"] = (block.clear_frac >= MIN_CLEAR) & block[S2_BANDS].notna().all(axis=1)
        s["n_usable"] = int(block.usable.sum())
        rows.append(block)
        samples.append(s)
    ps = pd.concat(rows, ignore_index=True)
    ps.to_csv(os.path.join(OUT, "plot_series.csv"), index=False, float_format="%.5g")
    st = pd.DataFrame(samples)
    st.to_csv(os.path.join(OUT, "sample_table.csv"), index=False)
    lab = st[st.YIELD > 0]
    print("rows", len(ps), "samples", len(st), "harvest samples", len(lab))
    print("CSV vs mask harvest day mismatches:", int((lab.HARVESTING_DAY != lab.mask_harvest_day).sum()))
    print("usable dates per harvest sample:", lab.n_usable.describe().round(1).to_dict())


if __name__ == "__main__":
    main()
