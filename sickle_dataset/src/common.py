"""Shared constants and helpers. Everything here mirrors the official SICKLE code
(github.com/Depanshu-Sani/SICKLE, utils/dataset.py) unless explicitly noted."""
import json
import os
from datetime import date

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(PROJECT_ROOT, "config", "config.json")))

S2_BANDS = CFG["sickle"]["s2_bands"]                      # official band order
BS_FEATURES = CFG["bare_soil_model"]["feature_order"]     # order stored inside the .pkl
IMG = CFG["sickle"]["img_size"]
INVALID_SCL = np.array(CFG["cloud_mask"]["invalid_scl_classes"])
BS_THRESHOLD = CFG["bare_soil_model"]["binary_threshold"]
MIN_CLEAR = CFG["cloud_mask"]["plot_date_usable_min_clear_fraction"]

MON_TO_INT = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
              "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def p(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def season_start(standard_season, year):
    """Official: start_date = date(YEAR, month of first season month, 1)."""
    return date(int(year), MON_TO_INT[standard_season.split("-")[0]], 1)


def s2_file_date(basename):
    """Official S2 date parsing (utils/dataset.py::_get_data_index)."""
    if basename[0] == "T":
        s = basename.split("_")[1][:8]
    else:
        s = basename.split("_")[0][:8]
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


def day_index(d, start):
    """Official SICKLE_Dataset index = (date - start).days + 1.
    The CSV HARVESTING_DAY uses the same convention (verified in the audit), so
    observation days and harvest labels are on one consistent scale."""
    return (d - start).days + 1


def nn_resize(a, H=IMG, W=IMG):
    """Nearest-neighbour resize identical to albumentations.Resize(..., cv2.INTER_NEAREST)
    used by the official SICKLE_Dataset (verified bit-exact against cv2 in the audit)."""
    h, w = a.shape[-2:]
    ri = np.minimum(np.floor(np.arange(H) * (h / H)).astype(int), h - 1)
    ci = np.minimum(np.floor(np.arange(W) * (w / W)).astype(int), w - 1)
    return a[..., ri, :][..., ci]


def ndvi(b4, b8):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (b8 - b4) / (b8 + b4)


def bsi(b2, b4, b8, b11):
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2))
