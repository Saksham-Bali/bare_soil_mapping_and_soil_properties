"""
train_soil_classifier.py
-------------------------
Trains multiple multiclass classifiers to predict soil type (s1/s2/s3/s4)
from Sentinel-2 bare-soil pixels.

Strategy:
  - 16 features: 11 raw bands + 5 spectral indices
  - 80/20 stratified train/test split
  - SMOTE applied to training set only (fixes severe s3 imbalance)
  - Models: XGBoost, RandomForest, ExtraTrees, (+LightGBM if installed)
  - Primary metric: macro F1 (best for imbalanced multiclass)
  - OOB score used for RF/ET stability (free, no retraining needed)
  - Best model saved as .joblib

Usage:
    python train_soil_classifier.py
"""

import os, time, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble          import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing     import LabelEncoder
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       f1_score, accuracy_score, ConfusionMatrixDisplay)
from xgboost                   import XGBClassifier
from imblearn.over_sampling    import SMOTE

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PARQUET_PATH = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\bare_soil_pixels.parquet'
OUTPUT_DIR   = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\soil_type_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Feature definitions ───────────────────────────────────────────────────────
BAND_COLS  = ['B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
INDEX_COLS = ['NDVI','BSI','NDWI','SWIR_ratio','brightness']
FEAT_COLS  = BAND_COLS + INDEX_COLS   # 16 features
TARGET_COL = 'soil_type'

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 65)
print("SOIL TYPE CLASSIFIER — TRAINING PIPELINE")
print("=" * 65)

df = pd.read_parquet(PARQUET_PATH)
print("\nLoaded: %d pixels  |  columns: %s" % (len(df), list(df.columns)))

# Drop non-feature columns
drop_cols = ['.geo', 'district', 'bare_soil_prob', 'bare_soil_pred']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── 2. Compute spectral indices ────────────────────────────────────────────────
eps = 1e-8
df['NDVI']       = (df['B8'] - df['B4']) / (df['B8']  + df['B4']  + eps)
df['BSI']        = ((df['B11']+df['B4'])-(df['B8']+df['B2'])) / \
                   ((df['B11']+df['B4'])+(df['B8']+df['B2'])+eps)
df['NDWI']       = (df['B3'] - df['B8']) / (df['B3']  + df['B8']  + eps)
df['SWIR_ratio'] = df['B11'] / (df['B12'] + eps)
df['brightness'] = (df['B2'] + df['B3'] + df['B4'] + df['B8']) / 4.0

print("\nClass distribution (raw):")
vc = df[TARGET_COL].value_counts().sort_index()
for cls, cnt in vc.items():
    print("  %-4s : %6d  (%.1f%%)" % (cls, cnt, 100*cnt/len(df)))

# ── 3. Encode labels ──────────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(df[TARGET_COL])          # s1→0, s2→1, s3→2, s4→3
X  = df[FEAT_COLS].values.astype(np.float32)
CLASS_NAMES = list(le.classes_)
print("\nLabel encoding: %s" % dict(zip(le.classes_, le.transform(le.classes_))))

# ── 4. Stratified 80/20 split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain: %d  |  Test: %d" % (len(X_train), len(X_test)))

# ── 5. SMOTE on training set only ────────────────────────────────────────────
print("\nApplying SMOTE to training set...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE: %d samples" % len(X_train_res))
print("Resampled class counts:", dict(zip(*np.unique(y_train_res, return_counts=True))))

# ── 6. Define models ──────────────────────────────────────────────────────────
n_classes = len(CLASS_NAMES)

# XGBoost: uses sample weights via compute_sample_weight
from sklearn.utils.class_weight import compute_sample_weight

models = {}

models['XGBoost'] = XGBClassifier(
    n_estimators      = 300,
    max_depth         = 7,
    learning_rate     = 0.08,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    use_label_encoder = False,
    eval_metric       = 'mlogloss',
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)

models['RandomForest'] = RandomForestClassifier(
    n_estimators = 300,
    max_depth    = None,
    class_weight = 'balanced',
    oob_score    = True,    # free stability estimate during training
    random_state = 42,
    n_jobs       = -1,
)

models['ExtraTrees'] = ExtraTreesClassifier(
    n_estimators = 300,
    max_depth    = None,
    class_weight = 'balanced',
    bootstrap    = True,    # needed for OOB
    oob_score    = True,
    random_state = 42,
    n_jobs       = -1,
)

# Add LightGBM if installed
try:
    from lightgbm import LGBMClassifier
    models['LightGBM'] = LGBMClassifier(
        n_estimators  = 300,
        max_depth     = 7,
        learning_rate = 0.08,
        class_weight  = 'balanced',
        random_state  = 42,
        n_jobs        = -1,
        verbose       = -1,
    )
    print("\nLightGBM detected -- will be included.")
except ImportError:
    print("\nLightGBM not installed — skipping.")

# ── 7. Train & evaluate each model ───────────────────────────────────────────
results = {}

print("\n" + "=" * 65)
print("TRAINING & EVALUATION")
print("=" * 65)

for name, model in models.items():
    print("\n--- %s ---" % name)
    t0 = time.time()

    # Fit on SMOTE-resampled training data
    if name == 'XGBoost':
        sw = compute_sample_weight('balanced', y_train_res)
        model.fit(X_train_res, y_train_res, sample_weight=sw)
    else:
        model.fit(X_train_res, y_train_res)

    elapsed = time.time() - t0
    print("  Train time: %.1fs" % elapsed)

    # Predict on HELD-OUT test set (never seen, no SMOTE applied)
    y_pred = model.predict(X_test)

    macro_f1    = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    oa          = accuracy_score(y_test, y_pred)

    print("  Macro F1    : %.4f" % macro_f1)
    print("  Weighted F1 : %.4f" % weighted_f1)
    print("  Overall Acc : %.4f" % oa)
    print("\n  Per-class report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    # OOB score (RF / ET only) -- computed for free during training
    oob = getattr(model, 'oob_score_', None)
    oob_str = ("%.4f" % oob) if oob is not None else "N/A"
    print("  OOB Accuracy: %s" % oob_str)

    results[name] = {
        'model':       model,
        'macro_f1':    macro_f1,
        'weighted_f1': weighted_f1,
        'oa':          oa,
        'oob':         oob,
        'y_pred':      y_pred,
    }

# ── 8. Pick best model (macro F1 on test set) ────────────────────────────────
best_name = max(results, key=lambda k: results[k]['macro_f1'])
best      = results[best_name]

print("\n" + "=" * 65)
print("BEST MODEL: %s  (test macro F1 = %.4f)" % (best_name, best['macro_f1']))
print("=" * 65)

# ── 9. Summary table ────────────────────────────────────────────────────────
print("\nModel Comparison:")
print("%-16s  %10s  %12s  %10s  %10s" % (
    "Model", "Macro F1", "Weighted F1", "Ovr Acc", "OOB Acc"))
print("-" * 62)
for name, res in sorted(results.items(), key=lambda x: -x[1]['macro_f1']):
    marker = " <-- BEST" if name == best_name else ""
    oob_str = ("%.4f" % res['oob']) if res['oob'] is not None else "  N/A  "
    print("%-16s  %10.4f  %12.4f  %10.4f  %10s%s" % (
        name,
        res['macro_f1'],
        res['weighted_f1'],
        res['oa'],
        oob_str,
        marker,
    ))

# ── 11. Confusion matrix plot for best model ─────────────────────────────────
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_%s.png' % best_name)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, best['y_pred']),
    display_labels=CLASS_NAMES
)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('%s — Confusion Matrix (Test Set)' % best_name)
plt.tight_layout()
plt.savefig(cm_path, dpi=150)
plt.close()
print("\nConfusion matrix saved: %s" % cm_path)

# Feature importance for best model (if available)
if hasattr(best['model'], 'feature_importances_'):
    fi = best['model'].feature_importances_
    fi_df = pd.DataFrame({'feature': FEAT_COLS, 'importance': fi})
    fi_df = fi_df.sort_values('importance', ascending=False)
    print("\nTop feature importances:")
    print(fi_df.to_string(index=False))

    fi_path = os.path.join(OUTPUT_DIR, 'feature_importance_%s.png' % best_name)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=fi_df, x='importance', y='feature', ax=ax, palette='viridis')
    ax.set_title('%s — Feature Importances' % best_name)
    plt.tight_layout()
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print("Feature importance plot saved: %s" % fi_path)

# ── 12. Save best model + label encoder ──────────────────────────────────────
bundle = {
    'model':        best['model'],
    'label_encoder': le,
    'feature_cols': FEAT_COLS,
    'band_cols':    BAND_COLS,
    'index_cols':   INDEX_COLS,
    'class_names':  CLASS_NAMES,
    'metrics': {
        'macro_f1':    best['macro_f1'],
        'weighted_f1': best['weighted_f1'],
        'oa':          best['oa'],
    }
}

save_path = os.path.join(OUTPUT_DIR, 'soil_type_%s.joblib' % best_name.lower())
joblib.dump(bundle, save_path, compress=3)
size_kb = os.path.getsize(save_path) / 1024
print("\nModel bundle saved: %s  (%.1f KB)" % (save_path, size_kb))
print("\nBundle keys: model, label_encoder, feature_cols, band_cols, index_cols, class_names, metrics")
print("\nDONE.")
