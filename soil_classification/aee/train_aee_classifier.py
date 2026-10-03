"""
train_aee_classifier.py
---------------------------
Trains the 3-class soil texture classifier using Alpha Earth Embeddings (AEE) + Sentinel-2.
Uses StratifiedGroupKFold across 33 polygons.

Performs a two-phase comparison:
  Phase A: Full 80 features (64 AEE + 16 S2)
  Phase B: Reduced features (Top 25 selected via feature importance)
"""

import os, time, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\aee\ae_bare_soil_pixels.parquet'
OUTPUT_DIR   = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\aee'

AEE_COLS   = [f'A{str(i).zfill(2)}' for i in range(64)]
BAND_COLS  = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
INDEX_COLS = ['NDVI','BSI','NDWI','SWIR_ratio','brightness','NDSI']
FEAT_COLS  = AEE_COLS + BAND_COLS + INDEX_COLS

TARGET_COL = 'soil_type'
GROUP_COL  = 'polygon_id'
RANDOM_STATE = 42
N_SPLITS = 5
TOP_K = 25  # Number of features for Phase B reduction

print("=" * 65)
print("AEE + S2 CLASSIFIER — TRAINING PIPELINE")
print("=" * 65)

# ── 1. Load & Data Prep ────────────────────────────────────────────────────────
print("Loading AEE dataset...")
df = pd.read_parquet(PARQUET_PATH)

drop_cols = ['.geo', 'bare_soil_prob', 'bare_soil_pred', 'B9']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

print("Engineering spectral indices...")
eps = 1e-8
df['NDVI']       = (df['B8'] - df['B4']) / (df['B8']  + df['B4']  + eps)
df['BSI']        = ((df['B11']+df['B4'])-(df['B8']+df['B2'])) / ((df['B11']+df['B4'])+(df['B8']+df['B2'])+eps)
df['NDWI']       = (df['B3'] - df['B8']) / (df['B3']  + df['B8']  + eps)
df['SWIR_ratio'] = df['B11'] / (df['B12'] + eps)
df['brightness'] = (df['B2'] + df['B3'] + df['B4'] + df['B8']) / 4.0
df['NDSI']       = (df['B11'] - df['B12']) / (df['B11'] + df['B12'] + eps)

print("\nEncoding labels...")
le = LabelEncoder()
y  = le.fit_transform(df[TARGET_COL])
X_full  = df[FEAT_COLS].values.astype(np.float32)
groups = df[GROUP_COL].values
CLASS_NAMES = list(le.classes_)

# ── Function: Run CV ───────────────────────────────────────────────────────────
def run_cv(X, feature_names):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    lgbm_mac = []
    xgb_mac = []
    
    oof_preds_lgbm = np.zeros(len(y), dtype=int)
    oof_preds_xgb = np.zeros(len(y), dtype=int)
    
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # LightGBM
        lgbm = LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                              class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        lgbm.fit(X_train, y_train)
        p_lgbm = lgbm.predict(X_test)
        lgbm_mac.append(f1_score(y_test, p_lgbm, average='macro'))
        oof_preds_lgbm[test_idx] = p_lgbm
        
        # XGBoost
        sw = compute_sample_weight('balanced', y_train)
        xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                            objective='multi:softmax', num_class=len(CLASS_NAMES),
                            random_state=RANDOM_STATE, n_jobs=-1)
        xgb.fit(X_train, y_train, sample_weight=sw)
        p_xgb = xgb.predict(X_test)
        xgb_mac.append(f1_score(y_test, p_xgb, average='macro'))
        oof_preds_xgb[test_idx] = p_xgb
        
    lgbm_mean = np.mean(lgbm_mac)
    xgb_mean = np.mean(xgb_mac)
    
    if lgbm_mean >= xgb_mean:
        return 'LightGBM', lgbm_mean, np.std(lgbm_mac), oof_preds_lgbm
    else:
        return 'XGBoost', xgb_mean, np.std(xgb_mac), oof_preds_xgb

# ── 2. Phase A: Full 80 Features ───────────────────────────────────────────────
print("\n--- PHASE A: Full 80 Features ---")
winner_full, mean_full, std_full, oof_full = run_cv(X_full, FEAT_COLS)
print(f"Winner: {winner_full} | Macro F1: {mean_full:.4f} ± {std_full:.4f}")

# ── 3. Phase B: Reduced Features ───────────────────────────────────────────────
print(f"\n--- PHASE B: Reduced Top {TOP_K} Features ---")
# Extract feature importance using a fast LightGBM on all data
lgbm_fs = LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                         class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
lgbm_fs.fit(X_full, y)
importances = lgbm_fs.feature_importances_

# Sort and keep top K
indices = np.argsort(importances)[::-1][:TOP_K]
reduced_features = [FEAT_COLS[i] for i in indices]
X_reduced = X_full[:, indices]

print(f"Top {TOP_K} selected features: {reduced_features[:10]}...")

winner_red, mean_red, std_red, oof_red = run_cv(X_reduced, reduced_features)
print(f"Winner: {winner_red} | Macro F1: {mean_red:.4f} ± {std_red:.4f}")

# ── 4. Final Comparison & Selection ────────────────────────────────────────────
print("\n" + "="*50)
print("FINAL PIPELINE COMPARISON")
print("="*50)
print(f"S2-Only Baseline (from v2) : 0.3408")
print(f"AEE+S2 (Full 80 Feats)     : {mean_full:.4f} ± {std_full:.4f} ({winner_full})")
print(f"AEE+S2 (Top {TOP_K} Feats)      : {mean_red:.4f} ± {std_red:.4f} ({winner_red})")

if mean_full >= mean_red:
    best_strategy = "Full 80 Features"
    best_model_name = winner_full
    best_X = X_full
    best_feats = FEAT_COLS
    best_oof = oof_full
    best_mean = mean_full
else:
    best_strategy = f"Top {TOP_K} Features"
    best_model_name = winner_red
    best_X = X_reduced
    best_feats = reduced_features
    best_oof = oof_red
    best_mean = mean_red

print(f"\nTraining final '{best_strategy}' model ({best_model_name}) on ALL data...")

if best_model_name == 'LightGBM':
    final_model = LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    final_model.fit(best_X, y)
else:
    sw = compute_sample_weight('balanced', y)
    final_model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, objective='multi:softmax', num_class=len(CLASS_NAMES), random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(best_X, y, sample_weight=sw)

# ── 5. Diagnostics & Save ──────────────────────────────────────────────────────
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_aee.png')
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix(y, best_oof), display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f'AEE+S2 ({best_strategy}) - OOF Confusion Matrix')
plt.tight_layout()
plt.savefig(cm_path, dpi=150)
plt.close()

fi_df = pd.DataFrame({'feature': best_feats, 'importance': final_model.feature_importances_}).sort_values('importance', ascending=False)
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance_aee.png')
fig, ax = plt.subplots(figsize=(9, 7))
sns.barplot(data=fi_df.head(30), x='importance', y='feature', ax=ax, palette='viridis')
ax.set_title(f'AEE+S2 ({best_strategy}) - Top Feature Importances')
plt.tight_layout()
plt.savefig(fi_path, dpi=150)
plt.close()

bundle = {
    'model': final_model,
    'label_encoder': le,
    'feature_cols': best_feats,
    'class_names': CLASS_NAMES,
    'strategy': best_strategy,
    'metrics': {
        'cv_macro_f1_mean': float(best_mean),
        'cv_macro_f1_std': float(np.std(best_mean)),
    }
}
model_path = os.path.join(OUTPUT_DIR, 'soil_type_aee_classifier.joblib')
joblib.dump(bundle, model_path, compress=3)

print("DONE.")
