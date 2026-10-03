"""
train_soil_classifier_v2.py
---------------------------
Trains the final 3-class soil texture classifier (s1_loam, s2_loam_clay, s3_sandy_loam).
Uses a strict StratifiedGroupKFold validation strategy to ensure no data leakage
between polygons (physical fields).

Compares LightGBM and XGBoost, and saves the champion model.
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\bare_soil_pixels.parquet'
OUTPUT_DIR   = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification'

BAND_COLS  = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
INDEX_COLS = ['NDVI','BSI','NDWI','SWIR_ratio','brightness', 'NDSI']
FEAT_COLS  = BAND_COLS + INDEX_COLS
TARGET_COL = 'soil_type'
GROUP_COL  = 'polygon_id'
RANDOM_STATE = 42
N_SPLITS = 5

print("=" * 65)
print("SOIL TYPE CLASSIFIER V2 — TRAINING PIPELINE")
print("=" * 65)

# ── 1. Load & Data Prep ────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(PARQUET_PATH)

# Check for polygon_id
if GROUP_COL not in df.columns:
    print(f"WARNING: '{GROUP_COL}' missing. Reconstructing from district + soil_type as fallback.")
    df[GROUP_COL] = df['district'] + "_" + df['soil_type']
else:
    print(f"Found '{GROUP_COL}' — grouping by authentic polygons.")

# Drop unused columns
drop_cols = ['.geo', 'bare_soil_prob', 'bare_soil_pred', 'B9']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── 2. Feature Engineering ─────────────────────────────────────────────────────
print("\nEngineering features (including new NDSI)...")
eps = 1e-8
df['NDVI']       = (df['B8'] - df['B4']) / (df['B8']  + df['B4']  + eps)
df['BSI']        = ((df['B11']+df['B4'])-(df['B8']+df['B2'])) / ((df['B11']+df['B4'])+(df['B8']+df['B2'])+eps)
df['NDWI']       = (df['B3'] - df['B8']) / (df['B3']  + df['B8']  + eps)
df['SWIR_ratio'] = df['B11'] / (df['B12'] + eps)
df['brightness'] = (df['B2'] + df['B3'] + df['B4'] + df['B8']) / 4.0
df['NDSI']       = (df['B11'] - df['B12']) / (df['B11'] + df['B12'] + eps)

# ── 3. Label Encoding ──────────────────────────────────────────────────────────
print("\nEncoding labels...")
le = LabelEncoder()
y  = le.fit_transform(df[TARGET_COL])
X  = df[FEAT_COLS].values.astype(np.float32)
groups = df[GROUP_COL].values
CLASS_NAMES = list(le.classes_)

print("\nClass distribution:")
for cls_name, cls_idx in zip(CLASS_NAMES, range(len(CLASS_NAMES))):
    count = np.sum(y == cls_idx)
    print(f"  {cls_name:<15}: {count:6d} ({(count/len(y)*100):.1f}%)")

# ── 4. Cross-Validation Setup ──────────────────────────────────────────────────
print(f"\nSetting up StratifiedGroupKFold ({N_SPLITS} folds)...")
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Metrics tracking
lgbm_metrics = {'macro': [], 'weighted': [], 'acc': []}
xgb_metrics = {'macro': [], 'weighted': [], 'acc': []}

# For out-of-fold predictions
oof_lgbm = np.zeros(len(y), dtype=int)
oof_xgb = np.zeros(len(y), dtype=int)

fold = 1
for train_idx, test_idx in cv.split(X, y, groups):
    print(f"\n--- FOLD {fold} ---")
    
    # Verify no group overlap
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    assert train_groups.isdisjoint(test_groups), "CRITICAL LEAKAGE DETECTED: Train and test groups overlap!"
    print(f"Group leakage check passed. (Train groups: {len(train_groups)}, Test groups: {len(test_groups)})")
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # -- Train LightGBM --
    t0 = time.time()
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    t_lgbm = time.time() - t0
    
    p_lgbm = lgbm.predict(X_test)
    oof_lgbm[test_idx] = p_lgbm
    
    l_mac = f1_score(y_test, p_lgbm, average='macro')
    l_wt = f1_score(y_test, p_lgbm, average='weighted')
    l_acc = accuracy_score(y_test, p_lgbm)
    lgbm_metrics['macro'].append(l_mac)
    lgbm_metrics['weighted'].append(l_wt)
    lgbm_metrics['acc'].append(l_acc)
    
    # -- Train XGBoost --
    t0 = time.time()
    sample_weights = compute_sample_weight('balanced', y_train)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        objective='multi:softmax', num_class=len(CLASS_NAMES),
        random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    t_xgb = time.time() - t0
    
    p_xgb = xgb.predict(X_test)
    oof_xgb[test_idx] = p_xgb
    
    x_mac = f1_score(y_test, p_xgb, average='macro')
    x_wt = f1_score(y_test, p_xgb, average='weighted')
    x_acc = accuracy_score(y_test, p_xgb)
    xgb_metrics['macro'].append(x_mac)
    xgb_metrics['weighted'].append(x_wt)
    xgb_metrics['acc'].append(x_acc)
    
    print(f"LightGBM ({t_lgbm:.1f}s) -> Macro F1: {l_mac:.4f}")
    print(f"XGBoost  ({t_xgb:.1f}s) -> Macro F1: {x_mac:.4f}")
    fold += 1

# ── 5. Comparison ──────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS (Out-of-Fold)")
print("="*50)

def print_stats(name, mets):
    print(f"[{name}]")
    print(f"  Macro F1    : {np.mean(mets['macro']):.4f} ± {np.std(mets['macro']):.4f}")
    print(f"  Weighted F1 : {np.mean(mets['weighted']):.4f} ± {np.std(mets['weighted']):.4f}")
    print(f"  Accuracy    : {np.mean(mets['acc']):.4f} ± {np.std(mets['acc']):.4f}")

print_stats("LightGBM", lgbm_metrics)
print("")
print_stats("XGBoost", xgb_metrics)

lgbm_mean_macro = np.mean(lgbm_metrics['macro'])
xgb_mean_macro = np.mean(xgb_metrics['macro'])

winner_name = "LightGBM" if lgbm_mean_macro >= xgb_mean_macro else "XGBoost"
print(f"\nWinner: {winner_name}")

# ── 6. Final Model & Diagnostics ───────────────────────────────────────────────
print(f"\nTraining final {winner_name} model on ALL data...")
if winner_name == "LightGBM":
    final_model = LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    final_model.fit(X, y)
    oof_preds = oof_lgbm
    importances = final_model.feature_importances_
else:
    sample_weights_all = compute_sample_weight('balanced', y)
    final_model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, objective='multi:softmax', num_class=len(CLASS_NAMES), random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(X, y, sample_weight=sample_weights_all)
    oof_preds = oof_xgb
    importances = final_model.feature_importances_

# Diagnostics: Confusion Matrix (using OOF preds)
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_v2.png')
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix(y, oof_preds), display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f'{winner_name} - OOF Confusion Matrix (3-Class)')
plt.tight_layout()
plt.savefig(cm_path, dpi=150)
plt.close()

# Diagnostics: Feature Importances
fi_df = pd.DataFrame({'feature': FEAT_COLS, 'importance': importances}).sort_values('importance', ascending=False)
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance_v2.png')
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=fi_df, x='importance', y='feature', ax=ax, palette='viridis')
ax.set_title(f'{winner_name} - Feature Importances')
plt.tight_layout()
plt.savefig(fi_path, dpi=150)
plt.close()

# Diagnostics: Per-District F1
print("\nPer-District Diagnostic (OOF Predictions):")
df['oof_pred'] = oof_preds
districts = df['district'].unique()
district_res = []
for d in districts:
    mask = df['district'] == d
    d_macro = f1_score(y[mask], oof_preds[mask], average='macro')
    district_res.append({'district': d, 'macro_f1': d_macro})
d_df = pd.DataFrame(district_res).sort_values('macro_f1')
print(d_df.to_string(index=False))

# ── 7. Save Final Model ────────────────────────────────────────────────────────
bundle = {
    'model': final_model,
    'label_encoder': le,
    'feature_cols': FEAT_COLS,
    'band_cols': BAND_COLS,
    'index_cols': INDEX_COLS,
    'class_names': CLASS_NAMES,
    'metrics': {
        'cv_macro_f1_mean': float(np.mean(lgbm_metrics['macro'] if winner_name == 'LightGBM' else xgb_metrics['macro'])),
        'cv_macro_f1_std': float(np.std(lgbm_metrics['macro'] if winner_name == 'LightGBM' else xgb_metrics['macro'])),
    }
}
model_path = os.path.join(OUTPUT_DIR, 'soil_type_classifier_v2.joblib')
joblib.dump(bundle, model_path, compress=3)

# ── 8. Final Summary ───────────────────────────────────────────────────────────
print("\n" + "="*50)
print("FINAL PIPELINE SUMMARY")
print("="*50)
print(f"Chosen Model  : {winner_name}")
mets = lgbm_metrics if winner_name == "LightGBM" else xgb_metrics
print(f"CV Macro F1   : {np.mean(mets['macro']):.4f} ± {np.std(mets['macro']):.4f}")
print(f"CV Weighted F1: {np.mean(mets['weighted']):.4f} ± {np.std(mets['weighted']):.4f}")
print(f"CV Accuracy   : {np.mean(mets['acc']):.4f} ± {np.std(mets['acc']):.4f}")
print("\nTop 5 Features:")
print(fi_df.head(5).to_string(index=False))
print("\nBottom 3 Districts (Lowest Macro F1):")
print(d_df.head(3).to_string(index=False))
print(f"\nArtifacts Saved:\n - {cm_path}\n - {fi_path}\n - {model_path}")
print("DONE.")
