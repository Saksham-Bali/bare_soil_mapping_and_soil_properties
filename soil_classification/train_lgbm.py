"""
train_lgbm.py
-------------
Trains a LightGBM classifier on bare-soil pixels and compares
against the existing best model (RandomForest, macro F1 = 0.6227).
If LightGBM is better, replaces the saved .joblib bundle.
"""

import os, time, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lightgbm                  import LGBMClassifier
from sklearn.preprocessing     import LabelEncoder
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       f1_score, accuracy_score, ConfusionMatrixDisplay)
from imblearn.over_sampling    import SMOTE

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH      = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\bare_soil_pixels.parquet'
OUTPUT_DIR        = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\soil_type_model'
BEST_MODEL_PATH   = os.path.join(OUTPUT_DIR, 'soil_type_randomforest.joblib')
CURRENT_BEST_F1   = 0.6227   # RandomForest test macro F1

BAND_COLS  = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
INDEX_COLS = ['NDVI','BSI','NDWI','SWIR_ratio','brightness']
FEAT_COLS  = BAND_COLS + INDEX_COLS
TARGET_COL = 'soil_type'
RANDOM_STATE = 42

# ── 1. Load & feature-engineer ────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(PARQUET_PATH)
df.drop(columns=[c for c in ['.geo','district','bare_soil_prob','bare_soil_pred']
                 if c in df.columns], inplace=True)

eps = 1e-8
df['NDVI']       = (df['B8'] - df['B4']) / (df['B8']  + df['B4']  + eps)
df['BSI']        = ((df['B11']+df['B4'])-(df['B8']+df['B2'])) / \
                   ((df['B11']+df['B4'])+(df['B8']+df['B2'])+eps)
df['NDWI']       = (df['B3'] - df['B8']) / (df['B3']  + df['B8']  + eps)
df['SWIR_ratio'] = df['B11'] / (df['B12'] + eps)
df['brightness'] = (df['B2'] + df['B3'] + df['B4'] + df['B8']) / 4.0

le = LabelEncoder()
y  = le.fit_transform(df[TARGET_COL])
X  = df[FEAT_COLS].values.astype(np.float32)
CLASS_NAMES = list(le.classes_)

# ── 2. Same split as before (same random_state = reproducible) ────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ── 3. SMOTE (same as before) ─────────────────────────────────────────────────
print("Applying SMOTE...")
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE: %d samples per class (balanced)" % (len(X_train_res) // len(CLASS_NAMES)))

# ── 4. Train LightGBM ─────────────────────────────────────────────────────────
print("\nTraining LightGBM...")
lgbm = LGBMClassifier(
    n_estimators   = 500,
    max_depth      = 8,
    learning_rate  = 0.05,
    num_leaves     = 63,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    class_weight   = 'balanced',
    random_state   = RANDOM_STATE,
    n_jobs         = -1,
    verbose        = -1,
)

t0 = time.time()
lgbm.fit(X_train_res, y_train_res)
elapsed = time.time() - t0
print("Train time: %.1fs" % elapsed)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred      = lgbm.predict(X_test)
macro_f1    = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
oa          = accuracy_score(y_test, y_pred)

print("\nLightGBM Results:")
print("  Macro F1    : %.4f  (current best RF = %.4f)" % (macro_f1, CURRENT_BEST_F1))
print("  Weighted F1 : %.4f" % weighted_f1)
print("  Overall Acc : %.4f" % oa)
print("\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

# Feature importance
fi = lgbm.feature_importances_
fi_df = pd.DataFrame({'feature': FEAT_COLS, 'importance': fi})
fi_df = fi_df.sort_values('importance', ascending=False)
print("Top feature importances:")
print(fi_df.to_string(index=False))

# ── 6. Compare and conditionally replace ─────────────────────────────────────
print("\n" + "=" * 55)
if macro_f1 > CURRENT_BEST_F1:
    improvement = macro_f1 - CURRENT_BEST_F1
    print("LightGBM WINS (+%.4f macro F1). Replacing saved model." % improvement)

    # Save confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_LightGBM.png')
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred),
        display_labels=CLASS_NAMES
    )
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('LightGBM -- Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print("Confusion matrix saved: %s" % cm_path)

    # Save feature importance plot
    import seaborn as sns
    fi_path = os.path.join(OUTPUT_DIR, 'feature_importance_LightGBM.png')
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=fi_df, x='importance', y='feature', ax=ax, palette='viridis')
    ax.set_title('LightGBM -- Feature Importances')
    plt.tight_layout()
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print("Feature importance saved: %s" % fi_path)

    # Save new best model bundle
    bundle = {
        'model':         lgbm,
        'label_encoder': le,
        'feature_cols':  FEAT_COLS,
        'band_cols':     BAND_COLS,
        'index_cols':    INDEX_COLS,
        'class_names':   CLASS_NAMES,
        'metrics': {
            'macro_f1':    macro_f1,
            'weighted_f1': weighted_f1,
            'oa':          oa,
        }
    }
    save_path = os.path.join(OUTPUT_DIR, 'soil_type_lightgbm.joblib')
    joblib.dump(bundle, save_path, compress=3)
    size_kb = os.path.getsize(save_path) / 1024
    print("\nNew best model saved: %s  (%.1f KB)" % (save_path, size_kb))
    print("VERDICT: LightGBM is the new champion.")

else:
    gap = CURRENT_BEST_F1 - macro_f1
    print("RandomForest HOLDS. LightGBM is worse by %.4f macro F1." % gap)
    print("Existing weights at: %s" % BEST_MODEL_PATH)
    print("VERDICT: No replacement made.")

print("=" * 55)
print("\nDONE.")
