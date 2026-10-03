"""
retrain_no_b9.py
-----------------
Retrains the soil type RandomForest classifier WITHOUT B9,
since Punjab district parquets don't contain the B9 band.
Features: 10 bands (no B9) + 5 spectral indices = 15 features
"""

import os, time, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     f1_score, accuracy_score, ConfusionMatrixDisplay)
from imblearn.over_sampling  import SMOTE

warnings.filterwarnings('ignore')

PARQUET_PATH = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\bare_soil_pixels.parquet'
OUTPUT_DIR   = r'D:\Bare_soil\Bare_Soil_Mapping\soil_classification\soil_type_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Features: NO B9 ───────────────────────────────────────────────────────────
BAND_COLS  = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']  # 10 bands
INDEX_COLS = ['NDVI','BSI','NDWI','SWIR_ratio','brightness']          # 5 indices
FEAT_COLS  = BAND_COLS + INDEX_COLS                                   # 15 total
TARGET_COL = 'soil_type'

print("=" * 60)
print("SOIL TYPE CLASSIFIER RETRAIN (no B9) — 15 features")
print("=" * 60)

# ── Load & engineer ───────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)
df.drop(columns=[c for c in ['.geo','district','bare_soil_prob','bare_soil_pred','B9']
                 if c in df.columns], inplace=True)

eps = 1e-8
df['NDVI']       = (df['B8'] - df['B4']) / (df['B8']  + df['B4']  + eps)
df['BSI']        = ((df['B11']+df['B4'])-(df['B8']+df['B2'])) / \
                   ((df['B11']+df['B4'])+(df['B8']+df['B2'])+eps)
df['NDWI']       = (df['B3'] - df['B8']) / (df['B3']  + df['B8']  + eps)
df['SWIR_ratio'] = df['B11'] / (df['B12'] + eps)
df['brightness'] = (df['B2'] + df['B3'] + df['B4'] + df['B8']) / 4.0

print("\nClass distribution:")
vc = df[TARGET_COL].value_counts().sort_index()
for cls, cnt in vc.items():
    print("  %-4s : %6d  (%.1f%%)" % (cls, cnt, 100*cnt/len(df)))

le = LabelEncoder()
y  = le.fit_transform(df[TARGET_COL])
X  = df[FEAT_COLS].values.astype(np.float32)
CLASS_NAMES = list(le.classes_)
print("\nLabel encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# ── Split & SMOTE ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain: %d  |  Test: %d" % (len(X_train), len(X_test)))

print("Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)
print("After SMOTE: %d samples, %d per class" % (len(X_res), len(X_res)//4))

# ── Train RandomForest ────────────────────────────────────────────────────────
print("\nTraining RandomForest (300 trees)...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators = 300,
    max_depth    = None,
    class_weight = 'balanced',
    oob_score    = True,
    random_state = 42,
    n_jobs       = -1,
)
rf.fit(X_res, y_res)
print("Train time: %.1fs" % (time.time() - t0))
print("OOB Accuracy: %.4f" % rf.oob_score_)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred      = rf.predict(X_test)
macro_f1    = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
oa          = accuracy_score(y_test, y_pred)

print("\nTest Results (no B9):")
print("  Macro F1    : %.4f" % macro_f1)
print("  Weighted F1 : %.4f" % weighted_f1)
print("  Overall Acc : %.4f" % oa)
print("\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_RF_noB9.png')
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred),
    display_labels=CLASS_NAMES
).plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('RandomForest (no B9) -- Confusion Matrix')
plt.tight_layout()
plt.savefig(cm_path, dpi=150)
plt.close()

# ── Feature importance ────────────────────────────────────────────────────────
fi_df = pd.DataFrame({'feature': FEAT_COLS, 'importance': rf.feature_importances_})
fi_df = fi_df.sort_values('importance', ascending=False)
print("\nFeature importances:")
print(fi_df.to_string(index=False))

fi_path = os.path.join(OUTPUT_DIR, 'feature_importance_RF_noB9.png')
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=fi_df, x='importance', y='feature', ax=ax, palette='viridis')
ax.set_title('RandomForest (no B9) -- Feature Importances')
plt.tight_layout()
plt.savefig(fi_path, dpi=150)
plt.close()

# ── Save bundle ───────────────────────────────────────────────────────────────
bundle = {
    'model':         rf,
    'label_encoder': le,
    'feature_cols':  FEAT_COLS,
    'band_cols':     BAND_COLS,
    'index_cols':    INDEX_COLS,
    'class_names':   CLASS_NAMES,
    'has_b9':        False,
    'metrics': {
        'macro_f1':    macro_f1,
        'weighted_f1': weighted_f1,
        'oa':          oa,
        'oob':         rf.oob_score_,
    }
}

save_path = os.path.join(OUTPUT_DIR, 'soil_type_rf_noB9.joblib')
joblib.dump(bundle, save_path, compress=3)
print("\nSaved: %s  (%.1f MB)" % (save_path, os.path.getsize(save_path)/1e6))
print("Features: %s" % FEAT_COLS)
print("\nDONE.")
