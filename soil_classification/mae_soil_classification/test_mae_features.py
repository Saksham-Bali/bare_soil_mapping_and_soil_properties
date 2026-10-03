import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report
import warnings
warnings.filterwarnings('ignore')

INTERMEDIATE_PARQUET = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\mae_soil_classification\bare_soil_with_mae.parquet"

print(f"Loading cached dataset from {INTERMEDIATE_PARQUET}...")
df = pd.read_parquet(INTERMEDIATE_PARQUET)
print(f"Dataset loaded. Total pixels: {len(df)}")

# Define feature sets
s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
s2_bands = [f for f in s2_bands if f in df.columns]

mae_cols = [f'MAE_{i:02d}' for i in range(32)]
mae_cols = [f for f in mae_cols if f in df.columns]

terrain_cols = ['elevation', 'slope', 'aspect']
terrain_cols = [c for c in terrain_cols if c in df.columns]

glcm_suffixes = ['asm', 'contrast', 'corr', 'var', 'idm', 'savg', 'svar', 'sent', 'ent', 'dvar', 'dent', 'ic1', 'ic2', 'maxcc']
glcm_cols = [c for c in df.columns if '_' in c and c.split('_')[-1] in glcm_suffixes]

feature_sets = {
    'S2_and_MAE': s2_bands + mae_cols,
    'S2_MAE_GLCM': s2_bands + mae_cols + glcm_cols,
    'S2_MAE_Terrain': s2_bands + mae_cols + terrain_cols,
    'S2_MAE_Terrain_GLCM': s2_bands + mae_cols + terrain_cols + glcm_cols
}

y_map = {'s1': 0, 's2': 1, 's3': 2}
y = df['soil_type'].map(y_map)
groups = df['polygon_id'] if 'polygon_id' in df.columns else df.index

n_splits = 3
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

for set_name, features in feature_sets.items():
    print(f"\n==================================================")
    print(f"Evaluating Feature Set: {set_name} ({len(features)} features)")
    print(f"==================================================")
    
    X = df[features]
    oof_preds = np.zeros(len(df))
    oof_y = np.zeros(len(df))
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        m = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr)
        
        p = m.predict(X_te)
        oof_preds[test_idx] = p
        oof_y[test_idx] = y_te
        
    macro_f1 = f1_score(oof_y, oof_preds, average='macro')
    acc = accuracy_score(oof_y, oof_preds)
    kappa = cohen_kappa_score(oof_y, oof_preds)
    
    print(f"\n--- {set_name} Out-of-Fold Results ---")
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Macro F1 Score   : {macro_f1:.4f}")
    print(f"Cohen's Kappa    : {kappa:.4f}")
    print("\nClassification Report:")
    print(classification_report(oof_y, oof_preds, target_names=['s1', 's2', 's3']))
