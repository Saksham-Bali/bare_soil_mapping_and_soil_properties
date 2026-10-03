import pandas as pd
import numpy as np
import os
import glob
import joblib
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, classification_report, cohen_kappa_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\aee_glcm_terrain\S2_Soil_Texture_Terrain_CSV"
BARE_SOIL_MODEL = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\bare_soil_pred_new\xgb_bare_soil_2024.joblib"
OUT_DIR = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\final_diagnostics"
MODEL_BUNDLE = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\aee_glcm_terrain\soil_type_final_ensemble.joblib"

print("1. Loading and Preprocessing Data...")
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
dfs = []
for f in csv_files:
    dfs.append(pd.read_csv(f))
df = pd.concat(dfs, ignore_index=True)

if 'brightness' in df.columns and 'Brightness' not in df.columns:
    df.rename(columns={'brightness': 'brightness_glcm_base'}, inplace=True)
if 'dist_to_water' in df.columns:
    df.drop(columns=['dist_to_water'], inplace=True)

eps = 1e-8
df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + eps)
df['BSI'] = ((df['B11'] + df['B4']) - (df['B8'] + df['B2'])) / ((df['B11'] + df['B4']) + (df['B8'] + df['B2']) + eps)
df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + eps)
df['SWIR_ratio'] = df['B11'] / (df['B12'] + eps)
df['Brightness'] = (df['B2'] + df['B3'] + df['B4'] + df['B8']) / 4.0
df['NDSI'] = (df['B3'] - df['B11']) / (df['B3'] + df['B11'] + eps)

bare_soil_model = joblib.load(BARE_SOIL_MODEL)
bs_features = bare_soil_model.feature_names_in_
for f in bs_features:
    if f not in df.columns and f == 'B1':
        df['B1'] = df['B2'] 

X_bs = df[bs_features]
df['bare_soil_prob'] = bare_soil_model.predict_proba(X_bs)[:, 1]
df = df[df['bare_soil_prob'] > 0.5].copy()
df['soil_type'] = df['soil_type'].replace('s4', 's3')
df = df.dropna()

from sklearn.inspection import permutation_importance

print(f"Top 40 features being calculated...")
s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
indices = ['NDVI', 'BSI', 'NDWI', 'SWIR_ratio', 'Brightness', 'NDSI']
glcm_cols = [c for c in df.columns if any(x in c for x in ['_asm', '_contrast', '_corr', '_ent', '_idm'])]
terrain_cols = ['elevation', 'slope', 'aspect']
aee_cols = [f'A{i:02d}' for i in range(64)]

all_features = s2_bands + indices + glcm_cols + terrain_cols + aee_cols
all_features = [f for f in all_features if f in df.columns]

X_temp = df[all_features]
y_map = {'s1': 0, 's2': 1, 's3': 2}
inv_y_map = {0: 's1', 1: 's2', 2: 's3'}
y = df['soil_type'].map(y_map)

# Subsample for very fast feature selection to exactly match prior fast_shap script
df_sub = df.sample(min(10000, len(df)), random_state=42)
X_sub = df_sub[all_features]
y_sub = df_sub['soil_type'].map(y_map)

sel_final = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=1, verbose=-1)
sel_final.fit(X_sub, y_sub)
perm_final = permutation_importance(sel_final, X_sub, y_sub, n_repeats=3, random_state=42, n_jobs=1)
best_k = 40
top_features = X_sub.columns[perm_final.importances_mean.argsort()[::-1][:best_k]]
top_features = list(top_features)

print(f"Top 40 features calculated and loaded.")

X = df[top_features]

print("\n--- 6. Feature Redundancy Sanity Check ---")
cat_counts = {'AEE': 0, 'S2_Bands': 0, 'Indices': 0, 'GLCM': 0, 'Terrain': 0}
s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
indices = ['NDVI', 'BSI', 'NDWI', 'SWIR_ratio', 'Brightness', 'NDSI']
glcm_cols = [c for c in df.columns if any(x in c for x in ['_asm', '_contrast', '_corr', '_ent', '_idm'])]
terrain_cols = ['elevation', 'slope', 'aspect']

for f in top_features:
    if f.startswith('A') and f[1:].isdigit(): cat_counts['AEE'] += 1
    elif f in s2_bands: cat_counts['S2_Bands'] += 1
    elif f in indices: cat_counts['Indices'] += 1
    elif f in glcm_cols: cat_counts['GLCM'] += 1
    elif f in terrain_cols: cat_counts['Terrain'] += 1
print(pd.Series(cat_counts))

print("\n--- 4. & 5. Per-class Breakdown & Kappa (5-fold CV) ---")
cv = StratifiedGroupKFold(n_splits=5)
oof_preds = np.zeros(len(df))
df_reset = df.reset_index(drop=True)
X_reset = X.reset_index(drop=True)
y_reset = y.reset_index(drop=True)

groups = df_reset['polygon_id']
# For speed in diagnostics, 500 iterations of CatBoost is sufficient for proxy testing
model = CatBoostClassifier(iterations=500, auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1)

for train_idx, val_idx in cv.split(X_reset, y_reset, groups):
    model.fit(X_reset.iloc[train_idx], y_reset.iloc[train_idx])
    oof_preds[val_idx] = model.predict(X_reset.iloc[val_idx]).flatten()

macro_f1 = f1_score(y_reset, oof_preds, average='macro')
acc = accuracy_score(y_reset, oof_preds)
kappa = cohen_kappa_score(y_reset, oof_preds)
print(f"5-Fold CV (Grouped by Polygon):")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Kappa:    {kappa:.4f}")
print("Classification Report:")
print(classification_report(y_reset, oof_preds, target_names=['s1', 's2', 's3']))

print("\n--- 1. Leave-One-District-Out (LODO) check ---")
districts = df_reset['district'].unique()
lodo_results = []

for d in districts:
    test_idx = df_reset[df_reset['district'] == d].index
    train_idx = df_reset[df_reset['district'] != d].index
    
    classes_present = y_reset.iloc[test_idx].nunique()
    if classes_present <= 1:
        lodo_results.append({'district': d, 'n_pixels': len(test_idx), 'classes_present': classes_present, 'macro_f1': np.nan, 'accuracy': np.nan, 'status': 'Not Evaluable'})
        continue
        
    model.fit(X_reset.iloc[train_idx], y_reset.iloc[train_idx])
    preds = model.predict(X_reset.iloc[test_idx]).flatten()
    
    d_f1 = f1_score(y_reset.iloc[test_idx], preds, average='macro')
    d_acc = accuracy_score(y_reset.iloc[test_idx], preds)
    
    lodo_results.append({'district': d, 'n_pixels': len(test_idx), 'classes_present': classes_present, 'macro_f1': d_f1, 'accuracy': d_acc, 'status': 'Evaluated'})

lodo_df = pd.DataFrame(lodo_results)
print(lodo_df.to_string())

evaluated = lodo_df[lodo_df['status'] == 'Evaluated']
lodo_avg_f1 = 0.0
if len(evaluated) > 0:
    lodo_avg_f1 = evaluated['macro_f1'].mean()
    print(f"\nAverage LODO Macro F1: {lodo_avg_f1:.4f} (Compare to Grouped CV: {macro_f1:.4f})")
    print(f"Gap: {macro_f1 - lodo_avg_f1:.4f}")

print("\n--- 2. Elevation / AEE dimension vs. district identity check ---")
aee_top5 = [f for f in top_features if f.startswith('A')][:5]
dist_features = ['elevation'] + aee_top5 if 'elevation' in df_reset.columns else aee_top5

dist_acc = 0.0
if len(dist_features) > 0:
    le = LabelEncoder()
    y_dist = le.fit_transform(df_reset['district'])
    
    # Simple CV to predict district
    dist_cv = StratifiedGroupKFold(n_splits=3)
    dist_preds = np.zeros(len(df_reset))
    dist_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    
    for train_idx, val_idx in dist_cv.split(df_reset[dist_features], y_dist, df_reset['polygon_id']):
        dist_model.fit(df_reset[dist_features].iloc[train_idx], y_dist[train_idx])
        dist_preds[val_idx] = dist_model.predict(df_reset[dist_features].iloc[val_idx])
        
    dist_acc = accuracy_score(y_dist, dist_preds)
    print(f"Predicting District from {dist_features}:")
    print(f"Accuracy: {dist_acc:.4f} (Random guessing: {1/len(le.classes_):.4f})")
else:
    print("Elevation or AEE features not available for district check.")

if 'elevation' in df_reset.columns:
    print("\nElevation variance within districts across soil types:")
    for d in evaluated['district']:
        subset = df_reset[df_reset['district'] == d]
        means = subset.groupby('soil_type')['elevation'].mean()
        stds = subset.groupby('soil_type')['elevation'].std()
        print(f"District {d}:")
        for st in means.index:
            print(f"  {st}: {means[st]:.1f} +- {stds[st]:.1f}m")

print("\n--- 3. AEE dimension correlation with physical covariates ---")
aee_dims = [f for f in top_features if f.startswith('A')]
phys_feats = ['elevation', 'slope', 'aspect', 'NDVI', 'BSI', 'Brightness', 'SWIR_ratio', 'NDSI']
phys_feats = [f for f in phys_feats if f in df_reset.columns]

if len(aee_dims) > 0 and len(phys_feats) > 0:
    corr_results = []
    df_corr = df_reset.sample(min(20000, len(df_reset)), random_state=42)
    for a in aee_dims:
        row = {'AEE_Dim': a}
        for p in phys_feats:
            r, _ = pearsonr(df_corr[a], df_corr[p])
            row[p] = r
        corr_results.append(row)
    
    corr_df = pd.DataFrame(corr_results).set_index('AEE_Dim')
    print(corr_df.round(3).to_string())
    corr_df.to_csv(os.path.join(OUT_DIR, "aee_physical_correlations.csv"))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title("AEE vs Physical Features Correlation")
    plt.savefig(os.path.join(OUT_DIR, "aee_correlation_heatmap.png"), bbox_inches='tight', dpi=300)
    plt.close()

with open(os.path.join(OUT_DIR, "diagnostic_summary.txt"), "w") as f:
    f.write(f"Grouped 5-Fold Macro F1: {macro_f1:.4f}\n")
    f.write(f"LODO Average Macro F1:   {lodo_avg_f1:.4f}\n")
    f.write(f"Generalization Gap:      {macro_f1 - lodo_avg_f1:.4f}\n")
    f.write(f"District Predictability: {dist_acc:.4f}\n")

print("\nAll diagnostics complete.")
