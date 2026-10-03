import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

INTERMEDIATE_PARQUET = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\mae_soil_classification\bare_soil_with_mae.parquet"
OUT_DIR = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\mae_soil_classification\exhaustive_results"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading cached dataset...")
df = pd.read_parquet(INTERMEDIATE_PARQUET)

s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
s2_bands = [f for f in s2_bands if f in df.columns]

mae_cols = [f'MAE_{i:02d}' for i in range(32)]
mae_cols = [f for f in mae_cols if f in df.columns]

terrain_cols = ['elevation', 'slope', 'aspect']
terrain_cols = [c for c in terrain_cols if c in df.columns]

glcm_suffixes = ['asm', 'contrast', 'corr', 'var', 'idm', 'savg', 'svar', 'sent', 'ent', 'dvar', 'dent', 'ic1', 'ic2', 'maxcc']
glcm_cols = [c for c in df.columns if '_' in c and c.split('_')[-1] in glcm_suffixes]

# We only need the missing two
feature_sets = {
    'S2_and_MAE': (s2_bands + mae_cols, 13),
    'S2_MAE_Terrain_GLCM': (s2_bands + mae_cols + terrain_cols + glcm_cols, 15)
}

y_map = {'s1': 0, 's2': 1, 's3': 2}
y = df['soil_type'].map(y_map)

for set_name, (raw_features, best_k) in feature_sets.items():
    print(f"\n=======================================================================")
    print(f"Generating SHAP for: {set_name} (Top {best_k} features)")
    
    # Clustering (same logic)
    X_feat = df[raw_features].sample(min(20000, len(df)), random_state=42)
    corr = spearmanr(X_feat).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr)
    distance_matrix[distance_matrix < 0] = 0
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    cluster_ids = hierarchy.fcluster(dist_linkage, 0.1, criterion='distance')
    
    cluster_dict = {}
    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = raw_features[i]
    selected_features = list(cluster_dict.values())
    X = df[selected_features]
    
    # Train full LGBM to get permutation importance on a sub-sample
    print("Training base selector model...")
    sel = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    sel.fit(X, y)
    
    sub_idx = np.random.choice(len(X), min(20000, len(X)), replace=False)
    perm = permutation_importance(sel, X.iloc[sub_idx], y.iloc[sub_idx], n_repeats=3, random_state=42, n_jobs=-1)
    
    top_features = X.columns[perm.importances_mean.argsort()[::-1][:best_k]]
    X_k = X[top_features]
    
    # Train final LGBM on the best features
    print("Training final model for SHAP...")
    final_m = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    final_m.fit(X_k, y)
    
    print("Generating SHAP plots...")
    plt.figure(figsize=(10, 8))
    explainer = shap.TreeExplainer(final_m)
    
    # Use a 5000-row sample for generating SHAP values to save time
    X_k_sample = X_k.sample(min(5000, len(X_k)), random_state=42)
    shap_values = explainer.shap_values(X_k_sample)
    
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_k_sample, plot_type="bar", show=False, class_names=['s1', 's2', 's3'])
    else:
        shap.summary_plot(shap_values, X_k_sample, show=False)
        
    plt.title(f"SHAP Summary: {set_name} (LightGBM)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'shap_{set_name}_LGBM.png'))
    plt.close()
    
    print(f"Saved shap_{set_name}_LGBM.png")
