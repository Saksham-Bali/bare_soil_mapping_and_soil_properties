import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

INTERMEDIATE_PARQUET = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\mae_soil_classification\bare_soil_with_mae.parquet"
OUT_DIR = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\mae_soil_classification\exhaustive_results"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading cached dataset from {INTERMEDIATE_PARQUET}...")
df = pd.read_parquet(INTERMEDIATE_PARQUET)
print(f"Dataset loaded. Total pixels: {len(df)}")

s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
s2_bands = [f for f in s2_bands if f in df.columns]

mae_cols = [f'MAE_{i:02d}' for i in range(32)]
mae_cols = [f for f in mae_cols if f in df.columns]

terrain_cols = ['elevation', 'slope', 'aspect']
terrain_cols = [c for c in terrain_cols if c in df.columns]

glcm_suffixes = ['asm', 'contrast', 'corr', 'var', 'idm', 'savg', 'svar', 'sent', 'ent', 'dvar', 'dent', 'ic1', 'ic2', 'maxcc']
glcm_cols = [c for c in df.columns if '_' in c and c.split('_')[-1] in glcm_suffixes]

feature_sets = {
    'S2_MAE_Terrain_GLCM': s2_bands + mae_cols + terrain_cols + glcm_cols
}

y_map = {'s1': 0, 's2': 1, 's3': 2}
y = df['soil_type'].map(y_map)
groups = df['polygon_id'] if 'polygon_id' in df.columns else df.index

splits_to_test = [3, 5]
top_k_options = [15, 25, 40]

for set_name, raw_features in feature_sets.items():
    print(f"\n=======================================================================")
    print(f"================ STARTING PIPELINE FOR: {set_name} ================")
    print(f"=======================================================================")
    
    print("1. Clustering highly correlated features (r > 0.90)...")
    X_feat = df[raw_features].sample(min(20000, len(df)), random_state=42)
    corr = spearmanr(X_feat).correlation
    
    # Handle single feature or errors safely
    if isinstance(corr, float) or corr.shape == ():
        selected_features = raw_features
    else:
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
        
    print(f"Reduced from {len(raw_features)} to {len(selected_features)} features.")
    
    X = df[selected_features]
    baseline_features = [f for f in s2_bands if f in selected_features]
    if len(baseline_features) == 0: baseline_features = selected_features
    
    results = []
    
    print("\n2. Starting Cross Validation & Feature Selection...")
    for n_splits in splits_to_test:
        print(f"\n--- n_splits = {n_splits} ---")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        valid_ks = [k for k in top_k_options if k <= len(selected_features)]
        if len(selected_features) not in valid_ks:
            valid_ks.append(len(selected_features))
        valid_ks = sorted(list(set(valid_ks)))
        
        for k in valid_ks:
            print(f"  Evaluating Top-{k} features...")
            fold_metrics = {'lgb': [], 'xgb': [], 'cat': []}
            
            for train_idx, test_idx in sgkf.split(X, y, groups):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
                
                sel = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
                sel.fit(X_tr, y_tr)
                
                sub_idx = np.random.choice(len(X_tr), min(20000, len(X_tr)), replace=False)
                X_tr_sub, y_tr_sub = X_tr.iloc[sub_idx], y_tr.iloc[sub_idx]
                
                perm = permutation_importance(sel, X_tr_sub, y_tr_sub, n_repeats=3, random_state=42, n_jobs=1)
                top_features = X.columns[perm.importances_mean.argsort()[::-1][:k]]
                
                X_tr_k = X_tr[top_features]
                X_te_k = X_te[top_features]
                
                lgb_m = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
                lgb_m.fit(X_tr_k, y_tr)
                fold_metrics['lgb'].append(f1_score(y_te, lgb_m.predict(X_te_k), average='macro'))
                
                classes, counts = np.unique(y_tr, return_counts=True)
                w_dict = dict(zip(classes, len(y_tr) / (len(classes) * counts)))
                sample_weights = y_tr.map(w_dict).values
                
                xgb_m = xgb.XGBClassifier(random_state=42, n_jobs=-1)
                xgb_m.fit(X_tr_k, y_tr, sample_weight=sample_weights)
                fold_metrics['xgb'].append(f1_score(y_te, xgb_m.predict(X_te_k), average='macro'))
                
                cat_m = CatBoostClassifier(auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1)
                cat_m.fit(X_tr_k, y_tr)
                fold_metrics['cat'].append(f1_score(y_te, cat_m.predict(X_te_k), average='macro'))
                
            for m_name, scores in fold_metrics.items():
                results.append({
                    'n_splits': n_splits,
                    'top_k': k,
                    'model': m_name,
                    'macro_f1': np.mean(scores),
                    'std_f1': np.std(scores)
                })
                
    res_df = pd.DataFrame(results)
    best_row = res_df.loc[res_df['macro_f1'].idxmax()]
    print("\n=== WINNING CONFIGURATION ===")
    print(best_row[['n_splits', 'top_k', 'model', 'macro_f1', 'std_f1']])
    
    best_n_splits = best_row['n_splits']
    best_k = best_row['top_k']
    best_model_name = best_row['model']
    
    print(f"\n3. Running Final Ensemble Evaluation for {set_name}...")
    sgkf = StratifiedGroupKFold(n_splits=best_n_splits, shuffle=True, random_state=42)
    
    oof_preds_single = np.zeros(len(df))
    oof_preds_ensemble = np.zeros((len(df), 3))
    
    stack_X_train_list = []
    stack_y_train_list = []
    oof_y = np.zeros(len(df))
    
    best_model_for_shap = None
    X_te_k_for_shap = None
    
    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        sel = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        sel.fit(X_tr, y_tr)
        
        sub_idx = np.random.choice(len(X_tr), min(20000, len(X_tr)), replace=False)
        perm = permutation_importance(sel, X_tr.iloc[sub_idx], y_tr.iloc[sub_idx], n_repeats=3, random_state=42, n_jobs=1)
        
        top_features = X.columns[perm.importances_mean.argsort()[::-1][:best_k]]
        
        X_tr_k = X_tr[top_features]
        X_te_k = X_te[top_features]
        
        lgb_m = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        lgb_m.fit(X_tr_k, y_tr)
        p_lgb = lgb_m.predict_proba(X_te_k)
        
        classes, counts = np.unique(y_tr, return_counts=True)
        w_dict = dict(zip(classes, len(y_tr) / (len(classes) * counts)))
        xgb_m = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        xgb_m.fit(X_tr_k, y_tr, sample_weight=y_tr.map(w_dict).values)
        p_xgb = xgb_m.predict_proba(X_te_k)
        
        cat_m = CatBoostClassifier(auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1)
        cat_m.fit(X_tr_k, y_tr)
        p_cat = cat_m.predict_proba(X_te_k)
        
        if best_model_name == 'lgb': 
            p_win = p_lgb
            best_model_for_shap = lgb_m
        elif best_model_name == 'xgb': 
            p_win = p_xgb
            best_model_for_shap = xgb_m
        else: 
            p_win = p_cat
            best_model_for_shap = cat_m
            
        X_te_k_for_shap = X_te_k
            
        oof_preds_single[test_idx] = np.argmax(p_win, axis=1)
        oof_preds_ensemble[test_idx] = (p_lgb + p_xgb + p_cat) / 3.0
        oof_y[test_idx] = y_te
        
        stack_X_train_list.append(np.hstack([p_lgb, p_xgb, p_cat]))
        stack_y_train_list.append(y_te)
        
    print(f"\n--- {set_name} : SINGLE WINNING MODEL (OOF) ---")
    print(f"Macro F1: {f1_score(oof_y, oof_preds_single, average='macro'):.4f}")
    
    oof_preds_ens_class = np.argmax(oof_preds_ensemble, axis=1)
    print(f"\n--- {set_name} : SIMPLE AVERAGE ENSEMBLE (OOF) ---")
    print(f"Macro F1: {f1_score(oof_y, oof_preds_ens_class, average='macro'):.4f}")
    
    print("\n4. Training Stacking Meta-Learner...")
    stack_X = np.vstack(stack_X_train_list)
    stack_y = np.concatenate(stack_y_train_list)
    meta_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    stack_oof_preds = cross_val_predict(meta_model, stack_X, stack_y, cv=5)
    
    print(f"\n--- {set_name} : STACKING ENSEMBLE (OOF) ---")
    print(f"Macro F1: {f1_score(stack_y, stack_oof_preds, average='macro'):.4f}")
    print("\nClassification Report (Stacking):")
    print(classification_report(stack_y, stack_oof_preds, target_names=['s1', 's2', 's3']))
    
    print("\n5. Generating SHAP plots...")
    try:
        plt.figure(figsize=(10, 8))
        explainer = shap.TreeExplainer(best_model_for_shap)
        shap_values = explainer.shap_values(X_te_k_for_shap)
        
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values, X_te_k_for_shap, plot_type="bar", show=False, class_names=['s1', 's2', 's3'])
        else:
            if len(np.shape(shap_values)) == 3:
                shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
                shap.summary_plot(shap_values, X_te_k_for_shap, plot_type="bar", show=False, class_names=['s1', 's2', 's3'])
            else:
                shap.summary_plot(shap_values, X_te_k_for_shap, show=False)
                
        plt.title(f"SHAP Summary: {set_name} ({best_model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'shap_{set_name}.png'))
        plt.close()
    except Exception as e:
        print(f"Failed to generate SHAP plots: {e}")
        plt.close()
    
    print(f"\nCOMPLETED {set_name} SUCCESSFULLY.\n")
