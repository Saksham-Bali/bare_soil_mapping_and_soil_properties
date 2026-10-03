import pandas as pd
import numpy as np
import os
import glob
import joblib
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import wilcoxon, spearmanr, pearsonr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\aee_glcm_terrain\S2_Soil_Texture_Terrain_CSV"
BARE_SOIL_MODEL = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\bare_soil_pred_new\xgb_bare_soil_2024.joblib"
OUT_DIR = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\aee_glcm_terrain"

# ── ✏️ EDIT THESE PATHS FOR SPECTRAL MAE ──────────────────────────────────
CKPT_PATH   = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\mae_soil_classification\spectral_mae_ckpt_32.pt.zip"
SCALER_PATH = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\mae_soil_classification\spectral_scaler_32.joblib"
INTERMEDIATE_PARQUET = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\mae_soil_classification\bare_soil_with_mae.parquet"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# SpectralMAE Config
BAND_COLS      = ['B11','B12','B2','B3','B4','B5','B6','B7','B8','B8A']
LAMBDAS_NM     = [1610, 2190, 490, 560, 665, 705, 740, 783, 842, 865]
FWHM_NM        = [  90,  180,  65,  35,  30,  15,  15,  20, 115,  20]
NUM_BANDS      = len(BAND_COLS)
EMBED_DIM  = 32
ENC_DEPTH  = 6
ENC_HEADS  = 4
DEC_DIM    = 32
DEC_DEPTH  = 3
DEC_HEADS  = 4
FOURIER_FF = 16

# ── SpectralMAE model definition ──────────────────────────────
class SpectralMetaEmbedding(nn.Module):
    def __init__(self, embed_dim, fourier_features=16, lam_min_nm=400., lam_max_nm=2500.):
        super().__init__()
        self.ff = fourier_features
        self.lam_min = lam_min_nm; self.lam_max = lam_max_nm
        in_dim = 2 + 2 * fourier_features
        self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, embed_dim))

    def forward(self, lambda_nm, fwhm_nm):
        lam  = (lambda_nm - self.lam_min) / (self.lam_max - self.lam_min)
        fwhm = fwhm_nm / (self.lam_max - self.lam_min)
        k    = torch.arange(1, self.ff + 1, device=lam.device, dtype=lam.dtype)[None, None, :]
        ang  = 2 * math.pi * lam.unsqueeze(-1) * k
        phi  = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        feat = torch.cat([lam.unsqueeze(-1), fwhm.unsqueeze(-1), phi], dim=-1)
        return self.mlp(feat)


def sample_band_mask(B, C, mask_ratio, device):
    K        = int(round((1. - mask_ratio) * C))
    noise    = torch.rand(B, C, device=device)
    ids      = torch.argsort(noise, dim=1)
    ids_keep = ids[:, :K]; ids_mask = ids[:, K:]
    mask     = torch.zeros(B, C, dtype=torch.bool, device=device)
    mask.scatter_(1, ids_mask, True)
    return mask, ids_keep, ids_mask


class SpectralMAEEncoder(nn.Module):
    def __init__(self, num_bands, embed_dim, enc_depth, enc_heads,
                 lambda_nm, fwhm_nm, fourier_features=16,
                 lam_min_nm=400., lam_max_nm=2500.):
        super().__init__()
        self.C = num_bands; D = embed_dim
        self.lambda_nm_const = torch.tensor(lambda_nm, dtype=torch.float32)
        self.fwhm_nm_const   = torch.tensor(fwhm_nm,   dtype=torch.float32)
        self.value_proj      = nn.Linear(1, D)
        self.cls             = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.meta            = SpectralMetaEmbedding(D, fourier_features, lam_min_nm, lam_max_nm)
        self.enc_mask_token  = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.enc_mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=D, nhead=enc_heads, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=enc_depth)

    def forward(self, x, mask_ratio=0.0):
        B, C = x.shape; D = self.cls.shape[-1]
        tok  = self.value_proj(x.unsqueeze(-1))
        lam  = self.lambda_nm_const.to(x.device).unsqueeze(0).expand(B, C)
        wid  = self.fwhm_nm_const.to(x.device).unsqueeze(0).expand(B, C)
        tok  = tok + self.meta(lam, wid)
        mask, ids_keep, ids_mask = sample_band_mask(B, C, mask_ratio, x.device)
        tok_vis = torch.gather(tok, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        cls_in  = self.cls.expand(B, 1, D)
        enc_out = self.encoder(torch.cat([cls_in, tok_vis], dim=1))
        cls_out = enc_out[:, 0, :]
        vis_out = enc_out[:, 1:, :]
        band_enc = torch.zeros(B, C, D, device=x.device, dtype=x.dtype)
        band_enc.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, D), vis_out)
        return {'cls': cls_out, 'band_enc': band_enc, 'mask': mask,
                'ids_keep': ids_keep, 'ids_mask': ids_mask}


class SpectralMAEDecoder(nn.Module):
    def __init__(self, num_bands, enc_dim, dec_dim, dec_depth, dec_heads,
                 lambda_nm, fwhm_nm, include_cls=True,
                 fourier_features=16, lam_min=400., lam_max=2500.):
        super().__init__()
        self.include_cls = include_cls
        self.lambda_nm_const = torch.tensor(lambda_nm, dtype=torch.float32)
        self.fwhm_nm_const   = torch.tensor(fwhm_nm,   dtype=torch.float32)
        self.mem_proj        = nn.Linear(enc_dim, dec_dim)
        self.dec_mask_token  = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.dec_mask_token, std=0.02)
        self.meta_q = SpectralMetaEmbedding(dec_dim, fourier_features, lam_min, lam_max)
        self.meta_m = SpectralMetaEmbedding(dec_dim, fourier_features, lam_min, lam_max)
        layer = nn.TransformerDecoderLayer(d_model=dec_dim, nhead=dec_heads, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=dec_depth)
        self.head    = nn.Linear(dec_dim, 1)

    def forward(self, enc_band, enc_cls, mask, ids_keep, ids_mask):
        B, C, D = enc_band.shape; dec_dim = self.head.in_features
        mem_vis = torch.gather(enc_band, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mem = mem_vis
        if self.include_cls:
            mem = torch.cat([enc_cls.unsqueeze(1), mem], dim=1)
        mem = self.mem_proj(mem)
        lam = self.lambda_nm_const.to(enc_band.device).unsqueeze(0).expand(B, C)
        wid = self.fwhm_nm_const.to(enc_band.device).unsqueeze(0).expand(B, C)
        lam_v  = torch.gather(lam, 1, ids_keep)
        wid_v  = torch.gather(wid, 1, ids_keep)
        meta_v = self.meta_m(lam_v, wid_v)
        if self.include_cls:
            meta_v = torch.cat([torch.zeros(B, 1, dec_dim, device=enc_band.device), meta_v], dim=1)
        mem = mem + meta_v
        M   = ids_mask.shape[1]
        q   = self.dec_mask_token.expand(B, M, dec_dim)
        lam_m = torch.gather(lam, 1, ids_mask)
        wid_m = torch.gather(wid, 1, ids_mask)
        q   = q + self.meta_q(lam_m, wid_m)
        dec_out = self.decoder(tgt=q, memory=mem)
        pred_m  = self.head(dec_out).squeeze(-1)
        out = torch.zeros(B, C, device=enc_band.device, dtype=pred_m.dtype)
        out.scatter_(1, ids_mask, pred_m)
        return out


class SpectralMAE(nn.Module):
    def __init__(self, num_bands, embed_dim, enc_depth, enc_heads,
                 dec_dim, dec_depth, dec_heads,
                 lambda_nm, fwhm_nm,
                 fourier_features=16, lam_min_nm=400., lam_max_nm=2500.,
                 include_cls_in_memory=True):
        super().__init__()
        self.encoder = SpectralMAEEncoder(
            num_bands, embed_dim, enc_depth, enc_heads,
            lambda_nm, fwhm_nm, fourier_features, lam_min_nm, lam_max_nm
        )
        self.decoder = SpectralMAEDecoder(
            num_bands, embed_dim, dec_dim, dec_depth, dec_heads,
            lambda_nm, fwhm_nm, include_cls_in_memory,
            fourier_features, lam_min_nm, lam_max_nm
        )

    def forward(self, x, mask_ratio=0.6):
        enc  = self.encoder(x, mask_ratio)
        pred = self.decoder(enc['band_enc'], enc['cls'], enc['mask'],
                            enc['ids_keep'], enc['ids_mask'])
        return {'cls': enc['cls'], 'pred_masked': pred, **enc}


@torch.no_grad()
def extract_cls_embeddings(model, X_np, batch_size=8192):
    model.eval()
    embeds = []
    for i in tqdm(range(0, len(X_np), batch_size), desc='Encoding CLS'):
        xb   = torch.from_numpy(X_np[i:i+batch_size]).to(DEVICE)
        enc  = model.encoder(xb, mask_ratio=0.0)
        embeds.append(enc['cls'].cpu().numpy())
    return np.concatenate(embeds, axis=0)


# ==============================================================================

if os.path.exists(INTERMEDIATE_PARQUET):
    print(f"\nLoading intermediate data from {INTERMEDIATE_PARQUET}...")
    df = pd.read_parquet(INTERMEDIATE_PARQUET)
    mae_col_names = [f'MAE_{i:02d}' for i in range(EMBED_DIM)]
else:
    print("1. Loading CSVs...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    dfs = []
    for f in csv_files:
        df_temp = pd.read_csv(f)
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw pixels loaded: {len(df)}")

    # Bare Soil Filtering
    print("2. Filtering for bare soil...")
    bare_soil_model = joblib.load(BARE_SOIL_MODEL)
    bs_features = bare_soil_model.feature_names_in_

    for f in bs_features:
        if f not in df.columns:
            if f == 'B1':
                df['B1'] = df['B2'] # B1 highly correlated with B2
            else:
                print(f"Warning: {f} not in dataset!")

    X_bs = df[bs_features]
    df['bare_soil_prob'] = bare_soil_model.predict_proba(X_bs)[:, 1]
    df = df[df['bare_soil_prob'] > 0.5].copy()
    print(f"Bare soil pixels retained: {len(df)}")

    # Class merge
    print("3. Class mapping (S4 -> S3)...")
    df['soil_type'] = df['soil_type'].replace('s4', 's3')
    print(df['soil_type'].value_counts())

    # Handle NaNs
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows containing NaNs.")


    # ==============================================================================
    # Extract Spectral MAE Embeddings
    # ==============================================================================
    print("\nLoading MAE model and scaler from paths...")
    if not os.path.exists(CKPT_PATH) or not os.path.exists(SCALER_PATH):
        print("WARNING: MAE checkpoint or scaler path not found!")
        print("Please update CKPT_PATH and SCALER_PATH in the script.")
        # Fallback to zeros just to keep the script running if paths are not updated yet
        mae_embeds = np.zeros((len(df), EMBED_DIM))
    else:
        mae_model = SpectralMAE(
            num_bands=NUM_BANDS,
            embed_dim=EMBED_DIM, enc_depth=ENC_DEPTH, enc_heads=ENC_HEADS,
            dec_dim=DEC_DIM,     dec_depth=DEC_DEPTH, dec_heads=DEC_HEADS,
            lambda_nm=LAMBDAS_NM, fwhm_nm=FWHM_NM,
            fourier_features=FOURIER_FF
        ).to(DEVICE)
        
        # Handle older torch versions without weights_only
        try:
            ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        except TypeError:
            ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
            
        mae_model.load_state_dict(ckpt, strict=True)
        mae_model.eval()
        
        mae_model.encoder.lambda_nm_const = mae_model.encoder.lambda_nm_const.to(DEVICE)
        mae_model.encoder.fwhm_nm_const   = mae_model.encoder.fwhm_nm_const.to(DEVICE)
        mae_model.decoder.lambda_nm_const = mae_model.decoder.lambda_nm_const.to(DEVICE)
        mae_model.decoder.fwhm_nm_const   = mae_model.decoder.fwhm_nm_const.to(DEVICE)
        
        scaler_mae = joblib.load(SCALER_PATH)
        
        # Scale 10 bands in exact order: 'B11','B12','B2','B3','B4','B5','B6','B7','B8','B8A'
        X_bands = df[BAND_COLS].values
        X_scaled = scaler_mae.transform(X_bands).astype(np.float32)
        
        print(f"Extracting embeddings for {len(X_scaled)} rows on {DEVICE}...")
        mae_embeds = extract_cls_embeddings(mae_model, X_scaled)

    # Append embeddings to DataFrame
    mae_col_names = [f'MAE_{i:02d}' for i in range(EMBED_DIM)]
    df_embeds = pd.DataFrame(mae_embeds, columns=mae_col_names, index=df.index)
    df = pd.concat([df, df_embeds], axis=1)

    print(f"\nSaving intermediate data to {INTERMEDIATE_PARQUET}...")
    df.to_parquet(INTERMEDIATE_PARQUET)

# ==============================================================================

# Feature lists
s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
all_features = s2_bands + mae_col_names
all_features = [f for f in all_features if f in df.columns]
print(f"Total initial features: {len(all_features)}")

# Correlation clustering
print("4. Clustering highly correlated features (r > 0.90)...")
X_feat = df[all_features].sample(min(20000, len(df)), random_state=42)
corr = spearmanr(X_feat).correlation
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
cluster_ids = hierarchy.fcluster(dist_linkage, 0.1, criterion='distance') # 1 - 0.90 = 0.1
cluster_dict = {}
for i, cluster_id in enumerate(cluster_ids):
    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = all_features[i]

selected_all_features = list(cluster_dict.values())
print(f"Reduced to {len(selected_all_features)} features after removing redundant clusters.")

X = df[selected_all_features]
y_map = {'s1': 0, 's2': 1, 's3': 2}
y = df['soil_type'].map(y_map)
groups = df['polygon_id'] if 'polygon_id' in df.columns else df.index

baseline_features = [f for f in s2_bands if f in selected_all_features]
print(f"Baseline features available: {len(baseline_features)}")

splits_to_test = [3, 5]
top_k_options = [15, 25, 40]
results = []
baseline_fold_scores = {}

# 5. Cross Validation
print("\n5. Starting Cross Validation & Feature Selection...")
for n_splits in splits_to_test:
    print(f"\n--- n_splits = {n_splits} ---")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Baseline
    fold_baseline = []
    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[train_idx][baseline_features], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx][baseline_features], y.iloc[test_idx]
        m = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr)
        fold_baseline.append(f1_score(y_te, m.predict(X_te), average='macro'))
    
    baseline_fold_scores[n_splits] = fold_baseline
    print(f"Baseline (S2 only) Macro F1: {np.mean(fold_baseline):.4f} ± {np.std(fold_baseline):.4f}")
    
    # Advanced configs
    for k in top_k_options:
        k = min(k, len(selected_all_features))
        print(f"  Evaluating Top-{k} features...")
        fold_metrics = {'lgb': [], 'xgb': [], 'cat': []}
        
        for train_idx, test_idx in sgkf.split(X, y, groups):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            
            # Feature selection
            sel = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
            sel.fit(X_tr, y_tr)
            
            # Subsample for fast permutation importance
            sub_idx = np.random.choice(len(X_tr), min(20000, len(X_tr)), replace=False)
            X_tr_sub, y_tr_sub = X_tr.iloc[sub_idx], y_tr.iloc[sub_idx]
            
            perm = permutation_importance(sel, X_tr_sub, y_tr_sub, n_repeats=3, random_state=42, n_jobs=1)
            top_features = X.columns[perm.importances_mean.argsort()[::-1][:k]]
            
            X_tr_k = X_tr[top_features]
            X_te_k = X_te[top_features]
            
            # Models
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
                'std_f1': np.std(scores),
                'scores': scores
            })

res_df = pd.DataFrame(results)
print("\n=== All Configurations ===")
print(res_df[['n_splits', 'top_k', 'model', 'macro_f1', 'std_f1']].sort_values('macro_f1', ascending=False).to_string(index=False))

best_row = res_df.loc[res_df['macro_f1'].idxmax()]
print("\n=== WINNING CONFIGURATION ===")
print(best_row[['n_splits', 'top_k', 'model', 'macro_f1', 'std_f1']])

best_n_splits = best_row['n_splits']
best_k = best_row['top_k']
best_model_name = best_row['model']
base_scores = baseline_fold_scores[best_n_splits]

stat, pval = wilcoxon(best_row['scores'], base_scores, alternative='greater')
print(f"\nWilcoxon Test (Best vs Baseline {best_n_splits}-fold): p-value = {pval:.4f}")

# 6. Final Evaluation & Ensembling
print(f"\n6. Running Final Ensemble Evaluation on n_splits={best_n_splits}...")
sgkf = StratifiedGroupKFold(n_splits=best_n_splits, shuffle=True, random_state=42)

oof_preds_single = np.zeros(len(df))
oof_preds_ensemble = np.zeros((len(df), 3))
oof_y = np.zeros(len(df))
oof_districts = np.array([''] * len(df), dtype=object)

all_top_features_list = []

for train_idx, test_idx in sgkf.split(X, y, groups):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    
    sel = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    sel.fit(X_tr, y_tr)
    
    sub_idx = np.random.choice(len(X_tr), min(20000, len(X_tr)), replace=False)
    perm = permutation_importance(sel, X_tr.iloc[sub_idx], y_tr.iloc[sub_idx], n_repeats=3, random_state=42, n_jobs=1)
    
    top_features = X.columns[perm.importances_mean.argsort()[::-1][:best_k]]
    all_top_features_list.append(top_features)
    
    X_tr_k = X_tr[top_features]
    X_te_k = X_te[top_features]
    
    # Train all 3 for ensemble
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
    
    # Store predictions
    if best_model_name == 'lgb': p_win = p_lgb
    elif best_model_name == 'xgb': p_win = p_xgb
    else: p_win = p_cat
        
    oof_preds_single[test_idx] = np.argmax(p_win, axis=1)
    oof_preds_ensemble[test_idx] = (p_lgb + p_xgb + p_cat) / 3.0
    oof_y[test_idx] = y_te
    if 'district' in df.columns:
        oof_districts[test_idx] = df['district'].iloc[test_idx]

print("\n--- SINGLE WINNING MODEL (OOF) ---")
print(f"Macro F1: {f1_score(oof_y, oof_preds_single, average='macro'):.4f}")
print(f"Accuracy: {accuracy_score(oof_y, oof_preds_single):.4f}")
print(f"Kappa:    {cohen_kappa_score(oof_y, oof_preds_single):.4f}")

oof_preds_ens_class = np.argmax(oof_preds_ensemble, axis=1)
print("\n--- TRI-MODEL ENSEMBLE (OOF) ---")
print(f"Macro F1: {f1_score(oof_y, oof_preds_ens_class, average='macro'):.4f}")
print(f"Accuracy: {accuracy_score(oof_y, oof_preds_ens_class):.4f}")
print(f"Kappa:    {cohen_kappa_score(oof_y, oof_preds_ens_class):.4f}")
print("\nClassification Report (Ensemble):")
print(classification_report(oof_y, oof_preds_ens_class, target_names=['s1', 's2', 's3']))

cm = confusion_matrix(oof_y, oof_preds_ens_class, normalize='true')
print("\nNormalized Confusion Matrix (Row %):")
print(np.round(cm, 3))

# 7. Final Model & Interpretability
print("\n7. Final Model Training & SHAP...")
sel_final = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
sel_final.fit(X, y)

sub_idx = np.random.choice(len(X), min(20000, len(X)), replace=False)
perm_final = permutation_importance(sel_final, X.iloc[sub_idx], y.iloc[sub_idx], n_repeats=3, random_state=42, n_jobs=1)

final_top_features = X.columns[perm_final.importances_mean.argsort()[::-1][:best_k]]
X_final = X[final_top_features]

if best_model_name == 'xgb':
    classes, counts = np.unique(y, return_counts=True)
    w_dict = dict(zip(classes, len(y) / (len(classes) * counts)))
    final_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    final_model.fit(X_final, y, sample_weight=y.map(w_dict).values)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X_final)
elif best_model_name == 'lgb':
    final_model = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    final_model.fit(X_final, y)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X_final)
else:
    final_model = CatBoostClassifier(auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1)
    final_model.fit(X_final, y)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X_final)

plt.figure(figsize=(10, 6))
try:
    if len(shap_values.shape) == 3:
        shap.summary_plot(shap_values[:, :, 0], X_final, show=False)
    elif isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_final, show=False)
    else:
        shap.summary_plot(shap_values, X_final, show=False)
    plt.savefig(os.path.join(OUT_DIR, "shap_summary_mae.png"), bbox_inches='tight', dpi=300)
    plt.close()
except Exception as e:
    print(f"Could not generate SHAP plot: {e}")

plt.figure(figsize=(10, 8))
feat_imps = perm_final.importances_mean[perm_final.importances_mean.argsort()[::-1][:best_k]]
sns.barplot(x=feat_imps, y=final_top_features)
plt.title("Permutation Feature Importance (Final Model)")
plt.savefig(os.path.join(OUT_DIR, "permutation_importance_mae.png"), bbox_inches='tight', dpi=300)
plt.close()

# Save final bundle
bundle = {
    'model': final_model,
    'features': list(final_top_features),
    'k': best_k,
    'metrics': {'macro_f1': f1_score(oof_y, oof_preds_ens_class, average='macro')}
}
joblib.dump(bundle, os.path.join(OUT_DIR, "soil_type_mae_ensemble.joblib"))

print("\nDone. All artifacts saved.")
