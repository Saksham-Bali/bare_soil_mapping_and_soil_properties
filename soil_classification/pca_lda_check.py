import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import euclidean, cosine

# 1. Load the parquet, drop .geo, bare_soil_prob, bare_soil_pred, B9.
parquet_path = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\bare_soil_pixels.parquet"
df = pd.read_parquet(parquet_path)

drop_cols = ['.geo', 'bare_soil_prob', 'bare_soil_pred', 'B9']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# 2. Engineer 5 indices
eps = 1e-8
B2, B3, B4, B8, B11, B12 = df['B2'], df['B3'], df['B4'], df['B8'], df['B11'], df['B12']
df['NDVI'] = (B8 - B4) / (B8 + B4 + eps)
df['BSI'] = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + eps)
df['NDWI'] = (B3 - B8) / (B3 + B8 + eps)
df['SWIR_ratio'] = B11 / (B12 + eps)
df['brightness'] = (B2 + B3 + B4 + B8) / 4.0

# 3. Feature set = 10 bands + 5 indices
bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
indices = ['NDVI', 'BSI', 'NDWI', 'SWIR_ratio', 'brightness']
feature_cols = bands + indices

X = df[feature_cols].values
y = df['soil_type'].values

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add to dataframe for easy filtering
df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
df_scaled['soil_type'] = y

output_dir = r"D:\Bare_soil\Bare_Soil_Mapping\soil_classification\soil_type_model"
os.makedirs(output_dir, exist_ok=True)

# 5. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['soil_type'] = y

print(f"PCA Explained Variance Ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")

colors = {'s1': 'red', 's2': 'blue', 's3': 'green', 's4': 'orange'}

# Plot PCA all classes
plt.figure(figsize=(10, 8))
for c in ['s1', 's2', 's3', 's4']:
    mask = df_pca['soil_type'] == c
    plt.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'], label=c, color=colors[c], alpha=0.5, s=1)
plt.title('PCA - All Classes')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(markerscale=10)
plt.savefig(os.path.join(output_dir, 'pca_all_classes.png'))
plt.close()

# Plot PCA s2, s3, s4 only
plt.figure(figsize=(10, 8))
for c in ['s2', 's3', 's4']:
    mask = df_pca['soil_type'] == c
    plt.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'], label=c, color=colors[c], alpha=0.5, s=1)
plt.title('PCA - s2, s3, s4 only')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(markerscale=10)
plt.savefig(os.path.join(output_dir, 'pca_s2_s3_s4_only.png'))
plt.close()

# 6. LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)
df_lda = pd.DataFrame(X_lda, columns=['LDA1', 'LDA2'])
df_lda['soil_type'] = y

plt.figure(figsize=(10, 8))
for c in ['s1', 's2', 's3', 's4']:
    mask = df_lda['soil_type'] == c
    plt.scatter(df_lda.loc[mask, 'LDA1'], df_lda.loc[mask, 'LDA2'], label=c, color=colors[c], alpha=0.5, s=1)
plt.title('LDA - All Classes')
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.legend(markerscale=10)
plt.savefig(os.path.join(output_dir, 'lda_all_classes.png'))
plt.close()

# Compute LDA centroids
centroids_lda = df_lda.groupby('soil_type')[['LDA1', 'LDA2']].mean()
print("\nLDA Centroids:")
print(centroids_lda)

dist_lda_s3_s1 = euclidean(centroids_lda.loc['s3'], centroids_lda.loc['s1'])
dist_lda_s3_s2 = euclidean(centroids_lda.loc['s3'], centroids_lda.loc['s2'])
dist_lda_s3_s4 = euclidean(centroids_lda.loc['s3'], centroids_lda.loc['s4'])

print("\nEuclidean distances in LDA space (s3 to others):")
print(f"  s3 to s1: {dist_lda_s3_s1:.4f}")
print(f"  s3 to s2: {dist_lda_s3_s2:.4f}")
print(f"  s3 to s4: {dist_lda_s3_s4:.4f}")

# 7. Compute class-wise mean feature vectors (raw feature space, standardized)
centroids_raw = df_scaled.groupby('soil_type')[feature_cols].mean()

# Cosine similarity and Euclidean distance for s3 vs s2, and s3 vs s4
raw_s1 = centroids_raw.loc['s1'].values
raw_s2 = centroids_raw.loc['s2'].values
raw_s3 = centroids_raw.loc['s3'].values
raw_s4 = centroids_raw.loc['s4'].values

dist_raw_s3_s1 = euclidean(raw_s3, raw_s1)
dist_raw_s3_s2 = euclidean(raw_s3, raw_s2)
dist_raw_s3_s4 = euclidean(raw_s3, raw_s4)

cos_s3_s1 = 1 - cosine(raw_s3, raw_s1)
cos_s3_s2 = 1 - cosine(raw_s3, raw_s2)
cos_s3_s4 = 1 - cosine(raw_s3, raw_s4)
cos_s1_s4 = 1 - cosine(raw_s1, raw_s4)

print("\nRaw Standardized Feature Space:")
print(f"  Euclidean distance s3 to s1: {dist_raw_s3_s1:.4f}")
print(f"  Euclidean distance s3 to s2: {dist_raw_s3_s2:.4f}")
print(f"  Euclidean distance s3 to s4: {dist_raw_s3_s4:.4f}")
print(f"  Cosine similarity  s3 to s1: {cos_s3_s1:.4f}")
print(f"  Cosine similarity  s3 to s2: {cos_s3_s2:.4f}")
print(f"  Cosine similarity  s3 to s4: {cos_s3_s4:.4f}")
print(f"  Cosine similarity  s1 to s4: {cos_s1_s4:.4f}")

# 8. Summary
print("\n=== FINAL SUMMARY ===")
lda_closer = 's2' if dist_lda_s3_s2 < dist_lda_s3_s4 else 's4'
raw_euclidean_closer = 's2' if dist_raw_s3_s2 < dist_raw_s3_s4 else 's4'
raw_cosine_closer = 's2' if cos_s3_s2 > cos_s3_s4 else 's4'

print(f"In LDA space, s3 is spectrally closer to: {lda_closer}")
print(f"In raw feature space (Euclidean), s3 is spectrally closer to: {raw_euclidean_closer}")
print(f"In raw feature space (Cosine), s3 is spectrally closer to: {raw_cosine_closer}")

if lda_closer == raw_euclidean_closer == raw_cosine_closer:
    print(f"\nCONCLUSION: ALL metrics strongly agree. s3 is spectrally closer to {lda_closer}.")
else:
    print("\nCONCLUSION: Metrics disagree. Check above for details.")
