import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import os
import json

# 1. Define file paths
bare_soil_path = '/Users/aaravsinghluthra/Plaksha/ilgc_sem7/Bare_Soil_Mapping/new_data/punjab_bare_soil_pixels_full.csv'
crop_path = '/Users/aaravsinghluthra/Plaksha/ilgc_sem7/punjab_crop_pixels_full.csv'
model_save_path = '/Users/aaravsinghluthra/Plaksha/ilgc_sem7/Bare_Soil_Mapping/ml_bare_soil_pred/xgb_bare_soil_2024.joblib'

print("Loading datasets...")
# 2. Load the data
df_bare = pd.read_csv(bare_soil_path)
df_crop = pd.read_csv(crop_path)

# Ensure target is correct (1 for bare soil, 0 for crop)
df_bare['target'] = 1
df_crop['target'] = 0

print(f"Bare soil samples: {len(df_bare)}")
print(f"Crop samples: {len(df_crop)}")

# 3. Combine and clean
df_combined = pd.concat([df_bare, df_crop], ignore_index=True)

# Drop unnecessary columns
cols_to_drop = ['system:index', '.geo']
df_combined = df_combined.drop(columns=[col for col in cols_to_drop if col in df_combined.columns])

# Drop any rows with NaN values
df_combined = df_combined.dropna()

print(f"Combined clean samples: {len(df_combined)}")

# Define features (X) and target (y)
X = df_combined.drop(columns=['target'])
y = df_combined['target']

# Feature names must match exactly what will be used in deployment
print(f"Features used: {X.columns.tolist()}")

# 4. Train/Test Split (80/20)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Train XGBoost Model
print("Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 6. Evaluate Model
print("Evaluating model...")
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, digits=4, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, digits=4))
print("--- Confusion Matrix ---")
print(conf_matrix)

# 7. Save Model
print(f"Saving model to {model_save_path}...")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(model, model_save_path)

# Also save metrics for agent
metrics = {
    "accuracy": report['accuracy'],
    "macro_avg_f1": report['macro avg']['f1-score'],
    "bare_soil_f1": report['1']['f1-score'],
    "crop_f1": report['0']['f1-score'],
    "conf_matrix": conf_matrix.tolist()
}
with open('/Users/aaravsinghluthra/Plaksha/ilgc_sem7/Bare_Soil_Mapping/scratch/metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Done!")
