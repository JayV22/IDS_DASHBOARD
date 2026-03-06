
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# =====================================
# 1. LOAD MODELS & SCALER
# =====================================
print("[INFO] Loading models...")

autoencoder = load_model(
    "autoencoder_ids_cicids.h5",
    compile=False   # FIX for Keras 'mse' error
)

iso_forest = joblib.load("isolation_forest_ids.pkl")
scaler = joblib.load("scaler_ids.pkl")


# =====================================
# 2. LOAD CIC-IDS2017 DATASET
# =====================================
print("[INFO] Loading dataset...")

df = pd.read_parquet("/Users/macbookair/Desktop/IDS_DASHBOARD/CIC-IDS2017/Benign-Monday-no-metadata.parquet")

# True labels (only for evaluation, not training)
y_true = (df['Label'] != 'BENIGN').astype(int)

# Drop label column
X = df.drop(columns=['Label'])

# Keep numeric features only
X = X.select_dtypes(include=[np.number])

# Clean data
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)


# =====================================
# 3. SCALE FEATURES
# =====================================
X_scaled = scaler.transform(X)


# =====================================
# 4. AUTOENCODER ANOMALY DETECTION
# =====================================
reconstruction_error = np.mean(
    np.square(X_scaled - autoencoder.predict(X_scaled, verbose=0)),
    axis=1
)

# Threshold (must match training logic)
threshold = np.percentile(reconstruction_error, 95)

ae_prediction = (reconstruction_error > threshold).astype(int)


# =====================================
# 5. ISOLATION FOREST ANOMALY DETECTION
# =====================================
if_prediction = iso_forest.predict(X_scaled)
if_prediction = np.where(if_prediction == -1, 1, 0)


# =====================================
# 6. HYBRID DECISION (OR RULE)
# =====================================
final_prediction = np.logical_or(
    ae_prediction,
    if_prediction
).astype(int)


# =====================================
# 7. SAVE RESULTS
# =====================================
df['AE_Anomaly'] = ae_prediction
df['IF_Anomaly'] = if_prediction
df['Final_Attack_Prediction'] = final_prediction

OUTPUT_FILE = "unsupervised_ids_output.csv"
df[['Label', 'Final_Attack_Prediction']].to_csv(OUTPUT_FILE, index=False)


# =====================================
# 8. SUMMARY
# =====================================
print("[SUCCESS] IDS detection completed")
print("[INFO] Output saved to:", OUTPUT_FILE)

print("\n[SUMMARY]")
print("Total Samples  :", len(final_prediction))
print("Detected Attacks:", final_prediction.sum())
print("Actual Attacks :", y_true.sum())
