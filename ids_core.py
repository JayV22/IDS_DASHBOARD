import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models once
autoencoder = load_model("autoencoder_ids_cicids.h5", compile=False)
iso_forest = joblib.load("isolation_forest_ids.pkl")
scaler = joblib.load("scaler_ids.pkl")


def detect_attack(X):
    """
    Input  : DataFrame (numeric features)
    Output : final_pred (0/1), severity (0–100)
    """

    X_scaled = scaler.transform(X)

    # Autoencoder
    recon_error = np.mean(
        np.square(X_scaled - autoencoder.predict(X_scaled, verbose=0)),
        axis=1
    )

    threshold = np.percentile(recon_error, 95)
    ae_pred = (recon_error > threshold).astype(int)

    # Isolation Forest
    if_pred = iso_forest.predict(X_scaled)
    if_pred = np.where(if_pred == -1, 1, 0)

    # Hybrid decision
    final_pred = np.logical_or(ae_pred, if_pred).astype(int)

    # Severity score
    severity = np.clip(
        (recon_error / np.max(recon_error)) * 100,
        0, 100
    )

    return final_pred, severity
