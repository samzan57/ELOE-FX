# models/proba_model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "rsi",
    "atr",
    "range_high_dist",   # (close - range_high) / atr
    "range_low_dist",    # (range_low - close) / atr
    "vol_regime",        # atr / rolling_median_atr
    "hour_sin", "hour_cos"
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # distances normalisées par ATR (évite l’échelle)
    out["range_high_dist"] = (out["close"] - out["range_high"]) / out["atr"]
    out["range_low_dist"]  = (out["range_low"] - out["close"]) / out["atr"]
    # régime de volatilité
    out["atr_med"] = out["atr"].rolling(60, min_periods=30).median()
    out["vol_regime"] = out["atr"] / out["atr_med"]

    # Encodage heure (sin/cos)
    idx = out.index.tz_convert("Europe/Paris") if out.index.tz is not None else out.index
    h = pd.Series(idx.hour, index=out.index).astype(float)
    out["hour_sin"] = np.sin(2*np.pi*h/24)
    out["hour_cos"] = np.cos(2*np.pi*h/24)

    return out

def fit_model(df: pd.DataFrame, side: str = "buy", model_type: str = "logit"):
    """
    side: 'buy' ou 'sell' -> cible 'y_buy' ou 'y_sell'
    model_type: 'logit' ou 'rf'
    """
    y_col = "y_buy" if side == "buy" else "y_sell"
    data = build_features(df)
    data = data.dropna(subset=FEATURES + [y_col])

    X = data[FEATURES].values
    y = data[y_col].values

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    else:
        model = LogisticRegression(max_iter=500, class_weight="balanced")

    model.fit(X, y)
    return model

def predict_proba(model, df_feat: pd.DataFrame):
    X = df_feat[FEATURES].fillna(0).values
    proba = model.predict_proba(X)[:, 1]
    return proba

def save_model(model, path: str):
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
