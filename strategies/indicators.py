# strategies/indicators.py
import numpy as np
import pandas as pd

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """
    RSI (style Wilder approx via EMA).
    series: prix de clôture (float)
    n: fenêtre RSI
    Retourne une série RSI avec NaN sur la période d'initialisation.
    """
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals

def compute_indicators(df: pd.DataFrame, atr_len: int = 14, rsi_len: int = 14, range_lookback: int = 30) -> pd.DataFrame:
    """
    Ajoute au DataFrame les colonnes:
      - atr: Average True Range simple (moyenne mobile sur 'atr_len')
      - rsi: RSI(rsi_len)
      - range_high / range_low: max/min rolling sur 'range_lookback'
    Exige colonnes: ['open','high','low','close'] (float).
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # S'assurer que les colonnes sont numériques
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # True Range
    prev_close = out["close"].shift()
    tr = np.maximum(out["high"] - out["low"],
                    np.maximum((out["high"] - prev_close).abs(),
                               (out["low"] - prev_close).abs()))
    out["atr"] = tr.rolling(atr_len, min_periods=atr_len).mean()

    # RSI
    out["rsi"] = rsi(out["close"], rsi_len)

    # Range (breakout)
    out["range_high"] = out["high"].rolling(range_lookback).max()
    out["range_low"]  = out["low"].rolling(range_lookback).min()

    return out
