# features/technical.py
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  1. Returns multi-horizon
# ─────────────────────────────────────────────
def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    for h in [1, 5, 15, 30, 60, 240]:
        df[f"ret_{h}"] = np.log(df["close"] / df["close"].shift(h))
    return df


# ─────────────────────────────────────────────
#  2. Volatilité
# ─────────────────────────────────────────────
def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift(1)).abs()
    lc  = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    df["atr_7"]  = tr.ewm(span=7,  adjust=False).mean()
    df["atr_14"] = tr.ewm(span=14, adjust=False).mean()

    log_r = np.log(df["close"] / df["close"].shift(1))
    ann   = np.sqrt(252 * 390)
    df["realized_vol_20"] = log_r.rolling(20).std() * ann
    df["realized_vol_60"] = log_r.rolling(60).std() * ann

    # régime volatilité : ratio ATR courant vs médiane 2 h
    med = df["atr_14"].rolling(120, min_periods=30).median()
    df["vol_ratio"] = df["atr_14"] / med.replace(0, np.nan)
    return df


# ─────────────────────────────────────────────
#  3. Oscillateurs
# ─────────────────────────────────────────────
def _add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    delta = df["close"].diff()
    for p in [7, 14, 21]:
        gain = delta.clip(lower=0).ewm(span=p, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(span=p, adjust=False).mean()
        df[f"rsi_{p}"] = 100 - 100 / (1 + gain / (loss + 1e-10))
    return df


def _add_stochastic(df: pd.DataFrame, k=14, d=3) -> pd.DataFrame:
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    df["stoch_k"] = 100 * (df["close"] - lo) / (hi - lo + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(d).mean()
    return df


def _add_cci(df: pd.DataFrame, p=14) -> pd.DataFrame:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(p).mean()
    mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_14"] = (tp - sma) / (0.015 * mad + 1e-10)
    return df


def _add_williams_r(df: pd.DataFrame, p=14) -> pd.DataFrame:
    hi = df["high"].rolling(p).max()
    lo = df["low"].rolling(p).min()
    df["williams_r"] = -100 * (hi - df["close"]) / (hi - lo + 1e-10)
    return df


# ─────────────────────────────────────────────
#  4. MACD
# ─────────────────────────────────────────────
def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    atr   = df.get("atr_14", pd.Series(np.nan, index=df.index))
    norm  = atr.replace(0, np.nan)
    df["macd_norm"]   = macd / norm
    df["macd_sig_norm"] = sig  / norm
    df["macd_hist_norm"] = (macd - sig) / norm
    return df


# ─────────────────────────────────────────────
#  5. Tendance (EMA, ADX)
# ─────────────────────────────────────────────
def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    e20  = df["close"].ewm(span=20,  adjust=False).mean()
    e50  = df["close"].ewm(span=50,  adjust=False).mean()
    e200 = df["close"].ewm(span=200, adjust=False).mean()

    df["ema_ratio_20_50"]  = e20 / e50 - 1
    df["ema_ratio_50_200"] = e50 / e200 - 1

    atr  = df.get("atr_14", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    df["price_vs_ema20"]  = (df["close"] - e20)  / atr
    df["price_vs_ema50"]  = (df["close"] - e50)  / atr
    df["price_vs_ema200"] = (df["close"] - e200) / atr

    # ADX
    pdm = df["high"].diff().clip(lower=0)
    ndm = (-df["low"].diff()).clip(lower=0)
    tr  = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    tr14  = tr.ewm(span=14, adjust=False).mean()
    pdi   = 100 * pdm.ewm(span=14, adjust=False).mean() / tr14.replace(0, np.nan)
    ndi   = 100 * ndm.ewm(span=14, adjust=False).mean() / tr14.replace(0, np.nan)
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-10)
    df["adx"]     = dx.ewm(span=14, adjust=False).mean()
    df["plus_di"]  = pdi
    df["minus_di"] = ndi
    return df


# ─────────────────────────────────────────────
#  6. Mean-reversion : VWAP + Bollinger
# ─────────────────────────────────────────────
def _add_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    # VWAP journalier (si volume absent → TWAP = moyenne des closes intraday)
    date_key = df.index.normalize() if hasattr(df.index, "normalize") else pd.Series(df.index).dt.normalize()
    if "volume" in df.columns and df["volume"].gt(0).any():
        vol = df["volume"]
    else:
        vol = pd.Series(1.0, index=df.index)

    pv      = df["close"] * vol
    cum_pv  = pv.groupby(df.index.date).transform("cumsum")
    cum_vol = vol.groupby(df.index.date).transform("cumsum")
    vwap    = cum_pv / cum_vol.replace(0, np.nan)
    df["vwap"] = vwap

    atr = df.get("atr_14", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    df["vwap_dist"] = (df["close"] - vwap) / atr

    # Bollinger 20
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    df["bb_position"] = (df["close"] - bb_lo) / (bb_up - bb_lo + 1e-10)
    df["bb_width"]    = (bb_up - bb_lo) / (bb_mid.replace(0, np.nan))

    # Distance normalisée au range session (pour signal breakout)
    df["range_high"] = df["high"].rolling(30).max()
    df["range_low"]  = df["low"].rolling(30).min()
    df["range_high_dist"] = (df["close"] - df["range_high"]) / atr
    df["range_low_dist"]  = (df["range_low"] - df["close"]) / atr
    return df


# ─────────────────────────────────────────────
#  7. Session / Temps
# ─────────────────────────────────────────────
def _add_session(df: pd.DataFrame, tz: str = "Europe/Paris") -> pd.DataFrame:
    idx = df.index.tz_convert(tz) if df.index.tzinfo is not None else df.index
    hour     = pd.Series(idx.hour + idx.minute / 60.0, index=df.index)
    weekday  = pd.Series(idx.weekday, index=df.index).astype(float)

    df["hour_sin"]     = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * hour / 24)
    df["weekday_sin"]  = np.sin(2 * np.pi * weekday / 5)
    df["weekday_cos"]  = np.cos(2 * np.pi * weekday / 5)

    # Sessions (heure Paris)
    df["is_london"]  = ((hour >= 9)  & (hour < 12)).astype(float)   # 9-12 Paris = Londres matin
    df["is_ny"]      = ((hour >= 14) & (hour < 18)).astype(float)   # 14-18 Paris = NY
    df["is_overlap"] = ((hour >= 14) & (hour < 17)).astype(float)   # overlap = pic liquidité

    # Minutes depuis ouverture de chaque session
    df["min_since_london"] = np.clip((hour - 9)  * 60, 0, 480).where(hour >= 9,  0.0)
    df["min_since_ny"]     = np.clip((hour - 14) * 60, 0, 300).where(hour >= 14, 0.0)
    return df


# ─────────────────────────────────────────────
#  8. Microstructure de la bougie
# ─────────────────────────────────────────────
def _add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    body      = df["close"] - df["open"]

    df["bar_efficiency"] = body / bar_range
    df["upper_wick"]     = (df["high"]  - df[["open", "close"]].max(axis=1)) / bar_range
    df["lower_wick"]     = (df[["open", "close"]].min(axis=1) - df["low"])   / bar_range

    atr = df.get("atr_14", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    df["range_atr_ratio"] = (df["high"] - df["low"]) / atr

    ma20 = (df["high"] - df["low"]).rolling(20).mean().replace(0, np.nan)
    df["range_ratio"] = (df["high"] - df["low"]) / ma20
    return df


# ─────────────────────────────────────────────
#  9. Statistiques rolling des rendements
# ─────────────────────────────────────────────
def _add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    log_r = np.log(df["close"] / df["close"].shift(1))
    ann   = np.sqrt(252 * 390)

    for w in [20, 60]:
        mu  = log_r.rolling(w).mean()
        sig = log_r.rolling(w).std()
        df[f"rolling_sharpe_{w}"] = (mu / (sig + 1e-10)) * ann

    df["ret_skew_20"] = log_r.rolling(20).skew()

    # Autocorrélation lag-5 (signal de mean-reversion : autocorr < 0)
    df["ret_autocorr_5"] = log_r.rolling(30).apply(
        lambda x: pd.Series(x).autocorr(lag=5) if len(x) >= 6 else 0.0, raw=False
    )
    return df


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit l'ensemble complet des features (~55 colonnes) à partir d'un OHLC 1-min."""
    df = df.copy()
    df = _add_volatility(df)      # atr avant les autres
    df = _add_returns(df)
    df = _add_rsi(df)
    df = _add_stochastic(df)
    df = _add_cci(df)
    df = _add_williams_r(df)
    df = _add_macd(df)
    df = _add_trend(df)
    df = _add_mean_reversion(df)
    df = _add_session(df)
    df = _add_microstructure(df)
    df = _add_rolling_stats(df)
    return df


# Liste canonique des features pour le modèle ML
FEATURE_COLS = [
    # Momentum multi-horizon
    "ret_1", "ret_5", "ret_15", "ret_60",
    # Volatilité & régime
    "atr_14", "vol_ratio", "realized_vol_20",
    # Oscillateurs
    "rsi_7", "rsi_14", "stoch_k", "cci_14", "williams_r",
    # MACD
    "macd_norm", "macd_hist_norm",
    # Tendance
    "ema_ratio_20_50", "ema_ratio_50_200", "price_vs_ema20", "adx",
    # Mean-reversion
    "vwap_dist", "bb_position", "bb_width",
    # Session
    "hour_sin", "hour_cos", "is_overlap", "min_since_london",
    # Microstructure
    "bar_efficiency", "range_atr_ratio", "upper_wick", "lower_wick",
    # Stats rolling
    "rolling_sharpe_20", "ret_skew_20", "ret_autocorr_5",
]
