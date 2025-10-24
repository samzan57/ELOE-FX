# backtests/labeling.py
import pandas as pd
import numpy as np

def make_tp_sl_levels(row, atr_k: float, min_stop_pips: float, rr: float, pip_size: float):
    atr = row.get("atr", np.nan)
    if not np.isfinite(atr):
        stop_pips = min_stop_pips
    else:
        stop_pips = max((atr / pip_size) * atr_k, min_stop_pips)
    stop_dist = stop_pips * pip_size
    entry = float(row["close"])
    tp_buy  = entry + rr * stop_dist
    sl_buy  = entry - stop_dist
    tp_sell = entry - rr * stop_dist
    sl_sell = entry + stop_dist
    return entry, stop_dist, tp_buy, sl_buy, tp_sell, sl_sell

def label_tp_before_sl(df: pd.DataFrame, horizon: int, atr_k: float, min_stop_pips: float, rr: float, pip_size: float):
    """
    Crée une colonne 'y_buy' et 'y_sell' (1 si TP avant SL dans le prochain 'horizon' bars, sinon 0).
    Hypothèse: entrée à la clôture de la bougie courante.
    """
    df = df.copy()
    y_buy  = np.zeros(len(df), dtype=np.float32)
    y_sell = np.zeros(len(df), dtype=np.float32)

    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    close = df["close"].to_numpy()

    for i in range(len(df) - horizon):
        entry, stop_dist, tp_buy, sl_buy, tp_sell, sl_sell = make_tp_sl_levels(df.iloc[i], atr_k, min_stop_pips, rr, pip_size)

        # Parcourt la fenêtre suivante pour voir ce qui touche en premier
        h_slice = highs[i+1:i+1+horizon]
        l_slice = lows[i+1:i+1+horizon]

        # BUY: TP si high >= tp_buy ; SL si low <= sl_buy
        hit_tp_idx = np.argmax(h_slice >= tp_buy) if np.any(h_slice >= tp_buy) else 0
        hit_sl_idx = np.argmax(l_slice <= sl_buy) if np.any(l_slice <= sl_buy) else 0

        if hit_tp_idx and hit_sl_idx:
            y_buy[i] = 1.0 if hit_tp_idx <= hit_sl_idx else 0.0
        elif hit_tp_idx:
            y_buy[i] = 1.0
        elif hit_sl_idx:
            y_buy[i] = 0.0
        else:
            y_buy[i] = 0.0

        # SELL: TP si low <= tp_sell ; SL si high >= sl_sell
        hit_tp_idx = np.argmax(l_slice <= tp_sell) if np.any(l_slice <= tp_sell) else 0
        hit_sl_idx = np.argmax(h_slice >= sl_sell) if np.any(h_slice >= sl_sell) else 0

        if hit_tp_idx and hit_sl_idx:
            y_sell[i] = 1.0 if hit_tp_idx <= hit_sl_idx else 0.0
        elif hit_tp_idx:
            y_sell[i] = 1.0
        elif hit_sl_idx:
            y_sell[i] = 0.0
        else:
            y_sell[i] = 0.0

    df["y_buy"]  = y_buy
    df["y_sell"] = y_sell
    return df
