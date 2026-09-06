# backtests/labeling.py
"""
Labeling réaliste pour classification binaire.

Corrections vs v1 :
  - Entrée au prix OPEN de la bougie i+1 (pas au close de i)
  - Spread + slippage modélisés
  - Bug argmax(0) corrigé → np.where()[0] utilisé à la place
  - Retourne aussi stop_dist pour le moteur de backtest
"""

import numpy as np
import pandas as pd


SPREAD_PIPS    = 0.20   # spread typique EURUSD au broker retail
SLIPPAGE_PIPS  = 0.05   # impact marché estimé


def label_tp_before_sl(
    df: pd.DataFrame,
    horizon: int,
    atr_k: float,
    min_stop_pips: float,
    rr: float,
    pip_sz: float,
    entry_delay: int = 1,       # bougies de délai avant exécution
) -> pd.DataFrame:
    """
    Pour chaque bougie i :
      - Entry price = open[i + entry_delay] ± (spread/2 + slippage)
      - TP / SL calculés via ATR × atr_k (minimum min_stop_pips)
      - y_buy  = 1 si TP long atteint avant SL dans les 'horizon' bougies
      - y_sell = 1 si TP short atteint avant SL dans les 'horizon' bougies
    """
    df = df.copy()
    n  = len(df)

    highs  = df["high"].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)
    opens  = df["open"].to_numpy(dtype=np.float64) if "open" in df.columns else df["close"].to_numpy(dtype=np.float64)

    # ATR : préfère atr_14 si disponible
    if "atr_14" in df.columns:
        atrs = df["atr_14"].to_numpy(dtype=np.float64)
    elif "atr" in df.columns:
        atrs = df["atr"].to_numpy(dtype=np.float64)
    else:
        atrs = np.full(n, np.nan)

    spread    = SPREAD_PIPS   * pip_sz
    slippage  = SLIPPAGE_PIPS * pip_sz

    y_buy  = np.zeros(n, dtype=np.float32)
    y_sell = np.zeros(n, dtype=np.float32)

    limit = n - horizon - entry_delay
    for i in range(limit):
        atr_val = atrs[i]
        if not np.isfinite(atr_val) or atr_val <= 0:
            continue

        stop_pips = max((atr_val / pip_sz) * atr_k, min_stop_pips)
        stop_dist = stop_pips * pip_sz

        entry_idx   = i + entry_delay
        entry_long  = opens[entry_idx] + spread / 2.0 + slippage
        entry_short = opens[entry_idx] - spread / 2.0 - slippage

        tp_buy  = entry_long  + rr * stop_dist
        sl_buy  = entry_long  - stop_dist
        tp_sell = entry_short - rr * stop_dist
        sl_sell = entry_short + stop_dist

        s = entry_idx + 1
        e = min(s + horizon, n)
        h_sl = highs[s:e]
        l_sl = lows[s:e]

        # ── BUY ──────────────────────────────────
        tp_hits = np.where(h_sl >= tp_buy)[0]
        sl_hits = np.where(l_sl <= sl_buy)[0]

        if len(tp_hits) and len(sl_hits):
            y_buy[i] = 1.0 if tp_hits[0] < sl_hits[0] else 0.0
        elif len(tp_hits):
            y_buy[i] = 1.0
        # else → 0 (SL ou expiré)

        # ── SELL ─────────────────────────────────
        tp_hits = np.where(l_sl <= tp_sell)[0]
        sl_hits = np.where(h_sl >= sl_sell)[0]

        if len(tp_hits) and len(sl_hits):
            y_sell[i] = 1.0 if tp_hits[0] < sl_hits[0] else 0.0
        elif len(tp_hits):
            y_sell[i] = 1.0

    df["y_buy"]  = y_buy
    df["y_sell"] = y_sell
    return df
