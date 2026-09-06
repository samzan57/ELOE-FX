# backtests/engine.py
"""
Moteur de backtest réaliste.

Règles d'exécution :
  - Signal généré à la clôture de la bougie i
  - Entrée à l'OPEN de la bougie i+1 (+ spread + slippage)
  - Sortie : TP, SL ou expiration à horizon bars
  - Maximum 1 position à la fois
  - Cooldown après perte : COOLDOWN_BARS bougies
"""

import numpy as np
import pandas as pd

from features.technical import build_all_features
from strategies.signals import compute_signals
from backtests.metrics import compute_metrics

PIP_SIZE       = 0.0001
SPREAD_PIPS    = 0.20
SLIPPAGE_PIPS  = 0.05
COOLDOWN_BARS  = 15    # bougies de repos après un SL


def _apply_cost(price: float, direction: str) -> float:
    cost = (SPREAD_PIPS + SLIPPAGE_PIPS) * PIP_SIZE
    return price + cost if direction == "long" else price - cost


def run_backtest(
    df_raw: pd.DataFrame,
    model_buy,
    model_sell,
    config: dict,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Retourne (trades_df, metrics_dict).

    config keys :
      atr_k, min_stop_pips, rr, horizon,
      signal_vwap_z  (seuil distance VWAP),
      signal_rsi_lo / signal_rsi_hi,
      adx_max        (filtre : pas de mean-reversion si tendance forte)
    """
    df = build_all_features(df_raw.copy())
    df = df.dropna(subset=["atr_14", "vwap_dist", "rsi_14", "adx"]).copy()
    df = df.reset_index()   # index 0..n pour accès numpy rapide

    signals = compute_signals(df, config)   # 'long' | 'short' | 'flat' par barre

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    open_p = df["open"].values
    atr    = df["atr_14"].values
    times  = df["index"].values if "index" in df.columns else df.index.values

    proba_buy  = model_buy.predict_proba(df)
    proba_sell = model_sell.predict_proba(df)

    trades        = []
    in_position   = False
    cooldown_left = 0
    pos           = {}    # trade en cours

    n       = len(df)
    horizon = config["horizon"]

    for i in range(1, n):
        # ── Mise à jour de la position ouverte ───────────────────────────
        if in_position:
            pos["bars_held"] += 1
            d = pos["direction"]

            hit_sl = low[i]  <= pos["sl"] if d == "long" else high[i] >= pos["sl"]
            hit_tp = high[i] >= pos["tp"] if d == "long" else low[i]  <= pos["tp"]

            # Priorité SL si les deux touchés dans la même bougie
            if hit_sl:
                result = "sl"; exit_p = pos["sl"]
            elif hit_tp:
                result = "tp"; exit_p = pos["tp"]
            elif pos["bars_held"] >= horizon:
                result = "expired"; exit_p = close[i]
            else:
                continue    # position encore ouverte

            pnl_pips = (exit_p - pos["entry"]) / PIP_SIZE if d == "long" \
                  else (pos["entry"] - exit_p) / PIP_SIZE

            trades.append({**pos,
                           "exit_time": times[i],
                           "exit":      exit_p,
                           "result":    result,
                           "pnl_pips":  pnl_pips})
            in_position   = False
            cooldown_left = COOLDOWN_BARS if result == "sl" else 0
            if verbose:
                print(f"  [{times[i]}] {d} {result} | pnl={pnl_pips:+.1f}p")
            continue

        # ── Cooldown ────────────────────────────────────────────────────
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        # ── Nouveau signal (sur bougie i-1, exécution à open[i]) ────────
        sig = signals[i - 1]
        if sig == "flat":
            continue

        if sig == "long"  and proba_buy[i - 1]  < model_buy.threshold:
            continue
        if sig == "short" and proba_sell[i - 1] < model_sell.threshold:
            continue

        atr_val   = atr[i - 1]
        stop_dist = max(atr_val * config["atr_k"],
                        config["min_stop_pips"] * PIP_SIZE)
        entry     = _apply_cost(open_p[i], sig)

        if sig == "long":
            tp = entry + config["rr"] * stop_dist
            sl = entry - stop_dist
        else:
            tp = entry - config["rr"] * stop_dist
            sl = entry + stop_dist

        pos = {
            "direction":  sig,
            "entry_time": times[i],
            "entry":      entry,
            "tp":         tp,
            "sl":         sl,
            "bars_held":  0,
        }
        in_position = True

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
        "direction", "entry_time", "exit_time", "entry", "exit",
        "tp", "sl", "result", "pnl_pips", "bars_held"
    ])

    metrics = compute_metrics(trades_df, config)
    return trades_df, metrics
