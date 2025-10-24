# backtests/optimizer.py
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from strategies.indicators import compute_indicators
from strategies.utils import pip_size
from backtests.labeling import label_tp_before_sl
from models.proba_model import fit_model, build_features, predict_proba

def expected_value(p, rr, fee=0.0):
    # EV (unités de risque) = p*rr - (1-p) - fee
    return p*rr - (1-p) - fee

def walk_forward_split(df, n_splits=4):
    n = len(df)
    fold = n // (n_splits+1)
    for i in range(n_splits):
        train = df.iloc[: (i+1)*fold]
        test  = df.iloc[(i+1)*fold : (i+2)*fold]
        yield train, test

def evaluate_params(df_raw: pd.DataFrame, atr_k, min_stop_pips, rsi_min, rr, horizon=120, fee=0.0, side="buy"):
    """
    df_raw: OHLC 1-min indexé en datetime tz-aware
    """
    df = compute_indicators(df_raw, atr_len=14, rsi_len=14, range_lookback=30)
    ps = pip_size("EURUSD")  # adapte si multi-paires
    df = label_tp_before_sl(df, horizon=horizon, atr_k=atr_k, min_stop_pips=min_stop_pips, rr=rr, pip_size=ps)

    # règle de signal "breakout + RSI" pour filtrer
    cond_buy  = (df["close"] > df["range_high"].shift(1)) & (df["rsi"] >= rsi_min)
    cond_sell = (df["close"] < df["range_low"].shift(1))  & (df["rsi"] <= (100 - rsi_min))

    results = []
    for train, test in walk_forward_split(df.dropna()):
        # fit proba sur train
        model = fit_model(train[cond_buy if side=="buy" else cond_sell], side=side, model_type="logit")

        # proba sur test
        test_feat = build_features(test)
        proba = predict_proba(model, test_feat.loc[test.index])

        # threshold optimal (théorique) + marge
        thr = 1.0/(1.0+rr) + 0.05

        sig_mask = cond_buy.loc[test.index] if side=="buy" else cond_sell.loc[test.index]
        take = test["y_buy" if side=="buy" else "y_sell"].astype(float)

        mask = sig_mask & (proba >= thr)
        p = proba[mask]
        y = take[mask]

        if p.size == 0:
            results.append((0, 0, 0, 0))
            continue

        ev_series = expected_value(p, rr, fee=fee)
        pnl_units = (y*rr - (1-y)) - fee  # par trade
        results.append((np.mean(pnl_units), np.mean(ev_series), y.mean(), mask.sum()))

    if not results:
        return {"ev":0,"pnl":0,"hit":0,"trades":0}

    pnl_avg = np.mean([r[0] for r in results])
    ev_avg  = np.mean([r[1] for r in results])
    hit_avg = np.mean([r[2] for r in results])
    n_trades = int(np.sum([r[3] for r in results]))
    return {"ev": ev_avg, "pnl": pnl_avg, "hit": hit_avg, "trades": n_trades}

def grid_search(df_raw, grid, horizon=120, fee=0.0, side="buy"):
    best = None
    best_score = -1e9
    for atr_k, min_stop_pips, rsi_min, rr in itertools.product(
        grid["ATR_K"], grid["MIN_STOP_PIPS"], grid["RSI_MIN"], grid["RR"]
    ):
        metrics = evaluate_params(df_raw, atr_k, min_stop_pips, rsi_min, rr,
                                  horizon=horizon, fee=fee, side=side)
        score = metrics["ev"]  # ou combine ev + pnl
        if score > best_score:
            best_score = score
            best = {
                "ATR_K": atr_k,
                "MIN_STOP_PIPS": min_stop_pips,
                "RSI_MIN": rsi_min,
                "RR": rr,
                **metrics
            }
    return best
