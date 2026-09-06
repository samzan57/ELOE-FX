# backtests/metrics.py
"""
Métriques de performance institutionnelles.
Inspiré des standards des hedge funds (Sharpe, Calmar, Sortino, MAR).
"""

import numpy as np
import pandas as pd


def compute_metrics(trades: pd.DataFrame, config: dict) -> dict:
    """
    Calcule les métriques à partir d'un DataFrame de trades.
    Attend les colonnes : pnl_pips, result, direction, entry_time, exit_time.
    """
    if trades.empty or "pnl_pips" not in trades.columns:
        return _empty_metrics()

    rr   = config.get("rr", 2.0)
    pnl  = trades["pnl_pips"].values
    n    = len(pnl)

    wins   = pnl > 0
    losses = pnl < 0

    win_rate     = float(wins.mean())
    avg_win      = float(pnl[wins].mean())  if wins.any()   else 0.0
    avg_loss     = float(pnl[losses].mean()) if losses.any() else 0.0
    profit_factor = (pnl[wins].sum() / -pnl[losses].sum()) if losses.any() else np.inf

    # Espérance par trade en unités de risque (stop_dist)
    stop_pips = 1.0 / rr * (avg_win if wins.any() else 1.0)  # proxy
    ev_per_trade = win_rate * rr - (1 - win_rate)             # en multiples de risque

    # ── Courbe de capital (pips cumulés) ─────────────────────────────────
    equity = np.cumsum(pnl)

    # Max drawdown en pips
    running_max = np.maximum.accumulate(equity)
    drawdown    = equity - running_max
    max_dd      = float(drawdown.min())

    # ── Sharpe annualisé (sur rendements par trade) ───────────────────────
    if pnl.std() > 0:
        trades_per_year = _estimate_trades_per_year(trades)
        sharpe = float(pnl.mean() / pnl.std() * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # ── Sortino (pénalise uniquement la volatilité négative) ─────────────
    downside = pnl[pnl < 0]
    if len(downside) > 0 and downside.std() > 0:
        trades_per_year = _estimate_trades_per_year(trades)
        sortino = float(pnl.mean() / downside.std() * np.sqrt(trades_per_year))
    else:
        sortino = 0.0

    # ── Calmar (Sharpe vs drawdown) ───────────────────────────────────────
    total_pnl = float(equity[-1]) if n > 0 else 0.0
    calmar    = float(total_pnl / abs(max_dd)) if max_dd != 0 else 0.0

    # ── Résultats par direction ───────────────────────────────────────────
    long_trades  = trades[trades["direction"] == "long"]["pnl_pips"]  if "direction" in trades else pd.Series(dtype=float)
    short_trades = trades[trades["direction"] == "short"]["pnl_pips"] if "direction" in trades else pd.Series(dtype=float)

    # ── Durée moyenne ─────────────────────────────────────────────────────
    avg_bars = float(trades["bars_held"].mean()) if "bars_held" in trades else 0.0

    return {
        "n_trades":       n,
        "win_rate":       round(win_rate, 4),
        "avg_win_pips":   round(avg_win, 2),
        "avg_loss_pips":  round(avg_loss, 2),
        "profit_factor":  round(profit_factor, 3),
        "ev_per_trade":   round(ev_per_trade, 4),
        "total_pips":     round(total_pnl, 1),
        "max_dd_pips":    round(max_dd, 1),
        "sharpe":         round(sharpe, 3),
        "sortino":        round(sortino, 3),
        "calmar":         round(calmar, 3),
        "avg_bars_held":  round(avg_bars, 1),
        "n_long":         int(len(long_trades)),
        "n_short":        int(len(short_trades)),
        "wr_long":        round(float((long_trades > 0).mean()), 4)  if len(long_trades)  else 0.0,
        "wr_short":       round(float((short_trades > 0).mean()), 4) if len(short_trades) else 0.0,
    }


def _estimate_trades_per_year(trades: pd.DataFrame) -> float:
    """Estime le nombre de trades annualisé."""
    if len(trades) < 2 or "entry_time" not in trades.columns:
        return 252.0   # fallback
    try:
        t0 = pd.to_datetime(trades["entry_time"].iloc[0])
        t1 = pd.to_datetime(trades["entry_time"].iloc[-1])
        days = max((t1 - t0).days, 1)
        return len(trades) * 365.0 / days
    except Exception:
        return 252.0


def _empty_metrics() -> dict:
    keys = ["n_trades","win_rate","avg_win_pips","avg_loss_pips","profit_factor",
            "ev_per_trade","total_pips","max_dd_pips","sharpe","sortino","calmar",
            "avg_bars_held","n_long","n_short","wr_long","wr_short"]
    return {k: 0 for k in keys}


def print_metrics(metrics: dict):
    """Affichage formaté dans le terminal."""
    print("\n" + "="*50)
    print(f"  Trades         : {metrics['n_trades']}")
    print(f"  Win Rate       : {metrics['win_rate']:.1%}")
    print(f"  Profit Factor  : {metrics['profit_factor']:.2f}")
    print(f"  EV / trade     : {metrics['ev_per_trade']:+.4f}  (×risque)")
    print(f"  Total pips     : {metrics['total_pips']:+.1f}")
    print(f"  Max Drawdown   : {metrics['max_dd_pips']:.1f} pips")
    print(f"  Sharpe         : {metrics['sharpe']:.3f}")
    print(f"  Sortino        : {metrics['sortino']:.3f}")
    print(f"  Calmar         : {metrics['calmar']:.3f}")
    print(f"  Long / Short   : {metrics['n_long']} / {metrics['n_short']}")
    print("="*50 + "\n")
