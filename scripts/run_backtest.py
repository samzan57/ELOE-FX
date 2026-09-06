#!/usr/bin/env python
# scripts/run_backtest.py
"""
Backtest complet avec les modèles entraînés.

Usage :
  python scripts/run_backtest.py --data data/historical/EURUSD_1min.csv

Affiche les métriques et génère un CSV de trades dans data/logs/.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # pas besoin de display GUI
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.lgbm_model import LGBMTradingModel
from backtests.engine import run_backtest
from backtests.metrics import print_metrics
from config import settings as S


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["datetime", "date", "time", "timestamp"]:
        if col in df.columns:
            df["datetime"] = pd.to_datetime(df[col])
            break
    df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])].copy()
    return df.sort_index().dropna()


def plot_equity(trades: pd.DataFrame, output: str):
    if trades.empty:
        return
    equity = trades["pnl_pips"].cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(equity.values, color="steelblue", linewidth=1.2)
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].fill_between(range(len(equity)), equity.values, 0,
                         where=equity.values >= 0, alpha=0.3, color="green")
    axes[0].fill_between(range(len(equity)), equity.values, 0,
                         where=equity.values < 0, alpha=0.3, color="red")
    axes[0].set_title("Courbe de capital (pips cumulés)")
    axes[0].set_ylabel("Pips")

    colors = ["green" if p > 0 else "red" for p in trades["pnl_pips"]]
    axes[1].bar(range(len(trades)), trades["pnl_pips"].values, color=colors, alpha=0.7)
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_title("P&L par trade (pips)")
    axes[1].set_ylabel("Pips")
    axes[1].set_xlabel("Trade #")

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    print(f"  Graphique sauvegardé → {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/historical/EURUSD_1min.csv")
    parser.add_argument("--horizon",    type=int,   default=120)
    parser.add_argument("--atr_k",      type=float, default=S.ATR_K)
    parser.add_argument("--min_sl",     type=float, default=S.MIN_STOP_PIPS)
    parser.add_argument("--rr",         type=float, default=S.RR)
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--plot",       default="data/logs/backtest_equity.png")
    parser.add_argument("--test_start", default=None,
                        help="Date de début du backtest OOS YYYY-MM-DD. "
                             "Seules les données après cette date sont utilisées.")
    args = parser.parse_args()

    config = {
        "horizon":         args.horizon,
        "atr_k":           args.atr_k,
        "min_stop_pips":   args.min_sl,
        "rr":              args.rr,
        "signal_vwap_z":   getattr(S, "SIGNAL_VWAP_Z",     1.2),
        "signal_rsi_lo":   getattr(S, "SIGNAL_RSI_LO",    40.0),
        "signal_rsi_hi":   getattr(S, "SIGNAL_RSI_HI",    60.0),
        "adx_max":         getattr(S, "ADX_MAX",           28.0),
        "adx_min_momentum":getattr(S, "ADX_MIN_MOMENTUM",  18.0),
    }

    print("\n" + "="*60)
    print("  ELOE-FX — Backtest réaliste")
    print("="*60)

    # Chargement modèles
    for p in [S.MODEL_PATH, S.SELL_MODEL_PATH]:
        if not Path(p).exists():
            print(f"[ERREUR] Modèle manquant : {p}")
            print("         Lance d'abord : python scripts/train_model.py")
            sys.exit(1)

    model_buy  = LGBMTradingModel.load(S.MODEL_PATH)
    model_sell = LGBMTradingModel.load(S.SELL_MODEL_PATH)
    print(f"  Modèles chargés — buy thr={model_buy.threshold:.4f}  sell thr={model_sell.threshold:.4f}")

    # Données
    df = load_csv(args.data)

    # Filtre OOS : on ne backteste que sur la période non vue pendant l'entraînement
    if args.test_start:
        import pandas as pd
        cutoff = pd.Timestamp(args.test_start, tz="UTC")
        df = df[df.index >= cutoff].copy()
        print(f"  Mode OOS — données à partir du {args.test_start}")
    else:
        # Vérifier si une date de coupure a été sauvegardée par train_model.py
        cutoff_file = Path("data/logs/train_cutoff.txt")
        if cutoff_file.exists():
            import pandas as pd
            saved = cutoff_file.read_text().strip()
            cutoff = pd.Timestamp(saved, tz="UTC")
            df = df[df.index >= cutoff].copy()
            print(f"  Mode OOS auto — données à partir du {saved} (lu depuis {cutoff_file})")

    print(f"  Données : {len(df):,} barres  ({df.index[0].date()} → {df.index[-1].date()})")

    # Backtest
    print("\n  Simulation en cours…")
    trades, metrics = run_backtest(df, model_buy, model_sell, config, verbose=args.verbose)

    print_metrics(metrics)

    if not trades.empty:
        out_csv = "data/logs/backtest_trades.csv"
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out_csv, index=False)
        print(f"  Trades exportés → {out_csv}")
        plot_equity(trades, args.plot)

        # Analyse par direction
        print("\n  ── Par direction ──────────────────────────────────")
        for d in ["long", "short"]:
            sub = trades[trades["direction"] == d]
            if sub.empty:
                continue
            wr  = (sub["pnl_pips"] > 0).mean()
            tot = sub["pnl_pips"].sum()
            print(f"  {d.upper():5s}  n={len(sub):4d}  WR={wr:.1%}  total={tot:+.1f} pips")

        # Analyse par résultat
        print("\n  ── Par résultat ───────────────────────────────────")
        for r in ["tp", "sl", "expired"]:
            sub = trades[trades["result"] == r]
            if sub.empty:
                continue
            print(f"  {r:8s}  n={len(sub):4d}  moy={sub['pnl_pips'].mean():+.1f} pips")
    else:
        print("\n  Aucun trade généré — vérifie les paramètres de signal.")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
