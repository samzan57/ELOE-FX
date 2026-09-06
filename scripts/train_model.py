#!/usr/bin/env python
# scripts/train_model.py
"""
Entraînement complet des modèles LightGBM.

Usage :
  python scripts/train_model.py --pair EURUSD --data data/historical/EURUSD_1min.csv

Le script effectue dans l'ordre :
  1. Chargement et nettoyage des données historiques
  2. Construction des features (~55 colonnes)
  3. Labeling avec TP/SL réaliste (délai d'exécution inclus)
  4. Walk-forward cross-validation (5 folds, gap 1 semaine) → métriques OOS
  5. Entraînement du modèle final sur 100 % des données
  6. Sauvegarde dans models/

Données attendues : CSV avec colonnes date,open,high,low,close
  (le script accepte aussi le format Dukascopy et MetaTrader)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Ajouter la racine du projet au PYTHONPATH ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.train import train_walk_forward, train_final_model
from backtests.metrics import print_metrics
from config import settings as S


# ─────────────────────────────────────────────
#  Chargement des données
# ─────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalisation des noms de colonnes (insensible à la casse)
    df.columns = [c.strip().lower() for c in df.columns]

    # Colonne datetime
    for col in ["datetime", "date", "time", "timestamp"]:
        if col in df.columns:
            df["datetime"] = pd.to_datetime(df[col])
            break
    else:
        raise ValueError("Colonne datetime introuvable dans le CSV.")

    df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Colonnes OHLC obligatoires
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante : {c}")

    df = df[["open", "high", "low", "close"] + (
        ["volume"] if "volume" in df.columns else []
    )].copy()

    df = df.sort_index().dropna()
    print(f"  Données chargées : {len(df):,} barres  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair",      default="EURUSD")
    parser.add_argument("--data",      default="data/historical/EURUSD_1min.csv")
    parser.add_argument("--horizon",   type=int,   default=120,   help="Horizon labeling (bougies)")
    parser.add_argument("--atr_k",     type=float, default=S.ATR_K)
    parser.add_argument("--min_sl",    type=float, default=S.MIN_STOP_PIPS)
    parser.add_argument("--rr",        type=float, default=S.RR)
    parser.add_argument("--folds",     type=int,   default=5)
    parser.add_argument("--train_end", default=None,
                        help="Date de fin d'entraînement YYYY-MM-DD (ex: 2025-10-01). "
                             "Les données après cette date sont réservées pour le backtest OOS.")
    args = parser.parse_args()

    config = {
        "horizon":          args.horizon,
        "atr_k":            args.atr_k,
        "min_stop_pips":    args.min_sl,
        "rr":               args.rr,
        # Paramètres signaux (pour le filtre pre-training)
        "signal_vwap_z":    S.SIGNAL_VWAP_Z,
        "signal_rsi_lo":    S.SIGNAL_RSI_LO,
        "signal_rsi_hi":    S.SIGNAL_RSI_HI,
        "adx_max":          S.ADX_MAX,
        "adx_min_momentum": S.ADX_MIN_MOMENTUM,
    }

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERREUR] Fichier introuvable : {data_path}")
        print("         Lance d'abord : python scripts/download_fx_csv.py")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"  ELOE-FX — Entraînement LightGBM  ({args.pair})")
    print("="*60)

    df_full = load_csv(str(data_path))

    # ── Séparation train / test OOS ───────────────────────────────────────
    if args.train_end:
        import pandas as pd
        cutoff = pd.Timestamp(args.train_end, tz="UTC")
        df      = df_full[df_full.index <  cutoff].copy()
        df_test = df_full[df_full.index >= cutoff].copy()
        print(f"\n  Train : {df.index[0].date()} → {df.index[-1].date()}  ({len(df):,} barres)")
        print(f"  Test  : {df_test.index[0].date()} → {df_test.index[-1].date()}  ({len(df_test):,} barres)  ← jamais vues")
        # Sauvegarde la date de coupure pour que run_backtest.py puisse l'utiliser
        cutoff_file = Path("data/logs/train_cutoff.txt")
        cutoff_file.parent.mkdir(parents=True, exist_ok=True)
        cutoff_file.write_text(str(cutoff.date()))
        print(f"  Date de coupure sauvegardée → {cutoff_file}")
    else:
        df = df_full
        print("\n  [INFO] Pas de --train_end : entraînement sur 100% des données.")
        print("         Pour un backtest honnête, utilise --train_end YYYY-MM-DD")

    for side in ["buy", "sell"]:
        print(f"\n── Side : {side.upper()} ──────────────────────────────────────")

        # Walk-forward (validation — sur les données d'entraînement seulement)
        _, wf_summary = train_walk_forward(
            df_raw  = df,
            config  = config,
            side    = side,
            n_folds = args.folds,
            verbose = True,
        )
        print(f"\n  Walk-forward summary ({args.folds} folds OOS) :")
        print(wf_summary.to_string(index=False))

        # Décision : EV moyen positif ?
        if wf_summary.empty or "ev" not in wf_summary.columns:
            print("\n  [INFO] Pas de validation walk-forward (dataset signal trop petit).")
            print("  Le modèle final sera entraîné sur toutes les barres signal.")
        else:
            mean_ev = wf_summary["ev"].mean()
            if mean_ev <= 0:
                print(f"\n  [ATTENTION] EV moyen OOS = {mean_ev:.4f} ≤ 0")
                print("  Le modèle n'a pas d'edge détectable sur ces données.")
            else:
                print(f"\n  [OK] EV moyen OOS = {mean_ev:+.4f} — edge détecté.")

        # Modèle final (entraîné uniquement sur df, pas df_full)
        save_path = S.MODEL_PATH if side == "buy" else S.SELL_MODEL_PATH
        print(f"\n  Entraînement final (données train seulement)…")
        train_final_model(df, config, side=side, save_path=save_path)

    print("\n" + "="*60)
    print("  Modèles prêts. Lance le backtest OOS :")
    if args.train_end:
        print(f"  python scripts/run_backtest.py --test_start {args.train_end}")
    else:
        print("  python scripts/run_backtest.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
