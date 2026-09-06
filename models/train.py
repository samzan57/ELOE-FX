# models/train.py
"""
Pipeline d'entraînement avec Purged Walk-Forward Cross-Validation.

Principe :
  - On divise la série temporelle en N folds consécutifs.
  - Entre chaque train et test, on laisse un GAP (1 semaine) pour éviter
    le leakage de données adjacentes (labels qui se chevauchent).
  - On évalue les métriques sur chaque fold out-of-sample.
  - Le modèle final est entraîné sur TOUTES les données.

Signal pre-filter :
  - On labellise d'abord toutes les barres (les labels nécessitent les prix futurs).
  - Puis on filtre pour ne garder que les barres où le signal VWAP/momentum
    se déclencherait (long pour buy, short pour sell).
  - Cela aligne le training set avec la distribution réelle des trades.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from features.technical import build_all_features, FEATURE_COLS
from backtests.labeling import label_tp_before_sl
from models.lgbm_model import LGBMTradingModel
from strategies.signals import compute_signals


def get_pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair.upper() else 0.0001


def _filter_by_signal(df: pd.DataFrame, config: dict, side: str) -> pd.DataFrame:
    """Garde uniquement les barres où le signal correspondant se déclenche."""
    signals = compute_signals(df, config)
    target  = "long" if side == "buy" else "short"
    mask    = signals == target
    return df[mask].copy()


# ─────────────────────────────────────────────
#  Walk-forward avec purge (gap temporel)
# ─────────────────────────────────────────────
def purged_walk_forward(df: pd.DataFrame, n_folds: int = 5, gap_days: int = 7):
    """
    Génère (train_df, test_df) avec un gap temporel fixe.
    Utilise les timestamps réels — fonctionne même avec un dataset filtré sparse.
    """
    if len(df) < 30:
        return

    start = df.index[0]
    end   = df.index[-1]
    total_days  = (end - start).days
    fold_days   = total_days // (n_folds + 1)

    if fold_days < 1:
        fold_days = 1

    gap = pd.Timedelta(days=gap_days)

    for i in range(n_folds):
        train_end   = start + pd.Timedelta(days=(i + 1) * fold_days)
        test_start  = train_end + gap
        test_end    = test_start + pd.Timedelta(days=fold_days)

        train_df = df[df.index <  train_end].copy()
        test_df  = df[(df.index >= test_start) & (df.index < test_end)].copy()

        if len(train_df) < 20 or len(test_df) < 5:
            continue

        yield train_df, test_df


# ─────────────────────────────────────────────
#  Métriques d'un fold
# ─────────────────────────────────────────────
def _eval_fold(model: LGBMTradingModel, test_df: pd.DataFrame, rr: float) -> dict:
    y_col  = model._y_col
    valid  = test_df.dropna(subset=model.features + [y_col])
    if len(valid) == 0:
        return {"ev": 0, "win_rate": 0, "n_trades": 0, "pnl_per_trade": 0}

    proba  = model.predict_proba(valid)
    y_true = valid[y_col].values
    mask   = proba >= model.threshold

    if mask.sum() == 0:
        return {"ev": 0, "win_rate": 0, "n_trades": 0, "pnl_per_trade": 0}

    p_f  = proba[mask]
    y_f  = y_true[mask]
    ev   = float((p_f * rr - (1 - p_f)).mean())
    pnl  = float((y_f * rr - (1 - y_f)).mean())
    wr   = float(y_f.mean())
    return {"ev": ev, "win_rate": wr, "n_trades": int(mask.sum()), "pnl_per_trade": pnl}


# ─────────────────────────────────────────────
#  Entraînement + évaluation walk-forward
# ─────────────────────────────────────────────
def train_walk_forward(
    df_raw: pd.DataFrame,
    config: dict,
    side: str = "buy",
    n_folds: int = 5,
    verbose: bool = True,
) -> tuple[list[LGBMTradingModel], pd.DataFrame]:
    """
    config keys attendues :
      horizon, atr_k, min_stop_pips, rr
      + toutes les clés signal (signal_vwap_z, signal_rsi_lo, …)
    Retourne (liste de modèles par fold, DataFrame de métriques).
    """
    ps  = get_pip_size("EURUSD")
    df  = build_all_features(df_raw)
    df  = label_tp_before_sl(
        df,
        horizon       = config["horizon"],
        atr_k         = config["atr_k"],
        min_stop_pips = config["min_stop_pips"],
        rr            = config["rr"],
        pip_sz        = ps,
    )
    df = df.dropna(subset=FEATURE_COLS)

    # Filtrer sur les barres où le signal se déclenche
    df = _filter_by_signal(df, config, side)
    if verbose:
        print(f"  Barres après filtre signal ({side}): {len(df):,}")

    models, rows = [], []
    for fold, (train_df, test_df) in enumerate(purged_walk_forward(df, n_folds=n_folds)):
        model = LGBMTradingModel(side=side)
        model.set_ev_threshold(rr=config["rr"])
        model.fit(train_df, val_df=test_df)

        metrics = _eval_fold(model, test_df, rr=config["rr"])
        metrics["fold"] = fold
        rows.append(metrics)
        models.append(model)

        if verbose:
            print(
                f"  Fold {fold} | EV={metrics['ev']:+.4f} "
                f"WR={metrics['win_rate']:.3f} "
                f"Trades={metrics['n_trades']}"
            )

    if not rows:
        if verbose:
            print("  [ATTENTION] Aucun fold valide généré — dataset trop petit après filtre signal.")
            print(f"  Barres disponibles : {len(df)}. Entraînement final sans validation walk-forward.")
        empty = pd.DataFrame(columns=["fold", "ev", "win_rate", "n_trades", "pnl_per_trade"])
        return models, empty

    summary = pd.DataFrame(rows)
    if verbose:
        print(f"\n  ── Moyenne OOS ──")
        print(f"  EV moyen       : {summary['ev'].mean():+.4f}")
        print(f"  Win rate moyen : {summary['win_rate'].mean():.3f}")
        print(f"  Trades / fold  : {summary['n_trades'].mean():.0f}")

    return models, summary


# ─────────────────────────────────────────────
#  Modèle final (entraîné sur tout le dataset)
# ─────────────────────────────────────────────
def train_final_model(
    df_raw: pd.DataFrame,
    config: dict,
    side: str = "buy",
    save_path: str | None = None,
) -> LGBMTradingModel:
    """Entraîne sur 100 % des données pour le déploiement live."""
    ps  = get_pip_size("EURUSD")
    df  = build_all_features(df_raw)
    df  = label_tp_before_sl(
        df,
        horizon       = config["horizon"],
        atr_k         = config["atr_k"],
        min_stop_pips = config["min_stop_pips"],
        rr            = config["rr"],
        pip_sz        = ps,
    )
    df = df.dropna(subset=FEATURE_COLS)

    # Filtrer sur les barres où le signal se déclenche
    df = _filter_by_signal(df, config, side)
    print(f"  Barres d'entraînement ({side}) : {len(df):,}")

    model = LGBMTradingModel(side=side)
    model.set_ev_threshold(rr=config["rr"])
    model.fit(df)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f"  Modèle {side} sauvegardé → {save_path}")

    return model
