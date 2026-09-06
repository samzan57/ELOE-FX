# strategies/signals.py
"""
Génération de signaux de trading — deux régimes combinés :

1. VWAP Mean-Reversion (principal)
   Fondement académique : les traders institutionnels utilisent le VWAP comme
   référence d'exécution. Quand le prix s'éloigne significativement du VWAP,
   leurs ordres de rééquilibrage créent une force de rappel.
   Conditions : prix loin du VWAP + oscillateur confirmant la surchauffe +
                absence de tendance forte (ADX < seuil) + session liquide.

2. Session Momentum (secondaire — premières 90 min de chaque session)
   Fondement : l'établissement de la direction directionnelle en début de
   session London et NY est documenté (Breedon & Ranaldo 2013).
   Conditions : breakout du range des 30 dernières bougies + ADX croissant.

Le filtre ML (LightGBM) vient ensuite confirmer / rejeter chaque setup.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Paramètres par défaut (surchargés par config)
# ─────────────────────────────────────────────
_DEFAULTS = {
    "signal_vwap_z":   1.2,   # distance VWAP en multiples d'ATR
    "signal_rsi_lo":   40.0,  # RSI oversold threshold
    "signal_rsi_hi":   60.0,  # RSI overbought threshold
    "adx_max":         28.0,  # ne pas faire de mean-reversion si tendance forte
    "adx_min_momentum": 18.0, # ADX minimum pour signaux momentum
    "breakout_pips":    0.10, # buffer anti "touch-and-go" en pips
}


def compute_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Retourne un array de chaînes 'long' | 'short' | 'flat' de longueur len(df).
    df doit déjà contenir toutes les features (résultat de build_all_features).
    """
    cfg = {**_DEFAULTS, **config}
    n   = len(df)
    sig = np.full(n, "flat", dtype=object)

    # ── Récupération des séries ───────────────────────────────────────────
    vwap_dist     = _safe(df, "vwap_dist",      0.0)
    rsi           = _safe(df, "rsi_14",         50.0)
    adx           = _safe(df, "adx",            0.0)
    is_overlap    = _safe(df, "is_overlap",     0.0)
    is_london     = _safe(df, "is_london",      0.0)
    is_ny         = _safe(df, "is_ny",          0.0)
    min_london    = _safe(df, "min_since_london", 999.0)
    min_ny        = _safe(df, "min_since_ny",   999.0)
    rh_dist       = _safe(df, "range_high_dist", 0.0)
    rl_dist       = _safe(df, "range_low_dist",  0.0)
    bb_pos        = _safe(df, "bb_position",    0.5)
    vol_ratio     = _safe(df, "vol_ratio",      1.0)

    pip = 0.0001
    buf = cfg["breakout_pips"] * pip

    vz    = cfg["signal_vwap_z"]
    rlo   = cfg["signal_rsi_lo"]
    rhi   = cfg["signal_rsi_hi"]
    adxmx = cfg["adx_max"]
    adxmn = cfg["adx_min_momentum"]
    in_session = (is_overlap > 0.5) | (is_london > 0.5) | (is_ny > 0.5)

    # Filtre de tendance : EMA50 vs EMA200 (1-min proxy pour tendance de fond)
    trend_up   = _safe(df, "ema_ratio_50_200", 0.0) > 0    # EMA50 > EMA200 → tendance haussière
    trend_down = _safe(df, "ema_ratio_50_200", 0.0) < 0    # EMA50 < EMA200 → tendance baissière

    # ── 1. VWAP Mean-Reversion ────────────────────────────────────────────
    # Long : prix sous le VWAP, oversold, ET tendance de fond haussière
    mr_long  = (
        (vwap_dist < -vz) &          # loin sous le VWAP
        (rsi < rlo) &                # oversold
        (adx < adxmx) &              # pas de forte tendance baissière
        (bb_pos < 0.35) &            # bas des bandes de Bollinger
        (vol_ratio < 2.0) &          # pas de spike de vol (gap/news)
        trend_up &                   # tendance de fond haussière → mean-reversion vers le haut crédible
        in_session
    )
    # Short : prix au-dessus du VWAP, overbought, ET tendance de fond baissière
    mr_short = (
        (vwap_dist > vz) &
        (rsi > rhi) &
        (adx < adxmx) &
        (bb_pos > 0.65) &
        (vol_ratio < 2.0) &
        trend_down &                 # tendance de fond baissière → mean-reversion vers le bas crédible
        in_session
    )

    # ── 2. Session Momentum (breakout des premières 90 min) ───────────────
    early_london = (min_london > 0) & (min_london <= 90)
    early_ny     = (min_ny     > 0) & (min_ny     <= 90)
    early_session = early_london | early_ny

    # Long : breakout au-dessus du range des 30 dernières bougies
    mom_long  = (
        (rh_dist > buf / 0.0001 * 0.01) &  # close > range_high + buffer
        (adx > adxmn) &
        early_session &
        (vol_ratio >= 0.8)          # volatilité suffisante
    )
    mom_short = (
        (rl_dist > buf / 0.0001 * 0.01) &
        (adx > adxmn) &
        early_session &
        (vol_ratio >= 0.8)
    )

    # ── Combinaison (mean-reversion prioritaire) ──────────────────────────
    for i in range(n):
        if mr_long[i]:
            sig[i] = "long"
        elif mr_short[i]:
            sig[i] = "short"
        elif mom_long[i]:
            sig[i] = "long"
        elif mom_short[i]:
            sig[i] = "short"

    return sig


def signal_for_bar(row: pd.Series, config: dict) -> str:
    """Version scalaire — utilisée par le bot live pour une seule bougie."""
    df_tmp = pd.DataFrame([row])
    # Reconstruire les arrays booléens sur 1 ligne
    return compute_signals(df_tmp, config)[0]


def _safe(df: pd.DataFrame, col: str, default: float) -> np.ndarray:
    if col in df.columns:
        return df[col].fillna(default).values.astype(float)
    return np.full(len(df), default, dtype=float)
