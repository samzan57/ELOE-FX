# strategies/risk.py
"""
Gestion du risque — approche institutionnelle.

Kelly fractionnel :
  f* = (p*(rr+1) - 1) / rr          (Kelly complet)
  f  = f* × kelly_fraction           (Kelly fractionnel, par défaut 0.25)

  Le Kelly fractionnel (25%) est utilisé par la plupart des fonds quant car
  le Kelly complet suppose une estimation parfaite de p, ce qui n'est jamais
  le cas en pratique. 0.25× Kelly réduit la variance d'environ 75%.

Circuit-breakers :
  - Drawdown journalier dépasse MAX_DAILY_DD → stop total
  - Perte consécutive ≥ MAX_CONSEC_LOSSES → cooldown
  - Spread trop large → skip
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import date


PIP_SIZE = 0.0001


@dataclass
class RiskState:
    """État mutable du risk manager — réinitialisé chaque jour."""
    capital:            float
    daily_start_equity: float        = 0.0
    today:              date | None  = None
    consec_losses:      int          = 0
    trades_today:       int          = 0
    daily_pnl:          float        = 0.0

    def reset_day(self, today: date, equity: float):
        if self.today != today:
            self.today              = today
            self.daily_start_equity = equity
            self.trades_today       = 0
            self.daily_pnl          = 0.0


# ─────────────────────────────────────────────
#  Calcul de taille de position
# ─────────────────────────────────────────────

def kelly_position_size(
    proba:          float,
    rr:             float,
    capital:        float,
    stop_dist_pips: float,
    pip_value:      float   = 10.0,   # $ par pip pour 100k (standard lot)
    kelly_fraction: float   = 0.25,
    min_qty:        int     = 1_000,
    qty_step:       int     = 1_000,
    max_risk_pct:   float   = 0.01,   # 1% max par trade (plafond de sécurité)
) -> int:
    """
    Retourne la taille en unités (ex: 5000 = 5k unités EURUSD).

    Algorithme :
      1. Kelly fractionnel → % du capital à risquer
      2. Plafonner à max_risk_pct
      3. Convertir en unités via stop_dist_pips × pip_value
      4. Arrondir au qty_step le plus proche
    """
    if proba <= 0 or rr <= 0 or stop_dist_pips <= 0:
        return 0

    # Kelly complet
    kelly_full = (proba * (rr + 1) - 1) / rr
    if kelly_full <= 0:
        return 0   # EV négatif → pas de trade

    risk_fraction = min(kelly_full * kelly_fraction, max_risk_pct)
    risk_dollars  = capital * risk_fraction

    # stop_dist_pips pips × pip_value = $ par unité standard (100k)
    # Pour qty unités : perte_max = qty/100_000 × stop_dist_pips × pip_value
    # qty = risk_dollars × 100_000 / (stop_dist_pips × pip_value)
    pip_val_per_unit = pip_value / 100_000   # $ par pip par unité
    qty_raw = risk_dollars / (stop_dist_pips * pip_val_per_unit)

    qty = int(qty_raw // qty_step) * qty_step
    return max(qty, min_qty) if qty_raw >= min_qty else 0


def fixed_risk_position_size(
    capital:        float,
    risk_pct:       float,
    stop_dist_pips: float,
    pip_value:      float = 10.0,
    min_qty:        int   = 1_000,
    qty_step:       int   = 1_000,
) -> int:
    """
    Sizing simplifié (fallback si Kelly non disponible) :
      qty = (capital × risk_pct) / (stop_dist_pips × pip_val_per_unit)
    """
    if stop_dist_pips <= 0:
        return 0
    pip_val_per_unit = pip_value / 100_000
    qty_raw = (capital * risk_pct) / (stop_dist_pips * pip_val_per_unit)
    qty = int(qty_raw // qty_step) * qty_step
    return max(qty, min_qty) if qty_raw >= min_qty else 0


# ─────────────────────────────────────────────
#  Circuit-breakers
# ─────────────────────────────────────────────

def check_daily_drawdown(
    daily_pnl_pct:    float,
    max_dd_pct:       float = 0.02,
) -> bool:
    """True si le drawdown journalier dépasse le seuil → STOP."""
    return daily_pnl_pct <= -max_dd_pct


def check_spread(spread_pips: float, max_spread_pips: float = 0.30) -> bool:
    """True si le spread est acceptable."""
    return spread_pips <= max_spread_pips


def check_consecutive_losses(
    consec_losses: int,
    max_losses:    int = 3,
) -> bool:
    """True si on dépasse le max de pertes consécutives → cooldown."""
    return consec_losses >= max_losses


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def stop_dist_pips(atr: float, atr_k: float, min_stop: float) -> float:
    """Calcule le stop en pips à partir de l'ATR."""
    return max((atr / PIP_SIZE) * atr_k, min_stop)


def ev_check(proba: float, rr: float, min_ev: float = 0.0) -> bool:
    """Vérifie que l'espérance mathématique est positive."""
    return (proba * rr - (1.0 - proba)) > min_ev
