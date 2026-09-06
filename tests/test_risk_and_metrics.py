"""
Unit tests for position sizing / circuit-breakers (strategies/risk.py) and
performance metrics (backtests/metrics.py).

Run with:
    pytest tests/
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.risk import (
    kelly_position_size,
    fixed_risk_position_size,
    check_daily_drawdown,
    check_spread,
    check_consecutive_losses,
    ev_check,
)
from strategies.utils import pip_size
from backtests.metrics import compute_metrics


def test_kelly_position_size_is_zero_for_negative_edge():
    # proba*rr - (1-proba) < 0 -> Kelly complet negatif -> pas de trade
    qty = kelly_position_size(proba=0.3, rr=1.0, capital=10_000, stop_dist_pips=10)
    assert qty == 0


def test_kelly_position_size_respects_max_risk_cap():
    # Avec une proba tres favorable, le sizing doit rester plafonne a max_risk_pct
    qty_capped = kelly_position_size(
        proba=0.9, rr=2.0, capital=10_000, stop_dist_pips=10, max_risk_pct=0.01
    )
    qty_uncapped = kelly_position_size(
        proba=0.9, rr=2.0, capital=10_000, stop_dist_pips=10, max_risk_pct=1.0
    )
    assert qty_capped <= qty_uncapped


def test_fixed_risk_position_size_scales_with_capital():
    qty_small = fixed_risk_position_size(capital=10_000, risk_pct=0.01, stop_dist_pips=10)
    qty_big = fixed_risk_position_size(capital=100_000, risk_pct=0.01, stop_dist_pips=10)
    assert qty_big > qty_small


def test_circuit_breakers():
    assert check_daily_drawdown(-0.03, max_dd_pct=0.02) is True
    assert check_daily_drawdown(-0.01, max_dd_pct=0.02) is False
    assert check_spread(0.20, max_spread_pips=0.30) is True
    assert check_spread(0.40, max_spread_pips=0.30) is False
    assert check_consecutive_losses(3, max_losses=3) is True
    assert check_consecutive_losses(2, max_losses=3) is False


def test_ev_check_positive_and_negative_edge():
    assert ev_check(proba=0.4, rr=2.0) is True    # EV = 0.4*2 - 0.6 = 0.2 > 0
    assert ev_check(proba=0.2, rr=2.0) is False   # EV = 0.2*2 - 0.8 = -0.4 < 0


def test_pip_size_jpy_vs_other_pairs():
    assert pip_size("USDJPY") == 0.01
    assert pip_size("EURUSD") == 0.0001


def test_compute_metrics_on_known_trades():
    trades = pd.DataFrame({
        "pnl_pips":   [16.0, -8.0, 16.0, -8.0, 16.0],
        "direction":  ["long", "short", "long", "long", "short"],
        "bars_held":  [30, 20, 40, 25, 35],
        "entry_time": [0, 1, 2, 3, 4],
        "exit_time":  [1, 2, 3, 4, 5],
    })

    metrics = compute_metrics(trades, config={"rr": 2.0})

    assert metrics["n_trades"] == 5
    assert metrics["win_rate"] == pytest.approx(0.6)
    assert metrics["total_pips"] == pytest.approx(32.0)
    assert metrics["profit_factor"] == pytest.approx(48.0 / 16.0)
    assert metrics["n_long"] == 3
    assert metrics["n_short"] == 2


def test_compute_metrics_on_empty_trades_returns_zeros():
    metrics = compute_metrics(pd.DataFrame(), config={"rr": 2.0})
    assert metrics["n_trades"] == 0
    assert metrics["sharpe"] == 0
