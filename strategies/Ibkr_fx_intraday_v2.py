# strategies/Ibkr_fx_intraday_v2.py
"""
Bot live IBKR v2 — architecture refactorisée.

Pipeline par bougie :
  1. Fetch bars IBKR (1-min, BAR_DURATION_DAYS jours)
  2. Build features (build_all_features)
  3. Compute signal (signals.compute_signals)
  4. ML filter (LGBMTradingModel.predict_proba)
  5. Risk checks (daily DD, spread, consecutive losses)
  6. Size position (Kelly fractionnel)
  7. Bracket order IBKR (MKT entry + LMT TP + STP SL)
  8. Attendre prochaine bougie
"""

from __future__ import annotations
import asyncio
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Python 3.14 ne crée plus de boucle asyncio automatiquement — ib_insync en a besoin
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import pandas as pd

from ib_insync import IB, Contract, MarketOrder, LimitOrder, StopOrder

from config import settings as S
from features.technical import build_all_features
from models.lgbm_model import LGBMTradingModel
from strategies.signals import signal_for_bar
from strategies.risk import (
    RiskState, kelly_position_size,
    check_daily_drawdown, check_spread, check_consecutive_losses,
    stop_dist_pips, ev_check,
)
from strategies.utils import (
    make_fx_contract, wait_for_price, fetch_bars,
    log_trade, pip_value_per_unit, pip_size, now_in_tz, in_session, ensure_paths,
)

logger = logging.getLogger("eloe_fx_v2")


# ─────────────────────────────────────────────
#  Chargement des modèles
# ─────────────────────────────────────────────
def load_models() -> tuple[LGBMTradingModel, LGBMTradingModel]:
    buy_path  = Path(S.MODEL_PATH)
    sell_path = Path(S.SELL_MODEL_PATH)
    if not buy_path.exists() or not sell_path.exists():
        raise FileNotFoundError(
            "Modèles non trouvés. Lance d'abord : python scripts/train_model.py"
        )
    model_buy  = LGBMTradingModel.load(str(buy_path))
    model_sell = LGBMTradingModel.load(str(sell_path))
    logger.info("Modèles chargés — threshold buy=%.4f sell=%.4f",
                model_buy.threshold, model_sell.threshold)
    return model_buy, model_sell


# ─────────────────────────────────────────────
#  Kill-switch
# ─────────────────────────────────────────────
def _should_kill(risk: RiskState) -> bool:
    if not getattr(S, "USE_KILL_SWITCH", True):
        return False
    dd_pct = risk.daily_pnl / max(risk.capital, 1)
    if check_daily_drawdown(dd_pct, getattr(S, "MAX_DAILY_DRAWDOWN", 0.02)):
        logger.warning("KILL-SWITCH déclenché — drawdown journalier %.2f%%", dd_pct * 100)
        return True
    return False


# ─────────────────────────────────────────────
#  Vérification position ouverte via IBKR
# ─────────────────────────────────────────────
def _has_open_position(ib: IB, contract: Contract) -> bool:
    """Vérifie si une position est ouverte sur ce contrat côté IBKR."""
    for pos in ib.positions():
        if (pos.contract.symbol   == contract.symbol and
                pos.contract.secType  == contract.secType and
                pos.position != 0):
            return True
    return False


# ─────────────────────────────────────────────
#  Bracket order (MKT + TP LMT + SL STP)
# ─────────────────────────────────────────────
def _place_bracket(
    ib:        IB,
    contract:  Contract,
    direction: str,
    qty:       int,
    tp:        float,
    sl:        float,
) -> list:
    action    = "BUY"  if direction == "long" else "SELL"
    close_act = "SELL" if direction == "long" else "BUY"

    # Demander un orderId AVANT de construire les ordres enfants
    parent_id = ib.client.getReqId()

    parent = MarketOrder(action, qty)
    parent.orderId  = parent_id
    parent.transmit = False

    take = LimitOrder(close_act, qty, round(tp, 5))
    take.orderId  = ib.client.getReqId()
    take.parentId = parent_id
    take.transmit = False
    take.tif      = "GTC"

    stop = StopOrder(close_act, qty, round(sl, 5))
    stop.orderId  = ib.client.getReqId()
    stop.parentId = parent_id
    stop.transmit = True   # transmit=True sur le dernier → envoie tout le bracket
    stop.tif      = "GTC"

    trades = []
    for o in [parent, take, stop]:
        trades.append(ib.placeOrder(contract, o))

    logger.info("Bracket %s qty=%d tp=%.5f sl=%.5f", direction, qty, tp, sl)

    try:
        from strategies.telegram_notify import send as tg
        tg(f"✅ ELOE-FX — Ordre placé\n{direction} qty={qty}\nTP={tp:.5f} | SL={sl:.5f}")
    except Exception:
        pass

    return trades


# ─────────────────────────────────────────────
#  Boucle principale
# ─────────────────────────────────────────────
def run(dry_run: bool | None = None):
    ensure_paths()

    if dry_run is None:
        dry_run = getattr(S, "DRY_RUN", False)

    log_file = Path("data/logs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(),
        ],
    )
    # Supprimer le spam ib_insync (updatePortfolio, position, etc.)
    logging.getLogger("ib_insync").setLevel(logging.WARNING)
    logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)
    logging.getLogger("ib_insync.client").setLevel(logging.WARNING)

    model_buy, model_sell = load_models()

    ib = IB()
    ib.connect(S.HOST, S.PORT, clientId=S.CLIENT_ID,
               readonly=getattr(S, "IB_CONNECT_READONLY", False),
               timeout=S.IB_RESQUEST_TIMEOUT)
    logger.info("Connecté à IBKR — DRY_RUN=%s", dry_run)

    pair     = S.PAIRS[0]
    contract = make_fx_contract(pair)
    ps       = pip_size(pair)
    pv       = pip_value_per_unit(pair)

    risk = RiskState(capital=S.CAPITAL)

    config = {
        "atr_k":            S.ATR_K,
        "min_stop_pips":    S.MIN_STOP_PIPS,
        "rr":               S.RR,
        "horizon":          120,
        "signal_vwap_z":    getattr(S, "SIGNAL_VWAP_Z",    1.6),
        "signal_rsi_lo":    getattr(S, "SIGNAL_RSI_LO",   35.0),
        "signal_rsi_hi":    getattr(S, "SIGNAL_RSI_HI",   65.0),
        "adx_max":          getattr(S, "ADX_MAX",          25.0),
        "adx_min_momentum": getattr(S, "ADX_MIN_MOMENTUM", 18.0),
    }

    loop_end      = time.time() + S.RUN_LOOP_MINUTES * 60
    consec_losses = 0
    scan_count    = 0

    while time.time() < loop_end:
        now = now_in_tz(S.TZ)

        # ── Vérification session ──────────────────────────────────────────
        if not getattr(S, "IGNORE_SESSION", False) and \
                not in_session(now, S.TZ, S.SESSION_START, S.SESSION_END):
            logger.info("Hors session — attente 60s")
            time.sleep(60)
            continue

        # ── Kill-switch ───────────────────────────────────────────────────
        risk.reset_day(now.date(), S.CAPITAL)
        if _should_kill(risk):
            break

        # ── Position déjà ouverte ? ───────────────────────────────────────
        if _has_open_position(ib, contract):
            logger.debug("Position ouverte — attente")
            time.sleep(S.SLEEP_BETWEEN_SEC)
            continue

        # ── Fetch données ─────────────────────────────────────────────────
        bars = fetch_bars(
            ib, contract,
            duration   = f"{S.BAR_DURATION_DAYS} D",
            barSize    = S.BAR_SIZE,
            whatToShow = S.WHAT_TO_SHOW,
        )
        if bars is None or len(bars) < 200:
            logger.warning("Données insuffisantes (%s barres) — retry dans %ds",
                           len(bars) if bars is not None else 0, S.SLEEP_BETWEEN_SEC)
            time.sleep(S.SLEEP_BETWEEN_SEC)
            continue

        # ── Features ──────────────────────────────────────────────────────
        try:
            df_feat = build_all_features(bars)
        except Exception as exc:
            logger.error("Erreur features : %s", exc)
            time.sleep(S.SLEEP_BETWEEN_SEC)
            continue

        last_row = df_feat.iloc[-2]   # dernière bougie complète (pas l'en-cours)

        # ── Signal ────────────────────────────────────────────────────────
        scan_count += 1
        sig = signal_for_bar(last_row, config)
        if sig == "flat":
            if scan_count % 10 == 0:  # log toutes les 10 scans (~3 min)
                logger.info("Scan #%d — aucun signal (flat)", scan_count)
            time.sleep(S.SLEEP_BETWEEN_SEC)
            continue
        logger.info("Scan #%d — signal détecté : %s", scan_count, sig.upper())

        # ── Filtre ML ─────────────────────────────────────────────────────
        row_df = pd.DataFrame([last_row])
        if sig == "long":
            proba = model_buy.predict_proba(row_df)[0]
            thr   = model_buy.threshold
        else:
            proba = model_sell.predict_proba(row_df)[0]
            thr   = model_sell.threshold

        if proba < thr:
            logger.info("Signal %s rejeté ML — proba=%.4f < thr=%.4f", sig, proba, thr)
            time.sleep(S.SLEEP_BETWEEN_SEC)
            continue

        # ── EV check ──────────────────────────────────────────────────────
        if not ev_check(proba, S.RR):
            time.sleep(S.SLEEP_BETWEEN_SEC)
            continue

        # ── Circuit-breaker pertes consécutives ───────────────────────────
        if check_consecutive_losses(consec_losses, max_losses=3):
            logger.info("Cooldown — %d pertes consécutives", consec_losses)
            time.sleep(S.SLEEP_BETWEEN_SEC * 5)
            consec_losses = 0
            continue

        # ── Prix live ─────────────────────────────────────────────────────
        live_price = wait_for_price(ib, contract, timeout=S.PRICE_TIMEOUT_SEC)
        if live_price is None:
            logger.warning("Prix live non disponible")
            time.sleep(S.SLEEP_BETWEEN_SEC)
            continue

        # ── Spread check ──────────────────────────────────────────────────
        if not check_spread(getattr(S, "MAX_SPREAD_PIPS", 0.30), S.MAX_SPREAD_PIPS):
            time.sleep(S.SLEEP_BETWEEN_SEC)
            continue

        # ── Sizing ────────────────────────────────────────────────────────
        atr_val  = float(last_row.get("atr_14", 0.0008))
        sd_pips  = stop_dist_pips(atr_val, S.ATR_K, S.MIN_STOP_PIPS)
        sd_price = sd_pips * ps

        qty = kelly_position_size(
            proba          = proba,
            rr             = S.RR,
            capital        = S.CAPITAL,
            stop_dist_pips = sd_pips,
            pip_value      = pv,
            kelly_fraction = getattr(S, "KELLY_FRACTION", 0.25),
            min_qty        = S.MIN_QTY,
            qty_step       = S.QTY_STEP,
            max_risk_pct   = S.RISK_PER_TRADE,
        )
        if qty == 0:
            logger.info("Sizing = 0 — EV insuffisant (proba=%.4f)", proba)
            time.sleep(S.SLEEP_BETWEEN_SEC)
            continue

        # ── Calcul TP / SL ────────────────────────────────────────────────
        if sig == "long":
            tp = round(live_price + S.RR * sd_price, 5)
            sl = round(live_price - sd_price, 5)
        else:
            tp = round(live_price - S.RR * sd_price, 5)
            sl = round(live_price + sd_price, 5)

        logger.info(
            "SETUP %s | price=%.5f tp=%.5f sl=%.5f qty=%d proba=%.4f",
            sig.upper(), live_price, tp, sl, qty, proba,
        )

        # ── Exécution ─────────────────────────────────────────────────────
        if not dry_run:
            _place_bracket(ib, contract, sig, qty, tp, sl)
            log_trade(pair, sig, qty, live_price, sl, tp, "open")
        else:
            logger.info("[DRY-RUN] Trade simulé — pas d'ordre envoyé")

        time.sleep(S.SLEEP_BETWEEN_SEC)

    ib.disconnect()
    logger.info("Bot arrêté proprement.")


if __name__ == "__main__":
    run()
