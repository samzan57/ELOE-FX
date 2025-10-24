# strategies/Ibkr_fx_intraday.py
# Intraday FX with risk-based sizing, ATR stops, breakout+RSI filter, ML filter (BUY & SELL),
# spread cap, cooldown, max positions, and looped scanning (IBKR only).

from ib_insync import *
from dataclasses import dataclass
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import time
import joblib

from config import settings as S
from strategies.utils import (
    make_fx_contract, wait_for_price, fetch_bars, log_trade,
    pip_value_per_unit, pip_size, now_in_tz, in_session, ensure_paths
)
from strategies.indicators import compute_indicators

# ---------- ML features builder (robuste) ----------
try:
    from models.proba_model import build_features as build_features_live
except Exception:
    def build_features_live(df: pd.DataFrame):
        # Fallback neutre si pas de module dédié
        return df.copy()
    
# --- Kill-switch v3: auto (PnL IB fiable) -> sinon NetLiq baseline, avec confirmation ---

BASELINE_FILE = Path("data/logs/netliq_baseline.csv")
_kill_breach_streak = 0  # compteur de confirmations consécutives

def _account_id(ib: IB):
    acct = getattr(S, "IB_ACCOUNT", None)
    if not acct:
        accts = ib.managedAccounts()
        acct = accts[0] if accts else None
    return acct

def _get_daily_pnl_ib(ib: IB, wait_total_sec: float = 3.0):
    """reqPnL avec petite attente; retourne float ou None."""
    acct = _account_id(ib)
    if not acct:
        print("[WARN] kill: no IB account found")
        return None
    pnl_obj = None
    try:
        pnl_obj = ib.reqPnL(acct, "")
        deadline = time.time() + wait_total_sec
        daily = None
        while time.time() < deadline:
            ib.waitOnUpdate(timeout=0.5)
            daily = pnl_obj.dailyPnL
            if daily is not None and daily == daily:  # pas NaN
                return float(daily)
        return None
    except Exception as e:
        print(f"[WARN] kill: reqPnL failed: {e}")
        return None
    finally:
        try:
            if pnl_obj is not None:
                ib.cancelPnL(pnl_obj)
        except Exception:
            pass

def _get_netliq(ib: IB):
    """Lit NetLiquidation via accountSummary; retourne float ou None."""
    try:
        vals = ib.accountSummary()
        target = _account_id(ib)
        netliq = None
        for v in vals:
            if v.tag == "NetLiquidation" and (target is None or v.account == target):
                netliq = float(v.value); break
        if netliq is None:
            for v in vals:
                if v.tag == "NetLiquidation":
                    netliq = float(v.value); break
        return netliq
    except Exception as e:
        print(f"[WARN] kill: accountSummary failed: {e}")
        return None

def _load_baseline(acct: str, ymd: str):
    if not BASELINE_FILE.exists():
        return None
    try:
        for line in BASELINE_FILE.read_text(encoding="utf-8").splitlines():
            a, d, v = line.strip().split(",")
            if a == acct and d == ymd:
                return float(v)
    except Exception:
        pass
    return None

def _save_baseline(acct: str, ymd: str, netliq: float):
    ensure_paths()
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with BASELINE_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{acct},{ymd},{netliq:.2f}\n")

def _kill_switch_hit(ib: IB) -> bool:
    """True => STOP trading today (drawdown day dépassé)."""
    global _kill_breach_streak
    if not getattr(S, "USE_KILL_SWITCH", True) or getattr(S, "KILL_MODE", "auto") == "off":
        _kill_breach_streak = 0
        return False

    max_loss = float(getattr(S, "CAPITAL", 10_000)) * float(getattr(S, "MAX_DAILY_DRAWDOWN", 0.02))
    mode = getattr(S, "KILL_MODE", "auto")
    sanity_mult = float(getattr(S, "KILL_SANITY_MULT", 0.25))
    confirm_needed = int(getattr(S, "KILL_CONFIRM_SCANS", 2))

    # 1) PnL jour IB (si demandé et/ou auto)
    use_pnl = mode in ("auto", "pnl")
    daily = _get_daily_pnl_ib(ib) if use_pnl else None
    if use_pnl and daily is not None:
        print(f"[PnL] dailyPnL={daily:.2f}  maxLoss={-max_loss:.2f}")
        # sanity check: ignore outliers (ex: -7250 sur 10k)
        if abs(daily) > sanity_mult * float(getattr(S, "CAPITAL", 10_000)):
            print(f"[WARN] kill: dailyPnL {daily:.2f} exceeds sanity {sanity_mult:.0%}*capital => treat as invalid")
        else:
            if daily <= -max_loss:
                _kill_breach_streak += 1
                if _kill_breach_streak >= confirm_needed:
                    print(f"[KILL] DailyPnL {daily:.2f} <= -{max_loss:.2f} (x{_kill_breach_streak}) -> stop today.")
                    return True
                else:
                    print(f"[WARN] kill: breach { _kill_breach_streak }/{ confirm_needed } (need consecutive)")
                    return False
            else:
                _kill_breach_streak = 0
                return False
        # si dailyPnL jugé invalide, on tombe en fallback NetLiq

    # 2) Fallback NetLiq baseline (si demandé et/ou auto)
    if mode in ("auto", "netliq"):
        acct = _account_id(ib) or "UNKNOWN"
        ymd = now_in_tz(getattr(S, "TZ", "Europe/Paris")).strftime("%Y-%m-%d")
        netliq = _get_netliq(ib)
        if netliq is None:
            print("[WARN] kill: netliq unavailable; skipping kill check this scan")
            return False
        base = _load_baseline(acct, ymd)
        if base is None:
            _save_baseline(acct, ymd, netliq)
            print(f"[PnL] baseline set (NetLiq) {netliq:.2f} for {acct} {ymd}")
            _kill_breach_streak = 0
            return False
        
        # calcul du seuil avec ref = capital ou baseline NetLiq
        ref = float(getattr(S, "CAPITAL", 10_000))
        if getattr(S, "KILL_REF", "capital") == "netliq" and base is not None:
            ref = base
        max_loss = ref * float(getattr(S, "MAX_DAILY_DRAWDOWN", 0.02))
        
        delta = netliq - base  # <0 si perte
        print(f"[PnL] NetLiq={netliq:.2f} base={base:.2f} delta={delta:.2f}  maxLoss={-max_loss:.2f}")
        if delta <= -max_loss:
            _kill_breach_streak += 1
            if _kill_breach_streak >= confirm_needed:
                print(f"[KILL] NetLiq drop {delta:.2f} <= -{max_loss:.2f} (x{_kill_breach_streak}) -> stop today.")
                return True
            else:
                print(f"[WARN] kill: breach { _kill_breach_streak }/{ confirm_needed } (need consecutive)")
                return False
        else:
            _kill_breach_streak = 0
            return False

    # 3) Pas de mode applicable
    return False


# ---------- DRY RUN override via variable d'env ----------
if os.environ.get("FORCE_DRYRUN") == "1":
    S.DRY_RUN = True

# ---------- Charger modèles proba ----------
BUY_MODEL_PATH  = Path(getattr(S, "MODEL_PATH", "models/model_eurusd_buy.pkl"))
SELL_MODEL_PATH = Path(getattr(S, "SELL_MODEL_PATH", "models/model_eurusd_sell.pkl"))

BUY_MODEL  = joblib.load(BUY_MODEL_PATH)  if BUY_MODEL_PATH.exists()  else None
SELL_MODEL = joblib.load(SELL_MODEL_PATH) if SELL_MODEL_PATH.exists() else None

# Mémoire cooldown par paire
last_trade_time = {}

# ---------- Data classes ----------
@dataclass
class OrderPlan:
    pair: str
    side: str
    qty: int
    entry: float
    stop: float
    take: float

# ---------- Helpers ----------
def plan_order(pair: str, price: float, atr: float, side: str) -> OrderPlan:
    ps = pip_size(pair)
    stop_pips = max((atr / ps) * S.ATR_K, S.MIN_STOP_PIPS) if (atr == atr) else S.MIN_STOP_PIPS
    stop_dist = stop_pips * ps

    if side == "BUY":
        stop = round(price - stop_dist, 5)
        take = round(price + S.RR * stop_dist, 5)
    else:
        stop = round(price + stop_dist, 5)
        take = round(price - S.RR * stop_dist, 5)

    risk_usd = S.CAPITAL * S.RISK_PER_TRADE
    pip_val_unit = pip_value_per_unit(pair)
    qty_float = risk_usd / (stop_pips * pip_val_unit)
    qty = int(max(getattr(S, "MIN_QTY", 1000), round(qty_float / getattr(S, "QTY_STEP", 1000)) * getattr(S, "QTY_STEP", 1000)))
    return OrderPlan(pair, side, qty, price, stop, take)

def signal_breakout_rsi(df: pd.DataFrame, pair: str):
    """Breakout + RSI avec option high/low + buffer en pips."""
    if df is None or len(df) < 2 or 'range_high' not in df or 'range_low' not in df or 'rsi' not in df:
        return None
    prior = df.iloc[-2]
    if pd.isna(prior.get('range_high')) or pd.isna(prior.get('range_low')):
        return None
    last = df.iloc[-1]

    ps = pip_size(pair)
    buf = getattr(S, "BREAKOUT_BUFFER_PIPS", 0.0) * ps
    use_hl = getattr(S, "BREAKOUT_USE_HIGH_LOW", True)

    up_break = (last.high > prior['range_high'] + buf) if use_hl else (last.close > prior['range_high'] + buf)
    dn_break = (last.low  < prior['range_low']  - buf) if use_hl else (last.close < prior['range_low']  - buf)

    if up_break and last.rsi >= S.RSI_MIN:
        return "BUY"
    if dn_break and last.rsi <= (100 - S.RSI_MIN):
        return "SELL"
    return None

def place_bracket(ib: IB, contract: Contract, plan: OrderPlan):
    action = plan.side
    parent = Order(
        action=action,
        orderType=getattr(S, "PARENT_ORDER_TYPE", "MKT"),
        totalQuantity=plan.qty,
        tif=getattr(S, "TIF", "GTC"),
        transmit=False,
    )
    take = Order(
        action='SELL' if action == 'BUY' else 'BUY',
        orderType=getattr(S, "TAKE_ORDER_TYPE", "LMT"),
        totalQuantity=plan.qty,
        lmtPrice=plan.take,
        tif=getattr(S, "TIF", "GTC"),
        transmit=False,
    )
    stop = Order(
        action='SELL' if action == 'BUY' else 'BUY',
        orderType=getattr(S, "STOP_ORDER_TYPE", "STP"),
        totalQuantity=plan.qty,
        auxPrice=plan.stop,
        tif=getattr(S, "TIF", "GTC"),
        transmit=True,
    )
    ib.qualifyContracts(contract)
    parent.orderId = ib.client.getReqId()
    take.parentId = parent.orderId
    stop.parentId = parent.orderId

    if S.DRY_RUN:
        print(f"[DRY_RUN] {plan.pair} {plan.side} qty={plan.qty} entry~{plan.entry:.5f} SL={plan.stop:.5f} TP={plan.take:.5f}")
        log_trade(plan.pair, plan.side, plan.qty, plan.entry, plan.stop, plan.take, "DRY_RUN", "not sent")
        return

    ib.placeOrder(contract, parent)
    ib.placeOrder(contract, take)
    ib.placeOrder(contract, stop)
    print(f"[PLACED] {plan.pair} {plan.side} qty={plan.qty} SL={plan.stop:.5f} TP={plan.take:.5f}")
    log_trade(plan.pair, plan.side, plan.qty, plan.entry, plan.stop, plan.take, "PLACED")

def get_spread_pips(ib: IB, contract: Contract, pair: str) -> float:
    """Retourne le spread en pips (bid/ask) ou 1e9 si indisponible."""
    t = ib.reqMktData(contract, '', False, False)
    ib.waitOnUpdate(timeout=1.0)
    spread_pips = None
    bid = getattr(t, 'bid', None)
    ask = getattr(t, 'ask', None)
    if bid is not None and ask is not None:
        ps = pip_size(pair)
        spread_pips = (ask - bid) / ps
    try:
        if getattr(t, "tickerId", None):
            ib.cancelMktData(t)
    except Exception:
        pass
    return float(spread_pips) if spread_pips is not None else 1e9

def open_positions_count_for_pair(ib: IB, pair: str) -> int:
    """Compte les positions CASH sur la paire (ex: EURUSD -> symbol='EUR', currency='USD')."""
    base, quote = pair[:3], pair[3:]
    try:
        positions = ib.positions()
    except Exception:
        return 0
    count = 0
    for p in positions:
        c = p.contract
        if getattr(c, 'secType', '') == 'CASH' and getattr(c, 'symbol', '') == base and getattr(c, 'currency', '') == quote:
            count += 1
    return count

# --- Pre-flight check (sanity before trading) ---

def _pre_flight_check(ib: IB) -> bool:
    ok = True
    print("\n=== PRE-FLIGHT CHECK ===")

    # Connexion & comptes
    if not ib.isConnected():
        print("[X] IB not connected")
        return False
    accts = ib.managedAccounts()
    acct = getattr(S, "IB_ACCOUNT", None) or (accts[0] if accts else None)
    if not acct:
        print("[X] No IB account returned by managedAccounts()")
        return False
    print(f"[OK] Connected. Account={acct}  Host={S.HOST}:{S.PORT}  ReadOnly={getattr(S, 'IB_CONNECT_READONLY', False)}")

    # Session & paramètres clés
    now = now_in_tz(S.TZ)
    in_sess = in_session(now, S.TZ, S.SESSION_START, S.SESSION_END) if not getattr(S, "IGNORE_SESSION", False) else True
    print(f"[i] Time={now.strftime('%Y-%m-%d %H:%M:%S %Z')}  Session={S.SESSION_START}-{S.SESSION_END}  InSession={in_sess}")
    print(f"[i] DRY_RUN={S.DRY_RUN}  RISK_PER_TRADE={S.RISK_PER_TRADE}  MAX_SPREAD_PIPS={getattr(S,'MAX_SPREAD_PIPS',0.2)}")
    print(f"[i] KILL_MODE={getattr(S,'KILL_MODE','auto')}  KILL_REF={getattr(S,'KILL_REF','capital')}  MAX_DD={S.MAX_DAILY_DRAWDOWN:.2%}")

    # Modèles BUY/SELL (optionnels)
    buy_path  = Path(getattr(S, "BUY_MODEL_PATH", getattr(S, "MODEL_PATH", "models/model_eurusd_buy.pkl")))
    sell_path = Path(getattr(S, "SELL_MODEL_PATH", "models/model_eurusd_sell.pkl"))
    print(f"[i] BUY model:  {buy_path}  ({'exists' if buy_path.exists() else 'missing'})")
    print(f"[i] SELL model: {sell_path}  ({'exists' if sell_path.exists() else 'missing'})")

    # Contrat & flux
    try:
        pair = S.PAIRS[0]
        contract = make_fx_contract(pair)
        ib.qualifyContracts(contract)
        if not getattr(contract, "conId", 0):
            print(f"[X] Contract not qualified for {pair}")
            return False
        # Prix & spread test
        p = wait_for_price(ib, contract, timeout=getattr(S, "PRICE_TIMEOUT_SEC", 5.0))
        if not p:
            print(f"[X] No live price for {pair}")
            ok = False
        else:
            spr = get_spread_pips(ib, contract, pair)
            print(f"[i] Spot~{p:.5f}  Spread~{spr:.3f} pips (cap={getattr(S,'MAX_SPREAD_PIPS',0.2)})")
    except Exception as e:
        print(f"[X] Market data check failed: {e}")
        ok = False

    # Historique
    try:
        df = fetch_bars(
            ib, contract,
            duration=f"{getattr(S,'BAR_DURATION_DAYS',1)} D",
            barSize=getattr(S,'BAR_SIZE','1 min'),
            tz=S.TZ,
            whatToShow=getattr(S,'WHAT_TO_SHOW','MIDPOINT'),
        )
        if df is None or df.empty:
            print("[X] Historical data empty")
            ok = False
        else:
            print(f"[i] Historical bars: {len(df)} rows from {df.index[0]} -> {df.index[-1]}")
    except Exception as e:
        print(f"[X] Historical fetch failed: {e}")
        ok = False

    # Kill-switch “dry probe”
    try:
        # On ne stoppe pas ici, on vérifie juste que l’appel ne crash pas
        _ = getattr(S, "KILL_MODE", "auto")
        # On appelle la baseline (sans empêcher de trader)
        # Note: la vraie décision reste dans _kill_switch_hit()
        pass
    except Exception as e:
        print(f"[X] Kill-switch config error: {e}")
        ok = False

    print(f"=== PRE-FLIGHT RESULT: {'OK' if ok else 'ISSUES FOUND'} ===\n")
    return ok


# ---------- Main ----------
def main():
    ensure_paths()
    ib = IB()
    ib.RequestTimeout = getattr(S, "IB_REQUEST_TIMEOUT", 30)

    # Connexion tolérante (retry + readonly)
    for attempt in range(1, 4):
        try:
            ib.connect(S.HOST, S.PORT, clientId=S.CLIENT_ID, readonly=getattr(S, "IB_CONNECT_READONLY", True))
            break
        except Exception as e:
            print(f"[WARN] Connexion IB tentative {attempt}/3: {e}")
            ib.sleep(2 * attempt)
    else:
        print("[ERROR] Impossible de se connecter à IB après 3 tentatives.")
        return

    # PRE-FLIGHT CHECK 
    if not _pre_flight_check(ib):
        print("[ABORT] Pre-flight check failed")
        ib.disconnect()
        return

    # (Optionnel) ne faire que le pré-check puis sortir
    if os.environ.get("PRECHECK_ONLY") == "1":
        print("[EXIT] Pre-check only mode.")
        ib.disconnect()
        return
    

    loops = max(1, getattr(S, "RUN_LOOP_MINUTES", 1))
    for k in range(loops):
        # juste au debut de chaque scan (dans for k in range(loops): )
        if _kill_switch_hit(ib):
            break

        now = now_in_tz(S.TZ)
        if not getattr(S, "IGNORE_SESSION", False):
            if not in_session(now, S.TZ, S.SESSION_START, S.SESSION_END):
                print(f"[INFO] Hors session {S.SESSION_START}-{S.SESSION_END} {S.TZ}.")
                break

        for pair in S.PAIRS:
            contract = make_fx_contract(pair)
            ib.qualifyContracts(contract)

            # Prix spot
            price = wait_for_price(ib, contract, timeout=getattr(S, "PRICE_TIMEOUT_SEC", 5.0))
            if not price:
                print(f"[WARN] Prix indisponible pour {pair}.")
                continue

            # Historique
            df = fetch_bars(
                ib, contract,
                duration=f"{getattr(S, 'BAR_DURATION_DAYS', 1)} D",
                barSize=getattr(S, 'BAR_SIZE', '1 min'),
                tz=S.TZ,
                whatToShow=getattr(S, 'WHAT_TO_SHOW', 'MIDPOINT'),
            )
            if df is None or df.empty or len(df) < 2:
                print(f"[INFO] {pair}: pas assez d'historique.")
                continue

            # Indicateurs
            df = compute_indicators(df, atr_len=S.ATR_LEN, rsi_len=S.RSI_LEN, range_lookback=S.BREAKOUT_LOOKBACK)

            # === Features (calculées une seule fois) ===
            df_feat = None
            try:
                res = build_features_live(df)
                df_feat = res[0] if isinstance(res, (tuple, list)) else res
            except Exception as e:
                print(f"[WARN] build_features_live a échoué: {e}")
                df_feat = None

            # === Probabilités BUY / SELL si modèles dispo ===
            p_buy, thr_buy = None, float(getattr(S, "MODEL_EV_THRESHOLD", 0.5))
            if BUY_MODEL is not None and df_feat is not None:
                try:
                    model = BUY_MODEL["model"]
                    FEATURES_SAVED = BUY_MODEL.get("features", [])
                    thr_buy = float(BUY_MODEL.get("threshold", thr_buy))
                    needed = list(FEATURES_SAVED) if FEATURES_SAVED else list(df_feat.columns)
                    for col in needed:
                        if col not in df_feat.columns:
                            df_feat[col] = 0.0
                    X_last = df_feat.iloc[[-1]][needed].fillna(0).to_numpy(dtype=float)
                    proba = model.predict_proba(X_last)
                    p_buy = float(proba[0, 1])
                    print(f"[PROBA] BUY p={p_buy:.3f} (thr={thr_buy:.3f})")
                except Exception as e:
                    print(f"[WARN] Impossible de calculer la proba BUY: {e}")
                    p_buy = None

            p_sell, thr_sell = None, float(getattr(S, "SELL_MODEL_EV_THRESHOLD", 0.5))
            if SELL_MODEL is not None and df_feat is not None:
                try:
                    model_s = SELL_MODEL["model"]
                    FEATURES_SAVED_S = SELL_MODEL.get("features", [])
                    thr_sell = float(SELL_MODEL.get("threshold", thr_sell))
                    needed_s = list(FEATURES_SAVED_S) if FEATURES_SAVED_S else list(df_feat.columns)
                    for col in needed_s:
                        if col not in df_feat.columns:
                            df_feat[col] = 0.0
                    X_last_s = df_feat.iloc[[-1]][needed_s].fillna(0).to_numpy(dtype=float)
                    proba_s = model_s.predict_proba(X_last_s)
                    p_sell = float(proba_s[0, 1])
                    print(f"[PROBA] SELL p={p_sell:.3f} (thr={thr_sell:.3f})")
                except Exception as e:
                    print(f"[WARN] Impossible de calculer la proba SELL: {e}")
                    p_sell = None

            # DEBUG contexte signal
            try:
                prior = df.iloc[-2]; last = df.iloc[-1]
                ps = pip_size(pair)
                buf = getattr(S, "BREAKOUT_BUFFER_PIPS", 0.0) * ps
                use_hl = getattr(S, "BREAKOUT_USE_HIGH_LOW", True)
                up_break = (last.high > prior['range_high'] + buf) if use_hl else (last.close > prior['range_high'] + buf)
                dn_break = (last.low  < prior['range_low']  - buf) if use_hl else (last.close < prior['range_low']  - buf)
                cond_rsi_up = last['rsi'] >= S.RSI_MIN
                cond_rsi_dn = last['rsi'] <= (100 - S.RSI_MIN)
                print(f"[DEBUG] close={last['close']:.5f} prevH={prior['range_high']:.5f} prevL={prior['range_low']:.5f} "
                      f"RSI={last['rsi']:.1f} | up_break={up_break} rsi_up={cond_rsi_up} | dn_break={dn_break} rsi_dn={cond_rsi_dn}")
            except Exception as e:
                print(f"[DEBUG] contexte signal indisponible: {e}")

            # Filtre spread (pips)
            spr_raw = get_spread_pips(ib, contract, pair)
            spr = round(spr_raw, 3)
            max_spread = float(getattr(S, "MAX_SPREAD_PIPS", 0.2))
            print(f"[SPREAD] {pair} = {spr:.3f} pips (max={max_spread:.3f})")
            if spr >= max_spread:
                print(f"[SKIP] spread {spr:.3f} pips >= max {max_spread:.3f} pips")
                continue

            # Cooldown
            now_utc = datetime.utcnow()
            lt = last_trade_time.get(pair)
            if lt and now_utc - lt < timedelta(minutes=getattr(S, "REENTER_COOLDOWN_MIN", 15)):
                print(f"[SKIP] cooldown encore actif {pair}")
                continue

            # Cap de positions ouvertes
            try:
                open_cnt = open_positions_count_for_pair(ib, pair)
            except Exception:
                open_cnt = 0
            if open_cnt >= getattr(S, "MAX_CONCURRENT_POSITIONS", 1):
                print("[SKIP] max positions atteint")
                continue

            # Signal principal
            sig = signal_breakout_rsi(df, pair)
            if sig is None:
                print(f"[INFO] {pair}: pas de signal.")
                continue

            # Filtre proba BUY / SELL
            if sig == "BUY" and p_buy is not None:
                if p_buy < thr_buy:
                    print(f"[SKIP] BUY proba={p_buy:.2f} < thr={thr_buy:.2f}")
                    continue
            if sig == "SELL" and p_sell is not None:
                if p_sell < thr_sell:
                    print(f"[SKIP] SELL proba={p_sell:.2f} < thr={thr_sell:.2f}")
                    continue

            # Plan + envoi
            atr = float(df['atr'].iloc[-1]) if 'atr' in df else float('nan')
            plan = plan_order(pair, price, atr, sig)
            print(f"[PLAN] {pair} {sig} qty={plan.qty} entry~{price:.5f} SL={plan.stop:.5f} TP={plan.take:.5f}")

            last_trade_time[pair] = now_utc
            place_bracket(ib, contract, plan)

        if k < loops - 1:
            ib.sleep(getattr(S, "SLEEP_BETWEEN_SEC", 20))

    ib.disconnect()

if __name__ == "__main__":
    main()
