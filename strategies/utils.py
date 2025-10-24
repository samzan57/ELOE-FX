# strategies/utils.py
from ib_insync import *
from datetime import datetime, time as dtime
import pytz
import numpy as np
import pandas as pd
from pathlib import Path
import time
from config import settings as S

# --- Chemins ---
DATA_DIR = Path("data")
LOG_DIR = DATA_DIR / "logs"
HIST_DIR = DATA_DIR / "historical"
LOG_TRADES = LOG_DIR / "trades.csv"
LOG_PNL = LOG_DIR / "pnl.csv"

def ensure_paths():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_TRADES.exists():
        LOG_TRADES.write_text("timestamp,pair,side,qty,entry,stop,take,status,info\n", encoding="utf-8")
    if not LOG_PNL.exists():
        LOG_PNL.write_text("timestamp,day_pnl\n", encoding="utf-8")

# --- Timezone & session ---
def now_in_tz(tz: str):
    return datetime.now(pytz.timezone(tz))

def in_session(dt: datetime, tz: str, start_tuple, end_tuple):
    """
    dt: datetime (naive ou aware). On convertit dans `tz`.
    start_tuple / end_tuple: (hour, minute)
    """
    zone = pytz.timezone(tz)
    if dt.tzinfo is None:
        dt = zone.localize(dt)
    else:
        dt = dt.astimezone(zone)
    start_naive = datetime(dt.year, dt.month, dt.day, start_tuple[0], start_tuple[1])
    end_naive   = datetime(dt.year, dt.month, dt.day, end_tuple[0], end_tuple[1])
    tstart = zone.localize(start_naive)
    tend   = zone.localize(end_naive)
    return tstart <= dt <= tend

# --- Marché ---
def make_fx_contract(pair: str) -> Contract:
    return Forex(pair, exchange='IDEALPRO')

def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001

def pip_value_per_unit(pair: str) -> float:
    # Pour XXXUSD (EURUSD, GBPUSD...) : approx pip value par unité = pip_size (USD)
    return pip_size(pair)

# --- Prix & Historique ---
def wait_for_price(ib: IB, contract: Contract, timeout: float = None):
    """
    Récupère un prix 'raisonnable' (mid, last, ou moyenne bid/ask) dans la fenêtre 'timeout'.
    Annule proprement l'abonnement market data pour éviter le warning cancelMktData.
    """
    if timeout is None:
        timeout = getattr(S, "PRICE_TIMEOUT_SEC", 5.0)

    ticker = ib.reqMktData(contract, genericTickList='', snapshot=False)
    start = time.time()
    price = None
    while time.time() - start < timeout:
        ib.waitOnUpdate(timeout=1.0)
        price = ticker.midpoint()
        if price is None or not np.isfinite(price):
            price = ticker.last
        if price is None or not np.isfinite(price):
            if ticker.bid is not None and ticker.ask is not None:
                price = (ticker.bid + ticker.ask) / 2.0
        if price is not None and np.isfinite(price) and price > 0:
            break

    # Annuler proprement si on a un abonnement actif
    try:
        if getattr(ticker, "tickerId", None):
            ib.cancelMktData(ticker)
    except Exception:
        pass

    return float(price) if price else None

def fetch_bars(
    ib: IB,
    contract: Contract,
    duration: str = None,
    barSize: str = None,
    tz: str = None,
    whatToShow: str = None,
):
    """
    Télécharge l'historique et renvoie un DataFrame OHLC indexé en timezone 'tz'.
    Gère correctement les index tz-aware (pas de tz_localize si déjà tz-aware).
    """
    # valeurs par défaut depuis settings
    duration = duration or f"{getattr(S, 'BAR_DURATION_DAYS', 2)} D"
    barSize = barSize or getattr(S, "BAR_SIZE", "1 min")
    tz = tz or getattr(S, "TZ", "Europe/Paris")
    whatToShow = whatToShow or getattr(S, "WHAT_TO_SHOW", "MIDPOINT")

    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=barSize,
        whatToShow=whatToShow,
        useRTH=False,
        formatDate=1,
        keepUpToDate=False
    )
    if not bars:
        return pd.DataFrame()

    df = util.df(bars)

    # S'assurer que la colonne temps est bien l’index
    if 'date' in df.columns:
        df = df.set_index('date')

    # Normaliser le fuseau sans planter si déjà tz-aware
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    df.index = idx.tz_convert(tz)

    return df[['open', 'high', 'low', 'close']]

# --- Logging ---
def log_trade(pair, side, qty, entry, stop, take, status, info=""):
    ensure_paths()
    line = f"{datetime.utcnow().isoformat()}Z,{pair},{side},{qty},{entry:.5f},{stop:.5f},{take:.5f},{status},{info}\n"
    with LOG_TRADES.open("a", encoding="utf-8") as f:
        f.write(line)
