# config/settings.py  (v2 — rebuild institutionnel)

import os
from dotenv import load_dotenv

load_dotenv()

# ===== Connexion IBKR =====
# Host/port/client ID configurables via .env (voir .env.example) — le port
# dépend de ta configuration (paper/live, TWS/Gateway), voir la documentation IBKR.
HOST = os.getenv("IBKR_HOST", "127.0.0.1")
PORT = os.getenv("IBKR_PORT")
if not PORT:
    raise ValueError("IBKR_PORT non défini — renseigne-le dans ton .env (voir .env.example).")
PORT = int(PORT)
CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "15"))
IB_RESQUEST_TIMEOUT = 30
IB_CONNECT_READONLY = False
IB_ACCOUNT = None              # ex "DU1234567" ou None pour 1er compte

# ===== Compte & Risque =====
CAPITAL = 10_000
RISK_PER_TRADE = 0.01          # 1% max par trade (plafond Kelly)
RR = 2.0                       # ratio TP/SL
KELLY_FRACTION = 0.25          # Kelly fractionnel 25% (standard quant)

# ===== Marchés & Session =====
PAIRS = ["EURUSD"]
TZ = "Europe/Paris"
SESSION_START = (9, 0)         # 9h Paris = ouverture Londres
SESSION_END   = (18, 0)        # 18h Paris = fermeture NY

# ===== Indicateurs / Stops =====
ATR_LEN = 14
ATR_K = 1.5                    # stop = ATR × ATR_K
MIN_STOP_PIPS = 8

# ===== Paramètres signaux (tunable) =====
SIGNAL_VWAP_Z     = 1.6        # distance VWAP — plus sélectif (1.2 → 1.6)
SIGNAL_RSI_LO     = 35.0       # RSI oversold — seulement les extrêmes (40 → 35)
SIGNAL_RSI_HI     = 65.0       # RSI overbought — seulement les extrêmes (60 → 65)
ADX_MAX           = 25.0       # filtre tendance plus strict (28 → 25)
ADX_MIN_MOMENTUM  = 18.0       # ADX minimum pour signaux momentum

# ===== Sécurité =====
IGNORE_SESSION    = False
DRY_RUN           = False
MAX_DAILY_DRAWDOWN = 0.02      # 2% du capital → kill switch
MAX_SPREAD_PIPS   = 0.30       # spread max toléré en pips
RUN_LOOP_MINUTES  = 480        # durée de la boucle live (8h = session complète)
SLEEP_BETWEEN_SEC = 20         # scan toutes les 20s

# ===== Marché / Données =====
BAR_DURATION_DAYS = 2          # historique intraday (2 jours pour VWAP stable)
BAR_SIZE          = "1 min"
WHAT_TO_SHOW      = "MIDPOINT"
PRICE_TIMEOUT_SEC = 30.0

# ===== Ordres =====
PARENT_ORDER_TYPE = "MKT"
TAKE_ORDER_TYPE   = "LMT"
STOP_ORDER_TYPE   = "STP"
TIF = "GTC"

# ===== Sizing =====
MIN_QTY  = 1_000
QTY_STEP = 1_000

# ===== Gestion exposition =====
MAX_CONCURRENT_POSITIONS = 1
REENTER_COOLDOWN_MIN = 15

# ===== Modèles ML (LightGBM v2) =====
MODEL_PATH      = "models/model_eurusd_buy.pkl"
SELL_MODEL_PATH = "models/model_eurusd_sell.pkl"

# ===== Kill-switch =====
USE_KILL_SWITCH  = True
KILL_MODE        = "netliq"    # ‘auto’ | ‘pnl’ | ‘netliq’ | ‘off’
KILL_REF         = "netliq"
KILL_SANITY_MULT = 0.25
KILL_CONFIRM_SCANS = 2

# --- Note DST ---
# SESSION_START/END sont en heure locale Europe/Paris (gère été/hiver automatiquement).
