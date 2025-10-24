# config/settings.py

# ===== Connexion IBKR (Paper) =====
HOST = "127.0.0.1"
PORT = 7497           # 7497 = Paper, 7496 = Live
CLIENT_ID = 15
IB_RESQUEST_TIMEOUT = 30
IB_CONNECT_READONLY = False

# ===== Compte & Risque =====
CAPITAL = 10_000
RISK_PER_TRADE = 0.005         # 0.5% du capital par trade
RR = 2.0                       # ratio TP/SL optimisé

# ===== Marchés & Session =====
PAIRS = ["EURUSD"]             # mono-paire pour l’instant
TZ = "Europe/Paris"
SESSION_START = (14, 0)        # 14:00 Paris (été)
SESSION_END   = (18, 0)        # 18:00 Paris

# ===== Indicateurs / Stops (optimisés) =====
ATR_LEN = 14
RSI_LEN = 14
BREAKOUT_LOOKBACK = 30         # (garde ce nom, le code l’utilise)
ATR_K = 1.5
MIN_STOP_PIPS = 8
RSI_MIN = 55

# Filtre breakout
BREAKOUT_USE_HIGH_LOW = True     # True: casse sur high/low (plus réactif)
BREAKOUT_BUFFER_PIPS = 0.05      # buffer anti "touch-and-go"

# ===== Sécurité =====
IGNORE_SESSION = False  # temporaire pour tests
DRY_RUN = False
MAX_DAILY_DRAWDOWN = 0.02      # 2% du capital
MAX_SPREAD_PIPS = 0.21            # ex: 0.2 pip (EURUSD)
RUN_LOOP_MINUTES = 60        # écoute pendant 1h
SLEEP_BETWEEN_SEC = 20       # scan toutes les 20s

# ===== Marché / Données =====
BAR_DURATION_DAYS = 1          # historique à récupérer
BAR_SIZE = "1 min"             # granularité IBKR
WHAT_TO_SHOW = "MIDPOINT"      # ou 'BID_ASK'
PRICE_TIMEOUT_SEC = 30.0       # timeout pour attendre un prix live

# ===== Ordres =====
PARENT_ORDER_TYPE = "MKT"      # exécution rapide intraday
TAKE_ORDER_TYPE   = "LMT"
STOP_ORDER_TYPE   = "STP"
TIF = "GTC"

# ===== Sizing / Arrondis =====
MIN_QTY = 1000                 # taille mini (1k)
QTY_STEP = 1000                # pas d'arrondi (1k)

# ===== Gestion session / Exposition =====
MAX_CONCURRENT_POSITIONS = 1
REENTER_COOLDOWN_MIN = 15

# ===== Filtre volatilité (optionnel) =====
USE_VOL_FILTER = False
ATR_QTL = 0.4

# ===== Modèle ML =====
MODEL_PATH = "models/model_eurusd_buy.pkl"
MODEL_EV_THRESHOLD = 0.505     # seuil proba (EV>0) issu du notebook
SELL_MODEL_PATH = "models/model_eurusd_sell.pkl"
SELL_MODEL_EV_THRESHOLD = 0.505  # (sera remplacé par threshold du pickle si tu lis le champ)

# --- Kill-switch ---
USE_KILL_SWITCH = True
KILL_MODE = "netliq"           # 'auto' | 'pnl' | 'netliq' | 'off'
KILL_SANITY_MULT = 0.25      # si |dailyPnL| > 25% * CAPITAL => suspect, on ignore
KILL_CONFIRM_SCANS = 2       # nb de scans consécutifs sous seuil avant STOP
IB_ACCOUNT = None            # ex "DU1234567" ou None pour 1er compte
KILL_REF = "netliq"         # "capital" ou "netliq"



# --- Note DST ---
# SESSION_START/END sont en heure locale Europe/Paris (gère été/hiver automatiquement).
