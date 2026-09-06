# ELOE-FX

**Système de trading algorithmique EUR/USD institutionnel : signaux VWAP + classification LightGBM, sizing Kelly fractionnel, backtest walk-forward purgé, exécution IBKR.**

![Courbe de capital du backtest](data/logs/backtest_equity.png)

---

## 🇫🇷 Version française

### Aperçu

ELOE-FX est un système de trading intraday sur EUR/USD qui va du signal à l'exécution, avec un souci constant de réalisme (spread, slippage, délai d'exécution, coûts) plutôt que d'un backtest optimiste :

1. **Signal** — mean-reversion sur écart au VWAP, confirmé par RSI/ADX, combiné à une classification binaire LightGBM (probabilité que le take-profit soit atteint avant le stop-loss)
2. **Labeling réaliste** — entrée au prix OPEN de la bougie suivante (pas au close, qui serait un biais de type "regarder dans le futur"), spread et slippage modélisés
3. **Validation** — Purged Walk-Forward Cross-Validation, avec un gap entre train et test pour éviter la fuite de données entre labels qui se chevauchent
4. **Sizing** — Kelly fractionnel (25%), plafonné à 1% de risque max par trade
5. **Circuit-breakers** — arrêt sur drawdown journalier, cooldown après pertes consécutives, filtre de spread
6. **Exécution** — connexion IBKR (paper trading), notifications Telegram

### Résultat clé

Backtest réaliste (entrée OPEN+1, spread+slippage inclus) sur données EUR/USD 1-minute, période out-of-sample :

| Métrique | Valeur |
|---|---|
| Trades | 34 |
| Win rate | 52.9% |
| Profit factor | 2.01 |
| Total | +129.4 pips |
| Max drawdown | -24.0 pips |

Sur cet échantillon, la stratégie est profitable avec un profit factor > 2 (les gains cumulés valent plus de 2x les pertes cumulées). **Point de vigilance assumé** : sur seulement 34 trades, les métriques annualisées (Sharpe, Sortino, Calmar) calculées par `backtests/metrics.py` deviennent statistiquement peu fiables — l'extrapolation à l'année d'un échantillon aussi court peut gonfler artificiellement un Sharpe apparent. Je préfère le dire plutôt que d'afficher un chiffre impressionnant mais trompeur : la vraie preuve de robustesse viendra d'un nombre de trades plus élevé (voir *Limites*).

### Méthodologie détaillée

**Signal (`strategies/signals.py`)**
- Mean-reversion sur le VWAP : quand le prix s'éloigne significativement de sa référence d'exécution institutionnelle, les ordres de rééquilibrage créent une force de rappel
- Confirmation par oscillateur (RSI) et filtre de tendance (ADX < seuil pour éviter le contre-trend en marché directionnel)

**Modèle (`models/`)**
- Classification binaire LightGBM : P(TP atteint avant SL) pour le sens long et le sens short séparément
- `models/train.py` : **Purged Walk-Forward Cross-Validation** — la série est découpée en folds consécutifs avec un gap d'une semaine entre train et test pour éviter le chevauchement de labels (fenêtres de labeling qui regardent plusieurs bougies en avant)

**Labeling réaliste (`backtests/labeling.py`)**
- Entrée au prix OPEN de la bougie i+1 (délai d'exécution réaliste, pas au close de la bougie de signal)
- Spread (0.20 pip) et slippage (0.05 pip) modélisés
- TP/SL dynamiques basés sur l'ATR

**Moteur de backtest (`backtests/engine.py`)**
- Une seule position à la fois, cooldown après un stop-loss
- Sortie sur TP, SL, ou expiration de l'horizon

**Sizing & risque (`strategies/risk.py`)**
- Kelly fractionnel : `f = 0.25 × (p×(rr+1) - 1) / rr`, plafonné à 1% du capital par trade
- Circuit-breakers : drawdown journalier max 2%, cooldown après pertes consécutives, filtre de spread max

**Exécution (`strategies/Ibkr_fx_intraday.py`)**
- Connexion IBKR (paper trading), notifications Telegram, automatisation via `run_bot.ps1`

### Structure du projet

```
ELOE-FX/
├── config/
│   └── settings.py          # Paramètres centralisés (risque, session, IBKR)
├── features/
│   └── technical.py         # Features multi-horizon (RSI, ATR, VWAP, momentum)
├── strategies/
│   ├── signals.py           # Signal VWAP mean-reversion + confirmation
│   ├── risk.py               # Kelly fractionnel, circuit-breakers
│   ├── indicators.py         # RSI, ATR, ADX, VWAP
│   ├── utils.py               # Utilitaires (pip_size, etc.)
│   ├── telegram_notify.py     # Notifications Telegram
│   └── Ibkr_fx_intraday.py    # Boucle live (IBKR paper trading)
├── models/
│   ├── lgbm_model.py          # Wrapper LightGBM
│   ├── proba_model.py         # Modèles de probabilité (logistique / RF)
│   ├── train.py                # Purged Walk-Forward CV
│   ├── model_eurusd_buy.pkl    # Modèle entraîné (sens long)
│   └── model_eurusd_sell.pkl   # Modèle entraîné (sens short)
├── backtests/
│   ├── labeling.py             # Labeling réaliste (OPEN+1, spread, slippage)
│   ├── engine.py                # Moteur de backtest
│   ├── metrics.py               # Sharpe, Sortino, Calmar, profit factor
│   └── optimizer.py              # Recherche d'hyperparamètres
├── scripts/
│   ├── download_fx_csv.py      # Téléchargement Dukascopy (1-min, gratuit)
│   ├── train_model.py           # Entraînement complet
│   └── run_backtest.py          # Backtest + export trades/graphique
├── notebooks/                  # Notebooks d'analyse et de backtest
├── data/
│   ├── historical/              # CSV de prix (EURUSD_1min.csv)
│   └── logs/                    # Résultats livrés (backtest_trades.csv, courbe de capital)
├── run_bot.ps1                  # Lance la stratégie en continu (Windows)
├── requirements.txt
├── .env.example
└── LICENSE
```

### Installation

```bash
git clone <url-du-dépôt>
cd ELOE-FX
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # requis pour l'exécution live (IBKR) et Telegram
```

### Utilisation

```bash
# Télécharger des données historiques (ex : EURUSD, 2 ans, gratuit via Dukascopy)
python scripts/download_fx_csv.py --pair EURUSD --years 2

# Entraîner les modèles (Purged Walk-Forward CV)
python scripts/train_model.py --pair EURUSD --data data/historical/EURUSD_1min.csv

# Backtest réaliste sur période out-of-sample
python scripts/run_backtest.py --data data/historical/EURUSD_1min.csv

# Lancer le bot en continu (Windows, nécessite TWS/IB Gateway + .env configuré)
.\run_bot.ps1
```

### Tests

```bash
pytest tests/
```

Les tests couvrent le sizing Kelly (EV négative → pas de trade, plafond de risque respecté), les circuit-breakers (drawdown, spread, pertes consécutives), et les métriques de performance (win rate, profit factor, drawdown) sur un jeu de trades connu.

### Stack technique

Python · pandas · NumPy · LightGBM · scikit-learn · ib_insync · Matplotlib · pytest

### Limites & pistes d'amélioration

- Échantillon de backtest réduit (34 trades) — les métriques annualisées (Sharpe/Sortino/Calmar) ne sont pas statistiquement robustes à cette taille ; il faudrait accumuler plusieurs mois/années de données pour des conclusions fiables.
- Un seul actif (EUR/USD) — le framework (labeling, purged CV, Kelly sizing) est généralisable à d'autres paires.
- Testé exclusivement en paper trading IBKR — **ce n'est pas un conseil en investissement**.

---

## 🇬🇧 English version

### Overview

ELOE-FX is an intraday EUR/USD trading system that goes from signal to execution, with a constant focus on realism (spread, slippage, execution delay, costs) rather than an optimistic backtest:

1. **Signal** — VWAP mean-reversion, confirmed by RSI/ADX, combined with a binary LightGBM classifier (probability that take-profit is hit before stop-loss)
2. **Realistic labeling** — entry at the OPEN of the next candle (not the close, which would be a look-ahead bias), spread and slippage modeled
3. **Validation** — Purged Walk-Forward Cross-Validation, with a gap between train and test to avoid data leakage between overlapping labels
4. **Sizing** — fractional Kelly (25%), capped at 1% max risk per trade
5. **Circuit-breakers** — stop on daily drawdown, cooldown after consecutive losses, spread filter
6. **Execution** — IBKR connection (paper trading), Telegram notifications

### Key result

Realistic backtest (OPEN+1 entry, spread+slippage included) on 1-minute EUR/USD data, out-of-sample period:

| Metric | Value |
|---|---|
| Trades | 34 |
| Win rate | 52.9% |
| Profit factor | 2.01 |
| Total | +129.4 pips |
| Max drawdown | -24.0 pips |

Over this sample, the strategy is profitable with a profit factor above 2 (cumulative gains are worth more than 2x cumulative losses). **An honest caveat**: with only 34 trades, the annualized metrics (Sharpe, Sortino, Calmar) computed by `backtests/metrics.py` become statistically unreliable — extrapolating such a short sample to a full year can artificially inflate an apparent Sharpe ratio. I'd rather say so than show an impressive-looking but misleading number: real proof of robustness will come from a larger trade count (see *Limitations*).

### Detailed methodology

**Signal (`strategies/signals.py`)**
- VWAP mean-reversion: when price moves significantly away from its institutional execution reference, rebalancing flows create a pull-back force
- Confirmed by an oscillator (RSI) and a trend filter (ADX below a threshold to avoid counter-trend trades in a directional market)

**Model (`models/`)**
- Binary LightGBM classification: P(TP hit before SL) for the long and short direction separately
- `models/train.py`: **Purged Walk-Forward Cross-Validation** — the series is split into consecutive folds with a one-week gap between train and test to avoid overlapping-label leakage (labeling windows that look several candles ahead)

**Realistic labeling (`backtests/labeling.py`)**
- Entry at the OPEN of candle i+1 (realistic execution delay, not the close of the signal candle)
- Spread (0.20 pip) and slippage (0.05 pip) modeled
- ATR-based dynamic TP/SL

**Backtest engine (`backtests/engine.py`)**
- One position at a time, cooldown after a stop-loss
- Exit on TP, SL, or horizon expiration

**Sizing & risk (`strategies/risk.py`)**
- Fractional Kelly: `f = 0.25 × (p×(rr+1) - 1) / rr`, capped at 1% of capital per trade
- Circuit-breakers: max 2% daily drawdown, cooldown after consecutive losses, max spread filter

**Execution (`strategies/Ibkr_fx_intraday.py`)**
- IBKR connection (paper trading), Telegram notifications, automated via `run_bot.ps1`

### Project structure

```
ELOE-FX/
├── config/
│   └── settings.py          # Centralized settings (risk, session, IBKR)
├── features/
│   └── technical.py         # Multi-horizon features (RSI, ATR, VWAP, momentum)
├── strategies/
│   ├── signals.py           # VWAP mean-reversion signal + confirmation
│   ├── risk.py               # Fractional Kelly, circuit-breakers
│   ├── indicators.py         # RSI, ATR, ADX, VWAP
│   ├── utils.py               # Utilities (pip_size, etc.)
│   ├── telegram_notify.py     # Telegram notifications
│   └── Ibkr_fx_intraday.py    # Live loop (IBKR paper trading)
├── models/
│   ├── lgbm_model.py          # LightGBM wrapper
│   ├── proba_model.py         # Probability models (logistic / RF)
│   ├── train.py                # Purged Walk-Forward CV
│   ├── model_eurusd_buy.pkl    # Trained model (long side)
│   └── model_eurusd_sell.pkl   # Trained model (short side)
├── backtests/
│   ├── labeling.py             # Realistic labeling (OPEN+1, spread, slippage)
│   ├── engine.py                # Backtest engine
│   ├── metrics.py               # Sharpe, Sortino, Calmar, profit factor
│   └── optimizer.py              # Hyperparameter search
├── scripts/
│   ├── download_fx_csv.py      # Dukascopy download (1-min, free)
│   ├── train_model.py           # Full training pipeline
│   └── run_backtest.py          # Backtest + trades/chart export
├── notebooks/                  # Analysis and backtest notebooks
├── data/
│   ├── historical/              # Price CSVs (EURUSD_1min.csv)
│   └── logs/                    # Delivered results (backtest_trades.csv, equity curve)
├── run_bot.ps1                  # Runs the strategy continuously (Windows)
├── requirements.txt
├── .env.example
└── LICENSE
```

### Installation

```bash
git clone <repo-url>
cd ELOE-FX
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # required for live execution (IBKR) and Telegram
```

### Usage

```bash
# Download historical data (e.g. EURUSD, 2 years, free via Dukascopy)
python scripts/download_fx_csv.py --pair EURUSD --years 2

# Train the models (Purged Walk-Forward CV)
python scripts/train_model.py --pair EURUSD --data data/historical/EURUSD_1min.csv

# Realistic backtest on the out-of-sample period
python scripts/run_backtest.py --data data/historical/EURUSD_1min.csv

# Run the bot continuously (Windows, requires TWS/IB Gateway + configured .env)
.\run_bot.ps1
```

### Tests

```bash
pytest tests/
```

Tests cover Kelly sizing (negative EV → no trade, risk cap respected), circuit-breakers (drawdown, spread, consecutive losses), and performance metrics (win rate, profit factor, drawdown) on a known set of trades.

### Tech stack

Python · pandas · NumPy · LightGBM · scikit-learn · ib_insync · Matplotlib · pytest

### Limitations & next steps

- Small backtest sample (34 trades) — annualized metrics (Sharpe/Sortino/Calmar) aren't statistically robust at this size; several months/years of data would be needed for reliable conclusions.
- Single asset (EUR/USD) — the framework (labeling, purged CV, Kelly sizing) generalizes to other pairs.
- Tested exclusively in IBKR paper trading — **this is not investment advice**.

---

## Author

**Deo ZANTOKO** — Engineering student in Applied Mathematics, Mathematical Modelling for Finance & Insurance (MMFA), CY Tech

## License

MIT — see [LICENSE](LICENSE).
