#!/usr/bin/env python
# scripts/download_fx_csv.py
"""
Télécharge les données historiques Forex 1-min depuis Dukascopy.

Source : https://datafeed.dukascopy.com  (gratuit, données tick qualité institutionnelle)
Format sortie : CSV avec colonnes datetime,open,high,low,close,volume
                Compatible avec tous les autres scripts du projet.

Fonctionnement :
  - Télécharge les fichiers tick heure par heure (format .bi5 = LZMA + binaire)
  - Calcule mid = (ask + bid) / 2
  - Rééchantillonne en OHLCV 1-min
  - Reprend là où il s'est arrêté (resume automatique)

Usage :
  python scripts/download_fx_csv.py                          # EURUSD 2 ans
  python scripts/download_fx_csv.py --pair EURUSD --years 3
  python scripts/download_fx_csv.py --pair GBPUSD --years 1
"""

import argparse
import lzma
import struct
import time
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Racine du projet dans PYTHONPATH ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─────────────────────────────────────────────
#  Configuration Dukascopy
# ─────────────────────────────────────────────
BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Valeur du point par paire (diviseur pour retrouver le prix réel)
POINT_VALUE = {
    "EURUSD": 100_000,
    "GBPUSD": 100_000,
    "USDJPY": 1_000,
    "USDCHF": 100_000,
    "AUDUSD": 100_000,
    "NZDUSD": 100_000,
    "USDCAD": 100_000,
    "EURGBP": 100_000,
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ELOE-FX data downloader)",
    "Referer":    "https://www.dukascopy.com/",
}

OUT_DIR = Path("data/historical")


# ─────────────────────────────────────────────
#  Téléchargement & parsing d'un fichier .bi5
# ─────────────────────────────────────────────
def _fetch_hour(pair: str, year: int, month: int, day: int, hour: int,
                session: requests.Session, retries: int = 3) -> bytes | None:
    # Dukascopy : le mois est 0-indexé (janvier = 00)
    url = (f"{BASE_URL}/{pair}/{year}/{month - 1:02d}/{day:02d}"
           f"/{hour:02d}h_ticks.bi5")
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200 and len(r.content) > 0:
                return r.content
            if r.status_code == 404:
                return None      # pas de données cette heure (week-end, férié)
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def _parse_bi5(raw_bytes: bytes, base_dt: datetime,
               point_value: int) -> list[tuple]:
    """Décompresse et parse un .bi5 → liste de (timestamp, ask, bid, ask_vol, bid_vol)."""
    if not raw_bytes:
        return []
    try:
        raw = lzma.decompress(raw_bytes)
    except lzma.LZMAError:
        return []

    record_size = 20
    n = len(raw) // record_size
    if n == 0:
        return []

    ticks = []
    for i in range(n):
        off = i * record_size
        ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(
            ">IIIff", raw[off: off + record_size]
        )
        ts      = base_dt + timedelta(milliseconds=int(ms))
        ask     = ask_raw / point_value
        bid     = bid_raw / point_value
        mid     = (ask + bid) / 2.0
        ticks.append((ts, mid, ask_vol + bid_vol))
    return ticks


# ─────────────────────────────────────────────
#  Rééchantillonnage ticks → OHLCV 1-min
# ─────────────────────────────────────────────
def _ticks_to_ohlcv(ticks: list[tuple]) -> pd.DataFrame:
    if not ticks:
        return pd.DataFrame()

    df = pd.DataFrame(ticks, columns=["datetime", "mid", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    ohlcv = df["mid"].resample("1min").ohlc()
    ohlcv["volume"] = df["volume"].resample("1min").sum()
    ohlcv = ohlcv.dropna(subset=["open"])
    return ohlcv.reset_index()


# ─────────────────────────────────────────────
#  Boucle principale
# ─────────────────────────────────────────────
def download(pair: str = "EURUSD", years: int = 2,
             output: str | None = None) -> Path:

    if pair not in POINT_VALUE:
        raise ValueError(f"Paire non supportée : {pair}. Supportées : {list(POINT_VALUE)}")

    out_file = Path(output) if output else OUT_DIR / f"{pair}_1min.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pv = POINT_VALUE[pair]

    # Plage de dates
    end_dt   = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=years * 365)

    # Reprise : lire la dernière date déjà téléchargée
    if out_file.exists():
        try:
            existing = pd.read_csv(out_file, usecols=["datetime"],
                                   parse_dates=["datetime"])
            last_dt = pd.to_datetime(existing["datetime"]).max()
            if last_dt.tzinfo is None:
                last_dt = last_dt.tz_localize("UTC")
            resume_dt = last_dt.to_pydatetime().replace(minute=0, second=0)
            print(f"  Reprise depuis {resume_dt.date()} (fichier existant : {len(existing):,} barres)")
            start_dt = resume_dt
        except Exception:
            pass

    # Génération de toutes les heures à télécharger
    hours = []
    cur = start_dt
    while cur < end_dt:
        # Dukascopy a des données 24h/24 week-end inclus (mais vides sam/dim)
        hours.append(cur)
        cur += timedelta(hours=1)

    total   = len(hours)
    all_dfs = []
    session = requests.Session()

    print(f"\n  Dukascopy → {pair} 1-min | {start_dt.date()} → {end_dt.date()}")
    print(f"  Fichiers à télécharger : ~{total:,} (heures)\n")

    downloaded = 0
    skipped    = 0

    for i, dt in enumerate(hours):
        raw = _fetch_hour(pair, dt.year, dt.month, dt.day, dt.hour,
                          session)
        ticks = _parse_bi5(raw, dt, pv) if raw else []

        if ticks:
            df_h = _ticks_to_ohlcv(ticks)
            if not df_h.empty:
                all_dfs.append(df_h)
                downloaded += len(df_h)

        else:
            skipped += 1

        # Progression
        if (i + 1) % 50 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"\r  [{bar}] {pct:5.1f}%  barres={downloaded:,}", end="", flush=True)

        # Politesse : ~4 requêtes/seconde max
        time.sleep(0.25)

    print()   # newline après la barre de progression

    if not all_dfs:
        print("  [ERREUR] Aucune donnée téléchargée.")
        return out_file

    # ── Fusion avec l'existant ────────────────────────────────────────────
    new_df = pd.concat(all_dfs, ignore_index=True)
    new_df.columns = ["datetime", "open", "high", "low", "close", "volume"]
    new_df["datetime"] = pd.to_datetime(new_df["datetime"], utc=True)

    if out_file.exists():
        old_df = pd.read_csv(out_file, parse_dates=["datetime"])
        if old_df["datetime"].dt.tz is None:
            old_df["datetime"] = old_df["datetime"].dt.tz_localize("UTC")
        new_df = pd.concat([old_df, new_df], ignore_index=True)

    new_df = (new_df
              .drop_duplicates(subset=["datetime"])
              .sort_values("datetime")
              .reset_index(drop=True))

    new_df.to_csv(out_file, index=False)

    print(f"\n  ✓ Sauvegardé → {out_file}")
    print(f"  Total barres : {len(new_df):,}")
    print(f"  Période      : {new_df['datetime'].iloc[0]}  →  {new_df['datetime'].iloc[-1]}")
    return out_file


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Télécharge les données historiques Forex depuis Dukascopy"
    )
    parser.add_argument("--pair",   default="EURUSD",
                        help="Paire Forex (défaut: EURUSD)")
    parser.add_argument("--years",  type=int, default=2,
                        help="Nombre d'années d'historique (défaut: 2)")
    parser.add_argument("--output", default=None,
                        help="Chemin du fichier CSV de sortie")
    args = parser.parse_args()

    download(pair=args.pair, years=args.years, output=args.output)


if __name__ == "__main__":
    main()
