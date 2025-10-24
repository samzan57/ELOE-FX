# scripts/download_fx_csv.py
import  os
import pandas as pd
from ib_insync import *
from config import settings as S


OUT_DIR = os.path.join("data", "historical")
OUT_FILE = os.path.join(OUT_DIR, "EURUSD_1min.csv")

def download_fx_csv(pair="EURUSD", duration=None, barSize=None, whatToShow=None):
    duration = duration or f"{getattr(S, 'BAR_DURATION_DAYS', 2)} D"
    barSize = barSize or getattr(S, "BAR_SIZE", "1 min")
    whatToShow = whatToShow or getattr(S, "WHAT_TO_SHOW", "MIDPOINT")

    ib = IB()
    ib.connect(S.HOST, S.PORT, clientId=getattr(S, "CLIENT_ID", 2))

    contract = Forex(pair)
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=barSize,
        whatToShow=whatToShow,
        useRTH=False,
        formatDate=1
    )

    df = util.df(bars)
    if df.empty:
        print("[WARN] Aucune donnée reçue.")
        ib.disconnect(); return

    df.rename(columns={"date": "datetime"}, inplace=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_FILE):
        old = pd.read_csv(OUT_FILE, parse_dates=["datetime"])
        df = pd.concat([old, df], ignore_index=True)
        df.drop_duplicates(subset=["datetime"], keep="last", inplace=True)
        df.sort_values("datetime", inplace=True)

    df.to_csv(OUT_FILE, index=False)
    print(f"[OK] {pair} -> {OUT_FILE} ({len(df)} lignes au total)")
    ib.disconnect()

if __name__ == "__main__":
    download_fx_csv()
