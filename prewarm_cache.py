# prewarm_cache.py
# Prewarm price cache and metadata for Super50 V2.0
import time, json, requests, yfinance as yf, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path(__file__).parent
CACHE = BASE / "cache_super50_v2"
PRICES = CACHE / "prices"
META = CACHE / "meta.json"
CACHE.mkdir(parents=True, exist_ok=True)
PRICES.mkdir(parents=True, exist_ok=True)

YEARS = 5
BATCH = 40
SLEEP = 0.8

NSE_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
]

def fetch_nse_symbols():
    s=set(); headers={"User-Agent":"Mozilla/5.0"}
    for u in NSE_URLS:
        try:
            r=requests.get(u, headers=headers, timeout=10)
            if r.status_code==200 and r.text:
                for line in r.text.splitlines()[1:]:
                    parts=line.split(",")
                    if parts and parts[0].strip(): s.add(parts[0].strip().upper())
        except: pass
    return sorted([x for x in s if x.isalnum()])

def price_path(tk): return PRICES / f"{tk.replace('/','_')}.parquet"

def have_price(tk, max_age_hours=72):
    p = price_path(tk)
    if not p.exists(): return False
    return (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() < max_age_hours*3600

def load_meta():
    if META.exists():
        try:
            with open(META,"r",encoding="utf-8") as f: return json.load(f)
        except: pass
    return {}

def save_meta(m):
    try:
        with open(META,"w",encoding="utf-8") as f: json.dump(m,f,indent=2,ensure_ascii=False)
    except: pass

def fetch_meta_batch(tickers):
    meta = load_meta(); need=[t for t in tickers if t not in meta]
    if not need: return meta
    def w(tk):
        try:
            info = yf.Ticker(tk).info
            ed = info.get("earningsTimestamp") or info.get("earningsDate")
            if isinstance(ed,(int,float)):
                try: ed = datetime.fromtimestamp(int(ed)).strftime("%Y-%m-%d")
                except: ed=None
            return tk, {"marketCap": info.get("marketCap",0) or 0, "sector": info.get("sector") or info.get("industry") or "Other", "name": info.get("shortName") or info.get("longName") or tk, "earningsDate": ed}
        except:
            return tk, {"marketCap":0,"sector":"Other","name":tk,"earningsDate":None}
    with ThreadPoolExecutor(max_workers=18) as ex:
        futures=[ex.submit(w,t) for t in need]
        for f in as_completed(futures):
            try:
                tk,md=f.result(); meta[tk]=md
            except: pass
    save_meta(meta); return meta

def download_prices(tickers):
    end = datetime.now(); start = end - timedelta(days=365*YEARS)
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        try:
            df = yf.download(batch, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), group_by="ticker", threads=True, progress=False)
        except: df = {}
        for tk in batch:
            p = price_path(tk)
            if p.exists(): continue
            try:
                sub = df[tk] if (isinstance(df, dict) and tk in df) else yf.download(tk, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
                if isinstance(sub, pd.DataFrame) and not sub.empty:
                    sub2 = sub[['Open','High','Low','Close','Volume']].dropna()
                    if not sub2.empty: sub2.to_parquet(p)
            except: pass
        time.sleep(SLEEP)

def main():
    print("Prewarm started...")
    syms = fetch_nse_symbols(); tickers = [s + ".NS" for s in syms]
    meta = fetch_meta_batch(tickers)
    ranked = sorted([(tk, meta.get(tk,{}).get("marketCap",0)) for tk in tickers], key=lambda x: x[1], reverse=True)
    top = [tk for tk,_ in ranked[:500 if len(ranked)>=500 else len(ranked)]]
    print(f"Top universe for caching: {len(top)}")
    to_dl = [t for t in top if not have_price(t)]
    print(f"Downloading {len(to_dl)} price files...")
    if to_dl: download_prices(to_dl)
    print("Prewarm finished. Meta saved at:", META)

if __name__=="__main__":
    main()
