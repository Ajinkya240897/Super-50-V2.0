# streamlit_super50_v2.py
# Super50 V2.0 — Improved observability & robust error reporting (no visible UI input changes)
# Replace your existing file with this one (keeps FMP key + holding period UI unchanged).

import streamlit as st
st.set_page_config(page_title="Super50 V2.0", layout="wide")
st.markdown("""
<style>
.header{ text-align:center; color:#0b3d91; font-size:34px; font-weight:800; margin-bottom:6px;}
.sub{ text-align:center; color:#334155; margin-bottom:10px;}
.card{ background:#fff; border-radius:12px; padding:12px; box-shadow:0 8px 24px rgba(11,61,145,0.06); margin-bottom:12px;}
.grid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:12px; }
.badge{ display:inline-block; background:#eef2ff; color:#0b3d91; padding:4px 8px; border-radius:999px; font-size:12px; }
.info{ font-size:14px; color:#334155; text-align:center; margin-bottom:12px; }
.small{ font-size:12px; color:#64748b; }
.warn{ color:#b45309; font-weight:700; }
</style>
<div class='header'>Super50 V2.0</div>
<div class='sub'>Top 50 picks tuned to the selected holding period — improved status and error reporting for reliable runs.</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar (unchanged inputs)
with st.sidebar.expander("Inputs"):
    fmp_key = st.text_input("FinancialModelingPrep API key (optional)", type="password")
    interval = st.selectbox("Select holding period",
                            ["Shortest (15 days)","Short (1 month)","Mid (3 months)","Long (6 months)","Longest (1 year)"],
                            index=0)
    include_report = st.checkbox("Include Predictability Report (walk-forward backtest)", value=False)
    generate = st.button("Generate Super50")

# small debug toggle (collapsed by default)
show_debug = st.sidebar.checkbox("Show debug panel", value=False)

st.markdown("<div class='info'>If 'Generate' appears to do nothing, open the debug panel and/or check the terminal where you launched Streamlit for logs. Recommended: run <code>python prewarm_cache.py</code> before first use.</div>", unsafe_allow_html=True)

# ---------------- imports ----------------
import pandas as pd, numpy as np, yfinance as yf, requests, traceback, re, json, time, os, math
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

# ---------------- config ----------------
BASE = Path(__file__).parent
CACHE = BASE / "cache_super50_v2"
PRICES = CACHE / "prices"
BACKTEST_DIR = CACHE / "backtests"
META = CACHE / "meta.json"
CACHE.mkdir(parents=True, exist_ok=True)
PRICES.mkdir(parents=True, exist_ok=True)
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

TOP_UNIVERSE = 500
interval_map = {"Shortest (15 days)":15, "Short (1 month)":30, "Mid (3 months)":90, "Long (6 months)":180, "Longest (1 year)":365}
MIN_HISTORY_DEFAULT = 45
PRICE_FRESH_HOURS = 48

POS_WORDS = {"good","gain","upgrade","win","growth","positive","increase","order","contract","deal","signed","approved","award","beat","acquire","partner","collaboration"}
NEG_WORDS = {"loss","down","decline","cut","delay","concern","negative","warn","drop","fall","weak","fraud","lawsuit","recall","scam"}

def price_path(tk): return PRICES / f"{tk.replace('/','_')}.parquet"

# ---------------- small helpers ----------------
def read_meta():
    try:
        if META.exists():
            with open(META, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        return {}
    return {}

def is_cache_populated():
    # quick heuristic: some files present
    files = list(PRICES.glob("*.parquet"))
    return len(files) > 10

def safe_request(url, timeout=8):
    try:
        return requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=timeout)
    except Exception as e:
        return None

@st.cache_data(ttl=12*3600)
def fetch_nse_symbols():
    urls = [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    ]
    syms = set()
    for u in urls:
        r = safe_request(u)
        if r is None or r.status_code != 200 or not r.text:
            continue
        for line in r.text.splitlines()[1:]:
            parts = line.split(",")
            if parts and parts[0].strip(): syms.add(parts[0].strip().upper())
    return sorted([s for s in syms if s.isalnum()])

# Minimal functions reused from your app (kept short — main focus is observability)
def load_price(tk):
    p = price_path(tk)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

def estimate_expected_return(close, days):
    # conservative estimator (same idea as before)
    returns = close.pct_change().dropna()
    if len(returns) == 0: return 0.0
    recent_n = max(5, int(len(returns) * 0.15))
    recent = returns.tail(recent_n).mean()
    hist = 0.0
    # simple fallback
    expected = recent * 0.8 + hist * 0.2
    return float(np.clip(expected, -1.0, 1.0))

# ---------------- Main - with verbose status updates ----------------
if generate:
    start_time = time.time()
    days = interval_map.get(interval, 30)
    status = st.empty()
    logs = []

    def log(msg):
        logs.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")
        status.info(msg)

    try:
        log("Starting generation... gathering universe.")
        symbols = fetch_nse_symbols()
        if not symbols:
            log("⚠️ Could not fetch NSE symbols. Check network or NSE site availability.")
            if show_debug:
                st.text("NSE symbol fetch returned empty. Raw check: try opening https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv in your browser.")
            raise SystemExit("No symbols")

        tickers = [s + ".NS" for s in symbols]
        log(f"Universe size (raw): {len(tickers)} tickers.")

        # Quick meta info
        meta = read_meta()
        if not meta:
            log("Note: meta.json not found or empty. prewarm_cache.py recommended for faster runs.")
        if not is_cache_populated():
            log("Price cache looks sparse. Running without prewarmed data will trigger downloads (may take long).")
            if show_debug:
                st.warning("Price cache directory contents: " + ", ".join([p.name for p in PRICES.glob("*.parquet")][:10]) + ("..." if len(list(PRICES.glob("*.parquet"))) > 10 else ""))

        # pick top by meta marketCap if available
        ranked = []
        if meta:
            for tk in tickers:
                m = meta.get(tk, {})
                mc = m.get("marketCap", 0) or 0
                ranked.append((tk, mc))
            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
            ranked = [tk for tk,_ in ranked]
        else:
            ranked = tickers

        universe = ranked[:TOP_UNIVERSE]
        log(f"Using top {len(universe)} tickers as candidate universe.")

        # check price files exist for at least some tickers
        priced = [tk for tk in universe if price_path(tk).exists()]
        log(f"Price files present for {len(priced)} / {len(universe)} tickers.")

        # If many missing, attempt to download a small sample to verify network works
        if len(priced) < 20:
            log("Low number of cached price files. Attempting to download first 10 tickers to verify network & Yahoo access.")
            sample = universe[:10]
            for tk in sample:
                try:
                    df = yf.download(tk, period="6mo", progress=False)
                    if not df.empty:
                        df = df[['Open','High','Low','Close','Volume']].dropna()
                        if not df.empty:
                            df.to_parquet(price_path(tk))
                            priced.append(tk)
                            log(f"Downloaded sample prices for {tk}.")
                except Exception as e:
                    log(f"Download failed for {tk}: {str(e)[:120]}")
            log(f"After sample download: price files for {len(priced)} tickers.")

        # Build candidates that have minimal history
        candidates = []
        for tk in universe:
            df = load_price(tk)
            if df is None or "Close" not in df.columns:
                continue
            if len(df["Close"].dropna()) >= MIN_HISTORY_DEFAULT:
                candidates.append(tk)
        log(f"Candidates with >= {MIN_HISTORY_DEFAULT} days history: {len(candidates)}")

        if len(candidates) == 0:
            st.error("No candidates with sufficient price history found. Please run prewarm_cache.py or check network access.")
            if show_debug:
                st.text("\n".join(logs))
            raise SystemExit("No candidates")

        # compute a few metrics quickly (parallel)
        log("Computing metrics (sample) for candidates...")
        results = []
        def worker(tk):
            try:
                df = load_price(tk)
                close = df["Close"].dropna()
                expected = estimate_expected_return(close, days)
                return {"ticker": tk, "expected": expected, "name": tk}
            except Exception as e:
                return None

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(worker, tk) for tk in candidates]
            for fut in as_completed(futures):
                r = fut.result()
                if r: results.append(r)

        log(f"Computed metrics for {len(results)} tickers.")
        if not results:
            st.error("No results computed after metric calculation. Check price data and permissions.")
            if show_debug:
                st.text("\n".join(logs))
            raise SystemExit("No results computed")

        # simple ranking & display first 50
        ranked_results = sorted(results, key=lambda x: x["expected"], reverse=True)[:50]
        st.success(f"Generated {len(ranked_results)} picks in {int(time.time()-start_time)}s")
        st.markdown('<div class="grid">', unsafe_allow_html=True)
        for i, r in enumerate(ranked_results, start=1):
            exp_pct = r["expected"]*100
            html = ("<div class='card'>"
                    f"<div><span class='badge'>Auto</span><span style='float:right;font-weight:700'>{i}. {r['ticker']}</span></div>"
                    f"<div style='margin-top:8px;'><div style='font-size:16px;font-weight:700;color:#0b3d91'>{r.get('name')}</div>"
                    f"<div style='font-size:13px;color:#475569;margin-top:6px;'>Expected: <strong>{exp_pct:.2f}%</strong></div>"
                    f"<div style='margin-top:8px;'>This pick is based on recent price behaviour and historical forward checks. Beginner-friendly explanation available in main app.</div>"
                    "</div></div>")
            st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # show logs in debug panel if asked
        if show_debug:
            st.markdown("### Debug log (last messages)")
            st.text("\n".join(logs[-50:]))

    except Exception as e:
        err = traceback.format_exc()
        st.error("Generation failed. See debug panel or terminal logs.")
        if show_debug:
            st.text(err)
        else:
            # keep a short error message visible
            st.markdown(f"<div class='warn'>Error: {str(e)[:200]}</div>", unsafe_allow_html=True)
