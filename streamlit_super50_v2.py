# streamlit_super50_v2.py
# Super50 V2.0 — Complete ready-to-paste file
# Purpose: produce Top 50 picks tuned to chosen holding period with predictability/backtest features.
# Visible UI unchanged: FMP API key (optional), Holding period, optional Predictability Report checkbox.
# Internal: prewarm-aware, walk-forward backtest (optional), bootstrap forward stats, horizon-aware scoring,
# calibrated expected returns, liquidity gating, sector & confidence handling, beginner-friendly "why" text.

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
.monotable { width:100%; border-collapse:collapse; }
.monotable th, .monotable td { padding:6px 8px; border-bottom:1px solid #eee; text-align:left; }
</style>
<div class='header'>Super50 V2.0</div>
<div class='sub'>Top 50 picks tuned to the selected holding period — predictability checks available (optional).</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar (unchanged inputs) ----------------
with st.sidebar.expander("Inputs"):
    fmp_key = st.text_input("FinancialModelingPrep API key (optional)", type="password")
    interval = st.selectbox("Select holding period",
                            ["Shortest (15 days)","Short (1 month)","Mid (3 months)","Long (6 months)","Longest (1 year)"],
                            index=0)
    include_report = st.checkbox("Include Predictability Report (walk-forward backtest)", value=False, help="Optional: run the historical walk-forward predictive validation (slower).")
    generate = st.button("Generate Super50")

show_debug = st.sidebar.checkbox("Show debug panel", value=False)

st.markdown("<div class='info'>Tip: run <code>python prewarm_cache.py</code> before first use to cache price history. Predictability report is optional and more compute-intensive.</div>", unsafe_allow_html=True)

# ---------------- imports ----------------
import pandas as pd, numpy as np, yfinance as yf, requests, math, traceback, re, json, time, os
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

# ---------------- config / cache ----------------
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

# thresholds (enforced)
IDEAL_THRESH = {15:0.03, 30:0.02, 90:0.02, 180:0.01, 365:0.01}
FALLBACK_THRESH = {15:0.02, 30:0.01, 90:0.01, 180:0.005, 365:0.005}

MIN_HISTORY_DEFAULT = 45
PRICE_FRESH_HOURS = 48

POS_WORDS = {"good","gain","upgrade","win","growth","positive","increase","order","contract","deal","signed","approved","award","beat","acquire","partner","collaboration"}
NEG_WORDS = {"loss","down","decline","cut","delay","concern","negative","warn","drop","fall","weak","fraud","lawsuit","recall","scam"}

# ---------------- file helpers ----------------
def price_path(tk): return PRICES / f"{tk.replace('/','_')}.parquet"

@st.cache_data(ttl=12*3600)
def fetch_nse_symbols():
    urls = [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    ]
    syms = set()
    headers = {"User-Agent":"Mozilla/5.0"}
    for u in urls:
        try:
            r = requests.get(u, headers=headers, timeout=8)
            if r.status_code == 200 and r.text:
                for line in r.text.splitlines()[1:]:
                    parts = line.split(",")
                    if parts and parts[0].strip(): syms.add(parts[0].strip().upper())
        except:
            continue
    return sorted([s for s in syms if s.isalnum()])

def load_meta():
    try:
        if META.exists():
            with open(META,"r",encoding="utf-8") as f: return json.load(f)
    except:
        pass
    return {}

def save_meta(m):
    try:
        with open(META,"w",encoding="utf-8") as f: json.dump(m, f, indent=2, ensure_ascii=False)
    except:
        pass

def fetch_meta_batch(tickers):
    meta = load_meta()
    need = [t for t in tickers if t not in meta]
    if not need:
        return {t: meta.get(t,{}) for t in tickers}
    def worker(tk):
        try:
            info = yf.Ticker(tk).info
            ed = info.get("earningsTimestamp") or info.get("earningsDate")
            if isinstance(ed, (int, float)):
                try: ed = datetime.fromtimestamp(int(ed)).strftime("%Y-%m-%d")
                except: ed = None
            return tk, {"marketCap": info.get("marketCap",0) or 0,
                        "sector": info.get("sector") or info.get("industry") or "Other",
                        "name": info.get("shortName") or info.get("longName") or tk,
                        "earningsDate": ed}
        except:
            return tk, {"marketCap":0,"sector":"Other","name":tk,"earningsDate":None}
    with ThreadPoolExecutor(max_workers=18) as ex:
        futures = {ex.submit(worker, tk): tk for tk in need}
        for fut in as_completed(futures):
            try:
                tk, md = fut.result(); meta[tk] = md
            except:
                continue
    save_meta(meta); return {t: meta.get(t,{}) for t in tickers}

def price_fresh(tk, hours=PRICE_FRESH_HOURS):
    p = price_path(tk)
    if not p.exists(): return False
    try:
        return (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)) < timedelta(hours=hours)
    except:
        return False

def download_prices(tickers, years=5):
    if not tickers:
        return
    end = datetime.now()
    start = end - timedelta(days=365*years)
    for i in range(0, len(tickers), 60):
        batch = tickers[i:i+60]
        try:
            df = yf.download(batch, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), group_by='ticker', threads=True, progress=False)
        except:
            df = {}
        if len(batch) == 1:
            tk = batch[0]
            try:
                if isinstance(df, pd.DataFrame):
                    sub = df[['Open','High','Low','Close','Volume']].dropna()
                    if not sub.empty:
                        sub.to_parquet(price_path(tk))
            except:
                pass
        else:
            for tk in batch:
                try:
                    if isinstance(df, dict) or tk not in df:
                        single = yf.download(tk, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
                        if not single.empty:
                            sub = single[['Open','High','Low','Close','Volume']].dropna()
                            if not sub.empty:
                                sub.to_parquet(price_path(tk))
                    else:
                        sub = df[tk][['Open','High','Low','Close','Volume']].dropna()
                        if not sub.empty:
                            sub.to_parquet(price_path(tk))
                except Exception:
                    try:
                        single = yf.download(tk, start=start.strftime("%Y-%m-%d"), end=start.strftime("%Y-%m-%d"), progress=False)
                        if not single.empty:
                            sub = single[['Open','High','Low','Close','Volume']].dropna()
                            if not sub.empty:
                                sub.to_parquet(price_path(tk))
                    except Exception:
                        pass

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

# ---------------- FMP helpers (optional) ----------------
def fmp_profile(sym, key):
    if not key:
        return {}
    try:
        r = requests.get(f"https://financialmodelingprep.com/api/v3/profile/{sym}?apikey={key}", timeout=8)
        if r.status_code == 200:
            data = r.json(); prof = data[0] if isinstance(data, list) and data else None
            if prof:
                return {"sector": prof.get("sector") or prof.get("industry"),
                        "pe": prof.get("pe"),
                        "marketCap": prof.get("mktCap"),
                        "description": prof.get("description"),
                        "earningsDate": prof.get("mktCap")}
    except Exception:
        pass
    return {}

def fmp_news_count(sym, key, days=30):
    if not key:
        return 0
    try:
        r = requests.get(f"https://financialmodelingprep.com/api/v3/stock_news?symbol={sym}&limit=50&apikey={key}", timeout=8)
        if r.status_code == 200:
            news = r.json()
            cnt = 0
            for n in news:
                dt = n.get("publishedDate") or n.get("date")
                if dt:
                    try:
                        d = datetime.fromisoformat(dt.replace('Z',''))
                        if (datetime.now() - d).days <= days:
                            cnt += 1
                    except Exception:
                        cnt += 1
            return cnt
    except Exception:
        pass
    return 0

# ---------------- sentiment via Google News RSS ----------------
def google_news_sentiment(sym, days=15):
    try:
        q = f"{sym} stock India"
        url = "https://news.google.com/rss/search?q=" + requests.utils.requote_uri(q) + "&hl=en-IN&gl=IN&ceid=IN:en"
        r = requests.get(url, timeout=6, headers={'User-Agent':'Mozilla/5.0'})
        if r.status_code != 200 or not r.text:
            return 0.0, 0
        root = ET.fromstring(r.text)
        items = root.findall('.//item')
        total_score = 0; count = 0
        for it in items[:12]:
            title = it.find('title').text if it.find('title') is not None else ""
            desc = it.find('description').text if it.find('description') is not None else ""
            txt = (title + " " + desc).lower()
            words = re.findall(r"\w+", txt)
            pos = sum(1 for w in words if w in POS_WORDS)
            neg = sum(1 for w in words if w in NEG_WORDS)
            score = pos - neg
            total_score += score; count += 1
        if count == 0:
            return 0.0, 0
        avg = total_score / count
        norm = max(-1.0, min(1.0, avg / 3.0))
        return float(norm), count
    except Exception:
        return 0.0, 0

# ---------------- technical helpers ----------------
def rsi(series, period=14):
    delta = series.diff().dropna()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def moving_average(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def atr(high, low, close, window=14):
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean().iloc[-1]

def rolling_sharpe(returns, window=90):
    if len(returns) < window:
        window = max(10, len(returns))
    mean = returns.rolling(window=window).mean()
    std = returns.rolling(window=window).std() + 1e-9
    sharpe = (mean / std) * math.sqrt(252)
    return float(sharpe.dropna().iloc[-1]) if not sharpe.dropna().empty else 0.0

def hist_forward_stats(close, days, windows=[252, 252*2, 252*3]):
    stats = []
    for w in windows:
        if len(close) < w + days:
            continue
        sub = close[-(w + days):]
        fwd = sub.shift(-days) / sub - 1
        fwd = fwd.dropna()
        if not fwd.empty:
            stats.append({"mean": float(fwd.mean()), "std": float(fwd.std()), "pos_prop": float((fwd > 0).mean())})
    if not stats:
        return {"mean": 0.0, "std": 0.0, "pos_prop": 0.0, "count": 0}
    mean = float(np.mean([s["mean"] for s in stats]))
    std = float(np.mean([s["std"] for s in stats]))
    pos = float(np.mean([s["pos_prop"] for s in stats]))
    return {"mean": mean, "std": std, "pos_prop": pos, "count": len(stats)}

def bootstrap_forward_mean(close, days, n_boot=300):
    if len(close) < max(2*days, 60):
        return {"mean": 0.0, "p25": 0.0, "p75": 0.0, "pos_rate": 0.0}
    fwd = close.shift(-days) / close - 1
    fwd = fwd.dropna().values
    if len(fwd) == 0:
        return {"mean": 0.0, "p25": 0.0, "p75": 0.0, "pos_rate": 0.0}
    samples = np.random.choice(fwd, size=(n_boot, min(len(fwd), n_boot)), replace=True)
    sample_means = samples.mean(axis=1)
    mean = float(np.mean(sample_means))
    p25 = float(np.percentile(sample_means, 25))
    p75 = float(np.percentile(sample_means, 75))
    pos = float((sample_means > 0).mean())
    return {"mean": mean, "p25": p25, "p75": p75, "pos_rate": pos}

def kelly_estimate(expected, vol):
    if vol <= 0:
        return 0.0
    f = expected / (vol * vol + 1e-9)
    return float(np.clip(f, -0.5, 0.5))

def momentum_percentiles(universe_closes):
    mom_scores = {}
    for tk, c in universe_closes.items():
        try:
            mom_scores[tk] = (c.iloc[-1] / c.shift(60).iloc[-1] - 1) if len(c) > 60 else 0.0
        except:
            mom_scores[tk] = 0.0
    ser = pd.Series(mom_scores)
    pct = ser.rank(pct=True).to_dict()
    return pct

# ---------------- expected return estimator ----------------
def estimate_expected_return(close, days):
    returns = close.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    recent_n = max(5, int(max(5, len(returns) * 0.15)))
    recent = returns.tail(recent_n).mean()
    hist = hist_forward_stats(close, days)["mean"]
    if days <= 30:
        expected = 0.85 * recent + 0.15 * hist
    elif days <= 90:
        expected = 0.7 * recent + 0.3 * hist
    elif days <= 180:
        expected = 0.55 * recent + 0.45 * hist
    else:
        expected = 0.45 * recent + 0.55 * hist
    mom_factor = recent * (1.0 + min(0.8, np.tanh(np.nan_to_num(recent) * 10)))
    expected = 0.7 * expected + 0.3 * mom_factor
    return float(np.clip(expected, -1.0, 1.0))

# ---------------- core per-ticker compute ----------------
def compute_metrics(tk, days, universe_closes=None, fmp_key=None):
    df = load_price(tk)
    if df is None or "Close" not in df.columns or len(df["Close"].dropna()) < 90:
        return None
    close = df["Close"].dropna()
    expected = estimate_expected_return(close, days)
    if expected < -0.5:
        return None
    returns = close.pct_change().dropna()
    vol = float(returns.std()) if not returns.empty else 0.0
    atr_val = atr(df["High"], df["Low"], df["Close"]) if "High" in df.columns and "Low" in df.columns else 0.0
    sharpe = rolling_sharpe(returns, window=min(252, len(returns)))
    bootstrap = bootstrap_forward_mean(close, days, n_boot=300)
    kelly = kelly_estimate(expected, vol)
    mom5 = (close.iloc[-1] / close.shift(5).iloc[-1] - 1) if len(close) > 5 else 0.0
    mom20 = (close.iloc[-1] / close.shift(20).iloc[-1] - 1) if len(close) > 20 else 0.0
    mom = 0.6 * (mom5 if not math.isnan(mom5) else 0) + 0.4 * (mom20 if not math.isnan(mom20) else 0)
    mom_pct = 0.0
    if universe_closes is not None and tk in universe_closes:
        try:
            percentiles = momentum_percentiles(universe_closes)
            mom_pct = percentiles.get(tk, 0.0)
        except:
            mom_pct = 0.0
    g_sent, g_cnt = google_news_sentiment(tk.replace(".NS",""), days=15)
    try:
        info = yf.Ticker(tk).info
    except:
        info = {}
    mc = info.get("marketCap", 0) or 0
    name = info.get("shortName") or info.get("longName") or tk
    sector = info.get("sector") or info.get("industry") or "Other"
    avg_vol = float(df["Volume"].dropna().tail(60).mean()) if "Volume" in df.columns and len(df["Volume"].dropna())>0 else 0.0

    # scoring (ensemble) - bias for short intervals to expected & bootstrap
    if days <= 30:
        score = (expected * 0.7) + (mom * 0.12) + (bootstrap["mean"] * 0.06) + (mom_pct * 0.05) + (g_sent * 0.02)
    else:
        score = (expected * 0.45) + (sharpe * 0.15) + (bootstrap["mean"] * 0.15) + (0.05 * math.log10(max(mc,1)+1)) + (g_sent * 0.05)

    # confidence boost by market cap, liquidity, bootstrap positive rate and sharpe
    conf = 50
    if mc > 5e10:
        conf += 12
    elif mc > 1e10:
        conf += 8
    if avg_vol > 200000:
        conf += 10
    conf += int(min(12, bootstrap.get("pos_rate", 0) * 12))
    if sharpe > 1.0:
        conf += 6
    conf = int(max(30, min(99, conf)))

    # beginner-friendly why text (includes bootstrap summary & Kelly guidance)
    try:
        pos_pct = int(round(bootstrap.get("pos_rate", 0) * 100))
    except:
        pos_pct = 0
    exp_pct = expected * 100.0
    if exp_pct >= 3.0:
        why = f"{name} ({tk}) — Strong recent momentum and historical forward checks suggest notable upside in the chosen short horizon. Bootstrap shows {pos_pct}% positive chance."
    elif exp_pct >= 1.0:
        why = f"{name} ({tk}) — Positive momentum and stable fundamentals; reasonable upside expected in the selected period."
    elif exp_pct > 0:
        why = f"{name} ({tk}) — Modest upside expected; safer pick for this period because the company is stable."
    else:
        why = f"{name} ({tk}) — Limited near-term upside; included for balance and risk control."
    return {"ticker": tk, "name": name, "expected": float(expected), "score": float(score), "sector": sector,
            "confidence": conf, "avg_vol": avg_vol, "marketCap": mc, "why": why, "bootstrap": bootstrap, "kelly": kelly, "sharpe": sharpe}

# ---------------- selection logic ----------------
def select_with_thresholds(results, days, ideal_thresh_map=IDEAL_THRESH, fallback_map=FALLBACK_THRESH):
    ideal = ideal_thresh_map.get(days, 0.0)
    fallback = fallback_map.get(days, 0.0)
    primary = [r for r in results if r.get("expected_calibrated", r["expected"]) >= ideal and r["expected"] > 0]
    primary_sorted = sorted(primary, key=lambda x: (x["score"], x["bootstrap"]["mean"]), reverse=True)
    if len(primary_sorted) >= 50:
        for r in primary_sorted[:50]:
            r["threshold_level"] = "ideal"
        return primary_sorted[:50], "ideal"
    fallback_pool = [r for r in results if r not in primary_sorted and r.get("expected_calibrated", r["expected"]) >= fallback and r["expected"] > 0]
    combined = primary_sorted + sorted(fallback_pool, key=lambda x: (x["score"], x["bootstrap"]["mean"]), reverse=True)
    if len(combined) >= 50:
        for r in combined[:len(primary_sorted)]:
            r["threshold_level"] = "ideal"
        for r in combined[len(primary_sorted):50]:
            r["threshold_level"] = "fallback"
        return combined[:50], "fallback"
    positive_pool = [r for r in results if r["expected"] > 0 and r not in combined]
    combined2 = combined + sorted(positive_pool, key=lambda x: (x["score"], x["bootstrap"]["mean"]), reverse=True)
    if len(combined2) < 50:
        remaining = [r for r in sorted(results, key=lambda x:(x["expected"], x["score"]), reverse=True) if r not in combined2]
        for r in remaining:
            if r["expected"] < -0.05:
                continue
            combined2.append(r)
            if len(combined2) >= 50:
                break
    for r in combined2[:50]:
        lvl = "ideal" if r.get("expected_calibrated", r["expected"]) >= ideal else ("fallback" if r.get("expected_calibrated", r["expected"]) >= fallback else "relaxed")
        r["threshold_level"] = lvl
    used_level = "relaxed" if any(r["threshold_level"]=="relaxed" for r in combined2[:50]) else ("fallback" if any(r["threshold_level"]=="fallback" for r in combined2[:50]) else "ideal")
    return combined2[:50], used_level

# ---------------- backtesting / walk-forward ----------------
def backtest_universe_walkforward(candidates, days, checkpoints=6, months_between=1, lookback_years=3):
    now = datetime.now()
    checkpoints_dates = [now - pd.DateOffset(months=months_between*i) for i in range(1, checkpoints+1)]
    aggregated = {"checkpoints": [], "per_ticker": {}}
    for dt in checkpoints_dates:
        cp_date = pd.to_datetime(dt.date())
        cp_results = []
        for tk in candidates:
            df_full = load_price(tk)
            if df_full is None or "Close" not in df_full.columns:
                continue
            df_trunc = df_full[df_full.index <= cp_date]
            if df_trunc is None or len(df_trunc["Close"].dropna()) < 60:
                continue
            close_trunc = df_trunc["Close"].dropna()
            pred = estimate_expected_return(close_trunc, days)
            try:
                price_now = float(close_trunc.iloc[-1])
            except:
                continue
            target_date = cp_date + pd.Timedelta(days=days)
            df_after = df_full[df_full.index >= target_date]
            if df_after is None or df_after.empty:
                continue
            price_future = float(df_after["Close"].iloc[0])
            realized = (price_future / price_now) - 1.0
            cp_results.append({"ticker":tk, "pred": pred, "realized": realized})
            aggregated["per_ticker"].setdefault(tk, {"preds": [], "realized": []})
            aggregated["per_ticker"][tk]["preds"].append(pred)
            aggregated["per_ticker"][tk]["realized"].append(realized)
        if cp_results:
            preds = np.array([p["pred"] for p in cp_results])
            reals = np.array([p["realized"] for p in cp_results])
            hit_rate = float((reals > 0).mean())
            mean_pred = float(preds.mean())
            mean_real = float(reals.mean())
            mae = float(np.mean(np.abs(preds - reals)))
            coverage = len(cp_results)
        else:
            hit_rate = mean_pred = mean_real = mae = 0.0; coverage = 0
        aggregated["checkpoints"].append({"checkpoint_date": str(cp_date.date()), "hit_rate": hit_rate, "mean_pred": mean_pred, "mean_real": mean_real, "mae": mae, "coverage": coverage})
    cps = aggregated["checkpoints"]
    if cps:
        agg_hit = float(np.mean([c["hit_rate"] for c in cps]))
        agg_mean_pred = float(np.mean([c["mean_pred"] for c in cps]))
        agg_mean_real = float(np.mean([c["mean_real"] for c in cps]))
        agg_mae = float(np.mean([c["mae"] for c in cps]))
        total_coverage = int(np.sum([c["coverage"] for c in cps]))
    else:
        agg_hit = agg_mean_pred = agg_mean_real = agg_mae = 0.0; total_coverage = 0
    report = {"agg_hit_rate": agg_hit, "agg_mean_pred": agg_mean_pred, "agg_mean_real": agg_mean_real, "agg_mae": agg_mae, "total_coverage": total_coverage, "details": aggregated}
    fname = BACKTEST_DIR / f"backtest_{days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    except:
        pass
    return report

# ---------------- main generation flow ----------------
if generate:
    days = interval_map.get(interval, 30)
    st.info(f"Generating Super50 for holding period = {days} trading days...")
    logs = []
    def log(msg):
        logs.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

    try:
        symbols = fetch_nse_symbols()
        if not symbols:
            st.error("Could not fetch NSE symbols. Check network or NSE feed.")
            if show_debug: st.text("NSE symbol fetch failed or returned empty.")
            raise SystemExit("No symbols")

        tickers = [s + ".NS" for s in symbols]
        meta_map_full = fetch_meta_batch(tickers)
        ranked = sorted([(tk, meta_map_full.get(tk,{}).get("marketCap",0)) for tk in tickers], key=lambda x: x[1], reverse=True)
        ranked = [tk for tk,_ in ranked]
        if days <= 15:
            universe = ranked[:400]
        elif days <= 30:
            universe = ranked[:450]
        else:
            universe = ranked[:TOP_UNIVERSE]

        log(f"Universe selected: {len(universe)} tickers.")

        missing = [t for t in universe if not price_fresh(t)]
        if missing:
            with st.spinner(f"Downloading price history for {len(missing)} tickers (this may take a few minutes)..."):
                download_prices(missing, years=5)
            log(f"Downloaded missing price files for {len(missing)} tickers.")

        # preload closes for momentum percentiles
        universe_closes = {}
        for tk in universe:
            df = load_price(tk)
            if df is not None and "Close" in df.columns:
                universe_closes[tk] = df["Close"].dropna()

        # build candidate list
        candidates = []
        for tk in universe:
            df = load_price(tk)
            if df is None or "Close" not in df.columns:
                continue
            if len(df["Close"].dropna()) >= MIN_HISTORY_DEFAULT:
                candidates.append(tk)
        log(f"Candidates with >= {MIN_HISTORY_DEFAULT} days: {len(candidates)}")
        if len(candidates) == 0:
            # relax to shorter history
            for tk in universe:
                df = load_price(tk)
                if df is None or "Close" not in df.columns:
                    continue
                if len(df["Close"].dropna()) >= 30:
                    candidates.append(tk)
        if len(candidates) == 0:
            candidates = universe[:200]
        log(f"Final candidate pool: {len(candidates)}")

        # optionally run walk-forward backtest (may be slow)
        backtest_stats = backtest_universe_walkforward(candidates, days, checkpoints=6, months_between=1, lookback_years=3) if include_report else {"agg_hit_rate":None}
        if include_report:
            log("Walk-forward backtest completed.")

        # compute metrics
        results = []
        with ThreadPoolExecutor(max_workers=18) as ex:
            futures = {ex.submit(compute_metrics, tk, days, universe_closes, fmp_key=(fmp_key or None)): tk for tk in candidates}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if res is not None and res.get("expected", 0) > -0.5:
                        results.append(res)
                except Exception:
                    continue

        if not results:
            st.error("No results computed. Check prewarm cache, network, or permissions.")
            if show_debug:
                st.text("Debug logs:\n" + "\n".join(logs[-50:]))
            raise SystemExit("No results")

        # calibration using backtest aggregated stats
        global_mult = 1.0
        if backtest_stats and backtest_stats.get("agg_hit_rate") is not None:
            gr = backtest_stats.get("agg_hit_rate", 0.0)
            if gr <= 0.4: global_mult = 0.8
            elif gr <= 0.48: global_mult = 0.9
            elif gr >= 0.6: global_mult = 1.08
            global_mult = float(np.clip(global_mult, 0.7, 1.15))

        for r in results:
            br = r.get("bootstrap",{}).get("pos_rate",0.0)
            per_mult = 1.0 + (br - 0.5) * 0.35
            per_mult = float(np.clip(per_mult, 0.75, 1.25))
            r["expected_calibrated"] = r["expected"] * global_mult * per_mult
            if r["expected_calibrated"] < -0.2:
                r["expected_calibrated"] = -0.2

        # liquidity gating (relaxed for shortest)
        if days <= 15:
            filtered = [r for r in results if (r["avg_vol"] > 10 or r["marketCap"] > 1e8)]
        elif days <= 30:
            filtered = [r for r in results if (r["avg_vol"] > 40 or r["marketCap"] > 2e8)]
        else:
            filtered = [r for r in results if (r["avg_vol"] > 100 or r["marketCap"] > 4e8)]
        if len(filtered) < 50:
            filtered = results

        for r in filtered:
            r["_rank_key"] = (r.get("expected_calibrated", r["expected"]), r["score"], r["confidence"])

        filtered = sorted(filtered, key=lambda x: x["_rank_key"], reverse=True)
        final50, level_used = select_with_thresholds(filtered, days)

        # sector soft diversification (soft cap 10 per sector)
        sector_counts = {}
        diversified = []
        for r in final50:
            sec = r.get("sector","Other") or "Other"
            cnt = sector_counts.get(sec, 0)
            if cnt < 10:
                diversified.append(r)
                sector_counts[sec] = cnt + 1
            else:
                diversified.append(r)  # keep original if replacement not found (soft)
        if len(diversified) == 50:
            final50 = diversified

        st.success(f"Super50 V2.0 — Final Picks ({len(final50)})")
        if level_used != "ideal":
            st.warning(f"Could not fill all 50 with ideal thresholds; used '{level_used}' fallback. See flagged picks.")

        st.markdown('<div class="grid">', unsafe_allow_html=True)
        for i, m in enumerate(final50, start=1):
            exp_pct = m.get("expected_calibrated", m.get("expected", 0)) * 100.0
            conf = m.get("confidence", 50)
            thr = m.get("threshold_level", "unknown")
            flag_html = ""
            if thr == "fallback":
                flag_html = "<div class='warn'>Below ideal threshold - fallback used</div>"
            elif thr == "relaxed":
                flag_html = "<div class='warn'>Relaxed selection used (insufficient candidates met thresholds)</div>"
            html = ("<div class='card'>"
                    f"<div><span class='badge'>{m.get('sector','Other')}</span><span style='float:right;font-weight:700'>{i}. {m['ticker']}</span></div>"
                    "<div style='margin-top:8px;'>"
                    f"<div style='font-size:16px;font-weight:700;color:#0b3d91'>{m.get('name','')}</div>"
                    f"<div style='font-size:13px;color:#475569;margin-top:6px;'>Expected: <strong>{exp_pct:.2f}%</strong> | Confidence: <strong>{conf}/100</strong></div>"
                    f"<div style='margin-top:8px;'>{m.get('why')}</div>"
                    f"{flag_html}"
                    "</div></div>")
            st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # CSV download
        df_out = pd.DataFrame([{
            "rank": i+1,
            "ticker": m.get("ticker"),
            "name": m.get("name",""),
            "expected_pct": (m.get("expected_calibrated", m.get("expected",0))*100),
            "marketCap": m.get("marketCap",0),
            "sector": m.get("sector",""),
            "confidence": m.get("confidence",0),
            "threshold_level": m.get("threshold_level","")
        } for i,m in enumerate(final50)])
        st.download_button("Download Super50 CSV", df_out.to_csv(index=False), file_name="Super50_V2.0_top50.csv", mime="text/csv")

        # show predictability report if requested
        if include_report and backtest_stats:
            st.markdown("<hr/>")
            st.markdown("<h3>Predictability Report (walk-forward summary)</h3>", unsafe_allow_html=True)
            st.write(f"Aggregated hit-rate (positive realized returns): **{backtest_stats.get('agg_hit_rate'):.2f}**")
            st.write(f"Aggregated predicted mean (avg across checkpoints): **{backtest_stats.get('agg_mean_pred'):.4f}**")
            st.write(f"Aggregated realized mean (avg across checkpoints): **{backtest_stats.get('agg_mean_real'):.4f}**")
            st.write(f"Aggregated MAE (avg across checkpoints): **{backtest_stats.get('agg_mae'):.4f}**")
            st.write(f"Total coverage (sum of tickers with realized forward prices across checkpoints): **{backtest_stats.get('total_coverage')}**")
            rows = backtest_stats.get("details", {}).get("checkpoints", [])
            if rows:
                dfc = pd.DataFrame(rows)
                st.dataframe(dfc)
            st.info(f"Backtest saved at: {BACKTEST_DIR}")

        if show_debug:
            st.markdown("### Debug logs (last messages)")
            st.text("\n".join(logs[-200:]))

    except Exception as e:
        st.error("Generation failed: " + str(e))
        if show_debug:
            st.text(traceback.format_exc())
