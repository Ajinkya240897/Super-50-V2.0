
Super50 V2.0 - Rebuilt Package
Files:
- streamlit_super50_v2.py  (main Streamlit app)
- prewarm_cache.py         (prewarm cache script)
- requirements.txt         (python dependencies)
- README.txt               (this file)

Quick start:
1. pip install -r requirements.txt
2. python prewarm_cache.py   # optional but recommended
3. streamlit run streamlit_super50_v2.py

Notes:
- Enter FMP API key in sidebar (optional) to enrich fundamentals & news
- Use "Include Predictability Report" option to run walk-forward backtest (slower)
- Cached data stored in cache_super50_v2/
