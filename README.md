# Pre-Earnings Analysis Dashboard

Analyze implied vs historical volatility for upcoming earnings plays.

## What It Does
A Streamlit web app that analyzes stocks before earnings by comparing **Implied Volatility (IV)** vs **Historical Volatility (HV)** — key metrics for options traders.

## Key Features

1. **IV/HV Ratio Analysis**
   - Fetches current ATM implied volatility from options chains
   - Calculates historical volatility based on configurable lookback (10-90 days)
   - Shows IV/HV ratio — if IV > HV, options are expensive (premium rich)

2. **IV Percentile**
   - Ranks current IV against past volatility (30-365 day lookback)
   - Helps identify if current IV is high/low relative to history

3. **IV Curve Visualization**
   - Plots implied volatility by strike price for upcoming expirations
   - Shows calls vs puts separately

4. **Large-Cap Filter**
   - Uses `nasdaq_largecap.csv` (pre-scraped) to focus on big companies ($10B+)

5. **Export**
   - Download results as CSV

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit dashboard |
| `scrape_market.py` | Scraper to build `nasdaq_largecap.csv` |
| `nasdaq_largecap.csv` | Pre-scraped dataset of large-cap stocks |
| `requirements.txt` | Dependencies |

## Tech Stack
- **Streamlit** — UI
- **yfinance** — Market data + options chains
- **pandas/numpy** — Data processing
- **plotly** — Interactive charts

## How to Run
```bash
pip install -r requirements.txt
python scrape_market.py  # one-time setup (requires nasdaqlisted.txt)
streamlit run app.py
```
