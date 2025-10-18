import streamlit as st
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import timedelta, date

# ----------------------
# Polygon API Key
# ----------------------
POLYGON_API_KEY = "pm85pW39kP0RVJaPt8fbhJJjujMOP2vE"
client = RESTClient(POLYGON_API_KEY)

# ----------------------
# Page Setup
# ----------------------
st.set_page_config(page_title="Earnings Edge Analyzer", layout="wide")
st.title("📊 Earnings Edge: IV/HV & Sector Analysis (Polygon Data)")
st.markdown("""
Analyze upcoming earnings using **IV/HV**, **historical earnings performance**, and **sector peer comparison**.
""")

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_area(
    "Enter tickers (comma-separated) or leave blank to use large-cap Nasdaq CSV",
    ""
)
hv_lookback = st.sidebar.slider("HV Lookback Period (days)", 10, 90, 30)
peer_count = st.sidebar.slider("Max Sector Peers to Compare", 1, 10, 5)

# ----------------------
# Load Large-Cap Nasdaq CSV
# ----------------------
@st.cache_data
def load_largecap_csv():
    df = pd.read_csv("nasdaq_largecap.csv")  # Columns: Symbol, Sector, MarketCap
    df.rename(columns={"Symbol":"Ticker"}, inplace=True)
    df["MarketCap"] = pd.to_numeric(df["MarketCap"], errors="coerce")
    return df

largecap_df = load_largecap_csv()

# ----------------------
# Determine tickers
# ----------------------
if tickers_input.strip():
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
else:
    tickers = largecap_df["Ticker"].tolist()

st.sidebar.markdown(f"✅ {len(tickers)} tickers loaded for analysis")

# ----------------------
# Polygon Helper Functions
# ----------------------
@st.cache_data
def fetch_upcoming_earnings(ticker):
    try:
        today = date.today()
        one_month = today + timedelta(days=30)
        data = client.reference_earnings(ticker=ticker, from_=today.isoformat(), to=one_month.isoformat())
        if data and "results" in data and len(data["results"])>0:
            return pd.to_datetime(data["results"][0]["reporting_date"])
        return None
    except:
        return None

@st.cache_data
def fetch_historical_earnings(ticker, last_n=10):
    try:
        data = client.reference_earnings(ticker=ticker, sort="desc", limit=last_n)
        if data and "results" in data:
            dates = [pd.to_datetime(item["reporting_date"]) for item in data["results"]]
            return dates
        return []
    except:
        return []

@st.cache_data
def fetch_historical_prices(ticker, start, end):
    try:
        bars = client.stocks_equities_aggregates(ticker, 1, "day", start.isoformat(), end.isoformat())
        df = pd.DataFrame(bars['results'])
        if df.empty:
            return None
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('t', inplace=True)
        return df
    except:
        return None

def post_earnings_move(ticker, earnings_dates, days_post=3):
    moves = []
    for date_item in earnings_dates:
        pre_prices = fetch_historical_prices(ticker, date_item - timedelta(days=2), date_item)
        post_prices = fetch_historical_prices(ticker, date_item, date_item + timedelta(days_post))
        if pre_prices is None or post_prices is None or pre_prices.empty or post_prices.empty:
            continue
        move_pct = (post_prices['c'][-1] - pre_prices['c'][-1]) / pre_prices['c'][-1] * 100
        moves.append(move_pct)
    return round(np.mean(moves),2) if moves else None

@st.cache_data
def fetch_sector_peers(ticker, max_peers=10):
    sector = largecap_df.loc[largecap_df["Ticker"]==ticker,"Sector"].values
    if len(sector)==0:
        return []
    sector = sector[0]
    peers = largecap_df[(largecap_df["Sector"]==sector) & (largecap_df["Ticker"]!=ticker)]
    peers = peers.sort_values("MarketCap", ascending=False).head(max_peers)
    return peers["Ticker"].tolist()

@st.cache_data
def sector_average_post_earnings(sector, last_n=10):
    if sector is None:
        return None
    peers = largecap_df[largecap_df["Sector"]==sector]["Ticker"].tolist()
    moves = []
    for t in peers:
        dates = fetch_historical_earnings(t, last_n)
        move = post_earnings_move(t, dates)
        if move is not None:
            moves.append(move)
    return round(np.mean(moves),2) if moves else None

@st.cache_data
def peer_correlation(ticker, max_peers=10, last_n=10):
    peers = fetch_sector_peers(ticker, max_peers)
    if not peers:
        return None
    ticker_dates = fetch_historical_earnings(ticker, last_n)
    if not ticker_dates:
        return None

    ticker_moves = []
    peers_moves_avg = []

    for date_item in ticker_dates:
        ticker_prices_pre = fetch_historical_prices(ticker, date_item - timedelta(days=2), date_item)
        ticker_prices_post = fetch_historical_prices(ticker, date_item, date_item + timedelta(days=3))
        if ticker_prices_pre is None or ticker_prices_post is None:
            continue
        ticker_move = (ticker_prices_post['c'][-1] - ticker_prices_pre['c'][-1])/ticker_prices_pre['c'][-1]*100
        ticker_moves.append(ticker_move)

        peer_moves = []
        for p in peers:
            p_prices_pre = fetch_historical_prices(p, date_item - timedelta(days=2), date_item)
            p_prices_post = fetch_historical_prices(p, date_item, date_item + timedelta(days=3))
            if p_prices_pre is None or p_prices_post is None:
                continue
            move_p = (p_prices_post['c'][-1] - p_prices_pre['c'][-1])/p_prices_pre['c'][-1]*100
            peer_moves.append(move_p)
        if peer_moves:
            peers_moves_avg.append(np.mean(peer_moves))

    if ticker_moves and peers_moves_avg:
        return round(np.corrcoef(ticker_moves, peers_moves_avg)[0,1],2)
    return None

# ----------------------
# Main Analysis
# ----------------------
results = []

if st.button("Run Analysis"):
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for idx, ticker in enumerate(tickers):
        upcoming_earnings = fetch_upcoming_earnings(ticker)
        hist_dates = fetch_historical_earnings(ticker, last_n=10)
        hist_move = post_earnings_move(ticker, hist_dates)
        sector_name = largecap_df.loc[largecap_df["Ticker"]==ticker,"Sector"].values[0] if len(largecap_df.loc[largecap_df["Ticker"]==ticker])>0 else None
        sector_avg = sector_average_post_earnings(sector_name, last_n=10)
        peer_corr = peer_correlation(ticker, max_peers=peer_count, last_n=10)
        peers = fetch_sector_peers(ticker, peer_count)

        results.append({
            "Ticker": ticker,
            "Upcoming Earnings": upcoming_earnings,
            "Historical Post-Earnings Move (%)": hist_move,
            "Sector": sector_name,
            "Sector Avg Post-Earnings (%)": sector_avg,
            "Peer Correlation": peer_corr,
            "Sector Peers": ", ".join(peers)
        })

        progress_bar.progress((idx+1)/len(tickers))
        progress_text.text(f"Processing {ticker} ({idx+1}/{len(tickers)})...")

    df = pd.DataFrame(results)
    st.success("✅ Analysis Complete!")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "earnings_analysis.csv", "text/csv")
