import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
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
st.title("📊 Earnings Edge: IV/HV & Sector Analysis")
st.markdown("""
Analyze upcoming earnings using **IV/HV**, **IV curves**, **historical earnings performance**, and **sector peer comparison**.
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
show_iv_curve = st.sidebar.checkbox("Show IV Curve for Tickers", value=True)

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
# Determine tickers to analyze
# ----------------------
if tickers_input.strip():
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
else:
    tickers = largecap_df["Ticker"].tolist()

st.sidebar.markdown(f"✅ {len(tickers)} tickers loaded for analysis")

# ----------------------
# Helper Functions
# ----------------------
@st.cache_data
def fetch_historical_volatility(ticker, days):
    try:
        data = yf.download(ticker, period=f"{days*2}d", progress=False)
        data["returns"] = np.log(data["Close"]/data["Close"].shift(1))
        hv = data["returns"].std() * np.sqrt(252)
        return round(hv*100,2)
    except:
        return None

@st.cache_data
def fetch_iv_chain(ticker, expirations=2):
    try:
        tk = yf.Ticker(ticker)
        exp_dates = tk.options[:expirations]
        chains = {}
        for exp in exp_dates:
            opt = tk.option_chain(exp)
            opt.calls["Type"] = "Call"
            opt.puts["Type"] = "Put"
            opt.calls["Expiration"] = exp
            opt.puts["Expiration"] = exp
            chains[exp] = pd.concat([opt.calls, opt.puts], ignore_index=True)
        return chains
    except:
        return {}

@st.cache_data
def fetch_iv_avg(ticker):
    chain = fetch_iv_chain(ticker, expirations=1)
    if not chain:
        return None
    df = list(chain.values())[0]
    if df.empty:
        return None
    return round(df["impliedVolatility"].mean()*100,2)

# ----------------------
# Polygon Earnings Functions
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
    except Exception as e:
        print(f"Error fetching upcoming earnings for {ticker}: {e}")
        return None

@st.cache_data
def fetch_historical_earnings(ticker, last_n=10):
    try:
        data = client.reference_earnings(ticker=ticker, sort="desc", limit=last_n)
        if data and "results" in data:
            dates = [pd.to_datetime(item["reporting_date"]) for item in data["results"]]
            return dates
        return []
    except Exception as e:
        print(f"Error fetching historical earnings for {ticker}: {e}")
        return []

def post_earnings_move(ticker, earnings_dates, days_post=1):
    tk = yf.Ticker(ticker)
    moves = []
    for date_item in earnings_dates:
        pre = tk.history(start=date_item - timedelta(days=1), end=date_item)
        post = tk.history(start=date_item, end=date_item + timedelta(days_post))
        if not pre.empty and not post.empty:
            move_pct = (post["Close"][-1] - pre["Close"][-1])/pre["Close"][-1]*100
            moves.append(move_pct)
    if moves:
        return round(np.mean(moves),2)
    return None

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
    if moves:
        return round(np.mean(moves),2)
    return None

@st.cache_data
def peer_correlation(ticker, max_peers=10, last_n=10):
    peers = fetch_sector_peers(ticker, max_peers)
    if not peers:
        return None
    tk = yf.Ticker(ticker)
    earnings_dates = fetch_historical_earnings(ticker, last_n)
    if not earnings_dates:
        return None

    ticker_moves = []
    peers_moves_avg = []

    for date_item in earnings_dates:
        pre = tk.history(start=date_item - timedelta(days=1), end=date_item)
        post = tk.history(start=date_item, end=date_item + timedelta(days=1))
        if pre.empty or post.empty:
            continue
        ticker_move = (post["Close"][-1] - pre["Close"][-1])/pre["Close"][-1]*100
        ticker_moves.append(ticker_move)

        peers_data = yf.download(peers, start=date_item - timedelta(days=1), end=date_item + timedelta(days=1), progress=False)
        peer_moves = []
        for p in peers:
            if (p, "Close") in peers_data.columns:
                move_p = (peers_data[p]["Close"].iloc[-1] - peers_data[p]["Close"].iloc[0])/peers_data[p]["Close"].iloc[0]*100
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
        iv = fetch_iv_avg(ticker)
        hv = fetch_historical_volatility(ticker, hv_lookback)
        earnings_upcoming = fetch_upcoming_earnings(ticker)
        hist_dates = fetch_historical_earnings(ticker, last_n=10)
        hist_move = post_earnings_move(ticker, hist_dates)
        ratio = round(iv/hv,2) if iv and hv else None
        sector_name = largecap_df.loc[largecap_df["Ticker"]==ticker,"Sector"].values[0] if len(largecap_df.loc[largecap_df["Ticker"]==ticker])>0 else None
        sector_avg = sector_average_post_earnings(sector_name, last_n=10)
        peer_corr = peer_correlation(ticker, max_peers=peer_count, last_n=10)
        peers = fetch_sector_peers(ticker, peer_count)

        results.append({
            "Ticker": ticker,
            "Upcoming Earnings": earnings_upcoming,
            "IV (%)": iv,
            "HV (%)": hv,
            "IV/HV Ratio": ratio,
            "Historical Post-Earnings Move (%)": hist_move,
            "Sector": sector_name,
            "Sector Avg Post-Earnings (%)": sector_avg,
            "Peer Correlation": peer_corr,
            "Sector Peers": ", ".join(peers)
        })

        # Update progress bar
        progress_bar.progress((idx+1)/len(tickers))
        progress_text.text(f"Processing {ticker} ({idx+1}/{len(tickers)})...")

    df = pd.DataFrame(results)
    df.sort_values("IV/HV Ratio", inplace=True)  # sort by lowest IV/HV
    st.success("✅ Analysis Complete!")
    st.dataframe(df)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "earnings_analysis.csv", "text/csv")

    # IV Curves
    if show_iv_curve:
        for ticker in tickers:
            chains = fetch_iv_chain(ticker)
            for exp, df_chain in chains.items():
                st.subheader(f"{ticker} IV Curve - Expiration {exp}")
                fig = px.scatter(df_chain, x="strike", y="impliedVolatility", color="Type",
                                 labels={"impliedVolatility":"IV","strike":"Strike"},
                                 hover_data=["lastPrice","volume"])
                fig.update_traces(marker=dict(size=8), mode='lines+markers')
                st.plotly_chart(fig)
