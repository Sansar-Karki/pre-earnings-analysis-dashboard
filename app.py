import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Earnings Edge Analyzer", layout="wide")
st.title("📊 Earnings Edge: IV/HV & Sector Analysis")

st.markdown("""
Analyze upcoming earnings using **IV/HV**, **IV curves**, **historical earnings performance**, and **sector peer comparison**.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_area(
    "Enter tickers (comma-separated)",
    "AAPL,MSFT,TSLA,NVDA,AMZN"
)
hv_lookback = st.sidebar.slider("HV Lookback Period (days)", 10, 90, 30)
peer_count = st.sidebar.slider("Max Sector Peers to Compare", 1, 10, 5)
show_iv_curve = st.sidebar.checkbox("Show IV Curve for Tickers", value=True)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# --- Load S&P 500 CSV ---
@st.cache_data
def load_sp500():
    df = pd.read_csv("sp500.csv")  # Columns: Ticker, Sector, MarketCap
    df["MarketCap"] = pd.to_numeric(df["MarketCap"], errors="coerce")
    return df

sp500_df = load_sp500()

# --- Helper Functions ---
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

@st.cache_data
def fetch_iv_avg(ticker):
    try:
        chain = fetch_iv_chain(ticker, expirations=1)
        df = list(chain.values())[0]
        return round(df["impliedVolatility"].mean()*100,2)
    except:
        return None

@st.cache_data
def fetch_earnings(ticker):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is not None and not cal.empty:
            return cal.iloc[0]["Earnings Date"]
        return None
    except:
        return None

@st.cache_data
def fetch_sector_peers(ticker, max_peers=5):
    sector = sp500_df.loc[sp500_df["Ticker"]==ticker,"Sector"].values
    if len(sector)==0:
        return []
    sector = sector[0]
    peers = sp500_df[(sp500_df["Sector"]==sector) & (sp500_df["Ticker"]!=ticker) & (sp500_df["MarketCap"]>10e9)]
    peers = peers.sort_values("MarketCap", ascending=False).head(max_peers)
    return peers["Ticker"].tolist()

@st.cache_data
def historical_earnings_move(ticker, last_n=10, days_post=1):
    try:
        tk = yf.Ticker(ticker)
        if hasattr(tk, 'earnings_dates'):
            earnings = tk.earnings_dates.tail(last_n)
            moves = []
            for date in earnings.index:
                date = pd.to_datetime(date)
                pre = tk.history(start=date - pd.Timedelta(days=1), end=date)
                post = tk.history(start=date, end=date + pd.Timedelta(days=days_post))
                if not pre.empty and not post.empty:
                    move_pct = (post["Close"][-1] - pre["Close"][-1]) / pre["Close"][-1] * 100
                    moves.append(move_pct)
            if moves:
                return round(np.mean(moves),2)
        return None
    except:
        return None

# --- Main Analysis ---
results = []

if st.button("Run Analysis"):
    with st.spinner("Fetching data..."):
        for ticker in tickers:
            iv = fetch_iv_avg(ticker)
            hv = fetch_historical_volatility(ticker, hv_lookback)
            earnings = fetch_earnings(ticker)
            peers = fetch_sector_peers(ticker, peer_count)
            hist_move = historical_earnings_move(ticker, last_n=10)
            ratio = round(iv/hv,2) if hv and iv else None
            results.append({
                "Ticker": ticker,
                "Earnings": earnings,
                "IV (%)": iv,
                "HV (%)": hv,
                "IV/HV Ratio": ratio,
                "Historical Post-Earnings Move (%)": hist_move,
                "Sector": sp500_df.loc[sp500_df["Ticker"]==ticker,"Sector"].values[0] if len(sp500_df.loc[sp500_df["Ticker"]==ticker])>0 else None,
                "Sector Peers": ", ".join(peers)
            })

    df = pd.DataFrame(results)
    st.success("✅ Analysis Complete!")
    st.dataframe(df)

    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "earnings_analysis.csv", "text/csv")

    # IV Curve
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
