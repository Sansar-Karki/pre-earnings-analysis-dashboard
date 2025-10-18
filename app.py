import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# -------------------------------
# App setup
# -------------------------------

st.set_page_config(page_title="Earnings Edge: IV/HV Analyzer", layout="wide")
st.title("📊 Earnings Edge: IV/HV Analyzer")

st.markdown("""
Analyze implied vs historical volatility for upcoming earnings plays.  
Now includes IV Percentile for richer context.
""")

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.header("Settings")
tickers_input = st.sidebar.text_area("Enter tickers (comma-separated)", "AAPL,MSFT,TSLA,NVDA,AMZN")
hv_lookback = st.sidebar.slider("HV Lookback Period (days)", 10, 90, 30)
iv_percentile_lookback = st.sidebar.slider("IV Percentile Lookback (days)", 30, 365, 180)
show_iv_curve = st.sidebar.checkbox("Show IV Curve for Tickers", value=True)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# -------------------------------
# Load largecap CSV (from scraper)
# -------------------------------

@st.cache_data
def load_largecap():
    try:
        df = pd.read_csv("nasdaq_largecap.csv")
        df["MarketCap"] = pd.to_numeric(df["MarketCap"], errors="coerce")
        return df
    except FileNotFoundError:
        st.warning("nasdaq_largecap.csv not found. Please run scrape_market.py first.")
        return pd.DataFrame(columns=["Symbol", "Sector", "MarketCap"])

largecap_df = load_largecap()

# -------------------------------
# Helper functions
# -------------------------------

@st.cache_data
def fetch_historical_volatility(ticker, days):
    """Calculate annualized historical volatility based on lookback period."""
    try:
        data = yf.download(ticker, period=f"{days}d", progress=False)
        data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
        hv = data["returns"].std() * np.sqrt(252)
        return round(hv * 100, 2)
    except Exception as e:
        print(f"Error fetching HV for {ticker}: {e}")
        return None


@st.cache_data
def fetch_iv_chain(ticker, expirations=1):
    """Fetch options chain data for given expirations."""
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
    except Exception as e:
        print(f"Error fetching chain for {ticker}: {e}")
        return None


@st.cache_data
def fetch_iv_avg(ticker):
    """Calculate average ATM implied volatility (±5% from spot)."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1d")
        if hist.empty:
            return None
        price = hist["Close"][-1]

        chain = fetch_iv_chain(ticker, expirations=1)
        if not chain:
            return None

        df = list(chain.values())[0]
        if df.empty:
            return None

        # Filter near ATM (±5%)
        df = df[(df["strike"] >= price * 0.95) & (df["strike"] <= price * 1.05)]
        df = df[df["impliedVolatility"].notnull()]
        if df.empty:
            return None

        iv = df["impliedVolatility"].mean() * 100
        return round(iv, 2)
    except Exception as e:
        print(f"Error fetching IV for {ticker}: {e}")
        return None


@st.cache_data
def fetch_iv_percentile(ticker, lookback_days):
    """
    Approximate IV Percentile:
    - Uses daily ATM IV snapshots (if available) or proxies with HV variation.
    - Returns percentile rank of current IV within lookback range.
    """
    try:
        # Try to approximate using HV variation as proxy (since yfinance lacks IV history)
        data = yf.download(ticker, period=f"{lookback_days}d", progress=False)
        data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data["hv_daily"] = data["returns"].rolling(20).std() * np.sqrt(252) * 100

        hv_series = data["hv_daily"].dropna()
        if hv_series.empty:
            return None

        current_iv = fetch_iv_avg(ticker)
        if current_iv is None:
            return None

        min_hv, max_hv = hv_series.min(), hv_series.max()
        iv_percentile = 100 * (current_iv - min_hv) / (max_hv - min_hv)
        iv_percentile = np.clip(iv_percentile, 0, 100)
        return round(iv_percentile, 1)
    except Exception as e:
        print(f"Error fetching IV percentile for {ticker}: {e}")
        return None

# -------------------------------
# Main analysis
# -------------------------------

results = []

if st.button("Run Analysis"):
    with st.spinner("Fetching volatility data..."):
        for ticker in tickers:
            iv = fetch_iv_avg(ticker)
            hv = fetch_historical_volatility(ticker, hv_lookback)
            ivp = fetch_iv_percentile(ticker, iv_percentile_lookback)
            ratio = round(iv / hv, 2) if hv and iv else None
            results.append({
                "Ticker": ticker,
                "IV (%)": iv,
                "HV (%)": hv,
                "IV/HV Ratio": ratio,
                "IV Percentile (%)": ivp,
            })

    df = pd.DataFrame(results).sort_values(by="IV/HV Ratio", ascending=True)
    st.success("✅ Analysis Complete!")

    # Display results
    st.dataframe(df)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "iv_hv_analysis.csv", "text/csv")

    # Optional IV Curve plots
    if show_iv_curve:
        for ticker in tickers:
            chains = fetch_iv_chain(ticker)
            if not chains:
                continue
            for exp, df_chain in chains.items():
                st.subheader(f"{ticker} IV Curve - Expiration {exp}")
                fig = px.scatter(
                    df_chain,
                    x="strike",
                    y="impliedVolatility",
                    color="Type",
                    labels={"impliedVolatility": "IV", "strike": "Strike"},
                    hover_data=["lastPrice", "volume"]
                )
                fig.update_traces(marker=dict(size=8), mode='lines+markers')
                st.plotly_chart(fig)
