import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from polygon import RESTClient
from datetime import date, timedelta

# ----------------------
# Polygon API
# ----------------------
POLYGON_API_KEY = "pm85pW39kP0RVJaPt8fbhJJjujMOP2vE"
client = RESTClient(POLYGON_API_KEY)

# ----------------------
# Page Setup
# ----------------------
st.set_page_config(page_title="Earnings IV/HV Dashboard", layout="wide")
st.title("📊 Earnings IV/HV Dashboard")
st.markdown("""
Analyze upcoming earnings using **IV/HV**, **IV curves**, and a simple **Earnings Calendar**.
""")

# ----------------------
# Sidebar
# ----------------------
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_area(
    "Enter tickers (comma-separated)",
    "AAPL,MSFT,TSLA,NVDA,AMZN"
)
hv_lookback = st.sidebar.slider("HV Lookback Period (days)", 10, 90, 30)
show_iv_curve = st.sidebar.checkbox("Show IV Curve", True)
show_calendar = st.sidebar.checkbox("Show Earnings Calendar", True)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

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
# Earnings Calendar
# ----------------------
@st.cache_data
def fetch_next_earnings(ticker):
    try:
        today = date.today()
        one_month = today + timedelta(days=30)
        data = client.reference_earnings(ticker=ticker, from_=today.isoformat(), to=one_month.isoformat())
        if data and "results" in data and len(data["results"])>0:
            return pd.to_datetime(data["results"][0]["reporting_date"])
        return None
    except:
        return None

# ----------------------
# Run Analysis
# ----------------------
results = []

if st.button("Run Analysis"):
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(tickers):
        hv = fetch_historical_volatility(ticker, hv_lookback)
        iv = fetch_iv_avg(ticker)
        ratio = round(iv/hv,2) if hv and iv else None
        earnings_date = fetch_next_earnings(ticker)

        results.append({
            "Ticker": ticker,
            "IV (%)": iv,
            "HV (%)": hv,
            "IV/HV Ratio": ratio,
            "Next Earnings": earnings_date
        })

        # Update progress
        progress_bar.progress((idx+1)/len(tickers))
        progress_text.text(f"Processing {ticker} ({idx+1}/{len(tickers)})...")

    df = pd.DataFrame(results)
    df.sort_values("IV/HV Ratio", inplace=True)
    st.success("✅ Analysis Complete!")
    st.dataframe(df)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "iv_hv_earnings.csv", "text/csv")

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

    # Earnings Calendar
    if show_calendar:
        st.subheader("📅 Upcoming Earnings Calendar")
        calendar_df = df[["Ticker","Next Earnings"]].dropna().sort_values("Next Earnings")
        st.dataframe(calendar_df)
