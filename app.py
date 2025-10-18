import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------
# Page Setup
# ----------------------
st.set_page_config(page_title="Earnings IV/HV Analyzer", layout="wide")
st.title("📊 Earnings IV/HV Analyzer")
st.markdown("""
Analyze upcoming earnings using **IV/HV**, **IV percentile**, and **expected vs historical move**.
""")

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_area(
    "Enter tickers (comma-separated)",
    "AAPL,MSFT,TSLA,NVDA,AMZN"
)
hv_lookback = st.sidebar.slider("HV Lookback Period (days)", 10, 90, 30)
show_iv_curve = st.sidebar.checkbox("Show IV Curve for Tickers", value=True)
earnings_window = st.sidebar.slider("Days around earnings to estimate move", 1, 5, 1)

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
        return round(hv*100,2), data
    except:
        return None, None

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

@st.cache_data
def fetch_iv_percentile(ticker, lookback_days=252):
    _, data = fetch_historical_volatility(ticker, lookback_days)
    if data is None:
        return None
    tk = yf.Ticker(ticker)
    try:
        chain = fetch_iv_chain(ticker, expirations=1)
        if not chain:
            return None
        iv_now = list(chain.values())[0]["impliedVolatility"].mean()*100
        # approximate IV percentile: compare to historical daily HV
        data["iv_percentile"] = ((iv_now - data["returns"].rolling(lookback_days).std()*np.sqrt(252)*100) /
                                 (data["returns"].rolling(lookback_days).std()*np.sqrt(252)*100).rolling(lookback_days).max())*100
        return min(max(iv_now,0),100)
    except:
        return None

@st.cache_data
def fetch_expected_move(ticker):
    chain = fetch_iv_chain(ticker, expirations=1)
    if not chain:
        return None
    df = list(chain.values())[0]
    if df.empty:
        return None
    atm = df.iloc[(df["strike"]-yf.Ticker(ticker).history(period="1d")["Close"][-1]).abs().argsort()[0]]
    return round((atm["lastPrice"]*2)/yf.Ticker(ticker).history(period="1d")["Close"][-1]*100,2)

# ----------------------
# Main Analysis
# ----------------------
results = []

if st.button("Run Analysis"):
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(tickers):
        hv, hist_data = fetch_historical_volatility(ticker, hv_lookback)
        iv = fetch_iv_avg(ticker)
        iv_percent = fetch_iv_percentile(ticker)
        expected_move = fetch_expected_move(ticker)
        ratio = round(iv/hv,2) if hv and iv else None
        
        results.append({
            "Ticker": ticker,
            "IV (%)": iv,
            "HV (%)": hv,
            "IV/HV Ratio": ratio,
            "IV Percentile (%)": iv_percent,
            "Expected Move (%)": expected_move,
            "Historical Move (%)": round(hv*np.sqrt(earnings_window/252)*100,2) if hv else None
        })

        # Update progress
        progress_bar.progress((idx+1)/len(tickers))
        progress_text.text(f"Processing {ticker} ({idx+1}/{len(tickers)})...")

    df = pd.DataFrame(results)
    df.sort_values("IV/HV Ratio", inplace=True)
    st.success("✅ Analysis Complete!")
    st.dataframe(df)

    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "iv_hv_analysis.csv", "text/csv")

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
