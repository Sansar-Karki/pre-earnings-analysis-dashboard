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
Analyze upcoming earnings using **IV/HV** and **IV curves**.
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
# Main Analysis
# ----------------------
results = []

if st.button("Run Analysis"):
    progress_text = st.empty()
    progress_bar = st.progress(0)
    for idx, ticker in enumerate(tickers):
        hv = fetch_historical_volatility(ticker, hv_lookback)
        iv = fetch_iv_avg(ticker)
        ratio = round(iv/hv,2) if hv and iv else None

        results.append({
            "Ticker": ticker,
            "IV (%)": iv,
            "HV (%)": hv,
            "IV/HV Ratio": ratio
        })

        # Update progress bar
        progress_bar.progress((idx+1)/len(tickers))
        progress_text.text(f"Processing {ticker} ({idx+1}/{len(tickers)})...")

    df = pd.DataFrame(results)
    df.sort_values("IV/HV Ratio", inplace=True)  # sort by lowest IV/HV
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
