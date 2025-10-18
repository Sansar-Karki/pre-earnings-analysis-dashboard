import pandas as pd
import yfinance as yf
import time

# ----------------------------
# Step 1: Load Nasdaq FTP file directly
# ----------------------------
url = "http://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"
nasdaq_df = pd.read_csv(url, sep="|", dtype=str)

# Remove the last row (contains "File Creation Time" summary)
nasdaq_df = nasdaq_df[:-1]

print(f"Total tickers loaded from Nasdaq FTP: {len(nasdaq_df)}")

# ----------------------------
# Step 2: Filter common stocks
# ----------------------------
# Keep only:
# - Market Category Q (Nasdaq Global Market) or S (Nasdaq SmallCap Market)
# - ETF == 'N'
# - Remove symbols ending with W, U, or R (warrants/units/rights)
filtered_df = nasdaq_df[
    (nasdaq_df['Market Category'].isin(['Q','S'])) &
    (nasdaq_df['ETF'] == 'N') &
    (~nasdaq_df['Symbol'].str.endswith(('W','U','R')))
].copy()

print(f"Filtered tickers (common stocks only): {len(filtered_df)}")

# ----------------------------
# Step 3: Add Sector and Market Cap using yfinance
# ----------------------------
filtered_df["Sector"] = None
filtered_df["MarketCap"] = None

for i, row in filtered_df.iterrows():
    ticker = row["Symbol"]
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        filtered_df.at[i, "Sector"] = info.get("sector")
        filtered_df.at[i, "MarketCap"] = info.get("marketCap")
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        continue
    time.sleep(0.5)  # avoid throttling Yahoo Finance

# ----------------------------
# Step 4: Filter large-cap companies (MarketCap > 10B)
# ----------------------------
largecap_df = filtered_df[filtered_df["MarketCap"].astype(float) > 10e9].copy()
largecap_df.reset_index(drop=True, inplace=True)
print(f"Large-cap tickers (>10B): {len(largecap_df)}")

# ----------------------------
# Step 5: Save to CSV
# ----------------------------
largecap_df.to_csv("nasdaq_largecap.csv", index=False)
print("CSV saved: nasdaq_largecap.csv")
print(largecap_df.head())
