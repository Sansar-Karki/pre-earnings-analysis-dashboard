import pandas as pd
import yfinance as yf
import time
from tqdm import tqdm
from pathlib import Path

# ----------------------------
# Step 1: Load Nasdaq file
# ----------------------------
url = "/Users/sansarkarki/Downloads/nasdaqlisted.txt"  # your local file
nasdaq_df = pd.read_csv(url, sep="|", dtype=str)

# Ensure 'Symbol' column is string
nasdaq_df['Symbol'] = nasdaq_df['Symbol'].astype(str)

# ----------------------------
# Step 2: Filter common stocks
# ----------------------------
filtered_df = nasdaq_df[
    (nasdaq_df['Market Category'].isin(['Q','S'])) &
    (nasdaq_df['ETF'] == 'N') &
    (~nasdaq_df['Symbol'].fillna("").str.endswith(('W','U','R')))
].copy()

print(f"Filtered tickers (common stocks only): {len(filtered_df)}")

# ----------------------------
# Step 3: Add Sector and Market Cap with progress bar
# ----------------------------
filtered_df["Sector"] = None
filtered_df["MarketCap"] = None

# Optional: save periodically in case of interruption
save_interval = 50  # save every 50 tickers
output_file = Path("nasdaq_largecap.csv")

for i, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Fetching yfinance data"):
    ticker = row["Symbol"]
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        filtered_df.at[i, "Sector"] = info.get("sector")
        filtered_df.at[i, "MarketCap"] = info.get("marketCap")
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        continue

    time.sleep(0.5)  # prevent throttling

    # Periodically save
    if i % save_interval == 0 and i != 0:
        filtered_df.to_csv(output_file, index=False)
        print(f"Intermediate save at ticker {i}")

# ----------------------------
# Step 4: Filter large-cap companies (MarketCap > 10B)
# ----------------------------
# Some MarketCap might be None, so convert safely
filtered_df["MarketCap"] = pd.to_numeric(filtered_df["MarketCap"], errors="coerce")
largecap_df = filtered_df[filtered_df["MarketCap"] > 10e9].copy()
largecap_df.reset_index(drop=True, inplace=True)
print(f"Large-cap tickers (>10B): {len(largecap_df)}")

# ----------------------------
# Step 5: Save final CSV
# ----------------------------
largecap_df.to_csv(output_file, index=False)
print(f"CSV saved: {output_file}")
print(largecap_df.head())
