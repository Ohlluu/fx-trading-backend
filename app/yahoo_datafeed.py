# backend/app/yahoo_datafeed.py
import yfinance as yf
import pandas as pd
import os
from typing import Optional

# Yahoo Finance symbol mapping for indices
YAHOO_SYMBOL_MAP = {
    "NAS100": "^NDX",    # Nasdaq 100
    "DE30": "^GDAXI",    # German DAX 30
    "UK100": "^FTSE",    # FTSE 100
    "SPX500": "^GSPC"    # S&P 500
}

def fetch_yahoo_data(symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """
    Fetch data from Yahoo Finance for indices
    """
    if symbol not in YAHOO_SYMBOL_MAP:
        print(f"❌ {symbol} not supported in Yahoo mapping")
        return None

    yahoo_symbol = YAHOO_SYMBOL_MAP[symbol]
    print(f"📈 Fetching {symbol} ({yahoo_symbol}) from Yahoo Finance...")

    try:
        ticker = yf.Ticker(yahoo_symbol)
        data = ticker.history(period=period, interval="1h")

        if data.empty:
            print(f"❌ No data returned for {symbol}")
            return None

        # Convert to our standard format
        df = pd.DataFrame({
            "time": data.index,
            "open": data["Open"],
            "high": data["High"],
            "low": data["Low"],
            "close": data["Close"],
            "volume": data["Volume"]
        })

        # Reset index to make time a column
        df = df.reset_index(drop=True)
        df["time"] = df["time"].dt.tz_convert("UTC")

        print(f"✅ {symbol}: Got {len(df)} hourly bars")
        return df

    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

def save_yahoo_data(symbol: str) -> bool:
    """Fetch and save Yahoo data to CSV"""
    df = fetch_yahoo_data(symbol)
    if df is None:
        return False

    # Save to data directory
    os.makedirs("data", exist_ok=True)
    filename = f"data/{symbol}_H1.csv"
    df.to_csv(filename, index=False)
    print(f"💾 Saved {symbol} data to {filename}")
    return True

def collect_all_index_data():
    """Collect data for all target indices"""
    print("=== COLLECTING INDEX DATA FROM YAHOO FINANCE ===")
    indices = ["NAS100", "DE30", "UK100", "SPX500"]

    results = {}
    for symbol in indices:
        success = save_yahoo_data(symbol)
        results[symbol] = "✅ Success" if success else "❌ Failed"
        print()

    print("📊 COLLECTION SUMMARY:")
    for symbol, status in results.items():
        print(f"{symbol}: {status}")

    return results

if __name__ == "__main__":
    collect_all_index_data()