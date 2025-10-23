#!/usr/bin/env python3
"""
Indices Data Feed using indices-api.com
Provides real-time index data for confluence analysis
"""

import os
import httpx
import pandas as pd
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
import pytz

# Get API key from environment
def _get_indices_api_key() -> str:
    key = os.getenv("INDICES_API_KEY", "9gzpiyem4ga03klqi1yd7zhzc8fvoe6yqpdpn7073rk033ukq03xx08oqnj6").strip()
    if not key:
        raise RuntimeError("INDICES_API_KEY env var is not set.")
    return key

# Map our index names to indices-api.com symbols
# Updated to focus on SPX500 and NAS100 only (confirmed accurate data)
INDICES_SYMBOL_MAP = {
    "SPX500": "GSPC",     # S&P 500 - confirmed accurate
    "US500": "GSPC",      # alias
    "NAS100": "NDX",      # NASDAQ 100 - confirmed accurate
    "US100": "NDX",       # alias
    # Note: UK100 and DE30 symbols in this API don't match broker prices
    # "DE30": "GDAXI",    # German DAX shows wrong price
    # "UK100": "FTSE",    # FTSE 100 shows wrong price
    "US30": "DJI",        # Dow Jones (if needed later)
}

async def fetch_current_index_price(index: str) -> Optional[float]:
    """
    Fetch current price for a single index
    Returns the actual index value (not scaled)
    """
    api_key = _get_indices_api_key()

    if index.upper() not in INDICES_SYMBOL_MAP:
        raise ValueError(f"Index {index} not supported. Available: {list(INDICES_SYMBOL_MAP.keys())}")

    symbol = INDICES_SYMBOL_MAP[index.upper()]

    url = "https://indices-api.com/api/latest"
    params = {
        "access_key": api_key,
        "symbols": symbol
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        if data.get("data", {}).get("success"):
            rates = data["data"]["rates"]
            # The actual index value is in USD{SYMBOL} format
            price_key = f"USD{symbol}"
            if price_key in rates:
                return float(rates[price_key])
            else:
                # Fallback: look for the symbol directly
                return float(rates.get(symbol, 0))
        else:
            return None

    except Exception as e:
        print(f"Error fetching {index} price: {e}")
        return None

async def fetch_multiple_index_prices(indices: list) -> Dict[str, float]:
    """
    Fetch current prices for multiple indices
    """
    api_key = _get_indices_api_key()

    # Map indices to symbols
    symbols = []
    index_to_symbol = {}
    for index in indices:
        if index.upper() in INDICES_SYMBOL_MAP:
            symbol = INDICES_SYMBOL_MAP[index.upper()]
            symbols.append(symbol)
            index_to_symbol[index] = symbol

    if not symbols:
        return {}

    url = "https://indices-api.com/api/latest"
    params = {
        "access_key": api_key,
        "symbols": ",".join(symbols)
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        result = {}
        if data.get("data", {}).get("success"):
            rates = data["data"]["rates"]
            timestamp_str = data["data"].get("date", "")

            for index, symbol in index_to_symbol.items():
                price_key = f"USD{symbol}"
                if price_key in rates:
                    result[index] = float(rates[price_key])
                elif symbol in rates:
                    result[index] = float(rates[symbol])

            # Add metadata
            result["timestamp"] = timestamp_str
            result["source"] = "indices-api.com"

        return result

    except Exception as e:
        print(f"Error fetching multiple index prices: {e}")
        return {}

def create_synthetic_h1_candle(index: str, current_price: float) -> pd.DataFrame:
    """
    Create a synthetic H1 candle for current price
    This is a temporary solution until we can get proper OHLC data
    """
    current_time = pd.Timestamp.now(tz="UTC").floor("h")

    # Create a simple candle where OHLC are all the current price
    # In reality, you'd want proper OHLC data
    data = {
        "time": [current_time],
        "open": [current_price],
        "high": [current_price],
        "low": [current_price],
        "close": [current_price],
        "volume": [1000]  # Dummy volume
    }

    df = pd.DataFrame(data)
    df = df.set_index("time", drop=False)

    return df

async def get_current_index_data(index: str) -> Dict[str, Any]:
    """
    Get current index data with metadata
    """
    price = await fetch_current_index_price(index)

    if price is None:
        return {"error": f"Could not fetch price for {index}"}

    chicago_tz = pytz.timezone('America/Chicago')
    utc_now = datetime.now(pytz.UTC)
    chicago_now = utc_now.astimezone(chicago_tz)

    return {
        "index": index,
        "current_price": price,
        "timestamp_utc": utc_now.isoformat(),
        "timestamp_chicago": chicago_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source": "indices-api.com",
        "is_live": True
    }

# Test function
async def test_indices_api():
    """Test the indices API with our symbols"""
    print("Testing indices-api.com connection...")

    test_indices = ["SPX500", "NAS100", "DE30", "UK100"]
    prices = await fetch_multiple_index_prices(test_indices)

    print("Results:")
    for index in test_indices:
        if index in prices:
            print(f"{index}: {prices[index]:,.2f}")
        else:
            print(f"{index}: No data")

    if "timestamp" in prices:
        print(f"Data timestamp: {prices['timestamp']}")

if __name__ == "__main__":
    asyncio.run(test_indices_api())