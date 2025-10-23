#!/usr/bin/env python3
"""
OANDA Data Feed for XAUUSD Trading System
Provides real-time and historical XAUUSD data that matches TradeLocker/TradingView pricing
"""

import os
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import httpx
import pytz
from datetime import datetime as dt_helper

# OANDA API Configuration
OANDA_API_KEY = "1c2ee716aac27b425f2fd7a8ffbe9b9a-7a3b3da61668a804b56e2974ff21aaa0"
OANDA_BASE_URL = "https://api-fxpractice.oanda.com/v3"
ACCOUNT_ID = "101-001-37143591-001"

async def get_current_xauusd_price() -> Optional[float]:
    """
    Get current XAUUSD price from OANDA API
    Returns mid price that matches TradeLocker/TradingView
    """
    url = f"{OANDA_BASE_URL}/accounts/{ACCOUNT_ID}/pricing"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "UNIX"
    }
    params = {"instruments": "XAU_USD"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if "prices" in data and data["prices"]:
                    price_info = data["prices"][0]

                    # Extract bid/ask
                    bid = None
                    ask = None

                    if "bids" in price_info and price_info["bids"]:
                        bid = float(price_info["bids"][0]["price"])

                    if "asks" in price_info and price_info["asks"]:
                        ask = float(price_info["asks"][0]["price"])

                    if bid and ask:
                        mid_price = (bid + ask) / 2
                        return round(mid_price, 2)

            print(f"❌ OANDA pricing error: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ OANDA API error: {e}")
        return None

async def get_xauusd_candles(count: int = 1000) -> Optional[pd.DataFrame]:
    """
    Get historical XAUUSD hourly candles from OANDA
    Returns DataFrame with columns: time, open, high, low, close, volume
    """
    url = f"{OANDA_BASE_URL}/instruments/XAU_USD/candles"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "UNIX"
    }
    params = {
        "granularity": "H1",  # 1-hour candles
        "count": count,
        "price": "M",  # Mid prices (matches our mid price calculation)
        "includeIncompleteCandles": "false"  # Exclude current open candle
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if "candles" in data and data["candles"]:
                    candles = []

                    for candle in data["candles"]:
                        if candle["complete"]:  # Only completed candles
                            # Convert UNIX timestamp to datetime (handle float timestamps)
                            timestamp = datetime.fromtimestamp(
                                float(candle["time"]),
                                tz=pytz.UTC
                            )

                            # Extract OHLC from mid prices
                            mid = candle["mid"]
                            candles.append([
                                timestamp,
                                float(mid["o"]),  # open
                                float(mid["h"]),  # high
                                float(mid["l"]),  # low
                                float(mid["c"]),  # close
                                int(candle.get("volume", 0))  # volume
                            ])

                    if candles:
                        # Create DataFrame
                        df = pd.DataFrame(candles, columns=[
                            "time", "open", "high", "low", "close", "volume"
                        ])

                        # Sort by time (ascending)
                        df = df.sort_values("time")

                        # Extra filter: ensure we exclude the current hour to prevent signal instability
                        current_hour = dt_helper.now(pytz.UTC).replace(minute=0, second=0, microsecond=0)
                        df = df[df["time"] < current_hour]

                        # Set time as index for compatibility with existing system
                        df = df.set_index("time", drop=False)

                        return df

            print(f"❌ OANDA candles error: {response.status_code}")
            if response.status_code != 200:
                print(f"Response: {response.text}")
            return None

    except Exception as e:
        print(f"❌ OANDA candles API error: {e}")
        return None

async def get_latest_candle_data() -> Optional[Dict[str, Any]]:
    """
    Get just the latest completed hourly candle
    Returns dict with datetime, OHLC, and volume
    """
    url = f"{OANDA_BASE_URL}/instruments/XAU_USD/candles"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "UNIX"
    }
    params = {
        "granularity": "H1",
        "count": "2",  # Get 2 candles to ensure we get the latest complete one
        "price": "M",
        "includeIncompleteCandles": "false"
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if "candles" in data and data["candles"]:
                    # Find the latest completed candle
                    completed_candle = None
                    for candle in data["candles"]:
                        if candle["complete"]:
                            completed_candle = candle
                            break

                    if completed_candle:
                        timestamp = datetime.fromtimestamp(
                            float(completed_candle["time"]),
                            tz=pytz.UTC
                        )

                        mid = completed_candle["mid"]
                        return {
                            "datetime": timestamp.isoformat(),
                            "open": float(mid["o"]),
                            "high": float(mid["h"]),
                            "low": float(mid["l"]),
                            "close": float(mid["c"]),
                            "volume": int(completed_candle.get("volume", 0))
                        }

            return None

    except Exception as e:
        print(f"❌ Latest candle error: {e}")
        return None

if __name__ == "__main__":
    async def test_oanda_feed():
        print("=== TESTING OANDA FEED ===")

        # Test current price
        print("1. Testing current price...")
        current_price = await get_current_xauusd_price()
        if current_price:
            print(f"✅ Current XAUUSD price: ${current_price}")
        else:
            print("❌ Failed to get current price")

        print()

        # Test latest candle
        print("2. Testing latest candle...")
        latest_candle = await get_latest_candle_data()
        if latest_candle:
            print(f"✅ Latest candle: {latest_candle['datetime']}")
            print(f"   OHLC: {latest_candle['open']}/{latest_candle['high']}/{latest_candle['low']}/{latest_candle['close']}")
        else:
            print("❌ Failed to get latest candle")

        print()

        # Test historical data (small sample)
        print("3. Testing historical data (10 candles)...")
        df = await get_xauusd_candles(count=10)
        if df is not None and not df.empty:
            print(f"✅ Retrieved {len(df)} historical candles")
            print("Latest 3 candles:")
            for i in range(-3, 0):
                row = df.iloc[i]
                time = df.index[i]
                print(f"  {time}: Close ${row['close']}")
        else:
            print("❌ Failed to get historical data")

    asyncio.run(test_oanda_feed())

async def get_current_price(instrument: str) -> Optional[float]:
    """
    Get current price for any OANDA instrument (generic function)
    instrument: OANDA format like 'GBP_USD', 'XAU_USD', etc.
    """
    url = f"{OANDA_BASE_URL}/accounts/{ACCOUNT_ID}/pricing"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "UNIX"
    }
    params = {"instruments": instrument}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if "prices" in data and data["prices"]:
                    price_info = data["prices"][0]

                    # Extract bid/ask
                    bid = None
                    ask = None

                    if "bids" in price_info and price_info["bids"]:
                        bid = float(price_info["bids"][0]["price"])

                    if "asks" in price_info and price_info["asks"]:
                        ask = float(price_info["asks"][0]["price"])

                    if bid and ask:
                        mid_price = (bid + ask) / 2
                        # Different precision for different instruments
                        if instrument == "XAU_USD":
                            return round(mid_price, 2)
                        else:  # For major FX pairs like GBP_USD
                            return round(mid_price, 5)

            print(f"❌ OANDA pricing error for {instrument}: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ OANDA API error for {instrument}: {e}")
        return None

async def get_gbpusd_candles(count: int = 1000) -> Optional[pd.DataFrame]:
    """
    Get historical GBPUSD hourly candles from OANDA
    Returns DataFrame with columns: time, open, high, low, close, volume
    """
    url = f"{OANDA_BASE_URL}/instruments/GBP_USD/candles"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "UNIX"
    }
    params = {
        "granularity": "H1",  # 1-hour candles
        "count": count,
        "price": "M",  # Mid prices
        "includeIncompleteCandles": "false"
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if "candles" in data and data["candles"]:
                    candles = data["candles"]

                    # Convert to our standard format
                    rows = []
                    for candle in candles:
                        # Convert timestamp to datetime
                        timestamp = datetime.fromtimestamp(float(candle["time"]), tz=pytz.UTC)

                        # Extract OHLC from mid prices
                        mid = candle.get("mid", {})

                        row = {
                            "time": timestamp,
                            "open": float(mid.get("o", 0)),
                            "high": float(mid.get("h", 0)),
                            "low": float(mid.get("l", 0)),
                            "close": float(mid.get("c", 0)),
                            "volume": int(candle.get("volume", 0))
                        }
                        rows.append(row)

                    # Create DataFrame
                    df = pd.DataFrame(rows)
                    df.set_index("time", inplace=True)
                    df.sort_index(inplace=True)

                    # Extra filter: ensure we exclude the current hour to prevent signal instability
                    current_hour = dt_helper.now(pytz.UTC).replace(minute=0, second=0, microsecond=0)
                    df = df[df.index < current_hour]

                    print(f"✅ OANDA GBPUSD: Retrieved {len(df)} candles from {df.index[0]} to {df.index[-1]} (excluding current hour)")
                    return df
                else:
                    print(f"❌ No GBPUSD candles in response")
                    return None
            else:
                print(f"❌ OANDA API error: {response.status_code}")
                return None

    except Exception as e:
        print(f"❌ OANDA GBPUSD candle fetch error: {e}")
        return None