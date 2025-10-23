#!/usr/bin/env python3
"""
Real-time XAUUSD price fetcher using OANDA API
Provides accurate pricing that matches TradeLocker/TradingView
"""

import asyncio
from typing import Optional

async def get_current_xauusd_price() -> Optional[float]:
    """
    Get current XAUUSD price using OANDA API
    Returns mid price that matches TradeLocker/TradingView pricing
    """
    try:
        from .oanda_feed import get_current_xauusd_price as oanda_get_current_price
        return await oanda_get_current_price()
    except ImportError:
        from oanda_feed import get_current_xauusd_price as oanda_get_current_price
        return await oanda_get_current_price()

async def get_latest_candle_data() -> Optional[dict]:
    """
    Get just the latest hourly candle for XAUUSD using OANDA API
    Returns the most recent completed OHLC data
    """
    try:
        from .oanda_feed import get_latest_candle_data as oanda_get_latest_candle
        return await oanda_get_latest_candle()
    except ImportError:
        from oanda_feed import get_latest_candle_data as oanda_get_latest_candle
        return await oanda_get_latest_candle()

if __name__ == "__main__":
    async def test_current_price():
        print("=== TESTING REAL-TIME PRICE FETCHER ===")

        # Test real-time price
        current_price = await get_current_xauusd_price()
        if current_price:
            print(f"✅ Current XAUUSD price: ${current_price:.2f}")
        else:
            print("❌ Failed to get current price")

        print()

        # Test latest candle
        latest_candle = await get_latest_candle_data()
        if latest_candle:
            print("✅ Latest hourly candle:")
            print(f"  Time: {latest_candle['datetime']}")
            print(f"  OHLC: {latest_candle['open']} / {latest_candle['high']} / {latest_candle['low']} / {latest_candle['close']}")
        else:
            print("❌ Failed to get latest candle")

    asyncio.run(test_current_price())