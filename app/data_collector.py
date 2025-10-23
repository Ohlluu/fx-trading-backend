#!/usr/bin/env python3
"""
XAUUSD Historical Data Collector
Gathers 2+ years of OANDA data for professional analysis
"""

import asyncio
import os
from datetime import datetime, timedelta
import pandas as pd
import pytz
from app.oanda_feed import get_xauusd_candles
import httpx

# OANDA API Configuration
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "1c2ee716aac27b425f2fd7a8ffbe9b9a-7a3b3da61668a804b56e2974ff21aaa0")
OANDA_BASE_URL = "https://api-fxpractice.oanda.com/v3"

async def get_historical_candles_range(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Get XAUUSD candles for a specific date range using OANDA's 'from' and 'to' parameters
    """
    url = f"{OANDA_BASE_URL}/instruments/XAU_USD/candles"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "RFC3339"
    }

    # Convert to RFC3339 format for OANDA
    from_time = start_date.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
    to_time = end_date.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')

    params = {
        "granularity": "H1",
        "from": from_time,
        "to": to_time,
        "price": "M",  # Mid prices
        "includeIncompleteCandles": "false"
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if "candles" in data and data["candles"]:
                    candles = []

                    for candle in data["candles"]:
                        if candle["complete"]:
                            # Parse RFC3339 timestamp
                            timestamp = pd.to_datetime(candle["time"], utc=True)

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
                        df = pd.DataFrame(candles, columns=[
                            "time", "open", "high", "low", "close", "volume"
                        ])

                        df = df.sort_values("time")
                        df = df.set_index("time", drop=False)
                        return df

            print(f"❌ OANDA range request failed: {response.status_code}")
            if response.status_code != 200:
                print(f"Error response: {response.text}")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Range request error: {e}")
        return pd.DataFrame()

async def collect_2_years_data() -> pd.DataFrame:
    """
    Collect 2+ years of XAUUSD hourly data by making multiple requests
    """
    print("=== COLLECTING 2+ YEARS OF XAUUSD DATA ===")

    # Define date ranges (working backwards from now)
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=730)  # 2 years ago

    print(f"Target range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    all_data = []

    # Split into 6-month chunks to avoid API limits
    chunk_size = timedelta(days=180)  # ~6 months
    current_start = start_date
    chunk_number = 1

    while current_start < end_date:
        current_end = min(current_start + chunk_size, end_date)

        print(f"\nChunk {chunk_number}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")

        # Get data for this chunk
        chunk_data = await get_historical_candles_range(current_start, current_end)

        if not chunk_data.empty:
            print(f"✅ Retrieved {len(chunk_data)} candles for chunk {chunk_number}")
            all_data.append(chunk_data)
        else:
            print(f"❌ No data for chunk {chunk_number}")

        current_start = current_end
        chunk_number += 1

        # Small delay to be nice to the API
        await asyncio.sleep(1)

    if all_data:
        # Combine all chunks
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates and sort
        combined_df = combined_df.drop_duplicates(subset=['time'])
        combined_df = combined_df.sort_values('time')
        combined_df = combined_df.set_index('time', drop=False)

        print(f"\n✅ TOTAL DATA COLLECTED: {len(combined_df)} candles")

        # Calculate actual time span
        actual_start = combined_df.index[0]
        actual_end = combined_df.index[-1]
        time_span = actual_end - actual_start
        months_covered = time_span.days / 30.44

        print(f"Actual range: {actual_start} to {actual_end}")
        print(f"Time span: {time_span.days} days ({months_covered:.1f} months)")

        return combined_df
    else:
        print("❌ No data collected")
        return pd.DataFrame()

async def analyze_session_patterns(df: pd.DataFrame):
    """
    Analyze XAUUSD patterns by trading sessions (New York time focus)
    """
    if df.empty:
        print("No data to analyze")
        return

    print("\n=== SESSION PATTERN ANALYSIS (NY TIME) ===")

    # Convert to Chicago/New York time
    chicago_tz = pytz.timezone('America/Chicago')
    df_ny = df.copy()
    df_ny['ny_time'] = df_ny.index.tz_convert(chicago_tz)
    df_ny['ny_hour'] = df_ny['ny_time'].dt.hour

    # Define NY time sessions
    sessions = {
        'Asian_Overlap': list(range(19, 24)) + list(range(0, 3)),  # 7PM-3AM
        'London_Fix_AM': [5, 6],  # 5-6AM (10:30 GMT)
        'London_NY_Overlap': list(range(8, 12)),  # 8AM-12PM
        'London_Fix_PM': [10, 11],  # 10-11AM (15:00 GMT)
        'NY_Afternoon': list(range(14, 17)),  # 2-5PM
    }

    print("SESSION VOLATILITY ANALYSIS:")

    for session_name, hours in sessions.items():
        session_data = df_ny[df_ny['ny_hour'].isin(hours)]

        if not session_data.empty:
            # Calculate average volatility (high-low range)
            volatility = (session_data['high'] - session_data['low']).mean()

            # Count significant moves (>$20 range)
            significant_moves = len(session_data[(session_data['high'] - session_data['low']) > 20])
            total_candles = len(session_data)

            print(f"{session_name:20}: Avg volatility ${volatility:.2f}, Big moves {significant_moves}/{total_candles} ({significant_moves/total_candles*100:.1f}%)")

    return df_ny

if __name__ == "__main__":
    async def main():
        # Collect 2 years of data
        df = await collect_2_years_data()

        if not df.empty:
            # Save to CSV for analysis
            df.to_csv('data/XAUUSD_2YEARS_H1.csv', index=False)
            print(f"✅ Data saved to data/XAUUSD_2YEARS_H1.csv")

            # Run session analysis
            await analyze_session_patterns(df)

    asyncio.run(main())