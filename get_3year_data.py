#!/usr/bin/env python3
"""
Get 3 Years of XAUUSD Data for Comprehensive Confluence Analysis
"""

import asyncio
import httpx
import pandas as pd
from datetime import datetime, timedelta

async def get_3_years_xauusd():
    print('🔄 FETCHING 3 YEARS OF XAUUSD DATA FOR CONFLUENCE ANALYSIS')
    print('This will get maximum data available...')

    api_key = '0e24ff3eb6ef415dba0cebcf04593e4f'

    # Get maximum data (5000 candles = ~7 months of hourly)
    # We'll need to make multiple calls to get 3 years
    all_data = []

    # Start from current date and work backwards
    end_date = datetime.now()

    for batch in range(6):  # 6 batches should cover ~3 years
        print(f'📡 Fetching batch {batch + 1}/6...')

        params = {
            'symbol': 'XAU/USD',
            'interval': '1h',
            'outputsize': '5000',
            'timezone': 'UTC',
            'format': 'JSON',
            'apikey': api_key,
        }

        if batch > 0:
            # For subsequent batches, use end_date parameter
            end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
            params['end_date'] = end_date_str

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get('https://api.twelvedata.com/time_series', params=params)
                data = r.json()

            if data.get('status') == 'ok' and data.get('values'):
                values = data['values']
                print(f'   ✅ Got {len(values)} candles')
                all_data.extend(values)

                # Update end_date to the oldest candle from this batch
                oldest_candle = values[-1]  # Last in array is oldest
                end_date = datetime.strptime(oldest_candle['datetime'], '%Y-%m-%d %H:%M:%S')

                # Subtract 1 hour to avoid overlap
                end_date -= timedelta(hours=1)

            else:
                print(f'   ❌ API Error: {data.get("message", "Unknown")}')
                break

            # Add delay between requests to respect rate limits
            await asyncio.sleep(1)

        except Exception as e:
            print(f'   ❌ Exception: {e}')
            break

    if all_data:
        print(f'\n📊 TOTAL DATA COLLECTED: {len(all_data)} hourly candles')

        # Convert to DataFrame and remove duplicates
        df = pd.DataFrame(all_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')

        # Convert price columns
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        # Add timezone info
        import pytz
        utc = pytz.UTC
        ny_tz = pytz.timezone('America/New_York')

        df['datetime'] = df['datetime'].dt.tz_localize(utc)
        df['ny_time'] = df['datetime'].dt.tz_convert(ny_tz)
        df['ny_hour'] = df['ny_time'].dt.hour
        df['ny_date_str'] = df['ny_time'].dt.date.astype(str)

        # Save the comprehensive dataset
        df.to_csv('data/XAUUSD_3YEAR_DATA.csv', index=False)

        print(f'💾 Saved {len(df)} unique candles to data/XAUUSD_3YEAR_DATA.csv')
        print(f'📅 Date range: {df.iloc[0]["datetime"]} to {df.iloc[-1]["datetime"]}')
        print(f'💰 Price range: ${df["low"].min():.2f} to ${df["high"].max():.2f}')

        return len(df)
    else:
        print('❌ No data collected')
        return 0

if __name__ == "__main__":
    result = asyncio.run(get_3_years_xauusd())
    print(f'\n🎯 Ready for 3-year confluence analysis with {result} candles!')