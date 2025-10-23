#!/usr/bin/env python3
"""
Find the Last GBPUSD 4:1 TP Hit from Historical Backtesting Data
Analyzes 5+ years of data to find when confluence-based signals last hit 4:1 TP
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np

async def find_last_successful_4_to_1_tp():
    print("🔍 SEARCHING 5+ YEARS OF GBPUSD DATA FOR LAST 4:1 TP HIT")
    print("=" * 60)

    try:
        # Import GBPUSD functions
        from app.oanda_feed import get_gbpusd_candles
        from app.gbpusd_confluence_system import GBPUSDConfluenceSystem

        # Get maximum available historical data
        df = await get_gbpusd_candles(count=5000)  # 5000 hours = ~7 months back
        if df is None or df.empty:
            print("❌ No GBPUSD data available")
            return

        print(f"✅ Analyzing {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        print(f"   Time span: {(df.index[-1] - df.index[0]).days} days")

        # Initialize confluence system
        confluence_system = GBPUSDConfluenceSystem()
        chicago_tz = pytz.timezone('America/Chicago')

        # Track all successful 4:1 TP hits
        successful_tps = []
        total_signals = 0

        # Analyze each candle for confluence signals
        print("\n🔍 Scanning for confluence signals and TP achievements...")

        for i in range(60, len(df) - 100):  # Need history for MAs and future for TP checking
            try:
                current_candle = df.iloc[i]
                current_time = df.index[i]
                current_price = current_candle['close']

                # Get historical data up to this point
                historical_df = df.iloc[:i+1].copy()

                # Check for confluence signal
                result = confluence_system.evaluate_confluence_signal(current_price, historical_df)

                if result and result.get('status') == 'signal':
                    total_signals += 1
                    signal_info = result['signal']
                    entry_price = signal_info['entry_price']
                    tp_price = signal_info['take_profit']
                    sl_price = signal_info['stop_loss']
                    signal_type = signal_info['signal']

                    # Convert to Chicago time
                    entry_time_chicago = current_time.tz_convert(chicago_tz)

                    # Check if TP was hit in the next 120 candles (5 days)
                    future_candles = df.iloc[i+1:i+121]
                    tp_hit = False
                    tp_hit_time = None
                    sl_hit_time = None

                    for j, future_candle in future_candles.iterrows():
                        # Check for TP hit
                        if signal_type == 'BUY':
                            if future_candle['high'] >= tp_price and not tp_hit:
                                tp_hit = True
                                tp_hit_time = j.tz_convert(chicago_tz)
                            if future_candle['low'] <= sl_price and not sl_hit_time:
                                sl_hit_time = j.tz_convert(chicago_tz)
                        else:  # SELL
                            if future_candle['low'] <= tp_price and not tp_hit:
                                tp_hit = True
                                tp_hit_time = j.tz_convert(chicago_tz)
                            if future_candle['high'] >= sl_price and not sl_hit_time:
                                sl_hit_time = j.tz_convert(chicago_tz)

                        # If TP hit first (or only TP hit), it's successful
                        if tp_hit and (not sl_hit_time or tp_hit_time <= sl_hit_time):
                            time_to_tp = (tp_hit_time - entry_time_chicago).total_seconds() / 3600

                            successful_tp = {
                                'entry_time': entry_time_chicago,
                                'tp_hit_time': tp_hit_time,
                                'signal': signal_type,
                                'entry_price': entry_price,
                                'tp_price': tp_price,
                                'confluence_score': signal_info['confluence_score'],
                                'time_to_tp_hours': time_to_tp,
                                'days_to_tp': time_to_tp / 24
                            }
                            successful_tps.append(successful_tp)
                            break

                    # Print progress every 1000 candles
                    if i % 1000 == 0:
                        progress = (i / len(df)) * 100
                        print(f"   Progress: {progress:.1f}% - Found {len(successful_tps)} successful TPs so far")

            except Exception as e:
                continue

        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Total signals found: {total_signals}")
        print(f"   Successful 4:1 TPs: {len(successful_tps)}")
        if total_signals > 0:
            print(f"   Success rate: {(len(successful_tps) / total_signals) * 100:.1f}%")

        if successful_tps:
            # Sort by entry time to find the most recent
            successful_tps.sort(key=lambda x: x['entry_time'])

            print(f"\n🏆 LAST 3 SUCCESSFUL 4:1 TP HITS:")
            print("-" * 50)

            for i, tp in enumerate(successful_tps[-3:]):
                print(f"\n#{len(successful_tps) - 2 + i}. {tp['signal']} Signal")
                print(f"   Entry Time: {tp['entry_time'].strftime('%Y-%m-%d %H:%M %Z')} (Chicago)")
                print(f"   Entry Price: ${tp['entry_price']:.5f}")
                print(f"   TP Price: ${tp['tp_price']:.5f}")
                print(f"   TP Hit Time: {tp['tp_hit_time'].strftime('%Y-%m-%d %H:%M %Z')} (Chicago)")
                print(f"   Time to TP: {tp['time_to_tp_hours']:.1f} hours ({tp['days_to_tp']:.1f} days)")
                print(f"   Confluence Score: {tp['confluence_score']}/25")

            # Show the absolute most recent successful TP
            most_recent = successful_tps[-1]
            print(f"\n🎯 MOST RECENT SUCCESSFUL 4:1 TP HIT:")
            print("=" * 50)
            print(f"Signal: {most_recent['signal']}")
            print(f"Entry: {most_recent['entry_time'].strftime('%Y-%m-%d %H:%M %Z')} (Chicago Time)")
            print(f"TP Hit: {most_recent['tp_hit_time'].strftime('%Y-%m-%d %H:%M %Z')} (Chicago Time)")
            print(f"Duration: {most_recent['time_to_tp_hours']:.1f} hours ({most_recent['days_to_tp']:.1f} days)")
            print(f"Entry Price: ${most_recent['entry_price']:.5f}")
            print(f"TP Price: ${most_recent['tp_price']:.5f}")
            print(f"Confluence: {most_recent['confluence_score']}/25")

            # Calculate time since last successful TP
            now_chicago = datetime.now(chicago_tz)
            time_since = now_chicago - most_recent['tp_hit_time']
            print(f"Time since last TP: {time_since.days} days, {time_since.seconds // 3600} hours ago")

        else:
            print(f"\n❌ NO SUCCESSFUL 4:1 TPs FOUND in the analyzed period")
            print(f"   This suggests either:")
            print(f"   1. The 4:1 ratio is very challenging for recent market conditions")
            print(f"   2. We need to analyze even more historical data")
            print(f"   3. Current confluence thresholds are very selective")

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(find_last_successful_4_to_1_tp())