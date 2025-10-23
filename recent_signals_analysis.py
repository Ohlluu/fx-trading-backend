#!/usr/bin/env python3
"""
Recent Signals Analysis for XAUUSD and GBPUSD
Find the last confluence signals and their TP achievement times
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import sys
import os

# Add the current directory to Python path
sys.path.append('/Users/user/fx-app/backend')

async def analyze_recent_xauusd_signals():
    print("=== ANALYZING RECENT XAUUSD SIGNALS ===")

    try:
        from app.oanda_feed import get_xauusd_candles
        from app.smart_confluence_system import evaluate_smart_confluence_signal

        # Get recent data
        df = await get_xauusd_candles(count=1000)
        if df is None or df.empty:
            print("❌ No XAUUSD data available")
            return

        print(f"✅ Got {len(df)} XAUUSD candles from {df.index[0]} to {df.index[-1]}")

        # Look for recent signals (last 100 candles)
        recent_signals = []
        chicago_tz = pytz.timezone('America/Chicago')

        for i in range(len(df)-100, len(df)):
            if i < 50:  # Need MA history
                continue

            current_price = df['close'].iloc[i]
            historical_df = df.iloc[:i+1].copy()

            try:
                result = evaluate_smart_confluence_signal(current_price, historical_df)
                if result and result.get('signal') in ['BUY', 'SELL']:
                    entry_time_utc = df.index[i]
                    entry_time_chicago = entry_time_utc.tz_convert(chicago_tz)

                    signal_data = {
                        'entry_time_utc': entry_time_utc,
                        'entry_time_chicago': entry_time_chicago,
                        'signal': result['signal'],
                        'entry_price': result['entry_price'],
                        'take_profit': result['take_profit'],
                        'stop_loss': result['stop_loss'],
                        'confluence_score': result['confluence_score'],
                        'candle_index': i
                    }

                    # Check if TP was hit in subsequent candles
                    tp_hit_time = None
                    for j in range(i+1, min(i+73, len(df))):  # Check next 72 hours (3 days)
                        future_candle = df.iloc[j]
                        if result['signal'] == 'BUY':
                            if future_candle['high'] >= result['take_profit']:
                                tp_hit_time = df.index[j].tz_convert(chicago_tz)
                                break
                        else:  # SELL
                            if future_candle['low'] <= result['take_profit']:
                                tp_hit_time = df.index[j].tz_convert(chicago_tz)
                                break

                    if tp_hit_time:
                        time_to_tp = (tp_hit_time - entry_time_chicago).total_seconds() / 3600  # hours
                        signal_data['tp_hit_time'] = tp_hit_time
                        signal_data['time_to_tp_hours'] = time_to_tp

                    recent_signals.append(signal_data)

            except Exception as e:
                continue

        print(f"\n📊 Found {len(recent_signals)} recent XAUUSD signals")

        # Show the most recent signals
        for i, signal in enumerate(recent_signals[-3:]):  # Last 3 signals
            print(f"\n🔹 Signal #{len(recent_signals)-2+i}:")
            print(f"   Entry Time: {signal['entry_time_chicago'].strftime('%Y-%m-%d %H:%M %Z')}")
            print(f"   Signal: {signal['signal']}")
            print(f"   Entry Price: ${signal['entry_price']:.2f}")
            print(f"   Take Profit: ${signal['take_profit']:.2f}")
            print(f"   Confluence: {signal['confluence_score']}/20")

            if 'tp_hit_time' in signal:
                print(f"   TP Hit Time: {signal['tp_hit_time'].strftime('%Y-%m-%d %H:%M %Z')}")
                print(f"   Time to TP: {signal['time_to_tp_hours']:.1f} hours")
                print(f"   ✅ SUCCESSFUL TRADE")
            else:
                print(f"   ❌ TP not hit within 72 hours")

        # Find the most recent successful trade
        successful_trades = [s for s in recent_signals if 'tp_hit_time' in s]
        if successful_trades:
            latest_success = successful_trades[-1]
            print(f"\n🏆 MOST RECENT SUCCESSFUL XAUUSD TRADE:")
            print(f"   Entry: {latest_success['entry_time_chicago'].strftime('%Y-%m-%d %H:%M %Z')}")
            print(f"   TP Hit: {latest_success['tp_hit_time'].strftime('%Y-%m-%d %H:%M %Z')}")
            print(f"   Duration: {latest_success['time_to_tp_hours']:.1f} hours")
            print(f"   Signal: {latest_success['signal']} at ${latest_success['entry_price']:.2f}")

    except Exception as e:
        print(f"❌ XAUUSD analysis error: {e}")

async def analyze_recent_gbpusd_signals():
    print("\n=== ANALYZING RECENT GBPUSD SIGNALS ===")

    try:
        from app.oanda_feed import get_gbpusd_candles
        from app.gbpusd_confluence_system import evaluate_gbpusd_confluence_signal

        # Get recent data
        df = await get_gbpusd_candles(count=1000)
        if df is None or df.empty:
            print("❌ No GBPUSD data available")
            return

        print(f"✅ Got {len(df)} GBPUSD candles from {df.index[0]} to {df.index[-1]}")

        # Look for recent signals (last 100 candles)
        recent_signals = []
        chicago_tz = pytz.timezone('America/Chicago')

        for i in range(len(df)-100, len(df)):
            if i < 50:  # Need MA history
                continue

            current_price = df['close'].iloc[i]
            historical_df = df.iloc[:i+1].copy()

            try:
                result = evaluate_gbpusd_confluence_signal(current_price, historical_df)
                if result and result.get('status') == 'signal':
                    signal_info = result['signal']
                    entry_time_utc = df.index[i]
                    entry_time_chicago = entry_time_utc.tz_convert(chicago_tz)

                    signal_data = {
                        'entry_time_utc': entry_time_utc,
                        'entry_time_chicago': entry_time_chicago,
                        'signal': signal_info['signal'],
                        'entry_price': signal_info['entry_price'],
                        'take_profit': signal_info['take_profit'],
                        'stop_loss': signal_info['stop_loss'],
                        'confluence_score': signal_info['confluence_score'],
                        'candle_index': i
                    }

                    # Check if TP was hit in subsequent candles (up to 100 hours)
                    tp_hit_time = None
                    for j in range(i+1, min(i+101, len(df))):  # Check next 100 hours
                        future_candle = df.iloc[j]
                        if signal_info['signal'] == 'BUY':
                            if future_candle['high'] >= signal_info['take_profit']:
                                tp_hit_time = df.index[j].tz_convert(chicago_tz)
                                break
                        else:  # SELL
                            if future_candle['low'] <= signal_info['take_profit']:
                                tp_hit_time = df.index[j].tz_convert(chicago_tz)
                                break

                    if tp_hit_time:
                        time_to_tp = (tp_hit_time - entry_time_chicago).total_seconds() / 3600  # hours
                        signal_data['tp_hit_time'] = tp_hit_time
                        signal_data['time_to_tp_hours'] = time_to_tp

                    recent_signals.append(signal_data)

            except Exception as e:
                continue

        print(f"\n📊 Found {len(recent_signals)} recent GBPUSD signals")

        # Show the most recent signals
        for i, signal in enumerate(recent_signals[-3:]):  # Last 3 signals
            print(f"\n🔹 Signal #{len(recent_signals)-2+i}:")
            print(f"   Entry Time: {signal['entry_time_chicago'].strftime('%Y-%m-%d %H:%M %Z')}")
            print(f"   Signal: {signal['signal']}")
            print(f"   Entry Price: ${signal['entry_price']:.5f}")
            print(f"   Take Profit: ${signal['take_profit']:.5f}")
            print(f"   Confluence: {signal['confluence_score']}/25")

            if 'tp_hit_time' in signal:
                print(f"   TP Hit Time: {signal['tp_hit_time'].strftime('%Y-%m-%d %H:%M %Z')}")
                print(f"   Time to TP: {signal['time_to_tp_hours']:.1f} hours")
                print(f"   ✅ SUCCESSFUL TRADE")
            else:
                print(f"   ❌ TP not hit within 100 hours")

        # Find the most recent successful trade
        successful_trades = [s for s in recent_signals if 'tp_hit_time' in s]
        if successful_trades:
            latest_success = successful_trades[-1]
            print(f"\n🏆 MOST RECENT SUCCESSFUL GBPUSD TRADE:")
            print(f"   Entry: {latest_success['entry_time_chicago'].strftime('%Y-%m-%d %H:%M %Z')}")
            print(f"   TP Hit: {latest_success['tp_hit_time'].strftime('%Y-%m-%d %H:%M %Z')}")
            print(f"   Duration: {latest_success['time_to_tp_hours']:.1f} hours")
            print(f"   Signal: {latest_success['signal']} at ${latest_success['entry_price']:.5f}")

    except Exception as e:
        print(f"❌ GBPUSD analysis error: {e}")

async def main():
    print("🔍 RECENT CONFLUENCE SIGNALS ANALYSIS")
    print("=====================================")

    await analyze_recent_xauusd_signals()
    await analyze_recent_gbpusd_signals()

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    asyncio.run(main())