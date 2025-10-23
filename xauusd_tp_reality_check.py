#!/usr/bin/env python3
"""
XAUUSD 2:1 TP Reality Check
Analyze actual 2:1 TP hit rates for XAUUSD confluence signals
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np

async def analyze_xauusd_2_to_1_reality():
    print("🔍 XAUUSD 2:1 TP REALITY CHECK")
    print("=" * 50)
    print("⚠️  Analyzing actual 2:1 TP achievement rates...")

    try:
        from app.oanda_feed import get_xauusd_candles
        from app.smart_confluence_system import evaluate_smart_confluence_signal

        # Get comprehensive historical data
        df = await get_xauusd_candles(count=5000)  # Maximum available
        if df is None or df.empty:
            print("❌ No XAUUSD data available")
            return

        print(f"✅ Analyzing {len(df)} XAUUSD candles from {df.index[0]} to {df.index[-1]}")
        print(f"   Time span: {(df.index[-1] - df.index[0]).days} days")

        chicago_tz = pytz.timezone('America/Chicago')

        # Track all signals and their outcomes
        all_signals = []
        successful_2to1_tps = []

        print("\n🔍 Scanning for confluence signals and 2:1 TP achievements...")

        for i in range(60, len(df) - 150):  # Need history and future data
            try:
                current_price = df['close'].iloc[i]
                current_time = df.index[i]
                historical_df = df.iloc[:i+1].copy()

                # Check for confluence signal
                result = evaluate_smart_confluence_signal(current_price, historical_df)

                if result and result.get('signal') in ['BUY', 'SELL']:
                    entry_price = result['entry_price']
                    tp_price = result['take_profit']
                    sl_price = result['stop_loss']
                    signal_type = result['signal']

                    entry_time_chicago = current_time.tz_convert(chicago_tz)

                    signal_data = {
                        'entry_time': entry_time_chicago,
                        'signal': signal_type,
                        'entry_price': entry_price,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'confluence_score': result['confluence_score']
                    }
                    all_signals.append(signal_data)

                    # Check if 2:1 TP was hit in next 100 candles (4+ days)
                    future_candles = df.iloc[i+1:i+101]
                    tp_hit = False
                    sl_hit = False
                    tp_hit_time = None
                    sl_hit_time = None

                    for j, future_candle in future_candles.iterrows():
                        # Check for TP hit (2:1 ratio)
                        if signal_type == 'BUY':
                            if future_candle['high'] >= tp_price and not tp_hit:
                                tp_hit = True
                                tp_hit_time = j.tz_convert(chicago_tz)
                            if future_candle['low'] <= sl_price and not sl_hit:
                                sl_hit = True
                                sl_hit_time = j.tz_convert(chicago_tz)
                        else:  # SELL
                            if future_candle['low'] <= tp_price and not tp_hit:
                                tp_hit = True
                                tp_hit_time = j.tz_convert(chicago_tz)
                            if future_candle['high'] >= sl_price and not sl_hit:
                                sl_hit = True
                                sl_hit_time = j.tz_convert(chicago_tz)

                        # Determine outcome
                        if tp_hit and sl_hit:
                            if tp_hit_time <= sl_hit_time:
                                # TP hit first = WIN
                                time_to_tp = (tp_hit_time - entry_time_chicago).total_seconds() / 3600
                                signal_data.update({
                                    'outcome': 'TP_HIT',
                                    'tp_hit_time': tp_hit_time,
                                    'time_to_tp_hours': time_to_tp
                                })
                                successful_2to1_tps.append(signal_data.copy())
                                break
                            else:
                                # SL hit first = LOSS
                                signal_data['outcome'] = 'SL_HIT'
                                break
                        elif tp_hit:
                            # Only TP hit = WIN
                            time_to_tp = (tp_hit_time - entry_time_chicago).total_seconds() / 3600
                            signal_data.update({
                                'outcome': 'TP_HIT',
                                'tp_hit_time': tp_hit_time,
                                'time_to_tp_hours': time_to_tp
                            })
                            successful_2to1_tps.append(signal_data.copy())
                            break
                        elif sl_hit:
                            # Only SL hit = LOSS
                            signal_data['outcome'] = 'SL_HIT'
                            break

                    # If neither hit within 100 candles
                    if 'outcome' not in signal_data:
                        signal_data['outcome'] = 'NO_EXIT'

                # Progress indicator
                if i % 1000 == 0:
                    progress = (i / len(df)) * 100
                    print(f"   Progress: {progress:.1f}% - Found {len(successful_2to1_tps)} successful 2:1 TPs")

            except Exception as e:
                continue

        print(f"\n📊 XAUUSD 2:1 TP ANALYSIS RESULTS:")
        print("-" * 40)
        print(f"   Total signals analyzed: {len(all_signals)}")
        print(f"   Successful 2:1 TPs: {len(successful_2to1_tps)}")

        if len(all_signals) > 0:
            success_rate = (len(successful_2to1_tps) / len(all_signals)) * 100
            print(f"   🎯 ACTUAL 2:1 TP SUCCESS RATE: {success_rate:.1f}%")

            if success_rate < 30:
                print(f"   ⚠️  WARNING: Success rate is very low!")
            elif success_rate < 50:
                print(f"   ⚠️  CAUTION: Success rate is below 50%")
            else:
                print(f"   ✅ Success rate looks reasonable")

        # Analyze timing patterns
        if successful_2to1_tps:
            tp_times = [tp['time_to_tp_hours'] for tp in successful_2to1_tps]
            avg_time = np.mean(tp_times)
            median_time = np.median(tp_times)

            print(f"\n⏰ 2:1 TP TIMING ANALYSIS:")
            print(f"   Average time to TP: {avg_time:.1f} hours")
            print(f"   Median time to TP: {median_time:.1f} hours")
            print(f"   Fastest TP: {min(tp_times):.1f} hours")
            print(f"   Slowest TP: {max(tp_times):.1f} hours")

            # Show recent successful TPs
            successful_2to1_tps.sort(key=lambda x: x['entry_time'])

            print(f"\n🏆 LAST 3 SUCCESSFUL XAUUSD 2:1 TPs:")
            print("-" * 45)

            for i, tp in enumerate(successful_2to1_tps[-3:]):
                print(f"\n#{len(successful_2to1_tps) - 2 + i}. {tp['signal']} Signal")
                print(f"   Entry: {tp['entry_time'].strftime('%Y-%m-%d %H:%M %Z')}")
                print(f"   Entry Price: ${tp['entry_price']:.2f}")
                print(f"   TP Price: ${tp['tp_price']:.2f}")
                print(f"   TP Hit: {tp['tp_hit_time'].strftime('%Y-%m-%d %H:%M %Z')}")
                print(f"   Time to TP: {tp['time_to_tp_hours']:.1f} hours")
                print(f"   Confluence: {tp['confluence_score']}/20")

            # Most recent successful TP
            most_recent = successful_2to1_tps[-1]
            now_chicago = datetime.now(chicago_tz)
            time_since = now_chicago - most_recent['tp_hit_time']

            print(f"\n🎯 MOST RECENT SUCCESSFUL 2:1 TP:")
            print("=" * 40)
            print(f"Entry: {most_recent['entry_time'].strftime('%Y-%m-%d %H:%M %Z')}")
            print(f"TP Hit: {most_recent['tp_hit_time'].strftime('%Y-%m-%d %H:%M %Z')}")
            print(f"Time ago: {time_since.days} days, {time_since.seconds // 3600} hours")

        else:
            print(f"\n❌ NO SUCCESSFUL 2:1 TPs FOUND!")
            print(f"   This is concerning for the XAUUSD strategy")

        # Reality check recommendation
        print(f"\n💡 RECOMMENDATIONS:")
        if len(all_signals) > 0:
            success_rate = (len(successful_2to1_tps) / len(all_signals)) * 100
            if success_rate < 20:
                print(f"   🚨 URGENT: 2:1 TP rarely achievable ({success_rate:.1f}%)")
                print(f"   📉 Consider 1.5:1 or 1:1 ratios instead")
                print(f"   🔧 Re-evaluate confluence thresholds")
            elif success_rate < 40:
                print(f"   ⚠️  CAUTION: Low 2:1 success rate ({success_rate:.1f}%)")
                print(f"   📊 Consider partial profit taking strategy")
                print(f"   ⏰ Implement time-based exit rules")
            else:
                print(f"   ✅ 2:1 TP success rate acceptable ({success_rate:.1f}%)")

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_xauusd_2_to_1_reality())